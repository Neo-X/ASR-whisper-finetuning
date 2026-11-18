# eval_dictation.py
"""
Baseline WER on the personal dictation dataset (data/dictation, built with add_clip.py).

Two numbers are reported:

  recorded  WER of the hypothesis dictate.py actually produced at dictation time
            (openai-whisper "turbo" + its initial prompt). No model is run.
  model     WER of re-transcribing the clips with --backend:
              whisper  openai/whisper-large-v3-turbo (same weights as "turbo"),
                       optionally + a LoRA adapter, with dictate.py's prompt
              granite  ibm-granite/granite-speech-4.1-2b, optional --keywords biasing
              cohere   CohereLabs/cohere-transcribe-03-2026
              qwen3    Qwen/Qwen3-ASR-1.7B-hf, optional --keywords as context/hotwords
              parakeet nvidia/parakeet-tdt-0.6b-v3
              kyutai   kyutai/stt-2.6b-en-trfs
            The whisper number is the one a fine-tuned model has to beat.

Speed is reported as RTF (processing time / audio duration, lower is faster),
measured on this machine one clip at a time after a warm-up clip.

Each is shown raw (case and punctuation count as errors) and normalized (Whisper's
English text normalizer: lowercase, no punctuation, standardised spellings).

Usage:
    python src/eval_dictation.py                      # recorded + base turbo on split=test
    python src/eval_dictation.py --skip_model         # recorded only, no GPU needed
    python src/eval_dictation.py --lora_dir ./whisper-dictation-LoRA
    python src/eval_dictation.py --backend granite --keywords "Asana, RLC, LoRA"
    python src/eval_dictation.py --backend cohere
"""
import argparse
import json
import logging
import os
import time
from collections import defaultdict

import jiwer
from datasets import Audio, concatenate_datasets, load_dataset

logging.basicConfig(format="%(message)s", level=logging.INFO)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(REPO_ROOT, "data", "dictation")
SAMPLING_RATE = 16000
# Must match WHISPER_INITIAL_PROMPT in transcriber/dictate.py.
DICTATE_INITIAL_PROMPT = (
    "Transcribed in British/Canadian English: colour, favourite, organise, realise, "
    "centre, behaviour, analyse, travelling, licence."
)
DEFAULT_MODELS = {
    "whisper": "openai/whisper-large-v3-turbo",
    "granite": "ibm-granite/granite-speech-4.1-2b",
    "cohere": "CohereLabs/cohere-transcribe-03-2026",
    "qwen3": "Qwen/Qwen3-ASR-1.7B-hf",
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
    "kyutai": "kyutai/stt-2.6b-en-trfs",
}


def load_dictation_dataset(args):
    """Load the audiofolder dataset and keep only the requested split/tag."""
    # Splits come from our own "split" column, not from datasets' folder/file-name
    # inference, so merge whatever splits it infers back into one dataset.
    loaded = load_dataset("audiofolder", data_dir=args.dataset_dir)
    dataset = concatenate_datasets(list(loaded.values()))
    if args.split != "all":
        dataset = dataset.filter(lambda r: r["split"] == args.split)
    if args.tag:
        dataset = dataset.filter(lambda r: r["tag"] == args.tag)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    return dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))


def get_normalizer():
    """Whisper's English normalizer (from the tokenizer), so WER ignores formatting differences."""
    from transformers import WhisperTokenizer

    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo")
    return lambda text: tokenizer.normalize(text)


def wer(references, predictions):
    """Corpus WER over non-empty references."""
    pairs = [(r, p) for r, p in zip(references, predictions) if r.strip()]
    if not pairs:
        return float("nan")
    refs, preds = zip(*pairs)
    return jiwer.wer(list(refs), list(preds))


def get_device():
    import torch

    # On AMD APUs (Strix Halo), ROCm needs this to enumerate the GPU; matches dictate.py.
    os.environ.setdefault("ROCR_VISIBLE_DEVICES", "0")
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_whisper(args, device):
    """HF Whisper (+ optional LoRA), mirroring dictate.py's initial prompt."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name, dtype=dtype)
    if args.lora_dir:
        from peft import PeftModel

        logging.info(f"Loading LoRA adapter from {args.lora_dir}...")
        model = PeftModel.from_pretrained(model, args.lora_dir).merge_and_unload()
    model.to(device).eval()

    def transcribe(sample):
        # Use the prompt dictate.py used for this clip unless one is given explicitly.
        # Clips dictated with another model have no initial_prompt; fall back to dictate.py's.
        prompt = args.prompt if args.prompt is not None else (sample["initial_prompt"] or DICTATE_INITIAL_PROMPT)
        # truncation=False + attention mask lets generate() do long-form decoding for clips > 30 s.
        inputs = processor(
            sample["audio"]["array"],
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
        if inputs.input_features.shape[-1] < 3000:
            # Short clips still need the full 30 s window.
            inputs = processor(sample["audio"]["array"], sampling_rate=SAMPLING_RATE,
                               return_tensors="pt", return_attention_mask=True)
        gen_kwargs = dict(language="english", task="transcribe", num_beams=args.num_beams, do_sample=False)
        if inputs.input_features.shape[-1] > 3000:
            gen_kwargs["return_timestamps"] = True  # required for long-form (> 30 s) decoding
        if prompt:
            gen_kwargs["prompt_ids"] = processor.get_prompt_ids(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(
                inputs.input_features.to(device, dtype),
                attention_mask=inputs.attention_mask.to(device),
                **gen_kwargs,
            )
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        # Decoding with prompt_ids can echo the prompt at the start; strip it.
        if prompt and text.startswith(prompt):
            text = text[len(prompt):].strip()
        return text

    return transcribe


def load_granite(args, device):
    """IBM Granite Speech (speech LLM); --keywords are appended to the instruction for biasing."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_name, device_map=device, dtype=torch.bfloat16
    ).eval()

    instruction = "<|audio|>transcribe the speech with proper punctuation and capitalization."
    if args.keywords:
        instruction += f" Keywords: {args.keywords}"
    chat = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def transcribe(sample):
        audio = sample["audio"]["array"]
        wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)
        # Leave room for fast speech: ~4 tokens per second of audio, at least 200.
        max_new_tokens = max(200, int(4 * len(audio) / SAMPLING_RATE))
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=args.num_beams)
        new_tokens = ids[:, inputs["input_ids"].shape[-1]:]
        return tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)[0].strip()

    return transcribe


def load_cohere(args, device):
    """Cohere Transcribe (needs transformers>=5.4); the processor chunks long audio itself."""
    import torch
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = CohereAsrForConditionalGeneration.from_pretrained(args.model_name, device_map=device).eval()

    def transcribe(sample):
        audio = sample["audio"]["array"]
        inputs = processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt", language="en")
        audio_chunk_index = inputs.get("audio_chunk_index")
        inputs.to(model.device, dtype=model.dtype)
        max_new_tokens = max(256, int(4 * len(audio) / SAMPLING_RATE))
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = processor.decode(ids, skip_special_tokens=True, audio_chunk_index=audio_chunk_index, language="en")
        if isinstance(text, list):
            text = text[0]
        return text.strip()

    return transcribe


def load_qwen3(args, device):
    """Qwen3-ASR (speech LLM); --keywords go in the system prompt as context/hotwords."""
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_name, device_map=device, dtype=torch.bfloat16
    ).eval()

    def transcribe(sample):
        audio = sample["audio"]["array"]
        inputs = processor.apply_transcription_request(audio, language="English", prompt=args.keywords or None)
        inputs = inputs.to(model.device, dtype=model.dtype)
        max_new_tokens = max(256, int(5 * len(audio) / SAMPLING_RATE))
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=args.num_beams)
        new_tokens = ids[:, inputs["input_ids"].shape[-1]:]
        return processor.decode(new_tokens[0], return_format="transcription_only").strip()

    return transcribe


def load_parakeet(args, device):
    """NVIDIA Parakeet TDT (transducer, no prompting)."""
    import torch
    from transformers import AutoModelForCTC, AutoProcessor, ParakeetForTDT

    processor = AutoProcessor.from_pretrained(args.model_name)
    cls = AutoModelForCTC if "ctc" in args.model_name else ParakeetForTDT
    model = cls.from_pretrained(args.model_name, device_map=device).eval()

    def transcribe(sample):
        inputs = processor(sample["audio"]["array"], sampling_rate=SAMPLING_RATE, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs)
        ids = out.sequences if hasattr(out, "sequences") else out
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    return transcribe


def load_kyutai(args, device):
    """Kyutai STT (streaming decoder-only); expects 24 kHz audio."""
    import librosa
    import torch
    from transformers import KyutaiSpeechToTextForConditionalGeneration, KyutaiSpeechToTextProcessor

    processor = KyutaiSpeechToTextProcessor.from_pretrained(args.model_name)
    model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(args.model_name, device_map=device).eval()
    sr = processor.feature_extractor.sampling_rate

    def transcribe(sample):
        audio = librosa.resample(sample["audio"]["array"], orig_sr=SAMPLING_RATE, target_sr=sr)
        inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    return transcribe


BACKENDS = {"whisper": load_whisper, "granite": load_granite, "cohere": load_cohere,
            "qwen3": load_qwen3, "parakeet": load_parakeet, "kyutai": load_kyutai}


def transcribe_with_model(args, dataset):
    """Re-transcribe every clip with the chosen backend; returns (texts, seconds per clip)."""
    device = get_device()
    logging.info(f"Loading {args.model_name} ({args.backend}) on {device}...")
    transcribe = BACKENDS[args.backend](args, device)

    # Warm up once so one-time GPU kernel compilation doesn't count against the first clip's speed.
    transcribe(dataset[0])

    predictions, seconds = [], []
    for i, sample in enumerate(dataset):
        start = time.perf_counter()
        text = transcribe(sample)
        seconds.append(time.perf_counter() - start)
        predictions.append(text)
        logging.info(f"[{i + 1}/{len(dataset)}] {sample['id']} ({seconds[-1]:.2f}s): {text}")
    return predictions, seconds


def report(name, references, predictions, tags, normalize):
    norm_refs = [normalize(r) for r in references]
    norm_preds = [normalize(p) for p in predictions]
    logging.info(f"\n{name}:")
    logging.info(f"  WER raw        {wer(references, predictions):.4f}")
    logging.info(f"  WER normalized {wer(norm_refs, norm_preds):.4f}")
    by_tag = defaultdict(list)
    for i, tag in enumerate(tags):
        by_tag[tag].append(i)
    for tag, idx in sorted(by_tag.items()):
        tag_wer = wer([norm_refs[i] for i in idx], [norm_preds[i] for i in idx])
        logging.info(f"  WER normalized [{tag}] {tag_wer:.4f}  (n={len(idx)})")
    return norm_refs, norm_preds


def main(args):
    dataset = load_dictation_dataset(args)
    if len(dataset) == 0:
        logging.info(f"No clips match split={args.split} tag={args.tag or 'any'} in {args.dataset_dir}.")
        return
    logging.info(f"Evaluating {len(dataset)} clips (split={args.split}, tag={args.tag or 'any'}).")

    normalize = get_normalizer()
    references = dataset["transcription"]
    recorded = dataset["hypothesis"]
    tags = dataset["tag"]

    norm_refs, norm_recorded = report("recorded (dictate.py at dictation time)", references, recorded, tags, normalize)

    model_preds = None
    if not args.skip_model:
        model_preds, model_seconds = transcribe_with_model(args, dataset)
        label = args.model_name + (f" + {args.lora_dir}" if args.lora_dir else "")
        if args.keywords and args.backend in ("granite", "qwen3"):
            label += f", keywords: {args.keywords}"
        _, norm_model = report(f"model ({label})", references, model_preds, tags, normalize)
        audio_seconds = sum(dataset["duration_s"])
        logging.info(f"  Speed: {sum(model_seconds):.2f}s for {audio_seconds:.1f}s of audio, "
                     f"RTF {sum(model_seconds) / audio_seconds:.3f} "
                     f"(mean {sum(model_seconds) / len(model_seconds):.2f}s per clip)")

    # Per-clip results, worst first, so failures are easy to inspect.
    rows = []
    for i, sample_id in enumerate(dataset["id"]):
        row = {
            "id": sample_id,
            "tag": tags[i],
            "transcription": references[i],
            "recorded": recorded[i],
            "recorded_wer": wer([norm_refs[i]], [norm_recorded[i]]),
        }
        if model_preds is not None:
            row["model"] = model_preds[i]
            row["model_wer"] = wer([norm_refs[i]], [norm_model[i]])
            row["model_seconds"] = round(model_seconds[i], 3)
            row["backend"] = args.backend
            row["model_name"] = args.model_name
        rows.append(row)
    sort_key = "model_wer" if model_preds is not None else "recorded_wer"
    rows.sort(key=lambda r: r[sort_key], reverse=True)

    logging.info("\nPer clip (normalized WER, worst first):")
    for r in rows:
        line = f"  {r['id']}  recorded={r['recorded_wer']:.2f}"
        if model_preds is not None:
            line += f"  model={r['model_wer']:.2f}"
        logging.info(line)

    # Kept outside dataset_dir so result files aren't picked up as dataset metadata.
    results_dir = os.path.join(os.path.dirname(os.path.abspath(args.dataset_dir)), "dictation-results")
    os.makedirs(results_dir, exist_ok=True)
    run_name = args.tag_name or (args.backend if not args.skip_model else "recorded")
    out_path = os.path.join(results_dir, f"eval_{time.strftime('%Y%m%d_%H%M%S')}_{run_name}.jsonl")
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logging.info(f"\nPer-clip results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline WER on the personal dictation dataset")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--split", type=str, default="test", help="test, train or all")
    parser.add_argument("--tag", type=str, default=None, help="Only evaluate clips with this tag (failure/good)")
    parser.add_argument("--backend", type=str, default="whisper", choices=sorted(BACKENDS))
    parser.add_argument("--model_name", type=str, default=None, help="Defaults to the backend's model")
    parser.add_argument("--lora_dir", type=str, default=None, help="whisper: optional LoRA adapter on top of --model_name")
    parser.add_argument("--prompt", type=str, default=None,
                        help="whisper: initial prompt; defaults to the one dictate.py used for each clip ('' for none)")
    parser.add_argument("--keywords", type=str, default=None,
                        help="granite/qwen3: comma-separated keywords for biasing, e.g. 'Asana, RLC, LoRA'")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N clips (smoke tests)")
    parser.add_argument("--tag_name", type=str, default=None, help="Run name used in the results file name (default: backend)")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--skip_model", action="store_true", help="Only report the recorded-hypothesis WER")
    args = parser.parse_args()
    args.model_name = args.model_name or DEFAULT_MODELS[args.backend]
    main(args)
