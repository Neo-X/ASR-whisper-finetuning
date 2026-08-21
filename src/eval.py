# eval.py
import logging
import os

import evaluate
import torch
from datasets import Audio, load_dataset
from huggingface_hub import login
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Setting up logging to show information level messages
logging.basicConfig(format="%(message)s", level=logging.INFO)

# Must match the base model used in train.py
BASE_MODEL_NAME = "distil-whisper/distil-large-v3"

# Common transcript keys across datasets
TRANSCRIPT_KEYS = ["text", "sentence", "normalized_text", "transcript", "transcription"]


def load_processor(model_dir):
    """
    Load the processor saved with the fine-tuned checkpoint (falling back to the base model's).

    Args:
        model_dir (str): Directory containing the saved LoRA checkpoint and processor.

    Returns:
        WhisperProcessor: The processor used for feature extraction and decoding.
    """
    if os.path.isfile(os.path.join(model_dir, "preprocessor_config.json")) and os.path.isfile(
        os.path.join(model_dir, "tokenizer_config.json")
    ):
        logging.info(f"Loading processor from {model_dir}...")
        return WhisperProcessor.from_pretrained(model_dir)

    logging.info(f"No processor found in {model_dir}, loading from {BASE_MODEL_NAME}...")
    return WhisperProcessor.from_pretrained(BASE_MODEL_NAME)


def load_model(model_dir, device, dtype):
    """
    Load the base Whisper model, apply the fine-tuned LoRA adapter, and merge it for inference.

    Args:
        model_dir (str): Directory containing the saved LoRA adapter.
        device (str): Target device ("cuda" or "cpu").
        dtype (torch.dtype): Model dtype (fp16 on GPU, fp32 on CPU).

    Returns:
        WhisperForConditionalGeneration: The base model with the LoRA adapter merged in.
    """
    logging.info(f"Loading base model '{BASE_MODEL_NAME}'...")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_NAME, torch_dtype=dtype)

    logging.info(f"Loading LoRA adapter from {model_dir}...")
    model = PeftModel.from_pretrained(model, model_dir)
    model = model.merge_and_unload()

    model.to(device)
    model.eval()
    return model


def prepare_eval_dataset(args, sampling_rate):
    """
    Load the evaluation dataset, keep only audio and transcript columns, and filter out long samples.

    Args:
        args (argparse.Namespace): Parsed arguments.
        sampling_rate (int): Target audio sampling rate.

    Returns:
        datasets.Dataset: The evaluation dataset with 'audio' and 'transcription' columns.
    """
    logging.info(f"Loading dataset '{args.dataset_name}' ({args.language}, split={args.split})...")
    try:
        dataset = load_dataset(args.dataset_name, args.language, split=args.split, trust_remote_code=True)
    except Exception:
        if args.split != "validation":
            logging.warning(f"Split '{args.split}' not found, falling back to 'validation'.")
            dataset = load_dataset(
                args.dataset_name, args.language, split="validation", trust_remote_code=True
            )
        else:
            raise

    # Keep only the audio and transcript columns
    columns_to_keep = ["audio"] + TRANSCRIPT_KEYS
    remove_columns = [col for col in dataset.features.keys() if col not in columns_to_keep]
    dataset = dataset.remove_columns(remove_columns)

    # Normalize the transcript column name
    transcript_key = next(key for key in TRANSCRIPT_KEYS if key in dataset.features)
    if transcript_key != "transcription":
        dataset = dataset.rename_column(transcript_key, "transcription")

    # Adjust the audio to the required sampling rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # Filter out long audio samples (match the training-time filter)
    def filter_length(batch):
        return [len(item["array"]) / sampling_rate <= args.max_input_length for item in batch["audio"]]

    logging.info("Filtering long audio samples...")
    dataset = dataset.filter(
        filter_length,
        batched=True,
        batch_size=1000,
        num_proc=args.num_workers,
        desc="Filtering evaluation set",
    )

    return dataset


def evaluate_model(args):
    """
    Evaluate a LoRA fine-tuned Whisper checkpoint on a dataset and report WER.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        float: Word Error Rate over the full evaluation set.
    """
    if args.auth_token:
        login(token=args.auth_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    logging.info(f"Evaluating on: {device}")

    processor = load_processor(args.model_dir)
    model = load_model(args.model_dir, device, dtype)

    sampling_rate = processor.feature_extractor.sampling_rate
    dataset = prepare_eval_dataset(args, sampling_rate)
    logging.info(f"Evaluating on {len(dataset)} samples...")

    # WER metric
    wer_metric = evaluate.load("wer")

    # Force the same language/task prompt used during training
    forced_decoder_ids = processor.get_decoder_prompt_ids(task="transcribe", language=args.prompt_language)

    references = []
    predictions = []

    for start in tqdm(range(0, len(dataset), args.batch_size), desc="Generating"):
        batch = dataset.select(range(start, min(start + args.batch_size, len(dataset))))

        audio_arrays = [sample["audio"]["array"] for sample in batch]
        inputs = processor(audio=audio_arrays, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_features = inputs.input_features.to(device)
        if device == "cuda":
            input_features = input_features.to(dtype)

        # Generate transcriptions
        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=225,
                do_sample=False,
                num_beams=args.num_beams,
            )
        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Collect references and predictions
        references.extend([sample["transcription"] for sample in batch])
        predictions.extend(transcriptions)

    # Compute WER
    wer = wer_metric.compute(predictions=predictions, references=references)
    logging.info(f"WER on {args.dataset_name} ({args.language}, split={args.split}): {wer:.4f}")
    return wer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a LoRA fine-tuned Whisper checkpoint")
    parser.add_argument("--dataset_name", type=str, default="google/fleurs")
    parser.add_argument("--language", type=str, default="fr_fr")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model_dir", type=str, default="./whisper-fr-LoRA")
    parser.add_argument("--max_input_length", type=float, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prompt_language", type=str, default="french")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--auth_token", type=str, default="")
    evaluate_model(parser.parse_args())
