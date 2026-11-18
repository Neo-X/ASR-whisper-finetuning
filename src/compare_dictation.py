# compare_dictation.py
"""
Compare eval_dictation.py runs side by side on the clips they all share.

For each run (one results jsonl) reports normalized WER on the common clip set, how many
clips it got better / same / worse than what turbo produced at dictation time, how many it
got exactly right, and speed.

Usage:
    python src/compare_dictation.py                       # latest run per name
    python src/compare_dictation.py eval_a.jsonl eval_b.jsonl
    python src/compare_dictation.py --show 20261019_094650_070   # one clip across runs

Every comparison also writes two CSVs to data/dictation-results/ (override with --csv_prefix):
    comparison_summary.csv   one row per run (WER, better/same/worse, exact, corrected terms right, RTF)
    comparison_per_clip.csv  one row per clip: reference, recorded, and each run's text + WER
"""
import argparse
import csv
import glob
import json
import os
import re

import jiwer

def note_targets(note, reference):
    """Corrected words from an add_clip note ("wrong -> right; ..."), kept if they occur in the reference."""
    targets = []
    for part in (note or "").split(";"):
        if "->" in part:
            t = re.sub(r"\(.*?\)", "", part.split("->")[1]).strip()
            if t and t.lower() in reference.lower():
                targets.append(t)
    return targets


def contains(text, term):
    squash = lambda x: re.sub(r"[^a-z0-9& ]", "", x.lower())
    return squash(term) in squash(text)


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dictation-results")


def run_name(path):
    m = re.match(r"eval_\d{8}_\d{6}_(.+)\.jsonl", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def latest_runs():
    """Latest results file per run name, skipping smoke tests and single-clip runs."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "eval_*.jsonl"))):
        rows = [json.loads(l) for l in open(path)]
        if len(rows) < 20 or "model" not in rows[0]:
            continue
        runs[run_name(path)] = path  # sorted by timestamp, so later files win
    return runs


def main(args):
    runs = {run_name(p): p for p in args.files} if args.files else latest_runs()
    data = {name: {r["id"]: r for r in map(json.loads, open(path))} for name, path in runs.items()}
    common = set.intersection(*(set(d) for d in data.values()))
    any_run = next(iter(data.values()))

    from transformers import WhisperTokenizer
    normalize = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo").normalize

    if args.show:
        r = any_run[args.show]
        print(f"reference : {r['transcription']}\nrecorded  : {r['recorded']}")
        for name, d in data.items():
            print(f"{name:<10}: {d[args.show]['model']}")
        return

    ids = sorted(common)
    refs = [normalize(any_run[i]["transcription"]) for i in ids]
    recorded = [normalize(any_run[i]["recorded"]) for i in ids]
    print(f"{len(ids)} clips in common across {len(data)} runs\n")
    rows = []
    for name, d in data.items():
        preds = [normalize(d[i]["model"]) for i in ids]
        better = same = worse = 0
        for ref, rec, pred in zip(refs, recorded, preds):
            a, b = jiwer.wer(ref, pred), jiwer.wer(ref, rec)
            better += a < b
            same += a == b
            worse += a > b
        secs = sum(d[i]["model_seconds"] for i in ids)
        rows.append((jiwer.wer(refs, preds), name, better, same, worse,
                     sum(r == p for r, p in zip(refs, preds)), secs))
    meta = {}
    meta_path = os.path.join(os.path.dirname(RESULTS_DIR), "dictation", "metadata.jsonl")
    if os.path.exists(meta_path):
        meta = {r["id"]: r["duration_s"] for r in map(json.loads, open(meta_path))}
    audio = sum(meta.get(i, 0) for i in ids) or float("nan")
    notes = {}
    if os.path.exists(meta_path):
        notes = {r["id"]: r.get("note", "") for r in map(json.loads, open(meta_path))}
    targets = [(i, t) for i in ids for t in note_targets(notes.get(i, ""), any_run[i]["transcription"])]
    terms_fixed = lambda texts: sum(contains(texts[i], t) for i, t in targets)
    summary = [{"run": "recorded turbo", "model_name": "turbo (dictate.py at dictation time)",
                "n_clips": len(ids), "wer_normalized": round(jiwer.wer(refs, recorded), 4),
                "better": "", "same": "", "worse": "",
                "exact": sum(r == p for r, p in zip(refs, recorded)),
                "terms_correct": terms_fixed({i: any_run[i]["recorded"] for i in ids}), "terms_total": len(targets),
                "rtf": "", "results_file": ""}]
    for w, name, better, same, worse, exact, secs in sorted(rows):
        first = data[name][ids[0]]
        summary.append({"run": name, "model_name": first.get("model_name", ""), "n_clips": len(ids),
                        "wer_normalized": round(w, 4), "better": better, "same": same, "worse": worse,
                        "exact": exact,
                        "terms_correct": terms_fixed({i: data[name][i]["model"] for i in ids}),
                        "terms_total": len(targets), "rtf": round(secs / audio, 4),
                        "results_file": os.path.basename(runs[name])})

    header = f"{'run':<16}{'WER':>7}{'better':>8}{'same':>6}{'worse':>7}{'exact':>7}{'terms':>9}{'RTF':>7}"
    print(header + "\n" + "-" * len(header))
    for r in summary:
        terms = f"{r['terms_correct']}/{r['terms_total']}"
        rtf = f"{r['rtf']:.3f}" if r["rtf"] != "" else ""
        print(f"{r['run']:<16}{r['wer_normalized']:>7.3f}{r['better']:>8}{r['same']:>6}{r['worse']:>7}"
              f"{r['exact']:>7}{terms:>9}{rtf:>7}")

    prefix = args.csv_prefix or os.path.join(RESULTS_DIR, "comparison")
    with open(f"{prefix}_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    names = [r["run"] for r in summary[1:]]
    with open(f"{prefix}_per_clip.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "note", "reference", "recorded", "recorded_wer"]
                        + [c for n in names for c in (n, f"{n}_wer")])
        for i, ref, rec in zip(ids, refs, recorded):
            row = [i, notes.get(i, ""), any_run[i]["transcription"], any_run[i]["recorded"],
                   round(jiwer.wer(ref, rec), 4)]
            for n in names:
                row += [data[n][i]["model"], round(jiwer.wer(ref, normalize(data[n][i]["model"])), 4)]
            writer.writerow(row)
    print(f"\nWrote {prefix}_summary.csv and {prefix}_per_clip.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="Results files (default: latest per run name)")
    parser.add_argument("--csv_prefix", help="Path prefix for the CSV outputs (default: data/dictation-results/comparison)")
    parser.add_argument("--show", help="Print one clip's transcription from every run")
    main(parser.parse_args())
