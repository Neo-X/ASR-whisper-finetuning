"""
Promote dictation clips from dictate.py's staging folder into the personal
dictation dataset.

dictate.py saves every recording to /tmp/dictate-clips (wiped on reboot). Clips
worth keeping — especially ones Whisper got wrong — are copied here with a
corrected reference transcription, so they can be used for evaluation (WER) and
possibly training.

The dataset uses the Hugging Face "audiofolder" layout:

    data/dictation/
        clips/<id>.wav
        metadata.jsonl      one record per clip

and loads with:

    load_dataset("audiofolder", data_dir="data/dictation")
    # then filter on the "split" column ("test" / "train")

Usage:
    python src/add_clip.py --list                       # show staged clips
    python src/add_clip.py 20261018_181909_104 \
        --text "...whether or not this particular audio clip saves." \
        --tag failure --note "audio -> auto"
"""

import argparse
import json
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STAGING = os.environ.get("DICTATE_SAVE_DIR", "/tmp/dictate-clips")
DEFAULT_DATASET = os.path.join(REPO_ROOT, "data", "dictation")
TAGS = ["failure", "good"]
SPLITS = ["test", "train"]


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def list_staged(staging_dir, dataset_dir, limit):
    kept = {r["id"] for r in read_jsonl(os.path.join(dataset_dir, "metadata.jsonl"))}
    staged = read_jsonl(os.path.join(staging_dir, "metadata.jsonl"))
    for r in staged[-limit:]:
        mark = "*" if r["id"] in kept else " "
        print(f"{mark} {r['id']}  {r['duration_s']:5.1f}s  {r['hypothesis']}")
    print(f"\n{len(staged)} staged clips in {staging_dir}; * = already in dataset")


def add_clip(args):
    staged = {r["id"]: r for r in read_jsonl(os.path.join(args.staging_dir, "metadata.jsonl"))}
    if args.clip_id not in staged:
        sys.exit(f"Clip {args.clip_id} not found in {args.staging_dir}/metadata.jsonl")
    src = staged[args.clip_id]

    clips_dir = os.path.join(args.dataset_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    shutil.copy2(os.path.join(args.staging_dir, src["file_name"]),
                 os.path.join(clips_dir, f"{args.clip_id}.wav"))

    text = args.text if args.text is not None else src["hypothesis"]
    record = {
        "file_name": f"clips/{args.clip_id}.wav",
        "id": args.clip_id,
        # Reference (corrected) text — the column train.py/eval.py read.
        "transcription": text,
        "hypothesis": src["hypothesis"],
        "tag": args.tag,
        "split": args.split,
        "note": args.note or "",
        "timestamp": src["timestamp"],
        "duration_s": src["duration_s"],
        "model": src["model"],
        "initial_prompt": src["initial_prompt"],
    }

    # Re-adding a clip replaces its record (e.g. to fix the reference text).
    meta_path = os.path.join(args.dataset_dir, "metadata.jsonl")
    records = [r for r in read_jsonl(meta_path) if r["id"] != args.clip_id]
    records.append(record)
    write_jsonl(meta_path, records)

    print(f"Added {args.clip_id} [{args.tag}/{args.split}]")
    print(f"  hypothesis:    {record['hypothesis']}")
    print(f"  transcription: {record['transcription']}")
    if args.tag == "failure" and record["transcription"] == record["hypothesis"]:
        print("  warning: tagged failure but transcription equals hypothesis — pass --text")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip_id", nargs="?", help="Clip id from dictate.py (e.g. 20261018_181909_104)")
    parser.add_argument("--text", help="Corrected reference transcription (defaults to Whisper's hypothesis)")
    parser.add_argument("--tag", choices=TAGS, default="failure")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--note", help="Free-text note, e.g. what went wrong")
    parser.add_argument("--list", action="store_true", help="List recently staged clips")
    parser.add_argument("--limit", type=int, default=20, help="Number of clips shown by --list")
    parser.add_argument("--staging_dir", default=DEFAULT_STAGING)
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET)
    args = parser.parse_args()

    if args.list:
        list_staged(args.staging_dir, args.dataset_dir, args.limit)
    elif args.clip_id:
        add_clip(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
