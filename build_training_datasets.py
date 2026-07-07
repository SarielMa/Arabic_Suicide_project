#!/usr/bin/env python3
"""Build 5 binary-classification datasets (one per prediction label).

For each of the 5 prediction labels we build an independent single-label binary
dataset:

  * Start from the usable matches table (``usable_matches.csv``).
  * Keep only rows whose value for THAT label is TRUE or FALSE (drop
    does_not_apply / operator_did_not_ask / no_response / blank).
  * Attach the deidentified transcript text for each file.
  * Stratified 80/20 train/test split so the TRUE/FALSE proportion is preserved
    in both splits.

Output: one folder per task, each containing ``train.json`` and ``test.json``.
Each JSON file is a list of records::

    {"file_id": "37914",
     "text_file": "...deidentified.txt",
     "text": "<transcript>",
     "label": 0,            # FALSE -> 0, TRUE -> 1
     "label_text": "FALSE"}

Usage::

    python build_training_datasets.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from sklearn.model_selection import train_test_split


LABEL_COLUMNS = [
    "Wish To Be Dead",
    "Non Specific Active Suicidal Thoughts",
    "Active Suicidal Ideation With Any Methods",
    "Active Suicidal With Some Intent To Act",
    "Active Suicidal Ideation With Specific Plan And Intent",
]

# Only these two values are valid targets for the binary tasks.
LABEL_MAP = {"TRUE": 1, "FALSE": 0}


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return name or "task"


def read_transcript(text_dir: Path, filename: str) -> str:
    path = text_dir / filename
    return path.read_text(encoding="utf-8").strip()


def load_usable_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_task_records(
    rows: list[dict[str, str]], column: str, text_dir: Path
) -> list[dict]:
    """Keep TRUE/FALSE rows for one label and attach transcript text."""
    records: list[dict] = []
    for row in rows:
        value = (row.get(column) or "").strip().upper()
        if value not in LABEL_MAP:
            continue
        records.append(
            {
                "file_id": row["file_id"],
                "text_file": row["text_file"],
                "text": read_transcript(text_dir, row["text_file"]),
                "label": LABEL_MAP[value],
                "label_text": value,
            }
        )
    return records


def write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--usable-csv",
        type=Path,
        default=Path("matched_info_0707/usable_matches.csv"),
    )
    parser.add_argument("--text-dir", type=Path, default=Path("Output_Deidentified"))
    parser.add_argument("--output-dir", type=Path, default=Path("training_datasets_0707"))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_usable_rows(args.usable_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Usable rows: {len(rows)}")
    print(f"Split: {int((1-args.test_size)*100)}/{int(args.test_size*100)} "
          f"stratified, seed={args.seed}")
    print("=" * 70)

    for column in LABEL_COLUMNS:
        records = build_task_records(rows, column, args.text_dir)
        labels = [r["label"] for r in records]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos

        # Stratify on the label so TRUE/FALSE proportions match across splits.
        train, test = train_test_split(
            records,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=labels,
        )

        task_dir = args.output_dir / safe_name(column)
        write_json(task_dir / "train.json", train)
        write_json(task_dir / "test.json", test)

        def prop(recs: list[dict]) -> str:
            n = len(recs)
            p = sum(r["label"] for r in recs)
            return f"{n} (TRUE={p}/{p/n*100:.1f}%, FALSE={n-p})"

        print(f"{column}")
        print(f"  total usable: {len(records)} (TRUE={n_pos}, FALSE={n_neg})")
        print(f"  train: {prop(train)}")
        print(f"  test:  {prop(test)}")
        print(f"  -> {task_dir}/")

    print("=" * 70)
    print(f"Done. Datasets written under: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
