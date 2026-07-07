#!/usr/bin/env python3
"""Step 1: convert the raw binary datasets into LLM instruction format.

Reads the raw ``train.json`` / ``test.json`` produced by
``build_training_datasets.py`` and writes, for each task, instruction-formatted
JSONL files with the fields::

    {"file_id": "...",
     "instruction": "<question + 'answer Yes/No'>",
     "input": "<Arabic transcript>",
     "output": "Yes" | "No",
     "label": 0 | 1}

This format is model-agnostic; the chat template is applied later in train.py /
evaluate.py so the same files work for any model.

Usage::

    python prepare_data.py \
        --raw-dir ../training_datasets_0707 \
        --out-dir processed_datasets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks import TASKS, build_instruction, label_to_text


def convert_split(raw_path: Path, out_path: Path, question: str) -> tuple[int, int]:
    with raw_path.open(encoding="utf-8") as handle:
        records = json.load(handle)

    instruction = build_instruction(question)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_pos = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            label = int(rec["label"])
            n_pos += label
            item = {
                "file_id": rec["file_id"],
                "instruction": instruction,
                "input": rec["text"],
                "output": label_to_text(label),
                "label": label,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(records), n_pos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("../training_datasets_0707"))
    parser.add_argument("--out-dir", type=Path, default=Path("processed_datasets"))
    args = parser.parse_args()

    for task_key, meta in TASKS.items():
        for split in ("train", "test"):
            raw_path = args.raw_dir / task_key / f"{split}.json"
            out_path = args.out_dir / task_key / f"{split}.jsonl"
            if not raw_path.exists():
                print(f"WARNING: missing {raw_path}, skipping.")
                continue
            n, n_pos = convert_split(raw_path, out_path, meta["question"])
            print(f"{task_key:52} {split:5} -> {n:4} rows (pos={n_pos})  {out_path}")

    print(f"\nWrote instruction-formatted datasets under: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
