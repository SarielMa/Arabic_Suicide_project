#!/usr/bin/env python3
"""Export the usable (consistent-label) matched text files to a CSV.

A file is "usable" when its filename ID exists in the metadata AND every
metadata row referencing that ID agrees on all 5 prediction labels. This
covers both:

  - files matching exactly one metadata row, and
  - files matching multiple rows that all carry identical labels.

Files whose rows disagree on any of the 5 labels are excluded.

The 5 prediction-label columns (spreadsheet columns AO..AS):
  - Wish To Be Dead
  - Non Specific Active Suicidal Thoughts
  - Active Suicidal Ideation With Any Methods
  - Active Suicidal With Some Intent To Act
  - Active Suicidal Ideation With Specific Plan And Intent

Usage::

    python export_usable_matches.py
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


SUFFIX = "_output_transcription_output_deidentified.txt"

LABEL_COLUMNS = [
    "Wish To Be Dead",
    "Non Specific Active Suicidal Thoughts",
    "Active Suicidal Ideation With Any Methods",
    "Active Suicidal With Some Intent To Act",
    "Active Suicidal Ideation With Specific Plan And Intent",
]


def normalize_id(value: object) -> str:
    """Return a clean integer-string ID, or "" if the token is not numeric."""
    text = str(value or "").strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text if re.fullmatch(r"\d+", text) else ""


def extract_file_id(name: str) -> str:
    """Extract the ID (block before ``_output``) from a text filename."""
    prefix = name[: -len(SUFFIX)] if name.endswith(SUFFIX) else Path(name).stem
    return normalize_id(prefix.rsplit("_", 1)[-1])


def detect_encoding(path: Path) -> str:
    """UTF-8 with a Windows-1252 fallback (Excel export contains cp1252 bytes)."""
    raw = path.read_bytes()
    try:
        raw.decode("utf-8-sig")
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "cp1252"


def load_metadata(
    csv_path: Path, matched_ids_header: str = "matched_ids"
) -> dict[str, list[tuple[str, ...]]]:
    """Map each metadata ID -> list of its 5-label tuples (one per row)."""
    encoding = detect_encoding(csv_path)
    label_tuples_by_id: dict[str, list[tuple[str, ...]]] = defaultdict(list)

    with csv_path.open(newline="", encoding=encoding) as handle:
        reader = csv.DictReader(handle)
        for column in [matched_ids_header, *LABEL_COLUMNS]:
            if column not in reader.fieldnames:
                raise ValueError(
                    f"Column {column!r} not found. Available: {reader.fieldnames}"
                )
        for row in reader:
            labels = tuple(str(row.get(c) or "").strip() for c in LABEL_COLUMNS)
            for token in str(row.get(matched_ids_header) or "").split(","):
                clean = normalize_id(token)
                if clean:
                    label_tuples_by_id[clean].append(labels)

    return label_tuples_by_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-dir", type=Path, default=Path("Output_Deidentified"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("matched_results_before0707/meta_data_0707version.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("usable_matches.csv")
    )
    args = parser.parse_args()

    label_tuples_by_id = load_metadata(args.metadata)

    rows_written = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["text_file", "file_id", "n_metadata_rows", "match_type", *LABEL_COLUMNS]
        )
        for path in sorted(args.text_dir.glob("*.txt")):
            file_id = extract_file_id(path.name)
            if not file_id or file_id not in label_tuples_by_id:
                continue
            tuples = label_tuples_by_id[file_id]
            # Usable only if all rows agree on the 5 labels.
            if len(set(tuples)) != 1:
                continue
            labels = tuples[0]
            match_type = "single_row" if len(tuples) == 1 else "multi_row_identical"
            writer.writerow([path.name, file_id, len(tuples), match_type, *labels])
            rows_written += 1

    print(f"Wrote {rows_written} usable matches to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
