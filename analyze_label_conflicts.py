#!/usr/bin/env python3
"""Analyze deidentified text files against the metadata CSV.

This reproduces the three analyses we ran:

  1. How many text-file IDs exist in the metadata (`matched_ids`) and how many
     do not.
  2. Among the matched files, how many map to exactly one metadata row vs.
     multiple rows.
  3. Among the multi-row files, how many have IDENTICAL prediction labels across
     all their rows, and how many have AT LEAST ONE conflicting label.

The 5 prediction-label columns (spreadsheet columns AO..AS) are:
  - Wish To Be Dead
  - Non Specific Active Suicidal Thoughts
  - Active Suicidal Ideation With Any Methods
  - Active Suicidal With Some Intent To Act
  - Active Suicidal Ideation With Specific Plan And Intent

How the file ID is derived
--------------------------
A text filename looks like::

    <date>_<epoch>_<agent>_<ID>_output_transcription_output_deidentified.txt

The ID is the underscore block immediately BEFORE ``_output`` (e.g.
``..._4144_170_output_...`` -> ``170``). Metadata rows store one or more
comma-separated IDs in the ``matched_ids`` column, so one metadata row can
reference several IDs, and the same ID can appear in several rows.

Usage::

    python analyze_label_conflicts.py
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


SUFFIX = "_output_transcription_output_deidentified.txt"

# The 5 prediction-label columns, matched by header name (columns AO..AS).
LABEL_COLUMNS = [
    "Wish To Be Dead",
    "Non Specific Active Suicidal Thoughts",
    "Active Suicidal Ideation With Any Methods",
    "Active Suicidal With Some Intent To Act",
    "Active Suicidal Ideation With Specific Plan And Intent",
]


def normalize_id(value: object) -> str:
    """Return a clean integer-string ID, or "" if the token is not numeric.

    Handles Excel float artifacts like "170.0" -> "170".
    """
    text = str(value or "").strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text if re.fullmatch(r"\d+", text) else ""


def extract_file_id(name: str) -> str:
    """Extract the ID (block before ``_output``) from a text filename."""
    prefix = name[: -len(SUFFIX)] if name.endswith(SUFFIX) else Path(name).stem
    return normalize_id(prefix.rsplit("_", 1)[-1])


def detect_encoding(path: Path) -> str:
    """Return an encoding that can decode the file.

    The metadata is exported from Excel and contains Windows-1252 bytes
    (e.g. curly quotes), so we fall back from UTF-8 to cp1252.
    """
    raw = path.read_bytes()
    try:
        raw.decode("utf-8-sig")
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "cp1252"


def load_metadata(
    csv_path: Path, matched_ids_header: str = "matched_ids"
) -> dict[str, list[tuple[str, ...]]]:
    """Map each metadata ID -> list of its 5-label tuples (one per row).

    A metadata row that lists several comma-separated IDs contributes its label
    tuple to each of those IDs.
    """
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


def scan_text_files(text_dir: Path) -> list[tuple[str, str]]:
    """Return (filename, file_id) for every .txt file, skipping invalid IDs."""
    results: list[tuple[str, str]] = []
    for path in sorted(text_dir.glob("*.txt")):
        file_id = extract_file_id(path.name)
        if file_id:
            results.append((path.name, file_id))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-dir", type=Path, default=Path("Output_Deidentified"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("matched_results_before0707/meta_data_0707version.csv"),
    )
    parser.add_argument(
        "--conflict-output",
        type=Path,
        default=Path("multi_row_label_conflicts.csv"),
        help="Write per-ID conflict details for the multi-row matched files.",
    )
    args = parser.parse_args()

    label_tuples_by_id = load_metadata(args.metadata)
    metadata_ids = set(label_tuples_by_id)
    text_files = scan_text_files(args.text_dir)

    # ---- Analysis 1: files with vs. without a metadata ID -------------------
    matched = [(name, fid) for name, fid in text_files if fid in metadata_ids]
    not_matched = [(name, fid) for name, fid in text_files if fid not in metadata_ids]

    print("=" * 60)
    print("Analysis 1: text files vs. metadata IDs")
    print("=" * 60)
    print(f"Total text files scanned:   {len(text_files)}")
    print(f"Files WITH metadata ID:     {len(matched)}")
    print(f"Files WITHOUT metadata ID:  {len(not_matched)}")

    # ---- Analysis 2: single-row vs. multi-row matches -----------------------
    single_row = [(n, f) for n, f in matched if len(label_tuples_by_id[f]) == 1]
    multi_row = [(n, f) for n, f in matched if len(label_tuples_by_id[f]) > 1]

    print()
    print("=" * 60)
    print("Analysis 2: matched files by number of metadata rows")
    print("=" * 60)
    print(f"Matched files total:        {len(matched)}")
    print(f"  match exactly ONE row:    {len(single_row)}")
    print(f"  match MULTIPLE rows:      {len(multi_row)}")

    # ---- Analysis 3: identical vs. conflicting labels (multi-row) -----------
    # Compare the 5-label tuples across a file's rows. If every row has the same
    # tuple -> identical; otherwise at least one label conflicts.
    identical: list[tuple[str, str]] = []
    conflicting: list[tuple[str, str]] = []
    for name, fid in multi_row:
        distinct_tuples = set(label_tuples_by_id[fid])
        if len(distinct_tuples) == 1:
            identical.append((name, fid))
        else:
            conflicting.append((name, fid))

    print()
    print("=" * 60)
    print("Analysis 3: label agreement among multi-row files")
    print("=" * 60)
    print(f"Multi-row matched files:            {len(multi_row)}")
    print(f"  IDENTICAL labels (all 5 same):    {len(identical)}")
    print(f"  AT LEAST ONE conflicting label:   {len(conflicting)}")

    # Write conflict details so they can be inspected.
    with args.conflict_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_id", "text_file", "n_rows", "distinct_label_tuple", *LABEL_COLUMNS])
        for name, fid in sorted(conflicting, key=lambda x: int(x[1])):
            tuples = label_tuples_by_id[fid]
            for i, tup in enumerate(sorted(set(tuples))):
                writer.writerow([fid, name if i == 0 else "", len(tuples), i + 1, *tup])
    print()
    print(f"Wrote conflict details:     {args.conflict_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
