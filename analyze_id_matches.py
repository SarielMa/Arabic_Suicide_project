#!/usr/bin/env python3
"""Analyze how many deidentified text file IDs are present in the metadata.

Each text filename looks like::

    <date>_<epoch>_<agent>_<id>_output_transcription_output_deidentified.txt

The file ID is the underscore block immediately before ``_output``. Metadata
rows store one or more comma-separated IDs in the ``matched_ids`` column.

Usage::

    python analyze_id_matches.py
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


SUFFIX = "_output_transcription_output_deidentified.txt"


def normalize_id(value: object) -> str:
    """Return a clean integer-string ID, or "" if the token is not numeric."""
    text = str(value or "").strip()
    if re.fullmatch(r"\d+\.0+", text):  # handle values like "170.0"
        text = text.split(".", 1)[0]
    return text if re.fullmatch(r"\d+", text) else ""


def extract_file_id(name: str) -> str:
    """Extract the ID (block before ``_output``) from a text filename."""
    prefix = name[: -len(SUFFIX)] if name.endswith(SUFFIX) else Path(name).stem
    return normalize_id(prefix.rsplit("_", 1)[-1])


def load_metadata_ids(csv_path: Path, matched_ids_header: str = "matched_ids") -> set[str]:
    """Collect every individual ID appearing in the matched_ids column."""
    ids: set[str] = set()
    # The metadata is exported from Excel and contains Windows-1252 bytes
    # (e.g. curly quotes), so fall back from UTF-8 to cp1252.
    raw = csv_path.read_bytes()
    try:
        raw.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        encoding = "cp1252"
    with csv_path.open(newline="", encoding=encoding) as handle:
        reader = csv.DictReader(handle)
        if matched_ids_header not in reader.fieldnames:
            raise ValueError(
                f"Column {matched_ids_header!r} not found. Available: {reader.fieldnames}"
            )
        for row in reader:
            for token in str(row.get(matched_ids_header) or "").split(","):
                clean = normalize_id(token)
                if clean:
                    ids.add(clean)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text-dir", type=Path, default=Path("Output_Deidentified")
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("matched_results_before0707/meta_data_0707version.csv"),
    )
    parser.add_argument("--matched-ids-header", default="matched_ids")
    parser.add_argument(
        "--unmatched-output",
        type=Path,
        default=Path("files_not_in_metadata.csv"),
        help="Write the list of text files whose ID is not in the metadata.",
    )
    args = parser.parse_args()

    metadata_ids = load_metadata_ids(args.metadata, args.matched_ids_header)

    matched: list[tuple[str, str]] = []       # (filename, file_id)
    not_matched: list[tuple[str, str]] = []    # (filename, file_id)
    invalid: list[str] = []                    # filenames without a numeric ID

    for path in sorted(args.text_dir.glob("*.txt")):
        file_id = extract_file_id(path.name)
        if not file_id:
            invalid.append(path.name)
        elif file_id in metadata_ids:
            matched.append((path.name, file_id))
        else:
            not_matched.append((path.name, file_id))

    total = len(matched) + len(not_matched) + len(invalid)

    # Count distinct IDs for context (some files can share an ID).
    matched_unique = {fid for _, fid in matched}
    not_matched_unique = {fid for _, fid in not_matched}

    print(f"Metadata file:            {args.metadata}")
    print(f"Distinct IDs in metadata: {len(metadata_ids)}")
    print(f"Text directory:           {args.text_dir}")
    print("-" * 50)
    print(f"Total text files scanned: {total}")
    print(f"Files WITH metadata ID:   {len(matched)}  (distinct IDs: {len(matched_unique)})")
    print(f"Files WITHOUT metadata ID:{len(not_matched)}  (distinct IDs: {len(not_matched_unique)})")
    if invalid:
        print(f"Files with invalid ID:    {len(invalid)}")

    # Save the unmatched files so they can be inspected.
    with args.unmatched_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text_file", "file_id", "reason"])
        for name, fid in not_matched:
            writer.writerow([name, fid, "id_not_in_metadata"])
        for name in invalid:
            writer.writerow([name, "", "invalid_filename_id"])
    print("-" * 50)
    print(f"Wrote unmatched report:   {args.unmatched_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
