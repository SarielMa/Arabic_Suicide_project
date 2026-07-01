#!/usr/bin/env python3
"""Match deidentified text filenames to metadata labels.

The text filename ID is the final underscore-delimited block before the
``_output_transcription_output_deidentified.txt`` suffix. Metadata rows may
contain one or more comma-separated IDs in ``matched_ids``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from zipfile import ZipFile


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
DEFAULT_SUFFIX = "_output_transcription_output_deidentified.txt"


@dataclass(frozen=True)
class TextFile:
    path: Path
    file_id: str


@dataclass(frozen=True)
class MetadataRecord:
    row_number: int
    matched_ids: str
    labels: dict[str, str]


def qname(name: str) -> str:
    return f"{{{MAIN_NS}}}{name}"


def column_letters_to_index(letters: str) -> int:
    value = 0
    for char in letters.strip().upper():
        if not ("A" <= char <= "Z"):
            raise ValueError(f"Invalid Excel column: {letters!r}")
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def index_to_column_letters(index: int) -> str:
    if index < 0:
        raise ValueError(f"Invalid zero-based column index: {index}")
    letters = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def cell_ref_to_index(cell_ref: str) -> int:
    letters = "".join(char for char in cell_ref if char.isalpha())
    return column_letters_to_index(letters)


def parse_column_range(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(":")]
    if len(parts) == 1:
        return [column_letters_to_index(parts[0])]
    if len(parts) != 2:
        raise ValueError(f"Invalid column range: {value!r}")
    start = column_letters_to_index(parts[0])
    end = column_letters_to_index(parts[1])
    if start > end:
        raise ValueError(f"Column range starts after it ends: {value!r}")
    return list(range(start, end + 1))


def normalize_id(value: object) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text if re.fullmatch(r"\d+", text) else ""


def split_matched_ids(value: object) -> list[str]:
    ids: list[str] = []
    for token in str(value or "").split(","):
        matched_id = normalize_id(token)
        if matched_id:
            ids.append(matched_id)
    return ids


def extract_file_id(path: Path, suffix: str) -> str:
    name = path.name
    prefix = name[: -len(suffix)] if name.endswith(suffix) else path.stem
    return normalize_id(prefix.rsplit("_", 1)[-1])


def scan_text_files(text_dir: Path, suffix: str) -> tuple[list[TextFile], list[Path]]:
    if not text_dir.exists():
        raise FileNotFoundError(f"Text directory does not exist: {text_dir}")
    if not text_dir.is_dir():
        raise NotADirectoryError(f"Text path is not a directory: {text_dir}")

    text_files: list[TextFile] = []
    invalid_files: list[Path] = []
    for path in sorted(text_dir.glob("*.txt")):
        file_id = extract_file_id(path, suffix)
        if file_id:
            text_files.append(TextFile(path=path, file_id=file_id))
        else:
            invalid_files.append(path)
    return text_files, invalid_files


def load_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    shared_strings: list[str] = []
    with zf.open("xl/sharedStrings.xml") as handle:
        for _event, elem in ET.iterparse(handle, events=("end",)):
            if elem.tag == qname("si"):
                shared_strings.append(
                    "".join(text.text or "" for text in elem.iter(qname("t")))
                )
                elem.clear()
    return shared_strings


def cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "s":
        value = cell.find(qname("v"))
        if value is None or value.text is None:
            return ""
        return shared_strings[int(value.text)]
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.iter(qname("t")))

    value = cell.find(qname("v"))
    return value.text if value is not None and value.text is not None else ""


def workbook_sheets(zf: ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    relationships = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in relationships}

    sheets: dict[str, str] = {}
    for sheet in workbook.findall(f"{qname('sheets')}/{qname('sheet')}"):
        rel_id = sheet.attrib[f"{{{REL_NS}}}id"]
        target = rel_targets[rel_id].lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        sheets[sheet.attrib["name"]] = target
    return sheets


def iter_sheet_rows(
    zf: ZipFile, sheet_path: str, shared_strings: list[str]
) -> Iterator[tuple[int, dict[int, str]]]:
    fallback_row_number = 0
    with zf.open(sheet_path) as handle:
        for _event, elem in ET.iterparse(handle, events=("end",)):
            if elem.tag != qname("row"):
                continue

            fallback_row_number += 1
            row_number = int(elem.attrib.get("r", fallback_row_number))
            values: dict[int, str] = {}
            for cell in elem.findall(qname("c")):
                values[cell_ref_to_index(cell.attrib["r"])] = cell_value(
                    cell, shared_strings
                )
            elem.clear()
            yield row_number, values


def load_metadata_records(
    metadata_path: Path,
    sheet_name: str | None,
    matched_ids_header: str,
    label_columns: list[int],
) -> tuple[dict[str, list[MetadataRecord]], list[str], str]:
    with ZipFile(metadata_path) as zf:
        sheets = workbook_sheets(zf)
        if sheet_name is None:
            selected_sheet_name = next(iter(sheets))
        elif sheet_name in sheets:
            selected_sheet_name = sheet_name
        else:
            available = ", ".join(sheets)
            raise ValueError(f"Sheet {sheet_name!r} not found. Available: {available}")

        shared_strings = load_shared_strings(zf)
        rows = iter_sheet_rows(zf, sheets[selected_sheet_name], shared_strings)
        try:
            _header_row_number, header_row = next(rows)
        except StopIteration as exc:
            raise ValueError(f"Sheet {selected_sheet_name!r} is empty") from exc

        headers = {index: value.strip() for index, value in header_row.items()}
        matched_ids_column = None
        for index, header in headers.items():
            if header.lower() == matched_ids_header.lower():
                matched_ids_column = index
                break
        if matched_ids_column is None:
            raise ValueError(f"Could not find metadata column {matched_ids_header!r}")

        label_headers: list[str] = []
        for index in label_columns:
            label_headers.append(headers.get(index) or index_to_column_letters(index))

        records_by_id: dict[str, list[MetadataRecord]] = defaultdict(list)
        for row_number, row in rows:
            matched_ids = row.get(matched_ids_column, "")
            labels = {header: row.get(index, "") for header, index in zip(label_headers, label_columns)}
            for matched_id in split_matched_ids(matched_ids):
                records_by_id[matched_id].append(
                    MetadataRecord(
                        row_number=row_number,
                        matched_ids=matched_ids,
                        labels=labels,
                    )
                )

    return records_by_id, label_headers, selected_sheet_name


def write_matches(
    output_path: Path,
    text_files: list[TextFile],
    records_by_id: dict[str, list[MetadataRecord]],
    label_headers: list[str],
) -> tuple[int, set[str], dict[str, int]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["text_file", "file_id", "metadata_row", "matched_ids", *label_headers]
    matched_file_ids: set[str] = set()
    duplicate_metadata_counts: dict[str, int] = {}
    output_rows = 0

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for text_file in text_files:
            records = records_by_id.get(text_file.file_id, [])
            if not records:
                continue
            matched_file_ids.add(text_file.file_id)
            if len(records) > 1:
                duplicate_metadata_counts[text_file.file_id] = len(records)
            for record in records:
                writer.writerow(
                    {
                        "text_file": str(text_file.path),
                        "file_id": text_file.file_id,
                        "metadata_row": record.row_number,
                        "matched_ids": record.matched_ids,
                        **record.labels,
                    }
                )
                output_rows += 1

    return output_rows, matched_file_ids, duplicate_metadata_counts


def write_unmatched(
    output_path: Path,
    text_files: list[TextFile],
    invalid_files: list[Path],
    matched_file_ids: set[str],
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unmatched_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text_file", "file_id", "reason"])
        writer.writeheader()
        for path in invalid_files:
            writer.writerow(
                {"text_file": str(path), "file_id": "", "reason": "invalid_filename_id"}
            )
            unmatched_count += 1
        for text_file in text_files:
            if text_file.file_id not in matched_file_ids:
                writer.writerow(
                    {
                        "text_file": str(text_file.path),
                        "file_id": text_file.file_id,
                        "reason": "no_metadata_match",
                    }
                )
                unmatched_count += 1
    return unmatched_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match deidentified text files to metadata AP:AT labels."
    )
    parser.add_argument("--text-dir", type=Path, default=Path("Output_Deidentified"))
    parser.add_argument("--metadata", type=Path, default=Path("meta_data.xlsx"))
    parser.add_argument("--sheet", default=None, help="Excel sheet name; defaults to the first sheet.")
    parser.add_argument("--matched-ids-header", default="matched_ids")
    parser.add_argument("--label-columns", default="AP:AT")
    parser.add_argument("--filename-suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--output", type=Path, default=Path("matched_text_metadata_labels.csv"))
    parser.add_argument(
        "--unmatched-output",
        type=Path,
        default=Path("unmatched_text_metadata_labels.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    label_columns = parse_column_range(args.label_columns)

    text_files, invalid_files = scan_text_files(args.text_dir, args.filename_suffix)
    records_by_id, label_headers, sheet_name = load_metadata_records(
        metadata_path=args.metadata,
        sheet_name=args.sheet,
        matched_ids_header=args.matched_ids_header,
        label_columns=label_columns,
    )
    output_rows, matched_file_ids, duplicate_metadata_counts = write_matches(
        output_path=args.output,
        text_files=text_files,
        records_by_id=records_by_id,
        label_headers=label_headers,
    )
    unmatched_count = write_unmatched(
        output_path=args.unmatched_output,
        text_files=text_files,
        invalid_files=invalid_files,
        matched_file_ids=matched_file_ids,
    )

    print(f"Metadata sheet: {sheet_name}")
    print("Label columns:", ", ".join(label_headers))
    print(f"Text files scanned: {len(text_files) + len(invalid_files)}")
    print(f"Matched text files: {len(matched_file_ids)}")
    print(f"Output rows: {output_rows}")
    print(f"Unmatched/invalid text files: {unmatched_count}")
    print(f"Text IDs with multiple metadata rows: {len(duplicate_metadata_counts)}")
    print(f"Wrote matches: {args.output}")
    print(f"Wrote unmatched report: {args.unmatched_output}")

    if invalid_files:
        print("WARNING: some text files did not have a valid numeric filename ID.", file=sys.stderr)
    if duplicate_metadata_counts:
        print("WARNING: some text IDs matched multiple metadata rows.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
