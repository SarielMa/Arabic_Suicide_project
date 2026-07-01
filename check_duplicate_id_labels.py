#!/usr/bin/env python3
"""Check whether duplicate file IDs have identical label columns."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("matched_text_metadata_labels.csv")
DEFAULT_OUTPUT = Path("duplicate_id_label_check.csv")
DEFAULT_MISMATCH_OUTPUT = Path("duplicate_id_label_mismatches.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Group matched metadata rows by file_id and check whether duplicate "
            "IDs have the same prediction/label values."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--id-column", default="file_id")
    parser.add_argument(
        "--prediction-columns",
        nargs="+",
        default=None,
        help=(
            "Prediction/label columns to compare. Defaults to all columns after "
            "'matched_ids'."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mismatch-output", type=Path, default=DEFAULT_MISMATCH_OUTPUT)
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with status 1 when duplicate IDs have different predictions.",
    )
    return parser.parse_args()


def infer_prediction_columns(fieldnames: list[str]) -> list[str]:
    if "matched_ids" not in fieldnames:
        raise ValueError(
            "Could not infer prediction columns because 'matched_ids' is missing."
        )
    start = fieldnames.index("matched_ids") + 1
    prediction_columns = fieldnames[start:]
    if not prediction_columns:
        raise ValueError("No prediction columns found after 'matched_ids'.")
    return prediction_columns


def label_signature(row: dict[str, str], prediction_columns: list[str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in prediction_columns)


def main() -> int:
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header: {args.input}")
        fieldnames = reader.fieldnames
        if args.id_column not in fieldnames:
            raise ValueError(f"ID column {args.id_column!r} not found in {args.input}")

        prediction_columns = (
            args.prediction_columns
            if args.prediction_columns is not None
            else infer_prediction_columns(fieldnames)
        )
        missing_columns = [col for col in prediction_columns if col not in fieldnames]
        if missing_columns:
            raise ValueError(f"Prediction columns not found: {missing_columns}")

        rows_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            rows_by_id[row[args.id_column]].append(row)

    duplicate_groups = {
        file_id: rows for file_id, rows in rows_by_id.items() if len(rows) > 1
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        args.id_column,
        "row_count",
        "unique_prediction_count",
        "predictions_same",
        "metadata_rows",
        "matched_ids_values",
        *prediction_columns,
    ]

    mismatch_fields = fieldnames + ["prediction_variant_index"]
    inconsistent_ids: set[str] = set()
    summary_rows = []
    mismatch_rows = []

    for file_id, rows in sorted(
        duplicate_groups.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]
    ):
        signatures = []
        signature_to_variant: dict[tuple[str, ...], int] = {}
        for row in rows:
            signature = label_signature(row, prediction_columns)
            signatures.append(signature)
            if signature not in signature_to_variant:
                signature_to_variant[signature] = len(signature_to_variant) + 1

        predictions_same = len(signature_to_variant) == 1
        if not predictions_same:
            inconsistent_ids.add(file_id)
            for row, signature in zip(rows, signatures):
                mismatch_rows.append(
                    {
                        **row,
                        "prediction_variant_index": str(signature_to_variant[signature]),
                    }
                )

        first_signature = signatures[0]
        summary_rows.append(
            {
                args.id_column: file_id,
                "row_count": str(len(rows)),
                "unique_prediction_count": str(len(signature_to_variant)),
                "predictions_same": str(predictions_same).lower(),
                "metadata_rows": ",".join(row.get("metadata_row", "") for row in rows),
                "matched_ids_values": " | ".join(
                    dict.fromkeys(row.get("matched_ids", "") for row in rows)
                ),
                **dict(zip(prediction_columns, first_signature)),
            }
        )

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    with args.mismatch_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=mismatch_fields)
        writer.writeheader()
        writer.writerows(mismatch_rows)

    print(f"Input rows: {sum(len(rows) for rows in rows_by_id.values())}")
    print(f"Unique IDs: {len(rows_by_id)}")
    print(f"Duplicate IDs: {len(duplicate_groups)}")
    print(f"Duplicate IDs with same predictions: {len(duplicate_groups) - len(inconsistent_ids)}")
    print(f"Duplicate IDs with different predictions: {len(inconsistent_ids)}")
    print("Prediction columns:", ", ".join(prediction_columns))
    print(f"Wrote duplicate summary: {args.output}")
    print(f"Wrote mismatch details: {args.mismatch_output}")
    return 1 if args.fail_on_mismatch and inconsistent_ids else 0


if __name__ == "__main__":
    raise SystemExit(main())
