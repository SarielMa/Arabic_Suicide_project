#!/usr/bin/env python3
"""Create distribution plots for usable AP:AT label data.

Usable rows are defined at the text-file ID level:
  * IDs with one metadata row are kept.
  * IDs with multiple metadata rows are kept only when all prediction columns
    have identical values across those rows.
  * IDs with conflicting prediction values are excluded.

The script writes a deduplicated usable CSV, a long-format count CSV, and one
PNG bar plot per prediction column.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_INPUT = Path("matched_text_metadata_labels.csv")
DEFAULT_OUTPUT_DIR = Path("usable_label_distribution_png_plots")
DEFAULT_USABLE_OUTPUT = Path("usable_text_metadata_labels.csv")
DEFAULT_COUNTS_OUTPUT = Path("usable_label_distribution_counts.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot label distributions for usable matched text-file IDs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--id-column", default="file_id")
    parser.add_argument(
        "--prediction-columns",
        nargs="+",
        default=None,
        help="Prediction/label columns. Defaults to all columns after 'matched_ids'.",
    )
    parser.add_argument("--usable-output", type=Path, default=DEFAULT_USABLE_OUTPUT)
    parser.add_argument("--counts-output", type=Path, default=DEFAULT_COUNTS_OUTPUT)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--blank-label", default="<blank>")
    return parser.parse_args()


def infer_prediction_columns(fieldnames: list[str]) -> list[str]:
    if "matched_ids" not in fieldnames:
        raise ValueError("Cannot infer prediction columns: 'matched_ids' is missing.")
    prediction_columns = fieldnames[fieldnames.index("matched_ids") + 1 :]
    if not prediction_columns:
        raise ValueError("No prediction columns found after 'matched_ids'.")
    return prediction_columns


def label_signature(row: dict[str, str], prediction_columns: list[str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in prediction_columns)


def safe_filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return name or "plot"


def load_usable_rows(
    input_path: Path,
    id_column: str,
    prediction_columns: list[str] | None,
) -> tuple[list[dict[str, str]], list[str], dict[str, int]]:
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header: {input_path}")
        fieldnames = reader.fieldnames
        if id_column not in fieldnames:
            raise ValueError(f"ID column {id_column!r} not found in {input_path}")

        if prediction_columns is None:
            prediction_columns = infer_prediction_columns(fieldnames)
        missing = [column for column in prediction_columns if column not in fieldnames]
        if missing:
            raise ValueError(f"Prediction columns not found: {missing}")

        rows_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            rows_by_id[row[id_column]].append(row)

    usable_rows: list[dict[str, str]] = []
    stats = {
        "input_rows": sum(len(rows) for rows in rows_by_id.values()),
        "matched_ids": len(rows_by_id),
        "single_match_ids": 0,
        "duplicate_same_label_ids": 0,
        "duplicate_conflict_ids": 0,
    }

    for file_id, rows in sorted(
        rows_by_id.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]
    ):
        signatures = {label_signature(row, prediction_columns) for row in rows}
        if len(rows) == 1:
            stats["single_match_ids"] += 1
            status = "single_match"
        elif len(signatures) == 1:
            stats["duplicate_same_label_ids"] += 1
            status = "duplicate_same_labels"
        else:
            stats["duplicate_conflict_ids"] += 1
            continue

        kept = rows[0].copy()
        kept["source_match_status"] = status
        kept["source_row_count"] = str(len(rows))
        kept["metadata_rows"] = ",".join(row.get("metadata_row", "") for row in rows)
        kept["matched_ids_values"] = " | ".join(
            dict.fromkeys(row.get("matched_ids", "") for row in rows)
        )
        usable_rows.append(kept)

    stats["usable_ids"] = len(usable_rows)
    return usable_rows, prediction_columns, stats


def write_usable_csv(
    path: Path, rows: list[dict[str, str]], prediction_columns: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "text_file",
        "file_id",
        "source_match_status",
        "source_row_count",
        "metadata_rows",
        "matched_ids",
        "matched_ids_values",
        *prediction_columns,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def count_labels(
    rows: list[dict[str, str]], prediction_columns: list[str], blank_label: str
) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = {}
    for column in prediction_columns:
        counter: Counter[str] = Counter()
        for row in rows:
            value = row.get(column, "").strip() or blank_label
            counter[value] += 1
        counts[column] = counter
    return counts


def write_counts_csv(path: Path, counts: dict[str, Counter[str]], total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["prediction_column", "label_value", "count", "percent"],
        )
        writer.writeheader()
        for column, counter in counts.items():
            for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
                percent = (count / total * 100) if total else 0.0
                writer.writerow(
                    {
                        "prediction_column": column,
                        "label_value": label,
                        "count": count,
                        "percent": f"{percent:.2f}",
                    }
                )


def wrap_text(text: str, width: int = 24) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        next_len = len(word) if not current else current_len + 1 + len(word)
        if current and next_len > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = next_len
    if current:
        lines.append(" ".join(current))
    return lines


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def write_png_bar_plot(
    path: Path,
    title: str,
    counter: Counter[str],
    total: int,
) -> None:
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    row_height = 54
    top = 86
    bottom = 42
    label_width = 250
    bar_left = label_width + 32
    bar_width = 520
    value_left = bar_left + bar_width + 18
    width = value_left + 145
    height = top + row_height * max(len(ordered), 1) + bottom
    max_count = max((count for _label, count in ordered), default=1)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(22, bold=True)
    subtitle_font = load_font(14)
    label_font = load_font(13)

    draw.text((28, 12), title, font=title_font, fill="#111827")
    draw.text((28, 45), f"Usable text-file IDs: {total}", font=subtitle_font, fill="#4b5563")

    for index, (label, count) in enumerate(ordered):
        y = top + index * row_height
        percent = (count / total * 100) if total else 0.0
        bar_len = int((count / max_count) * bar_width) if max_count else 0

        label_lines = wrap_text(label)
        line_start_y = y + 8 if len(label_lines) > 1 else y + 10
        for line_index, line in enumerate(label_lines[:2]):
            draw.text(
                (28, line_start_y + line_index * 16),
                line,
                font=label_font,
                fill="#111827",
            )
        if len(label_lines) > 2:
            draw.text(
                (28, line_start_y + 32),
                "...",
                font=label_font,
                fill="#111827",
            )

        draw.rounded_rectangle(
            (bar_left, y, bar_left + bar_width, y + 30),
            radius=3,
            fill="#eef2f7",
        )
        draw.rounded_rectangle(
            (bar_left, y, bar_left + bar_len, y + 30),
            radius=3,
            fill="#2563eb",
        )
        draw.text(
            (value_left, y + 8),
            f"{count} ({percent:.1f}%)",
            font=label_font,
            fill="#111827",
        )

    image.save(path)


def write_plots(
    plot_dir: Path,
    counts: dict[str, Counter[str]],
    total: int,
) -> list[Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []
    for column, counter in counts.items():
        path = plot_dir / f"{safe_filename(column)}.png"
        write_png_bar_plot(path, column, counter, total)
        plot_paths.append(path)
    return plot_paths


def main() -> int:
    args = parse_args()
    usable_rows, prediction_columns, stats = load_usable_rows(
        input_path=args.input,
        id_column=args.id_column,
        prediction_columns=args.prediction_columns,
    )
    counts = count_labels(usable_rows, prediction_columns, args.blank_label)

    write_usable_csv(args.usable_output, usable_rows, prediction_columns)
    write_counts_csv(args.counts_output, counts, len(usable_rows))
    plot_paths = write_plots(args.plot_dir, counts, len(usable_rows))

    print(f"Input rows: {stats['input_rows']}")
    print(f"Matched text-file IDs: {stats['matched_ids']}")
    print(f"Single-match usable IDs: {stats['single_match_ids']}")
    print(f"Duplicate same-label usable IDs: {stats['duplicate_same_label_ids']}")
    print(f"Duplicate conflict IDs excluded: {stats['duplicate_conflict_ids']}")
    print(f"Usable IDs plotted: {stats['usable_ids']}")
    print("Prediction columns:", ", ".join(prediction_columns))
    print(f"Wrote usable CSV: {args.usable_output}")
    print(f"Wrote distribution counts: {args.counts_output}")
    print(f"Wrote {len(plot_paths)} PNG plots to: {args.plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
