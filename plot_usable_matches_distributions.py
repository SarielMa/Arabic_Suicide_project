#!/usr/bin/env python3
"""Plot label-value distributions for the usable matched text files.

Reads ``usable_matches.csv`` (produced by ``export_usable_matches.py``) and
writes one PNG bar plot per prediction label, showing how many of the usable
files carry each label value (TRUE / FALSE / does_not_apply / ...).

Uses Pillow for rendering (matplotlib is not installed in this environment),
following the style of ``plot_usable_label_distributions.py``.

Usage::

    python plot_usable_matches_distributions.py
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


LABEL_COLUMNS = [
    "Wish To Be Dead",
    "Non Specific Active Suicidal Thoughts",
    "Active Suicidal Ideation With Any Methods",
    "Active Suicidal With Some Intent To Act",
    "Active Suicidal Ideation With Specific Plan And Intent",
]

BLANK_LABEL = "<blank>"


def safe_filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return name or "plot"


def load_counts(input_path: Path) -> tuple[dict[str, Counter[str]], int]:
    """Return per-column value counts and the total number of usable rows."""
    counts: dict[str, Counter[str]] = {c: Counter() for c in LABEL_COLUMNS}
    total = 0
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in LABEL_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns in {input_path}: {missing}")
        for row in reader:
            total += 1
            for column in LABEL_COLUMNS:
                value = (row.get(column) or "").strip() or BLANK_LABEL
                counts[column][value] += 1
    return counts, total


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


def write_png_bar_plot(path: Path, title: str, counter: Counter[str], total: int) -> None:
    """Draw a horizontal bar plot of value counts, sorted by frequency."""
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    row_height = 54
    top = 86
    bottom = 42
    label_width = 250
    bar_left = label_width + 32
    bar_width = 520
    value_left = bar_left + bar_width + 18
    width = value_left + 160
    height = top + row_height * max(len(ordered), 1) + bottom
    max_count = max((count for _label, count in ordered), default=1)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(22, bold=True)
    subtitle_font = load_font(14)
    label_font = load_font(14)

    draw.text((28, 12), title, font=title_font, fill="#111827")
    draw.text((28, 45), f"Usable files: {total}", font=subtitle_font, fill="#4b5563")

    for index, (label, count) in enumerate(ordered):
        y = top + index * row_height
        percent = (count / total * 100) if total else 0.0
        bar_len = int((count / max_count) * bar_width) if max_count else 0

        draw.text((28, y + 8), label, font=label_font, fill="#111827")
        draw.rounded_rectangle(
            (bar_left, y, bar_left + bar_width, y + 30), radius=3, fill="#eef2f7"
        )
        draw.rounded_rectangle(
            (bar_left, y, bar_left + bar_len, y + 30), radius=3, fill="#2563eb"
        )
        draw.text(
            (value_left, y + 8),
            f"{count} ({percent:.1f}%)",
            font=label_font,
            fill="#111827",
        )

    image.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("usable_matches.csv"))
    parser.add_argument(
        "--plot-dir", type=Path, default=Path("usable_matches_distribution_plots")
    )
    args = parser.parse_args()

    counts, total = load_counts(args.input)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    for column in LABEL_COLUMNS:
        path = args.plot_dir / f"{safe_filename(column)}.png"
        write_png_bar_plot(path, column, counts[column], total)
        print(f"  wrote {path}")

    print(f"Total usable files: {total}")
    print(f"Wrote {len(LABEL_COLUMNS)} PNG plots to: {args.plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
