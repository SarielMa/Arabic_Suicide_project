#!/usr/bin/env python3
"""Rebuild the five task datasets in English from a translations JSONL.

Emits processed_datasets_en/<task>/{train,test}.jsonl with the SAME file_ids,
splits, instructions, outputs and labels as the Arabic originals -- only the
``input`` transcript is swapped for its English translation. Because the schema is
identical, train.py / evaluate.py consume it unchanged via ``--data-dir``.

Refuses to build if any transcript is missing a translation, and by default
refuses to include transcripts whose translation failed the quality checks --
a summarized or refused translation silently deletes the label's evidence, and a
model trained on it would look like a language effect. Override deliberately with
--allow-flagged if you have inspected them.

Usage::

    python build_english_data.py --pred translations/qwen2.5-72b-instruct.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from translate import check_translation, hard_flags


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", type=Path, required=True, help="translations JSONL")
    p.add_argument("--data-dir", type=Path, default=Path("processed_datasets"))
    p.add_argument("--out-dir", type=Path, default=Path("processed_datasets_en"))
    p.add_argument("--allow-flagged", action="store_true",
                   help="Include transcripts whose translation failed the QC checks.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    translations = {r["file_id"]: r for r in read_jsonl(args.pred)}
    # Flags are RE-DERIVED here rather than read from the record. The stored list was
    # written by whatever version of check_translation ran at translation time, so a
    # check added later would never be applied to an existing file -- which is exactly
    # how 29 looped translations reached the English data: they were written before
    # `degenerate` existed, and their stored flags say only 'pers'/'too_long'.
    # Re-checking costs nothing and makes the build fail closed instead of trusting a
    # stale annotation. Falls back to the stored flags if the source text is absent.
    def current_flags(rec: dict) -> list[str]:
        if "arabic" in rec and "english" in rec:
            return check_translation(rec["arabic"], rec["english"])
        return rec.get("flags", [])

    flags_now = {fid: current_flags(r) for fid, r in translations.items()}
    # Only hard flags (refusal / summarization / truncation / degeneration /
    # untranslated Arabic) block the build: those delete the evidence the label rests
    # on. A soft flag -- a dropped <PERS> placeholder, a verbose rendering -- leaves
    # the risk content intact and is not a reason to withhold the transcript.
    damaged = {fid for fid, f in flags_now.items() if hard_flags(f)}
    soft = {fid for fid, f in flags_now.items() if f and fid not in damaged}
    print(f"Translations: {len(translations)}  "
          f"(content-damaged: {len(damaged)}, cosmetic flags: {len(soft)})")

    tasks = sorted(p for p in args.data_dir.iterdir() if p.is_dir())
    missing: set[str] = set()
    for task_dir in tasks:
        for split in ("train", "test"):
            path = task_dir / f"{split}.jsonl"
            if path.exists():
                for rec in read_jsonl(path):
                    if rec["file_id"] not in translations:
                        missing.add(rec["file_id"])
    if missing:
        raise SystemExit(
            f"{len(missing)} transcripts have no translation (e.g. "
            f"{sorted(missing)[:5]}). Finish translate.py before building."
        )
    if damaged and not args.allow_flagged:
        by_kind: dict[str, int] = {}
        for fid in damaged:
            for f in hard_flags(flags_now[fid]):
                by_kind[f.split("(")[0]] = by_kind.get(f.split("(")[0], 0) + 1
        raise SystemExit(
            f"{len(damaged)} translations lost content "
            f"({', '.join(f'{k}={v}' for k, v in sorted(by_kind.items()))}).\n"
            f"Inspect them:\n"
            f"  python inspect_translations.py --pred {args.pred} --only-flagged --show 5\n"
            f"Re-translate them in place:\n"
            f"  python translate.py --out {args.pred} --redo\n"
            f"Or pass --allow-flagged to include them anyway (not recommended: a "
            f"degenerate translation is filler, not content)."
        )

    print(f"\n{'task':<52} {'split':<6} {'n':>5} {'pos':>5}")
    for task_dir in tasks:
        for split in ("train", "test"):
            src = task_dir / f"{split}.jsonl"
            if not src.exists():
                continue
            records = read_jsonl(src)
            out_path = args.out_dir / task_dir.name / f"{split}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            n_pos = 0
            with out_path.open("w", encoding="utf-8") as handle:
                for rec in records:
                    # Only the transcript changes; label/instruction/split are carried
                    # over verbatim, so English vs Arabic is a like-for-like comparison.
                    new = dict(rec)
                    new["input"] = translations[rec["file_id"]]["english"]
                    n_pos += int(rec["label"])
                    handle.write(json.dumps(new, ensure_ascii=False) + "\n")
            print(f"{task_dir.name:<52} {split:<6} {len(records):>5} {n_pos:>5}")

    print(f"\nWrote {args.out_dir}/")
    print("Run the pipeline on it with:  DATA_DIR=processed_datasets_en ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
