#!/usr/bin/env python3
"""Build the coarse two-level (med_risk / high_risk) datasets from the 5 tasks.

A merged label is the OR of its constituent task labels:

    med_risk  = wish_to_be_dead OR non_specific_active_suicidal_thoughts
    high_risk = any_methods OR some_intent_to_act OR specific_plan_and_intent

Two properties of the source data drive the logic here.

**The 5 tasks annotate different subsets of calls.** Only 301 of the 333 calls
with any high_risk annotation carry all three labels. When a constituent label
is absent the OR may still be determined: ``T ? ?`` is True whatever the missing
values are. It is undetermined only when every *present* label is 0 and at least
one is missing (``F ? ?``), since the unrecorded value decides the outcome. We
keep every call whose OR is determined and drop the rest -- filling absent
labels with 0 would manufacture negatives in a risk-detection dataset.

**The per-task train/test splits were drawn independently.** The same call sits
in one task's train and another's test (e.g. 62 calls are in wish_to_be_dead
train and non_specific test), so the original splits cannot be carried over
without leaking. We draw a fresh file-level stratified split instead.

Arabic and English share one partition, keyed on file_id, so the two stay
parallel.

Usage::

    python build_merged_data.py                  # both languages, default dirs
    python build_merged_data.py --test-size 0.2 --seed 13
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from tasks import MERGED_TASKS, TASKS, build_instruction, label_to_text

SPLITS = ("train", "test")


def load_task_labels(src: Path) -> tuple[dict[str, dict[str, int]], dict[str, str]]:
    """Return ({file_id: {task_key: label}}, {file_id: transcript}) for one language."""
    labels: dict[str, dict[str, int]] = defaultdict(dict)
    transcripts: dict[str, str] = {}
    for task_key in TASKS:
        for split in SPLITS:
            path = src / task_key / f"{split}.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    rec = json.loads(line)
                    fid = rec["file_id"]
                    labels[fid][task_key] = int(rec["label"])
                    prev = transcripts.setdefault(fid, rec["input"])
                    if prev != rec["input"]:
                        raise ValueError(
                            f"file_id {fid} has different transcripts across tasks"
                        )
    return labels, transcripts


def merge_label(per_task: dict[str, int], components: list[str]) -> int | None:
    """OR the constituent labels; None when the result is undetermined.

    Undetermined means no present component is 1 *and* at least one component
    was never annotated -- the missing value would decide the label.
    """
    present = [t for t in components if t in per_task]
    if not present:
        return None
    if any(per_task[t] for t in present):
        return 1
    return 0 if len(present) == len(components) else None


def stratified_split(
    labelled: dict[str, int], test_size: float, seed: int
) -> dict[str, str]:
    """Assign each file_id to 'train' or 'test', stratified on the merged label."""
    assignment: dict[str, str] = {}
    rng = random.Random(seed)
    for label in (0, 1):
        # Sort before shuffling so the result depends only on the seed, not on
        # dict iteration order.
        group = sorted(fid for fid, lab in labelled.items() if lab == label)
        rng.shuffle(group)
        n_test = round(len(group) * test_size)
        for fid in group[:n_test]:
            assignment[fid] = "test"
        for fid in group[n_test:]:
            assignment[fid] = "train"
    return assignment


def write_split(
    path: Path,
    file_ids: list[str],
    labelled: dict[str, int],
    transcripts: dict[str, str],
    instruction: str,
) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_pos = 0
    with path.open("w", encoding="utf-8") as handle:
        for fid in file_ids:
            label = labelled[fid]
            n_pos += label
            item = {
                "file_id": fid,
                "instruction": instruction,
                "input": transcripts[fid],
                "output": label_to_text(label),
                "label": label,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(file_ids), n_pos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, default=Path("processed_datasets"))
    parser.add_argument("--out-dir", type=Path, default=Path("processed_datasets_merged"))
    parser.add_argument(
        "--en-src-dir", type=Path, default=Path("processed_datasets_en")
    )
    parser.add_argument(
        "--en-out-dir", type=Path, default=Path("processed_datasets_merged_en")
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude", type=Path, default=None,
        help="File of file_ids to drop, one per line ('#' comments allowed). "
             "Applied to BOTH languages before the split is drawn, so Arabic and "
             "English stay a like-for-like comparison.",
    )
    args = parser.parse_args()

    # Excluded calls are removed from both languages, not just the one with the
    # defect. Dropping them from English alone would leave the two corpora covering
    # different populations, and an Arabic-vs-English difference would no longer be
    # attributable to language. The cost is real -- a sound Arabic transcript is
    # discarded because its translation failed -- and it is paid deliberately to keep
    # the comparison controlled.
    excluded: set[str] = set()
    if args.exclude:
        for line in args.exclude.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                excluded.add(line)
        print(f"Exclusion list: {len(excluded)} file_ids from {args.exclude}")

    ar_labels, ar_text = load_task_labels(args.src_dir)
    langs = [("ar", args.out_dir, ar_text)]
    if args.en_src_dir.exists():
        en_labels, en_text = load_task_labels(args.en_src_dir)
        # The split is keyed on file_id, so the two languages must cover the
        # same calls with the same labels for the partition to apply to both.
        if set(en_labels) != set(ar_labels) or any(
            en_labels[f] != ar_labels[f] for f in ar_labels
        ):
            raise ValueError("Arabic and English label maps disagree; cannot share a split")
        langs.append(("en", args.en_out_dir, en_text))
    else:
        print(f"NOTE: {args.en_src_dir} not found, building Arabic only.\n")

    manifest: dict[str, dict] = {}
    for merged_key, meta in MERGED_TASKS.items():
        components = meta["components"]
        labelled: dict[str, int] = {}
        n_dropped = 0
        n_excluded = 0
        for fid, per_task in ar_labels.items():
            if fid in excluded:
                # Dropped before labelling, so the call cannot influence the split
                # or the class balance in either language.
                if any(t in per_task for t in components):
                    n_excluded += 1
                continue
            label = merge_label(per_task, components)
            if label is None:
                # Either no component annotated this call at all, or the OR is
                # undetermined; both are excluded.
                if any(t in per_task for t in components):
                    n_dropped += 1
                continue
            labelled[fid] = label

        assignment = stratified_split(labelled, args.test_size, args.seed)
        by_split = {
            split: sorted(f for f, s in assignment.items() if s == split)
            for split in SPLITS
        }
        instruction = build_instruction(meta["question"])

        print(f"{merged_key}  (OR of {len(components)} tasks)")
        print(f"  kept {len(labelled)} calls, dropped {n_dropped} undetermined"
              + (f", {n_excluded} excluded" if n_excluded else ""))
        for lang, out_dir, transcripts in langs:
            for split in SPLITS:
                out_path = out_dir / merged_key / f"{split}.jsonl"
                n, n_pos = write_split(
                    out_path, by_split[split], labelled, transcripts, instruction
                )
                print(
                    f"  [{lang}] {split:5} {n:4} rows  pos={n_pos:3} "
                    f"({n_pos / n:.1%})  -> {out_path}"
                )
        overlap = set(by_split["train"]) & set(by_split["test"])
        assert not overlap, f"{merged_key}: {len(overlap)} file_ids in both splits"
        manifest[merged_key] = {
            "components": components,
            "n_kept": len(labelled),
            "n_dropped_undetermined": n_dropped,
            "n_excluded": n_excluded,
            "excluded_file_ids": sorted(f for f in excluded if f in ar_labels),
            "exclusion_list": str(args.exclude) if args.exclude else None,
            "test_size": args.test_size,
            "seed": args.seed,
            "splits": by_split,
        }
        print()

    for _, out_dir, _ in langs:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
    print("Wrote split_manifest.json (file_id partition) alongside each dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
