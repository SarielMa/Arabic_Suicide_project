#!/usr/bin/env python3
"""Quality report over a translations JSONL. CPU-only; run it on the login node.

The number to watch is not the mean length ratio, it is *which* transcripts got
flagged. If refusals and short outputs cluster on the calls that carry positive
labels, the translation has selectively deleted the risk content and the English
experiment is measuring translation damage rather than language.

Usage::

    python inspect_translations.py --pred translations/qwen2.5-72b-instruct.jsonl
    python inspect_translations.py --pred ... --show 3        # print 3 side-by-side
    python inspect_translations.py --pred ... --only-flagged  # just the problems
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

TASKS = [
    "wish_to_be_dead",
    "non_specific_active_suicidal_thoughts",
    "active_suicidal_ideation_with_any_methods",
    "active_suicidal_with_some_intent_to_act",
    "active_suicidal_ideation_with_specific_plan_and_intent",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def labels_by_file_id(data_dir: Path) -> dict[str, dict[str, int]]:
    """file_id -> {task: label}, so flags can be cross-tabulated against the labels."""
    out: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for task in TASKS:
        for split in ("train", "test"):
            path = data_dir / task / f"{split}.jsonl"
            if not path.exists():
                continue
            for rec in read_jsonl(path):
                out[rec["file_id"]][task] = int(rec["label"])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("processed_datasets"))
    p.add_argument("--show", type=int, default=0, help="Print N transcripts side by side.")
    p.add_argument("--only-flagged", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    recs = read_jsonl(args.pred)
    if not recs:
        raise SystemExit(f"No records in {args.pred}")
    labels = labels_by_file_id(args.data_dir)

    flagged = [r for r in recs if r["flags"]]
    print(f"=== {args.pred}  ({recs[0].get('model', '?')})")
    print(f"translated : {len(recs)}")
    print(f"clean      : {len(recs) - len(flagged)}")
    print(f"flagged    : {len(flagged)}")

    kinds = collections.Counter(
        f.split("(")[0] for r in flagged for f in r["flags"]
    )
    if kinds:
        print("\nflag types:")
        for kind, n in kinds.most_common():
            print(f"  {kind:20} {n}")

    ratios = [r["len_ratio"] for r in recs]
    print(f"\nlength ratio (en/ar chars): median={statistics.median(ratios):.2f} "
          f"min={min(ratios):.2f} max={max(ratios):.2f}")

    # The critical check: are the failures concentrated on positive-label calls?
    print("\n=== flag rate by label (is translation loss correlated with the label?) ===")
    print(f"{'task':<52} {'pos':>12} {'neg':>12}")
    for task in TASKS:
        pos = [r for r in recs if labels.get(r["file_id"], {}).get(task) == 1]
        neg = [r for r in recs if labels.get(r["file_id"], {}).get(task) == 0]
        if not pos and not neg:
            continue
        fp = sum(1 for r in pos if r["flags"])
        fn = sum(1 for r in neg if r["flags"])
        pos_s = f"{fp}/{len(pos)}" + (f" ({100 * fp / len(pos):.0f}%)" if pos else "")
        neg_s = f"{fn}/{len(neg)}" + (f" ({100 * fn / len(neg):.0f}%)" if neg else "")
        print(f"{task:<52} {pos_s:>12} {neg_s:>12}")
    print("\nA higher flag rate on positives than negatives means the translator is "
          "sanitizing\nthe risk content specifically -- do NOT build the English "
          "datasets until that is fixed.")

    to_show = (flagged if args.only_flagged else recs)[: args.show]
    for r in to_show:
        print("\n" + "=" * 78)
        print(f"file_id={r['file_id']}  flags={r['flags']}  len_ratio={r['len_ratio']}")
        print("-" * 78 + "\nARABIC:\n" + r["arabic"][:900])
        print("-" * 78 + "\nENGLISH:\n" + r["english"][:900])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
