#!/usr/bin/env python3
"""Step 4: re-derive metrics at any decision threshold, offline, from saved scores.

evaluate.py stores P(Yes) per test example in ``predictions.jsonl``. Because the
threshold is applied *after* the model has spoken, every operating point can be
explored on CPU without touching a GPU or reloading a 70B model.

Two uses:

  * sweep a run and see where it would land at any threshold::

        python rescore.py --pred runs_balanced/acegpt-v2-70b-chat/*/eval

  * collect one row per (run, threshold) into a CSV for the paper::

        python rescore.py --pred 'runs*/**/eval' --thresholds 0.5,prior \\
            --out-csv runs_balanced/threshold_sweep.csv

``prior`` resolves per task to the train-split positive rate.
"""

from __future__ import annotations

import argparse
import csv
import json
from glob import glob
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

CSV_FIELDS = [
    "run", "base_model", "task", "threshold_name", "threshold",
    "accuracy", "macro_f1", "precision_pos", "recall_pos", "f1_pos",
    "tn", "fp", "fn", "tp", "n", "n_pred_pos",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prior_for(data_dir: Path, task: str) -> float:
    records = read_jsonl(data_dir / task / "train.jsonl")
    return sum(int(r["label"]) for r in records) / len(records)


def metrics_at(y_true: list[int], scores: list[float], threshold: float) -> dict:
    y_pred = [int(s >= threshold) for s in scores]
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
    )
    tn = sum(1 for t, q in zip(y_true, y_pred) if t == 0 and q == 0)
    fp = sum(1 for t, q in zip(y_true, y_pred) if t == 0 and q == 1)
    fn = sum(1 for t, q in zip(y_true, y_pred) if t == 1 and q == 0)
    tp = sum(1 for t, q in zip(y_true, y_pred) if t == 1 and q == 1)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(macro_f1, 4),
        "precision_pos": round(p[1], 4),
        "recall_pos": round(r[1], 4),
        "f1_pos": round(f1[1], 4),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "n": len(y_true), "n_pred_pos": tp + fp,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", nargs="+", required=True,
                   help="One or more eval dirs (or globs) holding predictions.jsonl.")
    p.add_argument("--data-dir", type=Path, default=Path("processed_datasets"))
    p.add_argument("--thresholds", default="0.5,prior",
                   help="Comma-separated floats and/or the keyword 'prior'.")
    p.add_argument("--sweep", action="store_true",
                   help="Also sweep 0.05..0.95 in steps of 0.05.")
    p.add_argument("--out-csv", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    eval_dirs: list[Path] = []
    for pattern in args.pred:
        hits = [Path(p) for p in glob(pattern, recursive=True)]
        eval_dirs.extend(hits if hits else [Path(pattern)])
    eval_dirs = [d for d in sorted(set(eval_dirs)) if (d / "predictions.jsonl").exists()]
    if not eval_dirs:
        raise SystemExit("No eval dirs with predictions.jsonl matched --pred.")

    names = [t.strip() for t in args.thresholds.split(",") if t.strip()]
    if args.sweep:
        names += [f"{t / 100:.2f}" for t in range(5, 100, 5)]

    rows = []
    for eval_dir in eval_dirs:
        preds = read_jsonl(eval_dir / "predictions.jsonl")
        if any(p.get("p_yes") is None for p in preds):
            print(f"[skip] {eval_dir}: no p_yes scores (produced before scoring "
                  f"existed; re-run evaluate.py to add them)")
            continue
        meta = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
        task = meta["task"]
        y_true = [int(p["true"]) for p in preds]
        scores = [float(p["p_yes"]) for p in preds]

        print(f"\n=== {eval_dir}  ({task})")
        print(f"{'threshold':>16}  {'macro-F1':>8} {'acc':>6} "
              f"{'P+':>6} {'R+':>6}  confusion")
        for name in names:
            threshold = prior_for(args.data_dir, task) if name == "prior" else float(name)
            m = metrics_at(y_true, scores, threshold)
            rows.append({
                "run": str(eval_dir), "base_model": meta["base_model"], "task": task,
                "threshold_name": name, "threshold": round(threshold, 4), **m,
            })
            print(f"{name + f' ({threshold:.3f})':>16}  "
                  f"{m['macro_f1'] * 100:8.2f} {m['accuracy'] * 100:6.2f} "
                  f"{m['precision_pos'] * 100:6.2f} {m['recall_pos'] * 100:6.2f}  "
                  f"[{m['tn']},{m['fp']} / {m['fn']},{m['tp']}]")

    if args.out_csv and rows:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
