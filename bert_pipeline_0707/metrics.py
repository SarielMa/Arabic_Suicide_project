"""Imbalanced binary-classification metrics + CSV helpers (shared)."""

from __future__ import annotations

import csv
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Column order for the flat CSV summary row.
CSV_FIELDS = [
    "model", "task", "split", "n",
    "accuracy", "macro_f1", "weighted_f1",
    "roc_auc", "pr_auc",
    "precision_pos", "recall_pos", "f1_pos",
    "precision_neg", "recall_neg", "f1_neg",
    "support_pos", "support_neg",
    "tn", "fp", "fn", "tp",
]


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """Metrics for imbalanced binary classification (positive class = 1).

    ``y_score`` is the predicted probability of the positive class. When given,
    the threshold-free ranking metrics ROC-AUC and PR-AUC are added; both are
    ``None`` if it is omitted, or if ``y_true`` contains a single class (they
    are undefined there, and sklearn raises rather than returning NaN).
    """
    labels = [0, 1]
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    both_classes = len(set(y_true)) > 1
    roc_auc = pr_auc = None
    if y_score is not None and both_classes:
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "positive_class": 1,
        "per_class": {
            "negative(0)": {"precision": p[0], "recall": r[0], "f1": f1[0], "support": int(support[0])},
            "positive(1)": {"precision": p[1], "recall": r[1], "f1": f1[1], "support": int(support[1])},
        },
        "precision_pos": p[1],
        "recall_pos": r[1],
        "f1_pos": f1[1],
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "weighted": {"precision": w_p, "recall": w_r, "f1": w_f1},
        "confusion_matrix": {"labels": labels, "matrix": cm},  # rows=true, cols=pred
    }


def metrics_to_row(metrics: dict, model: str, task: str, split: str) -> dict:
    cm = metrics["confusion_matrix"]["matrix"]  # [[tn, fp], [fn, tp]]
    neg = metrics["per_class"]["negative(0)"]
    pos = metrics["per_class"]["positive(1)"]
    return {
        "model": model,
        "task": task,
        "split": split,
        "n": metrics["n"],
        "accuracy": round(metrics["accuracy"], 4),
        "macro_f1": round(metrics["macro"]["f1"], 4),
        "weighted_f1": round(metrics["weighted"]["f1"], 4),
        # Empty cell rather than a sentinel when scores were unavailable, so the
        # column reads as missing instead of as a real (and misleading) value.
        "roc_auc": round(metrics["roc_auc"], 4) if metrics.get("roc_auc") is not None else "",
        "pr_auc": round(metrics["pr_auc"], 4) if metrics.get("pr_auc") is not None else "",
        "precision_pos": round(metrics["precision_pos"], 4),
        "recall_pos": round(metrics["recall_pos"], 4),
        "f1_pos": round(metrics["f1_pos"], 4),
        "precision_neg": round(neg["precision"], 4),
        "recall_neg": round(neg["recall"], 4),
        "f1_neg": round(neg["f1"], 4),
        "support_pos": pos["support"],
        "support_neg": neg["support"],
        "tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1],
    }


def append_summary_csv(path: Path, row: dict) -> None:
    """Append one row to a shared CSV, writing the header if the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def write_run_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)
