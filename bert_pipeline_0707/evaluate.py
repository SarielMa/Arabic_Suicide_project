#!/usr/bin/env python3
"""Evaluate a fine-tuned Arabic BERT on a task's test set.

Reports imbalanced-classification metrics (per-class precision/recall/F1, macro
& weighted, accuracy, confusion matrix) and writes JSON + CSV outputs, mirroring
the LLM pipeline. The positive class is label 1 (TRUE).

Example::

    python evaluate.py \
        --task wish_to_be_dead \
        --model runs/camelbert-da/wish_to_be_dead \
        --out runs/camelbert-da/wish_to_be_dead/eval \
        --summary-csv runs/camelbert-da/summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tasks import TASK_KEYS
from data_utils import load_split, maybe_arabert_preprocessor, preprocess_texts
from chunk_model import (
    ChunkCollator,
    ChunkedModelForSequenceClassification,
    build_chunks,
)
from metrics import (
    append_summary_csv,
    compute_metrics,
    metrics_to_row,
    write_run_csv,
)


@torch.no_grad()
def predict(model, tokenizer, texts, max_length, batch_size):
    device = next(model.parameters()).device
    preds = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch, truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        print(f"  predicted {min(start + batch_size, len(texts))}/{len(texts)}")
    return preds


@torch.no_grad()
def predict_chunked(model, tokenizer, texts, max_length, max_chunks, truncation, batch_size):
    device = next(model.parameters()).device
    collator = ChunkCollator(tokenizer)
    preds = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch = [
            {"chunks": build_chunks(t, tokenizer, max_length, max_chunks, truncation),
             "label": 0}
            for t in batch_texts
        ]
        enc = collator(batch)
        enc = {k: v.to(device) for k, v in enc.items() if k != "labels"}
        logits = model(**enc).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        print(f"  predicted {min(start + batch_size, len(texts))}/{len(texts)}")
    return preds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=TASK_KEYS)
    p.add_argument("--model", required=True, help="Path to the fine-tuned model dir.")
    p.add_argument("--data-dir", type=Path, default=Path("../training_datasets_0707"))
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--truncation", choices=["head", "tail"], default="head")
    p.add_argument("--chunking", action="store_true",
                   help="Force chunked inference. Auto-detected from the model's "
                        "run_config.json if omitted.")
    p.add_argument("--max-chunks", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--model-name", default=None,
                   help="Name to record in the CSV (defaults to --model).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"torch sees {torch.cuda.device_count()} GPU(s)")
    use_cuda = torch.cuda.is_available()

    # Auto-detect chunking / truncation from the model's saved run_config.json.
    chunking, max_chunks, truncation = args.chunking, args.max_chunks, args.truncation
    cfg_path = Path(args.model) / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        chunking = args.chunking or bool(cfg.get("chunking"))
        max_chunks = cfg.get("max_chunks", max_chunks)
        truncation = cfg.get("truncation", truncation)
    print(f"Mode: {'chunking (max_chunks=%d)' % max_chunks if chunking else 'truncation (%s)' % truncation}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.truncation_side = "right" if truncation == "head" else "left"
    if chunking:
        model = ChunkedModelForSequenceClassification.from_pretrained(args.model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model)
    if use_cuda:
        model = model.cuda()
    model.eval()

    # AraBERT preprocessing is keyed off the original base model name if recorded.
    base_name = args.model_name or args.model
    preprocessor = maybe_arabert_preprocessor(base_name)

    texts, labels, file_ids = load_split(args.data_dir, args.task, args.split)
    texts = preprocess_texts(texts, preprocessor)
    print(f"Evaluating {len(texts)} examples with {args.model}")

    if chunking:
        preds = predict_chunked(
            model, tokenizer, texts, args.max_length, max_chunks,
            truncation, args.batch_size,
        )
    else:
        preds = predict(model, tokenizer, texts, args.max_length, args.batch_size)

    metrics = compute_metrics(labels, preds)
    metrics["task"] = args.task
    metrics["model"] = base_name

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for fid, t, pr in zip(file_ids, labels, preds):
            handle.write(json.dumps({"file_id": fid, "true": t, "pred": pr}) + "\n")

    row = metrics_to_row(metrics, base_name, args.task, args.split)
    write_run_csv(args.out / "metrics.csv", row)
    if args.summary_csv:
        append_summary_csv(args.summary_csv, row)
        print(f"Appended summary row to: {args.summary_csv}")

    print("\n=== Metrics ===")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Positive-class (1): P={metrics['precision_pos']:.4f} "
          f"R={metrics['recall_pos']:.4f} F1={metrics['f1_pos']:.4f}")
    print(f"Macro F1:           {metrics['macro']['f1']:.4f}")
    print(f"Confusion [tn,fp / fn,tp]: {metrics['confusion_matrix']['matrix']}")
    print(f"Saved to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
