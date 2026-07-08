#!/usr/bin/env python3
"""Fine-tune an Arabic BERT for one binary task (sequence classification).

Each task gets its own 2-class classification head fine-tuned on the transcript
text. Because the classes are imbalanced, class-weighted cross-entropy is used
by default.

IMPORTANT — length: BERT is capped at 512 tokens, but the transcripts are much
longer (median ~1250 tokens). Plain truncation therefore drops most of each
call. `--truncation head` keeps the start, `--truncation tail` keeps the end.
Consider this choice carefully (or a chunking approach) for best results.

Example::

    python train.py \
        --task wish_to_be_dead \
        --model CAMeL-Lab/bert-base-arabic-camelbert-da \
        --output-dir runs/camelbert-da/wish_to_be_dead

Run once per task; loop over all 5 in run_all.sh.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from tasks import TASK_KEYS
from data_utils import load_split, maybe_arabert_preprocessor, preprocess_texts
from metrics import compute_metrics
from chunk_model import (
    ChunkCollator,
    ChunkDataset,
    ChunkedModelForSequenceClassification,
    build_chunks,
)


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy for imbalanced labels."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = (
            self.class_weights.to(outputs.logits.device)
            if self.class_weights is not None
            else None
        )
        loss = nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def hf_compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    m = compute_metrics(list(labels), list(preds))
    return {
        "accuracy": m["accuracy"],
        "f1_pos": m["f1_pos"],
        "precision_pos": m["precision_pos"],
        "recall_pos": m["recall_pos"],
        "macro_f1": m["macro"]["f1"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=TASK_KEYS)
    p.add_argument("--model", default="CAMeL-Lab/bert-base-arabic-camelbert-da")
    p.add_argument("--data-dir", type=Path, default=Path("../training_datasets_0707"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--truncation", choices=["head", "tail"], default="head",
                   help="Keep the start (head) or end (tail) of long transcripts.")
    p.add_argument("--chunking", action="store_true",
                   help="Read the FULL transcript by splitting into 512-token "
                        "windows and mean-pooling their [CLS] vectors.")
    p.add_argument("--max-chunks", type=int, default=6,
                   help="Max 512-token windows per transcript when --chunking.")
    p.add_argument("--class-weights", choices=["balanced", "none"], default="balanced")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"torch sees {torch.cuda.device_count()} GPU(s)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # head -> truncate from the right (keep start); tail -> from the left.
    tokenizer.truncation_side = "right" if args.truncation == "head" else "left"

    preprocessor = maybe_arabert_preprocessor(args.model)
    print(f"Mode: {'chunking (max_chunks=%d)' % args.max_chunks if args.chunking else 'truncation (%s)' % args.truncation}")

    def load(split):
        texts, labels, _ = load_split(args.data_dir, args.task, split)
        texts = preprocess_texts(texts, preprocessor)
        if args.chunking:
            chunked = [
                build_chunks(t, tokenizer, args.max_length, args.max_chunks, args.truncation)
                for t in texts
            ]
            return ChunkDataset(chunked, labels), labels
        enc = tokenizer(texts, truncation=True, max_length=args.max_length)
        return TokenizedDataset(enc, labels), labels

    train_ds, train_labels = load("train")
    eval_ds, _ = load("test")
    print(f"Train: {len(train_labels)}  (pos={sum(train_labels)})  Test: {len(eval_ds)}")

    # Class weights inversely proportional to frequency.
    class_weights = None
    if args.class_weights == "balanced":
        counts = np.bincount(train_labels, minlength=2).astype(float)
        counts[counts == 0] = 1.0
        w = counts.sum() / (2.0 * counts)
        class_weights = torch.tensor(w, dtype=torch.float)
        print(f"Class weights (0,1): {class_weights.tolist()}")

    if args.chunking:
        model = ChunkedModelForSequenceClassification.from_encoder(args.model, num_labels=2)
        model.class_weights = class_weights  # loss handled inside the model
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_pos",
        greater_is_better=True,
        save_total_limit=1,
        bf16=use_cuda,
        report_to="none",
        seed=args.seed,
    )

    if args.chunking:
        # The chunked model computes the (weighted) loss internally, so a plain
        # Trainer is used with the chunk collator.
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            data_collator=ChunkCollator(tokenizer),
            compute_metrics=hf_compute_metrics,
        )
    else:
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=hf_compute_metrics,
        )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    (args.output_dir / "run_config.json").write_text(
        json.dumps({"model": args.model, "task": args.task, **vars(args)},
                   default=str, indent=2),
        encoding="utf-8",
    )
    print(f"Saved fine-tuned model to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
