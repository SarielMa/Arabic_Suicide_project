#!/usr/bin/env python3
"""Step 3: evaluate a (fine-tuned or base) model on a task's test set.

Generates a Yes/No answer per test example and reports metrics suited to
imbalanced binary classification: per-class precision/recall/F1, macro and
weighted averages, accuracy, and the confusion matrix. The positive class is
label 1 (TRUE / "Yes").

Works for:
  * a QLoRA-tuned adapter:  --adapter runs/.../<task>
  * the base model zero-shot: omit --adapter

Example::

    python evaluate.py \
        --task wish_to_be_dead \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter runs/qwen2.5-1.5b/wish_to_be_dead \
        --data-dir processed_datasets \
        --out runs/qwen2.5-1.5b/wish_to_be_dead/eval
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from tasks import (
    answer_first_token_ids,
    messages_from_instruction,
    set_chat_template_if_missing,
    text_to_label,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def positive_prior(data_dir: Path, task: str) -> float:
    """Positive rate of the task's own train split, used as the decision threshold.

    Greedy decoding implicitly thresholds P(Yes) at 0.5, which is only calibrated
    when the classes are balanced. Thresholding at the training prior instead is the
    balanced-class decision rule: it asks whether the transcript raises the odds of
    risk above the base rate the model was trained on, rather than above a coin flip.
    """
    records = read_jsonl(data_dir / task / "train.jsonl")
    return sum(int(r["label"]) for r in records) / len(records)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Metrics for imbalanced binary classification (positive class = 1)."""
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
    return {
        "n": len(y_true),
        "n_pred_pos": int(sum(y_pred)),
        "accuracy": accuracy_score(y_true, y_pred),
        "positive_class": 1,
        "per_class": {
            "negative(0)": {"precision": p[0], "recall": r[0], "f1": f1[0], "support": int(support[0])},
            "positive(1)": {"precision": p[1], "recall": r[1], "f1": f1[1], "support": int(support[1])},
        },
        # Headline numbers for the imbalanced positive class:
        "precision_pos": p[1],
        "recall_pos": r[1],
        "f1_pos": f1[1],
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "weighted": {"precision": w_p, "recall": w_r, "f1": w_f1},
        "confusion_matrix": {"labels": labels, "matrix": cm},  # rows=true, cols=pred
    }


# Column order for the flat CSV summary row.
CSV_FIELDS = [
    "base_model", "adapter", "task", "split", "n",
    "decision", "threshold", "class_weight", "pos_weight",
    "accuracy",
    "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
    "roc_auc", "pr_auc",
    "precision_pos", "recall_pos", "f1_pos",
    "precision_neg", "recall_neg", "f1_neg",
    "support_pos", "support_neg",
    "tn", "fp", "fn", "tp", "n_unparsed",
]


def metrics_to_row(metrics: dict, split: str) -> dict:
    """Flatten the nested metrics dict into a single CSV row."""
    cm = metrics["confusion_matrix"]["matrix"]  # [[tn, fp], [fn, tp]]
    neg = metrics["per_class"]["negative(0)"]
    pos = metrics["per_class"]["positive(1)"]
    return {
        "base_model": metrics["base_model"],
        "adapter": metrics["adapter"] or "",
        "task": metrics["task"],
        "split": split,
        "n": metrics["n"],
        "decision": metrics["decision"],
        "threshold": round(metrics["threshold"], 4),
        "class_weight": metrics["class_weight"],
        "pos_weight": metrics["pos_weight"],
        "roc_auc": round(metrics["roc_auc"], 4) if metrics["roc_auc"] is not None else "",
        "pr_auc": round(metrics["pr_auc"], 4) if metrics["pr_auc"] is not None else "",
        "accuracy": round(metrics["accuracy"], 4),
        "macro_precision": round(metrics["macro"]["precision"], 4),
        "macro_recall": round(metrics["macro"]["recall"], 4),
        "macro_f1": round(metrics["macro"]["f1"], 4),
        "weighted_f1": round(metrics["weighted"]["f1"], 4),
        "precision_pos": round(metrics["precision_pos"], 4),
        "recall_pos": round(metrics["recall_pos"], 4),
        "f1_pos": round(metrics["f1_pos"], 4),
        "precision_neg": round(neg["precision"], 4),
        "recall_neg": round(neg["recall"], 4),
        "f1_neg": round(neg["f1"], 4),
        "support_pos": pos["support"],
        "support_neg": neg["support"],
        "tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1],
        "n_unparsed": metrics["n_unparsed"],
    }


def append_summary_csv(path: Path, row: dict) -> None:
    """Append one row to a shared CSV, writing the header if the file is new.

    An existing file keeps its own header: the summaries written by earlier runs
    predate the scoring columns, and appending a wider row under a narrower header
    would silently misalign every column.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    fieldnames = CSV_FIELDS
    if not new_file:
        with path.open(encoding="utf-8") as handle:
            existing = next(csv.reader(handle), None)
        if existing:
            fieldnames = existing
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def build_prompts(tokenizer, batch, max_input_tokens) -> list[str]:
    prompts = []
    for rec in batch:
        ids = tokenizer(rec["input"], add_special_tokens=False)["input_ids"]
        transcript = (
            rec["input"] if len(ids) <= max_input_tokens
            else tokenizer.decode(ids[:max_input_tokens])
        )
        messages = messages_from_instruction(rec["instruction"], transcript)
        prompts.append(
            tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        )
    return prompts


@torch.no_grad()
def generate_predictions(model, tokenizer, records, max_input_tokens, max_new_tokens, batch_size):
    device = next(model.parameters()).device
    raw_outputs: list[str] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prompts = build_prompts(tokenizer, batch, max_input_tokens)
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False
        ).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1] :]
        raw_outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        print(f"  generated {min(start + batch_size, len(records))}/{len(records)}")
    return raw_outputs


@torch.no_grad()
def score_predictions(model, tokenizer, records, max_input_tokens, batch_size):
    """P(Yes) for each example, from the logits of the first answer token.

    A softmax restricted to the {Yes, No} token ids, at the last prompt position.
    This is the same quantity greedy decoding thresholds at 0.5, but kept as a
    continuous score, so the operating point becomes a choice rather than an
    artefact of argmax --- and a model that "collapsed" under greedy decoding can
    still be ranked by how much probability it puts on the positive class.
    """
    device = next(model.parameters()).device
    yes_id, no_id = answer_first_token_ids(tokenizer)
    print(f"Scoring first answer token: yes_id={yes_id} no_id={no_id} "
          f"({tokenizer.convert_ids_to_tokens([yes_id, no_id])})")

    scores: list[float] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prompts = build_prompts(tokenizer, batch, max_input_tokens)
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False
        ).to(device)
        logits = model(**enc).logits[:, -1, :].float()  # left padding => last is next
        pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=-1)
        p_yes = torch.softmax(pair, dim=-1)[:, 1]
        scores.extend(p_yes.tolist())
        print(f"  scored {min(start + batch_size, len(records))}/{len(records)}")
    return scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for base model.")
    p.add_argument("--data-dir", type=Path, default=Path("processed_datasets"))
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Append this run's metrics as one row to a shared CSV "
        "(header written once). Ideal for collecting a cluster sweep.",
    )
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument(
        "--decision",
        choices=["greedy", "prob"],
        default="greedy",
        help="'greedy' parses the generated token (the original behaviour); "
        "'prob' thresholds P(Yes) from the first-token logits.",
    )
    p.add_argument(
        "--threshold",
        default="prior",
        help="Decision threshold for --decision prob: a float, or 'prior' to use "
        "the task's train-split positive rate (the balanced-class rule).",
    )
    p.add_argument(
        "--no-score",
        action="store_true",
        help="Skip the P(Yes) forward pass (no scores, no AUC).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    use_4bit = not args.no_4bit and use_cuda

    # GPUs come from CUDA_VISIBLE_DEVICES (set by SLURM); device_map="auto" uses
    # all visible GPUs. Print for log-side confirmation.
    import os
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"torch sees {torch.cuda.device_count()} GPU(s); 4-bit={use_4bit}")

    # Prefer the tokenizer saved with the adapter (identical special tokens).
    tok_src = args.adapter if args.adapter else args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    set_chat_template_if_missing(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left padding for batched generation

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    records = read_jsonl(args.data_dir / args.task / f"{args.split}.jsonl")
    print(f"Evaluating {len(records)} examples "
          f"({'adapter: ' + args.adapter if args.adapter else 'base model'})")

    raw_outputs = generate_predictions(
        model, tokenizer, records,
        max_input_tokens=args.max_len - 128,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    scores = None
    if not args.no_score:
        scores = score_predictions(
            model, tokenizer, records,
            max_input_tokens=args.max_len - 128,
            batch_size=args.batch_size,
        )

    threshold = 0.5
    if args.decision == "prob":
        if args.threshold == "prior":
            threshold = positive_prior(args.data_dir, args.task)
        else:
            threshold = float(args.threshold)
        if scores is None:
            raise SystemExit("--decision prob requires scoring; drop --no-score.")
        print(f"Decision: P(Yes) >= {threshold:.4f} "
              f"({'train prior' if args.threshold == 'prior' else 'fixed'})")

    y_true, y_pred, n_unparsed = [], [], 0
    pred_rows = []
    for i, (rec, raw) in enumerate(zip(records, raw_outputs)):
        greedy = text_to_label(raw)
        if greedy is None:
            n_unparsed += 1
            greedy = 0  # count unparseable as negative
        p_yes = scores[i] if scores is not None else None
        pred = greedy if args.decision == "greedy" else int(p_yes >= threshold)
        y_true.append(int(rec["label"]))
        y_pred.append(pred)
        pred_rows.append({
            "file_id": rec["file_id"],
            "true": int(rec["label"]),
            "pred": pred,
            "pred_greedy": greedy,
            "p_yes": p_yes,
            "raw_output": raw.strip(),
        })

    metrics = compute_metrics(y_true, y_pred)
    metrics["n_unparsed"] = n_unparsed
    metrics["task"] = args.task
    metrics["base_model"] = args.model
    metrics["adapter"] = args.adapter
    metrics["decision"] = args.decision
    metrics["threshold"] = threshold
    # Threshold-free ranking quality: undefined if the split is single-class.
    both_classes = len(set(y_true)) == 2
    metrics["roc_auc"] = (
        roc_auc_score(y_true, scores) if scores is not None and both_classes else None
    )
    metrics["pr_auc"] = (
        average_precision_score(y_true, scores)
        if scores is not None and both_classes else None
    )
    # Provenance of the adapter being evaluated, so the balanced and baseline arms
    # are distinguishable in a shared summary CSV.
    metrics["class_weight"], metrics["pos_weight"] = "", ""
    if args.adapter:
        cfg_path = Path(args.adapter) / "run_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            metrics["class_weight"] = cfg.get("class_weight", "")
            metrics["pos_weight"] = cfg.get("pos_weight", "")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in pred_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Flat CSV outputs: a per-run one-liner, plus an optional shared summary.
    csv_row = metrics_to_row(metrics, args.split)
    with (args.out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(csv_row)
    if args.summary_csv:
        append_summary_csv(args.summary_csv, csv_row)
        print(f"Appended summary row to: {args.summary_csv}")

    print("\n=== Metrics ===")
    print(f"Decision rule:       {args.decision} (threshold {threshold:.4f})")
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"Positive-class (1):  P={metrics['precision_pos']:.4f} "
          f"R={metrics['recall_pos']:.4f} F1={metrics['f1_pos']:.4f}")
    print(f"Macro F1:            {metrics['macro']['f1']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC / PR-AUC:    {metrics['roc_auc']:.4f} / {metrics['pr_auc']:.4f}")
    print(f"Confusion [tn,fp / fn,tp]: {metrics['confusion_matrix']['matrix']}")
    print(f"Predicted positive:  {metrics['n_pred_pos']}/{metrics['n']}")
    print(f"Unparseable outputs: {n_unparsed}")
    print(f"Saved to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
