#!/usr/bin/env python3
"""Step 2: QLoRA supervised fine-tuning for one binary task.

Fine-tunes a causal LM (e.g. Qwen/Qwen2.5-1.5B-Instruct) with 4-bit QLoRA on
the instruction-formatted dataset for a single task. Loss is computed only on
the assistant answer tokens (the prompt is masked with -100), so the model
learns to emit "Yes"/"No".

Uses the plain transformers Trainer (no trl dependency) so it works across
library versions and any HF causal LM.

Example::

    python train.py \
        --task wish_to_be_dead \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --data-dir processed_datasets \
        --output-dir runs/qwen2.5-1.5b/wish_to_be_dead

Run per task; loop over all 5 tasks in a shell script for a full sweep.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from tasks import messages_from_instruction

# LoRA target modules for Llama/Qwen-style attention+MLP blocks.
QWEN_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

IGNORE_INDEX = -100


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class SFTDataset(Dataset):
    """Tokenizes instruction examples with prompt tokens masked out."""

    def __init__(self, records, tokenizer, max_len: int, max_input_tokens: int):
        self.records = records
        self.tok = tokenizer
        self.max_len = max_len
        self.max_input_tokens = max_input_tokens

    def __len__(self) -> int:
        return len(self.records)

    def _truncate_input(self, text: str) -> str:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.max_input_tokens:
            return text
        return self.tok.decode(ids[: self.max_input_tokens])

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        transcript = self._truncate_input(rec["input"])
        messages = messages_from_instruction(rec["instruction"], transcript)

        prompt_ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        full_ids = self.tok.apply_chat_template(
            messages + [{"role": "assistant", "content": rec["output"]}],
            add_generation_prompt=False,
            tokenize=True,
        )
        full_ids = full_ids[: self.max_len]
        labels = list(full_ids)
        # Mask everything that belongs to the prompt.
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = IGNORE_INDEX
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }


@dataclass
class PadCollator:
    tokenizer: object
    pad_to_multiple_of: int = 8

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        pad_id = self.tokenizer.pad_token_id
        input_ids, attn, labels = [], [], []
        for x in batch:
            n = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * n)
            attn.append(x["attention_mask"] + [0] * n)
            labels.append(x["labels"] + [IGNORE_INDEX] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--data-dir", type=Path, default=Path("processed_datasets"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    use_4bit = not args.no_4bit and use_cuda

    # GPUs are inherited from CUDA_VISIBLE_DEVICES (set by SLURM --gpus). We do
    # not pass a count in; device_map="auto" uses all visible GPUs. Print it so
    # the log confirms Python sees the same GPUs SLURM allocated.
    import os
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"torch sees {torch.cuda.device_count()} GPU(s); 4-bit={use_4bit}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
    model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=QWEN_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_records = read_jsonl(args.data_dir / args.task / "train.jsonl")
    train_ds = SFTDataset(
        train_records,
        tokenizer,
        max_len=args.max_len,
        max_input_tokens=args.max_len - 128,
    )
    print(f"Training examples: {len(train_ds)}")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_cuda,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=PadCollator(tokenizer),
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))  # saves the LoRA adapter
    tokenizer.save_pretrained(str(args.output_dir))
    # Record what produced this run for reproducibility / evaluation.
    (args.output_dir / "run_config.json").write_text(
        json.dumps({"base_model": args.model, "task": args.task, **vars(args)},
                   default=str, indent=2),
        encoding="utf-8",
    )
    print(f"Saved adapter + tokenizer to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
