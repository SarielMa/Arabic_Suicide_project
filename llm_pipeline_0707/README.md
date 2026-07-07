# LLM pipeline for the 5 binary suicide-risk tasks

QLoRA SFT + evaluation for classifying Arabic crisis-helpline transcripts.
Each of the 5 prediction labels is an independent binary task. The pipeline is
model-agnostic: pass any HF causal LM via `--model`.

Environment: `conda activate finben_b200`.

## Files
- `tasks.py` — task definitions, prompt/chat formatting, label parsing (shared).
- `prepare_data.py` — **Step 1**: raw `train.json`/`test.json` → instruction JSONL.
- `train.py` — **Step 2**: 4-bit QLoRA SFT for one task (transformers Trainer).
- `evaluate.py` — **Step 3**: generate predictions + precision/recall/F1 metrics.
- `run_all.sh` — prepare + train + eval all 5 tasks for one model.
- `run_zeroshot.sh` — zero-shot eval of all 5 tasks for one model.
- `models.txt` — the list of HF model repos to run (one per line; edit freely).
- `run_pipeline.sh` — loop over `models.txt`, running zero-shot and/or SFT.
- `apply_server.sh` — SLURM batch script (B200) that sets up the env and calls
  `run_pipeline.sh`.

## Instruction format (Step 1)
Each example becomes:
```json
{"file_id": "...", "instruction": "<question + 'answer Yes/No'>",
 "input": "<Arabic transcript>", "output": "Yes"|"No", "label": 0|1}
```
The chat template is applied at train/eval time; loss is on the answer only.

## Quick start
```bash
cd llm_pipeline_0707
python prepare_data.py                          # writes processed_datasets/

# one task
python train.py --task wish_to_be_dead \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir runs/qwen2.5-1.5b/wish_to_be_dead
python evaluate.py --task wish_to_be_dead \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter runs/qwen2.5-1.5b/wish_to_be_dead \
    --out runs/qwen2.5-1.5b/wish_to_be_dead/eval

# all tasks for a model — TRAIN + EVAL
bash run_all.sh Qwen/Qwen2.5-1.5B-Instruct qwen2.5-1.5b
bash run_all.sh Qwen/Qwen2.5-14B-Instruct  qwen2.5-14b

# all tasks for a model — ZERO-SHOT ONLY (no training)
bash run_zeroshot.sh Qwen/Qwen2.5-1.5B-Instruct qwen2.5-1.5b
bash run_zeroshot.sh Qwen/Qwen2.5-14B-Instruct  qwen2.5-14b
```

- `run_all.sh` → trains a QLoRA adapter per task, then evaluates it. Outputs
  under `runs/<run_name>/`.
- `run_zeroshot.sh` → evaluates the base model with no fine-tuning. Outputs
  under `runs/zeroshot/<run_name>/`.

Single-task zero-shot: run `evaluate.py` without `--adapter`.

## Cluster (SLURM, B200)
Edit `models.txt` to choose which models to run, then submit:
```bash
sbatch apply_server.sh                              # MODE=both (zero-shot + SFT)
sbatch --export=ALL,MODE=zeroshot apply_server.sh   # zero-shot only
sbatch --export=ALL,MODE=sft      apply_server.sh   # SFT only
```
`apply_server.sh` sets up CUDA 12.8 + the `finben_b200` conda env and runs
`run_pipeline.sh`, which loops over every model in `models.txt`. To run the
pipeline on an already-allocated node: `bash run_pipeline.sh [both|zeroshot|sft]`.

## Metrics (Step 3)
Because the classes are imbalanced, metrics include per-class
precision/recall/F1, macro & weighted averages, accuracy, and the confusion
matrix. Headline fields `precision_pos`/`recall_pos`/`f1_pos` are for the
positive class (label 1 = TRUE).

Each eval run writes into `--out`:
- `metrics.json` — full nested metrics.
- `metrics.csv` — the same run flattened to a single CSV row.
- `predictions.jsonl` — per-example true/pred/raw output.

Add `--summary-csv path.csv` to **append** the run's row to a shared CSV (header
written once). `run_all.sh` uses this to collect all 5 tasks into
`runs/<run_name>/summary.csv` — one table per model, ready to submit.

## Notes
- 4-bit QLoRA is on by default and requires a GPU; pass `--no-4bit` on CPU/debug.
- Transcripts are truncated to `--max-len` (default 2048) tokens, keeping the
  answer; raise it for the 14B model if you have memory.
- Task hyperparameters (LoRA r/alpha, lr, epochs, batch size) are `train.py` CLI
  flags — tune per model.
