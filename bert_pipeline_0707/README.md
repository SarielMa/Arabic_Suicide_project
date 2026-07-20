# Arabic BERT pipeline for the 5 binary suicide-risk tasks

Fine-tunes Arabic BERT encoders (sequence classification) on the transcripts.
Each of the 5 prediction labels is an independent binary task with its own
2-class head. Model-agnostic: any HF encoder works via `--model`.

Environment: `conda activate finben_b200`.

## Models (edit `models.txt`)
Defaults, chosen for **dialectal** Arabic transcripts:
- `CAMeL-Lab/bert-base-arabic-camelbert-da` — dialectal-Arabic BERT (recommended).
- `aubmindlab/bert-base-arabertv02` — most-used Arabic BERT (MSA + dialect).
- `Mo7amed3bdelghany/marbert-sakenni-sentiment` — MARBERT (sentiment-tuned; the
  head is discarded and a fresh 2-class head is trained).

## Files
- `tasks.py` — the 5 task keys. `data_utils.py` — loading + AraBERT preprocessing.
- `metrics.py` — imbalanced metrics + CSV helpers.
- `train.py` — fine-tune one task (class-weighted loss for imbalance).
- `evaluate.py` — predict + metrics; writes JSON + CSV.
- `models.txt`, `run_all.sh`, `run_pipeline.sh` — orchestration.
- `apply_server.sh` — SLURM (B200) batch script.

## Quick start
```bash
cd bert_pipeline_0707

# one task
python train.py --task wish_to_be_dead \
    --model CAMeL-Lab/bert-base-arabic-camelbert-da \
    --output-dir runs/camelbert-da/wish_to_be_dead
python evaluate.py --task wish_to_be_dead \
    --model runs/camelbert-da/wish_to_be_dead \
    --model-name CAMeL-Lab/bert-base-arabic-camelbert-da \
    --out runs/camelbert-da/wish_to_be_dead/eval

# all tasks for one model
bash run_all.sh CAMeL-Lab/bert-base-arabic-camelbert-da camelbert-da

# every model in models.txt
bash run_pipeline.sh
```

## Cluster (SLURM, B200)
Compute nodes may lack internet, so **prefetch models on the login node first**:
```bash
python prefetch_models.py   # downloads every model in models.txt into the HF cache
sbatch apply_server.sh      # uses the settings at the top of apply_server.sh
```
Long-transcript handling is set by two variables at the top of `apply_server.sh`
(edit them directly — no submit-time flags needed):
- `CHUNKING=1` (default) reads the full transcript via 512-token windows + pooling;
  set `0` for plain 512-token truncation.
- `TRUNCATION=head|tail` picks which end to keep.

They still accept an override if you prefer, e.g.
`sbatch --export=ALL,CHUNKING=0 apply_server.sh`.

## English Translated-Transcript Run
The translated English data from the LLM pipeline is supported directly:
```bash
MODELS_FILE=models_english.txt python prefetch_models.py
sbatch apply_english.sh
```
`apply_english.sh` reads `../llm_pipeline_0707/processed_datasets_en`, fine-tunes
`google-bert/bert-large-uncased` and `google-bert/bert-base-uncased`, and writes
outputs under `runs_en/<model>/`.

## Metrics
Same as the LLM pipeline: per-class precision/recall/F1, macro & weighted,
accuracy, confusion matrix. Positive class = label 1 (TRUE). Each run writes
`metrics.json`, `metrics.csv`, `predictions.jsonl`; `run_all.sh` collects a
`runs/<model>/summary.csv`.

## Important notes
- **512-token limit vs. long calls.** Transcripts are much longer than 512
  tokens (median ~1250), so BERT sees only part of each call. `--truncation head`
  keeps the start; `--truncation tail` keeps the end. Try both; a chunking /
  long-document approach would use the full transcript (not yet implemented).
- **Class imbalance.** `--class-weights balanced` (default) weights the loss by
  inverse class frequency; pass `--class-weights none` to disable.
- **AraBERT preprocessing** is auto-applied for `arabert` models *if* the
  `arabert` package is installed; otherwise it is skipped with a warning
  (`pip install arabert` to enable).
- **No zero-shot.** Unlike the LLM pipeline, BERT needs a trained head, so there
  is no zero-shot mode.
