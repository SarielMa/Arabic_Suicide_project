#!/usr/bin/env bash
# Fine-tune + evaluate one BERT model on all 5 tasks.
#
# Usage:
#   bash run_all.sh CAMeL-Lab/bert-base-arabic-camelbert-da camelbert-da
#
# Arg 1: HF model id (default CAMeL-Lab/bert-base-arabic-camelbert-da)
# Arg 2: short run name for the output path (default derived from model)
# Env:   TRUNCATION=head|tail (default head) controls which end of long
#        transcripts is kept.
set -euo pipefail

MODEL="${1:-CAMeL-Lab/bert-base-arabic-camelbert-da}"
RUN_NAME="${2:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}"
TRUNCATION="${TRUNCATION:-head}"
# CHUNKING=1 reads the FULL transcript (512-token windows + [CLS] mean-pool).
CHUNKING="${CHUNKING:-0}"
CHUNK_FLAG=""; [[ "$CHUNKING" == "1" ]] && CHUNK_FLAG="--chunking"

TASKS=(
  wish_to_be_dead
  non_specific_active_suicidal_thoughts
  active_suicidal_ideation_with_any_methods
  active_suicidal_with_some_intent_to_act
  active_suicidal_ideation_with_specific_plan_and_intent
)

for TASK in "${TASKS[@]}"; do
  OUT="runs/${RUN_NAME}/${TASK}"
  echo "================ TRAIN: ${TASK} (${MODEL}) ================"
  python train.py --task "$TASK" --model "$MODEL" \
      --truncation "$TRUNCATION" $CHUNK_FLAG --output-dir "$OUT"

  echo "================ EVAL:  ${TASK} (${MODEL}) ================"
  # Chunking is auto-detected from the model's run_config.json.
  python evaluate.py --task "$TASK" --model "$OUT" --model-name "$MODEL" \
      --truncation "$TRUNCATION" --out "${OUT}/eval" \
      --summary-csv "runs/${RUN_NAME}/summary.csv"
done

echo "All tasks done for ${MODEL}."
echo "Per-task metrics: runs/${RUN_NAME}/<task>/eval/metrics.{json,csv}"
echo "Combined summary: runs/${RUN_NAME}/summary.csv"
