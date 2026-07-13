#!/usr/bin/env bash
# End-to-end sweep: prepare data, then train + evaluate every task for a model.
#
# Usage:
#   bash run_all.sh Qwen/Qwen2.5-1.5B-Instruct qwen2.5-1.5b
#   bash run_all.sh Qwen/Qwen2.5-14B-Instruct  qwen2.5-14b
#
# Arg 1: HF model id (default Qwen/Qwen2.5-1.5B-Instruct)
# Arg 2: short run name used in the output path (default derived from model)
#
# Env knobs (used by the class-imbalance experiment; defaults reproduce the
# original baseline sweep exactly):
#   RUNS_DIR    output tree                        (default: runs)
#   TRAIN_ARGS  extra flags for train.py           (e.g. --class-weight balanced)
#   EVAL_ARGS   extra flags for evaluate.py        (e.g. --decision prob)
#
# Balanced arm (writes to runs_balanced/, leaving runs/ untouched):
#   RUNS_DIR=runs_balanced TRAIN_ARGS="--class-weight balanced" \
#     EVAL_ARGS="--decision prob --threshold prior" \
#     bash run_all.sh meta-llama/Llama-3.3-70B-Instruct llama-3.3-70b-instruct
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
RUN_NAME="${2:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}"
RUNS_DIR="${RUNS_DIR:-runs}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
EVAL_ARGS="${EVAL_ARGS:-}"

TASKS=(
  wish_to_be_dead
  non_specific_active_suicidal_thoughts
  active_suicidal_ideation_with_any_methods
  active_suicidal_with_some_intent_to_act
  active_suicidal_ideation_with_specific_plan_and_intent
)

# Step 1: build instruction-formatted data (idempotent; safe to re-run).
python prepare_data.py

for TASK in "${TASKS[@]}"; do
  OUT="${RUNS_DIR}/${RUN_NAME}/${TASK}"
  echo "================ TRAIN: ${TASK} (${MODEL}) ================"
  python train.py --task "$TASK" --model "$MODEL" --output-dir "$OUT" ${TRAIN_ARGS}

  echo "================ EVAL:  ${TASK} (${MODEL}) ================"
  python evaluate.py --task "$TASK" --model "$MODEL" \
      --adapter "$OUT" --out "${OUT}/eval" \
      --summary-csv "${RUNS_DIR}/${RUN_NAME}/summary.csv" ${EVAL_ARGS}
done

echo "All tasks done for ${MODEL}."
echo "Per-task metrics: ${RUNS_DIR}/${RUN_NAME}/<task>/eval/metrics.{json,csv}"
echo "Combined summary: ${RUNS_DIR}/${RUN_NAME}/summary.csv"
