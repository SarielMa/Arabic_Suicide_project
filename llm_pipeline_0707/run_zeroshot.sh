#!/usr/bin/env bash
# Zero-shot evaluation (no training) of the base model on all 5 tasks.
#
# Usage:
#   bash run_zeroshot.sh Qwen/Qwen2.5-1.5B-Instruct qwen2.5-1.5b
#   bash run_zeroshot.sh Qwen/Qwen2.5-14B-Instruct  qwen2.5-14b
#
# Arg 1: HF model id (default Qwen/Qwen2.5-1.5B-Instruct)
# Arg 2: short run name used in the output path (default derived from model)
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
RUN_NAME="${2:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}"
# Same knobs as run_all.sh; defaults reproduce the original zero-shot sweep.
RUNS_DIR="${RUNS_DIR:-runs}"
DATA_DIR="${DATA_DIR:-processed_datasets}"
EVAL_ARGS="${EVAL_ARGS:-}"

TASKS=(
  wish_to_be_dead
  non_specific_active_suicidal_thoughts
  active_suicidal_ideation_with_any_methods
  active_suicidal_with_some_intent_to_act
  active_suicidal_ideation_with_specific_plan_and_intent
)

# Ensure instruction-formatted data exists (idempotent).
python prepare_data.py

for TASK in "${TASKS[@]}"; do
  OUT="${RUNS_DIR}/zeroshot/${RUN_NAME}/${TASK}"

  # Resume, as in run_all.sh: skip a task that already has metrics, so a job killed
  # by a node fault or the wall clock can be resubmitted without redoing the sweep.
  if [[ -f "${OUT}/eval/metrics.json" ]]; then
    echo "======== SKIP (already done): ${TASK} (${MODEL}) ========"
    continue
  fi

  echo "================ ZERO-SHOT EVAL: ${TASK} (${MODEL}) ================"
  # No --adapter => evaluate the base model directly (no fine-tuning).
  python evaluate.py --task "$TASK" --model "$MODEL" \
      --out "${OUT}/eval" --data-dir "$DATA_DIR" \
      --summary-csv "${RUNS_DIR}/zeroshot/${RUN_NAME}/summary.csv" ${EVAL_ARGS}
done

echo "Zero-shot done for ${MODEL}."
echo "Per-task metrics: ${RUNS_DIR}/zeroshot/${RUN_NAME}/<task>/eval/metrics.{json,csv}"
echo "Combined summary: ${RUNS_DIR}/zeroshot/${RUN_NAME}/summary.csv"
