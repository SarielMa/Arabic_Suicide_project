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
#   TASK_LIST   space-separated task dirs to sweep (default: the 5 C-SSRS tasks)
#   PREPARE_CMD data-build command run first       (default: python prepare_data.py)
#
# Balanced arm (writes to runs_balanced/, leaving runs/ untouched):
#   RUNS_DIR=runs_balanced TRAIN_ARGS="--class-weight balanced" \
#     EVAL_ARGS="--decision prob --threshold prior" \
#     bash run_all.sh meta-llama/Llama-3.3-70B-Instruct llama-3.3-70b-instruct
#
# Merged two-level arm (med_risk / high_risk; see build_merged_data.py):
#   RUNS_DIR=runs_merged DATA_DIR=processed_datasets_merged \
#     TASK_LIST="med_risk high_risk" PREPARE_CMD="python build_merged_data.py" \
#     bash run_all.sh Qwen/Qwen2.5-1.5B-Instruct qwen2.5-1.5b
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
RUN_NAME="${2:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}"
RUNS_DIR="${RUNS_DIR:-runs}"
DATA_DIR="${DATA_DIR:-processed_datasets}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
EVAL_ARGS="${EVAL_ARGS:-}"

# train.py / evaluate.py treat --task as a plain directory name under DATA_DIR and
# read the prompt from each record's `instruction` field, so any dataset laid out
# as <DATA_DIR>/<task>/{train,test}.jsonl can be swept by overriding TASK_LIST.
read -r -a TASKS <<< "${TASK_LIST:-wish_to_be_dead \
non_specific_active_suicidal_thoughts \
active_suicidal_ideation_with_any_methods \
active_suicidal_with_some_intent_to_act \
active_suicidal_ideation_with_specific_plan_and_intent}"

# Step 1: build instruction-formatted data (idempotent; safe to re-run). Override
# for datasets built by a different script; set to "true" to skip entirely.
${PREPARE_CMD:-python prepare_data.py}

for TASK in "${TASKS[@]}"; do
  OUT="${RUNS_DIR}/${RUN_NAME}/${TASK}"

  # Resume: a finished task leaves an adapter and an eval. Skip it, so a job that
  # died partway (node fault, wall clock) can simply be resubmitted instead of
  # retraining every model from scratch. Delete the task dir to force a redo.
  if [[ -f "${OUT}/adapter_model.safetensors" && -f "${OUT}/eval/metrics.json" ]]; then
    echo "======== SKIP (already done): ${TASK} (${MODEL}) ========"
    continue
  fi

  echo "================ TRAIN: ${TASK} (${MODEL}) ================"
  python train.py --task "$TASK" --model "$MODEL" --output-dir "$OUT" \
      --data-dir "$DATA_DIR" ${TRAIN_ARGS}

  echo "================ EVAL:  ${TASK} (${MODEL}) ================"
  python evaluate.py --task "$TASK" --model "$MODEL" \
      --adapter "$OUT" --out "${OUT}/eval" --data-dir "$DATA_DIR" \
      --summary-csv "${RUNS_DIR}/${RUN_NAME}/summary.csv" ${EVAL_ARGS}
done

echo "All tasks done for ${MODEL}."
echo "Per-task metrics: ${RUNS_DIR}/${RUN_NAME}/<task>/eval/metrics.{json,csv}"
echo "Combined summary: ${RUNS_DIR}/${RUN_NAME}/summary.csv"
