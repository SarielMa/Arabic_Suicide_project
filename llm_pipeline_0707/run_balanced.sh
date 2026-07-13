#!/usr/bin/env bash
# Class-imbalance experiment: re-run SFT with a class-weighted loss and a
# prior-corrected decision threshold, for every model in models.txt.
#
# Two independent interventions, both aimed at the majority-class collapse that
# dominates the baseline table:
#
#   Arm A (training):  --class-weight balanced  weights positive examples by
#                      (N_neg/N_pos), so predicting "No" everywhere is no longer
#                      the cheapest way down. Requires retraining -> writes new
#                      adapters under runs_balanced/.
#   Arm B (inference): --decision prob --threshold prior  thresholds P(Yes) at the
#                      training prior instead of implicitly at 0.5 via argmax.
#                      Needs no training: it also runs over the BASELINE adapters
#                      already in runs/ (see rescore_baseline below).
#
# Usage:
#   bash run_balanced.sh              # Arm A + B: retrain all models, prob decoding
#   bash run_balanced.sh rescore      # Arm B only: re-evaluate the EXISTING runs/
#                                     # adapters with scoring; no training at all
#
# Outputs land in runs_balanced/ (Arm A) and runs_scored/ (Arm B on baselines), so
# the published runs/ tree is never touched.
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-train}"
MODELS_FILE="${MODELS_FILE:-models.txt}"

# Experiment knobs (apply_balanced.sh sets these; defaults are the paper config).
CLASS_WEIGHT="${CLASS_WEIGHT:-balanced}"       # balanced | none
CLASS_WEIGHT_ALPHA="${CLASS_WEIGHT_ALPHA:-1.0}"  # w_pos = (N_neg/N_pos)^alpha
CLASS_WEIGHT_CAP="${CLASS_WEIGHT_CAP:-4.0}"    # upper bound on w_pos
DECISION="${DECISION:-prob}"                   # prob | greedy
THRESHOLD="${THRESHOLD:-prior}"                # 'prior' or a float in (0,1)
DATA_DIR="${DATA_DIR:-processed_datasets}"
BALANCED_RUNS_DIR="${BALANCED_RUNS_DIR:-runs_balanced}"
SCORED_RUNS_DIR="${SCORED_RUNS_DIR:-runs_scored}"

echo "--- imbalance experiment config ---"
echo "  class weight : ${CLASS_WEIGHT} (alpha=${CLASS_WEIGHT_ALPHA}, cap=${CLASS_WEIGHT_CAP})"
echo "  decision     : ${DECISION} (threshold=${THRESHOLD})"
echo "  data dir     : ${DATA_DIR}"
echo "  models file  : ${MODELS_FILE}"
echo "-----------------------------------"

TASKS=(
  wish_to_be_dead
  non_specific_active_suicidal_thoughts
  active_suicidal_ideation_with_any_methods
  active_suicidal_with_some_intent_to_act
  active_suicidal_ideation_with_specific_plan_and_intent
)

mapfile -t MODELS < <(grep -vE '^[[:space:]]*(#|$)' "$MODELS_FILE")
[[ ${#MODELS[@]} -gt 0 ]] || { echo "No models in $MODELS_FILE" >&2; exit 1; }

python prepare_data.py

case "$MODE" in
  train)
    echo "Arm A+B: class-weighted SFT + thresholded decoding -> ${BALANCED_RUNS_DIR}/"
    for MODEL in "${MODELS[@]}"; do
      MODEL="$(echo "$MODEL" | xargs)"
      RUN_NAME="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
      RUNS_DIR="${BALANCED_RUNS_DIR}" DATA_DIR="${DATA_DIR}" \
      TRAIN_ARGS="--class-weight ${CLASS_WEIGHT} --class-weight-alpha ${CLASS_WEIGHT_ALPHA} --class-weight-cap ${CLASS_WEIGHT_CAP}" \
      EVAL_ARGS="--decision ${DECISION} --threshold ${THRESHOLD}" \
        bash run_all.sh "$MODEL" "$RUN_NAME"
    done
    echo "Done. Summaries: ${BALANCED_RUNS_DIR}/<model>/summary.csv"
    ;;

  rescore)
    # Arm B alone: the baseline adapters are already trained, so this is pure
    # inference. It answers whether the collapses in the paper are dead models or
    # merely mis-set thresholds -- without spending a single training step.
    echo "Arm B only: re-evaluating EXISTING runs/ adapters with P(Yes) scoring"
    for MODEL in "${MODELS[@]}"; do
      MODEL="$(echo "$MODEL" | xargs)"
      RUN_NAME="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
      for TASK in "${TASKS[@]}"; do
        ADAPTER="runs/${RUN_NAME}/${TASK}"
        [[ -d "$ADAPTER" ]] || { echo "[skip] no adapter: $ADAPTER"; continue; }
        OUT="${SCORED_RUNS_DIR}/${RUN_NAME}/${TASK}"
        echo "======== RESCORE: ${TASK} (${MODEL}) ========"
        python evaluate.py --task "$TASK" --model "$MODEL" \
            --adapter "$ADAPTER" --out "${OUT}/eval" --data-dir "${DATA_DIR}" \
            --decision "${DECISION}" --threshold "${THRESHOLD}" \
            --summary-csv "${SCORED_RUNS_DIR}/${RUN_NAME}/summary.csv"
      done
    done
    echo "Done. Summaries: ${SCORED_RUNS_DIR}/<model>/summary.csv"
    echo "Sweep thresholds offline (no GPU):"
    echo "  python rescore.py --pred '${SCORED_RUNS_DIR}/*/*/eval' --sweep"
    ;;

  *)
    echo "Unknown mode '$MODE' (use: train | rescore)" >&2
    exit 1
    ;;
esac
