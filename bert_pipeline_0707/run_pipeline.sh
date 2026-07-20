#!/usr/bin/env bash
# Fine-tune + evaluate every BERT model in models.txt across all 5 tasks.
#
# Usage:
#   bash run_pipeline.sh
# Env:
#   MODELS_FILE=my_models.txt
#   DATA_DIR=../training_datasets_0707
#   RUNS_DIR=runs
#   TRUNCATION=head|tail
set -euo pipefail

cd "$(dirname "$0")"

MODELS_FILE="${MODELS_FILE:-models.txt}"
export DATA_DIR="${DATA_DIR:-../training_datasets_0707}"
export RUNS_DIR="${RUNS_DIR:-runs}"
export TRUNCATION="${TRUNCATION:-head}"
export CHUNKING="${CHUNKING:-0}"  # 1 = full transcript via chunk+pool

[[ -f "$MODELS_FILE" ]] || { echo "Missing model list: $MODELS_FILE" >&2; exit 1; }
[[ -d "$DATA_DIR" ]] || { echo "Missing data directory: $DATA_DIR" >&2; exit 1; }

mapfile -t MODELS < <(grep -vE '^[[:space:]]*(#|$)' "$MODELS_FILE")
[[ ${#MODELS[@]} -gt 0 ]] || { echo "No models found in $MODELS_FILE" >&2; exit 1; }

echo "DATA_DIR=${DATA_DIR}"
echo "RUNS_DIR=${RUNS_DIR}"
echo "TRUNCATION=${TRUNCATION}  CHUNKING=${CHUNKING}"
echo "Models (${#MODELS[@]}):"
printf '  %s\n' "${MODELS[@]}"

for MODEL in "${MODELS[@]}"; do
  RUN_NAME="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
  echo "############################################################"
  echo "# MODEL: ${MODEL}  (run name: ${RUN_NAME})"
  echo "############################################################"
  bash run_all.sh "$MODEL" "$RUN_NAME"
done

echo "All BERT models done. Summaries: ${RUNS_DIR}/<model>/summary.csv"
