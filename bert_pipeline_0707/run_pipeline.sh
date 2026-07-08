#!/usr/bin/env bash
# Fine-tune + evaluate every BERT model in models.txt across all 5 tasks.
#
# Usage:
#   bash run_pipeline.sh
# Env:
#   MODELS_FILE=my_models.txt   # override model list
#   TRUNCATION=head|tail        # which end of long transcripts to keep (default head)
set -euo pipefail

cd "$(dirname "$0")"

MODELS_FILE="${MODELS_FILE:-models.txt}"
export TRUNCATION="${TRUNCATION:-head}"

[[ -f "$MODELS_FILE" ]] || { echo "Missing model list: $MODELS_FILE" >&2; exit 1; }

mapfile -t MODELS < <(grep -vE '^[[:space:]]*(#|$)' "$MODELS_FILE")
[[ ${#MODELS[@]} -gt 0 ]] || { echo "No models found in $MODELS_FILE" >&2; exit 1; }

echo "TRUNCATION=${TRUNCATION}"
echo "Models (${#MODELS[@]}):"
printf '  %s\n' "${MODELS[@]}"

for MODEL in "${MODELS[@]}"; do
  RUN_NAME="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
  echo "############################################################"
  echo "# MODEL: ${MODEL}  (run name: ${RUN_NAME})"
  echo "############################################################"
  bash run_all.sh "$MODEL" "$RUN_NAME"
done

echo "All BERT models done. Summaries: runs/<model>/summary.csv"
