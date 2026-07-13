#!/usr/bin/env bash
# Drive the whole benchmark: for every model in models.txt, run zero-shot and/or
# SFT (QLoRA train + eval) across all 5 tasks.
#
# Usage:
#   bash run_pipeline.sh            # MODE=both  (default): zero-shot + SFT
#   bash run_pipeline.sh zeroshot   # zero-shot only
#   bash run_pipeline.sh sft        # SFT only
#
# Model list: edit models.txt (one HF repo per line). Override the file with:
#   MODELS_FILE=my_models.txt bash run_pipeline.sh
set -euo pipefail

# Always run from the pipeline directory (where the python scripts live).
cd "$(dirname "$0")"

MODE="${1:-both}"
MODELS_FILE="${MODELS_FILE:-models.txt}"

case "$MODE" in
  zeroshot|sft|both) ;;
  *) echo "Unknown MODE '$MODE' (use: zeroshot | sft | both)" >&2; exit 1 ;;
esac

[[ -f "$MODELS_FILE" ]] || { echo "Missing model list: $MODELS_FILE" >&2; exit 1; }

# Read models, skipping comments (#...) and blank lines.
mapfile -t MODELS < <(grep -vE '^[[:space:]]*(#|$)' "$MODELS_FILE")
[[ ${#MODELS[@]} -gt 0 ]] || { echo "No models found in $MODELS_FILE" >&2; exit 1; }

echo "MODE=${MODE}"
echo "Models (${#MODELS[@]}):"
printf '  %s\n' "${MODELS[@]}"

# Build instruction-formatted data once (idempotent).
python prepare_data.py

for MODEL in "${MODELS[@]}"; do
  # Strip leading/trailing whitespace (a stray space makes an invalid HF repo id).
  MODEL="${MODEL#"${MODEL%%[![:space:]]*}"}"
  MODEL="${MODEL%"${MODEL##*[![:space:]]}"}"
  [[ -n "$MODEL" ]] || continue
  RUN_NAME="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
  echo "############################################################"
  echo "# MODEL: ${MODEL}  (run name: ${RUN_NAME})"
  echo "############################################################"

  if [[ "$MODE" == "zeroshot" || "$MODE" == "both" ]]; then
    bash run_zeroshot.sh "$MODEL" "$RUN_NAME"
  fi
  if [[ "$MODE" == "sft" || "$MODE" == "both" ]]; then
    bash run_all.sh "$MODEL" "$RUN_NAME"
  fi
done

echo "All models done (MODE=${MODE})."
echo "Zero-shot summaries: runs/zeroshot/<model>/summary.csv"
echo "SFT summaries:       runs/<model>/summary.csv"
