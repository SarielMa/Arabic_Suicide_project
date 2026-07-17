#!/bin/bash
#SBATCH --job-name=arabic_suicide_balanced
#SBATCH --mail-type=ALL
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:2
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=%j_arabic_suicide_balanced_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu
#
# Class-imbalance experiment. Submit with no arguments:
#
#     sbatch apply_balanced.sh
#
# Everything is configured in the CONFIG block below -- edit it, then sbatch.
# This script NEVER writes into runs/: the baseline results behind the paper
# table are safe no matter what you set here.
set -euo pipefail

# ============================== CONFIG ======================================
# What to run:
#   rescore       Arm B only. Re-evaluates the EXISTING runs/ adapters with P(Yes)
#                 scoring + a corrected threshold. NO training: finishes in minutes,
#                 costs nothing, and tells you whether the collapses in the paper are
#                 dead models or just a mis-set decision threshold. -> runs_scored/
#   balanced      Arm A+B. Retrains every model with the class-weighted loss and
#                 evaluates with the corrected threshold. Full sweep. -> runs_balanced/
#   balance_only  Arm A only. Re-evaluates the EXISTING runs_balanced/ (class-
#                 weighted) adapters with plain greedy decoding at 0.5. NO training:
#                 isolates the class-weighting fix from the threshold fix, so the
#                 ablation can separate the two. Requires 'balanced' to have run
#                 first. -> runs_balanced_only/
EXPERIMENT="balanced"

# --- Arm A: training-side fix (used only when EXPERIMENT=balanced) ---
# Positive examples get loss weight w_pos = (N_neg / N_pos)^ALPHA, capped at CAP.
# With the real splits that gives: WD 1.18, NA 2.07, IM 3.20, SI 3.23, PI 4.0(cap).
#   CLASS_WEIGHT=none  -> disables Arm A (ablation: threshold fix alone)
#   ALPHA=0.5          -> softer, square-root weighting (if alpha=1 over-triggers)
#   CAP                -> stops the rarest task (PI, 35 positives) from getting a
#                         6x weight, which at that support flips collapse into
#                         indiscriminate "Yes"
CLASS_WEIGHT="balanced"
CLASS_WEIGHT_ALPHA="1.0"
CLASS_WEIGHT_CAP="4.0"

# --- Arm B: inference-side fix ---
# DECISION=prob thresholds P(Yes) from the first answer token's logits.
# THRESHOLD=prior uses each task's own train positive rate (WD .46 ... PI .14)
# instead of the 0.5 that greedy decoding hard-codes. Set a float to override.
#   DECISION=greedy -> disables Arm B (ablation: class weighting alone)
DECISION="prob"
THRESHOLD="prior"

# Which models to sweep (one HF repo per line; '#' comments allowed).
MODELS_FILE="models.txt"
# ===========================================================================

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/Arabic_data_match/llm_pipeline_0707"

case "${EXPERIMENT}" in
  rescore|balanced|balance_only) ;;
  *) echo "EXPERIMENT must be 'rescore', 'balanced', or 'balance_only', got '${EXPERIMENT}'" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------- environment
for var in CONDA_EXE CONDA_PREFIX CONDA_PREFIX_1 CONDA_PREFIX_2 CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE CONDA_PKGS_DIRS CONDA_ENVS_PATH _CE_CONDA _CE_M _CONDA_EXE _CONDA_ROOT; do
  unset "${var}" || true
done
unset -f conda 2>/dev/null || true
unset -f __conda_activate 2>/dev/null || true
unset -f __conda_reactivate 2>/dev/null || true
unset -f __conda_hashr 2>/dev/null || true

if ! command -v conda >/dev/null 2>&1; then
  conda() { return 0; }
  export -f conda
  _FAKE_CONDA_FOR_PURGE=1
fi

module --force purge || true
if [[ "${_FAKE_CONDA_FOR_PURGE:-0}" == "1" ]]; then
  unset -f conda || true
  unset _FAKE_CONDA_FOR_PURGE
fi

module load StdEnv || true
module load CUDA/12.8.0

export CUDA_HOME
CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export TRITON_CACHE_DIR="/tmp/${USER}/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

module load miniconda

if [[ -n "${EBROOTMINICONDA:-}" && -f "${EBROOTMINICONDA}/etc/profile.d/conda.sh" ]]; then
  source "${EBROOTMINICONDA}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
  CONDA_BASE="$(cd "$(dirname "${CONDA_BIN}")/.." && pwd)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "Failed to initialize conda after loading the miniconda module." >&2
  exit 1
fi

conda activate finben_b200

which python
python -c "import torch; print('torch cuda:', torch.version.cuda); print('gpus:', torch.cuda.device_count())"
nvidia-smi

cd "${REPO_ROOT}"

echo "=========================================================="
echo " EXPERIMENT   = ${EXPERIMENT}"
echo " CLASS_WEIGHT = ${CLASS_WEIGHT} (alpha=${CLASS_WEIGHT_ALPHA}, cap=${CLASS_WEIGHT_CAP})"
echo " DECISION     = ${DECISION} (threshold=${THRESHOLD})"
echo " MODELS_FILE  = ${MODELS_FILE}"
echo " CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "=========================================================="

# run_balanced.sh reads these from the environment.
export CLASS_WEIGHT CLASS_WEIGHT_ALPHA CLASS_WEIGHT_CAP DECISION THRESHOLD MODELS_FILE

case "${EXPERIMENT}" in
  rescore)      bash run_balanced.sh rescore ;;
  balanced)     bash run_balanced.sh train ;;
  balance_only) bash run_balanced.sh balance_only ;;
esac

echo "Done (EXPERIMENT=${EXPERIMENT})."
