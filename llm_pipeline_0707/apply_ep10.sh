#!/bin/bash
#SBATCH --job-name=ep10
#SBATCH --mail-type=ALL
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:2
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=%j_arabic_ep10_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu
#
# Longer-training experiment: plain SFT for 10 epochs, on BOTH Arabic and English,
# for every model in models.txt. Submit with no arguments:
#
#     sbatch apply_ep10.sh
#
# This is the *plain* pipeline -- standard cross-entropy, greedy decoding. No class
# weighting (Arm A), no thresholded scoring (Arm B). The ONLY thing that changes
# vs. the baseline is the epoch count (3 -> 10), so any difference is attributable
# to training length alone.
#
# Reads:  processed_datasets/      (Arabic; built by prepare_data.py)
#         processed_datasets_en/    (English; built by build_english_data.py)
# Writes: runs_ep10/<model>/<task>/      Arabic  SFT, 10 epochs, greedy decoding
#         runs_en_ep10/<model>/<task>/   English SFT, 10 epochs, greedy decoding
#
# Nothing under runs/, runs_en/, runs_scored/, runs_balanced/ or runs_balanced_only/
# is touched: the 3-epoch baselines stay exactly as they are.
#
# Zero-shot is NOT re-run here: it does not depend on the epoch count, so the
# existing runs/zeroshot/ (Arabic) and runs_en/zeroshot/ (English) are the correct
# zero-shot columns for the 10-epoch table.
#
# Resume: run_all.sh skips any task whose adapter + eval already exist, so if this
# job hits the wall clock just resubmit it -- finished tasks are not retrained.
set -euo pipefail

# ============================== CONFIG ======================================
# Number of SFT epochs (this is the whole point of the experiment).
EPOCHS="10"

# Which language(s) to sweep:
#   arabic   processed_datasets/     -> runs_ep10/
#   english  processed_datasets_en/  -> runs_en_ep10/
#   both     Arabic first, then English
LANG="both"

# Which models to sweep (one HF repo per line; '#' comments allowed).
MODELS_FILE="models.txt"
# ===========================================================================

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/Arabic_data_match/llm_pipeline_0707"

case "${LANG}" in
  arabic|english|both) ;;
  *) echo "LANG must be arabic|english|both, got '${LANG}'" >&2; exit 1 ;;
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

# Build the (data dir, runs dir) pairs to sweep, in order.
DATA_DIRS=()
RUNS_DIRS=()
if [[ "${LANG}" == "arabic" || "${LANG}" == "both" ]]; then
  DATA_DIRS+=("processed_datasets");    RUNS_DIRS+=("runs_ep10")
fi
if [[ "${LANG}" == "english" || "${LANG}" == "both" ]]; then
  DATA_DIRS+=("processed_datasets_en"); RUNS_DIRS+=("runs_en_ep10")
fi

# Fail fast if a requested language has no data dir.
for i in "${!DATA_DIRS[@]}"; do
  if [[ ! -d "${DATA_DIRS[$i]}" ]]; then
    echo "No ${DATA_DIRS[$i]}/. Build it first (prepare_data.py / build_english_data.py)." >&2
    exit 1
  fi
done

echo "=========================================================="
echo " EXPERIMENT  = 10-epoch plain SFT (no class weight, greedy decoding)"
echo " EPOCHS      = ${EPOCHS}"
echo " LANG        = ${LANG}"
echo " MODELS_FILE = ${MODELS_FILE}"
echo "=========================================================="

mapfile -t MODELS < <(grep -vE '^[[:space:]]*(#|$)' "${MODELS_FILE}")
[[ ${#MODELS[@]} -gt 0 ]] || { echo "No models in ${MODELS_FILE}" >&2; exit 1; }

# Plain SFT at EPOCHS epochs. EVAL_ARGS is left empty on purpose so evaluate.py
# defaults to --decision greedy: exactly the baseline condition, only longer.
for i in "${!DATA_DIRS[@]}"; do
  DATA_DIR="${DATA_DIRS[$i]}"
  RUNS_DIR="${RUNS_DIRS[$i]}"
  echo "########## SWEEP: ${DATA_DIR} -> ${RUNS_DIR} (${EPOCHS} epochs) ##########"
  for MODEL in "${MODELS[@]}"; do
    MODEL="$(echo "${MODEL}" | xargs)"
    RUN_NAME="$(basename "${MODEL}" | tr '[:upper:]' '[:lower:]')"
    RUNS_DIR="${RUNS_DIR}" DATA_DIR="${DATA_DIR}" \
    TRAIN_ARGS="--epochs ${EPOCHS}" \
      bash run_all.sh "${MODEL}" "${RUN_NAME}"
  done
done

echo "Done (10-epoch SFT, LANG=${LANG})."
echo "Arabic SFT summaries:  runs_ep10/<model>/summary.csv"
echo "English SFT summaries: runs_en_ep10/<model>/summary.csv"
echo "Zero-shot columns (unchanged): runs/zeroshot/ and runs_en/zeroshot/"
