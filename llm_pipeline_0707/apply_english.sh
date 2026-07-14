#!/bin/bash
#SBATCH --job-name=arabic_english
#SBATCH --mail-type=ALL
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:2
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=%j_arabic_english_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu
#
# The English arm: rerun the two BASELINE conditions on the machine-translated
# transcripts, so that English-vs-Arabic is the only thing that differs.
#
#     sbatch apply_english.sh
#
# This is deliberately the *plain* pipeline -- standard cross-entropy, greedy
# decoding. No class weighting (Arm A), no thresholded scoring (Arm B). Mixing the
# imbalance fixes in here would confound the two questions: we could no longer say
# whether a change came from the language or from the intervention. The English
# counterparts of runs_scored/ and runs_balanced/ can be produced later, on top of
# this, once the language effect is established on its own.
#
# Reads:  processed_datasets_en/   (built by build_english_data.py)
# Writes: runs_en/zeroshot/<model>/<task>/   zero-shot, no fine-tuning
#         runs_en/<model>/<task>/            SFT, standard loss + greedy decoding
#
# Nothing under runs/, runs_scored/ or runs_balanced/ is touched: the Arabic
# results stay exactly as they are.
set -euo pipefail

# ============================== CONFIG ======================================
# Which conditions to run:
#   zeroshot  base models, no training           -> runs_en/zeroshot/
#   sft       fine-tune + evaluate               -> runs_en/
#   both      zero-shot first, then SFT
EXPERIMENT="both"

# The English data. Must exist already -- build_english_data.py makes it from
# translations/<model>.jsonl and it refuses to build if any translation lost content.
DATA_DIR="processed_datasets_en"
RUNS_DIR="runs_en"

# Which models to sweep (one HF repo per line; '#' comments allowed).
MODELS_FILE="models.txt"
# ===========================================================================

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/Arabic_data_match/llm_pipeline_0707"

case "${EXPERIMENT}" in
  zeroshot|sft|both) ;;
  *) echo "EXPERIMENT must be zeroshot|sft|both, got '${EXPERIMENT}'" >&2; exit 1 ;;
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

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "No ${DATA_DIR}/. Build it first:" >&2
  echo "  python build_english_data.py --pred translations/<model>.jsonl" >&2
  exit 1
fi

echo "=========================================================="
echo " EXPERIMENT  = ${EXPERIMENT}  (plain SFT: no class weight, greedy decoding)"
echo " DATA_DIR    = ${DATA_DIR}"
echo " RUNS_DIR    = ${RUNS_DIR}"
echo " MODELS_FILE = ${MODELS_FILE}"
echo "=========================================================="

mapfile -t MODELS < <(grep -vE '^[[:space:]]*(#|$)' "${MODELS_FILE}")
[[ ${#MODELS[@]} -gt 0 ]] || { echo "No models in ${MODELS_FILE}" >&2; exit 1; }

# Zero-shot for every model first: it needs no training, so if the job dies later
# we still have the untrained reference point for the whole sweep.
if [[ "${EXPERIMENT}" == "zeroshot" || "${EXPERIMENT}" == "both" ]]; then
  for MODEL in "${MODELS[@]}"; do
    MODEL="$(echo "${MODEL}" | xargs)"
    RUN_NAME="$(basename "${MODEL}" | tr '[:upper:]' '[:lower:]')"
    RUNS_DIR="${RUNS_DIR}" DATA_DIR="${DATA_DIR}" \
      bash run_zeroshot.sh "${MODEL}" "${RUN_NAME}"
  done
fi

# Plain SFT. TRAIN_ARGS and EVAL_ARGS are left empty on purpose -- train.py then
# defaults to --class-weight none and evaluate.py to --decision greedy, which is
# exactly the baseline condition, just pointed at the English data.
if [[ "${EXPERIMENT}" == "sft" || "${EXPERIMENT}" == "both" ]]; then
  for MODEL in "${MODELS[@]}"; do
    MODEL="$(echo "${MODEL}" | xargs)"
    RUN_NAME="$(basename "${MODEL}" | tr '[:upper:]' '[:lower:]')"
    RUNS_DIR="${RUNS_DIR}" DATA_DIR="${DATA_DIR}" \
      bash run_all.sh "${MODEL}" "${RUN_NAME}"
  done
fi

echo "Done (EXPERIMENT=${EXPERIMENT})."
echo "Zero-shot summaries: ${RUNS_DIR}/zeroshot/<model>/summary.csv"
echo "SFT summaries:       ${RUNS_DIR}/<model>/summary.csv"
echo "Compare to Arabic:   runs/zeroshot/<model>/summary.csv and runs/<model>/summary.csv"
