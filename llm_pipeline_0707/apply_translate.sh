#!/bin/bash
#SBATCH --job-name=arabic_translate
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:2
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=%j_arabic_translate.txt
#SBATCH --mail-user=linhai.ma@yale.edu
#
# Translate the Arabic transcripts into English, then rebuild the 5 task datasets.
#
#     sbatch apply_translate.sh
#
# Qwen2.5-72B is ~145GB and is not in the HF cache yet; the node has internet, so
# the first run downloads it (add ~30-60 min to the first job, cached after that).
#
# Outputs -- the Arabic originals and all model results are left untouched:
#   translations/<model>.jsonl        Arabic + English side by side, plus QC flags
#   processed_datasets_en/<task>/...  training-ready English data (BUILD_DATASETS=1)
set -euo pipefail

# ============================== CONFIG ======================================
TRANSLATOR="Qwen/Qwen2.5-72B-Instruct"

# PILOT: translate only the first N transcripts (0 = all 438).
# START WITH A PILOT. Read the output before spending the full run: if the model
# sanitizes or refuses the self-harm content, the translation deletes exactly the
# evidence the PI/SI/IM labels depend on, and the English experiment would be
# measuring translation damage rather than language.
PILOT_N=20

# Re-attempts for a transcript whose translation fails the QC checks
# (refusal, summarization, untranslated Arabic, dropped <PERS> tags).
RETRIES=2

# Build processed_datasets_en/ after translating? Only meaningful on a full run,
# and it will refuse to build if any translation is missing or flagged.
BUILD_DATASETS=0
# ===========================================================================

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/Arabic_data_match/llm_pipeline_0707"
RUN_NAME="$(basename "${TRANSLATOR}" | tr '[:upper:]' '[:lower:]')"
OUT_JSONL="translations/${RUN_NAME}.jsonl"

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
echo " TRANSLATOR = ${TRANSLATOR}"
echo " PILOT_N    = ${PILOT_N}  (0 = all 438 unique transcripts)"
echo " OUT        = ${OUT_JSONL}"
echo "=========================================================="

python prepare_data.py

python translate.py \
    --model "${TRANSLATOR}" \
    --out "${OUT_JSONL}" \
    --limit "${PILOT_N}" \
    --retries "${RETRIES}"

# QC report: flag rate overall, and -- the number that matters -- whether the
# failures cluster on positive-label calls.
python inspect_translations.py --pred "${OUT_JSONL}"

if [[ "${BUILD_DATASETS}" == "1" ]]; then
  python build_english_data.py --pred "${OUT_JSONL}"
fi

echo "Done."
echo "Review side-by-side:  python inspect_translations.py --pred ${OUT_JSONL} --show 5"
echo "Review the problems:  python inspect_translations.py --pred ${OUT_JSONL} --only-flagged --show 5"
