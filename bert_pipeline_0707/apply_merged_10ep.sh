#!/bin/bash
#SBATCH --job-name=merged_bert10
#SBATCH --mail-type=ALL
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=%j_merged_bert_ep10_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# Two-level risk experiment for the BERT pipeline: fine-tune + evaluate the merged
# med_risk / high_risk tasks on BOTH Arabic and English. Submit with no arguments:
#
#     sbatch apply_merged_10ep.sh
#
# Reads:  ../llm_pipeline_0707/processed_datasets_merged/     (Arabic)
#         ../llm_pipeline_0707/processed_datasets_merged_en/  (English)
#         Both are built by the LLM pipeline's build_merged_data.py -- run it there
#         first if those directories are missing.
# Writes: runs_merged_ep10/<model>/<task>/     Arabic
#         runs_en_merged_ep10/<model>/<task>/  English
#
# where <task> is med_risk or high_risk. Nothing under runs/ or runs_en/ is touched.
#
# Each language uses its own model list, mirroring apply_server.sh / apply_english.sh:
# Arabic encoders for the Arabic transcripts, English encoders for the translations.
#
# Epochs are set to 10, matching the merged LLM-pipeline 10-epoch arm. This differs
# from train.py's default of 4, which the 5-task BERT baseline in runs/ used, and from
# the 3-epoch arm in runs_merged_ep3/ -- which must NOT share this output tree, or
# run_all.sh resume would let the two jobs skip each other and mix epoch counts.
#
# Resume: run_all.sh skips any task whose model + eval already exist, so a job that
# hits the wall clock can simply be resubmitted.

# ============================== CONFIG ======================================
# SFT epochs (matches the merged LLM 10-epoch arm; train.py's own default is 4).
EPOCHS="10"

# The two merged tasks, swept in place of the 5 C-SSRS tasks.
export TASK_LIST="med_risk high_risk"
export TRAIN_ARGS="--epochs ${EPOCHS}"

# Long-transcript handling: match the Arabic/English baselines unless overridden.
export CHUNKING="${CHUNKING:-1}"
export TRUNCATION="${TRUNCATION:-head}"

# Which language(s) to sweep: arabic | english | both
# NOTE: deliberately NOT named LANG -- that is the POSIX locale variable, which
# SLURM propagates into the job (LANG=C.UTF-8), so a "${LANG:-both}" default would
# silently inherit the locale instead of falling back to "both".
SWEEP_LANG="${SWEEP_LANG:-both}"
# ===========================================================================

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/Arabic_data_match/bert_pipeline_0707"
PIPELINE_SH="${REPO_ROOT}/run_pipeline.sh"
MERGED_ROOT="../llm_pipeline_0707"

case "${SWEEP_LANG}" in
  arabic|english|both) ;;
  *) echo "SWEEP_LANG must be arabic|english|both, got '${SWEEP_LANG}'" >&2; exit 1 ;;
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
[[ -f "${PIPELINE_SH}" ]] || { echo "Missing pipeline script: ${PIPELINE_SH}" >&2; exit 1; }

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
  NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
fi
if [[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -lt 1 ]]; then
  echo "Unable to detect available GPUs." >&2
  exit 1
fi
export NUM_GPUS

# Build the (data dir, runs dir, model list) triples to sweep, in order. Each
# language keeps its own model list, as in apply_server.sh / apply_english.sh.
SWEEPS=()
if [[ "${SWEEP_LANG}" == "arabic" || "${SWEEP_LANG}" == "both" ]]; then
  SWEEPS+=("${MERGED_ROOT}/processed_datasets_merged|runs_merged_ep10|models.txt")
fi
if [[ "${SWEEP_LANG}" == "english" || "${SWEEP_LANG}" == "both" ]]; then
  SWEEPS+=("${MERGED_ROOT}/processed_datasets_merged_en|runs_en_merged_ep10|models_english.txt")
fi

# Fail fast before burning GPU time: every task split must exist up front.
for SWEEP in "${SWEEPS[@]}"; do
  IFS='|' read -r D R M <<< "${SWEEP}"
  [[ -f "${M}" ]] || { echo "Missing model list: ${M}" >&2; exit 1; }
  for TASK in ${TASK_LIST}; do
    for SPLIT in train test; do
      F="${D}/${TASK}/${SPLIT}.jsonl"
      [[ -f "${F}" ]] || {
        echo "Missing ${F}." >&2
        echo "Build it first: cd ${MERGED_ROOT} && python build_merged_data.py" >&2
        exit 1
      }
    done
  done
done

echo "=========================================================="
echo " EXPERIMENT  = merged two-level BERT (med_risk / high_risk)"
echo " TASKS       = ${TASK_LIST}"
echo " EPOCHS      = ${EPOCHS}"
echo " SWEEP_LANG  = ${SWEEP_LANG}"
echo " TRUNCATION  = ${TRUNCATION}  CHUNKING=${CHUNKING}"
echo " NUM_GPUS    = ${NUM_GPUS}"
echo "=========================================================="

for SWEEP in "${SWEEPS[@]}"; do
  IFS='|' read -r D R M <<< "${SWEEP}"
  echo "########## SWEEP: ${D} -> ${R} (models: ${M}) ##########"
  DATA_DIR="${D}" RUNS_DIR="${R}" MODELS_FILE="${M}" bash "${PIPELINE_SH}"
done

echo "Done (merged two-level BERT, ${EPOCHS} epochs, SWEEP_LANG=${SWEEP_LANG})."
echo "Arabic summaries:  runs_merged_ep10/<model>/summary.csv"
echo "English summaries: runs_en_merged_ep10/<model>/summary.csv"
