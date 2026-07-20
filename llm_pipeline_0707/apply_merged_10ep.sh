#!/bin/bash
#SBATCH --job-name=merged10
#SBATCH --mail-type=ALL
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:2
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=%j_merged_ep10_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu
#
# Two-level risk experiment: plain SFT for 10 epochs on the merged med_risk /
# high_risk datasets, on BOTH Arabic and English, for every model in models.txt.
# Submit with no arguments:
#
#     sbatch apply_merged_10ep.sh
#
# This is the *plain* pipeline -- standard cross-entropy, greedy decoding. No class
# weighting (Arm A), no thresholded scoring (Arm B). Epoch count is 10, so against the
# 3-epoch merged arm in runs_merged_ep3/ the only thing that differs is training
# length -- the merged-label analogue of the runs_ep10/ experiment.
#
# Reads:  processed_datasets_merged/     (Arabic)   PRE-BUILT -- see below
#         processed_datasets_merged_en/  (English)  PRE-BUILT -- see below
# Writes: runs_merged_ep10/<model>/<task>/     Arabic  SFT, 10 epochs, greedy decoding
#         runs_en_merged_ep10/<model>/<task>/  English SFT, 10 epochs, greedy decoding
#
# where <task> is med_risk or high_risk. Nothing under runs/, runs_en/, runs_scored/,
# runs_balanced/, runs_balanced_only/, runs_ep10/, runs_en_ep10/, runs_merged_ep3/ or
# runs_en_merged_ep3/ is touched -- in particular this arm must NOT share an output
# tree with apply_merged.sh (3 epochs), or run_all.sh's resume check would let the
# two jobs skip each other's tasks and silently mix epoch counts in one directory.
#
# There is no zero-shot column for these tasks yet: the merged questions differ from
# the 5 C-SSRS questions, so runs/zeroshot/ does NOT transfer. Run run_zeroshot.sh
# against the merged data dirs if a zero-shot comparison is wanted.
#
# Resume: run_all.sh skips any task whose adapter + eval already exist, so if this
# job hits the wall clock just resubmit it -- finished tasks are not retrained. The
# merged datasets are NOT rebuilt by this job, so a resubmission reads the identical
# split rather than reshuffling under the already-trained adapters.
set -euo pipefail

# ============================== CONFIG ======================================
# Number of SFT epochs. train.py defaults to 3; this arm deliberately trains longer.
EPOCHS="10"

# Which language(s) to sweep:
#   arabic   processed_datasets_merged/     -> runs_merged_ep10/
#   english  processed_datasets_merged_en/  -> runs_en_merged_ep10/
#   both     Arabic first, then English
# NOTE: deliberately NOT named LANG -- that is the POSIX locale variable, which
# SLURM propagates into the job. Assigning it here would clobber the locale for
# every child process, and a "${LANG:-both}" default would inherit C.UTF-8 instead.
SWEEP_LANG="both"

# The two merged tasks, swept in place of the 5 C-SSRS tasks.
TASK_LIST="med_risk high_risk"

# Which models to sweep (one HF repo per line; '#' comments allowed).
MODELS_FILE="models.txt"
# ===========================================================================

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/Arabic_data_match/llm_pipeline_0707"

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

# NOTE: this job deliberately does NOT build the merged datasets. They are a shared,
# pre-built artifact -- run `python build_merged_data.py` once, by hand, before
# submitting. Rebuilding here would rewrite the very files every concurrent job is
# reading (torn reads), and would make each arm re-derive its own split instead of
# reading one fixed one: if the 5-task sources ever changed between two submissions,
# the arms would silently train and test on different data with nothing to flag it.
# The guard below refuses to start if the artifact is missing.

# Build the (data dir, runs dir) pairs to sweep, in order.
DATA_DIRS=()
RUNS_DIRS=()
if [[ "${SWEEP_LANG}" == "arabic" || "${SWEEP_LANG}" == "both" ]]; then
  DATA_DIRS+=("processed_datasets_merged");    RUNS_DIRS+=("runs_merged_ep10")
fi
if [[ "${SWEEP_LANG}" == "english" || "${SWEEP_LANG}" == "both" ]]; then
  DATA_DIRS+=("processed_datasets_merged_en"); RUNS_DIRS+=("runs_en_merged_ep10")
fi

# Fail fast if a requested language is missing a task split.
for i in "${!DATA_DIRS[@]}"; do
  for TASK in ${TASK_LIST}; do
    for SPLIT in train test; do
      F="${DATA_DIRS[$i]}/${TASK}/${SPLIT}.jsonl"
      [[ -f "${F}" ]] || {
        echo "Missing ${F}." >&2
        echo "Build the merged datasets first: python build_merged_data.py" >&2
        exit 1
      }
    done
  done
done

echo "=========================================================="
echo " EXPERIMENT  = merged two-level SFT (no class weight, greedy decoding)"
echo " TASKS       = ${TASK_LIST}"
echo " EPOCHS      = ${EPOCHS}"
echo " SWEEP_LANG  = ${SWEEP_LANG}"
echo " MODELS_FILE = ${MODELS_FILE}"
echo "=========================================================="

mapfile -t MODELS < <(grep -vE '^[[:space:]]*(#|$)' "${MODELS_FILE}")
[[ ${#MODELS[@]} -gt 0 ]] || { echo "No models in ${MODELS_FILE}" >&2; exit 1; }

# Both conditions use the ORIGINAL decision rule: EVAL_ARGS is left empty, so
# evaluate.py defaults to --decision greedy -- neither the class-weighted arm (A)
# nor the prior-thresholded arm (B). PREPARE_CMD=true skips run_all.sh's / run_
# zeroshot.sh's prepare_data.py step, which would rebuild the 5-task
# processed_datasets/ tree that is not used here.
for i in "${!DATA_DIRS[@]}"; do
  DATA_DIR="${DATA_DIRS[$i]}"
  RUNS_DIR="${RUNS_DIRS[$i]}"
  echo "########## SWEEP: ${DATA_DIR} -> ${RUNS_DIR} (${EPOCHS} epochs) ##########"
  for MODEL in "${MODELS[@]}"; do
    MODEL="$(echo "${MODEL}" | xargs)"
    RUN_NAME="$(basename "${MODEL}" | tr '[:upper:]' '[:lower:]')"

    # Zero-shot (base model, no adapter) into ${RUNS_DIR}/zeroshot/. Independent of
    # EPOCHS, so each epoch-arm computes its own copy and stays self-contained.
    RUNS_DIR="${RUNS_DIR}" DATA_DIR="${DATA_DIR}" \
    TASK_LIST="${TASK_LIST}" PREPARE_CMD="true" \
      bash run_zeroshot.sh "${MODEL}" "${RUN_NAME}"

    # Supervised fine-tuning + greedy eval.
    RUNS_DIR="${RUNS_DIR}" DATA_DIR="${DATA_DIR}" \
    TASK_LIST="${TASK_LIST}" PREPARE_CMD="true" \
    TRAIN_ARGS="--epochs ${EPOCHS}" \
      bash run_all.sh "${MODEL}" "${RUN_NAME}"
  done
done

echo "Done (merged two-level: zero-shot + ${EPOCHS}-epoch SFT, SWEEP_LANG=${SWEEP_LANG})."
echo "Arabic  zero-shot: runs_merged_ep10/zeroshot/<model>/summary.csv"
echo "Arabic  SFT:       runs_merged_ep10/<model>/summary.csv"
echo "English zero-shot: runs_en_merged_ep10/zeroshot/<model>/summary.csv"
echo "English SFT:       runs_en_merged_ep10/<model>/summary.csv"
