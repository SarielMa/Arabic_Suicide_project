#!/bin/bash
#SBATCH --job-name=arabic_suicide_bert
#SBATCH --mail-type=ALL
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=%j_arabic_suicide_bert_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# ==== Long-transcript handling (transcripts far exceed BERT's 512-token cap) ====
# Edit these two values directly, then just `sbatch apply_server.sh`.
#   CHUNKING   = 1 -> read the FULL transcript (512-token windows + pooling)
#                0 -> truncate to 512 tokens
#   TRUNCATION = head | tail  (which end to keep; also picks the kept windows
#                when a transcript needs more than max_chunks windows)
# (Both still accept a submit-time override, e.g. --export=ALL,CHUNKING=0.)
export CHUNKING="${CHUNKING:-1}"
export TRUNCATION="${TRUNCATION:-head}"

REPO_ROOT="/nfs/roberts/project/pi_sjf37/lm2445/Arabic_data_match/bert_pipeline_0707"
PIPELINE_SH="${REPO_ROOT}/run_pipeline.sh"

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

which nvcc
nvcc --version
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

echo "REPO_ROOT=${REPO_ROOT}"
echo "PIPELINE_SH=${PIPELINE_SH}"
echo "TRUNCATION=${TRUNCATION}  CHUNKING=${CHUNKING}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NUM_GPUS=${NUM_GPUS}"

# run_pipeline.sh reads models.txt and fine-tunes + evaluates each model.
bash "${PIPELINE_SH}"
