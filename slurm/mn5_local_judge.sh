#!/bin/bash
#SBATCH --job-name=icl-local-judge
#SBATCH --account=ugr92
#SBATCH --qos=acc_bsccs
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Offline local LLM-as-a-judge run on MareNostrum5 (BSC).
#
# Model:   Qwen3-VL-32B-Thinking (Apache 2.0, ~64 GB VRAM in BF16 on H100-80GB).
# Default judge mode: query_only (paper-comparable).
#
# Submit with, for example:
#     sbatch slurm/mn5_local_judge.sh \
#         "pipeline/openrouter_runs/full_experiment_single__20260427_213833" \
#         "query_only" \
#         "10"        # optional --limit (omit for full run)
#
# Smoke test recommended: run with --limit 10 (third positional arg) before
# launching the full evaluation, especially before the Sunday deadline.

set -euo pipefail

RUN_DIR="${1:?ERROR: pass the experiment run directory as the first arg}"
JUDGE_MODE="${2:-query_only}"   # query_only | query_and_support
LIMIT="${3:-}"                  # optional cap on number of trials

# --- Offline / cache environment -------------------------------------------
export HF_HOME="/gpfs/projects/ugr92/ICL/hf_cache"
export HF_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Repro / determinism (we do greedy decoding anyway).
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- Activate the project's conda env ---------------------------------------
# Adjust the conda init path if your MN5 setup differs.
source "${CONDA_PREFIX_ROOT:-/gpfs/projects/ugr92/conda}/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate research-explain

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "[+] Host        : $(hostname)"
echo "[+] CUDA devices: ${CUDA_VISIBLE_DEVICES}"
echo "[+] Run dir     : ${RUN_DIR}"
echo "[+] Judge mode  : ${JUDGE_MODE}"
echo "[+] Limit       : ${LIMIT:-<full>}"
nvidia-smi || true

# --- Launch the judge -------------------------------------------------------
EXTRA_ARGS=()
if [[ -n "${LIMIT}" ]]; then
    EXTRA_ARGS+=("--limit" "${LIMIT}")
fi

python execute_local_judge.py \
    --run-dir "${RUN_DIR}" \
    --judge-mode "${JUDGE_MODE}" \
    --model "qwen3-vl-32b-thinking" \
    --quantization "auto" \
    --debug \
    "${EXTRA_ARGS[@]}"

echo "[+] Done."
