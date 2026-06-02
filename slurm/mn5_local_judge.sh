#!/bin/bash
#SBATCH --job-name=icl-local-judge
#SBATCH --account=ugr92
#SBATCH --qos=acc_resb
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-7                     # 4 datasets × 2 modos = 8 tareas paralelas
#SBATCH --output=logs/%x-%A_%a.out      # %A = array job id, %a = task id
#SBATCH --error=logs/%x-%A_%a.err

# =============================================================================
# ICL — LOCAL LLM-AS-A-JUDGE EN PARALELO (MareNostrum5 / BSC)
#
# Array de 8 tareas: cada una corre el judge sobre un (dataset × modo) distinto.
# Las 8 tareas se lanzan a la vez si hay GPUs disponibles → 8× speedup.
#
# Mapa de tareas:
#   ID  Dataset   Modo
#    0  flowers   query_only
#    1  flowers   query_and_support
#    2  pets      query_only
#    3  pets      query_and_support
#    4  cifar10   query_only
#    5  cifar10   query_and_support
#    6  dtd       query_only
#    7  dtd       query_and_support
#
# Modelo: Qwen3-VL-32B-Thinking (BF16, ~64 GB VRAM por tarea, Apache 2.0)
#
# USO:
#   # Todas las tareas (máximo paralelismo):
#   sbatch slurm/mn5_local_judge.sh <RUN_DIR>
#
#   # Solo un dataset/modo concreto para probar:
#   sbatch --array=0 slurm/mn5_local_judge.sh <RUN_DIR>   # flowers + query_only
#   sbatch --array=5 slurm/mn5_local_judge.sh <RUN_DIR>   # cifar10 + query_and_support
#
#   # Solo los query_only (IDs pares):
#   sbatch --array=0,2,4,6 slurm/mn5_local_judge.sh <RUN_DIR>
#
#   # Smoke test (limitar a 10 trials por tarea):
#   sbatch slurm/mn5_local_judge.sh <RUN_DIR> 10
#
# NOTA: Para resultados de inferencia LOCAL (pipeline/local_runs/) los CSV ya
# están divididos por dataset/prompt_type. Pasa el directorio padre como
# RUN_DIR o múltiples dirs separados por espacio entre comillas:
#   sbatch slurm/mn5_local_judge.sh "dir1 dir2 dir3 ..."
# =============================================================================

set -euo pipefail

RUN_DIR="${1:?ERROR: pasa el directorio del experimento como primer argumento}"
LIMIT="${2:-}"    # segundo arg opcional: cap de trials por tarea (útil para smoke)

# --- Mapa dataset × modo según SLURM_ARRAY_TASK_ID ---------------------------
DATASETS=("flowers" "pets" "cifar10" "dtd")
MODES=("query_only" "query_and_support")

DATASET_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
MODE_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

DATASET="${DATASETS[$DATASET_IDX]}"
JUDGE_MODE="${MODES[$MODE_IDX]}"

# --- Entorno offline ----------------------------------------------------------
export HF_HOME="/gpfs/projects/ugr92/ICL/hf_cache"
export HF_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42

# --- Conda -------------------------------------------------------------------
module load anaconda/2023.07
source activate /gpfs/projects/ugr92/ICL/research-explain

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "[+] Host         : $(hostname)"
echo "[+] Array task   : ${SLURM_ARRAY_TASK_ID}/7  →  dataset=${DATASET}  mode=${JUDGE_MODE}"
echo "[+] Run dir(s)   : ${RUN_DIR}"
echo "[+] Limit        : ${LIMIT:-<full>}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

# --- Construir args extras ---------------------------------------------------
EXTRA_ARGS=()
if [[ -n "${LIMIT}" ]]; then
    EXTRA_ARGS+=("--limit" "${LIMIT}")
fi

# RUN_DIR puede contener múltiples dirs separados por espacios
# shellcheck disable=SC2086
python execute_local_judge.py \
    --run-dir ${RUN_DIR} \
    --dataset "${DATASET}" \
    --judge-mode "${JUDGE_MODE}" \
    --model qwen3-vl-32b-thinking \
    --quantization nf4 \
    --max-new-tokens 2048 \
    --repetition-penalty 1.05 \
    --debug \
    "${EXTRA_ARGS[@]}"

echo "[+] Done — task ${SLURM_ARRAY_TASK_ID} (${DATASET} / ${JUDGE_MODE})"
