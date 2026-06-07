#!/bin/bash
#SBATCH --job-name=icl-local-judge
#SBATCH --account=ugr92
#SBATCH --qos=acc_resb
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=47:00:00
#SBATCH --array=0-31                    # 4 datasets × 4 prompt_types × 2 modos = 32 tareas paralelas
#SBATCH --output=logs/%x-%A_%a.out      # %A = array job id, %a = task id
#SBATCH --error=logs/%x-%A_%a.err

# =============================================================================
# ICL — LOCAL LLM-AS-A-JUDGE EN PARALELO (MareNostrum5 / BSC)
#
# Array de 32 tareas: cada una corre el judge sobre un (dataset × prompt_type × modo).
# Las 32 tareas se lanzan a la vez si hay GPUs disponibles → 32× speedup.
# Con ~288 trials/tarea a ~3 min/trial: ~14-16 h por tarea.
#
# Mapa de tareas (task_id = dataset_idx*8 + prompt_idx*2 + mode_idx):
#   dataset:      flowers(0) pets(1) cifar10(2) dtd(3)
#   prompt_type:  nle(0) features(1) rulebased(2) axioms_ontology_v2(3)
#   mode:         query_only(0) query_and_support(1)
#
# Ejemplos:
#   ID  0 → flowers   × nle               × query_only
#   ID  1 → flowers   × nle               × query_and_support
#   ID  8 → flowers   × features          × query_only
#   ID 16 → pets      × nle               × query_only
#   ID 31 → dtd       × axioms_ontology_v2 × query_and_support
#
# Modelo: Qwen3-VL-32B-Thinking (nf4, ~40 GB VRAM por tarea)
#
# USO:
#   # Todas las tareas (máximo paralelismo):
#   sbatch slurm/mn5_local_judge.sh <RUN_DIR>
#
#   # Solo un dataset/prompt/modo concreto:
#   sbatch --array=0  slurm/mn5_local_judge.sh <RUN_DIR>   # flowers × nle × query_only
#   sbatch --array=16 slurm/mn5_local_judge.sh <RUN_DIR>   # pets × nle × query_only
#
#   # Solo los query_only (IDs pares):
#   sbatch --array=0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 slurm/mn5_local_judge.sh <RUN_DIR>
#
#   # Smoke test (limitar a 5 trials por tarea):
#   sbatch --array=0 slurm/mn5_local_judge.sh <RUN_DIR> 5
#
#   # Reanudar una tarea que hizo timeout (ej. job 6 = flowers×axioms×query_only):
#   sbatch --array=6 slurm/mn5_local_judge.sh <RUN_DIR> "" resume
#
# NOTA: Para resultados de inferencia LOCAL (pipeline/local_runs/) los CSV ya
# están divididos por dataset/prompt_type. Pasa el directorio padre como
# RUN_DIR o múltiples dirs separados por espacio entre comillas:
#   sbatch slurm/mn5_local_judge.sh "dir1 dir2 dir3 ..."
# =============================================================================

set -euo pipefail

RUN_DIR="${1:?ERROR: pasa el directorio del experimento como primer argumento}"
LIMIT="${2:-}"    # segundo arg opcional: cap de trials por tarea (útil para smoke)
RESUME="${3:-}"   # tercer arg opcional: cualquier valor no vacío activa --resume

# --- Mapa dataset × prompt_type × modo según SLURM_ARRAY_TASK_ID -------------
# task_id = dataset_idx*8 + prompt_idx*2 + mode_idx
DATASETS=("flowers" "pets" "cifar10" "dtd")
PROMPT_TYPES=("nle" "features" "rulebased" "axioms_ontology_v2")
MODES=("query_only" "query_and_support")

DATASET_IDX=$(( SLURM_ARRAY_TASK_ID / 8 ))
PROMPT_IDX=$(( (SLURM_ARRAY_TASK_ID % 8) / 2 ))
MODE_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

DATASET="${DATASETS[$DATASET_IDX]}"
PROMPT_TYPE="${PROMPT_TYPES[$PROMPT_IDX]}"
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
echo "[+] Array task   : ${SLURM_ARRAY_TASK_ID}/31  →  dataset=${DATASET}  prompt=${PROMPT_TYPE}  mode=${JUDGE_MODE}"
echo "[+] Run dir(s)   : ${RUN_DIR}"
echo "[+] Limit        : ${LIMIT:-<full>}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

# --- Presupuesto de tokens: axioms_ontology_v2 requiere más margen -----------
# Los axiomas DL generan cadenas de razonamiento largas en el modelo thinking;
# 32768 evita truncaciones y los loops que causan parse_error (~28% sin este fix).
MAX_NEW_TOKENS=16384
if [[ "${PROMPT_TYPE}" == "axioms_ontology_v2" ]]; then
    MAX_NEW_TOKENS=32768
fi

# --- Construir args extras ---------------------------------------------------
EXTRA_ARGS=()
if [[ -n "${LIMIT}" ]]; then
    EXTRA_ARGS+=("--limit" "${LIMIT}")
fi
if [[ -n "${RESUME}" ]]; then
    EXTRA_ARGS+=("--resume")
fi

# RUN_DIR puede contener múltiples dirs separados por espacios
# shellcheck disable=SC2086
python execute_local_judge.py \
    --run-dir ${RUN_DIR} \
    --dataset "${DATASET}" \
    --prompt-type "${PROMPT_TYPE}" \
    --judge-mode "${JUDGE_MODE}" \
    --model qwen3-vl-32b-thinking \
    --quantization nf4 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --repetition-penalty 1.05 \
    --debug \
    "${EXTRA_ARGS[@]}"

echo "[+] Done — task ${SLURM_ARRAY_TASK_ID} (${DATASET} / ${PROMPT_TYPE} / ${JUDGE_MODE})"
