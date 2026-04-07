#!/usr/bin/env bash
set -euo pipefail

# Minimal GenEval pipeline:
# 1) Generate images in `davae` env
# 2) Evaluate images in `geneval` env
# 3) Write report.json (overall_score)
#
# This script intentionally does NOT try to create/update conda envs or download models.
# Prepare them once beforehand if needed:
# - `eval/setup_geneval_env.sh` (creates `geneval` env + downloads Mask2Former weights)
#
# Usage:
#   bash eval/run_geneval_simple.sh
#
# Outputs:
#   eval/outputs/geneval/<run_name>/
#     images/
#     results/results.jsonl
#     results/report.json

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_EXE="conda"

# ---- Config (override via env vars) ----
MODEL_ID="${MODEL_ID:-stabilityai/stable-diffusion-3.5-medium}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
STEPS="${STEPS:-30}"
GUIDANCE="${GUIDANCE:-4.5}"
SHIFT="${SHIFT:-}"
N_SAMPLES="${N_SAMPLES:-4}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-bfloat16}"
SD_LOCAL_FILES_ONLY="${SD_LOCAL_FILES_ONLY:-1}"
MAX_PROMPTS="${MAX_PROMPTS:-}"

GEN_ENV="${GEN_ENV:-davae}"
SCORE_ENV="${SCORE_ENV:-geneval}"
GENEVAL_MODEL_DIR="${GENEVAL_MODEL_DIR:-${REPO_ROOT}/eval/assets/geneval_models}"
PROMPTS_JSONL="${PROMPTS_JSONL:-${REPO_ROOT}/geneval/prompts/evaluation_metadata.jsonl}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-geneval_${TS}_${WIDTH}x${HEIGHT}_s${STEPS}_g${GUIDANCE}_n${N_SAMPLES}_seed${SEED}}"
OUTROOT="${REPO_ROOT}/eval/outputs/geneval/${RUN_NAME}"
IMAGEDIR="${OUTROOT}/images"
RESULTS_DIR="${OUTROOT}/results"
LOGFILE="${OUTROOT}/run.log"

mkdir -p "${OUTROOT}" "${RESULTS_DIR}"
exec > >(tee -a "${LOGFILE}") 2>&1

echo "[geneval_simple] outroot=${OUTROOT}"
echo "[geneval_simple] logfile=${LOGFILE}"
echo "[geneval_simple] gen_env=${GEN_ENV} score_env=${SCORE_ENV}"
echo "[geneval_simple] prompts=${PROMPTS_JSONL}"
echo "[geneval_simple] geneval_model_dir=${GENEVAL_MODEL_DIR}"

echo "[geneval_simple] step1: generate images..."
GEN_ARGS=(
  --metadata-file "${PROMPTS_JSONL}"
  --outdir "${IMAGEDIR}"
  --model-id "${MODEL_ID}"
  --dtype "${DTYPE}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --steps "${STEPS}"
  --guidance "${GUIDANCE}"
  --n-samples "${N_SAMPLES}"
  --seed "${SEED}"
  --resume
)
if [[ "${SD_LOCAL_FILES_ONLY}" == "1" ]]; then
  GEN_ARGS+=( --local-files-only )
fi
if [[ -n "${SHIFT}" ]]; then
  GEN_ARGS+=( --shift "${SHIFT}" )
fi
if [[ -n "${MAX_PROMPTS}" ]]; then
  GEN_ARGS+=( --max-prompts "${MAX_PROMPTS}" )
fi

"${CONDA_EXE}" run -n "${GEN_ENV}" python "${REPO_ROOT}/eval/generate_sd35m_geneval.py" "${GEN_ARGS[@]}"

echo "[geneval_simple] step2: evaluate images..."
"${CONDA_EXE}" run -n "${SCORE_ENV}" bash -c \
  'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; \
   python "'"${REPO_ROOT}/geneval/evaluation/evaluate_images.py"'" "'"${IMAGEDIR}"'" \
     --outfile "'"${RESULTS_DIR}/results.jsonl"'" \
     --model-path "'"${GENEVAL_MODEL_DIR}"'" \
     --verbose --timing --log-every 1 --heartbeat-secs 60 --clip-num-workers 0'

echo "[geneval_simple] step3: write report.json..."
"${CONDA_EXE}" run -n "${SCORE_ENV}" bash -c \
  'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; \
   python "'"${REPO_ROOT}/eval/geneval_report.py"'" \
     --results "'"${RESULTS_DIR}/results.jsonl"'" \
     --out "'"${RESULTS_DIR}/report.json"'"'

echo "[geneval_simple] done. report=${RESULTS_DIR}/report.json"


