#!/usr/bin/env bash
set -euo pipefail

# End-to-end GenEval pipeline for SD3.5M at 512x512.
#
# This runner:
# 1) Creates/updates a dedicated `geneval` conda env for scoring
# 2) Runs generation in `davae` env (SD3.5 pipeline)
# 3) Runs evaluation + writes a JSON report
#
# Outputs:
# eval/outputs/geneval/<run_name>/
#   images/   (GenEval image folder structure)
#   results/results.jsonl
#   results/report.json

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_EXE="conda"

# ---- Run config (override via env vars) ----
MODEL_ID="${MODEL_ID:-stabilityai/stable-diffusion-3.5-medium}"
SD_LOCAL_FILES_ONLY="${SD_LOCAL_FILES_ONLY:-1}"   # 1 => do not download SD weights (use local HF cache / local dir)
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
STEPS="${STEPS:-30}"
GUIDANCE="${GUIDANCE:-4.5}"
SHIFT="${SHIFT:-}"
N_SAMPLES="${N_SAMPLES:-4}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-bfloat16}"
MAX_PROMPTS="${MAX_PROMPTS:-}"

GENEVAL_ENV_NAME="${GENEVAL_ENV_NAME:-geneval}"
GENEVAL_MODEL_DIR="${GENEVAL_MODEL_DIR:-${REPO_ROOT}/eval/assets/geneval_models}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-sd35m_geneval_${TS}_${WIDTH}x${HEIGHT}_s${STEPS}_g${GUIDANCE}_n${N_SAMPLES}_seed${SEED}}"
OUTROOT="${REPO_ROOT}/eval/outputs/geneval/${RUN_NAME}"
IMAGEDIR="${OUTROOT}/images"
RESULTS_DIR="${OUTROOT}/results"
LOGFILE="${OUTROOT}/run.log"

PROMPTS_JSONL="${REPO_ROOT}/geneval/prompts/evaluation_metadata.jsonl"

if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "[run_geneval] ERROR: conda executable not found: ${CONDA_EXE}" >&2
  exit 2
fi
if [[ ! -f "${PROMPTS_JSONL}" ]]; then
  echo "[run_geneval] ERROR: prompts not found: ${PROMPTS_JSONL}" >&2
  exit 2
fi

mkdir -p "${OUTROOT}" "${RESULTS_DIR}"

# Stream all stdout/stderr to a persistent log file for easy debugging:
#   tail -f eval/outputs/geneval/<run_name>/run.log
exec > >(tee -a "${LOGFILE}") 2>&1

echo "[run_geneval] outroot=${OUTROOT}"
echo "[run_geneval] logfile=${LOGFILE}"

# Step 1: ensure scoring env + detector weights (optional).
# Recommended: run this once on CPU using `cpu` prefix, then set SKIP_GENEVAL_ENV_SETUP=1 for GPU runs.
if [[ "${SKIP_GENEVAL_ENV_SETUP:-0}" != "1" ]]; then
  GENEVAL_ENV_NAME="${GENEVAL_ENV_NAME}" GENEVAL_MODEL_DIR="${GENEVAL_MODEL_DIR}" bash "${REPO_ROOT}/eval/setup_geneval_env.sh"
else
  echo "[run_geneval] SKIP_GENEVAL_ENV_SETUP=1 (skip conda env/model setup)"
fi

echo "[run_geneval] generating images (conda env: davae)..."
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

"${CONDA_EXE}" run -n davae python "${REPO_ROOT}/eval/generate_sd35m_geneval.py" \
  "${GEN_ARGS[@]}"

echo "[run_geneval] evaluating images (conda env: ${GENEVAL_ENV_NAME})..."
# Some clusters inject a system GCC path into LD_LIBRARY_PATH, which can override conda's libstdc++ and
# cause `GLIBCXX_* not found` errors (often via PIL/libLerc). Force conda libs first for scoring.
"${CONDA_EXE}" run -n "${GENEVAL_ENV_NAME}" bash -c \
  'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; python "'"${REPO_ROOT}/geneval/evaluation/evaluate_images.py"'" "'"${IMAGEDIR}"'" --outfile "'"${RESULTS_DIR}/results.jsonl"'" --model-path "'"${GENEVAL_MODEL_DIR}"'" --verbose --timing --log-every 1 --heartbeat-secs 60 --clip-num-workers 0'

echo "[run_geneval] writing report.json (conda env: ${GENEVAL_ENV_NAME})..."
"${CONDA_EXE}" run -n "${GENEVAL_ENV_NAME}" bash -c \
  'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; python "'"${REPO_ROOT}/eval/geneval_report.py"'" --results "'"${RESULTS_DIR}/results.jsonl"'" --out "'"${RESULTS_DIR}/report.json"'"'

echo "[run_geneval] done. report=${RESULTS_DIR}/report.json"


