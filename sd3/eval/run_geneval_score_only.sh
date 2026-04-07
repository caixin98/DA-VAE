#!/usr/bin/env bash
set -euo pipefail

# Score-only GenEval runner.
#
# This script assumes you already have a GenEval-formatted image directory:
# <imagedir>/
#   00000/metadata.jsonl
#   00000/samples/0000.png
#   00001/...
#
# It will run ONLY the scoring + report steps (no image generation).
#
# Usage:
#   bash eval/run_geneval_score_only.sh /path/to/images
#
# Output (by default):
#   <outroot>/results/results.jsonl
#   <outroot>/results/report.json
#   <outroot>/score_only.log
#
# Notes:
# - GenEval scoring requires GPU (evaluate_images.py asserts CUDA).
# - On some clusters LD_LIBRARY_PATH may point to a system GCC libstdc++ that breaks imports.
#   We force conda's ${CONDA_PREFIX}/lib to be first.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_EXE="conda"

IMAGEDIR="${1:-${IMAGEDIR:-}}"
if [[ -z "${IMAGEDIR}" ]]; then
  echo "[score_only] Usage: bash eval/run_geneval_score_only.sh /path/to/images" >&2
  echo "[score_only] Or set IMAGEDIR=/path/to/images" >&2
  exit 2
fi
if [[ ! -d "${IMAGEDIR}" ]]; then
  echo "[score_only] ERROR: IMAGEDIR is not a directory: ${IMAGEDIR}" >&2
  exit 2
fi
if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "[score_only] ERROR: conda executable not found: ${CONDA_EXE}" >&2
  exit 2
fi

GENEVAL_ENV_NAME="${GENEVAL_ENV_NAME:-geneval}"
GENEVAL_MODEL_DIR="${GENEVAL_MODEL_DIR:-${REPO_ROOT}/eval/assets/geneval_models}"
# Derive OUTROOT: if IMAGEDIR ends with "/images", write outputs next to it.
if [[ "$(basename "${IMAGEDIR}")" == "images" ]]; then
  OUTROOT="$(cd "$(dirname "${IMAGEDIR}")" && pwd)"
else
  OUTROOT="$(cd "${IMAGEDIR}" && pwd)"
fi

RESULTS_DIR="${RESULTS_DIR:-${OUTROOT}/results}"
LOGFILE="${LOGFILE:-${OUTROOT}/score_only.log}"

mkdir -p "${RESULTS_DIR}"

exec > >(tee -a "${LOGFILE}") 2>&1

echo "[score_only] imagedir=${IMAGEDIR}"
echo "[score_only] results_dir=${RESULTS_DIR}"
echo "[score_only] logfile=${LOGFILE}"
echo "[score_only] geneval_env=${GENEVAL_ENV_NAME}"
echo "[score_only] geneval_model_dir=${GENEVAL_MODEL_DIR}"

echo "[score_only] scoring images..."
"${CONDA_EXE}" run -n "${GENEVAL_ENV_NAME}" bash -c \
  'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; \
   python "'"${REPO_ROOT}/geneval/evaluation/evaluate_images.py"'" "'"${IMAGEDIR}"'" \
     --outfile "'"${RESULTS_DIR}/results.jsonl"'" \
     --model-path "'"${GENEVAL_MODEL_DIR}"'" \
     --verbose \
     --log-every "${LOG_EVERY:-1}" \
     --heartbeat-secs "${HEARTBEAT_SECS:-60}" \
     --timing \
     --clip-num-workers "${CLIP_NUM_WORKERS:-0}" \
     --clip-batch-size "${CLIP_BATCH_SIZE:-16}"'

echo "[score_only] writing report.json..."
"${CONDA_EXE}" run -n "${GENEVAL_ENV_NAME}" bash -c \
  'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; python "'"${REPO_ROOT}/eval/geneval_report.py"'" --results "'"${RESULTS_DIR}/results.jsonl"'" --out "'"${RESULTS_DIR}/report.json"'"'

echo "[score_only] done."
echo "[score_only] report=${RESULTS_DIR}/report.json"


