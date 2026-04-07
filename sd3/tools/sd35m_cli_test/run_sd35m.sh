#!/usr/bin/env bash
set -euo pipefail

# SD3.5M CLI runner.
# Edit the variables below to control generation results.

# ===== Model / runtime =====
MODEL_ID="${MODEL_ID:-stabilityai/stable-diffusion-3.5-medium}"  # HF repo id or local directory
DEVICE="${DEVICE:-cuda}"                                       # cuda | cpu
DTYPE="${DTYPE:-bfloat16}"                                     # bfloat16 | float16 | float32
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"                      # 1 => only local cache

# ===== Generation knobs =====
PROMPT="${PROMPT:-A cinematic photo of a futuristic city at night, rain, neon lights, ultra detailed, 8k}"
# NEG_PROMPT="${NEG_PROMPT:-deformed, low quality, bad aesthetics, blurry}"
NEG_PROMPT="${NEG_PROMPT:-}"
WIDTH="${WIDTH:-2048}"
HEIGHT="${HEIGHT:-2048}"
STEPS="${STEPS:-30}"
GUIDANCE="${GUIDANCE:-3.5}"
SEED="${SEED:-0}"                 # -1 => random
BATCH_SIZE="${BATCH_SIZE:-1}"     # num_images_per_prompt

# ===== Scheduler / memory =====
SHIFT="${SHIFT:-1.0}"
USE_DYNAMIC_SHIFTING="${USE_DYNAMIC_SHIFTING:-0}"   # 1 => enable
# Default to safer settings so the first run is more likely to succeed on limited VRAM.
CPU_OFFLOAD="${CPU_OFFLOAD:-1}"                     # 1 => enable
ATTN_SLICING="${ATTN_SLICING:-1}"                   # 1 => enable

# ===== Output =====
OUTDIR="${OUTDIR:-outputs/sd35m}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# IMPORTANT: On many clusters, the gpuX/srun job does NOT inherit your interactive shell's conda activation.
# Use an explicit python path from the davae env by default so dependencies resolve on compute nodes.
DEFAULT_DAVAE_PY="python"
PY="${PYTHON_BIN:-${DEFAULT_DAVAE_PY}}"

if [[ ! -x "${PY}" ]]; then
  echo "[sd35m_cli] ERROR: python not found or not executable: ${PY}" >&2
  echo "[sd35m_cli] Tip: set PYTHON_BIN=/path/to/python (e.g. your conda env python) and retry." >&2
  exit 2
fi

ARGS=(
  --model-id "${MODEL_ID}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --prompt "${PROMPT}"
  --negative-prompt "${NEG_PROMPT}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --steps "${STEPS}"
  --guidance "${GUIDANCE}"
  --seed "${SEED}"
  --batch-size "${BATCH_SIZE}"
  --shift "${SHIFT}"
  --output-dir "${OUTDIR}"
)

if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  ARGS+=( --local-files-only )
fi
if [[ "${USE_DYNAMIC_SHIFTING}" == "1" ]]; then
  ARGS+=( --use-dynamic-shifting )
fi
if [[ "${CPU_OFFLOAD}" == "1" ]]; then
  ARGS+=( --enable-cpu-offload )
fi
if [[ "${ATTN_SLICING}" == "1" ]]; then
  ARGS+=( --enable-attention-slicing )
fi

exec "${PY}" "${SCRIPT_DIR}/sd35m_infer.py" "${ARGS[@]}"


