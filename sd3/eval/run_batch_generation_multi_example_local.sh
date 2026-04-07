#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${PYTHON:=python}"

if [[ -n "${GPUS:-}" ]]; then
    DEFAULT_GPU_LIST="${GPUS}"
else
    if command -v nvidia-smi >/dev/null 2>&1; then
        DEFAULT_GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
    else
        DEFAULT_GPU_LIST=""
    fi
fi

DEFAULT_CONFIG="inference/config/token_text_image_da_vae_with_lora_diff_sd35_local_residual.yaml"
DEFAULT_OUTPUT_ROOT="${PROJECT_ROOT}/evaluation_outputs"
DEFAULT_METADATA="${SCRIPT_DIR}/meta_data_test10.json"

GUIDANCE_SCALE="${GUIDANCE_SCALE:-2.5}"
NUM_STEPS="${NUM_STEPS:-30}"
SHIFT_VALUE="${SHIFT_VALUE:-4.0}"
SEED_VALUE="${SEED_VALUE:-1234}"
RUN_SD35_BASELINE_COMPARISON=0
BASELINE_MODE="${BASELINE_MODE:-original}"

usage() {
    cat <<EOF
Usage: ${0##*/} [--gpus GPU_LIST] [--config PATH] [--output-root PATH] [--metadata PATH] [--steps N] [--cfg VALUE] [--shift VALUE] [--seed VALUE] [--compare-sd35] [--baseline-mode MODE] [CONFIG_PATH] [--extra-args ...]

Runs evaluation/batch_generate_from_metadata.py across multiple GPUs by sharding the metadata IDs.

Environment overrides:
  GPUS              default: ${DEFAULT_GPU_LIST}
  CONFIG            default: ${DEFAULT_CONFIG}
  OUTPUT_ROOT       default: ${DEFAULT_OUTPUT_ROOT}
  METADATA          default: ${DEFAULT_METADATA}
  GUIDANCE_SCALE    default: 3.5
  NUM_STEPS         default: 50
  SHIFT_VALUE       default: 5.0
  SEED_VALUE        default: 1234
  PYTHON            default: python
  BASELINE_MODE     default: ${BASELINE_MODE} (low|original)

Remaining arguments are forwarded to the Python script (e.g. --categories, --skip-existing).
You may also provide CONFIG_PATH as the first positional argument (yaml/yml/json) instead of --config.

Flags:
  --compare-sd35       Additionally run SD3.5 baseline at 512 and 1024 for comparison
  --baseline-mode M    Baseline mode to use when --compare-sd35 is set (low|original, default: ${BASELINE_MODE})
EOF
}

CONFIG_PATH="${CONFIG:-$DEFAULT_CONFIG}"
CONFIG_SPECIFIED=0
if [[ -n "${CONFIG:-}" ]]; then
    CONFIG_SPECIFIED=1
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
METADATA_PATH="${METADATA:-$DEFAULT_METADATA}"
GPU_LIST="${DEFAULT_GPU_LIST}"

ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPU_LIST="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            CONFIG_SPECIFIED=1
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --metadata)
            METADATA_PATH="$2"
            shift 2
            ;;
        --steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --cfg)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --shift)
            SHIFT_VALUE="$2"
            shift 2
            ;;
        --seed)
            SEED_VALUE="$2"
            shift 2
            ;;
        --compare-sd35)
            RUN_SD35_BASELINE_COMPARISON=1
            shift 1
            ;;
        --baseline-mode)
            BASELINE_MODE="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            ARGS+=("$@")
            break
            ;;
        *)
            if [[ ${CONFIG_SPECIFIED} -eq 0 && "$1" != -* && ( "$1" == *.yaml || "$1" == *.yml || "$1" == *.json ) ]]; then
                CONFIG_PATH="$1"
                CONFIG_SPECIFIED=1
                shift
            else
                ARGS+=("$1")
                shift
            fi
            ;;
    esac
done

IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NUM_SHARDS=${#GPU_ARRAY[@]}

if (( NUM_SHARDS == 0 )); then
    echo "[Multi] No GPUs specified. Use --gpus, set GPUS env var, or ensure nvidia-smi is available." >&2
    exit 1
fi

echo "[Multi] Launching ${NUM_SHARDS} shard(s) across GPU(s): ${GPU_LIST}"

pids=()

for idx in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$idx]}"
    echo "[Multi] Starting shard ${idx}/${NUM_SHARDS} on GPU ${gpu}"

    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        set -x
        "${PYTHON}" "${PROJECT_ROOT}/evaluation/batch_generate_from_metadata.py" \
            --config "${CONFIG_PATH}" \
            --output-root "${OUTPUT_ROOT}" \
            --metadata "${METADATA_PATH}" \
            --num-inference-steps "${NUM_STEPS}" \
            --guidance-scale "${GUIDANCE_SCALE}" \
            --shift "${SHIFT_VALUE}" \
            --seed "${SEED_VALUE}" \
            --num-shards "${NUM_SHARDS}" \
            --shard-index "${idx}" \
            "${ARGS[@]}"
    ) &

    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

if (( status == 0 )) && (( RUN_SD35_BASELINE_COMPARISON == 1 )); then
    echo "[Multi] Main run finished successfully. Launching SD3.5 baseline comparison runs (512 and 1024)."

    # Run baseline at 512x512
    echo "[Multi][Baseline] Starting 512x512 generation with mode '${BASELINE_MODE}'"
    pids=()
    for idx in "${!GPU_ARRAY[@]}"; do
        gpu="${GPU_ARRAY[$idx]}"
        echo "[Multi][Baseline] 512px shard ${idx}/${NUM_SHARDS} on GPU ${gpu}"
        (
            export CUDA_VISIBLE_DEVICES="${gpu}"
            set -x
            "${PYTHON}" "${PROJECT_ROOT}/evaluation/batch_generate_from_metadata.py" \
                --config "${CONFIG_PATH}" \
                --output-root "${OUTPUT_ROOT}/baseline_sd35_512" \
                --metadata "${METADATA_PATH}" \
                --num-inference-steps "${NUM_STEPS}" \
                --guidance-scale "${GUIDANCE_SCALE}" \
                --shift "${SHIFT_VALUE}" \
                --seed "${SEED_VALUE}" \
                --height 512 \
                --width 512 \
                --baseline-mode "${BASELINE_MODE}" \
                --num-shards "${NUM_SHARDS}" \
                --shard-index "${idx}" \
                "${ARGS[@]}"
        ) &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done

    # Run baseline at 1024x1024
    if (( status == 0 )); then
        echo "[Multi][Baseline] Starting 1024x1024 generation with mode '${BASELINE_MODE}'"
        pids=()
        for idx in "${!GPU_ARRAY[@]}"; do
            gpu="${GPU_ARRAY[$idx]}"
            echo "[Multi][Baseline] 1024px shard ${idx}/${NUM_SHARDS} on GPU ${gpu}"
            (
                export CUDA_VISIBLE_DEVICES="${gpu}"
                set -x
                "${PYTHON}" "${PROJECT_ROOT}/evaluation/batch_generate_from_metadata.py" \
                    --config "${CONFIG_PATH}" \
                    --output-root "${OUTPUT_ROOT}/baseline_sd35_1024" \
                    --metadata "${METADATA_PATH}" \
                    --num-inference-steps "${NUM_STEPS}" \
                    --guidance-scale "${GUIDANCE_SCALE}" \
                    --shift "${SHIFT_VALUE}" \
                    --seed "${SEED_VALUE}" \
                    --height 1024 \
                    --width 1024 \
                    --baseline-mode "${BASELINE_MODE}" \
                    --num-shards "${NUM_SHARDS}" \
                    --shard-index "${idx}" \
                    "${ARGS[@]}"
            ) &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                status=1
            fi
        done
    fi
fi

exit ${status}


