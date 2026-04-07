config_path=${1:-$CONFIG_PATH}

if [ -z "$config_path" ]; then
    echo "Usage:"
    echo "  bash run_train.sh path/to/config.yaml"
    echo "  # or"
    echo "  CONFIG_PATH=path/to/config.yaml bash run_train.sh"
    echo "Optional overrides: NPROC_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT"
    exit 1
fi
shift || true

# W&B is non-interactive under torchrun/srun; 默认在线上传（可通过 WANDB_MODE 覆盖）。
export WANDB_MODE=${WANDB_MODE:-online}
unset WANDB_DISABLED
# 默认不静默：方便你在 stdout 看到 "View run / Syncing run"；如需静默可显式 export WANDB_SILENT=true
export WANDB_SILENT=${WANDB_SILENT:-false}
# Avoid mixing ~/.local site-packages with the conda env (can cause numpy/torch type mismatches).
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}

nproc_per_node=${NPROC_PER_NODE:-8}
# nnodes=${NNODES:-$WORLD_SIZE}
nnodes=1
# node_rank=${NODE_RANK:-${RANK:-0}}
node_rank=0
# master_addr=${MASTER_ADDR:-127.0.0.1}
master_addr=127.0.0.1
# master_port=${MASTER_PORT:-29502}
master_port=29502

echo "Launching torchrun with nnodes=${nnodes}, nproc_per_node=${nproc_per_node}, node_rank=${node_rank}."

# 可选：边训练边同步 ckpt/images 到对象存储（S3/OSS 等），并在训练结束时清理掉后台同步进程。
# 需提前配置：CKPT_DIR、IMG_DIR（可选）；对象存储路径建议在 ~/.bashrc 中设置：
#   export RCLONE_SYNC_DEST=caixin:caixin_hdd/cache
# 本脚本默认同步到 ${RCLONE_SYNC_DEST}/lightningdit/davae，如需覆盖可显式传入 SYNC_DEST。
SYNC_DEST=${SYNC_DEST:-${RCLONE_SYNC_DEST:+${RCLONE_SYNC_DEST}/lightningdit/davae}}
SYNC_INTERVAL=${SYNC_INTERVAL:-300}
# 同步后保留的本地 ckpt 数量（按修改时间倒序），默认保留 2 个
CKPT_KEEP=${CKPT_KEEP:-2}
# 匹配 ckpt 文件的模式，默认 *.ckpt，可按需改成 *.pt 等
CKPT_PATTERN=${CKPT_PATTERN:-*.ckpt}

start_background_sync() {
    if [ -z "$SYNC_DEST" ] || { [ ! -d "${CKPT_DIR:-}" ] && [ ! -d "${IMG_DIR:-}" ]; }; then
        return
    fi

    echo "Starting rclone sync to ${SYNC_DEST} every ${SYNC_INTERVAL}s..."
    while true; do
        if [ -n "${CKPT_DIR:-}" ] && [ -d "$CKPT_DIR" ]; then
            rclone sync --progress --transfers 200 --checkers 200 "$CKPT_DIR" "${SYNC_DEST}/ckpt" --create-empty-src-dirs
            # 清理本地旧 ckpt，仅保留最新 CKPT_KEEP 个
            if [ "${CKPT_KEEP:-0}" -gt 0 ]; then
                find "$CKPT_DIR" -maxdepth 1 -type f -name "$CKPT_PATTERN" -printf '%T@ %p\n' \
                  | sort -nr \
                  | tail -n +$((CKPT_KEEP+1)) \
                  | cut -d' ' -f2- \
                  | xargs -r rm -f
            fi
        fi
        if [ -n "${IMG_DIR:-}" ] && [ -d "$IMG_DIR" ]; then
            rclone sync --progress --transfers 200 --checkers 200 "$IMG_DIR" "${SYNC_DEST}/images" --create-empty-src-dirs
        fi
        sleep "$SYNC_INTERVAL"
    done &
    SYNC_PID=$!
}

cleanup_sync() {
    if [ -n "${SYNC_PID:-}" ]; then
        kill "$SYNC_PID" >/dev/null 2>&1 || true
        wait "$SYNC_PID" 2>/dev/null || true
    fi
}

trap cleanup_sync EXIT INT TERM
start_background_sync

torchrun --nproc_per_node="${nproc_per_node}" \
    --nnodes="${nnodes}" \
    --node_rank="${node_rank}" \
    --master_addr="${master_addr}" \
    --master_port="${master_port}" \
    main.py \
    --base "$config_path" \
    --train \
    "$@"

