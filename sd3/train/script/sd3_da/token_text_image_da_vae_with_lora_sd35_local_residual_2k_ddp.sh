#!/usr/bin/env bash
set -euo pipefail

# 让脚本在任意工作目录下运行都能找到开源代码
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SD3_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${SD3_ROOT}:${PYTHONPATH:-}"

# ====== 你的原始配置（支持从第一个参数传入）======
if [ $# -ge 1 ]; then
  CONFIG_PATH="$1"
else
  CONFIG_PATH="${CONFIG_PATH:-train/config/sd3_da/token_text_image_da_vae_with_lora_diff_sd35_local_residual_new_2k.yaml}"
fi
# 若传入的是相对路径，基于 SD3_ROOT 解析为绝对路径
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${SD3_ROOT}/${CONFIG_PATH}"
fi
export OMINI_CONFIG="${OMINI_CONFIG:-$CONFIG_PATH}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PRECISION="${PRECISION:-bf16}"

if [ ! -f "$OMINI_CONFIG" ]; then
  echo "错误: 配置文件 $OMINI_CONFIG 不存在" >&2
  exit 1
fi

# ====== 分布式参数（两边相同）======
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
# 兼容你的参考脚本：从 WORLD_SIZE 读取“节点数”，默认为 1
export NNODES="${NNODES:-${WORLD_SIZE:-1}}"
export MASTER_PORT="${MASTER_PORT:-1235}"
export MASTER_ADDR="${MASTER_ADDR:-}"

# ====== 网络/NCCL 建议（跨 Pod 更稳）======
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DISTRIBUTED_DEFAULT_BACKEND="${TORCH_DISTRIBUTED_DEFAULT_BACKEND:-nccl}"

# ====== 机器序号（两种方式择一）======
# A) 明确传入（推荐）：NODE_RANK / RANK / MACHINE_RANK
# B) 从 HOSTNAME 自动推断（支持尾部随机后缀）
NODE_RANK="${NODE_RANK:-${RANK:-${MACHINE_RANK:-}}}"
if [ -z "${NODE_RANK}" ]; then
  _hn="${HOSTNAME:-}"
  # 如果主机名以 -<字母数字串> 结尾（如 -2u9gkx），先去掉该随机后缀
  if [[ "${_hn}" =~ ^(.+)-[[:alpha:]][[:alnum:]]*$ ]]; then
    _hn="${BASH_REMATCH[1]}"
  fi
  # 再从尾部的 -<数字> 中提取 NODE_RANK
  if [[ "${_hn}" =~ -([0-9]+)$ ]]; then
    NODE_RANK="${BASH_REMATCH[1]}"
  fi
fi
: "${NODE_RANK:?请设置 NODE_RANK/RANK=0..N-1，或使用 StatefulSet 的序号命名}"
MACHINE_RANK="${NODE_RANK}"

# ====== MASTER_ADDR 设置（关键！）======
if [ -z "${MASTER_ADDR}" ]; then
  if [ "${MACHINE_RANK}" = "0" ]; then
    # 主节点默认取本机内网 IP
    MASTER_ADDR="$(hostname -i | awk '{print $1}')"
  else
    # 从节点：从 HOSTNAME 推断主节点名为尾号置 0；否则退化为 127.0.0.1（需显式传入更可靠）
    host_base="${HOSTNAME:-}"
    if [[ "$host_base" =~ ^(.+)-[[:alpha:]][[:alnum:]]*$ ]]; then
      host_base="${BASH_REMATCH[1]}"
    fi
    if [[ "$host_base" =~ ^(.*)-([0-9]+)$ ]]; then
      MASTER_ADDR="${BASH_REMATCH[1]}-0"
    else
      MASTER_ADDR="127.0.0.1"
    fi
  fi
fi
# MASTER_ADDR="pluto-prod-xcai-temp-2025-10-30-15-8-0-2u9gkx"
export MASTER_ADDR
echo "[rank=$MACHINE_RANK] MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE"

# ====== 启动（Accelerate 多机多卡）======
exec accelerate launch \
  --multi_gpu \
  --num_processes "$(($GPUS_PER_NODE*$NNODES))" \
  --num_machines "${NNODES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --mixed_precision "${PRECISION}" \
  -m omini.train_sd3_davae.train_sd3_tokenizer "$@"
