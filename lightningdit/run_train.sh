#!/bin/bash
CONFIG_PATH=$1

PRECISION=${PRECISION:-bf16}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=1235
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export WANDB_MODE=${WANDB_MODE:-online}
unset WANDB_DISABLED
export WANDB_SILENT=${WANDB_SILENT:-true}
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}

# Ensure project packages are importable for all ranks
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/davae:${PYTHONPATH}"

accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    train.py \
    --config $CONFIG_PATH
