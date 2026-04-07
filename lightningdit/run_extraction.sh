#!/bin/bash
CONFIG_PATH=$1

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=1
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-1235}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
PRECISION=${PRECISION:-bf16}
IMAGE_SIZE=${IMAGE_SIZE:-512}
DATA_SPLIT=${DATA_SPLIT:-imagenet_train}
OUTPUT_PATH=${OUTPUT_PATH:-/path/to/imagenet/latents}

CONFIG_BASENAME_WITHOUT_EXT=$(basename "${CONFIG_PATH%.*}")
SUBDIR="$CONFIG_BASENAME_WITHOUT_EXT/${DATA_SPLIT}_${IMAGE_SIZE}"

accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    extract_features.py \
    --config $CONFIG_PATH \
    --output_path $OUTPUT_PATH \
    --data_split $DATA_SPLIT \
    --image_size $IMAGE_SIZE \
    --batch_size 40 \
    --num_workers 16
