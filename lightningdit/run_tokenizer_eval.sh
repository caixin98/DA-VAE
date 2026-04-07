CONFIG_PATH=$1
MODEL_TYPE=${2:-'davae'}

PRECISION=${PRECISION:-bf16}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-1235}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

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
    evaluate_tokenizer.py \
    --config_path $CONFIG_PATH \
    --model_type $MODEL_TYPE \
    ${DATA_PATH:+--data_path $DATA_PATH} \
    ${OUTPUT_PATH:+--output_path $OUTPUT_PATH} \
    ${BATCH_SIZE:+--batch_size $BATCH_SIZE} \
    ${NUM_WORKERS:+--num_workers $NUM_WORKERS} \
    ${MAX_IMAGES:+--max_images $MAX_IMAGES}