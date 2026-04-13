#!/bin/bash

# 设置环境变量
# 请根据你的实际路径修改
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SD3_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH=$PYTHONPATH:$SD3_ROOT

# 训练数据目录 - 请根据你的实际路径修改
TRAIN_DATA_DIR="/path/to/your/train/data"
VAL_DATA_DIR="/path/to/your/val/data"

# 启动训练
accelerate launch --num_machines=1 --num_processes=8 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_davae.py config=configs/training/SD3DAVAE/base.yaml \
    experiment.project="sd3_da_vae_run1_freeze_vae_alignment_mean" \
    experiment.name="sd3_da_vae_run1_freeze_vae_alignment_mean" \
    experiment.output_dir="sd3_da_vae_run1_freeze_vae_alignment_mean" \
    local_train.enabled=true \
    local_train.image_dir="$TRAIN_DATA_DIR" \
    local_train.max_samples=100000 \
    local_train.batch_size=2 \
    local_train.num_workers=16 \
    local_train.load_mode="simple" \
    base_size=1024 \
    local_eval.enabled=true \
    local_eval.image_dir="$VAL_DATA_DIR" \
    local_eval.max_samples=250 \
    experiment.max_train_examples=100000 \
    training.per_gpu_batch_size=8 \
    experiment.skip_cond_image=true \
    experiment.generate_every=2000 \
    model.freeze_vae_encoder=True \
    optimizer.params.learning_rate=1e-4 \
    lr_scheduler.params.warmup_steps=0 \
    model.use_sd3_vae=True \
    losses.encoder_alignment_loss.weight=1.0 \
    losses.encoder_alignment_loss.align_method=mean \
    losses.encoder_alignment_loss.encoder_alignment_mse_weight=0.2 \
    # experiment.init_weight=sd3_da_vae_run1_freeze_vae/checkpoint-12000/unwrapped_model/pytorch_model.bin \
