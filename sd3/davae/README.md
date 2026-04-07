# DA-VAE Tokenizer Training for SD3

This directory contains the training code for the DA-VAE (Detail-Aligned VAE) tokenizer built on top of the SD3 VAE.

## Directory Structure

```
davae/
├── scripts/
│   └── train_davae.py               # Main training script
├── configs/
│   ├── training/SD3DAVAE/base.yaml   # Training configuration
│   └── test/sd3davae_diff.yaml       # Test configuration
├── freeze_vae_alignment_mean.sh      # Launch script
└── requirements.txt
```

## Usage

### 1. Configure Training

Edit `configs/training/SD3DAVAE/base.yaml`:

```yaml
model:
    sd3_model_path: "stabilityai/stable-diffusion-3-medium-diffusers"

local_train:
    image_dir: "/path/to/your/training/images"

local_eval:
    image_dir: "/path/to/your/validation/images"
```

### 2. Run Training

```bash
accelerate launch --num_processes=8 \
    scripts/train_davae.py \
    --config_path configs/training/SD3DAVAE/base.yaml
```

Or use the launch script:

```bash
bash freeze_vae_alignment_mean.sh
```

## Key Configuration

| Parameter | Description |
|-----------|-------------|
| `model.tokenizer_type` | Set to `"sd3_da_vae"` |
| `model.sd3_model_path` | Path to SD3 diffusers model |
| `losses.perceptual_weight` | Weight for LPIPS perceptual loss |
| `losses.reconstruction_weight` | Weight for L1 reconstruction loss |
| `training.per_gpu_batch_size` | Batch size per GPU |
| `training.mixed_precision` | `"bf16"`, `"fp16"`, or `"no"` |

## Model Architecture

The DA-VAE adds detail channels to the pretrained SD3 VAE:
1. **Base encoder** (frozen): Encodes base-resolution images to C-channel latents
2. **Detail encoder** (trainable): Encodes high-resolution details to D additional channels
3. **Joint decoder** (trainable): Decodes concatenated [C+D] latents to high-resolution images

The **alignment loss** (Eq. 3-4 in the paper) ensures detail channels share structure with base channels.
