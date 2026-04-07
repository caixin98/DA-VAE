# LightningDiT + DA-VAE (CVPR 2026)

This directory contains the code for reproducing the **ImageNet experiments** from the DA-VAE paper (Section 4.1). It implements DA-VAE (Detail-Aligned VAE) integrated with the LightningDiT training and sampling pipeline.

## Key Components

- **`train.py`**: Training entry point for LightningDiT with DA-VAE / VA-VAE tokenizer.
- **`inference.py`**: Class-conditional sampling with CFG, interval guidance, and timestep shift.
- **`evaluate_tokenizer.py`**: Tokenizer reconstruction evaluation (rFID / PSNR / LPIPS / SSIM).
- **`extract_features.py`**: Extract and save latent features from ImageNet as safetensors.

### Model & Tokenizer

- **`models/`**: LightningDiT transformer and components (SwiGLU, RoPE, RMSNorm, LPIPS).
- **`tokenizer/davae.py`**: DA-VAE inference wrapper — the core tokenizer for this paper.
- **`tokenizer/vavae.py`**: VA-VAE baseline wrapper for comparison.
- **`davae/`**: DA-VAE core implementation (LDM-based encoder/decoder with detail alignment).

### Infrastructure

- **`transport/`**: Rectified flow transport, ODE/SDE sampling integrators.
- **`local_datasets/`**: `ImgLatentDataset` for loading precomputed latents from safetensors.
- **`checkpoints/`**: Checkpoint management utilities.
- **`tools/`**: FID and Inception Score computation.

## Setup

```bash
conda create -n davae python=3.10
conda activate davae
pip install -r requirements.txt
```

## Training

### Step 1: Train DA-VAE Tokenizer (Optional — use pretrained weights if available)

```bash
cd davae
bash run_train.sh
```

### Step 2: Extract Latent Features

```bash
bash run_extraction.sh
```

### Step 3: Train LightningDiT with DA-VAE

**Final configuration (paper results, FID=1.68):**

```bash
bash run_train.sh configs/lightningdit_xl_davae_f32d128_detail_align_mean_loss_schedule.yaml
```

Key settings in this config:
- `detail_align_mean` mode: explicit base/detail separation with group-mean alignment (Eq. 4)
- `channel_loss_schedule`: cosine warmup over 20k steps for the extra 96 detail channels (Eq. 12–13)
- `use_qknorm: true`, `freeze_dit: false` (full fine-tuning)
- Sampling: `cfg_scale=4.0`, `cfg_interval_start=0.2`, `timestep_shift=0.3`

### Step 4: Generate and Evaluate

```bash
# Fast sampling
bash run_fast_inference.sh configs/lightningdit_xl_davae_f32d128_detail_align_mean_loss_schedule.yaml

# Full sampling + FID
bash run_inference.sh configs/lightningdit_xl_davae_f32d128_detail_align_mean_loss_schedule.yaml

# Tokenizer reconstruction metrics
python evaluate_tokenizer.py \
    --config_path tokenizer/configs/da_f16d32_piat_detail_align_mean.yaml \
    --model_type davae \
    --data_path /path/to/imagenet/val \
    --output_path /tmp/davae_results
```

## Configuration Files

| Config | Description |
|--------|-------------|
| `configs/lightningdit_xl_davae_f32d128_detail_align_mean_loss_schedule.yaml` | **Final config** — detail alignment + loss schedule (FID=1.68) |
| `configs/config_details.yaml` | Reference VA-VAE baseline settings |
| `tokenizer/configs/da_f16d32_piat_detail_align_mean.yaml` | DA-VAE tokenizer config |
| `tokenizer/configs/davae_f16d32.yaml` | DA-VAE base config |
| `tokenizer/configs/vavae_f16d32.yaml` | VA-VAE baseline tokenizer config |
| `davae/configs/da_f16d32_piat_diff_align_mean.yaml` | DA-VAE LDM training config |

## Pretrained Weights

Before training, you need the following pretrained weights (update paths in configs):

1. **VA-VAE checkpoint**: `vavae-imagenet256-f16d32-dinov2.pt` — base VAE encoder (frozen during DA-VAE training)
2. **LightningDiT-XL checkpoint**: `lightningdit-xl-imagenet256-800ep.pt` — pretrained DiT for warm-start fine-tuning

## License

See `LICENSE`.
