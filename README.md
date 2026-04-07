# DA-VAE: Detail-Aligned VAE for Efficient High-Resolution Diffusion (CVPR 2026)

This repository contains the official implementation of **Detail-Aligned VAE (DA-VAE)**, a method that increases the compression ratio of a pretrained VAE while requiring only lightweight adaptation for the pretrained diffusion backbone.

## Overview

DA-VAE reduces the token count for latent diffusion models by increasing the spatial compression rate while preserving the structure of the original latent space. Key ideas:

- **Structured Latent Space**: The first C channels retain the pretrained VAE latent at base resolution; an additional D channels encode high-resolution details.
- **Detail Alignment Loss**: A channel-wise grouped reduction that aligns detail channels with the base latent structure (Eq. 3–4 in the paper).
- **Zero-Init Warm Start**: New patch embedder and output layer are zero-initialized to preserve pretrained DiT behavior at the start of fine-tuning.
- **Gradual Loss Scheduling**: Cosine-annealed weighting that gradually introduces detail channel losses during diffusion fine-tuning.

## Results

| Setting | Base Resolution | Target Resolution | Tokens | Speedup |
|---------|----------------|-------------------|--------|---------|
| ImageNet (LightningDiT-XL) | 256×256 | 512×512 | 16×16 | — |
| SD3.5 Medium | 512×512 | 1024×1024 | 32×32 | ~4× |
| SD3.5 Medium | 1024×1024 | 2048×2048 | 32×32 | ~6× |

## Repository Structure

```
open_source/
├── lightningdit/          # ImageNet experiments (Section 4.1)
│   ├── train.py           # DiT training with DA-VAE
│   ├── inference.py       # Sampling / generation
│   ├── evaluate_tokenizer.py  # rFID / LPIPS / SSIM evaluation
│   ├── models/            # LightningDiT transformer
│   ├── tokenizer/         # DA-VAE and VA-VAE wrappers
│   ├── davae/             # DA-VAE core LDM implementation
│   ├── transport/         # Rectified flow sampling
│   └── configs/           # Training configurations
│
└── sd3/                   # SD3.5 text-to-image experiments (Section 4.2)
    ├── modeling/           # SD3 DA-VAE model modules
    ├── davae/              # DA-VAE training for SD3
    ├── omini/
    │   ├── train_sd3_davae/  # DA-VAE tokenizer training
    │   ├── train_sd3_hr/     # SD3.5M LoRA fine-tuning
    │   └── pipeline/         # Inference pipelines
    ├── train/              # Training scripts and configs
    ├── eval/               # Evaluation scripts
    └── data/               # Data loaders
```

## Quick Start

### ImageNet Experiments

```bash
cd open_source/lightningdit
pip install -r requirements.txt

# Train DA-VAE + LightningDiT (final paper config)
bash run_train.sh configs/lightningdit_xl_davae_f32d128_detail_align_mean_loss_schedule.yaml

# Inference
bash run_inference.sh configs/lightningdit_xl_davae_f32d128_detail_align_mean_loss_schedule.yaml

# Tokenizer evaluation (rFID, LPIPS, SSIM)
python evaluate_tokenizer.py \
    --config_path tokenizer/configs/da_f16d32_piat_detail_align_mean.yaml \
    --model_type davae \
    --data_path /path/to/imagenet/val \
    --output_path /tmp/davae_results
```

### SD3.5 Text-to-Image Experiments

```bash
cd open_source/sd3
pip install -r requirements.txt

# Train DA-VAE tokenizer for SD3
OMINI_CONFIG=train/config/sd3_da/token_text_image_da_vae_with_lora_diff_sd35_local_residual_new_2k.yaml \
  accelerate launch -m omini.train_sd3_davae.train_sd3_tokenizer

# Fine-tune SD3.5M with LoRA
bash train/script/sd3_da/token_text_image_da_vae_with_lora_sd35_local_residual_2k_ddp.sh
```

See individual subdirectory READMEs for detailed instructions.

## Citation

```bibtex
@inproceedings{davae2026,
  title={Detail-Aligned VAE for Efficient High-Resolution Diffusion},
  booktitle={CVPR},
  year={2026}
}
```

## License

See `lightningdit/LICENSE` for details.
