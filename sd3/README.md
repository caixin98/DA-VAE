# SD3.5 + DA-VAE (CVPR 2026)

This directory contains the code for reproducing the **text-to-image experiments** from the DA-VAE paper (Section 4.2). It implements DA-VAE integrated with Stable Diffusion 3.5 Medium for high-resolution image generation.

## Overview

The pipeline consists of two stages:
1. **DA-VAE Training**: Train the detail encoder on top of the SD3 VAE (f8→f16, adding D=16 detail channels to C=16 base channels).
2. **SD3.5M LoRA Fine-tuning**: Fine-tune SD3.5 Medium with LoRA (rank=256) on the new structured latent space.

## Directory Structure

```
sd3/
├── modeling/              # SD3 DA-VAE model and modules
├── utils/                 # Training utilities (logger, LR schedulers, etc.)
├── data/                  # Data loaders (PIAT, local, bucket)
├── davae/                 # DA-VAE training for SD3
│   ├── scripts/           # Training entry (train_davae.py)
│   └── configs/           # DA-VAE training configs
├── omini/
│   ├── config.py          # Shared config utilities
│   ├── train_sd3_davae/   # DA-VAE tokenizer training
│   ├── train_sd3_hr/      # SD3.5M LoRA fine-tuning
│   │   ├── trainer.py     # Core training loop with zero-init & loss schedule
│   │   └── train_sd3_hr.py  # Entry point
│   ├── train_sd3_text_token_image/  # Data loading utilities
│   └── pipeline/          # SD3 inference pipeline with DA-VAE
├── train/                 # Training scripts and configs
│   ├── config/            # YAML configurations
│   └── script/            # Launch scripts
├── inference/             # Inference configs
├── checkpoints/           # Checkpoint configs
├── eval/                  # Evaluation scripts (FID, GenEval)
└── tools/                 # CLI inference tool
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

### Stage 1: Train DA-VAE Tokenizer

Train the detail encoder/decoder on top of the pretrained SD3 VAE:

```bash
accelerate launch davae/scripts/train_davae.py \
  --config_path davae/configs/training/SD3DAVAE/base.yaml
```

Edit dataset paths in `davae/configs/training/SD3DAVAE/base.yaml`:

```yaml
local_train:
    image_dir: "/path/to/your/training/images"   # e.g. SAM dataset
local_eval:
    image_dir: "/path/to/your/validation/images"
```

### Stage 2: Fine-tune SD3.5M with LoRA

After DA-VAE is trained, fine-tune the SD3.5M diffusion backbone on the new structured latent space:

```bash
OMINI_CONFIG=train/config/sd3_da/token_text_image_da_vae_with_lora_diff_sd35_local_residual_new_2k.yaml \
  bash train/script/sd3_da/token_text_image_da_vae_with_lora_sd35_local_residual_2k_ddp.sh
```

The fine-tuning uses:
- **Zero-initialized patch embedder** for detail channels (paper Section 3.2)
- **Cosine loss scheduling** for gradual detail channel adaptation (Eq. 12–13)
- **LoRA** (rank=256) on attention and FFN layers

### Configuration

Edit paths in `train/config/sd3_da/token_text_image_da_vae_with_lora_diff_sd35_local_residual_new_2k.yaml`:

```yaml
tokenizer_vae:
  config_path: "checkpoints/sd3_davae/config.yaml"        # DA-VAE config from Stage 1
  checkpoint_path: "checkpoints/sd3_davae/pytorch_model.bin"  # DA-VAE weights from Stage 1

data:
  local_train:
    image_dir: "/path/to/your/dataset"    # synthetic data generated from DiffusionDB prompts
  local_eval:
    image_dir: "/path/to/your/dataset"
```

## Evaluation

```bash
# FID evaluation on MJHQ-30K
python eval/eval.py --input_dir /path/to/generated --ref_dir /path/to/mjhq30k

# GenEval
python eval/generate_sd35m_geneval.py
```

## Key Implementation Files

| File | Paper Section |
|------|--------------|
| `omini/pipeline/sd3_transformer_wrapper_tokenizer.py` | Zero-init patch embedder (Section 3.2, Fig. 1 right) |
| `omini/train_sd3_hr/trainer.py` | LoRA fine-tuning with loss scheduling |
| `modeling/modules/sd3_da_vae.py` | DA-VAE encoder/decoder for SD3 |
| `modeling/sd3_da_vae_wrapper.py` | DA-VAE wrapper for inference |

## Requirements

- Stable Diffusion 3.5 Medium weights from [Stability AI](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
- DA-VAE checkpoint (trained in Stage 1)
- Training data (SAM dataset for DA-VAE, synthetic data from DiffusionDB prompts for LoRA fine-tuning)
