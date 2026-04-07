# SD3 DA-VAE Training Code

This directory contains the code for training SD3 Deep-Compressed VAE (DA-VAE) models with frozen VAE encoder and alignment loss.

## Directory Structure

```
davae/
├── scripts/
│   └── train_dititok.py          # Main training script
├── configs/
│   └── training/
│       └── SD3DAVAE/
│           └── base.yaml         # Training configuration
├── modeling/
│   ├── modules/
│   │   ├── sd3_da_vae.py         # SD3 DA-VAE model implementation
│   │   ├── losses.py             # Loss functions including encoder alignment loss
│   │   └── ...                   # Other supporting modules
│   ├── flux/
│   │   └── sd3_vae.py            # SD3 VAE wrapper
│   └── sd3_da_vae_wrapper.py     # Model wrapper for inference
├── utils/                         # Utility functions
├── data/                          # Data loading utilities
└── freeze_vae_alignment_mean.sh   # Training script

```

## Features

- **Frozen VAE Encoder**: Option to freeze the VAE encoder during training
- **Encoder Alignment Loss**: Alignment loss between encoder hidden states and teacher latents
- **Deep Compression**: Configurable compression factor for latent space
- **Multiple Alignment Methods**: Support for 'mean' and 'proj' alignment methods

## Requirements

- Python 3.8+
- PyTorch
- Accelerate
- Diffusers
- OmegaConf
- Other dependencies (see requirements.txt in parent directory)

## Usage

### 1. Update Paths

Edit `freeze_vae_alignment_mean.sh` and update:
- `PYTHONPATH`: Set to the directory containing this code
- `TRAIN_DATA_DIR`: Path to your training images
- `VAL_DATA_DIR`: Path to your validation images

### 2. Configure Training

Edit `configs/training/SD3DAVAE/base.yaml` to configure:
- Model settings (SD3 model path, compression factor, etc.)
- Training hyperparameters (learning rate, batch size, etc.)
- Loss weights (reconstruction, perceptual, alignment, etc.)

### 3. Run Training

```bash
bash freeze_vae_alignment_mean.sh
```

Or run directly with accelerate:

```bash
accelerate launch --num_machines=1 --num_processes=8 \
    scripts/train_dititok.py \
    config=configs/training/SD3DAVAE/base.yaml \
    model.freeze_vae_encoder=True \
    model.use_sd3_vae=True \
    losses.encoder_alignment_loss.weight=1.0 \
    losses.encoder_alignment_loss.align_method=mean
```

## Key Configuration Parameters

### Model Configuration
- `model.tokenizer_type`: Set to `"sd3_da_vae"`
- `model.sd3_model_path`: Path to SD3 diffusers model
- `model.freeze_vae_encoder`: Freeze VAE encoder (True/False)
- `model.use_sd3_vae`: Use SD3 VAE for teacher latents (True/False)
- `model.da_factor`: Deep compression factor (default: 2)
- `model.da_mode`: Compression mode - "full" or "diff"

### Loss Configuration
- `losses.encoder_alignment_loss.weight`: Weight for alignment loss
- `losses.encoder_alignment_loss.align_method`: Alignment method - "mean" or "proj"
- `losses.encoder_alignment_loss.encoder_alignment_mse_weight`: MSE weight in alignment loss
- `losses.reconstruction_weight`: Weight for reconstruction loss
- `losses.perceptual_weight`: Weight for perceptual loss

### Training Configuration
- `training.per_gpu_batch_size`: Batch size per GPU
- `training.gradient_accumulation_steps`: Gradient accumulation steps
- `training.mixed_precision`: Mixed precision mode ("bf16", "fp16", or "no")
- `optimizer.params.learning_rate`: Learning rate

## Model Architecture

The SD3 DA-VAE model consists of:
1. **SD3 VAE Encoder**: Encodes images to latent space
2. **DC Down Block**: Compresses encoder features spatially
3. **DC Up Block**: Expands compressed latents back to original size
4. **SD3 VAE Decoder**: Decodes latents back to pixel space

When `freeze_vae_encoder=True`, only the DC blocks and decoder are trained.

## Alignment Loss

The encoder alignment loss aligns the compressed encoder hidden states with teacher latents from the original SD3 VAE:

- **mean method**: Projects channels by grouped mean pooling
- **proj method**: Uses a learnable 1x1 convolution projection

This helps the compressed representation maintain semantic alignment with the original VAE latents.

## Notes

- Make sure your input images are in [0, 1] range
- The model supports both local image directories and S3 data sources
- Checkpoint saving and resuming are handled automatically
- Training logs are saved to the output directory

## License

See LICENSE file in parent directory.

