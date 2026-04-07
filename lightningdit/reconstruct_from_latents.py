import os
import argparse
from datetime import datetime
from typing import List

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from safetensors.torch import load_file


def _detect_vae_variant(config_path: str) -> str:
    try:
        cfg = OmegaConf.load(config_path)
        target = None
        if "model" in cfg and cfg.model is not None:
            target = cfg.model.get("target", None)
        if isinstance(target, str):
            if "da_autoencoder.DAVAE" in target:
                return "davae"
            if "autoencoder.AutoencoderKL" in target:
                return "vavae"
        params = cfg.model.get("params", {}) if "model" in cfg else {}
        if isinstance(params, dict) and ("ddconfig_da" in params or "da_ckpt_path" in params):
            return "davae"
    except Exception:
        pass
    return "vavae"


def _save_batch_images(images_uint8: np.ndarray, out_dir: str, prefix: str, start_idx: int) -> int:
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for i, img in enumerate(images_uint8):
        fname = os.path.join(out_dir, f"{prefix}_{start_idx + i:09d}.png")
        Image.fromarray(img).save(fname)
        count += 1
    return count


def list_safetensors(latents_dir: str) -> List[str]:
    all_files = []
    for root, _, files in os.walk(latents_dir):
        for f in files:
            if f.endswith(".safetensors"):
                all_files.append(os.path.join(root, f))
    return sorted(all_files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--latents_dir", type=str, required=True, help="Directory containing *.safetensors from extract_features")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--variant", type=str, default="auto", choices=["auto", "vavae", "davae"])
    parser.add_argument("--which", type=str, default="latents", choices=["latents", "latents_flip", "both"], help="Which tensor(s) to decode")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Reconstruction requires at least one CUDA device."
    device = torch.device(args.device)

    variant = args.variant
    if variant == "auto":
        variant = _detect_vae_variant(args.config)

    if variant == "davae":
        from tokenizer.davae import DA_VAE as VAE
    elif variant == "vavae":
        from tokenizer.vavae import VA_VAE as VAE
    else:
        raise ValueError(f"Unknown variant: {variant}")

    vae = VAE(args.config)

    # Default output directory
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "recon_from_latents", ts)
    os.makedirs(args.output_dir, exist_ok=True)

    files = list_safetensors(args.latents_dir)
    if len(files) == 0:
        raise FileNotFoundError(f"No .safetensors files found under {args.latents_dir}")

    print(f"Found {len(files)} shards under {args.latents_dir}. Writing to {args.output_dir}")

    decode_keys = ["latents", "latents_flip"] if args.which == "both" else [args.which]

    total_decoded = 0
    with torch.no_grad():
        for shard_idx, shard_path in enumerate(files):
            data = load_file(shard_path)
            for key in decode_keys:
                if key not in data:
                    print(f"[Warn] Key '{key}' not in {shard_path}, skipping")
                    continue
                z = data[key]  # (N, C, H, W) on CPU
                # Process in batches
                n = z.shape[0]
                start = 0
                out_subdir = os.path.join(args.output_dir, key)
                prefix = f"recon_s{shard_idx:03d}_{key}"
                while start < n:
                    end = min(start + args.batch_size, n)
                    z_batch = z[start:end].to(device)
                    # Use wrapper's decode_to_images which returns uint8 numpy
                    imgs_uint8 = vae.decode_to_images(z_batch)
                    wrote = _save_batch_images(imgs_uint8, out_subdir, prefix, start)
                    total_decoded += wrote
                    start = end
            print(f"Processed shard {shard_idx+1}/{len(files)}: {os.path.basename(shard_path)}")

    print(f"Done. Total decoded images: {total_decoded}")


if __name__ == "__main__":
    main()


