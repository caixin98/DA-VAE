"""
Evaluate tokenizer performance by computing reconstruction metrics.

Metrics include:
- rFID (Reconstruction FID)
- PSNR (Peak Signal-to-Noise Ratio)
- LPIPS (Learned Perceptual Image Patch Similarity)
- SSIM (Structural Similarity Index)

This script supports:
- torchvision ImageFolder input (and auto-wrapping flat JPEG folders into an `unknown/` class via symlinks)
- vavae 256x256 reconstruction + bilinear upsample to 512 before computing metrics

by Jingfeng Yao (original), adapted for SRTokenizer evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models.lpips import LPIPS
from tools.calculate_fid import InceptionV3, calculate_frechet_distance


def print_with_prefix(content: str, prefix: str = "Tokenizer Evaluation", rank: int = 0) -> None:
    if rank == 0:
        print(f"\033[34m[{prefix}]\033[0m {content}")


def _maybe_init_distributed(backend: Optional[str] = None) -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size()


def _ensure_imagefolder_root(data_path: str) -> str:
    """If `data_path` is a flat folder of images, create an ImageFolder-compatible view via symlinks."""
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"data_path not found: {data_path}")

    # If it already looks like ImageFolder (has at least one subdir), keep it.
    has_subdir = any(x.is_dir() for x in p.iterdir())
    if has_subdir:
        return str(p)

    # Otherwise, wrap it as: <data_path>_imagefolder/unknown/*.JPEG (symlinks)
    wrapped = p.parent / (p.name + "_imagefolder")
    cls_dir = wrapped / "unknown"
    cls_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".JPEG", ".JPG", ".PNG"}
    imgs = [x for x in p.iterdir() if x.is_file() and x.suffix in exts]
    if len(imgs) == 0:
        # Some datasets use uppercase suffix; just fall back to all files.
        imgs = [x for x in p.iterdir() if x.is_file()]

    # Idempotent: only create missing symlinks.
    for src in tqdm(imgs, desc="Preparing ImageFolder symlinks"):
        dst = cls_dir / src.name
        if dst.exists():
            continue
        try:
            dst.symlink_to(src)
        except FileExistsError:
            pass
        except OSError:
            # If filesystem disallows symlinks, fall back to hardlink/copy (best-effort).
            try:
                os.link(str(src), str(dst))
            except Exception:
                # Last resort: copy (slow, but works)
                import shutil

                shutil.copy2(str(src), str(dst))

    return str(wrapped)


@torch.no_grad()
def _inception_features(inception: InceptionV3, x_01: torch.Tensor) -> torch.Tensor:
    """Return (B, dims) float32 features for input in [0,1]."""
    pred = inception(x_01)[0]
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
    pred = pred.squeeze(3).squeeze(2)
    return pred


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).contiguous()
    return w


@torch.no_grad()
def ssim_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 2.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Compute SSIM for x,y in [-1,1]. Returns scalar mean over batch."""
    if x.shape != y.shape:
        raise ValueError(f"SSIM expects same shape, got {x.shape} vs {y.shape}")
    if x.dim() != 4:
        raise ValueError(f"SSIM expects BCHW tensors, got dim={x.dim()}")

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    b, c, h, w = x.shape
    window = _gaussian_window(window_size, sigma, x.device, x.dtype)
    window = window.view(1, 1, window_size, window_size).repeat(c, 1, 1, 1)

    # valid conv (no padding) like standard SSIM
    mu_x = F.conv2d(x, window, groups=c)
    mu_y = F.conv2d(y, window, groups=c)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, window, groups=c) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    # mean over channels and spatial dims, then batch
    return ssim_map.mean(dim=(1, 2, 3)).mean()


@torch.no_grad()
def evaluate_tokenizer(
    config_path: str,
    model_type: str,
    data_path: str,
    output_path: str,
    ref_size: int = 512,
    token_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    max_images: Optional[int] = None,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    rank, world_size = _maybe_init_distributed()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")

    # Load tokenizer (use ref_size for direct 512x512 processing)
    if rank == 0:
        print_with_prefix("Loading model...")
    if model_type == "vavae":
        from tokenizer.vavae import VA_VAE

        tokenizer = VA_VAE(config_path, img_size=ref_size, device=device)
        model = tokenizer.model
    elif model_type == "davae":
        from tokenizer.davae import DA_VAE

        tokenizer = DA_VAE(config_path, img_size=ref_size, device=device)
        model = tokenizer.model
    else:
        raise ValueError(f"Unsupported tokenizer '{model_type}'. Choose 'vavae' or 'davae'.")

    if model is None:
        raise RuntimeError("Tokenizer model is not loaded.")

    # Data (keep ImageFolder as requested)
    imagefolder_root = _ensure_imagefolder_root(data_path)
    transform = tokenizer.img_transform(p_hflip=0.0, img_size=ref_size)
    dataset = ImageFolder(root=imagefolder_root, transform=transform)
    if max_images is not None:
        dataset.samples = dataset.samples[: int(max_images)]
        dataset.targets = dataset.targets[: int(max_images)]

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    val_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # Output dirs (kept for compatibility / logging)
    folder_name = os.path.splitext(os.path.basename(config_path))[0]
    out_dir = Path(output_path) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        print_with_prefix(f"Data root (ImageFolder): {imagefolder_root}")
        print_with_prefix(f"Output dir: {str(out_dir)}")

    # Metrics
    lpips = LPIPS().to(device).eval()

    # FID (online stats, no image dumping)
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    try:
        inception = InceptionV3([block_idx]).to(device).eval()
    except Exception as e:
        # Fallback to torchvision inception (may still require weights, but better than failing hard)
        print_with_prefix(f"Warning: failed to init FID inception weights: {e}", rank=rank)
        inception = InceptionV3([block_idx], use_fid_inception=False).to(device).eval()

    # Running sums for metrics (use Python floats to avoid tensor accumulation issues)
    n_seen = 0
    lpips_sum = 0.0
    ssim_sum = 0.0
    psnr_sum = 0.0

    # Running sums for FID moments
    feat_sum_ref = torch.zeros((dims,), dtype=torch.float64, device=device)
    feat_sum_rec = torch.zeros((dims,), dtype=torch.float64, device=device)
    feat_xxt_ref = torch.zeros((dims, dims), dtype=torch.float64, device=device)
    feat_xxt_rec = torch.zeros((dims, dims), dtype=torch.float64, device=device)

    # Save sample images (first batch only, rank 0)
    samples_dir = out_dir / "samples" if rank == 0 else None
    if samples_dir is not None:
        samples_dir.mkdir(exist_ok=True)
    saved_samples = 0
    max_samples = 10  # Save first 10 images

    if rank == 0:
        print_with_prefix("Generating reconstructions + computing metrics...")

    for batch_idx, batch in enumerate(tqdm(val_dataloader, disable=(rank != 0))):
        ref512 = batch[0].to(device, non_blocking=True)  # [-1,1], (B,3,512,512)

        # Direct 512x512 encoding and decoding (no down/up sampling)
        latents = tokenizer.encode_images(ref512)
        model_dtype = next(model.parameters()).dtype
        latents = latents.to(dtype=model_dtype)
        rec512 = model.decode(latents)

        # Metrics expect consistent dtypes. vavae typically decodes in fp16 while ref is fp32.
        # Cast to float32 for metric computation to avoid dtype mismatch in conv2d/LPIPS/inception.
        ref512_f = ref512.float()
        rec512_f = rec512.float()

        b = ref512.shape[0]
        n_seen += b

        # LPIPS / SSIM on [-1,1]
        # These return batch averages, so we need to multiply by batch size to get sum
        lpips_val = lpips(rec512_f, ref512_f).mean()
        ssim_val = ssim_torch(rec512_f, ref512_f, data_range=2.0)
        lpips_sum += float(lpips_val.item()) * b  # Multiply by batch size
        ssim_sum += float(ssim_val.item()) * b  # Multiply by batch size

        # PSNR on [0,255]
        ref_255 = torch.clamp((ref512_f + 1.0) * 0.5 * 255.0, 0.0, 255.0)
        rec_255 = torch.clamp((rec512_f + 1.0) * 0.5 * 255.0, 0.0, 255.0)
        mse = torch.mean((ref_255 - rec_255) ** 2, dim=(1, 2, 3))
        psnr = 20.0 * torch.log10(torch.tensor(255.0, device=device)) - 10.0 * torch.log10(mse + 1e-8)
        psnr_val = psnr.mean()
        psnr_sum += float(psnr_val.item()) * b  # Multiply by batch size

        # FID features on [0,1]
        ref01 = torch.clamp((ref512_f + 1.0) * 0.5, 0.0, 1.0)
        rec01 = torch.clamp((rec512_f + 1.0) * 0.5, 0.0, 1.0)
        f_ref = _inception_features(inception, ref01).to(torch.float64)
        f_rec = _inception_features(inception, rec01).to(torch.float64)
        feat_sum_ref += f_ref.sum(dim=0)
        feat_sum_rec += f_rec.sum(dim=0)
        feat_xxt_ref += f_ref.t().mm(f_ref)
        feat_xxt_rec += f_rec.t().mm(f_rec)

        # Save sample images (first batch, rank 0 only)
        if samples_dir is not None and saved_samples < max_samples:
            for i in range(min(b, max_samples - saved_samples)):
                idx = saved_samples + i
                # Save reference (512x512)
                ref_img = (ref512_f[i].cpu().clamp(-1, 1) + 1.0) * 0.5
                ref_img = (ref_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(ref_img).save(samples_dir / f"{idx:04d}_ref_512.png")
                # Save direct 512x512 reconstruction (no upsampling)
                rec512_img = (rec512_f[i].cpu().clamp(-1, 1) + 1.0) * 0.5
                rec512_img = (rec512_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(rec512_img).save(samples_dir / f"{idx:04d}_rec_512.png")
            saved_samples += min(b, max_samples - saved_samples)

    # Reduce across ranks (sums + counts)
    count_t = torch.tensor([n_seen], device=device, dtype=torch.long)
    lpips_t = torch.tensor(lpips_sum, device=device, dtype=torch.float64)
    ssim_t = torch.tensor(ssim_sum, device=device, dtype=torch.float64)
    psnr_t = torch.tensor(psnr_sum, device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(lpips_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(ssim_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(psnr_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(feat_sum_ref, op=dist.ReduceOp.SUM)
        dist.all_reduce(feat_sum_rec, op=dist.ReduceOp.SUM)
        dist.all_reduce(feat_xxt_ref, op=dist.ReduceOp.SUM)
        dist.all_reduce(feat_xxt_rec, op=dist.ReduceOp.SUM)

    total_n = int(count_t.item())
    avg_lpips = (lpips_t / max(total_n, 1)).item()
    avg_ssim = (ssim_t / max(total_n, 1)).item()
    avg_psnr = (psnr_t / max(total_n, 1)).item()

    # FID stats
    mu_ref = (feat_sum_ref / max(total_n, 1)).cpu().numpy()
    mu_rec = (feat_sum_rec / max(total_n, 1)).cpu().numpy()
    exxt_ref = (feat_xxt_ref / max(total_n, 1)).cpu().numpy()
    exxt_rec = (feat_xxt_rec / max(total_n, 1)).cpu().numpy()
    sigma_ref = exxt_ref - np.outer(mu_ref, mu_ref)
    sigma_rec = exxt_rec - np.outer(mu_rec, mu_rec)
    fid = float(calculate_frechet_distance(mu_ref, sigma_ref, mu_rec, sigma_rec))

    if rank == 0:
        print_with_prefix("Final Metrics:")
        print_with_prefix(f"rFID: {fid:.3f}")
        print_with_prefix(f"PSNR: {avg_psnr:.3f}")
        print_with_prefix(f"LPIPS: {avg_lpips:.3f}")
        print_with_prefix(f"SSIM: {avg_ssim:.3f}")

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="tokenizer/configs/vavae_f16d32.yaml")
    parser.add_argument("--model_type", type=str, default="vavae", choices=["davae", "vavae"])
    parser.add_argument("--data_path", type=str, default="/path/to/your/imagenet/ILSVRC2012_validation")
    parser.add_argument("--output_path", type=str, default="./tokenizer_eval_outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ref_size", type=int, default=512)
    parser.add_argument("--token_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    evaluate_tokenizer(
        config_path=args.config_path,
        model_type=args.model_type,
        data_path=args.data_path,
        output_path=args.output_path,
        ref_size=args.ref_size,
        token_size=args.token_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images=args.max_images,
        seed=args.seed,
    )