#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate images for GenEval prompts using SD3.5M (Stable Diffusion 3.5 Medium).

Output format matches GenEval README:
<outdir>/
  00000/
    metadata.jsonl   (actually a single JSON object; GenEval evaluation reads json.load)
    samples/
      0000.png
      0001.png
      0002.png
      0003.png

Notes:
- This script is meant to run in the existing `davae` environment where SD3.5 pipeline deps live.
- New/modified comments are English per repo style.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

import torch
from tqdm import tqdm

try:
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
        StableDiffusion3Pipeline,
    )
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency: diffusers with StableDiffusion3Pipeline. "
        "Run this in the repo's inference env (e.g. `davae`)."
    ) from e


def _parse_dtype(s: str) -> torch.dtype:
    s = (s or "").lower().strip()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s} (choose from bfloat16/float16/float32)")


def _ensure_even(x: int) -> int:
    return int(x) // 2 * 2


def _maybe_apply_scheduler_knobs(pipe: StableDiffusion3Pipeline, shift: Optional[float], use_dynamic_shifting: bool) -> None:
    # Best-effort: different diffusers versions may or may not expose these APIs.
    try:
        pipe.scheduler.config.use_dynamic_shifting = bool(use_dynamic_shifting)
    except Exception:
        pass
    if shift is not None:
        try:
            pipe.scheduler.set_shift(float(shift))
        except Exception:
            pass


def _maybe_apply_memory_knobs(
    pipe: StableDiffusion3Pipeline,
    enable_cpu_offload: bool,
    enable_attention_slicing: bool,
) -> None:
    if enable_attention_slicing:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    else:
        try:
            pipe.disable_attention_slicing()
        except Exception:
            pass

    if enable_cpu_offload:
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass


@dataclass(frozen=True)
class GenConfig:
    model_id: str
    local_files_only: bool
    dtype: str
    width: int
    height: int
    steps: int
    guidance: float
    shift: Optional[float]
    use_dynamic_shifting: bool
    negative_prompt: str
    n_samples: int
    seed: int
    outdir: str
    resume: bool
    max_prompts: Optional[int]


def _load_metadata_jsonl(path: str, max_prompts: Optional[int]) -> List[dict]:
    metas: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            metas.append(json.loads(line))
            if max_prompts is not None and len(metas) >= int(max_prompts):
                break
    return metas


def _count_existing_samples(sample_dir: str) -> int:
    if not os.path.isdir(sample_dir):
        return 0
    n = 0
    for fn in os.listdir(sample_dir):
        if fn.lower().endswith(".png") and fn.split(".")[0].isdigit():
            n += 1
    return n


def _save_metadata(outpath: str, metadata: dict) -> None:
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)


def _save_images(images: List[Image.Image], sample_dir: str) -> None:
    os.makedirs(sample_dir, exist_ok=True)
    for i, im in enumerate(images):
        im.save(os.path.join(sample_dir, f"{i:04d}.png"))


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser(description="Generate GenEval images with SD3.5M")
    p.add_argument("--metadata-file", type=str, required=True, help="Path to geneval prompts JSONL (evaluation_metadata.jsonl)")
    p.add_argument("--outdir", type=str, required=True, help="Output folder to write GenEval-formatted images")

    p.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument(
        "--shift",
        type=float,
        default=None,
        help="Optional scheduler shift override. If omitted, keep the scheduler's default behavior.",
    )
    p.add_argument("--use-dynamic-shifting", action="store_true")
    p.add_argument("--negative-prompt", type=str, default="")

    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="Skip prompts that already have >= n-samples images")
    p.add_argument("--max-prompts", type=int, default=None, help="Debug option: limit number of prompts")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SD3.5M generation.")

    cfg = GenConfig(
        model_id=str(args.model_id),
        local_files_only=bool(args.local_files_only),
        dtype=str(args.dtype),
        width=_ensure_even(int(args.width)),
        height=_ensure_even(int(args.height)),
        steps=max(1, int(args.steps)),
        guidance=float(args.guidance),
        shift=None if args.shift is None else float(args.shift),
        use_dynamic_shifting=bool(args.use_dynamic_shifting),
        negative_prompt=str(args.negative_prompt or ""),
        n_samples=max(1, int(args.n_samples)),
        seed=int(args.seed),
        outdir=str(args.outdir),
        resume=bool(args.resume),
        max_prompts=args.max_prompts if args.max_prompts is None else int(args.max_prompts),
    )

    metadatas = _load_metadata_jsonl(args.metadata_file, max_prompts=cfg.max_prompts)
    os.makedirs(cfg.outdir, exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=_parse_dtype(cfg.dtype),
        local_files_only=cfg.local_files_only,
    ).to("cuda")

    # Default to safer memory settings for long runs.
    _maybe_apply_scheduler_knobs(pipe, shift=cfg.shift, use_dynamic_shifting=cfg.use_dynamic_shifting)
    _maybe_apply_memory_knobs(pipe, enable_cpu_offload=True, enable_attention_slicing=True)

    for idx, meta in enumerate(tqdm(metadatas, desc="GenEval prompts")):
        outpath = os.path.join(cfg.outdir, f"{idx:05d}")
        sample_dir = os.path.join(outpath, "samples")

        if cfg.resume and _count_existing_samples(sample_dir) >= cfg.n_samples:
            continue

        _save_metadata(outpath, meta)

        prompt = str(meta.get("prompt", "")).strip()
        if not prompt:
            # Keep folder structure consistent even if prompt is empty.
            continue

        # Use deterministic but distinct seeds per sample and per prompt index.
        base = int(cfg.seed) + int(idx) * 1000
        generators = [torch.Generator(device="cuda").manual_seed(base + i) for i in range(cfg.n_samples)]

        out = pipe(
            prompt=prompt,
            negative_prompt=cfg.negative_prompt.strip() or None,
            num_inference_steps=cfg.steps,
            guidance_scale=float(cfg.guidance),
            width=cfg.width,
            height=cfg.height,
            num_images_per_prompt=cfg.n_samples,
            generator=generators,
        )

        images = list(getattr(out, "images", []) or [])
        if len(images) != cfg.n_samples:
            # Best-effort: still save whatever got generated.
            pass
        _save_images(images, sample_dir)


if __name__ == "__main__":
    main()




