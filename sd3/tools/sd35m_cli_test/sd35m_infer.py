#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI inference script for SD3.5 Medium (Stable Diffusion 3.5 Medium).

Notes:
- This script mirrors the inference knobs exposed by tools/sd35m_web_demo/app.py
  (shift, dynamic shifting, cpu offload, attention slicing, etc.).
- Comments are intentionally in English per repo style.
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

import torch
try:
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
        StableDiffusion3Pipeline,
    )
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency: diffusers. "
        "Install dependencies first, e.g.:\n"
        "  pip install -r tools/sd35m_cli_test/requirements.txt\n"
        "or:\n"
        "  pip install -r tools/sd35m_web_demo/requirements.txt\n"
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


def _seed_to_generator(device: str, seed: int) -> torch.Generator:
    g = torch.Generator(device=device if str(device).startswith("cuda") else "cpu")
    g.manual_seed(int(seed))
    return g


@dataclass
class RunConfig:
    model_id: str
    device: str
    dtype: str
    local_files_only: bool

    prompt: str
    negative_prompt: str

    width: int
    height: int
    steps: int
    guidance: float
    seed: int
    batch_size: int

    shift: float
    use_dynamic_shifting: bool
    enable_cpu_offload: bool
    enable_attention_slicing: bool

    output_dir: str


def _maybe_apply_scheduler_knobs(pipe: StableDiffusion3Pipeline, shift: float, use_dynamic_shifting: bool) -> None:
    # Best-effort: different diffusers versions may or may not expose these APIs.
    try:
        pipe.scheduler.config.use_dynamic_shifting = bool(use_dynamic_shifting)
    except Exception:
        pass
    try:
        pipe.scheduler.set_shift(float(shift))
    except Exception:
        pass


def _maybe_apply_memory_knobs(
    pipe: StableDiffusion3Pipeline,
    enable_cpu_offload: bool,
    enable_attention_slicing: bool,
) -> None:
    # Best-effort: keep running even if a knob is unavailable.
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


def _save_images(images: Any, out_dir: str, stem: str) -> int:
    if images is None:
        return 0
    if isinstance(images, Image.Image):
        images = [images]
    if not isinstance(images, list):
        return 0

    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for i, im in enumerate(images):
        if not isinstance(im, Image.Image):
            continue
        fp = os.path.join(out_dir, f"{stem}_{i:02d}.png")
        im.save(fp)
        n += 1
    return n


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser(description="SD3.5 Medium CLI Inference (txt2img)")
    p.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-3.5-medium", help="HF repo id or local directory")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Weights dtype")
    p.add_argument("--local-files-only", action="store_true", help="Load only from local cache (no network)")

    p.add_argument("--prompt", type=str, required=True, help="Text prompt")
    p.add_argument("--negative-prompt", type=str, default="", help="Negative prompt (optional)")

    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=0, help="Seed (-1 for random)")
    p.add_argument("--batch-size", type=int, default=1, help="num_images_per_prompt")

    p.add_argument("--shift", type=float, default=2.0, help="scheduler shift")
    p.add_argument("--use-dynamic-shifting", action="store_true", help="Enable dynamic shifting if supported")
    p.add_argument("--enable-cpu-offload", action="store_true", help="Enable CPU offload (lower VRAM)")
    p.add_argument("--enable-attention-slicing", action="store_true", help="Enable attention slicing (lower VRAM)")

    p.add_argument("--output-dir", type=str, default="outputs/sd35m", help="Where to save results")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available. Use --device cpu or check environment.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    width = _ensure_even(int(args.width))
    height = _ensure_even(int(args.height))
    steps = max(1, int(args.steps))
    batch_size = max(1, int(args.batch_size))

    seed = int(args.seed)
    if seed < 0:
        seed = int(np.random.randint(0, 2**31 - 1))

    cfg = RunConfig(
        model_id=str(args.model_id),
        device=str(args.device),
        dtype=str(args.dtype),
        local_files_only=bool(args.local_files_only),
        prompt=str(args.prompt),
        negative_prompt=str(args.negative_prompt or ""),
        width=width,
        height=height,
        steps=steps,
        guidance=float(args.guidance),
        seed=seed,
        batch_size=batch_size,
        shift=float(args.shift),
        use_dynamic_shifting=bool(args.use_dynamic_shifting),
        enable_cpu_offload=bool(args.enable_cpu_offload),
        enable_attention_slicing=bool(args.enable_attention_slicing),
        output_dir=str(args.output_dir),
    )

    t_load0 = time.time()
    pipe = StableDiffusion3Pipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=_parse_dtype(cfg.dtype),
        local_files_only=cfg.local_files_only,
    )
    if cfg.device == "cuda":
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    t_load = time.time() - t_load0

    _maybe_apply_scheduler_knobs(pipe, shift=cfg.shift, use_dynamic_shifting=cfg.use_dynamic_shifting)
    _maybe_apply_memory_knobs(pipe, enable_cpu_offload=cfg.enable_cpu_offload, enable_attention_slicing=cfg.enable_attention_slicing)

    dev_for_gen = "cuda" if (cfg.device == "cuda") else "cpu"
    gen = _seed_to_generator(dev_for_gen, cfg.seed)

    t_inf0 = time.time()
    out = pipe(
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt.strip() or None,
        num_inference_steps=cfg.steps,
        guidance_scale=float(cfg.guidance),
        width=cfg.width,
        height=cfg.height,
        num_images_per_prompt=cfg.batch_size,
        generator=gen,
    )
    t_inf = time.time() - t_inf0

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    stem = f"sd35m_{ts}_seed{cfg.seed}_{cfg.width}x{cfg.height}_s{cfg.steps}_g{cfg.guidance}_b{cfg.batch_size}"

    os.makedirs(cfg.output_dir, exist_ok=True)
    meta_path = os.path.join(cfg.output_dir, f"{stem}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_config": asdict(cfg),
                "timing": {"load_sec": t_load, "inference_sec": t_inf},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    n_saved = _save_images(getattr(out, "images", None), cfg.output_dir, stem=stem)
    print(f"[sd35m_cli] saved_images={n_saved} seed={cfg.seed} size={cfg.width}x{cfg.height} steps={cfg.steps} guidance={cfg.guidance}")
    print(f"[sd35m_cli] output_dir={os.path.abspath(cfg.output_dir)}")
    print(f"[sd35m_cli] meta={os.path.abspath(meta_path)}")
    print(f"[sd35m_cli] timing load={t_load:.2f}s infer={t_inf:.2f}s")


if __name__ == "__main__":
    main()


