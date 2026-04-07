#!/usr/bin/env python3
"""Convenience wrapper to run SD3 DAVAE inference over a prompt list."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from omini.train_sd3_davae.inference_sd3_davae import SD3TokenizerInference


def _read_prompts(prompt_file: Path) -> List[str]:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with prompt_file.open("r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]

    if not prompts:
        raise ValueError(f"Prompt file {prompt_file} does not contain any non-empty lines")

    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SD3 DAVAE inference for a list of prompts using shared parameters",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="inference/config/token_text_image_da_vae_with_lora_diff_sd35-01.yaml",
        help="Path to the inference configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Optional tokenizer checkpoint path (defaults to auto discovery)",
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        help="Optional LoRA checkpoint path (defaults to auto discovery)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="Text file with one prompt per line",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="Additional prompts passed directly via CLI",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./inference_output",
        help="Directory to store generated results",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of inference steps to run per prompt",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed (set the same seed for reproducible results)",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=5.0,
        help="Scheduler shift value",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target device (e.g., cuda, cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype used during inference",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Optional override for generated image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Optional override for generated image width",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        help="Directory to store per-step frames (defaults to <output-dir>/frames)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Save intermediate frames every N steps",
    )
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help="If set, persist latent tensors for each saved step",
    )
    parser.add_argument(
        "--sd3-baseline",
        action="store_true",
        help="Generate additional SD3 baseline comparison images",
    )

    args = parser.parse_args()

    prompts: List[str] = []
    if args.prompt_file:
        prompts.extend(_read_prompts(Path(args.prompt_file)))
    if args.prompts:
        prompts.extend([prompt for prompt in args.prompts if prompt.strip()])

    if not prompts:
        parser.error("No prompts provided. Use --prompt-file or --prompts to supply at least one prompt.")

    args.prompts = prompts
    return args


def main() -> None:
    args = parse_args()

    inference = SD3TokenizerInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        lora_checkpoint_path=args.lora_checkpoint,
        device=args.device,
        dtype=args.dtype,
    )

    inference.generate(
        prompts=args.prompts,
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
        frames_dir=args.frames_dir,
        save_interval=args.save_interval,
        save_latents=args.save_latents,
        sd3_baseline=args.sd3_baseline,
        scheduler_shift=args.shift,
    )


if __name__ == "__main__":
    main()


