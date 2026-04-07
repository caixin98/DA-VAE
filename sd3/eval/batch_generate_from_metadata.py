#!/usr/bin/env python3
"""Batch inference utility that reads prompts from metadata JSON sources and saves outputs by category."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml



PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omini.train_sd3_davae.inference_sd3_davae import SD3TokenizerInference


def _normalize_path(path_str: Optional[str], config_dir: Path) -> Optional[Path]:
    if path_str is None:
        return None
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        try:
            return candidate.resolve(strict=False)
        except Exception:
            return candidate

    relative_candidates = [config_dir / candidate, PROJECT_ROOT / candidate]
    for path in relative_candidates:
        if path.exists():
            try:
                return path.resolve(strict=False)
            except Exception:
                return path

    fallback = relative_candidates[0]
    try:
        return fallback.resolve(strict=False)
    except Exception:
        return fallback


def _normalize_existing(path_str: Optional[str], config_dir: Path) -> Optional[Path]:
    path = _normalize_path(path_str, config_dir)
    if path is not None and path.exists():
        return path
    return None


def _discover_run_dir(train_cfg: Dict[str, object], config_dir: Path) -> Optional[Path]:
    run_name = train_cfg.get("run_name") if isinstance(train_cfg, dict) else None
    if not isinstance(run_name, str) or not run_name:
        return None

    raw_save_path = train_cfg.get("save_path", "runs") if isinstance(train_cfg, dict) else "runs"
    if not raw_save_path:
        raw_save_path = "runs"
    save_base = _normalize_path(str(raw_save_path), config_dir)
    if save_base is None:
        return None

    run_dir = save_base / run_name
    try:
        return run_dir.resolve(strict=False)
    except Exception:
        return run_dir


def _parse_step(path: Path, prefix: str) -> int:
    try:
        name = path.stem
        if name.startswith(prefix):
            step_str = name[len(prefix):]
        else:
            parts = [part for part in name.split('_') if part.isdigit()]
            if parts:
                step_str = parts[-1]
            else:
                return -1
        return int(step_str)
    except Exception:
        return -1


def _find_latest_tokenizer_checkpoint(run_dir: Optional[Path]) -> Optional[Path]:
    if run_dir is None:
        return None
    ckpt_dir = run_dir / "tokenizer_checkpoints"
    if not ckpt_dir.exists():
        return None

    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        try:
            return latest.resolve(strict=False)
        except Exception:
            return latest

    candidates: List[Tuple[int, float, Path]] = []
    for path in ckpt_dir.glob("step_*.pt"):
        step = _parse_step(path, "step_")
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0.0
        candidates.append((step, mtime, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen = candidates[0][2]
    try:
        return chosen.resolve(strict=False)
    except Exception:
        return chosen


def _find_latest_lightning_checkpoint(run_dir: Optional[Path]) -> Optional[Path]:
    if run_dir is None:
        return None
    ckpt_dir = run_dir / "full_checkpoints"
    if not ckpt_dir.exists():
        return None

    # Look for versioned last checkpoints (last-v1.ckpt, last-v2.ckpt, etc.)
    versioned_candidates: List[Tuple[int, Path]] = []
    for path in ckpt_dir.glob("last-v*.ckpt"):
        try:
            # Extract version number from last-v{N}.ckpt
            stem = path.stem  # e.g., "last-v2"
            if stem.startswith("last-v"):
                version_str = stem[6:]  # Remove "last-v" prefix
                version = int(version_str)
                versioned_candidates.append((version, path))
        except (ValueError, IndexError):
            continue
    
    # If versioned checkpoints exist, use the one with highest version
    if versioned_candidates:
        versioned_candidates.sort(key=lambda item: item[0], reverse=True)
        chosen = versioned_candidates[0][1]
        try:
            return chosen.resolve(strict=False)
        except Exception:
            return chosen
    
    # Fall back to last.ckpt if no versioned checkpoints
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        try:
            return last_ckpt.resolve(strict=False)
        except Exception:
            return last_ckpt

    # Fall back to any .ckpt file sorted by modification time
    candidates = sorted(
        ckpt_dir.glob("*.ckpt"),
        key=lambda path: (path.stat().st_mtime if path.exists() else 0.0),
        reverse=True,
    )
    if not candidates:
        return None
    chosen = candidates[0]
    try:
        return chosen.resolve(strict=False)
    except Exception:
        return chosen


def _find_latest_lora_checkpoint(run_dir: Optional[Path]) -> Optional[Path]:
    if run_dir is None:
        return None
    lora_dir = run_dir / "lora_weights"
    if not lora_dir.exists():
        return None

    latest_dir = lora_dir / "latest"
    if latest_dir.exists():
        try:
            return latest_dir.resolve(strict=False)
        except Exception:
            return latest_dir

    dir_candidates: List[Tuple[int, float, Path]] = []
    for path in lora_dir.glob("lora_step_*"):
        if not path.is_dir():
            continue
        step = _parse_step(path, "lora_step_")
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0.0
        dir_candidates.append((step, mtime, path))

    if dir_candidates:
        dir_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        chosen = dir_candidates[0][2]
        try:
            return chosen.resolve(strict=False)
        except Exception:
            return chosen

    weight_files = sorted(
        lora_dir.glob("*.safetensors"),
        key=lambda path: (path.stat().st_mtime if path.exists() else 0.0),
        reverse=True,
    )
    if weight_files:
        chosen = weight_files[0]
        try:
            return chosen.resolve(strict=False)
        except Exception:
            return chosen

    return None


@dataclass(frozen=True)
class MetadataEntry:
    image_id: str
    prompt: str
    category: str
    output_basename: str
    metadata_path: Optional[Path] = None


def load_metadata_from_directory(metadata_dir: Path) -> Dict[str, Dict[str, object]]:
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")
    if not metadata_dir.is_dir():
        raise ValueError(f"Expected metadata directory but received: {metadata_dir}")

    entries: Dict[str, Dict[str, object]] = {}
    json_files = sorted(metadata_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under metadata directory: {metadata_dir}")

    skipped_missing_prompts = 0
    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            print(f"[Batch][Warning] Failed to load metadata JSON '{json_path}': {exc}", file=sys.stderr)
            continue

        if not isinstance(payload, dict):
            print(f"[Batch][Warning] Metadata JSON '{json_path}' is not an object; skipping", file=sys.stderr)
            continue

        prompt = payload.get("prompt")
        if not prompt:
            skipped_missing_prompts += 1
            continue

        image_field = payload.get("image") or json_path.stem
        output_basename = Path(image_field).stem

        relative_parent = json_path.parent.relative_to(metadata_dir)
        category = relative_parent.as_posix()
        if not category or category == ".":
            category = "uncategorized"

        if output_basename in entries:
            existing_path = entries[output_basename].get("metadata_path")
            print(
                "[Batch][Warning] Duplicate image id '{0}' encountered in '{1}'. "
                "Existing metadata from '{2}' will be kept; skipping duplicate.".format(
                    output_basename,
                    json_path,
                    existing_path,
                ),
                file=sys.stderr,
            )
            continue

        try:
            resolved_json_path = json_path.resolve(strict=False)
        except Exception:
            resolved_json_path = json_path

        entries[output_basename] = {
            "prompt": prompt,
            "category": category,
            "metadata_path": str(resolved_json_path),
            "output_basename": output_basename,
        }

    if skipped_missing_prompts:
        print(
            f"[Batch][Warning] Skipped {skipped_missing_prompts} metadata file(s) without prompts under {metadata_dir}",
            file=sys.stderr,
        )

    if not entries:
        raise ValueError(f"No valid metadata entries with prompts found under {metadata_dir}")

    return entries


def parse_args() -> argparse.Namespace:
    default_metadata = Path(__file__).resolve().parent / "meta_data.json"

    parser = argparse.ArgumentParser(description="Generate images from metadata prompts grouped by category")
    parser.add_argument(
        "--config",
        type=str,
        default="inference/config/token_text_image_da_vae_with_lora_diff_sd35-01.yaml",
        help="Path to inference configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Optional tokenizer checkpoint path (auto discovery if omitted)",
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        help="Optional LoRA checkpoint path (auto discovery if omitted)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(default_metadata),
        help="Metadata JSON path or directory of JSON files (expects id → {prompt, category})",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./evaluation_outputs",
        help="Directory where category folders will be created",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of inference steps per prompt",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=5.0,
        help="Scheduler shift value (set on the SD3 scheduler)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Optional override for image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Optional override for image width",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target device for inference",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype to run inference with",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        help="Optional directory to store intermediate latent frames (defaults inside run folder)",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save per-step x_t frames alongside outputs",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        help="Restrict generation to one or more categories (case-sensitive)",
    )
    parser.add_argument(
        "--include-ids",
        type=str,
        nargs="+",
        help="Restrict generation to these specific metadata IDs",
    )
    parser.add_argument(
        "--exclude-ids",
        type=str,
        nargs="+",
        help="Skip these metadata IDs",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="Limit the total number of prompts generated",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip prompts whose target image already exists",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        help="Total shard count when distributing work across multiple processes",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        help="Index (0-based) of the shard handled by this process. Requires --num-shards.",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Retain intermediate x_t frames when they are saved",
    )
    parser.add_argument(
        "--sd3-baseline",
        action="store_true",
        help="[Deprecated] Same as --baseline-mode low",
    )
    parser.add_argument(
        "--baseline-mode",
        type=str,
        choices=["off", "low", "original"],
        default="off",
        help="Baseline generation using original SD3: 'low' = low+up pair, 'original' = one image at requested size",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        choices=["jpg", "png"],
        default="png",
        help="Output image format/extension",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use EMA shadow weights for inference (if available) instead of current weights",
    )
    parser.add_argument(
        "--also-ema",
        action="store_true",
        help="Additionally generate with EMA weights and save as <id>_ema.<ext>",
    )

    return parser.parse_args()


def load_metadata(metadata_path: Path) -> Dict[str, Dict[str, object]]:
    if metadata_path.is_dir():
        return load_metadata_from_directory(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Metadata JSON must be a dictionary of id → info")

    normalized: Dict[str, Dict[str, object]] = {}
    for image_id, info in data.items():
        if not isinstance(info, dict):
            continue
        prompt = info.get("prompt")
        if not prompt:
            continue
        normalized_info = dict(info)
        category = str(normalized_info.get("category") or "uncategorized").strip() or "uncategorized"
        normalized_info["prompt"] = prompt
        normalized_info["category"] = category
        normalized_info.setdefault("output_basename", image_id)
        normalized_info["metadata_path"] = normalized_info.get("metadata_path")
        normalized[image_id] = normalized_info

    return normalized


def filter_entries(
    raw: Dict[str, Dict[str, object]],
    allowed_categories: Iterable[str] | None,
    include_ids: Iterable[str] | None,
    exclude_ids: Iterable[str] | None,
) -> List[MetadataEntry]:
    allowed_categories = set(allowed_categories or []) or None
    include_ids = set(include_ids or []) or None
    exclude_ids = set(exclude_ids or [])

    entries: List[MetadataEntry] = []
    for image_id, info in raw.items():
        if include_ids is not None and image_id not in include_ids:
            continue
        if image_id in exclude_ids:
            continue

        prompt = (info or {}).get("prompt")
        if not prompt:
            continue

        raw_category = (info or {}).get("category", "uncategorized")
        category = str(raw_category).strip() or "uncategorized"
        if allowed_categories is not None and category not in allowed_categories:
            continue

        metadata_path_raw = (info or {}).get("metadata_path")
        metadata_path = Path(metadata_path_raw).expanduser() if metadata_path_raw else None

        output_basename = (info or {}).get("output_basename") or image_id
        output_basename = str(Path(output_basename).stem)

        entries.append(
            MetadataEntry(
                image_id=image_id,
                prompt=prompt,
                category=category,
                output_basename=output_basename,
                metadata_path=metadata_path,
            )
        )

    return entries


def chunk_entries(entries: Iterable[MetadataEntry], limit: int | None) -> List[MetadataEntry]:
    if limit is None or limit <= 0:
        return list(entries)
    return list(entries)[:limit]


def apply_sharding(entries: List[MetadataEntry], num_shards: int | None, shard_index: int | None) -> List[MetadataEntry]:
    if num_shards is None:
        if shard_index is not None:
            raise ValueError("--shard-index provided without --num-shards")
        return entries

    if num_shards <= 0:
        raise ValueError("--num-shards must be greater than 0")

    if shard_index is None:
        raise ValueError("--num-shards requires --shard-index to be set")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in the range [0, num_shards)")

    return [entry for idx, entry in enumerate(entries) if idx % num_shards == shard_index]


def format_run_folder_name(config_path: Path, steps: int, guidance: float, shift: float) -> str:
    config_name = config_path.stem
    guidance_tag = format_float_for_tag(guidance)
    shift_tag = format_float_for_tag(shift)
    return f"{config_name}_steps{steps}_cfg{guidance_tag}_shift{shift_tag}"


def format_float_for_tag(value: float) -> str:
    rounded = round(float(value), 4)
    if abs(rounded - int(rounded)) < 1e-6:
        return str(int(rounded))
    return f"{rounded:.4f}".rstrip("0").rstrip(".")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    # Backward compatibility: --sd3-baseline implies --baseline-mode low unless explicitly set otherwise
    if getattr(args, "sd3_baseline", False) and getattr(args, "baseline_mode", "off") == "off":
        args.baseline_mode = "low"

    metadata_path = Path(args.metadata).expanduser()
    metadata = load_metadata(metadata_path)

    entries = filter_entries(
        metadata,
        allowed_categories=args.categories,
        include_ids=args.include_ids,
        exclude_ids=args.exclude_ids,
    )

    entries.sort(key=lambda item: item.image_id)
    entries = apply_sharding(entries, args.num_shards, args.shard_index)

    entries = chunk_entries(entries, args.max_items)
    if not entries:
        print("[Batch] No prompts matched the given filters; nothing to do.")
        return

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        try:
            config_path = (PROJECT_ROOT / config_path).resolve(strict=False)
        except Exception:
            config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config_dir = config_path.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}
    if not isinstance(config_data, dict):
        raise ValueError(f"Invalid config format: expected mapping but received {type(config_data).__name__}")

    train_cfg = config_data.get("train", {}) or {}
    if not isinstance(train_cfg, dict):
        train_cfg = {}
    use_lightning_resume = bool(train_cfg.get("use_lightning_resume", False))
    auto_resume = bool(train_cfg.get("auto_resume", False))
    run_dir = _discover_run_dir(train_cfg, config_dir)

    ckpt_request = (args.checkpoint or "").strip()
    if ckpt_request.lower() in {"", "auto", "latest"}:
        ckpt_request = ""

    if ckpt_request:
        resolved_ckpt_path = _normalize_path(ckpt_request, config_dir)
    else:
        resolved_ckpt_path = None
        for field in ("resume_from_checkpoint", "checkpoint_path"):
            candidate = _normalize_existing(train_cfg.get(field), config_dir)
            if candidate is not None:
                resolved_ckpt_path = candidate
                break
        if resolved_ckpt_path is None and use_lightning_resume:
            resolved_ckpt_path = _find_latest_lightning_checkpoint(run_dir)
        if resolved_ckpt_path is None and auto_resume:
            fallback = _find_latest_tokenizer_checkpoint(run_dir)
            resolved_ckpt_path = fallback if fallback is not None else resolved_ckpt_path

    lora_request = (args.lora_checkpoint or "").strip()
    if lora_request.lower() in {"", "auto", "latest"}:
        lora_request = ""

    if lora_request:
        resolved_lora_path = _normalize_path(lora_request, config_dir)
    else:
        lora_candidate = train_cfg.get("resume_from_lora_checkpoint") or train_cfg.get("lora_checkpoint_path")
        resolved_lora_path = _normalize_existing(lora_candidate, config_dir)
        if resolved_lora_path is None:
            resolved_lora_path = _find_latest_lora_checkpoint(run_dir)

    if resolved_ckpt_path:
        print(f"[Batch] Resolved checkpoint: {resolved_ckpt_path}")
    else:
        print("[Batch] No checkpoint resolved; using base weights")

    if resolved_lora_path:
        print(f"[Batch] Resolved LoRA checkpoint: {resolved_lora_path}")

    base_output = Path(args.output_root).expanduser()
    if args.baseline_mode and args.baseline_mode != "off":
        guidance_tag = format_float_for_tag(args.guidance_scale)
        shift_tag = format_float_for_tag(args.shift)
        run_folder_name = f"base_{args.baseline_mode}_steps{args.num_inference_steps}_cfg{guidance_tag}_shift{shift_tag}"
    else:
        run_folder_name = format_run_folder_name(
            config_path=config_path,
            steps=args.num_inference_steps,
            guidance=args.guidance_scale,
            shift=args.shift,
        )
    run_root = (base_output / run_folder_name).resolve()
    ensure_directory(run_root)

    frames_directory: Path | None = None
    save_frames = args.save_frames or args.frames_dir is not None
    if save_frames:
        if args.frames_dir:
            frames_directory = Path(args.frames_dir).expanduser()
        else:
            if args.num_shards and args.shard_index is not None:
                frames_directory = run_root / f"frames_shard{args.shard_index}"
            else:
                frames_directory = run_root / "frames"
        ensure_directory(frames_directory)

    print(f"[Batch] Loaded {len(entries)} prompts from {metadata_path}")
    print(f"[Batch] Outputs will be written under: {run_root}")

    inference = SD3TokenizerInference(
        config_path=str(config_path),
        checkpoint_path=str(resolved_ckpt_path) if resolved_ckpt_path else None,
        lora_checkpoint_path=str(resolved_lora_path) if resolved_lora_path else None,
        device=args.device,
        dtype=args.dtype,
    )

    image_ext = (args.image_format or "jpg").lower()

    grouped: Dict[str, List[MetadataEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.category, []).append(entry)

    total_to_generate = sum(len(items) for items in grouped.values())
    print(f"[Batch] Prepared {total_to_generate} prompts across {len(grouped)} categories")

    generated = 0
    for category, items in grouped.items():
        category_str = str(category).strip()
        if not category_str or category_str == ".":
            category_path = Path("uncategorized")
        else:
            category_path = Path(category_str)
        category_dir = run_root / category_path
        ensure_directory(category_dir)

        for entry in items:
            target_path = category_dir / f"{entry.output_basename}.{image_ext}"
            if args.skip_existing and target_path.exists():
                if entry.metadata_path and entry.metadata_path.exists():
                    json_target = target_path.with_suffix(".json")
                    if not json_target.exists():
                        try:
                            shutil.copy2(entry.metadata_path, json_target)
                        except Exception as exc:
                            print(
                                f"[Batch][Warning] Failed to copy metadata JSON for id '{entry.image_id}': {exc}",
                                file=sys.stderr,
                            )
                continue

            category_display = category_str or "uncategorized"
            print(f"[Batch] Generating prompt for id '{entry.image_id}' in category '{category_display}'")

            existing_images = {p.name for p in category_dir.glob(f"*.{image_ext}")}
            existing_frames = {p for p in frames_directory.iterdir() if p.is_dir()} if frames_directory else set()

            # First pass: default or EMA depending on --use-ema
            # Baseline 模式下忽略 EMA 相关开关（baseline 不需要 EMA）
            if args.baseline_mode and args.baseline_mode != "off":
                use_ema_flag = None
                also_ema_enabled = False
            else:
                use_ema_flag = True if args.use_ema else None
                also_ema_enabled = bool(args.also_ema)
            inference.generate(
                prompts=[entry.prompt],
                output_dir=str(category_dir),
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                seed=args.seed,
                frames_dir=str(frames_directory) if frames_directory else None,
                save_interval=1,
                save_latents=False,
                baseline_mode=args.baseline_mode,
                scheduler_shift=args.shift,
                save_frames=save_frames,
                image_format=image_ext,
                use_ema=use_ema_flag,
            )

            new_images = [p for p in category_dir.glob(f"*.{image_ext}") if p.name not in existing_images]
            if not new_images:
                print(f"[Batch][Warning] No new image found for id '{entry.image_id}'")
                continue

            # Filter out files that disappeared between glob and stat (concurrent writes)
            candidates: List[Tuple[float, Path]] = []
            for p in new_images:
                try:
                    mtime = p.stat().st_mtime
                    candidates.append((mtime, p))
                except FileNotFoundError:
                    continue

            if not candidates:
                print(f"[Batch][Warning] No stable image candidate for id '{entry.image_id}' (race), skipping this item")
                continue

            latest_image = max(candidates, key=lambda t: t[0])[1]
            final_path = category_dir / f"{entry.output_basename}.{image_ext}"
            if final_path.exists():
                try:
                    final_path.unlink()
                except FileNotFoundError:
                    pass
            try:
                latest_image.rename(final_path)
            except FileNotFoundError:
                # The chosen file was moved/removed after selection; attempt one quick rescan
                refreshed = [p for p in category_dir.glob(f"*.{image_ext}") if p.name not in existing_images]
                stable: List[Tuple[float, Path]] = []
                for p in refreshed:
                    try:
                        stable.append((p.stat().st_mtime, p))
                    except FileNotFoundError:
                        continue
                if stable:
                    try:
                        max(stable, key=lambda t: t[0])[1].rename(final_path)
                    except Exception:
                        print(f"[Batch][Warning] Failed to finalize image for id '{entry.image_id}' on retry; skipping", file=sys.stderr)
                        continue
                else:
                    print(f"[Batch][Warning] No image available for id '{entry.image_id}' after retry; skipping", file=sys.stderr)
                    continue

            if entry.metadata_path and entry.metadata_path.exists():
                json_target = final_path.with_suffix(".json")
                try:
                    shutil.copy2(entry.metadata_path, json_target)
                except Exception as exc:
                    print(
                        f"[Batch][Warning] Failed to copy metadata JSON for id '{entry.image_id}' to '{json_target}': {exc}",
                        file=sys.stderr,
                    )

            if frames_directory:
                new_frame_dirs = [
                    p for p in frames_directory.iterdir() if p.is_dir() and p not in existing_frames
                ]
                target_frame_dir = None
                for frame_dir in new_frame_dirs:
                    candidate = frames_directory / entry.image_id
                    if candidate.exists():
                        target_frame_dir = candidate
                        break
                    frame_dir.rename(candidate)
                    target_frame_dir = candidate
                    break

                if target_frame_dir and not args.keep_frames:
                    shutil.rmtree(target_frame_dir, ignore_errors=True)

            generated += 1

            # Optional second pass: EMA generation in addition to the above
            if also_ema_enabled:
                existing_images_ema = {p.name for p in category_dir.glob(f"*.{image_ext}")}
                existing_frames_ema = {p for p in frames_directory.iterdir() if p.is_dir()} if frames_directory else set()

                inference.generate(
                    prompts=[entry.prompt],
                    output_dir=str(category_dir),
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    seed=args.seed,
                    frames_dir=str(frames_directory) if frames_directory else None,
                    save_interval=1,
                    save_latents=False,
                    baseline_mode=args.baseline_mode,
                    scheduler_shift=args.shift,
                    save_frames=save_frames,
                    image_format=image_ext,
                    use_ema=True,
                )

                new_images_ema = [p for p in category_dir.glob(f"*.{image_ext}") if p.name not in existing_images_ema]
                if new_images_ema:
                    ema_candidates: List[Tuple[float, Path]] = []
                    for p in new_images_ema:
                        try:
                            ema_candidates.append((p.stat().st_mtime, p))
                        except FileNotFoundError:
                            continue
                    if ema_candidates:
                        latest_ema = max(ema_candidates, key=lambda t: t[0])[1]
                        ema_final = category_dir / f"{entry.output_basename}_ema.{image_ext}"
                        if ema_final.exists():
                            try:
                                ema_final.unlink()
                            except FileNotFoundError:
                                pass
                        try:
                            latest_ema.rename(ema_final)
                            print(f"[Batch] Saved EMA image as: {ema_final}")
                        except FileNotFoundError:
                            pass

                if frames_directory:
                    new_frame_dirs_ema = [
                        p for p in frames_directory.iterdir() if p.is_dir() and p not in existing_frames_ema
                    ]
                    target_frame_dir_ema = None
                    for frame_dir in new_frame_dirs_ema:
                        candidate = frames_directory / f"{entry.image_id}_ema"
                        if candidate.exists():
                            target_frame_dir_ema = candidate
                            break
                        frame_dir.rename(candidate)
                        target_frame_dir_ema = candidate
                        break

                    if target_frame_dir_ema and not args.keep_frames:
                        shutil.rmtree(target_frame_dir_ema, ignore_errors=True)

    if frames_directory and not args.keep_frames:
        shutil.rmtree(frames_directory, ignore_errors=True)

    print(f"[Batch] Done. Generated {generated} images in total.")


if __name__ == "__main__":
    main()


