#!/usr/bin/env python3

"""Compute FID between generated images and a reference directory."""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path
from typing import Optional

"""Minimize noisy logs before importing libraries that may touch CUDA/NCCL."""
os.environ.setdefault("NCCL_DEBUG", "WARN")
for var in ("WORLD_SIZE", "LOCAL_RANK", "RANK"):
    if var in os.environ:
        os.environ.pop(var)

import torch

from cleanfid import fid

try:
    # Prefer the shared implementation you added in eval/eval.py.
    # When running `python eval/compute_fid.py ...`, that directory is on sys.path,
    # so `eval.py` is importable as module `eval`.
    from eval import filter_corrupted_images  # type: ignore
except Exception:  # pragma: no cover
    # Fallback: if import fails for any reason, do a best-effort check.
    def filter_corrupted_images(root: Path) -> None:
        """Best-effort corrupted-image filter to avoid CleanFID crashing on bad files."""

        try:
            from PIL import Image  # type: ignore
        except Exception as exc:
            print(f"[compute_fid] WARNING: Pillow not available; skip corrupted-image check ({exc})")
            return

        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        moved = 0
        checked = 0
        corrupted_root = root / "__corrupted__"
        for img_path in root.rglob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in exts:
                continue
            checked += 1
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except Exception:
                rel = img_path.relative_to(root)
                dst = corrupted_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    img_path.rename(dst)
                    moved += 1
                except Exception:
                    pass

        if moved:
            print(f"[compute_fid] filtered corrupted images: moved {moved}/{checked} into {corrupted_root}")
        else:
            print(f"[compute_fid] corrupted-image check: ok ({checked} files)")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID score")
    parser.add_argument(
        "generated",
        type=Path,
        help="Generated image directory",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Reference image directory (optional when using --stats-name)",
    )
    parser.add_argument(
        "--stats-name",
        default="mjhq30k_clean",
        type=str,
        help="Optional CleanFID stats cache name (custom stats must exist)",
    )
    parser.add_argument(
        "--stats-mode",
        type=str,
        default="clean",
        help="CleanFID stats mode when using --stats-name (default: clean)",
    )
    parser.add_argument(
        "--dataset-res",
        type=int,
        default=1024,
        help="Dataset resolution tag used when loading custom stats (default: 512)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Primary torch device (default: cuda)",
    )
    parser.add_argument(
        "--single-device",
        action="store_true",
        help="Disable torch.nn.DataParallel and force single-device execution",
    )
    parser.add_argument(
        "--create-stats",
        action="store_true",
        help="When using --stats-name, create reference stats from --reference before computing FID",
    )
    return parser.parse_args(argv)


def validate_dir(path: Path, kind: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{kind} directory not found: {path}")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    generated = args.generated.expanduser().resolve()

    validate_dir(generated, "Generated")

    reference: Optional[Path]
    if args.reference is not None:
        reference = args.reference.expanduser().resolve()
        validate_dir(reference, "Reference")
    else:
        reference = None
        if not args.stats_name:
            raise ValueError("Reference path is required when --stats-name is not provided")

    # Filter corrupted images (outputs status to stdout)
    filter_corrupted_images(generated)
    if args.create_stats:
        if not args.stats_name:
            raise ValueError("--create-stats requires --stats-name")
        if reference is None:
            raise ValueError("--create-stats requires --reference (to build stats)")
        # Ensure reference images are sane before building stats
        filter_corrupted_images(reference)

    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        use_dataparallel = not args.single_device
    except Exception as exc:
        print(f"[compute_fid] Falling back to CPU ({exc})", file=sys.stderr)
        device = torch.device("cpu")
        use_dataparallel = False

    if args.stats_name:
        if args.create_stats:
            make_custom_stats = getattr(fid, "make_custom_stats", None)
            if make_custom_stats is None:
                raise RuntimeError(
                    "cleanfid.fid.make_custom_stats is not available in this cleanfid version; "
                    "please upgrade cleanfid or create stats externally."
                )
            # Build/overwrite custom stats for the reference directory.
            # NOTE: clean-fid versions differ a lot here; e.g. 0.1.35 signature is:
            #   (name, fdir, num=None, mode='clean', model_name='inception_v3', num_workers=0, batch_size=64, device=..., verbose=True)
            sig = inspect.signature(make_custom_stats)
            param_names = set(sig.parameters.keys())

            # Always provide required positional args if present in signature
            # (we still pass them positionally to avoid keyword mismatches).
            pos_args = [args.stats_name, str(reference)]

            kwargs: dict[str, object] = {}
            if "mode" in param_names:
                kwargs["mode"] = args.stats_mode
            if "device" in param_names:
                kwargs["device"] = device
            if "verbose" in param_names:
                kwargs["verbose"] = True

            # Some versions accept "num" to limit the number of images for stats.
            # We don't set it here (full stats), but keep compatibility if needed later.
            make_custom_stats(*pos_args, **kwargs)

        score = fid.compute_fid(
            str(generated),
            dataset_name=args.stats_name,
            dataset_split="custom",
            dataset_res=args.dataset_res,
            mode=args.stats_mode,
            device=device,
            use_dataparallel=use_dataparallel,
        )
    elif reference is not None:
        score = fid.compute_fid(
            str(reference),
            str(generated),
            device=device,
            use_dataparallel=use_dataparallel,
        )
    else:
        raise ValueError("Cannot compute FID without reference statistics or reference directory")

    print(score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

