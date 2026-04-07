
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import hf_hub_download

HF_VAE_REPO = "hustvl/vavae-imagenet256-f16d32-dinov2"
HF_VAE_FILENAME = "vavae-imagenet256-f16d32-dinov2.pt"
HF_DIT_REPO = "hustvl/lightningdit-xl-imagenet256-800ep"
HF_DIT_FILENAME = "lightningdit-xl-imagenet256-800ep.pt"

DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "LIGHTNINGDIT_CACHE",
        Path(os.environ.get("HF_HOME", Path.home() / ".cache")) / "lightningdit",
    )
)

PathLike = Union[str, os.PathLike]


def _prepare_path(path: PathLike | None) -> Optional[Path]:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _download_if_needed(
    *,
    repo_id: str,
    filename: str,
    target_path: Optional[Path],
    cache_dir: Optional[Path],
) -> Path:
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    if target_path and target_path.exists():
        return target_path

    downloaded = Path(
        hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache_dir))
    )

    if target_path is None:
        return downloaded

    if downloaded.samefile(target_path):
        return target_path

    shutil.copy2(downloaded, target_path)
    return target_path


def ensure_vae_checkpoint(
    local_path: Optional[PathLike] = None,
    *,
    cache_dir: Optional[PathLike] = None,
) -> Path:
    cache_path = _prepare_path(cache_dir) if cache_dir is not None else None
    target = _prepare_path(local_path)
    return _download_if_needed(
        repo_id=HF_VAE_REPO,
        filename=HF_VAE_FILENAME,
        target_path=target,
        cache_dir=cache_path,
    )


def ensure_dit_checkpoint(
    local_path: Optional[PathLike] = None,
    *,
    cache_dir: Optional[PathLike] = None,
) -> Path:
    cache_path = _prepare_path(cache_dir) if cache_dir is not None else None
    target = _prepare_path(local_path)
    return _download_if_needed(
        repo_id=HF_DIT_REPO,
        filename=HF_DIT_FILENAME,
        target_path=target,
        cache_dir=cache_path,
    )


def resolve_checkpoint_paths(
    *,
    dit_checkpoint: Optional[PathLike] = None,
    vae_checkpoint: Optional[PathLike] = None,
    cache_dir: Optional[PathLike] = None,
) -> dict[str, Path]:
    cache_path = _prepare_path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    return {
        "dit": ensure_dit_checkpoint(local_path=dit_checkpoint, cache_dir=cache_path),
        "vae": ensure_vae_checkpoint(local_path=vae_checkpoint, cache_dir=cache_path),
    }
