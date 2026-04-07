#!/usr/bin/env python3
"""Compute the Inception Score (IS) for a directory of images."""

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
try:
    from torchmetrics.image.inception import InceptionScore  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
    raise ModuleNotFoundError(
        "torchmetrics is required to compute Inception Score. "
        "Please install it with `pip install torchmetrics`."
    ) from exc

IMAGE_EXTENSIONS: Set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".webp",
    ".tiff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Inception Score for images in a directory."
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Path to the directory containing images.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for DataLoader (default: 32).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device to run on. Defaults to CUDA if available, otherwise CPU.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=299,
        help=(
            "Resize the shorter side of every image to this size before computing IS. "
            "Set to 0 to disable resizing (all images must share the same size)."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for images in subdirectories.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=",".join(sorted(IMAGE_EXTENSIONS)),
        help=(
            "Comma-separated list of image file extensions to include. "
            "Defaults to common image formats."
        ),
    )
    return parser.parse_args()


def collect_image_paths(
    directory: Path, recursive: bool, extensions: Sequence[str]
) -> List[Path]:
    if recursive:
        iterator: Iterable[Path] = directory.rglob("*")
    else:
        iterator = directory.glob("*")

    allowed = {f".{ext.lower().lstrip('.')}" for ext in extensions}

    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in allowed
    )


class ImageFolderDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], resize: Optional[int]) -> None:
        self.image_paths = list(image_paths)
        transform_ops = []
        if resize and resize > 0:
            transform_ops.append(transforms.Resize((resize, resize)))
        transform_ops.append(transforms.PILToTensor())
        self.transform = transforms.Compose(transform_ops)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        return self.transform(image)


def main() -> None:
    args = parse_args()

    image_dir: Path = args.image_dir.expanduser().resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory '{image_dir}' does not exist.")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Provided path '{image_dir}' is not a directory.")

    extensions = [ext.strip() for ext in args.extensions.split(",") if ext.strip()]
    image_paths = collect_image_paths(image_dir, args.recursive, extensions)

    if not image_paths:
        raise RuntimeError(
            f"No images found in '{image_dir}' with extensions {extensions}."
        )

    resize = args.resize if args.resize > 0 else None
    dataset = ImageFolderDataset(image_paths, resize=resize)

    if resize is None:
        sample_shapes = {tuple(t.shape) for t in (dataset[i] for i in range(min(8, len(dataset))))}
        if len(sample_shapes) > 1:
            raise ValueError(
                "Images have varying spatial dimensions. Please enable resizing or "
                "ensure all images share identical resolution."
            )

    device_str = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = torch.device(device_str)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    metric = InceptionScore(normalize=False).to(device)

    total_images = 0
    metric.reset()

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            metric.update(batch)
            total_images += batch.size(0)

    score, score_std = metric.compute()

    print(f"Processed {total_images} images from '{image_dir}'.")
    print(f"Inception Score: {score.item():.4f} ± {score_std.item():.4f}")


if __name__ == "__main__":
    main()
