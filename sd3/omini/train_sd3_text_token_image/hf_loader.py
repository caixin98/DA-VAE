"""Utilities for building dataloaders from HuggingFace datasets."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import DataLoader
from torchvision import transforms as T

try:
    from datasets import load_dataset, Image as HFImage
except Exception as exc:  # pragma: no cover - only triggered when datasets missing
    load_dataset = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _default_transform(image_size: int) -> Callable:
    return T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ]
    )


def create_hf_text2image_dataloader(
    dataset_name: str,
    split: str = "train",
    image_key: str = "image",
    caption_key: str = "text",
    batch_size: int = 1,
    num_workers: int = 0,
    image_size: int = 512,
    shuffle: bool = True,
    streaming: bool = False,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Build a dataloader that yields dicts with `image` tensor and `strText` caption."""
    if load_dataset is None:  # pragma: no cover - see above
        raise ImportError(
            "datasets.load_dataset is unavailable. Install `datasets` or check the recorded import error"
        ) from _IMPORT_ERROR

    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    transform = _default_transform(image_size)
    image_feature = HFImage()

    def _to_pil(image):
        if isinstance(image, PILImage.Image):
            return image.convert("RGB")
        if isinstance(image, dict):
            if "path" in image or "bytes" in image:
                return image_feature.decode_example(image).convert("RGB")
            if "array" in image:
                return PILImage.fromarray(np.array(image["array"])).convert("RGB")
            for value in image.values():
                try:
                    return PILImage.fromarray(np.array(value)).convert("RGB")
                except Exception:
                    continue
        if isinstance(image, (list, tuple)):
            arr = np.asarray(image)
            arr = np.squeeze(arr)
            while arr.ndim > 3 and arr.shape[0] == 1:
                arr = np.squeeze(arr, axis=0)
            if arr.ndim == 4 and arr.shape[-1] in (1, 3):
                arr = arr.reshape(arr.shape[-3:])
            arr = arr.astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            return PILImage.fromarray(arr).convert("RGB")
        try:
            return image_feature.decode_example(image).convert("RGB")
        except Exception:
            arr = np.asarray(image)
            arr = np.squeeze(arr)
            while arr.ndim > 3 and arr.shape[0] == 1:
                arr = np.squeeze(arr, axis=0)
            if arr.ndim == 4 and arr.shape[-1] in (1, 3):
                arr = arr.reshape(arr.shape[-3:])
            arr = arr.astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            return PILImage.fromarray(arr).convert("RGB")

    def _convert(example: Dict):
        images = example[image_key]
        captions = example.get(caption_key)

        if isinstance(images, list):
            converted_images = []
            converted_captions = []
            for idx, img in enumerate(images):
                converted_images.append(transform(_to_pil(img)))
                if isinstance(captions, list):
                    caption = captions[idx] if idx < len(captions) else ""
                else:
                    caption = captions or ""
                converted_captions.append(caption)
            example["image"] = converted_images
            example["strText"] = converted_captions
        else:
            example["image"] = transform(_to_pil(images))
            if isinstance(captions, list):
                example["strText"] = captions[0] if captions else ""
            else:
                example["strText"] = captions or ""
        return example

    if streaming:
        mapped = ds.map(_convert)

        def _collate(batch):
            images = torch.stack([item["image"] for item in batch])
            captions = [item["strText"] for item in batch]
            return {"image": images, "strText": captions}

        return DataLoader(
            mapped,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=_collate,
        )

    ds = ds.cast_column(image_key, HFImage())
    ds = ds.with_transform(_convert)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    def _collate(batch):
        images = torch.stack([item["image"] for item in batch])
        captions = [item["strText"] for item in batch]
        return {"image": images, "strText": captions}

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
    )
