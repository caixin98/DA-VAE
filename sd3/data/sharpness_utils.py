from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Optional, Dict, Literal

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment dependency
    raise ImportError("OpenCV (cv2) is required for sharpness utilities.") from exc

from PIL import Image

Method = Literal["laplacian", "tenengrad"]


def _resize_gray(gray: np.ndarray, resize_to: Optional[int]) -> np.ndarray:
    """Helper to resize a grayscale image (float32, [0,1]) so that its longer side == resize_to."""
    if resize_to is None:
        return gray

    h, w = gray.shape[:2]
    if max(h, w) == resize_to or resize_to <= 0:
        return gray

    scale = resize_to / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(gray, (new_w, new_h), interpolation=interp)


def load_gray(path: str | Path, resize_to: Optional[int] = 512) -> Optional[np.ndarray]:
    """
    Load a single-channel grayscale image and normalise to [0, 1] float32.
    Optionally resizes the image so that its longest side equals `resize_to`.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    gray = img.astype(np.float32) / 255.0
    return _resize_gray(gray, resize_to)


def pil_to_gray_array(
    pil_img: Image.Image,
    resize_to: Optional[int] = None,
) -> np.ndarray:
    """
    Convert a PIL image to a float32 grayscale ndarray in [0, 1].
    Optionally resizes the image so that its longest side equals `resize_to`.
    """
    gray = np.asarray(pil_img.convert("L"), dtype=np.float32) / 255.0
    return _resize_gray(gray, resize_to)


def sharpness_raw(gray: np.ndarray, method: Method = "laplacian", ksize: int = 3) -> float:
    """
    Compute the raw sharpness metric from a grayscale image.

    - laplacian: variance of Laplacian response.
    - tenengrad: mean energy of Sobel gradients.
    """
    if gray.ndim != 2:
        raise ValueError("sharpness_raw expects a 2D grayscale image.")

    if method == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
        return float(lap.var())
    if method == "tenengrad":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
        return float(np.mean(gx * gx + gy * gy))

    raise ValueError(f"Unknown sharpness method: {method}")


def calibrate_percentiles(
    image_paths: Iterable[str | Path],
    method: Method = "laplacian",
    resize_to: Optional[int] = 512,
    ksize: int = 3,
    p_low: float = 5.0,
    p_high: float = 95.0,
) -> Tuple[float, float]:
    """
    Compute robust percentile-based calibration bounds for sharpness scores.
    Returns (p_low_value, p_high_value).
    """
    vals = []
    for path in image_paths:
        gray = load_gray(path, resize_to=resize_to)
        if gray is None:
            continue
        try:
            vals.append(sharpness_raw(gray, method=method, ksize=ksize))
        except Exception:
            continue

    if not vals:
        return 0.0, 1.0

    lo, hi = np.percentile(vals, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def to_score_0_100(value: float, p_lo: float, p_hi: float) -> float:
    """
    Linearly map a raw sharpness value into the [0, 100] range using percentile bounds.
    """
    denom = max(p_hi - p_lo, 1e-6)
    score = (value - p_lo) / denom
    score = np.clip(score, 0.0, 1.0) * 100.0
    return float(score)


def score_image(
    image_path: str | Path,
    p_lo: float,
    p_hi: float,
    method: Method = "laplacian",
    resize_to: Optional[int] = 512,
    ksize: int = 3,
) -> Dict[str, float]:
    """
    Convenience helper to score a single image, returning {raw, score}.
    """
    gray = load_gray(image_path, resize_to=resize_to)
    if gray is None:
        return {"raw": float("nan"), "score": float("nan")}
    raw = sharpness_raw(gray, method=method, ksize=ksize)
    score = to_score_0_100(raw, p_lo, p_hi)
    return {"raw": raw, "score": score}


def compute_sharpness_from_pil(
    pil_img: Image.Image,
    method: Method = "laplacian",
    resize_to: Optional[int] = None,
    ksize: int = 3,
) -> float:
    """
    Compute the raw sharpness value directly from a PIL image.
    """
    gray = pil_to_gray_array(pil_img, resize_to=resize_to)
    return sharpness_raw(gray, method=method, ksize=ksize)


__all__ = [
    "Method",
    "load_gray",
    "pil_to_gray_array",
    "sharpness_raw",
    "calibrate_percentiles",
    "to_score_0_100",
    "score_image",
    "compute_sharpness_from_pil",
]

