"""
Detail-Aligned VAE (DA-VAE) wrapper for inference.

Mirrors the public API of `tokenizer.vavae.VA_VAE` so that callers can
switch variants transparently:
- img_transform(p_hflip, img_size)
- encode_images(images) -> latents
- decode_to_images(latents) -> uint8 numpy images

This wrapper instantiates `ldm.models.da_autoencoder.DAVAE` from the
local `vae/` package and uses its DA path for encode/decode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms

import sys


def _ensure_ldm_on_path():
    """Ensure that `ldm` can be imported.

    The DA-VAE autoencoder uses absolute imports like `from ldm...`, so we
    add the `davae` directory (which contains the `ldm` package) to sys.path.
    """
    this_file = Path(__file__).resolve()
    davae_root = this_file.parent.parent / "davae"
    if str(davae_root) not in sys.path:
        sys.path.insert(0, str(davae_root))


_ensure_ldm_on_path()
from ldm.models.da_autoencoder import DAVAE  # noqa: E402


class DA_VAE:
    """Detail-Aligned VAE Inference Wrapper."""

    def __init__(
        self,
        config: Union[str, Path, DictConfig],
        img_size: int = 256,
        horizon_flip: float = 0.5,
        fp16: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        if isinstance(config, (str, Path)):
            self.config = OmegaConf.load(config)
        elif isinstance(config, DictConfig):
            self.config = config
        else:
            self.config = OmegaConf.create(config)

        # Pull required fields from config
        model_cfg = self.config.model
        params = model_cfg.params

        self.base_embed_dim = params.base_embed_dim
        self.base_ckpt_path = params.get("base_ckpt_path", None)
        self.da_ckpt_path = params.get("da_ckpt_path", None)
        self.ddconfig_base = params.ddconfig_base
        self.ddconfig_da = params.ddconfig_da
        self.lossconfig = params.lossconfig
        self.enable_deep_compress = params.get("enable_deep_compress", True)
        self.upsample_interpolation = params.get("upsample_interpolation", "nearest")
        # Optional advanced params
        self.monitor = params.get("monitor", None)
        self.da_mode = params.get("da_mode", "full")
        self.da_factor = params.get("da_factor", 2)
        self.align_method = params.get("align_method", "proj")
        self.freeze_da_encoder = params.get("freeze_da_encoder", False)
        self.freeze_da_decoder = params.get("freeze_da_decoder", False)
        self.pe_only_mode = params.get("pe_only_mode", False)
        self.pe_only_weight = params.get("pe_only_weight", 1.0)
  

        self.img_size = img_size
        self.horizon_flip = horizon_flip
        self.fp16 = bool(fp16) and torch.cuda.is_available()
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = torch.device(device)
        self.model: Optional[DAVAE] = None
        self.load()

    def load(self) -> "DA_VAE":
        """Load and initialise the DA-VAE weights."""
        model = DAVAE(
            ddconfig_base=self.ddconfig_base,
            ddconfig_da=self.ddconfig_da,
            lossconfig=self.lossconfig,
            base_embed_dim=self.base_embed_dim,
            base_ckpt_path=self.base_ckpt_path,
            da_ckpt_path=self.da_ckpt_path,
            enable_deep_compress=self.enable_deep_compress,
            upsample_interpolation=self.upsample_interpolation,
            da_mode=self.da_mode,
            da_factor=self.da_factor,
            align_method=self.align_method,
        )
        model = model.to(self.device)
        if self.fp16 and self.device.type == "cuda":
            model = model.half()
        else:
            model = model.float()

        # Optional: load from top-level ckpt_path in config for inference
        try:
            top_ckpt_path = str(self.config.get("ckpt_path", "")).strip()
        except Exception:
            top_ckpt_path = ""
        if top_ckpt_path:
            try:
                obj = torch.load(top_ckpt_path, map_location="cpu")
                # prefer inner dicts commonly used in checkpoints
                sd = None
                if isinstance(obj, dict):
                    for k in ("ema", "state_dict", "model"):
                        if k in obj and isinstance(obj[k], dict):
                            sd = obj[k]
                            break
                if sd is None and isinstance(obj, dict):
                    sd = obj
                # strip potential DistributedDataParallel prefixes
                cleaned = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
                missing, unexpected = model.load_state_dict(cleaned, strict=False)
                print(f"[DA_VAE] loaded top-level ckpt: missing={len(missing)} unexpected={len(unexpected)}")
            except Exception as e:
                print(f"[DA_VAE] warning: failed to load top-level ckpt_path ({top_ckpt_path}): {e}")

        model.eval()
        self.model = model
        return self

    def img_transform(self, p_hflip: float = 0.0, img_size: Optional[int] = None):
        """
        Image preprocessing transforms.

        Args:
            p_hflip: Probability of horizontal flip.
            img_size: Target image size, use default if ``None``.
        Returns:
            torchvision.transforms.Compose
        """
        img_size = img_size if img_size is not None else self.img_size
        img_transforms = [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, img_size)),
            transforms.RandomHorizontalFlip(p=p_hflip),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
        return transforms.Compose(img_transforms)

    def encode_images(self, images: torch.Tensor, return_residual: bool = False) -> torch.Tensor:
        """Encode images to DC latent representations (z_hc).

        Returns z from the DA encoder posterior. This z can be passed to
        `decode_to_images` below for reconstruction.
        """
        if self.model is None:
            raise RuntimeError("VAE model is not loaded. Call `load()` first.")
        with torch.no_grad():
            model_dtype = next(self.model.parameters()).dtype
            x = images.to(self.device, dtype=model_dtype)
            # In diff mode, decoder expects concatenated [z_b, z_hc_student]
            if getattr(self, "da_mode", "full") == "detail":
                base_post = self.model.encode_base(x)
                z_b = base_post.mode().detach()
                hc_post = self.model.encode_hc(x)
                z_hc_student = hc_post.sample()
                z_hc_student = z_hc_student + torch.randn_like(z_hc_student)
                z = torch.cat([z_b, z_hc_student], dim=1)
                if return_residual:
                    return z, z_hc_student
            else:
                hc_post = self.model.encode_hc(x)
                z = hc_post.sample()
        return z.to(images.dtype)

    def decode_to_images(self, latents: torch.Tensor) -> np.ndarray:
        """Decode DC latent representations to uint8 numpy images."""
        if self.model is None:
            raise RuntimeError("VAE model is not loaded. Call `load()` first.")
        with torch.no_grad():
            model_dtype = next(self.model.parameters()).dtype
            z_hc = latents.to(self.device, dtype=model_dtype)
            images = self.model.decode(z_hc)
            images = (
                torch.clamp(127.5 * images + 128.0, 0, 255)
                .permute(0, 2, 3, 1)
                .to("cpu", dtype=torch.uint8)
                .numpy()
            )
        return images


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


if __name__ == "__main__":
    vae = DA_VAE('tokenizer/configs/davae_f16d32.yaml')
    vae.load()


