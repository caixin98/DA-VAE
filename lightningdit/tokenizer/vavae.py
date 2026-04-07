
"""
Vision Foundation Model Aligned VAE.
It has exactly the same architecture as the LDM VAE (or VQGAN-KL).
Here we first provide its inference implementation with diffusers.
The training code will be provided later.

"LightningDiT + VA_VAE" achieves state-of-the-art Latent Diffusion System
with 0.27 rFID and 1.35 FID on ImageNet 256x256.

by Maple (Jingfeng Yao) from HUST-VL
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms

from tokenizer.autoencoder import AutoencoderKL


class VA_VAE:
    """Vision Foundation Model Aligned VAE Implementation."""

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

        self.embed_dim = self.config.model.params.embed_dim
        self.ckpt_path = self.config.ckpt_path
        self.img_size = img_size
        self.horizon_flip = horizon_flip
        self.fp16 = bool(fp16) and torch.cuda.is_available()
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = torch.device(device)
        self.model: Optional[AutoencoderKL] = None
        self.load()

    def load(self) -> "VA_VAE":
        """Load and initialise the VAE weights."""
        model = AutoencoderKL(
            embed_dim=self.embed_dim,
            ch_mult=(1, 1, 2, 2, 4),
            ckpt_path=self.ckpt_path,
        )
        model = model.to(self.device)
        if self.fp16 and self.device.type == "cuda":
            model = model.half()
        else:
            model = model.float()
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

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent representations."""
        if self.model is None:
            raise RuntimeError("VAE model is not loaded. Call `load()` first.")
        with torch.no_grad():
            # Ensure input tensor matches model device and dtype (fp16/fp32)
            model_dtype = next(self.model.parameters()).dtype
            latents = images.to(self.device, dtype=model_dtype)
            posterior = self.model.encode(latents)
            sample = posterior.sample()
        return sample.to(images.dtype)

    def decode_to_images(self, latents: torch.Tensor) -> np.ndarray:
        """Decode latent representations to uint8 numpy images."""
        if self.model is None:
            raise RuntimeError("VAE model is not loaded. Call `load()` first.")
        with torch.no_grad():
            # Ensure latent tensor matches model device and dtype (fp16/fp32)
            model_dtype = next(self.model.parameters()).dtype
            latents = latents.to(self.device, dtype=model_dtype)
            images = self.model.decode(latents)
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
    vae = VA_VAE('tokenizer/configs/vavae_f16d32.yaml')
    vae.load()
