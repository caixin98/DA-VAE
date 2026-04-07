# wrap the stable diffusion 3 vae model, and provide a simple interface for the meta data
# here we directly use the sd3 pipeline from the diffusers library
# we do the pack and unpack operation for the latent code

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Tuple
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from einops import rearrange

class MinimalSD3Pipeline:
    def __init__(self, vae, vae_scale_factor, dtype, device):
        self.vae = vae
        self.vae_scale_factor = vae_scale_factor
        self.dtype = dtype
        self.device = device
        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

class SD3_VAE(nn.Module):
    """
    A wrapper class for the Stable Diffusion 3 VAE model that provides simple encode/decode interfaces.
    This version only loads the VAE and a minimal pipeline shell.
    """
    def __init__(self, model_path: str = "stabilityai/stable-diffusion-3-medium-diffusers", device: Optional[str] = None, deterministic_sampling: bool = False):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=self.dtype).to(self.device)
        # Ensure dtype reflects actual parameter dtype used by the loaded VAE
        try:
            self.dtype = next(self.vae.parameters()).dtype
        except StopIteration:
            pass
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self.vae, "config") else 8
        self.pipeline = MinimalSD3Pipeline(self.vae, vae_scale_factor, self.dtype, self.device)
        # 控制是否允许在 encode 时保留梯度
        self.allow_grad_encode = False
        # 控制是否使用确定性采样（使用均值而不是采样）
        self.deterministic_sampling = deterministic_sampling
        # self.vae.enable_slicing()
        # self.vae.enable_tiling()

    def encode(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        images = self.pipeline.image_processor.preprocess(image)
        # Align input dtype to actual VAE parameter dtype to avoid conv bias/input mismatch
        try:
            vae_dtype = next(self.vae.parameters()).dtype
            self.dtype = vae_dtype
        except StopIteration:
            vae_dtype = self.dtype
        images = images.to(self.device, dtype=vae_dtype)
        
        # 获取latent分布
        latent_dist = self.vae.encode(images).latent_dist
        
        if self.allow_grad_encode:
            if self.deterministic_sampling:
                # 使用均值进行确定性采样
                latents = latent_dist.mean
            else:
                # 使用随机采样
                latents = latent_dist.sample()
        else:
            with torch.no_grad():
                if self.deterministic_sampling:
                    # 使用均值进行确定性采样
                    latents = latent_dist.mean
                else:
                    # 使用随机采样
                    latents = latent_dist.sample()
        
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        # SD3 VAE不需要pack操作，直接返回latents
        return latents

    def decode(self, latents: torch.Tensor, height: int = None, width: int = None) -> torch.Tensor:
        try:
            vae_dtype = next(self.vae.parameters()).dtype
            self.dtype = vae_dtype
        except StopIteration:
            vae_dtype = self.dtype
        latents = latents.to(self.device, dtype=vae_dtype)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        # with torch.no_grad():
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image + 1) / 2
        image = torch.clamp(image, 0, 1)
        return image

    def generate_random_latents(self, batch_size: int, height: int, width: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        vae_scale_factor = self.pipeline.vae_scale_factor
        height = int(height) // vae_scale_factor
        width = int(width) // vae_scale_factor
        shape = (batch_size, self.vae.config.latent_channels, height, width)
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
        # SD3 VAE不需要pack操作，直接返回latents
        return latents

    def to(self, device: str, dtype: Optional[torch.dtype] = None) -> 'SD3_VAE':
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        # Move and optionally cast the underlying VAE
        self.vae = self.vae.to(device)
        if dtype is not None:
            self.vae = self.vae.to(dtype=self.dtype)
        # Keep self.dtype in sync with actual parameter dtype
        try:
            self.dtype = next(self.vae.parameters()).dtype
        except StopIteration:
            pass
        # Sync pipeline settings
        self.pipeline.device = device
        self.pipeline.dtype = self.dtype
        return self

    def enable_slicing(self):
        self.vae.enable_slicing()

    def disable_slicing(self):
        self.vae.disable_slicing()

    def enable_tiling(self):
        self.vae.enable_tiling()

    def disable_tiling(self):
        self.vae.disable_tiling()
