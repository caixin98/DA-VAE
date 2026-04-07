"""
VAE Upsampler Data Loader

基于piat_loader的VAE latent upsampler数据加载器
支持在线VAE编码处理

Copyright (2024) Bytedance Ltd. and/or its affiliates
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
from data.piat_loader import create_train_dataloader as create_piat_train_dataloader
from data.piat_loader import create_val_dataloader as create_piat_val_dataloader


class VAEUpsamplerDataLoader:
    """
    VAE Upsampler数据加载器
    基于piat_loader，支持在线VAE编码
    """
    
    @staticmethod
    def create_train_dataloader(
        config,
        vae_model,
        **kwargs
    ):
        """创建训练数据加载器"""
        # 使用piat_loader创建基础数据加载器
        dataloader = create_piat_train_dataloader(config)
        
        # 包装为VAE upsampler数据加载器
        return VAEUpsamplerWrapper(dataloader, vae_model, config)
    
    @staticmethod
    def create_val_dataloader(
        config,
        vae_model,
        **kwargs
    ):
        """创建验证数据加载器"""
        # 使用piat_loader创建基础数据加载器
        dataloader = create_piat_val_dataloader(config)
        
        # 包装为VAE upsampler数据加载器
        return VAEUpsamplerWrapper(dataloader, vae_model, config)


class VAEUpsamplerWrapper:
    """
    VAE Upsampler数据加载器包装器
    在piat_loader的基础上添加VAE编码功能
    """
    
    def __init__(self, base_dataloader, vae_model, config):
        self.base_dataloader = base_dataloader
        self.vae_model = vae_model
        self.config = config
        self.scale_factor = config.model.scale_factor
        self.interpolation_mode = config.dataloader.train.get('interpolation_mode', 'bicubic')
        
        # 设置VAE模型为评估模式
        self.vae_model.eval()
        for param in self.vae_model.parameters():
            param.requires_grad = False
    
    def __iter__(self):
        return VAEUpsamplerIterator(self.base_dataloader, self.vae_model, self.scale_factor, self.interpolation_mode)
    
    def __len__(self):
        return len(self.base_dataloader)
    
    @property
    def dataset(self):
        return self.base_dataloader.dataset if hasattr(self.base_dataloader, 'dataset') else None
    
    @property
    def batch_sampler(self):
        return self.base_dataloader.batch_sampler if hasattr(self.base_dataloader, 'batch_sampler') else None
    
    @property
    def sampler(self):
        return self.base_dataloader.sampler if hasattr(self.base_dataloader, 'sampler') else None
    
    @property
    def num_workers(self):
        return self.base_dataloader.num_workers if hasattr(self.base_dataloader, 'num_workers') else 0
    
    @property
    def pin_memory(self):
        return self.base_dataloader.pin_memory if hasattr(self.base_dataloader, 'pin_memory') else False
    
    @property
    def drop_last(self):
        return self.base_dataloader.drop_last if hasattr(self.base_dataloader, 'drop_last') else False


class VAEUpsamplerIterator:
    """
    VAE Upsampler迭代器
    处理VAE编码逻辑
    """
    
    def __init__(self, base_dataloader, vae_model, scale_factor, interpolation_mode):
        self.base_dataloader = base_dataloader
        self.vae_model = vae_model
        self.scale_factor = scale_factor
        self.interpolation_mode = interpolation_mode
        self.base_iterator = iter(base_dataloader)
    
    def __next__(self):
        # 获取基础数据
        batch = next(self.base_iterator)
        
        # 提取图像数据
        images = batch['image']  # (B, 3, H, W)
        cond_images = batch.get('cond_image', None)
        
        # 在线生成VAE latents
        with torch.no_grad():
            # 编码高分辨率图像
            hr_latents = self.vae_model.encode(images)
            
            # 生成低分辨率图像并编码
            if cond_images is not None:
                # 使用cond_images作为低分辨率图像
                lr_latents = self.vae_model.encode(cond_images)
            else:
                # 从高分辨率图像生成低分辨率图像
                lr_images = self._create_lr_images(images)
                lr_latents = self.vae_model.encode(lr_images)
        
        # 返回VAE upsampler需要的数据格式
        return {
            'hr_latent': hr_latents,
            'lr_latent': lr_latents,
            'image': images,
            'cond_image': cond_images,
            '__key__': batch.get('__key__', [])
        }
    
    def _create_lr_images(self, hr_images: torch.Tensor) -> torch.Tensor:
        """
        从高分辨率图像创建低分辨率图像
        
        Args:
            hr_images: 高分辨率图像 (B, 3, H, W)
            
        Returns:
            lr_images: 低分辨率图像 (B, 3, H//scale_factor, W//scale_factor)
        """
        batch_size, channels, height, width = hr_images.shape
        
        # 计算低分辨率尺寸
        lr_height = height // self.scale_factor
        lr_width = width // self.scale_factor
        
        # 下采样
        lr_images = F.interpolate(
            hr_images,
            size=(lr_height, lr_width),
            mode=self.interpolation_mode,
            align_corners=False if self.interpolation_mode in ['bilinear', 'bicubic'] else None,
            antialias=True if self.interpolation_mode in ['bilinear', 'bicubic'] else False
        )
        
        return lr_images
    
    def __iter__(self):
        return self
