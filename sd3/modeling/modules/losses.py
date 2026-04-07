"""This files contains training loss implementation.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Ref:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
"""
from typing import Mapping, Text, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.amp import autocast
import matplotlib.pyplot as plt
import numpy as np
from .pca_loss import StaticPCALoss

# 添加kornia导入用于边缘检测
try:
    import kornia
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("警告: kornia未安装，边缘加权功能将不可用。请安装: pip install kornia")

from modeling.diffusion import create_diffusion
from modeling.modules.blocks import SimpleMLPAdaLN
from .perceptual_loss import PerceptualLoss, PatchBasedLPIPSLoss, PatchBasedGramLoss
from .discriminator import NLayerDiscriminator
from .discriminator_qwen import QwenVLDiscriminator
from elatentlpips import ELatentLPIPS
from utils.latent_utils import unpack_latent_from_chw4
from .discriminator_convnext import ConvNeXtV2Discriminator 
from .ema_model import EMAModel
from .decoder_lpips import DecoderLPIPSLoss


class KLWeightScheduler:
    """KL权重调度器，支持多种调度策略"""
    
    def __init__(self, scheduler_type="linear", start_weight=0.0, end_weight=1e-6, 
                 warmup_steps=1000, total_steps=100000):
        """
        Args:
            scheduler_type: 调度器类型，支持 "linear", "cosine", "exponential", "step"
            start_weight: 起始权重
            end_weight: 结束权重
            warmup_steps: 预热步数
            total_steps: 总步数
        """
        self.scheduler_type = scheduler_type
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
    def get_weight(self, step):
        """根据当前步数获取KL权重"""
        if step < self.warmup_steps:
            # 预热阶段：从0线性增长到end_weight
            # progress = step / self.warmup_steps
            # return self.start_weight + (self.end_weight - self.start_weight) * progress
            return 0
        else:
            # 根据调度器类型计算权重
            if self.scheduler_type == "linear":
                # 线性增长到end_weight
                progress = min(1.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
                return self.end_weight * progress
            elif self.scheduler_type == "cosine":
                # 余弦退火
                progress = min(1.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
                return self.end_weight * (1 + np.cos(np.pi * (1 - progress))) / 2
            elif self.scheduler_type == "exponential":
                # 指数增长
                progress = min(1.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
                return self.end_weight * (1 - np.exp(-5 * progress))
            elif self.scheduler_type == "step":
                # 阶梯式增长
                if step < self.warmup_steps + (self.total_steps - self.warmup_steps) // 2:
                    return self.end_weight * 0.5
                else:
                    return self.end_weight
            else:
                return self.end_weight

def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discrminator.

    This function is borrowed from
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py#L20
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor
) -> torch.Tensor:
    """Computes the LeCam loss for the given average real and fake logits.

    Args:
        logits_real_mean -> torch.Tensor: The average real logits.
        logits_fake_mean -> torch.Tensor: The average fake logits.
        ema_logits_real_mean -> torch.Tensor: The EMA of the average real logits.
        ema_logits_fake_mean -> torch.Tensor: The EMA of the average fake logits.

    Returns:
        lecam_loss -> torch.Tensor: The LeCam loss.
    """
    lecam_loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss

class ReconstructionLoss_Stage1(nn.Module):
    """Reconstruction loss for stage 1 training."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reconstruction_loss = config.losses.reconstruction_loss
        self.reconstruction_weight = config.losses.reconstruction_weight
        
        # 初始化感知损失权重
        self.perceptual_weight = config.losses.get("perceptual_weight", 1.0)
        self.gram_loss_weight = config.losses.get("gram_weight", 0.0)
        
        # 检查是否使用基于patch的感知损失
        use_patch_based_perceptual = config.losses.get("use_patch_based_perceptual", False)
        patch_based_config = config.losses.get("patch_based_config", {})
        
        if use_patch_based_perceptual:
            # 创建基础的感知损失（用于提取LPIPS和Gram模型）
            base_perceptual_loss = PerceptualLoss(config.losses.perceptual_loss).eval()
            
            # 创建基于patch的LPIPS损失
            if self.perceptual_weight > 0.0:
                self.patch_lpips_loss = PatchBasedLPIPSLoss(
                    lpips_model=base_perceptual_loss,
                    patch_size=patch_based_config.get("patch_size", 256),
                    stride=patch_based_config.get("stride", 128),
                    sampling=patch_based_config.get("sampling", "grid"),
                    num_random_patches=patch_based_config.get("num_random_patches")
                ).eval()
                print(f"启用基于patch的LPIPS损失: {self.patch_lpips_loss.get_patch_info()}")
            else:
                self.patch_lpips_loss = None
            
            # 创建基于patch的Gram损失
            if self.gram_loss_weight > 0.0:
                from .lpips import GramLoss
                gram_model = GramLoss().eval()
                self.patch_gram_loss = PatchBasedGramLoss(
                    gram_model=gram_model,
                    patch_size=patch_based_config.get("patch_size", 256),
                    stride=patch_based_config.get("stride", 128)
                ).eval()
                print(f"启用基于patch的Gram损失: {self.patch_gram_loss.get_patch_info()}")
            else:
                self.patch_gram_loss = None
                
            # 为了向后兼容，保留perceptual_loss属性
            self.perceptual_loss = None
        else:
            # 不使用patch-based时，也分别创建LPIPS和Gram损失
            base_perceptual_loss = PerceptualLoss(config.losses.perceptual_loss).eval()
            
            # 创建独立的LPIPS损失
            if self.perceptual_weight > 0.0:
                self.lpips_loss = base_perceptual_loss
                print("启用独立LPIPS损失")
            else:
                self.lpips_loss = None
            
            # 创建独立的Gram损失
            if self.gram_loss_weight > 0.0:
                from .lpips import GramLoss
                self.gram_loss = GramLoss().eval()
                print("启用独立Gram损失")
            else:
                self.gram_loss = None
            
            # 为了向后兼容，保留perceptual_loss属性
            self.perceptual_loss = base_perceptual_loss
            self.patch_lpips_loss = None
            self.patch_gram_loss = None
        
        self.pca_loss = StaticPCALoss(pca_basis_path=config.losses.get("pca_basis_path", "pca_basis_5000.pth"))

    def _compute_pca_loss(self, inputs, reconstructions):
        """
        计算PCA Loss并返回总loss和channel loss统计信息
        """
        if self.reconstruction_loss == "pca":
            total_loss, channel_losses = self.pca_loss(inputs, reconstructions)
            return total_loss, channel_losses
        else:
            return None, {}

    def forward(self, inputs, reconstructions, extra_result_dict):
        """Forward pass."""
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "pca":
            reconstruction_loss, pca_channel_losses = self.pca_loss(inputs, reconstructions)
        else:
            raise ValueError(f"Unsupported reconstruction_loss {self.reconstruction_loss}")

        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual losses.
        total_perceptual_loss = 0.0
        loss_dict = dict(
            total_loss=0.0,  # 将在后面更新
            reconstruction_loss=reconstruction_loss.detach(),
        )
        
        # 计算LPIPS损失
        if self.patch_lpips_loss is not None:
            lpips_loss = self.patch_lpips_loss(inputs, reconstructions)
            total_perceptual_loss += self.perceptual_weight * lpips_loss
            loss_dict["lpips_loss"] = (self.perceptual_weight * lpips_loss).detach()
        elif self.perceptual_loss is not None:
            perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()
            total_perceptual_loss += self.perceptual_weight * perceptual_loss
            loss_dict["perceptual_loss"] = (self.perceptual_weight * perceptual_loss).detach()
        
        # 计算Gram损失
        if self.patch_gram_loss is not None:
            gram_loss = self.patch_gram_loss(inputs, reconstructions)
            total_perceptual_loss += self.gram_loss_weight * gram_loss
            loss_dict["gram_loss"] = (self.gram_loss_weight * gram_loss).detach()

        # Compute total loss.
        total_loss = reconstruction_loss + total_perceptual_loss
        loss_dict["total_loss"] = total_loss.clone().detach()
        
        # 如果使用PCA Loss，添加channel loss统计信息
        if self.reconstruction_loss == "pca":
            loss_dict.update(pca_channel_losses)

        return total_loss, loss_dict


class ReconstructionLoss_Stage2(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        """Initializes the losses module.

        Args:
            config: A dictionary, the configuration for the model and everything else.
        """
        super().__init__()
        loss_config = config.losses
        discriminator_type = loss_config.get("discriminator_type", "nlayer")
        if discriminator_type == "nlayer":
            self.discriminator = NLayerDiscriminator(num_channels=3)
        elif discriminator_type == "qwen":
            use_lora = config.model.get("discriminator", {}).get("use_lora", False)
            self.discriminator = QwenVLDiscriminator(use_lora=use_lora)
        elif discriminator_type == "convnext":
            # 从配置中读取是否使用预训练权重的选项
            use_pretrained_weights = loss_config.get("use_pretrained_weights", True)
            # Get model size from config, default to "base"
            convnext_model_size = loss_config.get("convnext_model_size", "base")
            self.discriminator = ConvNeXtV2Discriminator(
                model_size=convnext_model_size,
                use_pretrained_weights=use_pretrained_weights
            )
        self.pca_loss = StaticPCALoss(pca_basis_path=config.losses.get("pca_basis_path", "pca_basis_5000.pth"))

        # Load discriminator weights if specified; default to experiment.init_weight when absent
        weights_path = None
        if hasattr(config, 'discriminator_weights_path') and config.discriminator_weights_path:
            weights_path = config.discriminator_weights_path
        elif hasattr(config, 'experiment') and getattr(config.experiment, 'init_weight', None):
            weights_path = config.experiment.init_weight

        if weights_path:
            try:
                print(f"Loading discriminator weights from: {weights_path}")
                raw_obj = torch.load(weights_path, map_location="cpu")
                # Support both wrapped and flat state dicts
                if isinstance(raw_obj, dict) and 'state_dict' in raw_obj:
                    loaded_sd = raw_obj['state_dict']
                else:
                    loaded_sd = raw_obj

                # Try direct load first (if it's already a pure discriminator state_dict)
                try:
                    missing, unexpected = self.discriminator.load_state_dict(loaded_sd, strict=False)
                    # Heuristic: if nothing matched, fall back to extracting prefixed keys
                    if isinstance(missing, list) and len(missing) > 0 and len(loaded_sd) > 0:
                        raise RuntimeError("Fallback to prefixed extraction")
                    print("Discriminator weights loaded successfully (direct).")
                except Exception:
                    # Extract keys starting with 'discriminator.' and strip the prefix
                    disc_only_sd = {}
                    for k, v in loaded_sd.items():
                        if isinstance(k, str) and k.startswith('discriminator.'):
                            disc_only_sd[k[len('discriminator.'):]] = v
                    if disc_only_sd:
                        self.discriminator.load_state_dict(disc_only_sd, strict=False)
                        print(f"Discriminator weights loaded successfully (extracted {len(disc_only_sd)} params).")
                    else:
                        print("No discriminator.* keys found in provided weights; skipped discriminator init.")
            except Exception as e:
                print(f"Warning: Failed to load discriminator weights from {weights_path}: {e}")

        # Initialize discriminator EMA
        self.use_discriminator_ema = loss_config.get("use_discriminator_ema", False)
        if self.use_discriminator_ema:
            self.discriminator_ema = EMAModel(
                self.discriminator.parameters(),
                decay=loss_config.get("discriminator_ema_decay", 0.999),
                update_after_step=loss_config.get("discriminator_ema_update_after_step", 0),
                update_every=loss_config.get("discriminator_ema_update_every", 1),
                use_ema_warmup=loss_config.get("discriminator_ema_warmup", True),
                inv_gamma=loss_config.get("discriminator_ema_inv_gamma", 1.0),
                power=loss_config.get("discriminator_ema_power", 2/3),
            )
            print(f"Initialized discriminator EMA with decay={loss_config.get('discriminator_ema_decay', 0.999)}")
        else:
            self.discriminator_ema = None

        self.reconstruction_loss = loss_config.reconstruction_loss
        self.reconstruction_weight = loss_config.reconstruction_weight
        # Optional for VAE (KL) mode: default to 0.0 when not provided
        self.quantizer_weight = loss_config.get("quantizer_weight", 0.0)
        
        # 初始化感知损失权重
        self.perceptual_weight = loss_config.get("perceptual_weight", 1.0)
        self.gram_loss_weight = loss_config.get("gram_loss_weight", 0.0)
        
        # 检查是否使用基于patch的感知损失
        use_patch_based_perceptual = loss_config.get("use_patch_based_perceptual", False)
        patch_based_config = loss_config.get("patch_based_config", {})
        
        if use_patch_based_perceptual:
            # 创建基础的感知损失（用于提取LPIPS和Gram模型）
            base_perceptual_loss = PerceptualLoss(loss_config.perceptual_loss).eval()
            
            # 创建基于patch的LPIPS损失
            if self.perceptual_weight > 0.0:
                self.patch_lpips_loss = PatchBasedLPIPSLoss(
                    lpips_model=base_perceptual_loss,
                    patch_size=patch_based_config.get("patch_size", 256),
                    stride=patch_based_config.get("stride", 128),
                    sampling=patch_based_config.get("sampling", "grid"),
                    num_random_patches=patch_based_config.get("num_random_patches")
                ).eval()
                print(f"启用基于patch的LPIPS损失: {self.patch_lpips_loss.get_patch_info()}")
            else:
                self.patch_lpips_loss = None
                self.lpips_loss = None
            
            # 创建基于patch的Gram损失
            if self.gram_loss_weight > 0.0:
                from .lpips import GramLoss
                gram_model = GramLoss().eval()
                self.patch_gram_loss = PatchBasedGramLoss(
                    gram_model=gram_model,
                    patch_size=patch_based_config.get("patch_size", 256),
                    stride=patch_based_config.get("stride", 128)
                ).eval()
                print(f"启用基于patch的Gram损失: {self.patch_gram_loss.get_patch_info()}")
            else:
                self.patch_gram_loss = None
                self.gram_loss = None
                
            # 为了向后兼容，保留perceptual_loss属性
            self.perceptual_loss = None
        else:
            # 不使用patch-based时，也分别创建LPIPS和Gram损失
            base_perceptual_loss = PerceptualLoss(loss_config.perceptual_loss).eval()
            
            # 创建独立的LPIPS损失
            if self.perceptual_weight > 0.0:
                self.lpips_loss = base_perceptual_loss
                print("启用独立LPIPS损失")
            else:
                self.lpips_loss = None
            
            # 创建独立的Gram损失
            if self.gram_loss_weight > 0.0:
                from .lpips import GramLoss
                self.gram_loss = GramLoss().eval()
                print("启用独立Gram损失")
            else:
                self.gram_loss = None
            
            # 为了向后兼容，保留perceptual_loss属性
            self.perceptual_loss = base_perceptual_loss
            self.patch_lpips_loss = None
            self.patch_gram_loss = None
        


        # 添加控制是否使用对抗损失的超参数
        self.use_adversarial_loss = loss_config.get("use_adversarial_loss", True)
        
        # 检查是否使用基于patch的对抗损失
        use_patch_based_adversarial = loss_config.get("use_patch_based_adversarial", False)
        if use_patch_based_adversarial and self.use_adversarial_loss:
            # 创建基于patch的对抗损失
            patch_adversarial_config = loss_config.get("patch_adversarial_config", {})
            self.patch_adversarial_loss = PatchBasedAdversarialLoss(
                discriminator=self.discriminator,
                patch_size=patch_adversarial_config.get("patch_size", 256),
                num_patches=patch_adversarial_config.get("num_patches", 8),
                min_patch_size=patch_adversarial_config.get("min_patch_size", 192),
                max_patch_size=patch_adversarial_config.get("max_patch_size", 384),
                use_random_patch_size=patch_adversarial_config.get("use_random_patch_size", True),
                ensure_valid_crop=patch_adversarial_config.get("ensure_valid_crop", True),
                patch_sampling_strategy=patch_adversarial_config.get("patch_sampling_strategy", "random"),
                grid_stride=patch_adversarial_config.get("grid_stride")
            )
            print(f"启用基于patch的对抗损失: {self.patch_adversarial_loss.get_patch_info()}")
        else:
            self.patch_adversarial_loss = None
        
        # 如果禁用对抗损失，使用默认值，否则从配置中读取
        if not self.use_adversarial_loss:
            print("警告: 已禁用对抗损失 (use_adversarial_loss=False)")
            self.discriminator_iter_start = 0
            self.discriminator_factor = 0.0
            self.discriminator_weight = 0.0
        else:
            self.discriminator_iter_start = loss_config.discriminator_start
            self.discriminator_factor = loss_config.discriminator_factor
            self.discriminator_weight = loss_config.discriminator_weight
        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.lecam_ema_decay = loss_config.get("lecam_ema_decay", 0.999)
        self.discriminator_warmup_steps = loss_config.get("discriminator_warmup_steps", 0)
        
        # 如果禁用对抗损失，将相关权重设为0
        if not self.use_adversarial_loss:
            self.lecam_regularization_weight = 0.0
        
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        self.config = config
        
        # 添加对quantize_mode的支持
        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        
        # 添加边缘加权重建损失的配置
        self.use_edge_weighted_loss = loss_config.get("use_edge_weighted_loss", False)
        if self.use_edge_weighted_loss:
            if not KORNIA_AVAILABLE:
                print("警告: 启用了边缘加权损失但kornia不可用，将回退到普通重建损失")
                self.use_edge_weighted_loss = False
            else:
                self.edge_weight_alpha = loss_config.get("edge_weight_alpha", 50.0)
                self.edge_weight_normalization = loss_config.get("edge_weight_normalization", "max")  # "max" or "mean"
                print(f"启用边缘加权重建损失: alpha={self.edge_weight_alpha}, normalization={self.edge_weight_normalization}")

    def update_discriminator_ema(self, global_step: int):
        """Update discriminator EMA if enabled."""
        if self.use_discriminator_ema and self.discriminator_ema is not None:
            # Ensure EMA model is on the same device as discriminator parameters
            discriminator_params = list(self.discriminator.parameters())
            if discriminator_params:
                target_device = discriminator_params[0].device
                self.discriminator_ema.to(target_device)
            self.discriminator_ema.set_step(global_step)
            self.discriminator_ema.step(self.discriminator.parameters())

    def use_ema_discriminator(self, use_ema: bool = True):
        """Switch between EMA and original discriminator parameters."""
        if self.use_discriminator_ema and self.discriminator_ema is not None:
            if use_ema:
                # Use EMA parameters
                self.discriminator_ema.copy_to(self.discriminator.parameters())
            else:
                # Restore original parameters (this would need to be implemented if needed)
                pass

    def compute_edge_weight_map(self, inputs: torch.Tensor) -> torch.Tensor:
        """计算基于Sobel边缘的权重图。
        
        Args:
            inputs: 输入图像，shape为(B, C, H, W)，值在[0, 1]范围内
            
        Returns:
            W: 权重图，shape为(B, C, H, W)
        """
        if not self.use_edge_weighted_loss or not KORNIA_AVAILABLE:
            # 如果不使用边缘加权或kornia不可用，返回全1权重
            return torch.ones_like(inputs)
        
        try:
            # 计算Sobel边缘图
            sobel_map = kornia.filters.sobel(inputs)  # 输出的是边缘强度
            
            # 创建权重图 W
            W = torch.zeros_like(inputs)
            
            # 确保在batch中分别处理
            for i in range(inputs.shape[0]):
                batch_sobel_map = sobel_map[i]
                
                # 防止除以0，并选择归一化方式
                if self.edge_weight_normalization == "max":
                    norm_factor = batch_sobel_map.max() + 1e-6
                elif self.edge_weight_normalization == "mean":
                    norm_factor = batch_sobel_map.mean() + 1e-6
                else:
                    norm_factor = batch_sobel_map.max() + 1e-6
                
                norm_sobel = batch_sobel_map / norm_factor
                W[i] = 1.0 + self.edge_weight_alpha * norm_sobel
            
            # detach()确保权重图不参与梯度计算
            return W.detach()
            
        except Exception as e:
            print(f"计算边缘权重图时出现错误: {e}，回退到普通权重")
            return torch.ones_like(inputs)

    def compute_loss_gradients(self, 
                              inputs: torch.Tensor,
                              reconstructions: torch.Tensor,
                              extra_result_dict: Mapping[Text, torch.Tensor],
                              global_step: int) -> Mapping[Text, torch.Tensor]:
        """计算不同loss对output pixel的梯度量级。
        
        Args:
            inputs: 输入图像
            reconstructions: 重建图像
            extra_result_dict: 额外的结果字典
            global_step: 全局步数
            
        Returns:
            gradient_dict: 包含各种loss梯度的字典
        """
  
        gradient_dict = {}
        
   


        # 创建完全独立的tensor副本，避免修改原始tensor
        inputs_detached = inputs.detach().clone()
        reconstructions_detached = reconstructions.detach().clone()
        
        # 设置梯度计算
        reconstructions_detached.requires_grad_(True)
        reconstructions_detached.retain_grad()
        
        try:
            # 1. 计算reconstruction loss的梯度
            if self.reconstruction_loss == "l1":
                reconstruction_loss = F.l1_loss(inputs_detached, reconstructions_detached, reduction="mean")
            elif self.reconstruction_loss == "l2":
                reconstruction_loss = F.mse_loss(inputs_detached, reconstructions_detached, reduction="mean")
            elif self.reconstruction_loss == "pca":
                reconstruction_loss, _ = self.pca_loss(inputs_detached, reconstructions_detached)
            else:
                raise ValueError(f"Unsupported reconstruction_loss {self.reconstruction_loss}")
            
            reconstruction_loss *= self.reconstruction_weight
            reconstruction_loss.backward(retain_graph=True)
            reconstruction_grad = reconstructions_detached.grad.clone()
            gradient_dict['reconstruction_grad'] = reconstruction_grad.detach()
            reconstructions_detached.grad.zero_()
            
            # 清理第一个loss的计算图
            del reconstruction_loss, reconstruction_grad
            
     
            
            if self.perceptual_weight > 0.0:
                if self.patch_lpips_loss is not None:
                    # 使用patch-based LPIPS损失
                    perceptual_loss = self.patch_lpips_loss(inputs_detached, reconstructions_detached)
                    perceptual_loss *= self.perceptual_weight
                    perceptual_loss.backward(retain_graph=True)
                    perceptual_grad = reconstructions_detached.grad.clone()
                    gradient_dict['perceptual_grad'] = perceptual_grad.detach()
                    reconstructions_detached.grad.zero_()
                    
                    # 清理第二个loss的计算图
                    del perceptual_loss, perceptual_grad
                elif self.perceptual_loss is not None:
                    # 使用传统perceptual损失
                    perceptual_loss = self.perceptual_loss(inputs_detached, reconstructions_detached).mean()
                    perceptual_loss *= self.perceptual_weight
                    perceptual_loss.backward(retain_graph=True)
                    perceptual_grad = reconstructions_detached.grad.clone()
                    gradient_dict['perceptual_grad'] = perceptual_grad.detach()
                    reconstructions_detached.grad.zero_()
                    
                    # 清理第二个loss的计算图
                    del perceptual_loss, perceptual_grad
                   
               
            
            # 3. 计算discriminator loss的梯度
            discriminator_factor = self.discriminator_factor
            if discriminator_factor > 0.0 and self.discriminator_weight > 0.0 and self.use_adversarial_loss:
                # 禁用discriminator的梯度
                for param in self.discriminator.parameters():
                    param.requires_grad = False
                logits_fake = self.discriminator(reconstructions_detached.float())
                generator_loss = -torch.mean(logits_fake)
                generator_loss *= self.discriminator_weight * discriminator_factor
                generator_loss.backward(retain_graph=True)
                discriminator_grad = reconstructions_detached.grad.clone()
                gradient_dict['discriminator_grad'] = discriminator_grad.detach()
                reconstructions_detached.grad.zero_()
                
                # 清理discriminator loss的计算图
                del logits_fake, generator_loss, discriminator_grad
            else:
                gradient_dict['discriminator_grad'] = torch.zeros_like(reconstructions_detached)
            
            # 4. 计算total loss的梯度（重新计算，避免使用已backward的tensor）
            if self.reconstruction_loss == "l1":
                reconstruction_loss_new = F.l1_loss(inputs_detached, reconstructions_detached, reduction="mean")
            elif self.reconstruction_loss == "l2":
                reconstruction_loss_new = F.mse_loss(inputs_detached, reconstructions_detached, reduction="mean")
            elif self.reconstruction_loss == "pca":
                reconstruction_loss_new, _ = self.pca_loss(inputs_detached, reconstructions_detached)
            else:
                raise ValueError(f"Unsupported reconstruction_loss {self.reconstruction_loss}")
            
            reconstruction_loss_new *= self.reconstruction_weight
            perceptual_loss_new = self.perceptual_loss(inputs_detached, reconstructions_detached).mean() * self.perceptual_weight
            
            if discriminator_factor > 0.0 and self.discriminator_weight > 0.0 and self.use_adversarial_loss:
                discriminator_output = self.discriminator(reconstructions_detached.float())
                generator_loss_new = -torch.mean(discriminator_output) * self.discriminator_weight * discriminator_factor
            else:
                generator_loss_new = torch.zeros((), device=inputs_detached.device)
            
            total_loss = reconstruction_loss_new + perceptual_loss_new + generator_loss_new
            total_loss.backward(retain_graph=True)
            total_grad = reconstructions_detached.grad.clone()
            gradient_dict['total_grad'] = total_grad.detach()
            
            # 清理total loss的计算图
            del reconstruction_loss_new, perceptual_loss_new, generator_loss_new, total_loss, total_grad
            if discriminator_factor > 0.0 and self.discriminator_weight > 0.0 and self.use_adversarial_loss:
                del discriminator_output
            
        finally:
            # 清理所有梯度
            if reconstructions_detached.grad is not None:
                reconstructions_detached.grad.zero_()
            
            # 清理所有模块的梯度
            for module in [self.discriminator, self.perceptual_loss]:
                if hasattr(module, 'zero_grad'):
                    module.zero_grad()
            
            # 清理所有参数的梯度
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            
            # 清理所有子模块的梯度
            for child in self.children():
                if hasattr(child, 'zero_grad'):
                    child.zero_grad()
            
            # 清理tensor引用
            del inputs_detached, reconstructions_detached
            
            # 清理计算图和缓存
            torch.cuda.empty_cache()
            
            # 强制垃圾回收
            import gc
            gc.collect()
        
        return gradient_dict

    def visualize_loss_gradients(self, 
                                inputs: torch.Tensor,
                                reconstructions: torch.Tensor,
                                extra_result_dict: Mapping[Text, torch.Tensor],
                                global_step: int,
                                save_path: str = None) -> None:
        """可视化不同loss对output pixel的梯度量级。
        
        Args:
            inputs: 输入图像
            reconstructions: 重建图像
            extra_result_dict: 额外的结果字典
            global_step: 全局步数
            save_path: 保存路径，如果为None则显示图像
        """
        # 只在主进程中执行梯度可视化，避免分布式训练中的重复计算
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                print(f"DEBUG: Process rank {rank}/{world_size} - {'SKIPPING' if rank != 0 else 'EXECUTING'} gradient visualization at step {global_step}")
                if rank != 0:
                    return
        except:
            # 如果没有分布式环境，继续执行
            print("DEBUG: No distributed environment detected, executing gradient visualization")
            pass

       # 根据VAE模式处理输入
        if (getattr(self, 'use_flux_vae', False) or getattr(self, 'use_sd3_vae', False)):
            flux_vae_mode = getattr(self, 'flux_vae_mode', 'latent_to_latent')
            # 需要将reconstructions解码为pixel
            with torch.no_grad():
                inputs = self.original_vae_model.decode(inputs.detach())
                if flux_vae_mode == "latent_to_latent":
                    reconstructions = self.original_vae_model.decode(reconstructions.detach())
                elif flux_vae_mode == "latent_to_pixel":
                    reconstructions = reconstructions.detach()
        else:
            # 没有使用VAE，直接使用原始输入
            inputs = inputs.detach()
            reconstructions = reconstructions.detach()
        try:
            gradient_dict = self.compute_loss_gradients(inputs, reconstructions, extra_result_dict, global_step)
            
            # 计算梯度的L2范数
            grad_magnitudes = {}
            for loss_name, grad in gradient_dict.items():
                # 计算每个像素位置的梯度幅度
                grad_magnitude = torch.norm(grad, dim=1, keepdim=True)  # 在channel维度上计算L2范数
                # 转换为float32以避免BFloat16的numpy转换问题
                grad_magnitudes[loss_name] = grad_magnitude.detach().cpu().float().numpy()
            
            # 创建可视化
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            fig.suptitle(f'Loss Gradients Magnitude (Step {global_step})', fontsize=16)
            
            # 显示原始图像和重建图像（归一化到[0,1]范围）
            input_img = inputs[0].permute(1, 2, 0).detach().cpu().float().numpy()
            input_img = np.clip(input_img, 0, 1)  # 确保在[0,1]范围内
            axes[0, 0].imshow(input_img)
            axes[0, 0].set_title('Input Image')
            axes[0, 0].axis('off')
            
            recon_img = reconstructions[0].permute(1, 2, 0).detach().cpu().float().numpy()
            recon_img = np.clip(recon_img, 0, 1)  # 确保在[0,1]范围内
            axes[0, 1].imshow(recon_img)
            axes[0, 1].set_title('Reconstruction')
            axes[0, 1].axis('off')
            
            # 显示梯度幅度 - 只显示实际存在的梯度
            available_grads = list(grad_magnitudes.keys())
            print(f"Available gradients: {available_grads}")
            
            # 显示可用的梯度
            for i, grad_name in enumerate(available_grads[:3]):  # 最多显示3个梯度
                if i < 3:  # 确保不超出axes范围
                    grad_mag = grad_magnitudes[grad_name][0, 0]  # 取第一个batch，第一个channel
                    im = axes[1, i].imshow(grad_mag, cmap='hot', interpolation='nearest')
                    axes[1, i].set_title(f'{grad_name.replace("_grad", "").title()} Loss Grad')
                    axes[1, i].axis('off')
                    plt.colorbar(im, ax=axes[1, i])
            
            # 如果梯度数量少于3个，隐藏多余的axes
            for i in range(len(available_grads), 3):
                axes[1, i].axis('off')
            
            # 显示总梯度幅度（如果存在）
            if 'total_grad' in grad_magnitudes:
                total_grad_mag = grad_magnitudes['total_grad'][0, 0]
                im = axes[0, 2].imshow(total_grad_mag, cmap='hot', interpolation='nearest')
                axes[0, 2].set_title('Total Loss Grad')
                axes[0, 2].axis('off')
                plt.colorbar(im, ax=axes[0, 2])
            else:
                axes[0, 2].axis('off')
            
            # 显示梯度幅度的统计信息
            if available_grads:
                grad_names = [name.replace('_grad', '').title() for name in available_grads]
                grad_means = [np.mean(grad_magnitudes[name]) for name in available_grads]
                axes[2, 0].bar(grad_names, grad_means)
                axes[2, 0].set_title('Average Gradient Magnitude')
                axes[2, 0].set_ylabel('Magnitude')
                # 旋转x轴标签以避免重叠
                axes[2, 0].tick_params(axis='x', rotation=45)
            
            # 显示梯度幅度的分布
            if available_grads:
                for grad_name in available_grads:
                    grad_mag_flat = grad_magnitudes[grad_name].flatten()
                    axes[2, 1].hist(grad_mag_flat, bins=50, alpha=0.5, label=grad_name.replace('_grad', ''))
                axes[2, 1].set_title('Gradient Magnitude Distribution')
                axes[2, 1].set_xlabel('Magnitude')
                axes[2, 1].set_ylabel('Frequency')
                axes[2, 1].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"梯度可视化已保存到: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
            # 清理内存
            del gradient_dict, grad_magnitudes, input_img, recon_img
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"梯度可视化过程中出现错误: {e}")
            # 确保清理内存
            torch.cuda.empty_cache()
            if 'plt' in locals():
                plt.close('all')

    def log_gradient_statistics(self, 
                               inputs: torch.Tensor,
                               reconstructions: torch.Tensor,
                               extra_result_dict: Mapping[Text, torch.Tensor],
                               global_step: int) -> Mapping[Text, float]:
        """记录梯度统计信息，用于tensorboard等日志系统。
        
        Args:
            inputs: 输入图像
            reconstructions: 重建图像
            extra_result_dict: 额外的结果字典
            global_step: 全局步数
            
        Returns:
            stats_dict: 包含梯度统计信息的字典
        """
        try:
            gradient_dict = self.compute_loss_gradients(inputs, reconstructions, extra_result_dict, global_step)
            
            stats_dict = {}
            for loss_name, grad in gradient_dict.items():
                grad_magnitude = torch.norm(grad, dim=1)  # 在channel维度上计算L2范数
                stats_dict[f'{loss_name}_mean'] = grad_magnitude.mean().item()
                stats_dict[f'{loss_name}_std'] = grad_magnitude.std().item()
                stats_dict[f'{loss_name}_max'] = grad_magnitude.max().item()
                stats_dict[f'{loss_name}_min'] = grad_magnitude.min().item()
            
            # 清理内存
            del gradient_dict, grad_magnitude
            torch.cuda.empty_cache()
            
            return stats_dict
            
        except Exception as e:
            print(f"计算梯度统计信息时出现错误: {e}")
            # 返回空的统计信息
            return {
                'reconstruction_grad_mean': 0.0,
                'reconstruction_grad_std': 0.0,
                'reconstruction_grad_max': 0.0,
                'reconstruction_grad_min': 0.0,
                'perceptual_grad_mean': 0.0,
                'perceptual_grad_std': 0.0,
                'perceptual_grad_max': 0.0,
                'perceptual_grad_min': 0.0,
                'discriminator_grad_mean': 0.0,
                'discriminator_grad_std': 0.0,
                'discriminator_grad_max': 0.0,
                'discriminator_grad_min': 0.0,
                'total_grad_mean': 0.0,
                'total_grad_std': 0.0,
                'total_grad_max': 0.0,
                'total_grad_min': 0.0,
            }

    @autocast("cuda", enabled=False)
    def forward(self,
                inputs: torch.Tensor,
                reconstructions: torch.Tensor,
                extra_result_dict: Mapping[Text, torch.Tensor],
                global_step: int,
                mode: str = "generator",
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # Both inputs and reconstructions are in range [0, 1].
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, extra_result_dict, global_step)
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")
   
    def should_discriminator_be_trained(self, global_step : int):
        """检查是否应该训练判别器。
        
        这个方法只控制判别器本身的训练，不影响生成器训练中的判别器损失。
        生成器训练时，判别器损失始终参与，以确保生成器能够学习生成判别器认为真实的图像。
        
        Args:
            global_step: 当前全局步数
            
        Returns:
            bool: 如果应该训练判别器则返回True，否则返回False
        """
        # 如果禁用对抗损失，则不训练判别器
        if not self.use_adversarial_loss:
            return False
        return global_step >= self.discriminator_iter_start

    def is_discriminator_warmup_period(self, global_step: int) -> bool:
        """Check if we're in the discriminator warmup period.
        
        During warmup, only discriminator is trained, generator is frozen.
        
        Returns:
            bool: True if in warmup period, False otherwise
        """
        if not self.should_discriminator_be_trained(global_step):
            return False
        
        warmup_steps = getattr(self, 'discriminator_warmup_steps', 0)
        if warmup_steps <= 0:
            return False
            
        warmup_start = self.discriminator_iter_start
        warmup_end = warmup_start + warmup_steps
        
        return warmup_start <= global_step < warmup_end

    def get_discriminator_training_frequency(self, global_step: int) -> int:
        """Get discriminator training frequency based on global step.
        
        Returns:
            int: How many generator steps per discriminator step.
                1 means train discriminator every step.
                2 means train discriminator every 2 steps.
        """
        # 如果禁用对抗损失，则不训练判别器
        if not self.use_adversarial_loss:
            return 0  # Don't train discriminator
        
        if global_step < self.discriminator_iter_start:
            return 0  # Don't train discriminator
        
        # Check if progressive training is enabled
        progressive_training = getattr(self.config.losses, 'progressive_discriminator_training', False)
        if not progressive_training:
            return 1  # Default: train discriminator every step
        
        # Get stage boundaries from config, with defaults
        stage1_steps = getattr(self.config.losses, 'discriminator_frequency_stage1_steps', 10_000)
        stage2_steps = getattr(self.config.losses, 'discriminator_frequency_stage2_steps', 50_000)
        
        # Progressive training: start with more discriminator training, then reduce
        if global_step < self.discriminator_iter_start + stage1_steps:
            return 1  # Train discriminator every step initially
        elif global_step < self.discriminator_iter_start + stage2_steps:
            return 5  # Train discriminator every 2 steps
        else:
            return 10  # Train discriminator every 3 steps (default GAN ratio)

    def _forward_generator(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        # Skip generator training during discriminator warmup period
        if self.is_discriminator_warmup_period(global_step):
            # Return zero loss during warmup, but still compute reconstructions for discriminator
            total_loss = torch.zeros((), device=inputs.device)
            
            # 根据quantize_mode创建不同的loss_dict
            if self.quantize_mode == "direct":
                loss_dict = dict(
                    total_loss=total_loss.clone().detach(),
                    reconstruction_loss=torch.zeros((), device=inputs.device).detach(),
                    perceptual_loss=torch.zeros((), device=inputs.device).detach(),
                    weighted_gan_loss=torch.zeros((), device=inputs.device).detach(),
                    discriminator_factor=torch.tensor(0.0, device=inputs.device),
                    d_weight=torch.tensor(0.0, device=inputs.device),
                    gan_loss=torch.zeros((), device=inputs.device).detach(),
                    in_warmup=torch.tensor(True, device=inputs.device),
                )
            else:
                # 对于vq和vae模式，包含量化相关的损失
                loss_dict = dict(
                    total_loss=total_loss.clone().detach(),
                    reconstruction_loss=torch.zeros((), device=inputs.device).detach(),
                    perceptual_loss=torch.zeros((), device=inputs.device).detach(),
                    quantizer_loss=torch.zeros((), device=inputs.device).detach(),
                    weighted_gan_loss=torch.zeros((), device=inputs.device).detach(),
                    discriminator_factor=torch.tensor(0.0, device=inputs.device),
                    commitment_loss=extra_result_dict.get("commitment_loss", torch.zeros((), device=inputs.device)).detach(),
                    codebook_loss=extra_result_dict.get("codebook_loss", torch.zeros((), device=inputs.device)).detach(),
                    d_weight=torch.tensor(0.0, device=inputs.device),
                    gan_loss=torch.zeros((), device=inputs.device).detach(),
                    in_warmup=torch.tensor(True, device=inputs.device),
                )
            return total_loss, loss_dict

        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        
        # 计算边缘权重图（如果启用）
        edge_weights = self.compute_edge_weight_map(inputs)
        
        # 计算重建损失，支持边缘加权和PCA损失
        if self.reconstruction_loss == "l1":
            if self.use_edge_weighted_loss:
                # 加权L1损失
                pixel_wise_l1 = torch.abs(inputs - reconstructions)
                reconstruction_loss = (edge_weights * pixel_wise_l1).mean()
            else:
                reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            if self.use_edge_weighted_loss:
                # 加权L2损失（统一到float32，避免与bf16混用）
                pixel_wise_mse = (inputs - reconstructions) ** 2
                reconstruction_loss = (edge_weights.float() * pixel_wise_mse).mean()
            else:
                reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "pca":
            # PCA损失
            reconstruction_loss, pca_channel_losses = self.pca_loss(inputs, reconstructions)
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {self.reconstruction_loss}")
        
        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual losses.
        total_perceptual_loss = 0.0
        perceptual_loss_dict = {}
        
        # 计算LPIPS损失
        if self.patch_lpips_loss is not None:
            # 使用patch-based LPIPS损失
            lpips_loss = self.patch_lpips_loss(inputs, reconstructions)
            total_perceptual_loss += self.perceptual_weight * lpips_loss
            perceptual_loss_dict["lpips_loss"] = (self.perceptual_weight * lpips_loss).detach()
        elif self.lpips_loss is not None:
            # 使用独立的LPIPS损失
            lpips_loss = self.lpips_loss(inputs, reconstructions).mean()
            total_perceptual_loss += self.perceptual_weight * lpips_loss
            perceptual_loss_dict["lpips_loss"] = (self.perceptual_weight * lpips_loss).detach()
        elif self.perceptual_loss is not None:
            # 使用传统感知损失（向后兼容）
            perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()
            total_perceptual_loss += self.perceptual_weight * perceptual_loss
            perceptual_loss_dict["perceptual_loss"] = (self.perceptual_weight * perceptual_loss).detach()
        
        # 计算Gram损失
        if self.patch_gram_loss is not None:
            # 使用patch-based Gram损失
            gram_loss = self.patch_gram_loss(inputs, reconstructions)
            total_perceptual_loss += self.gram_loss_weight * gram_loss
            perceptual_loss_dict["gram_loss"] = (self.gram_loss_weight * gram_loss).detach()
        elif self.gram_loss is not None:
            # 使用独立的Gram损失
            gram_loss = self.gram_loss(inputs, reconstructions)
            total_perceptual_loss += self.gram_loss_weight * gram_loss
            perceptual_loss_dict["gram_loss"] = (self.gram_loss_weight * gram_loss).detach()

        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=inputs.device)
        # 生成器训练时，判别器损失应该始终参与，不受should_discriminator_be_trained影响
        discriminator_factor = self.discriminator_factor
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0 and self.use_adversarial_loss:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            
            # 使用基于patch的对抗损失或标准对抗损失
            if hasattr(self, 'patch_adversarial_loss') and self.patch_adversarial_loss is not None:
                generator_loss = self.patch_adversarial_loss(reconstructions, mode="generator")
            else:
                logits_fake = self.discriminator(reconstructions.float())
                generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        # Compute quantizer loss based on quantize_mode
        quantize_mode = getattr(self, 'quantize_mode', 'vq')  # Default to vq for backward compatibility
        
        # 检查是否跳过编码器
        skip_encoder = getattr(self, 'skip_encoder', False)
        
        if quantize_mode == "vq" and not skip_encoder:
            # Compute quantizer loss.
            quantizer_loss = extra_result_dict["quantizer_loss"]
            total_loss = (
                reconstruction_loss
                + total_perceptual_loss
                + self.quantizer_weight * quantizer_loss
                + d_weight * discriminator_factor * generator_loss
            )
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=total_perceptual_loss.detach(),
                quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor, device=inputs.device),
                commitment_loss=extra_result_dict["commitment_loss"].detach(),
                codebook_loss=extra_result_dict["codebook_loss"].detach(),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
                in_warmup=torch.tensor(False, device=inputs.device),
            )
            # 添加感知损失的详细信息
            loss_dict.update(perceptual_loss_dict)
            
            # 添加边缘权重相关的信息
            if self.use_edge_weighted_loss:
                edge_weights = self.compute_edge_weight_map(inputs)
                loss_dict["edge_weight_mean"] = edge_weights.mean().detach()
                loss_dict["edge_weight_max"] = edge_weights.max().detach()
                loss_dict["edge_weight_min"] = edge_weights.min().detach()
            
            # 如果使用PCA Loss，添加channel loss统计信息
            if self.reconstruction_loss == "pca":
                loss_dict.update(pca_channel_losses)
        elif quantize_mode == "direct" or skip_encoder:
            # Direct mode 或 skip_encoder: 没有量化，只有重建损失和感知损失
            total_loss = (
                reconstruction_loss
                + total_perceptual_loss
                + d_weight * discriminator_factor * generator_loss

            )
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=total_perceptual_loss.detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor, device=inputs.device),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
                in_warmup=torch.tensor(False, device=inputs.device),
            )
            # 添加感知损失的详细信息
            loss_dict.update(perceptual_loss_dict)
            
            # 添加边缘权重相关的信息
            if self.use_edge_weighted_loss:
                edge_weights = self.compute_edge_weight_map(inputs)
                loss_dict["edge_weight_mean"] = edge_weights.mean().detach()
                loss_dict["edge_weight_max"] = edge_weights.max().detach()
                loss_dict["edge_weight_min"] = edge_weights.min().detach()
            
            # 如果使用PCA Loss，添加channel loss统计信息
            if self.reconstruction_loss == "pca":
                loss_dict.update(pca_channel_losses)
        else:
            # For other modes (vae), let subclasses handle them
            raise NotImplementedError(f"Unsupported quantize_mode: {quantize_mode} in ReconstructionLoss_Stage2")

        return total_loss, loss_dict

    def _forward_discriminator(self,
                               inputs: torch.Tensor,
                               reconstructions: torch.Tensor,
                               global_step: int,
                               ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Discrminator training step."""
        # 判别器训练时，只有在should_discriminator_be_trained为True时才计算损失
        if not self.should_discriminator_be_trained(global_step):
            # 如果不需要训练判别器，返回零损失
            discriminator_loss = torch.zeros((), device=inputs.device)
            loss_dict = dict(
                discriminator_loss=discriminator_loss.detach(),
                logits_real=torch.zeros((), device=inputs.device),
                logits_fake=torch.zeros((), device=inputs.device),
                lecam_loss=torch.zeros((), device=inputs.device),
            )
            return discriminator_loss, loss_dict
            
        loss_dict = {}
        # Turn the gradients on.
        for param in self.discriminator.parameters():
            param.requires_grad = True

        # 根据VAE模式处理输入
        if (getattr(self, 'use_flux_vae', False) or getattr(self, 'use_sd3_vae', False)):
            flux_vae_mode = getattr(self, 'flux_vae_mode', 'latent_to_latent')
            # 需要将reconstructions解码为pixel
            with torch.no_grad():
                real_images = self.original_vae_model.decode(inputs.detach())
                if flux_vae_mode == "latent_to_latent":
                    fake_images = self.original_vae_model.decode(reconstructions.detach())
                elif flux_vae_mode == "latent_to_pixel":
                    fake_images = reconstructions.detach()
        else:
            # 没有使用VAE，直接使用原始输入
            real_images = inputs.detach()
            fake_images = reconstructions.detach()
            

        logits_real = self.discriminator(real_images.float())
        logits_fake = self.discriminator(fake_images.float())

        discriminator_loss = self.discriminator_factor * hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)

        # optional lecam regularization
        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                torch.mean(logits_real),
                torch.mean(logits_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            ) * self.lecam_regularization_weight

            # Ensure decay values are on the same device as the tensors
            decay_tensor = torch.tensor(self.lecam_ema_decay, device=self.ema_real_logits_mean.device, dtype=self.ema_real_logits_mean.dtype)
            one_minus_decay_tensor = torch.tensor(1 - self.lecam_ema_decay, device=self.ema_real_logits_mean.device, dtype=self.ema_real_logits_mean.dtype)
            
            self.ema_real_logits_mean = self.ema_real_logits_mean * decay_tensor + torch.mean(logits_real).detach() * one_minus_decay_tensor
            self.ema_fake_logits_mean = self.ema_fake_logits_mean * decay_tensor + torch.mean(logits_fake).detach() * one_minus_decay_tensor
        
        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )
        
        # Add EMA-related logging
        if self.use_discriminator_ema and self.discriminator_ema is not None:
            loss_dict["discriminator_ema_decay"] = torch.tensor(self.discriminator_ema.cur_decay_value or 0.0, device=inputs.device)
        
        return discriminator_loss, loss_dict


class ReconstructionLoss_Single_Stage(ReconstructionLoss_Stage2):
    def __init__(
        self,
        config
    ):
        super().__init__(config)
        loss_config = config.losses
        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        self.skip_encoder = config.model.vq_model.get("skip_encoder", False)
        
        if self.quantize_mode == "vae":
            # 基础KL权重
            self.base_kl_weight = loss_config.get("kl_weight", 1e-6)
            logvar_init = loss_config.get("logvar_init", 0.0)
            self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init, requires_grad=False)
            
            # KL权重调度器配置
            kl_scheduler_config = loss_config.get("kl_weight_scheduler", {})
            if kl_scheduler_config.get("enabled", False):
                # 创建KL权重调度器
                scheduler_type = kl_scheduler_config.get("type", "linear")
                start_weight = kl_scheduler_config.get("start_weight", 0.0)
                end_weight = kl_scheduler_config.get("end_weight", self.base_kl_weight)
                warmup_steps = kl_scheduler_config.get("warmup_steps", 2000)
                total_steps = kl_scheduler_config.get("total_steps", 100000)
                
                self.kl_weight_scheduler = KLWeightScheduler(
                    scheduler_type=scheduler_type,
                    start_weight=start_weight,
                    end_weight=end_weight,
                    warmup_steps=warmup_steps,
                    total_steps=total_steps
                )
                self.use_kl_scheduler = True
            else:
                # 不使用调度器，使用固定权重
                self.kl_weight = self.base_kl_weight
                self.use_kl_scheduler = False
        alignment_cfg = loss_config.get("encoder_alignment_loss", {})
        self.encoder_alignment_weight = alignment_cfg.get("weight", 0.0)
        self.encoder_alignment_dist_margin = alignment_cfg.get("distmat_margin", 0.0)
        self.encoder_alignment_cos_margin = alignment_cfg.get("cos_margin", 0.0)
        self.encoder_alignment_dist_weight = alignment_cfg.get("distmat_weight", 1.0)
        self.encoder_alignment_cos_weight = alignment_cfg.get("cos_weight", 1.0)
        self.encoder_alignment_max_positions = alignment_cfg.get("max_positions")
        if self.encoder_alignment_max_positions is not None:
            self.encoder_alignment_max_positions = int(self.encoder_alignment_max_positions)
        self.encoder_alignment_detach_cond = alignment_cfg.get("detach_lq_cond", True)
        self.encoder_alignment_use_dist = alignment_cfg.get("enable_dist_term", True)
        self.encoder_alignment_use_cos = alignment_cfg.get("enable_cos_term", True)
        self.encoder_alignment_eps = alignment_cfg.get("eps", 1e-6)
        self.encoder_alignment_enabled = self.encoder_alignment_weight > 0.0
        self.encoder_alignment_mse_weight = alignment_cfg.get("encoder_alignment_mse_weight", 1.0)
        # 控制在method == 'mean'时是否启用MSE对齐
        self.encoder_alignment_use_mse_mean = alignment_cfg.get("enable_mse_mean", True)
        self._encoder_alignment_warned = False
        # Debug flag removed after verification
        self.encoder_alignment_debug = False

    def _compute_encoder_alignment_loss(self, extra_result_dict: Mapping[Text, torch.Tensor]):
        _is_main = False
        if not self.encoder_alignment_enabled:
            return None
        if not isinstance(extra_result_dict, Mapping):
            return None
        encoder_hidden = extra_result_dict.get("encoder_hidden_spatial")
        lq_cond = extra_result_dict.get("lq_cond_spatial")
        alignment_method = extra_result_dict.get("encoder_alignment_method", None)
        if encoder_hidden is None or lq_cond is None:
            return None
        if encoder_hidden.ndim != 4 or lq_cond.ndim != 4:
            return None
        if encoder_hidden.shape[0] != lq_cond.shape[0]:
            return None
        if encoder_hidden.shape[-2:] != lq_cond.shape[-2:]:
            if not self._encoder_alignment_warned:
                print("encoder_alignment_loss skipped due to spatial mismatch between encoder hidden states and lq_cond")
                self._encoder_alignment_warned = True
            return None
        if self.encoder_alignment_detach_cond:
            lq_cond = lq_cond.detach()
        lq_cond = lq_cond.to(dtype=encoder_hidden.dtype)

        # If method == 'mean', use pure spatial MSE alignment and return
        if isinstance(alignment_method, str) and alignment_method.lower() == 'mean':
            # 如果配置关闭了mean模式下的MSE对齐，则跳过
            if not getattr(self, "encoder_alignment_use_mse_mean", True):
                if not self._encoder_alignment_warned:
                    print("encoder_alignment_loss 'mean' method disabled by config (enable_mse_mean=False)")
                    self._encoder_alignment_warned = True
                return None
            if encoder_hidden.shape[1] != lq_cond.shape[1]:
                if not self._encoder_alignment_warned:
                    print(f"encoder_alignment_loss mse (mean) skipped due to channel mismatch: {encoder_hidden.shape[1]} vs {lq_cond.shape[1]}")
                    self._encoder_alignment_warned = True
                return None
            mse_loss = F.mse_loss(encoder_hidden, lq_cond)
            mse_w = getattr(self, "encoder_alignment_mse_weight", 1.0)
            weighted_mse = mse_w * mse_loss
            logs = {
                "encoder_alignment_mse_loss": weighted_mse.to(encoder_hidden.dtype),
                "encoder_alignment_mse_raw": mse_loss.to(encoder_hidden.dtype),
            }
            return weighted_mse.to(encoder_hidden.dtype), logs
        hidden_flat = rearrange(encoder_hidden, "b c h w -> b c (h w)")
        cond_flat = rearrange(lq_cond, "b c h w -> b c (h w)")
        if self.encoder_alignment_max_positions is not None and hidden_flat.shape[-1] > self.encoder_alignment_max_positions:
            idx = torch.randperm(hidden_flat.shape[-1], device=hidden_flat.device)[: self.encoder_alignment_max_positions]
            hidden_flat = hidden_flat[:, :, idx]
            cond_flat = cond_flat[:, :, idx]
        hidden_norm = F.normalize(hidden_flat.float(), dim=1, eps=self.encoder_alignment_eps)
        cond_norm = F.normalize(cond_flat.float(), dim=1, eps=self.encoder_alignment_eps)
        total_loss = hidden_norm.new_zeros(())
        raw_loss = hidden_norm.new_zeros(())
        logs = {}
        if self.encoder_alignment_use_dist and self.encoder_alignment_dist_weight != 0.0:
            hidden_cos = torch.einsum("bci,bcj->bij", hidden_norm, hidden_norm)
            cond_cos = torch.einsum("bci,bcj->bij", cond_norm, cond_norm)
            diff = torch.abs(hidden_cos - cond_cos)
            dist_loss = F.relu(diff - self.encoder_alignment_dist_margin).mean()
            weighted_dist = self.encoder_alignment_dist_weight * dist_loss
            total_loss = total_loss + weighted_dist
            raw_loss = raw_loss + dist_loss
            logs["encoder_alignment_dist_loss"] = weighted_dist
            logs["encoder_alignment_dist_raw"] = dist_loss
        if self.encoder_alignment_use_cos and self.encoder_alignment_cos_weight != 0.0:
            if hidden_flat.shape[1] != cond_flat.shape[1]:
                if not self._encoder_alignment_warned:
                    print(f"encoder_alignment_loss cosine term skipped due to channel mismatch: {hidden_flat.shape[1]} vs {cond_flat.shape[1]}")
                    self._encoder_alignment_warned = True
            else:
                cos_sim = F.cosine_similarity(cond_flat, hidden_flat, dim=1)
                cos_loss = F.relu(1 - self.encoder_alignment_cos_margin - cos_sim).mean()
                weighted_cos = self.encoder_alignment_cos_weight * cos_loss
                total_loss = total_loss + weighted_cos
                raw_loss = raw_loss + cos_loss
                logs["encoder_alignment_cos_loss"] = weighted_cos
                logs["encoder_alignment_cos_raw"] = cos_loss
        if not logs:
            return None
        logs["encoder_alignment_raw"] = raw_loss.to(encoder_hidden.dtype)
        return total_loss.to(encoder_hidden.dtype), logs

    def get_kl_weight(self, global_step):
        """获取当前步数的KL权重"""
        if self.use_kl_scheduler:
            return self.kl_weight_scheduler.get_weight(global_step)
        else:
            return self.kl_weight

    def _forward_generator(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        
        # 计算边缘权重图（如果启用）
        edge_weights = self.compute_edge_weight_map(inputs)
        
        # 计算重建损失，支持边缘加权和PCA损失
        if self.reconstruction_loss == "l1":
            if self.use_edge_weighted_loss:
                # 加权L1损失
                pixel_wise_l1 = torch.abs(inputs - reconstructions)
                reconstruction_loss = (edge_weights * pixel_wise_l1).mean()
            else:
                reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            if self.use_edge_weighted_loss:
                # 加权L2损失（统一到float32，避免与bf16混用）
                pixel_wise_mse = (inputs - reconstructions) ** 2
                reconstruction_loss = (pixel_wise_mse * edge_weights.float()).mean()
            else:
                reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "pca":
            # PCA损失
            reconstruction_loss, pca_channel_losses = self.pca_loss(inputs, reconstructions)
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {self.reconstruction_loss}")
        
        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual losses.
        total_perceptual_loss = 0.0
        perceptual_loss_dict = {}
        
        # 计算LPIPS损失
        if self.patch_lpips_loss is not None:
            # 使用patch-based LPIPS损失
            lpips_loss = self.patch_lpips_loss(inputs, reconstructions)
            total_perceptual_loss += self.perceptual_weight * lpips_loss
            perceptual_loss_dict["lpips_loss"] = (self.perceptual_weight * lpips_loss).detach()
        elif self.lpips_loss is not None:
            # 使用独立的LPIPS损失
            lpips_loss = self.lpips_loss(inputs, reconstructions).mean()
            total_perceptual_loss += self.perceptual_weight * lpips_loss
            perceptual_loss_dict["lpips_loss"] = (self.perceptual_weight * lpips_loss).detach()
        elif self.perceptual_loss is not None:
            # 使用传统感知损失（向后兼容）
            perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()
            total_perceptual_loss += self.perceptual_weight * perceptual_loss
            perceptual_loss_dict["perceptual_loss"] = (self.perceptual_weight * perceptual_loss).detach()
        
        # 计算Gram损失
        if self.patch_gram_loss is not None:
            # 使用patch-based Gram损失
            gram_loss = self.patch_gram_loss(inputs, reconstructions)
            total_perceptual_loss += self.gram_loss_weight * gram_loss
            perceptual_loss_dict["gram_loss"] = (self.gram_loss_weight * gram_loss).detach()
        elif self.gram_loss is not None:
            # 使用独立的Gram损失
            gram_loss = self.gram_loss(inputs, reconstructions)
            total_perceptual_loss += self.gram_loss_weight * gram_loss
            perceptual_loss_dict["gram_loss"] = (self.gram_loss_weight * gram_loss).detach()

        encoder_alignment_loss = None
        encoder_alignment_logs = {}
        weighted_alignment = None
        if self.encoder_alignment_enabled:
            alignment_result = self._compute_encoder_alignment_loss(extra_result_dict)
            if alignment_result is not None:
                encoder_alignment_loss, encoder_alignment_logs = alignment_result
                weighted_alignment = self.encoder_alignment_weight * encoder_alignment_loss

        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=inputs.device)
        # 生成器训练时，判别器损失应该始终参与，不受should_discriminator_be_trained影响
        discriminator_factor = self.discriminator_factor
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0 and self.use_adversarial_loss:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            
            # 使用基于patch的对抗损失或标准对抗损失
            if hasattr(self, 'patch_adversarial_loss') and self.patch_adversarial_loss is not None:
                generator_loss = self.patch_adversarial_loss(reconstructions, mode="generator")
            else:
                logits_fake = self.discriminator(reconstructions.float())
                generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        if self.quantize_mode == "vq":
            # Compute quantizer loss.
            quantizer_loss = extra_result_dict["quantizer_loss"]
            total_loss = (
                reconstruction_loss
                + total_perceptual_loss
                + self.quantizer_weight * quantizer_loss
                + d_weight * discriminator_factor * generator_loss
            )
            if weighted_alignment is not None:
                total_loss = total_loss + weighted_alignment
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=total_perceptual_loss.detach(),
                quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor, device=inputs.device),
                commitment_loss=extra_result_dict["commitment_loss"].detach(),
                codebook_loss=extra_result_dict["codebook_loss"].detach(),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
            if weighted_alignment is not None:
                loss_dict["encoder_alignment_loss"] = weighted_alignment.detach()
                for key, value in encoder_alignment_logs.items():
                    if isinstance(value, torch.Tensor):
                        loss_dict[key] = value.detach()
                    else:
                        loss_dict[key] = value

            # 添加感知损失的详细信息
            loss_dict.update(perceptual_loss_dict)
            
            # 添加边缘权重相关的信息
            if self.use_edge_weighted_loss:
                edge_weights = self.compute_edge_weight_map(inputs)
                loss_dict["edge_weight_mean"] = edge_weights.mean().detach()
                loss_dict["edge_weight_max"] = edge_weights.max().detach()
                loss_dict["edge_weight_min"] = edge_weights.min().detach()
            
            # 如果使用PCA Loss，添加channel loss统计信息
            if self.reconstruction_loss == "pca":
                loss_dict.update(pca_channel_losses)
        elif self.quantize_mode == "vae":
            if self.skip_encoder:
                # 如果跳过编码器，只计算重建损失和感知损失
                total_loss = reconstruction_loss + total_perceptual_loss + d_weight * discriminator_factor * generator_loss
                if weighted_alignment is not None:
                    total_loss = total_loss + weighted_alignment
                loss_dict = dict(
                    total_loss=total_loss.clone().detach(),
                    reconstruction_loss=reconstruction_loss.detach(),
                    perceptual_loss=total_perceptual_loss.detach(),
                    weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                    discriminator_factor=torch.tensor(discriminator_factor, device=inputs.device),
                    d_weight=d_weight,
                    gan_loss=generator_loss.detach(),
                )
                if weighted_alignment is not None:
                    loss_dict["encoder_alignment_loss"] = weighted_alignment.detach()
                    for key, value in encoder_alignment_logs.items():
                        if isinstance(value, torch.Tensor):
                            loss_dict[key] = value.detach()
                        else:
                            loss_dict[key] = value

                
                # 添加边缘权重相关的信息
                if self.use_edge_weighted_loss:
                    edge_weights = self.compute_edge_weight_map(inputs)
                    loss_dict["edge_weight_mean"] = edge_weights.mean().detach()
                    loss_dict["edge_weight_max"] = edge_weights.max().detach()
                    loss_dict["edge_weight_min"] = edge_weights.min().detach()
                
                # 如果使用PCA Loss，添加channel loss统计信息
                if self.reconstruction_loss == "pca":
                    loss_dict.update(pca_channel_losses)
            else:
                # Compute kl loss.
                reconstruction_loss = reconstruction_loss / torch.exp(self.logvar)
                if "posteriors" in extra_result_dict:
                    posteriors = extra_result_dict["posteriors"]
                    if hasattr(posteriors, 'kl'):
                        kl_loss = posteriors.kl()
                        kl_loss = torch.mean(kl_loss)
                    else:
                        kl_loss = torch.tensor(0.0, device=inputs.device, dtype=reconstruction_loss.dtype)
                else:
                    kl_loss = torch.tensor(0.0, device=inputs.device, dtype=reconstruction_loss.dtype)
                # 获取当前步数的KL权重
                current_kl_weight = self.get_kl_weight(global_step)
                
                total_loss = (
                    reconstruction_loss
                    + total_perceptual_loss
                    + current_kl_weight * kl_loss
                    + d_weight * discriminator_factor * generator_loss
                )
                if weighted_alignment is not None:
                    total_loss = total_loss + weighted_alignment
                loss_dict = dict(
                    total_loss=total_loss.clone().detach(),
                    reconstruction_loss=reconstruction_loss.detach(),
                    perceptual_loss=total_perceptual_loss.detach(),
                    kl_loss=(kl_loss).detach(),  # 原始KL loss
                    kl_loss_weighted=(current_kl_weight * kl_loss).detach(),  # 加权后的KL loss
                    kl_weight=current_kl_weight,  # 记录当前KL权重
                    weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                    discriminator_factor=torch.tensor(discriminator_factor, device=inputs.device),
                    d_weight=d_weight,
                    gan_loss=generator_loss.detach(),
                )
                if weighted_alignment is not None:
                    loss_dict["encoder_alignment_loss"] = weighted_alignment.detach()
                    for key, value in encoder_alignment_logs.items():
                        if isinstance(value, torch.Tensor):
                            loss_dict[key] = value.detach()
                        else:
                            loss_dict[key] = value

                
                # 添加边缘权重相关的信息
                if self.use_edge_weighted_loss:
                    edge_weights = self.compute_edge_weight_map(inputs)
                    loss_dict["edge_weight_mean"] = edge_weights.mean().detach()
                    loss_dict["edge_weight_max"] = edge_weights.max().detach()
                    loss_dict["edge_weight_min"] = edge_weights.min().detach()
                
                # 如果使用PCA Loss，添加channel loss统计信息
                if self.reconstruction_loss == "pca":
                    loss_dict.update(pca_channel_losses)
        elif self.quantize_mode == "direct":
            # Direct mode: no quantization, only reconstruction and perceptual loss
            total_loss = (
                reconstruction_loss
                + total_perceptual_loss
                + d_weight * discriminator_factor * generator_loss
            )
            if weighted_alignment is not None:
                total_loss = total_loss + weighted_alignment
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=total_perceptual_loss.detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor, device=inputs.device),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
            if weighted_alignment is not None:
                loss_dict["encoder_alignment_loss"] = weighted_alignment.detach()
                for key, value in encoder_alignment_logs.items():
                    if isinstance(value, torch.Tensor):
                        loss_dict[key] = value.detach()
                    else:
                        loss_dict[key] = value

            # 添加感知损失的详细信息
            loss_dict.update(perceptual_loss_dict)
            
            # 添加边缘权重相关的信息
            if self.use_edge_weighted_loss:
                edge_weights = self.compute_edge_weight_map(inputs)
                loss_dict["edge_weight_mean"] = edge_weights.mean().detach()
                loss_dict["edge_weight_max"] = edge_weights.max().detach()
                loss_dict["edge_weight_min"] = edge_weights.min().detach()
        
        # 如果使用PCA Loss，添加channel loss统计信息
        if self.reconstruction_loss == "pca":
            loss_dict.update(pca_channel_losses)
       
        return total_loss, loss_dict


class ReconstructionLoss_FluxVAE(ReconstructionLoss_Single_Stage):
    """Reconstruction loss for models using FLUX VAE in latent space."""
    
    def __init__(self, config, model=None):
        super().__init__(config)
        self.model = model  # 保存model引用
        self.use_flux_vae = config.model.get("use_flux_vae", False)
        self.use_sd3_vae = config.model.get("use_sd3_vae", False)
        
        # 新增：VAE模式配置
        self.flux_vae_mode = config.model.get("flux_vae_mode", "latent_to_latent")
        if self.use_flux_vae or self.use_sd3_vae:
            if self.flux_vae_mode not in ["latent_to_latent", "latent_to_pixel", "latent_to_latent_decode"]:
                raise ValueError(f"Unsupported flux_vae_mode: {self.flux_vae_mode}. "
                               f"Supported modes: latent_to_latent, latent_to_pixel, latent_to_latent_decode")
            print(f"VAE模式: {self.flux_vae_mode}")
        
        # 新增：是否使用原始像素输入作为ground truth的选项
        self.use_original_pixel_supervision = config.losses.get("use_original_pixel_supervision", False)
        if self.use_original_pixel_supervision:
            print("启用使用原始像素输入作为ground truth")
        
        from elatentlpips import ELatentLPIPS
        # Initialize E-LatentLPIPS
        self.perceptual_loss_type = config.losses.get("perceptual_loss", "decoder-lpips")
        
        # 直接使用传入的model的original_vae_model
        self.original_vae_model = model.original_vae_model if model is not None else None
        

        # 初始化感知损失（在VAE model创建之后）
        if self.perceptual_loss_type == "elpips-flux":
            self.perceptual_loss = ELatentLPIPS(
                encoder="flux",
                augment="b"
            ).eval()
            for param in self.perceptual_loss.parameters():
                param.requires_grad = False
        elif self.perceptual_loss_type == "decoder-lpips":
            # 使用DecoderLPIPS作为感知损失
            if not (self.use_flux_vae or self.use_sd3_vae):
                raise ValueError("DecoderLPIPS requires FLUX VAE or SD3 VAE to be enabled")
            
            # 获取DecoderLPIPS的配置
            # decoder_lpips_config = config.losses.get("decoder_lpips_config", {})
            # use_dropout = decoder_lpips_config.get("use_dropout", True)
            # latent_channels = decoder_lpips_config.get("latent_channels", 16)
            # use_depth_weighting = decoder_lpips_config.get("use_depth_weighting", True)
            # use_feature_normalization = decoder_lpips_config.get("use_feature_normalization", True)
            
            # # 现在可以初始化DecoderLPIPS，因为original_vae_model已经创建
            # if hasattr(self, 'original_vae_model') and self.original_vae_model is not None:
            #     from modeling.modules.decoder_lpips import DecoderLPIPSLoss
                
            #     # 获取decoder
            #     if hasattr(self.original_vae_model, 'vae') and hasattr(self.original_vae_model.vae, 'decoder'):
            #         decoder = self.original_vae_model.vae.decoder
            #     else:
            #         decoder = self.original_vae_model
                
            #     # 初始化DecoderLPIPS
            #     self.perceptual_loss = DecoderLPIPSLoss(
            #         decoder=decoder,
            #         use_dropout=use_dropout,
            #         latent_channels=latent_channels,
            #         use_depth_weighting=use_depth_weighting,
            #         use_feature_normalization=use_feature_normalization
            #     ).eval()
                
            #     # 冻结DecoderLPIPS参数
            #     for param in self.perceptual_loss.parameters():
            #         param.requires_grad = False
                
            #     print("✅ DecoderLPIPS已初始化并冻结")
            # else:
            #     self.perceptual_loss = None
            #     print("❌ DecoderLPIPS初始化失败：original_vae_model未创建")
        
   
            
    def _forward_generator(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step with VAE support."""
        if not (self.use_flux_vae or self.use_sd3_vae):
            # Fall back to original implementation
            return super()._forward_generator(inputs, reconstructions, extra_result_dict, global_step)
        
        # VAE enabled - handle different modes
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        
        # 根据模式处理输入和重建图像
        if self.flux_vae_mode == "latent_to_latent":
            # 模式1: 输入latent，输出latent，在latent space监督
            return self._compute_latent_space_loss(inputs, reconstructions, extra_result_dict, global_step)
            
        elif self.flux_vae_mode == "latent_to_pixel":
            # 模式2: 输入latent，输出pixel，在pixel space监督
            # 从extra_result_dict中获取原始像素输入
            original_pixel_inputs = extra_result_dict.get("original_pixel_inputs", None)
            return self._compute_pixel_space_loss_from_latent_output(inputs, reconstructions, extra_result_dict, global_step, original_pixel_inputs)
            
        elif self.flux_vae_mode == "latent_to_latent_decode":
            # 模式3: 输入latent，输出latent但decode到pixel space监督
            # 从extra_result_dict中获取原始像素输入
            original_pixel_inputs = extra_result_dict.get("original_pixel_inputs", None)
            return self._compute_pixel_space_loss_from_decoded_latent(inputs, reconstructions, extra_result_dict, global_step, original_pixel_inputs)
        
        else:
            raise ValueError(f"Unsupported flux_vae_mode: {self.flux_vae_mode}")

    def _compute_latent_space_loss(self, inputs: torch.Tensor, reconstructions: torch.Tensor, 
                                   extra_result_dict: Mapping[Text, torch.Tensor], global_step: int):
        """计算latent space的损失（原有实现）"""
        # 计算边缘权重图（如果启用）
        edge_weights = self.compute_edge_weight_map(inputs)
        
        # 计算重建损失，支持边缘加权和PCA损失
        if self.reconstruction_loss == "l1":
            if self.use_edge_weighted_loss:
                # 加权L1损失
                pixel_wise_l1 = torch.abs(inputs - reconstructions)
                reconstruction_loss = (edge_weights * pixel_wise_l1).mean()
            else:
                reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            if self.use_edge_weighted_loss:
                # 加权L2损失（统一到float32，避免与bf16混用）
                pixel_wise_mse = (inputs - reconstructions) ** 2
                reconstruction_loss = (edge_weights.float() * pixel_wise_mse).mean()
            else:
                reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "pca":
            # PCA损失
            reconstruction_loss, pca_channel_losses = self.pca_loss(inputs, reconstructions)
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {self.reconstruction_loss}")
        
        reconstruction_loss *= self.reconstruction_weight

        # 计算感知损失
        if self.perceptual_weight > 0.0:
            if self.patch_lpips_loss is not None:
                # 使用patch-based LPIPS loss
                perceptual_loss = self.patch_lpips_loss(inputs, reconstructions).mean()
            elif self.perceptual_loss is not None:
                if self.perceptual_loss_type == "decoder-lpips":
                    # 对于DecoderLPIPS，直接在latent space计算
                    # 处理packed模式
                    if self.use_flux_vae and self.original_vae_model.use_packed:
                        unpacked_inputs = unpack_latent_from_chw4(inputs)
                        unpacked_reconstructions = unpack_latent_from_chw4(reconstructions)
                    else:
                        unpacked_inputs = inputs
                        unpacked_reconstructions = reconstructions
                    
                    # DecoderLPIPS的调用方式：pred_latent, target_latent
                    perceptual_loss = self.perceptual_loss(unpacked_reconstructions, unpacked_inputs).mean()
                else:
                    # 对于E-LatentLPIPS，只有在packed模式下才需要unpack
                    if self.use_flux_vae and self.original_vae_model.use_packed:
                        unpacked_inputs = unpack_latent_from_chw4(inputs)
                        unpacked_reconstructions = unpack_latent_from_chw4(reconstructions)
                    else:
                        unpacked_inputs = inputs
                        unpacked_reconstructions = reconstructions
                    perceptual_loss = self.perceptual_loss(unpacked_inputs, unpacked_reconstructions, normalize=False).mean()
            else:
                perceptual_loss = torch.zeros((), device=inputs.device)
        else:
            perceptual_loss = torch.zeros((), device=inputs.device)

        # 计算编码器对齐损失（如果启用）
        encoder_alignment_loss = None
        encoder_alignment_logs = {}
        weighted_alignment = None
        if self.encoder_alignment_enabled:
            alignment_result = self._compute_encoder_alignment_loss(extra_result_dict)
            if alignment_result is not None:
                encoder_alignment_loss, encoder_alignment_logs = alignment_result
                weighted_alignment = self.encoder_alignment_weight * encoder_alignment_loss

        # Compute discriminator loss (optional, can be disabled for latent space)
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0 and self.use_adversarial_loss:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        # 根据quantize_mode计算总损失
        # 检查是否跳过编码器
        skip_encoder = getattr(self, 'skip_encoder', False)
        
        if self.quantize_mode == "vq" and not skip_encoder:
            quantizer_loss = extra_result_dict["quantizer_loss"]
            total_loss = (
                reconstruction_loss
                + perceptual_loss
                + self.quantizer_weight * quantizer_loss
                + d_weight * discriminator_factor * generator_loss
            )
            if weighted_alignment is not None:
                total_loss = total_loss + weighted_alignment
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=perceptual_loss.detach(),
                quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor, device=inputs.device),
                commitment_loss=extra_result_dict["commitment_loss"].detach(),
                codebook_loss=extra_result_dict["codebook_loss"].detach(),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
            if weighted_alignment is not None:
                loss_dict["encoder_alignment_loss"] = weighted_alignment.detach()
                for key, value in encoder_alignment_logs.items():
                    loss_dict[key] = value.detach() if isinstance(value, torch.Tensor) else value
            # 添加感知损失的详细信息
        elif self.quantize_mode == "vae" and not skip_encoder:
            reconstruction_loss = reconstruction_loss / torch.exp(self.logvar)
            # 从extra_result_dict中获取DiagonalGaussianDistribution对象
            if "posteriors" in extra_result_dict:
                posteriors = extra_result_dict["posteriors"]
                if hasattr(posteriors, 'kl'):
                    kl_loss = posteriors.kl()
                    kl_loss = torch.mean(kl_loss)
                else:
                    kl_loss = torch.tensor(0.0, device=inputs.device, dtype=reconstruction_loss.dtype)
            else:
                kl_loss = torch.tensor(0.0, device=inputs.device, dtype=reconstruction_loss.dtype)
                
            # 获取当前步数的KL权重
            current_kl_weight = self.get_kl_weight(global_step)
            
            total_loss = (
                reconstruction_loss
                + perceptual_loss
                + current_kl_weight * kl_loss
                + d_weight * discriminator_factor * generator_loss
            )
            if weighted_alignment is not None:
                total_loss = total_loss + weighted_alignment
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=perceptual_loss.detach(),
                kl_loss=(kl_loss).detach(),  # 原始KL loss
                kl_loss_weighted=(current_kl_weight * kl_loss).detach(),  # 加权后的KL loss
                kl_weight=current_kl_weight,  # 记录当前KL权重
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor, device=inputs.device),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
            if weighted_alignment is not None:
                loss_dict["encoder_alignment_loss"] = weighted_alignment.detach()
                for key, value in encoder_alignment_logs.items():
                    loss_dict[key] = value.detach() if isinstance(value, torch.Tensor) else value
            # 添加感知损失的详细信息
        elif self.quantize_mode == "direct" or skip_encoder:
            # Direct mode 或 skip_encoder: 没有量化，只有重建损失和感知损失
            total_loss = (
                reconstruction_loss
                + self.perceptual_weight * perceptual_loss
                + d_weight * discriminator_factor * generator_loss
            )
            if weighted_alignment is not None:
                total_loss = total_loss + weighted_alignment
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=perceptual_loss.detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor, device=inputs.device),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
            if weighted_alignment is not None:
                loss_dict["encoder_alignment_loss"] = weighted_alignment.detach()
                for key, value in encoder_alignment_logs.items():
                    loss_dict[key] = value.detach() if isinstance(value, torch.Tensor) else value
            # 添加感知损失的详细信息
        else:
            raise NotImplementedError

        # 添加边缘权重相关的信息
        if self.use_edge_weighted_loss:
            loss_dict["edge_weight_mean"] = edge_weights.mean().detach()
            loss_dict["edge_weight_max"] = edge_weights.max().detach()
            loss_dict["edge_weight_min"] = edge_weights.min().detach()
        
        # 如果使用PCA Loss，添加channel loss统计信息
        if self.reconstruction_loss == "pca":
            loss_dict.update(pca_channel_losses)

        return total_loss, loss_dict

    def _compute_pixel_space_loss_from_latent_output(self, inputs: torch.Tensor, reconstructions: torch.Tensor,
                                                     extra_result_dict: Mapping[Text, torch.Tensor], global_step: int,
                                                     original_pixel_inputs: torch.Tensor = None):
        """计算pixel space的损失（模型输出直接是pixel space）"""
        # 输入是latent space，需要decode到pixel space用于计算损失
        
        # 检查是否使用原始像素输入作为ground truth
        use_original_pixel_supervision = getattr(self, 'use_original_pixel_supervision', False)
        
        if use_original_pixel_supervision and original_pixel_inputs is not None:
            # 使用原始像素输入作为ground truth
            inputs_pixel = original_pixel_inputs
        else:
            # 将latent space的输入decode到pixel space用于计算损失
            with torch.no_grad():
                # 使用原始VAE model生成ground truth
                if self.use_flux_vae:
                    inputs_pixel = self.original_vae_model.decode(inputs).float()
                elif self.use_sd3_vae:
                    inputs_pixel = self.original_vae_model.decode(inputs).float()
        
        # 重建图像已经在pixel space，直接使用
        reconstructions_pixel = reconstructions
        
        # 计算边缘权重图（在pixel space）
        edge_weights = self.compute_edge_weight_map(inputs_pixel)
        
        # 计算重建损失，支持边缘加权和PCA损失
        if self.reconstruction_loss == "l1":
            if self.use_edge_weighted_loss:
                pixel_wise_l1 = torch.abs(inputs_pixel - reconstructions_pixel)
                reconstruction_loss = (edge_weights * pixel_wise_l1).mean()
            else:
                reconstruction_loss = F.l1_loss(inputs_pixel, reconstructions_pixel, reduction="mean")
        elif self.reconstruction_loss == "l2":
            if self.use_edge_weighted_loss:
                pixel_wise_mse = (inputs_pixel - reconstructions_pixel) ** 2
                reconstruction_loss = (edge_weights.float() * pixel_wise_mse).mean()
            else:
                reconstruction_loss = F.mse_loss(inputs_pixel, reconstructions_pixel, reduction="mean")
        elif self.reconstruction_loss == "pca":
            # PCA损失
            reconstruction_loss, pca_channel_losses = self.pca_loss(inputs_pixel, reconstructions_pixel)
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {self.reconstruction_loss}")
        
        reconstruction_loss *= self.reconstruction_weight

         # Compute perceptual losses.
        total_perceptual_loss = 0.0
        perceptual_loss_dict = {}
        
        # 计算LPIPS损失
        if self.patch_lpips_loss is not None:
            # 使用patch-based LPIPS损失
            lpips_loss = self.patch_lpips_loss(inputs_pixel, reconstructions_pixel)
            total_perceptual_loss += self.perceptual_weight * lpips_loss
            perceptual_loss_dict["lpips_loss"] = (self.perceptual_weight * lpips_loss).detach()
        elif self.lpips_loss is not None:
            # 使用独立的LPIPS损失
            lpips_loss = self.lpips_loss(inputs_pixel, reconstructions_pixel).mean()
            total_perceptual_loss += self.perceptual_weight * lpips_loss
            perceptual_loss_dict["lpips_loss"] = (self.perceptual_weight * lpips_loss).detach()
        elif self.perceptual_loss is not None:
            # 使用传统感知损失（向后兼容）
            perceptual_loss = self.perceptual_loss(inputs_pixel, reconstructions_pixel).mean()
            total_perceptual_loss += self.perceptual_weight * perceptual_loss
            perceptual_loss_dict["perceptual_loss"] = (self.perceptual_weight * perceptual_loss).detach()
        
        # 计算编码器对齐损失（如果启用）
        encoder_alignment_loss = None
        encoder_alignment_logs = {}
        weighted_alignment = None
        if self.encoder_alignment_enabled:
            alignment_result = self._compute_encoder_alignment_loss(extra_result_dict)
            if alignment_result is not None:
                encoder_alignment_loss, encoder_alignment_logs = alignment_result
                weighted_alignment = self.encoder_alignment_weight * encoder_alignment_loss

        # 计算Gram损失
        if self.patch_gram_loss is not None:
            # 使用patch-based Gram损失
            gram_loss = self.patch_gram_loss(inputs_pixel, reconstructions_pixel)
            total_perceptual_loss += self.gram_loss_weight * gram_loss
            perceptual_loss_dict["gram_loss"] = (self.gram_loss_weight * gram_loss).detach()
        elif self.gram_loss is not None:
            # 使用独立的Gram损失
            gram_loss = self.gram_loss(inputs_pixel, reconstructions_pixel)
            total_perceptual_loss += self.gram_loss_weight * gram_loss
            perceptual_loss_dict["gram_loss"] = (self.gram_loss_weight * gram_loss).detach()

        # Compute discriminator loss
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0 and self.use_adversarial_loss:
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions_pixel.float())
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        # 根据quantize_mode计算总损失
        # 检查是否跳过编码器
        skip_encoder = getattr(self, 'skip_encoder', False)
        
        if self.quantize_mode == "vq" and not skip_encoder:
            quantizer_loss = extra_result_dict["quantizer_loss"]
            total_loss = (
                reconstruction_loss
                + total_perceptual_loss
                + self.quantizer_weight * quantizer_loss
                + d_weight * discriminator_factor * generator_loss
            )
            if weighted_alignment is not None:
                total_loss = total_loss + weighted_alignment
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=total_perceptual_loss.detach(),
                quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor, device=inputs.device),
                commitment_loss=extra_result_dict["commitment_loss"].detach(),
                codebook_loss=extra_result_dict["codebook_loss"].detach(),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
            # 添加感知损失的详细信息
            loss_dict.update(perceptual_loss_dict)
            if weighted_alignment is not None:
                loss_dict["encoder_alignment_loss"] = weighted_alignment.detach()
                for key, value in encoder_alignment_logs.items():
                    loss_dict[key] = value.detach() if isinstance(value, torch.Tensor) else value
        elif self.quantize_mode == "vae" and not skip_encoder:
            reconstruction_loss = reconstruction_loss / torch.exp(self.logvar)
            # 从extra_result_dict中获取DiagonalGaussianDistribution对象（健壮处理）
            if isinstance(extra_result_dict, dict):
                posteriors = extra_result_dict.get("posteriors", None)
            else:
                posteriors = extra_result_dict
            if hasattr(posteriors, 'kl'):
                kl_loss = posteriors.kl()
                kl_loss = torch.mean(kl_loss)
            else:
                kl_loss = torch.tensor(0.0, device=inputs.device, dtype=reconstruction_loss.dtype)
            
            # 获取当前步数的KL权重
            current_kl_weight = self.get_kl_weight(global_step)
            
            total_loss = (
                reconstruction_loss
                + total_perceptual_loss
                + current_kl_weight * kl_loss
                + d_weight * discriminator_factor * generator_loss
            )
            if weighted_alignment is not None:
                total_loss = total_loss + weighted_alignment
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=total_perceptual_loss.detach(),
                kl_loss=(current_kl_weight * kl_loss).detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor, device=inputs.device),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
            # 添加感知损失的详细信息
            loss_dict.update(perceptual_loss_dict)
            if weighted_alignment is not None:
                loss_dict["encoder_alignment_loss"] = weighted_alignment.detach()
                for key, value in encoder_alignment_logs.items():
                    loss_dict[key] = value.detach() if isinstance(value, torch.Tensor) else value
        elif self.quantize_mode == "direct" or skip_encoder:
            # Direct mode 或 skip_encoder: 没有量化，只有重建损失和感知损失
            total_loss = (
                reconstruction_loss
                + total_perceptual_loss
                + d_weight * discriminator_factor * generator_loss
            )
            if weighted_alignment is not None:
                total_loss = total_loss + weighted_alignment
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=total_perceptual_loss.detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor, device=inputs.device),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
            # 添加感知损失的详细信息
            loss_dict.update(perceptual_loss_dict)
            if weighted_alignment is not None:
                loss_dict["encoder_alignment_loss"] = weighted_alignment.detach()
                for key, value in encoder_alignment_logs.items():
                    loss_dict[key] = value.detach() if isinstance(value, torch.Tensor) else value
        else:
            raise NotImplementedError

        # 添加边缘权重相关的信息
        if self.use_edge_weighted_loss:
            loss_dict["edge_weight_mean"] = edge_weights.mean().detach()
            loss_dict["edge_weight_max"] = edge_weights.max().detach()
            loss_dict["edge_weight_min"] = edge_weights.min().detach()

        return total_loss, loss_dict

    def _compute_pixel_space_loss_from_decoded_latent(self, inputs: torch.Tensor, reconstructions: torch.Tensor,
                                                      extra_result_dict: Mapping[Text, torch.Tensor], global_step: int,
                                                      original_pixel_inputs: torch.Tensor = None):
        """计算pixel space的损失（模型输出是latent但decode到pixel space监督）"""
        # 输入和重建图像都在latent space，需要都decode到pixel space用于计算损失
        
        # 检查是否使用原始像素输入作为ground truth
        use_original_pixel_supervision = getattr(self, 'use_original_pixel_supervision', False)
        
        if use_original_pixel_supervision and original_pixel_inputs is not None:
            # 使用原始像素输入作为ground truth
            inputs_pixel = original_pixel_inputs
            # 重建图像已经是pixel，直接使用
            reconstructions_pixel = reconstructions
        else:
            # 将latent space的输入decode到pixel space，重建图像已经是pixel
            if self.use_flux_vae:
                with torch.no_grad():
                    inputs_pixel = self.original_vae_model.decode(inputs)
            elif self.use_sd3_vae:
                with torch.no_grad():
                    inputs_pixel = self.original_vae_model.decode(inputs)
            # 重建图像已经是pixel，直接使用
            reconstructions_pixel = reconstructions
        
        # 计算边缘权重图（在pixel space）
        edge_weights = self.compute_edge_weight_map(inputs_pixel)
        
        # 计算重建损失，支持边缘加权和PCA损失
        if self.reconstruction_loss == "l1":
            if self.use_edge_weighted_loss:
                pixel_wise_l1 = torch.abs(inputs_pixel - reconstructions_pixel)
                reconstruction_loss = (edge_weights * pixel_wise_l1).mean()
            else:
                reconstruction_loss = F.l1_loss(inputs_pixel, reconstructions_pixel, reduction="mean")
        elif self.reconstruction_loss == "l2":
            if self.use_edge_weighted_loss:
                pixel_wise_mse = (inputs_pixel - reconstructions_pixel) ** 2
                reconstruction_loss = (edge_weights.float() * pixel_wise_mse).mean()
            else:
                reconstruction_loss = F.mse_loss(inputs_pixel, reconstructions_pixel, reduction="mean")
        elif self.reconstruction_loss == "pca":
            # PCA损失
            reconstruction_loss, pca_channel_losses = self.pca_loss(inputs_pixel, reconstructions_pixel)
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {self.reconstruction_loss}")
        
        reconstruction_loss *= self.reconstruction_weight

        # 在pixel space计算perceptual loss
        if self.perceptual_weight > 0.0:
            if self.patch_lpips_loss is not None:
                # 使用patch-based LPIPS loss
                perceptual_loss = self.patch_lpips_loss(inputs, reconstructions).mean()
            elif self.perceptual_loss is not None:
                if self.perceptual_loss_type == "decoder-lpips":
                    if self.use_flux_vae:
                        # 只有在packed模式下才需要unpack
                        if self.original_vae_model.use_packed:
                            inputs_unpacked = unpack_latent_from_chw4(inputs)
                            # reconstructions 已经是pixel，不需要unpack
                        else:
                            inputs_unpacked = inputs
                    # 对于DecoderLPIPS，在latent space计算感知损失
                    perceptual_loss = self.perceptual_loss(inputs_unpacked, reconstructions).mean()
                else:
                    # 对于E-LatentLPIPS，在pixel space计算感知损失
                    perceptual_loss = self.perceptual_loss(inputs_pixel, reconstructions_pixel).mean()
            else:
                perceptual_loss = torch.zeros((), device=inputs.device)
            
            total_perceptual_loss = perceptual_loss * self.perceptual_weight
            perceptual_loss_dict = {"perceptual_loss": perceptual_loss.detach()}
        else:
            total_perceptual_loss = torch.zeros((), device=inputs.device)
            perceptual_loss_dict = {"perceptual_loss": torch.zeros((), device=inputs.device)}

        # 计算编码器对齐损失（如果启用）
        encoder_alignment_loss = None
        encoder_alignment_logs = {}
        weighted_alignment = None
        if self.encoder_alignment_enabled:
            alignment_result = self._compute_encoder_alignment_loss(extra_result_dict)
            if alignment_result is not None:
                encoder_alignment_loss, encoder_alignment_logs = alignment_result
                weighted_alignment = self.encoder_alignment_weight * encoder_alignment_loss
        # Compute discriminator loss
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0 and self.use_adversarial_loss:
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions_pixel.float())
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        # 根据quantize_mode计算总损失
        # 检查是否跳过编码器
        skip_encoder = getattr(self, 'skip_encoder', False)
        
        if self.quantize_mode == "vq" and not skip_encoder:
            quantizer_loss = extra_result_dict["quantizer_loss"]
            total_loss = (
                reconstruction_loss
                + total_perceptual_loss
                + self.quantizer_weight * quantizer_loss
                + d_weight * discriminator_factor * generator_loss
            )
            if weighted_alignment is not None:
                total_loss = total_loss + weighted_alignment
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=total_perceptual_loss.detach(),
                quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor, device=inputs.device),
                commitment_loss=extra_result_dict["commitment_loss"].detach(),
                codebook_loss=extra_result_dict["codebook_loss"].detach(),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
            # 添加感知损失的详细信息
            loss_dict.update(perceptual_loss_dict)
            if weighted_alignment is not None:
                loss_dict["encoder_alignment_loss"] = weighted_alignment.detach()
                for key, value in encoder_alignment_logs.items():
                    loss_dict[key] = value.detach() if isinstance(value, torch.Tensor) else value
        elif self.quantize_mode == "vae" and not skip_encoder:
            reconstruction_loss = reconstruction_loss / torch.exp(self.logvar)
            # 从extra_result_dict中获取DiagonalGaussianDistribution对象（健壮处理）
            if isinstance(extra_result_dict, dict):
                posteriors = extra_result_dict.get("posteriors", None)
            else:
                posteriors = extra_result_dict
            if hasattr(posteriors, 'kl'):
                kl_loss = posteriors.kl()
                kl_loss = torch.mean(kl_loss)
            else:
                kl_loss = torch.tensor(0.0, device=inputs.device, dtype=reconstruction_loss.dtype)
            
            # 获取当前步数的KL权重
            current_kl_weight = self.get_kl_weight(global_step)
          
            total_loss = (
                reconstruction_loss
                + total_perceptual_loss
                + current_kl_weight * kl_loss
                + d_weight * discriminator_factor * generator_loss
            )
            if weighted_alignment is not None:
                total_loss = total_loss + weighted_alignment
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=total_perceptual_loss.detach(),
                kl_loss=(current_kl_weight * kl_loss).detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor, device=inputs.device),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
            # 添加感知损失的详细信息
            loss_dict.update(perceptual_loss_dict)
            if weighted_alignment is not None:
                loss_dict["encoder_alignment_loss"] = weighted_alignment.detach()
                for key, value in encoder_alignment_logs.items():
                    loss_dict[key] = value.detach() if isinstance(value, torch.Tensor) else value
        elif self.quantize_mode == "direct" or skip_encoder:
            # Direct mode 或 skip_encoder: 没有量化，只有重建损失和感知损失
            total_loss = (
                reconstruction_loss
                + self.perceptual_weight * perceptual_loss
                + d_weight * discriminator_factor * generator_loss
            )
            if weighted_alignment is not None:
                total_loss = total_loss + weighted_alignment
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=total_perceptual_loss.detach(),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
                discriminator_factor=torch.tensor(discriminator_factor, device=inputs.device),
                d_weight=d_weight,
                gan_loss=generator_loss.detach(),
            )
            # 添加感知损失的详细信息
            loss_dict.update(perceptual_loss_dict)
            if weighted_alignment is not None:
                loss_dict["encoder_alignment_loss"] = weighted_alignment.detach()
                for key, value in encoder_alignment_logs.items():
                    loss_dict[key] = value.detach() if isinstance(value, torch.Tensor) else value
        else:
            raise NotImplementedError

        # 添加边缘权重相关的信息
        if self.use_edge_weighted_loss:
            loss_dict["edge_weight_mean"] = edge_weights.mean().detach()
            loss_dict["edge_weight_max"] = edge_weights.max().detach()
            loss_dict["edge_weight_min"] = edge_weights.min().detach()
        
        # 如果使用PCA Loss，添加channel loss统计信息
        if self.reconstruction_loss == "pca":
            loss_dict.update(pca_channel_losses)

        return total_loss, loss_dict

    def decode_to_pixel_space(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Decode VAE latents back to pixel space for visualization."""
        if self.use_flux_vae:
            return self.original_vae_model.decode(latents, height=height, width=width)
        elif self.use_sd3_vae:
            return self.original_vae_model.decode(latents, height=height, width=width)
        else:
            return latents
    
    # 移除set_vae_model方法，VAE model现在在loss module内部直接管理
    
    def _freeze_model_components(self):
        """
        冻结模型的其他部分，只保留VAE decoder可训练
        这个方法会在train_vae_decoder=True时被调用
        """
        print("开始冻结模型的其他部分...")
        
        # 这个方法需要在模型创建后调用，所以我们需要延迟执行
        # 在实际使用时，这个方法会被重写或通过其他方式调用
        self._model_freeze_pending = True
        print("模型冻结将在模型创建后执行")
    
    def apply_model_freeze(self, model):
        """
        应用模型冻结，冻结除VAE decoder外的所有参数
        这个方法需要在模型创建后调用
        """
        if not hasattr(self, '_model_freeze_pending') or not self._model_freeze_pending:
            return
        
        print("应用模型冻结 - 只保留VAE decoder可训练")
        
        # 冻结模型的所有参数
        for name, param in model.named_parameters():
            param.requires_grad = False
            # print(f"冻结模型参数: {name}")
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_params = total_params - frozen_params
        
        print(f"模型参数统计:")
        print(f"  - 总参数: {total_params:,}")
        print(f"  - 冻结参数: {frozen_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
        print(f"  - 冻结比例: {100.0 * frozen_params / total_params:.2f}%")
        
        # 标记冻结已完成
        self._model_freeze_pending = False
        print("模型冻结完成")
    
    def compute_loss_gradients(self, 
                              inputs: torch.Tensor,
                              reconstructions: torch.Tensor,
                              extra_result_dict: Mapping[Text, torch.Tensor],
                              global_step: int) -> Mapping[Text, torch.Tensor]:
        """计算不同loss对output pixel的梯度量级，支持VAE形状对齐。
        
        Args:
            inputs: 输入图像
            reconstructions: 重建图像
            extra_result_dict: 额外的结果字典
            global_step: 全局步数
            
        Returns:
            gradient_dict: 包含各种loss梯度的字典
        """
        if not (self.use_flux_vae or self.use_sd3_vae):
            # 如果没有使用VAE，回退到父类实现
            return super().compute_loss_gradients(inputs, reconstructions, extra_result_dict, global_step)
        
        gradient_dict = {}
        
        # 根据VAE模式处理输入和重建图像
        if self.flux_vae_mode == "latent_to_latent":
            # 模式1: 输入latent，输出latent，在latent space计算梯度
            return self._compute_latent_space_gradients(inputs, reconstructions, extra_result_dict, global_step)
            
        elif self.flux_vae_mode == "latent_to_pixel":
            # 模式2: 输入latent，输出pixel，在pixel space计算梯度
            original_pixel_inputs = extra_result_dict.get("original_pixel_inputs", None)
            return self._compute_pixel_space_gradients_from_latent_output(inputs, reconstructions, extra_result_dict, global_step, original_pixel_inputs)
            
        elif self.flux_vae_mode == "latent_to_latent_decode":
            # 模式3: 输入latent，输出latent但decode到pixel space计算梯度
            original_pixel_inputs = extra_result_dict.get("original_pixel_inputs", None)
            return self._compute_pixel_space_gradients_from_decoded_latent(inputs, reconstructions, extra_result_dict, global_step, original_pixel_inputs)
        
        else:
            raise ValueError(f"Unsupported flux_vae_mode: {self.flux_vae_mode}")

    def _compute_latent_space_gradients(self, inputs: torch.Tensor, reconstructions: torch.Tensor, 
                                       extra_result_dict: Mapping[Text, torch.Tensor], global_step: int):
        """在latent space计算梯度"""
        gradient_dict = {}
        
        # 创建完全独立的tensor副本，避免修改原始tensor
        inputs_detached = inputs.detach().clone()
        reconstructions_detached = reconstructions.detach().clone()
        
        # 设置梯度计算
        reconstructions_detached.requires_grad_(True)
        reconstructions_detached.retain_grad()
        
        try:
            # 1. 计算reconstruction loss的梯度
            if self.reconstruction_loss == "l1":
                reconstruction_loss = F.l1_loss(inputs_detached, reconstructions_detached, reduction="mean")
            elif self.reconstruction_loss == "l2":
                reconstruction_loss = F.mse_loss(inputs_detached, reconstructions_detached, reduction="mean")
            elif self.reconstruction_loss == "pca":
                reconstruction_loss, _ = self.pca_loss(inputs_detached, reconstructions_detached)
            else:
                raise ValueError(f"Unsupported reconstruction_loss {self.reconstruction_loss}")
            
            reconstruction_loss *= self.reconstruction_weight
            reconstruction_loss.backward(retain_graph=True)
            reconstruction_grad = reconstructions_detached.grad.clone()
            gradient_dict['reconstruction_grad'] = reconstruction_grad.detach()
            reconstructions_detached.grad.zero_()
            
            # 清理第一个loss的计算图
            del reconstruction_loss, reconstruction_grad
            
            # 2. 计算perceptual loss的梯度
            if self.perceptual_weight > 0.0:
                if self.patch_lpips_loss is not None:
                    # 使用patch-based LPIPS loss
                    perceptual_loss = self.patch_lpips_loss(inputs_detached, reconstructions_detached).mean()
                elif self.perceptual_loss is not None:
                    if self.perceptual_loss_type == "decoder-lpips":
                        # 对于DecoderLPIPS，直接在latent space计算
                        if self.use_flux_vae and self.original_vae_model.use_packed:
                            unpacked_inputs = unpack_latent_from_chw4(inputs_detached)
                            unpacked_reconstructions = unpack_latent_from_chw4(reconstructions_detached)
                        else:
                            unpacked_inputs = inputs_detached
                            unpacked_reconstructions = reconstructions_detached
                        
                        perceptual_loss = self.perceptual_loss(unpacked_reconstructions, unpacked_inputs).mean()
                    else:
                        # 对于E-LatentLPIPS
                        if self.use_flux_vae and self.original_vae_model.use_packed:
                            unpacked_inputs = unpack_latent_from_chw4(inputs_detached)
                            unpacked_reconstructions = unpack_latent_from_chw4(reconstructions_detached)
                        else:
                            unpacked_inputs = inputs_detached
                            unpacked_reconstructions = reconstructions_detached
                        perceptual_loss = self.perceptual_loss(unpacked_inputs, unpacked_reconstructions, normalize=False).mean()
                else:
                    perceptual_loss = torch.zeros((), device=inputs_detached.device)
                
                perceptual_loss *= self.perceptual_weight
                perceptual_loss.backward(retain_graph=True)
                perceptual_grad = reconstructions_detached.grad.clone()
                gradient_dict['perceptual_grad'] = perceptual_grad.detach()
                reconstructions_detached.grad.zero_()
                
                # 清理第二个loss的计算图
                del perceptual_loss, perceptual_grad
            
            # 3. 计算discriminator loss的梯度（在latent space通常不使用）
            # 这里可以添加latent space的判别器梯度计算，如果需要的话
            
        except Exception as e:
            print(f"计算梯度统计信息时出现错误: {e}")
            # 返回空的梯度字典
            return {}
        
        return gradient_dict

    def _compute_pixel_space_gradients_from_latent_output(self, inputs: torch.Tensor, reconstructions: torch.Tensor,
                                                         extra_result_dict: Mapping[Text, torch.Tensor], global_step: int,
                                                         original_pixel_inputs: torch.Tensor = None):
        """从latent输出在pixel space计算梯度"""
        gradient_dict = {}
        
        if original_pixel_inputs is None:
            print("警告: 在latent_to_pixel模式下需要original_pixel_inputs")
            return {}
        
        # 创建完全独立的tensor副本
        inputs_detached = original_pixel_inputs.detach().clone()  # 使用原始像素输入
        reconstructions_detached = reconstructions.detach().clone()  # 重建的像素输出
        
        # 设置梯度计算
        reconstructions_detached.requires_grad_(True)
        reconstructions_detached.retain_grad()
        
        try:
            # 1. 计算reconstruction loss的梯度
            if self.reconstruction_loss == "l1":
                reconstruction_loss = F.l1_loss(inputs_detached, reconstructions_detached, reduction="mean")
            elif self.reconstruction_loss == "l2":
                reconstruction_loss = F.mse_loss(inputs_detached, reconstructions_detached, reduction="mean")
            elif self.reconstruction_loss == "pca":
                reconstruction_loss, _ = self.pca_loss(inputs_detached, reconstructions_detached)
            else:
                raise ValueError(f"Unsupported reconstruction_loss {self.reconstruction_loss}")
            
            reconstruction_loss *= self.reconstruction_weight
            reconstruction_loss.backward(retain_graph=True)
            reconstruction_grad = reconstructions_detached.grad.clone()
            gradient_dict['reconstruction_grad'] = reconstruction_grad.detach()
            reconstructions_detached.grad.zero_()
            
            # 清理第一个loss的计算图
            del reconstruction_loss, reconstruction_grad
            
            # 2. 计算perceptual loss的梯度
            if self.perceptual_weight > 0.0:
                if self.patch_lpips_loss is not None:
                    # 使用patch-based LPIPS loss
                    perceptual_loss = self.patch_lpips_loss(inputs_detached, reconstructions_detached).mean()
                elif self.perceptual_loss is not None:
                    if self.perceptual_loss_type == "decoder-lpips":
                        # 对于DecoderLPIPS，需要将pixel转换回latent
                        # 这里需要VAE编码器，但为了梯度计算，我们可能需要特殊处理
                        print("警告: DecoderLPIPS在latent_to_pixel模式下的梯度计算需要特殊处理")
                        perceptual_loss = torch.zeros((), device=inputs_detached.device)
                    else:
                        # 对于E-LatentLPIPS，在pixel space计算
                        perceptual_loss = self.perceptual_loss(inputs_detached, reconstructions_detached).mean()
                else:
                    perceptual_loss = torch.zeros((), device=inputs_detached.device)
                
                perceptual_loss *= self.perceptual_weight
                perceptual_loss.backward(retain_graph=True)
                perceptual_grad = reconstructions_detached.grad.clone()
                gradient_dict['perceptual_grad'] = perceptual_grad.detach()
                reconstructions_detached.grad.zero_()
                
                # 清理第二个loss的计算图
                del perceptual_loss, perceptual_grad
            
            # 3. 计算discriminator loss的梯度
            discriminator_factor = self.discriminator_factor
            if discriminator_factor > 0.0 and self.discriminator_weight > 0.0 and self.use_adversarial_loss:
                # 禁用discriminator的梯度
                for param in self.discriminator.parameters():
                    param.requires_grad = False
                logits_fake = self.discriminator(reconstructions_detached.float())
                generator_loss = -torch.mean(logits_fake)
                generator_loss *= self.discriminator_weight * discriminator_factor
                generator_loss.backward(retain_graph=True)
                discriminator_grad = reconstructions_detached.grad.clone()
                gradient_dict['discriminator_grad'] = discriminator_grad.detach()
                reconstructions_detached.grad.zero_()
                
                # 清理discriminator loss的计算图
                del generator_loss, discriminator_grad
            
        except Exception as e:
            print(f"计算梯度统计信息时出现错误: {e}")
            return {}
        
        return gradient_dict

    def _compute_pixel_space_gradients_from_decoded_latent(self, inputs: torch.Tensor, reconstructions: torch.Tensor,
                                                          extra_result_dict: Mapping[Text, torch.Tensor], global_step: int,
                                                          original_pixel_inputs: torch.Tensor = None):
        """从解码的latent在pixel space计算梯度"""
        gradient_dict = {}
        
        if original_pixel_inputs is None:
            print("警告: 在latent_to_latent_decode模式下需要original_pixel_inputs")
            return {}
        
        # 创建完全独立的tensor副本
        inputs_detached = original_pixel_inputs.detach().clone()  # 使用原始像素输入
        
        # 将重建的latent解码为像素
        with torch.no_grad():
            if self.use_flux_vae:
                if self.original_vae_model.use_packed:
                    # 处理packed模式
                    reconstructions_unpacked = unpack_latent_from_chw4(reconstructions)
                else:
                    reconstructions_unpacked = reconstructions
                
                # 使用VAE解码器
                reconstructions_pixel = self.original_vae_model.decode(reconstructions_unpacked)
            else:
                # SD3 VAE处理
                reconstructions_pixel = self.original_vae_model.decode(reconstructions)
        
        reconstructions_detached = reconstructions_pixel.detach().clone()
        
        # 设置梯度计算
        reconstructions_detached.requires_grad_(True)
        reconstructions_detached.retain_grad()
        
        try:
            # 1. 计算reconstruction loss的梯度
            if self.reconstruction_loss == "l1":
                reconstruction_loss = F.l1_loss(inputs_detached, reconstructions_detached, reduction="mean")
            elif self.reconstruction_loss == "l2":
                reconstruction_loss = F.mse_loss(inputs_detached, reconstructions_detached, reduction="mean")
            elif self.reconstruction_loss == "pca":
                reconstruction_loss, _ = self.pca_loss(inputs_detached, reconstructions_detached)
            else:
                raise ValueError(f"Unsupported reconstruction_loss {self.reconstruction_loss}")
            
            reconstruction_loss *= self.reconstruction_weight
            reconstruction_loss.backward(retain_graph=True)
            reconstruction_grad = reconstructions_detached.grad.clone()
            gradient_dict['reconstruction_grad'] = reconstruction_grad.detach()
            reconstructions_detached.grad.zero_()
            
            # 清理第一个loss的计算图
            del reconstruction_loss, reconstruction_grad
            
            # 2. 计算perceptual loss的梯度
            if self.perceptual_weight > 0.0:
                if self.patch_lpips_loss is not None:
                    # 使用patch-based LPIPS loss
                    perceptual_loss = self.patch_lpips_loss(inputs_detached, reconstructions_detached).mean()
                elif self.perceptual_loss is not None:
                    if self.perceptual_loss_type == "decoder-lpips":
                        # 对于DecoderLPIPS，在latent space计算
                        if self.use_flux_vae and self.original_vae_model.use_packed:
                            unpacked_inputs = unpack_latent_from_chw4(inputs)
                            unpacked_reconstructions = unpack_latent_from_chw4(reconstructions)
                        else:
                            unpacked_inputs = inputs
                            unpacked_reconstructions = reconstructions
                        
                        perceptual_loss = self.perceptual_loss(unpacked_reconstructions, unpacked_inputs).mean()
                    else:
                        # 对于E-LatentLPIPS，在pixel space计算
                        perceptual_loss = self.perceptual_loss(inputs_detached, reconstructions_detached).mean()
                else:
                    perceptual_loss = torch.zeros((), device=inputs_detached.device)
                
                perceptual_loss *= self.perceptual_weight
                perceptual_loss.backward(retain_graph=True)
                perceptual_grad = reconstructions_detached.grad.clone()
                gradient_dict['perceptual_grad'] = perceptual_grad.detach()
                reconstructions_detached.grad.zero_()
                
                # 清理第二个loss的计算图
                del perceptual_loss, perceptual_grad
            
            # 3. 计算discriminator loss的梯度
            discriminator_factor = self.discriminator_factor
            if discriminator_factor > 0.0 and self.discriminator_weight > 0.0 and self.use_adversarial_loss:
                # 禁用discriminator的梯度
                for param in self.discriminator.parameters():
                    param.requires_grad = False
                logits_fake = self.discriminator(reconstructions_detached.float())
                generator_loss = -torch.mean(logits_fake)
                generator_loss *= self.discriminator_weight * discriminator_factor
                generator_loss.backward(retain_graph=True)
                discriminator_grad = reconstructions_detached.grad.clone()
                gradient_dict['discriminator_grad'] = discriminator_grad.detach()
                reconstructions_detached.grad.zero_()
                
                # 清理discriminator loss的计算图
                del generator_loss, discriminator_grad
            
        except Exception as e:
            print(f"计算梯度统计信息时出现错误: {e}")
            return {}
        
        return gradient_dict

    def to(self, device):
        """Move the loss module and its VAE model to the specified device."""
        super().to(device)
        if hasattr(self, 'original_vae_model') and self.original_vae_model is not None:
            self.original_vae_model.to(device)
        return self


class MLMLoss(torch.nn.Module):
    def __init__(self,
                 config):
        super().__init__()
        self.label_smoothing = config.losses.label_smoothing
        self.loss_weight_unmasked_token = config.losses.loss_weight_unmasked_token
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing,
                                                   reduction="none")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                weights=None) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        inputs = rearrange(inputs, "b n c -> b c n")
        loss = self.criterion(inputs, targets)
        if weights is not None:
            weights = weights.to(loss)
            loss_weights = (1.0 - weights) * self.loss_weight_unmasked_token + weights # set 0 to self.loss_weight_unasked_token
            loss = (loss * loss_weights).sum() / (loss_weights.sum() + 1e-8)
            # we only compute correct tokens on masked tokens
            correct_tokens = ((torch.argmax(inputs, dim=1) == targets) * weights).sum(dim=1) / (weights.sum(1) + 1e-8)
        else:
            loss = loss.mean()
            correct_tokens = (torch.argmax(inputs, dim=1) == targets).float().mean()
        return loss, {"loss": loss, "correct_tokens": correct_tokens}
    

class ARLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.target_vocab_size = config.model.vq_model.codebook_size
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        shift_logits = logits[..., :-1, :].permute(0, 2, 1).contiguous() # NLC->NCL
        shift_labels = labels.contiguous()
        shift_logits = shift_logits.view(shift_logits.shape[0], self.target_vocab_size, -1)
        shift_labels = shift_labels.view(shift_labels.shape[0], -1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = self.criterion(shift_logits, shift_labels)
        correct_tokens = (torch.argmax(shift_logits, dim=1) == shift_labels).sum(dim=1) / shift_labels.size(1)
        return loss, {"loss": loss, "correct_tokens": correct_tokens.mean()}
    

class DiffLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, config):
        super(DiffLoss, self).__init__()
        self.in_channels = config.model.vq_model.token_size

        self.net = SimpleMLPAdaLN(
            in_channels=self.in_channels,
            model_channels=config.losses.diffloss_w,
            out_channels=self.in_channels * 2,  # for vlb loss
            z_channels=config.model.maskgen.decoder_embed_dim,
            num_res_blocks=config.losses.diffloss_d,
            grad_checkpointing=config.get("training.grad_checkpointing", False),
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=config.losses.get("num_sampling_steps", "100"), noise_schedule="cosine")

    def forward(self, target, z, mask=None):
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()

        loss_dict = dict(
            diff_loss=loss.clone().mean().detach(),
        )

        return loss.mean(), loss_dict

    def sample(self, z, temperature=1.0, cfg=1.0):
        # diffusion loss sampling
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )

        return sampled_token_latent

# 添加基于patch的对抗损失类
class PatchBasedAdversarialLoss(nn.Module):
    """基于patch的对抗损失，通过随机crop patch来计算判别器损失。
    
    这个类包装了现有的判别器，并添加了基于patch的随机采样功能。
    通过随机crop多个patch来计算对抗损失，可以：
    1. 减少计算量（特别是对于大图像）
    2. 增加训练的随机性，提高泛化能力
    3. 关注局部细节，可能提高生成质量
    4. 减少判别器的过拟合风险
    """
    
    def __init__(self, 
                 discriminator: nn.Module,
                 patch_size: int = 128,
                 num_patches: int = 8,
                 min_patch_size: int = 64,
                 max_patch_size: int = 256,
                 use_random_patch_size: bool = True,
                 ensure_valid_crop: bool = True,
                 patch_sampling_strategy: str = "random",
                 grid_stride: Optional[int] = None):
        """初始化基于patch的对抗损失。
        
        Args:
            discriminator: 基础的判别器模块
            patch_size: 固定的patch大小（当use_random_patch_size=False时使用）
            num_patches: 每次计算损失时使用的patch数量
            min_patch_size: 随机patch大小的最小值（当use_random_patch_size=True时使用）
            max_patch_size: 随机patch大小的最大值（当use_random_patch_size=True时使用）
            use_random_patch_size: 是否使用随机的patch大小
            ensure_valid_crop: 是否确保crop的patch完全在图像范围内
            patch_sampling_strategy: patch采样策略，支持"random"、"grid"、"attention_weighted"
            grid_stride: 网格采样时的步长（像素），默认为patch大小
        """
        super().__init__()
        self.discriminator = discriminator
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.use_random_patch_size = use_random_patch_size
        self.ensure_valid_crop = ensure_valid_crop
        self.patch_sampling_strategy = patch_sampling_strategy
        self.grid_stride = grid_stride
        
        # 确保patch大小合理
        if self.use_random_patch_size:
            assert self.min_patch_size <= self.max_patch_size, "min_patch_size必须小于等于max_patch_size"
            assert self.min_patch_size > 0, "min_patch_size必须大于0"
        else:
            assert self.patch_size > 0, "patch_size必须大于0"

        if self.grid_stride is not None and self.grid_stride <= 0:
            raise ValueError("grid_stride必须大于0")
        
        # 验证采样策略
        valid_strategies = ["random", "grid", "attention_weighted"]
        if patch_sampling_strategy not in valid_strategies:
            raise ValueError(f"不支持的patch_sampling_strategy: {patch_sampling_strategy}。支持: {valid_strategies}")

    def _get_random_patch_size(self) -> int:
        """获取随机的patch大小。"""
        if not self.use_random_patch_size:
            return self.patch_size
        
        # 在min_patch_size和max_patch_size之间随机选择
        # 使用32的倍数以确保与某些网络架构兼容
        size_range = self.max_patch_size - self.min_patch_size
        random_size = self.min_patch_size + torch.randint(0, size_range // 32 + 1, (1,)).item() * 32
        return min(random_size, self.max_patch_size)

    def _get_random_crop_coords(self, image_shape: tuple, patch_size: int):
        """生成随机裁剪坐标。"""
        height, width = image_shape[-2:]
        if self.ensure_valid_crop:
            max_h_start = max(0, height - patch_size)
            max_w_start = max(0, width - patch_size)
            if max_h_start == 0 and height < patch_size:
                # 图像太小，直接裁剪整个图像（或通过插值），但这种情况我们返回None，让forward来处理
                return None, None
            h_start = torch.randint(0, max_h_start + 1, (1,)).item()
            w_start = torch.randint(0, max_w_start + 1, (1,)).item()
        else:
            h_start = torch.randint(0, height, (1,)).item()
            w_start = torch.randint(0, width, (1,)).item()
        return h_start, w_start

    def _grid_positions(self, length: int, patch_size: int) -> List[int]:
        """生成覆盖指定长度的网格起点，保证包含边缘。"""
        if self.ensure_valid_crop and length < patch_size:
            return []
        if length <= patch_size:
            return [0]

        stride = self.grid_stride if self.grid_stride is not None else patch_size
        stride = max(1, min(stride, patch_size))
        last_start = length - patch_size
        positions = list(range(0, last_start + 1, stride))
        if positions[-1] != last_start:
            positions.append(last_start)
        return positions

    def _enumerate_grid_coords(self, height: int, width: int, patch_size: int) -> List[Tuple[int, int]]:
        """枚举给定patch大小的所有网格坐标。"""
        y_positions = self._grid_positions(height, patch_size)
        x_positions = self._grid_positions(width, patch_size)
        return [(y, x) for y in y_positions for x in x_positions]

    def _crop_with_coords(self, image: torch.Tensor, h_start: int, w_start: int, patch_size: int) -> torch.Tensor:
        """根据给定的坐标裁剪patch。"""
        batch_size, channels, height, width = image.shape
        h_end = h_start + patch_size
        w_end = w_start + patch_size

        if self.ensure_valid_crop:
            return image[:, :, h_start:h_end, w_start:w_end]
        else:
            # 允许越界裁剪，并用0填充
            cropped_patch = torch.zeros(batch_size, channels, patch_size, patch_size, 
                                        device=image.device, dtype=image.dtype)
            
            valid_h_start = max(0, h_start)
            valid_h_end = min(height, h_end)
            valid_w_start = max(0, w_start)
            valid_w_end = min(width, w_end)
            
            patch_h_start = max(0, -h_start)
            patch_w_start = max(0, -w_start)
            
            if valid_h_end > valid_h_start and valid_w_end > valid_w_start:
                cropped_patch[:, :, 
                              patch_h_start:patch_h_start + (valid_h_end - valid_h_start),
                              patch_w_start:patch_w_start + (valid_w_end - valid_w_start)] = \
                    image[:, :, valid_h_start:valid_h_end, valid_w_start:valid_w_end]
            return cropped_patch

    def _compute_attention_weights(self, image: torch.Tensor) -> torch.Tensor:
        """计算图像的注意力权重，用于加权patch采样。"""
        # 简单的基于梯度的注意力权重计算
        if image.requires_grad:
            # 如果图像需要梯度，创建一个副本
            image_detached = image.detach()
        else:
            image_detached = image
        
        # 计算图像梯度作为注意力权重
        grad_x = torch.abs(image_detached[:, :, :, 1:] - image_detached[:, :, :, :-1])
        grad_y = torch.abs(image_detached[:, :, 1:, :] - image_detached[:, :, :-1, :])
        
        # 合并梯度
        attention_weights = torch.mean(grad_x, dim=1, keepdim=True) + torch.mean(grad_y, dim=1, keepdim=True)
        
        # 归一化到[0, 1]
        attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-8)
        
        return attention_weights

    def _get_attention_weighted_crop_coords(self, image: torch.Tensor, patch_size: int):
        """基于注意力权重生成裁剪坐标。"""
        attention_weights = self._compute_attention_weights(image)
        
        # 将注意力权重转换为概率分布
        weights_flat = attention_weights.view(attention_weights.shape[0], -1)
        weights_flat = weights_flat / (weights_flat.sum(dim=1, keepdim=True) + 1e-8)
        
        # 根据权重采样位置
        batch_size = image.shape[0]
        h_start_list = []
        w_start_list = []
        
        for b in range(batch_size):
            # 使用multinomial采样
            indices = torch.multinomial(weights_flat[b], 1, replacement=False)
            h_idx = indices.item() // attention_weights.shape[-1]
            w_idx = indices.item() % attention_weights.shape[-1]
            
            # 确保patch在图像范围内
            h_start = max(0, min(h_idx, attention_weights.shape[-2] - patch_size))
            w_start = max(0, min(w_idx, attention_weights.shape[-1] - patch_size))
            
            h_start_list.append(h_start)
            w_start_list.append(w_start)
        
        return h_start_list, w_start_list

    def forward(self, input: torch.Tensor, target: torch.Tensor = None, mode: str = "generator") -> torch.Tensor:
        """计算基于patch的对抗损失。
        
        Args:
            input: 输入图像，shape为(B, C, H, W)，值在[0, 1]范围内
            target: 目标图像（用于判别器训练），shape为(B, C, H, W)，值在[0, 1]范围内
            mode: 计算模式，"generator"或"discriminator"
            
        Returns:
            loss: 基于patch的对抗损失
        """
        batch_size = input.shape[0]
        total_loss = input.new_zeros(())
        
        # 如果图像太小，无法裁剪，则直接计算整张图的损失
        h, w = input.shape[-2:]
        if h < self.min_patch_size or w < self.min_patch_size:
            if mode == "generator":
                return -torch.mean(self.discriminator(input.float()))
            elif mode == "discriminator" and target is not None:
                logits_real = self.discriminator(input.float())
                logits_fake = self.discriminator(target.float())
                return hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)

        if self.patch_sampling_strategy == "grid":
            grid_patch_size = self._get_random_patch_size() if self.use_random_patch_size else self.patch_size
            coords = self._enumerate_grid_coords(h, w, grid_patch_size)
            if not coords:
                input_patch = F.interpolate(
                    input, size=(grid_patch_size, grid_patch_size), mode='bilinear', align_corners=False
                )
                target_patch = None
                if target is not None:
                    target_patch = F.interpolate(
                        target, size=(grid_patch_size, grid_patch_size), mode='bilinear', align_corners=False
                    )
                if mode == "generator":
                    return -torch.mean(self.discriminator(input_patch.float()))
                if mode == "discriminator" and target_patch is not None:
                    logits_real = self.discriminator(input_patch.float())
                    logits_fake = self.discriminator(target_patch.float())
                    return hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)
                raise ValueError(f"不支持的mode: {mode}")

            patch_count = 0
            for h_start, w_start in coords:
                input_patch = self._crop_with_coords(input, h_start, w_start, grid_patch_size)
                target_patch = None
                if target is not None:
                    target_patch = self._crop_with_coords(target, h_start, w_start, grid_patch_size)

                if mode == "generator":
                    patch_loss = -torch.mean(self.discriminator(input_patch.float()))
                elif mode == "discriminator" and target_patch is not None:
                    logits_real = self.discriminator(input_patch.float())
                    logits_fake = self.discriminator(target_patch.float())
                    patch_loss = hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)
                else:
                    raise ValueError(f"不支持的mode: {mode}")

                total_loss += patch_loss
                patch_count += 1

            if patch_count == 0:
                if mode == "generator":
                    return -torch.mean(self.discriminator(input.float()))
                if mode == "discriminator" and target is not None:
                    logits_real = self.discriminator(input.float())
                    logits_fake = self.discriminator(target.float())
                    return hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)
                raise ValueError(f"不支持的mode: {mode}")

            return total_loss / patch_count

        patch_count = 0
        num_samples = self.num_patches if self.num_patches and self.num_patches > 0 else 1

        for _ in range(num_samples):
            current_patch_size = self._get_random_patch_size()

            if self.patch_sampling_strategy == "random":
                h_start, w_start = self._get_random_crop_coords(input.shape, current_patch_size)
                if h_start is None:
                    input_patch = F.interpolate(
                        input, size=(current_patch_size, current_patch_size), mode='bilinear', align_corners=False
                    )
                    target_patch = None
                    if target is not None:
                        target_patch = F.interpolate(
                            target, size=(current_patch_size, current_patch_size), mode='bilinear', align_corners=False
                        )
                else:
                    input_patch = self._crop_with_coords(input, h_start, w_start, current_patch_size)
                    target_patch = None
                    if target is not None:
                        target_patch = self._crop_with_coords(target, h_start, w_start, current_patch_size)

            elif self.patch_sampling_strategy == "attention_weighted":
                h_start_list, w_start_list = self._get_attention_weighted_crop_coords(input, current_patch_size)
                input_patch = torch.stack([
                    self._crop_with_coords(input[i:i+1], h_start_list[i], w_start_list[i], current_patch_size)
                    for i in range(batch_size)
                ], dim=0)
                target_patch = None
                if target is not None:
                    target_patch = torch.stack([
                        self._crop_with_coords(target[i:i+1], h_start_list[i], w_start_list[i], current_patch_size)
                        for i in range(batch_size)
                    ], dim=0)
            else:
                raise ValueError(f"不支持的patch_sampling_strategy: {self.patch_sampling_strategy}")

            if mode == "generator":
                patch_loss = -torch.mean(self.discriminator(input_patch.float()))
            elif mode == "discriminator" and target_patch is not None:
                logits_real = self.discriminator(input_patch.float())
                logits_fake = self.discriminator(target_patch.float())
                patch_loss = hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)
            else:
                raise ValueError(f"不支持的mode: {mode}")

            total_loss += patch_loss
            patch_count += 1

        if patch_count == 0:
            if mode == "generator":
                return -torch.mean(self.discriminator(input.float()))
            if mode == "discriminator" and target is not None:
                logits_real = self.discriminator(input.float())
                logits_fake = self.discriminator(target.float())
                return hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)
            raise ValueError(f"不支持的mode: {mode}")

        return total_loss / patch_count
    
    def get_patch_info(self) -> dict:
        """获取当前patch配置信息。"""
        return {
            'patch_size': self.patch_size if not self.use_random_patch_size else 'random',
            'num_patches': self.num_patches,
            'min_patch_size': self.min_patch_size if self.use_random_patch_size else None,
            'max_patch_size': self.max_patch_size if self.use_random_patch_size else None,
            'use_random_patch_size': self.use_random_patch_size,
            'ensure_valid_crop': self.ensure_valid_crop,
            'patch_sampling_strategy': self.patch_sampling_strategy,
            'grid_stride': self.grid_stride
        }

class SD3SRDistillationLoss(nn.Module):
    """MSE alignment loss between student and teacher tokenizer encodings for SR distillation."""

    def __init__(self, config, model=None):
        super().__init__()
        losses_cfg = config.losses
        self.alignment_weight = losses_cfg.get('sr_alignment_weight', 1.0)
        self.quantizer_weight = losses_cfg.get('sr_quantizer_weight', 0.0)
        self.commitment_weight = losses_cfg.get('sr_commitment_weight', 0.0)
        self.mse = nn.MSELoss()
        self.model = model

    def forward(self, student_hidden, teacher_hidden, extra_result_dict=None, global_step=None, **kwargs):
        alignment_loss = self.mse(student_hidden, teacher_hidden.detach())
        base_loss = self.alignment_weight * alignment_loss
        quantizer_term = student_hidden.new_zeros(())
        commitment_term = student_hidden.new_zeros(())

        student_dict = None
        teacher_dict = None
        if isinstance(extra_result_dict, dict):
            student_dict = extra_result_dict.get('student_encoder_dict')
            teacher_dict = extra_result_dict.get('teacher_encoder_dict')

        if student_dict is not None:
            if 'quantizer_loss' in student_dict and self.quantizer_weight != 0.0:
                quantizer_term = self.quantizer_weight * student_dict['quantizer_loss']
            if 'commitment_loss' in student_dict and self.commitment_weight != 0.0:
                commitment_term = self.commitment_weight * student_dict['commitment_loss']

        total_loss = base_loss + quantizer_term + commitment_term

        loss_dict = {
            'total_loss': total_loss.detach(),
            'sr_alignment_loss': (self.alignment_weight * alignment_loss).detach(),
            'sr_alignment_raw': alignment_loss.detach(),
        }

        if quantizer_term.requires_grad or quantizer_term.detach().item() != 0.0:
            loss_dict['sr_quantizer_loss'] = quantizer_term.detach()
        if commitment_term.requires_grad or commitment_term.detach().item() != 0.0:
            loss_dict['sr_commitment_loss'] = commitment_term.detach()

        if teacher_dict is not None and 'quantizer_loss' in teacher_dict:
            loss_dict['teacher/quantizer_loss'] = teacher_dict['quantizer_loss'].detach()
        if student_dict is not None and 'codebook_loss' in student_dict:
            loss_dict['student/codebook_loss'] = student_dict['codebook_loss'].detach()

        return total_loss, loss_dict

    def is_discriminator_warmup_period(self, global_step: int) -> bool:
        """Check if we're in the discriminator warmup period.
        
        For distillation loss, we don't use adversarial training, so this always returns False.
        
        Args:
            global_step: Current global step number
            
        Returns:
            bool: Always False for distillation loss
        """
        return False

    def get_discriminator_training_frequency(self, global_step: int) -> int:
        """Get discriminator training frequency based on global step.
        
        For distillation loss, we don't use adversarial training, so this always returns 0.
        
        Args:
            global_step: Current global step number
            
        Returns:
            int: Always 0 for distillation loss (don't train discriminator)
        """
        return 0
