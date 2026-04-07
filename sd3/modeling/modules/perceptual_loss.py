"""This file contains perceptual loss module using LPIPS and ConvNeXt-S.

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
"""

from typing import List, Optional

import torch
import torch.nn as nn

from torchvision import models
from .lpips import LPIPS

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

 
class PerceptualLoss(torch.nn.Module):
    def __init__(self, model_name: str = "convnext_s"):
        """Initializes the PerceptualLoss class.

        Args:
            model_name: A string, the name of the perceptual loss model to use.

        Raise:
            ValueError: If the model_name does not contain "lpips" or "convnext_s".
        """
        super().__init__()
        if ("lpips" not in model_name) and (
            "convnext_s" not in model_name):
            raise ValueError(f"Unsupported Perceptual Loss model name {model_name}")
        self.lpips = None
        self.convnext = None
        self.loss_weight_lpips = None
        self.loss_weight_convnext = None

        # Parsing the model name. We support name formatted in
        # "lpips-convnext_s-{float_number}-{float_number}", where the 
        # {float_number} refers to the loss weight for each component.
        # E.g., lpips-convnext_s-1.0-2.0 refers to compute the perceptual loss
        # using both the convnext_s and lpips, and average the final loss with
        # (1.0 * loss(lpips) + 2.0 * loss(convnext_s)) / (1.0 + 2.0).
        if "lpips" in model_name:
            self.lpips = LPIPS().eval()

        if "convnext_s" in model_name:
            self.convnext = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1).eval()

        if "lpips" in model_name and "convnext_s" in model_name:
            loss_config = model_name.split('-')[-2:]
            self.loss_weight_lpips, self.loss_weight_convnext = float(loss_config[0]), float(loss_config[1])
            print(f"self.loss_weight_lpips, self.loss_weight_convnext: {self.loss_weight_lpips}, {self.loss_weight_convnext}")

        self.register_buffer("imagenet_mean", torch.Tensor(_IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("imagenet_std", torch.Tensor(_IMAGENET_STD)[None, :, None, None])

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        # Always in eval mode.
        self.eval()
        loss = 0.
        num_losses = 0.
        lpips_loss = 0.
        convnext_loss = 0.
        # Computes LPIPS loss, if available.
        if self.lpips is not None:
            lpips_loss = self.lpips(input, target)
            if self.loss_weight_lpips is None:
                loss += lpips_loss
                num_losses += 1
            else:
                num_losses += self.loss_weight_lpips
                loss += self.loss_weight_lpips * lpips_loss

        if self.convnext is not None:
            # Computes ConvNeXt-s loss, if available.
            input = torch.nn.functional.interpolate(input, size=224, mode="bilinear", align_corners=False, antialias=True)
            target = torch.nn.functional.interpolate(target, size=224, mode="bilinear", align_corners=False, antialias=True)
            pred_input = self.convnext((input - self.imagenet_mean) / self.imagenet_std)
            pred_target = self.convnext((target - self.imagenet_std) / self.imagenet_std)
            convnext_loss = torch.nn.functional.mse_loss(
                pred_input,
                pred_target,
                reduction="mean")
                
            if self.loss_weight_convnext is None:
                num_losses += 1
                loss += convnext_loss
            else:
                num_losses += self.loss_weight_convnext
                loss += self.loss_weight_convnext * convnext_loss
        
        # weighted avg.
        loss = loss / num_losses
        return loss


def extract_patches(image, patch_size, stride):
    """将图像分割成重叠的patches。
    
    Args:
        image: 输入图像，shape为(B, C, H, W)
        patch_size: patch的大小
        stride: patch之间的步长
        
    Returns:
        patches: 分割后的patches，shape为(B, C, num_patches, patch_size, patch_size)
    """
    # 使用unfold操作将图像分割成patches
    patches = image.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    # 重新整形为batch of patches
    patches = patches.contiguous().view(image.size(0), image.size(1), -1, patch_size, patch_size)
    return patches


class PatchBasedLPIPSLoss(nn.Module):
    """基于patch的LPIPS损失，支持固定网格或随机采样。
    
    这个类可以使用固定的网格划分或随机起始位置来提取局部patch并计算LPIPS损失，
    以在稳定性和覆盖范围之间取得平衡。
    """
    
    def __init__(self, 
                 lpips_model: nn.Module,
                 patch_size: int = 256,
                 stride: int = 128,
                 sampling: str = "grid",
                 num_random_patches: Optional[int] = None):
        """初始化基于patch的LPIPS损失。
        
        Args:
            lpips_model: 预训练的LPIPS模型
            patch_size: patch的大小，默认256
            stride: patch之间的步长，默认128（重叠度为50%）
            sampling: 采样方式，可选"grid"或"random"
            num_random_patches: 随机采样时的patch数量，默认按照stride计算
        """
        super().__init__()
        self.lpips_model = lpips_model
        self.patch_size = patch_size
        self.stride = stride
        self.sampling = sampling.lower()
        self.num_random_patches = num_random_patches
        
        # 确保patch_size和stride合理
        assert patch_size > 0, "patch_size必须大于0"
        assert stride > 0, "stride必须大于0"
        assert stride <= patch_size, "stride不能大于patch_size"
        if self.sampling not in {"grid", "random"}:
            raise ValueError("sampling必须是'grid'或'random'")
        if self.num_random_patches is not None and self.num_random_patches <= 0:
            raise ValueError("num_random_patches必须大于0")
        
        # 将LPIPS模型设为eval模式并冻结参数
        self.lpips_model.eval()
        for param in self.lpips_model.parameters():
            param.requires_grad = False

    def forward(self, real_images: torch.Tensor, recon_images: torch.Tensor) -> torch.Tensor:
        """计算基于patch的LPIPS损失。
        
        Args:
            real_images: 真实图像，shape为(B, C, H, W)，值在[0, 1]范围内
            recon_images: 重建图像，shape为(B, C, H, W)，值在[0, 1]范围内
            
        Returns:
            loss: 基于patch的LPIPS损失
        """
        # 确保输入和目标图像尺寸相同
        assert real_images.shape == recon_images.shape, "Real and recon images must have the same shape"
        
        # 如果图像太小，无法分割成patches，则直接计算整张图的损失
        h, w = real_images.shape[-2:]
        if h < self.patch_size or w < self.patch_size:
            return self.lpips_model(real_images, recon_images).mean()
        
        if self.sampling == "random":
            return self._random_sampling_loss(real_images, recon_images, h, w)
        return self._grid_sampling_loss(real_images, recon_images, h, w)

    def _grid_sampling_loss(
        self,
        real_images: torch.Tensor,
        recon_images: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """按照固定网格采样计算LPIPS损失，并覆盖边缘区域。"""
        y_positions = self._grid_positions(height)
        x_positions = self._grid_positions(width)
        num_patches = len(y_positions) * len(x_positions)

        lpips_loss = real_images.new_zeros(())
        for y in y_positions:
            for x in x_positions:
                real_patch = real_images[:, :, y:y + self.patch_size, x:x + self.patch_size].contiguous()
                recon_patch = recon_images[:, :, y:y + self.patch_size, x:x + self.patch_size].contiguous()
                patch_lpips_loss = self.lpips_model(real_patch, recon_patch).mean()
                if not torch.isfinite(patch_lpips_loss):
                    patch_lpips_loss = torch.tensor(0.0, device=real_patch.device, dtype=real_patch.dtype)
                lpips_loss += patch_lpips_loss

        return lpips_loss / num_patches

    def _grid_positions(self, length: int) -> List[int]:
        """生成覆盖指定长度的网格起点，保证包含边缘。"""
        if length <= self.patch_size:
            return [0]

        last_start = length - self.patch_size
        positions = list(range(0, last_start + 1, self.stride))
        if positions[-1] != last_start:
            positions.append(last_start)
        return positions

    def _random_sampling_loss(self,
                              real_images: torch.Tensor,
                              recon_images: torch.Tensor,
                              height: int,
                              width: int) -> torch.Tensor:
        """按照随机位置采样计算LPIPS损失。"""
        num_patches = max(1, self._default_random_patch_count(height, width))
        max_y = height - self.patch_size
        max_x = width - self.patch_size
        device = real_images.device
        
        if max_y == 0:
            y_coords = torch.zeros(num_patches, device=device, dtype=torch.long)
        else:
            y_coords = torch.randint(0, max_y + 1, (num_patches,), device=device)
        if max_x == 0:
            x_coords = torch.zeros(num_patches, device=device, dtype=torch.long)
        else:
            x_coords = torch.randint(0, max_x + 1, (num_patches,), device=device)
        
        lpips_loss = 0.0
        for idx in range(num_patches):
            y = int(y_coords[idx].item())
            x = int(x_coords[idx].item())
            real_patch = real_images[:, :, y:y + self.patch_size, x:x + self.patch_size].contiguous()
            recon_patch = recon_images[:, :, y:y + self.patch_size, x:x + self.patch_size].contiguous()
            patch_lpips_loss = self.lpips_model(real_patch, recon_patch).mean()
            if not torch.isfinite(patch_lpips_loss):
                patch_lpips_loss = torch.tensor(0.0, device=real_patch.device, dtype=real_patch.dtype)
            lpips_loss += patch_lpips_loss
        
        return lpips_loss / num_patches

    def _default_random_patch_count(self, height: int, width: int) -> int:
        """计算随机采样时的默认patch数量。"""
        if self.num_random_patches is not None:
            return self.num_random_patches
        vertical_steps = max(1, (max(height - self.patch_size, 0) // self.stride) + 1)
        horizontal_steps = max(1, (max(width - self.patch_size, 0) // self.stride) + 1)
        return vertical_steps * horizontal_steps
    
    def get_patch_info(self) -> dict:
        """获取当前patch配置信息。"""
        info = {
            'patch_size': self.patch_size,
            'stride': self.stride,
            'overlap_ratio': 1.0 - (self.stride / self.patch_size),
            'sampling': self.sampling,
        }
        if self.sampling == 'random':
            info['num_random_patches'] = self.num_random_patches
        return info



class PatchBasedGramLoss(nn.Module):
    """基于固定网格patch的Gram损失。
    
    这个类使用固定的网格划分而不是随机裁剪，提供更稳定的训练过程。
    通过将图像分割成重叠的patches并计算每个patch的Gram损失，可以：
    1. 提供稳定的训练过程（无随机性）
    2. 关注局部细节，提高重建质量
    3. 减少计算量（特别是对于大图像）
    4. 保持空间一致性
    """
    
    def __init__(self, 
                 gram_model: nn.Module,
                 patch_size: int = 256,
                 stride: int = 128):
        """初始化基于patch的Gram损失。
        
        Args:
            gram_model: 预训练的Gram损失模型
            patch_size: patch的大小，默认256
            stride: patch之间的步长，默认128（重叠度为50%）
        """
        super().__init__()
        self.gram_model = gram_model
        self.patch_size = patch_size
        self.stride = stride
        
        # 确保patch_size和stride合理
        assert patch_size > 0, "patch_size必须大于0"
        assert stride > 0, "stride必须大于0"
        assert stride <= patch_size, "stride不能大于patch_size"
        
        # 将Gram模型设为eval模式并冻结参数
        self.gram_model.eval()
        for param in self.gram_model.parameters():
            param.requires_grad = False

    def forward(self, real_images: torch.Tensor, recon_images: torch.Tensor) -> torch.Tensor:
        """计算基于patch的Gram损失。
        
        Args:
            real_images: 真实图像，shape为(B, C, H, W)，值在[0, 1]范围内
            recon_images: 重建图像，shape为(B, C, H, W)，值在[0, 1]范围内
            
        Returns:
            loss: 基于patch的Gram损失
        """
        # 确保输入和目标图像尺寸相同
        assert real_images.shape == recon_images.shape, "Real and recon images must have the same shape"
        
        # 如果图像太小，无法分割成patches，则直接计算整张图的损失
        h, w = real_images.shape[-2:]
        if h < self.patch_size or w < self.patch_size:
            return self.gram_model(real_images, recon_images)
        
        # 提取patches
        real_patches = extract_patches(real_images, self.patch_size, self.stride)
        recon_patches = extract_patches(recon_images, self.patch_size, self.stride)
        
        gram_loss = 0.0
        num_patches = real_patches.size(2)  # 获取patch数量
        
        # 遍历每个patch并累积Gram损失
        for i in range(num_patches):
            real_patch = real_patches[:, :, i, :, :].contiguous()
            recon_patch = recon_patches[:, :, i, :, :].contiguous()
            
            # 计算当前patch的Gram损失
            patch_gram_loss = self.gram_model(real_patch, recon_patch)
            
            # 处理非有限值（NaN或Inf）
            if not torch.isfinite(patch_gram_loss):
                patch_gram_loss = torch.tensor(0.0, device=real_patch.device, dtype=real_patch.dtype)
            
            gram_loss += patch_gram_loss

        # 按patch数量归一化
        return gram_loss / num_patches
    
    def get_patch_info(self) -> dict:
        """获取当前patch配置信息。"""
        return {
            'patch_size': self.patch_size,
            'stride': self.stride,
            'overlap_ratio': 1.0 - (self.stride / self.patch_size)
        }