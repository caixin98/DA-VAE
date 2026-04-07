"""This file contains code for Decoder LPIPS loss based on AutoencoderKL decoder features.

This file may have been modified by Bytedance Ltd. and/or its affiliates ("Bytedance's Modifications").
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference:
    https://github.com/richzhang/PerceptualSimilarity/
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/lpips.py
    https://github.com/CompVis/taming-transformers/blob/master/taming/util.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


def normalize_tensor(x, eps=1e-10):
    """Normalize tensor along channel dimension."""
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    """Compute spatial average over height and width dimensions."""
    return x.mean([2, 3], keepdim=keepdim)


def normalize_features_with_stats(features, reference_stats, eps=1e-8):
    """
    使用参考特征统计来归一化特征，确保一致的归一化。
    
    根据论文，这里应该实现类似L2归一化的效果，但使用去噪latents的统计信息
    来确保两个张量使用相同的归一化标准。
    
    Args:
        features: 要归一化的特征字典
        reference_stats: 参考特征统计字典，包含每个特征的均值和方差
        eps: 数值稳定性参数
    
    Returns:
        归一化后的特征字典
    """
    normalized_features = {}
    
    for name, feat in features.items():
        if name in reference_stats:
            # 使用参考特征的统计信息进行归一化
            # 这里实现类似L2归一化的效果，但使用参考统计
            mean = reference_stats[name]['mean']
            std = reference_stats[name]['std']
            
            # 先减去均值，然后除以标准差
            normalized_feat = (feat - mean) / (std + eps)
            
            # 然后进行L2归一化，确保每个通道的范数为1
            norm_factor = torch.sqrt(torch.sum(normalized_feat**2, dim=1, keepdim=True))
            normalized_feat = normalized_feat / (norm_factor + eps)
            
            normalized_features[name] = normalized_feat
        else:
            # 如果没有参考统计，使用原始L2归一化
            normalized_features[name] = normalize_tensor(feat)
    
    return normalized_features


def compute_feature_stats(features):
    """
    计算特征的统计信息（均值和方差）。
    
    Args:
        features: 特征字典
    
    Returns:
        统计信息字典，包含每个特征的均值和方差
    """
    stats = {}
    
    for name, feat in features.items():
        # 计算每个通道的均值和方差
        # feat shape: [B, C, H, W]
        mean = feat.mean(dim=[0, 2, 3], keepdim=True)  # [1, C, 1, 1]
        var = feat.var(dim=[0, 2, 3], keepdim=True)    # [1, C, 1, 1]
        std = torch.sqrt(var + 1e-8)
        
        stats[name] = {
            'mean': mean,
            'std': std
        }
    
    return stats


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv."""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = []
        if use_dropout:
            layers.append(nn.Dropout())

        layers.append(
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the linear layer."""
        return self.model(x)


class DecoderLPIPS(nn.Module):
    """Learned perceptual metric based on AutoencoderKL decoder features."""

    def __init__(self, decoder: nn.Module, use_dropout: bool = True,
                 feature_names: Optional[List[str]] = None, latent_channels: int = 16,
                 use_depth_weighting: bool = True, use_feature_normalization: bool = True):
        super().__init__()
        self.decoder = decoder
        self.latent_channels = latent_channels
        self.use_depth_weighting = use_depth_weighting
        self.use_feature_normalization = use_feature_normalization

        # 获取特征通道数 - 固定使用默认的6层
        self.feature_names = self._get_default_feature_names()
        self.chns = self._get_feature_channels()

        # 计算深度特定权重
        if self.use_depth_weighting:
            self.depth_weights = self._compute_depth_weights()
        else:
            self.depth_weights = None

        # 创建线性层来处理每个特征层
        self.lin_layers = nn.ModuleList([
            NetLinLayer(chn_in, chn_out=1, use_dropout=use_dropout)
            for chn_in in self.chns
        ])

        # 将线性层权重初始化为全1（非负），并保持冻结，避免产生负损
        for lin_layer in self.lin_layers:
            for layer in lin_layer.model:
                if hasattr(layer, 'weight') and layer.weight is not None:
                    with torch.no_grad():
                        layer.weight.data.fill_(1.0)

        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False
        
        # 确保线性层在正确的设备上
        if hasattr(self.decoder, 'vae'):
            target_device = next(self.decoder.vae.parameters()).device
            target_dtype = next(self.decoder.vae.parameters()).dtype
            for lin_layer in self.lin_layers:
                lin_layer.to(device=target_device, dtype=target_dtype)

    def to(self, device):
        """将模型移动到指定设备。"""
        super().to(device)
        # 确保线性层也在正确的设备上
        for lin_layer in self.lin_layers:
            lin_layer.to(device)
        # 确保decoder也在正确的设备上
        if hasattr(self.decoder, 'vae'):
            self.decoder.vae.to(device)
        return self

    def _get_default_feature_names(self) -> List[str]:
        """获取默认的特征层名称，基于decoder结构。"""
        return [
            'conv_in',
            'mid_block',
            'up_blocks.0',
            'up_blocks.1',
            'up_blocks.2',
            'up_blocks.3'
        ]

    def _get_feature_channels(self) -> List[int]:
        """获取每个特征层的通道数。根据VAE类型自动检测。"""
        # 检查是否是SD3 VAE
        if hasattr(self.decoder, 'vae') and hasattr(self.decoder.vae, 'config'):
            # SD3 VAE的通道数配置
            if hasattr(self.decoder.vae.config, 'block_out_channels'):
                block_out_channels = self.decoder.vae.config.block_out_channels
                # SD3 VAE: conv_in -> mid_block -> up_blocks
                # conv_in: 512, mid_block: 512, up_blocks: [512, 512, 256, 128]
                return [512, 512] + block_out_channels[::-1]  # 反向，因为up_blocks是倒序的
        
        # 默认FLUX VAE的通道数配置
        return [512, 512, 512, 512, 256, 128]

    def _compute_depth_weights(self) -> List[float]:
        """
        计算深度特定权重：ωl = 2^(-rl/r1)
        
        其中：
        - rl 是第l层的分辨率
        - r1 是第一层的分辨率（作为基准）
        
        根据VAE类型自动调整分辨率倍数：
        - SD3 VAE: 实际只有8倍上采样 (1x, 1x, 2x, 4x, 8x, 8x)
        - FLUX VAE: 可能有16倍上采样 (1x, 1x, 2x, 4x, 8x, 16x)
        """
        # 检查VAE类型并设置相应的分辨率倍数
        if hasattr(self.decoder, 'vae') and hasattr(self.decoder.vae, 'config'):
            # SD3 VAE: 实际只有8倍上采样，最后一个up_block没有继续上采样
            layer_resolution_factors = [1, 1, 2, 4, 8, 8]  # 6层
        else:
            # FLUX VAE: 可能有16倍上采样
            layer_resolution_factors = [1, 1, 2, 4, 8, 8]  # 6层
        
        # 确保特征层数量匹配
        num_layers = len(self.feature_names)
        if len(layer_resolution_factors) != num_layers:
            # 如果层数不匹配，使用默认配置
            layer_resolution_factors = [1, 1, 2, 4, 8, 8][:num_layers]
        
        # 计算权重：ωl = 2^(-rl/r1) = 2^(-resolution_factor)
        depth_weights = [2.0 ** (-factor) for factor in layer_resolution_factors]
        
        print(f"🔧 深度权重计算完成 (VAE类型: {'SD3' if hasattr(self.decoder, 'vae') else 'FLUX'}):")
        for i, (name, weight) in enumerate(zip(self.feature_names, depth_weights)):
            if i < len(layer_resolution_factors):
                print(f"  {name}: 分辨率倍数={layer_resolution_factors[i]}x, 权重={weight:.4f}")
            else:
                print(f"  {name}: 权重={weight:.4f}")
        
        return depth_weights

    def _extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """直接提取特征，不使用hooks。"""
        features = {}

        # 检查是否是SD3 VAE
        if hasattr(self.decoder, 'vae'):
            # SD3 VAE: 通过vae.decoder访问
            decoder = self.decoder.vae.decoder
        else:
            # 直接使用decoder
            decoder = self.decoder

        # 确保输入在正确的设备和数据类型上
        target_device = next(decoder.parameters()).device
        target_dtype = next(decoder.parameters()).dtype
        
        if x.device != target_device:
            x = x.to(target_device)
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        # 手动提取conv_in特征
        h = decoder.conv_in(x)
        features['conv_in'] = h

        # 手动提取mid_block特征
        h = decoder.mid_block(h)
        features['mid_block'] = h

        # 手动提取up_blocks特征
        for i, up_block in enumerate(decoder.up_blocks):
            h = up_block(h)
            features[f'up_blocks.{i}'] = h

        return features

    def compute_feature_stats(self, features):
        """计算特征的统计信息（均值和方差）。"""
        return compute_feature_stats(features)
    
    def normalize_features_with_stats(self, features, reference_stats):
        """使用参考特征统计来归一化特征。"""
        return normalize_features_with_stats(features, reference_stats)
    
    def normalize_tensor(self, x, eps=1e-10):
        """归一化张量。"""
        return normalize_tensor(x, eps)

    def forward(self, pred_latent: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
        """
        计算基于decoder中间特征的LPIPS loss。

        Args:
            pred_latent: 预测的latent tensor
            target_latent: 目标的latent tensor (去噪latents，用作归一化参考)

        Returns:
            LPIPS loss值
        """
        # 提取特征
        pred_features = self._extract_features(pred_latent)
        target_features = self._extract_features(target_latent)

        # 特征归一化
        if self.use_feature_normalization:
            # 使用去噪latents (target_latent) 的统计来归一化两个张量
            # 先计算目标特征的统计信息
            target_stats = compute_feature_stats(target_features)
            
            # 使用目标特征的统计信息来归一化两个张量，确保一致性
            pred_features = normalize_features_with_stats(pred_features, target_stats)
            target_features = normalize_features_with_stats(target_features, target_stats)
        else:
            # 使用原始的L2归一化
            pred_features = {name: normalize_tensor(feat) for name, feat in pred_features.items()}
            target_features = {name: normalize_tensor(feat) for name, feat in target_features.items()}

        # 计算每个特征层的差异
        res = []

        for i, (name, lin_layer) in enumerate(zip(self.feature_names, self.lin_layers)):
            if name in pred_features and name in target_features:
                pred_feat = pred_features[name]
                target_feat = target_features[name]

                # 确保特征形状匹配
                if pred_feat.shape != target_feat.shape:
                    if pred_feat.shape[2:] != target_feat.shape[2:]:
                        target_feat = F.interpolate(
                            target_feat, size=pred_feat.shape[2:], mode='bilinear', align_corners=False
                        )

                # 计算L2差异
                diff = (pred_feat - target_feat) ** 2

                # 确保数据类型匹配
                conv_layer = None
                for layer in lin_layer.model:
                    if hasattr(layer, 'weight'):
                        conv_layer = layer
                        break

                if conv_layer is not None and diff.dtype != conv_layer.weight.dtype:
                    diff = diff.to(conv_layer.weight.dtype)

                # 通过线性层并计算空间平均
                res.append(spatial_average(lin_layer(diff), keepdim=True))

        # 如果没有有效的特征，返回零
        if not res:
            return torch.tensor(0.0, device=pred_latent.device, dtype=pred_latent.dtype)

        # 累加所有层的loss，应用深度权重
        if self.use_depth_weighting and self.depth_weights is not None:
            # 使用深度权重平衡不同层的贡献
            val = res[0] * self.depth_weights[0]
            for l in range(1, len(res)):
                if l < len(self.depth_weights):
                    val += res[l] * self.depth_weights[l]
                else:
                    val += res[l]  # 如果权重数量不足，使用原始值
        else:
            # 不使用深度权重，直接累加
            val = res[0]
            for l in range(1, len(res)):
                val += res[l]

        # 确保返回标量
        if val.numel() > 1:
            val = val.mean()
        # 数值稳定：LPIPS按理应为非负，做一次下界截断
        return torch.clamp(val, min=0.0)


class DecoderLPIPSLoss(nn.Module):
    """简化的Decoder LPIPS Loss接口。"""

    def __init__(self, decoder: nn.Module, use_dropout: bool = True, latent_channels: int = 16,
                 use_depth_weighting: bool = True, use_feature_normalization: bool = True):
        super().__init__()
        self.decoder_lpips = DecoderLPIPS(decoder, use_dropout=use_dropout, latent_channels=latent_channels,
                                        use_depth_weighting=use_depth_weighting, 
                                        use_feature_normalization=use_feature_normalization)

    def forward(self, pred_latent: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
        """计算loss的简化接口。"""
        return self.decoder_lpips(pred_latent, target_latent)

    def to(self, device):
        """将模型移动到指定设备。"""
        super().to(device)
        self.decoder_lpips.to(device)
        return self
