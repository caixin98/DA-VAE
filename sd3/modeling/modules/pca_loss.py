import torch
import torch.nn as nn
import numpy as np
from utils.latent_utils import unpack_latent_from_chw4
import os

class StaticPCALoss(nn.Module):
    def __init__(self, pca_basis_path="pca_basis_5000.pth", k=None, device='cuda'):
        """
        初始化静态PCA Loss模块.
        Args:
            pca_basis_path (str): 预计算好的PCA基和均值文件的路径.
            k (int, optional): 如果指定，则只使用前k个主成分。默认为None。
            device (str): 计算所在的设备 ('cuda' or 'cpu').
        """
        super(StaticPCALoss, self).__init__()
        return
        # 加载预计算好的数据
        if os.path.exists(pca_basis_path):
            pca_data = torch.load(pca_basis_path, map_location='cpu')
            # 检查是否有完整的PCA对象
            if 'pca_object' in pca_data:
                # 新格式：使用sklearn PCA对象
                self.pca_object = pca_data['pca_object']
                self.mean = torch.from_numpy(self.pca_object.mean_).to(device)
                principal_components = torch.from_numpy(self.pca_object.components_.T).to(device)
                self.explained_variance_ratio = torch.from_numpy(self.pca_object.explained_variance_ratio_).to(device)
                print(f"加载sklearn PCA对象，解释方差比例: {self.explained_variance_ratio[:5]}")
            else:
                # 旧格式：直接使用保存的mean和principal_components
                self.pca_object = None
                self.mean = pca_data['mean'].to(device)
                principal_components = pca_data['principal_components'].to(device)
                self.explained_variance_ratio = None
                print("加载旧格式PCA数据")

            # 如果指定了k，只取前k个主成分
            if k is not None and k < principal_components.shape[1]:
                self.principal_components = principal_components[:, :k]
                if self.explained_variance_ratio is not None:
                    self.explained_variance_ratio = self.explained_variance_ratio[:k]
                    print(f"使用前{k}个主成分，累计解释方差: {self.explained_variance_ratio.sum().item():.4f}")
                else:
                    print(f"使用前{k}个主成分")
            else:
                self.principal_components = principal_components
                if self.explained_variance_ratio is not None:
                    print(f"使用所有主成分，累计解释方差: {self.explained_variance_ratio.sum().item():.4f}")
                else:
                    print(f"使用所有主成分")
            
    def forward(self, x, x_hat):
        """
        计算PCA Loss.
        Args:
            x (torch.Tensor): 原始输入张量，形状为 (B, C, H, W)。
            x_hat (torch.Tensor): VAE重建的输出张量，形状为 (B, C, H, W)。
        Returns:
            tuple: (total_loss, channel_losses_dict) 其中total_loss是总loss，channel_losses_dict包含每个channel的loss统计
        """

        if x.shape[1] == 64:
            unpacked_x = unpack_latent_from_chw4(x)
            unpacked_x_hat = unpack_latent_from_chw4(x_hat)
            x = unpacked_x
            x_hat = unpacked_x_hat
        channels = x.shape[1]
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(-1, channels)
        x_hat_reshaped = x_hat.permute(0, 2, 3, 1).contiguous().view(-1, channels)
        
        # 2. 中心化 (使用预计算的均值)
        x_centered = x_reshaped - self.mean
        x_hat_centered = x_hat_reshaped - self.mean

        # 3. 投影到主成分上
        coeffs_x = torch.matmul(x_centered, self.principal_components)
        coeffs_x_hat = torch.matmul(x_hat_centered, self.principal_components)

        # 4. 计算总MSE Loss
        total_loss = nn.functional.mse_loss(coeffs_x, coeffs_x_hat)
        
        # 5. 计算每个PCA channel的loss
        channel_losses_dict = {}
        n_components = coeffs_x.shape[1]
        
        # 计算每个PCA component的MSE loss
        for i in range(n_components):
            channel_loss = nn.functional.mse_loss(coeffs_x[:, i], coeffs_x_hat[:, i])
            channel_losses_dict[f'pca_channel_{i:03d}_loss'] = channel_loss.item()
            
            # 如果有解释方差比例信息，也记录下来
            if self.explained_variance_ratio is not None and i < len(self.explained_variance_ratio):
                channel_losses_dict[f'pca_channel_{i:03d}_explained_variance'] = self.explained_variance_ratio[i].item()
        
        # 添加一些统计信息
        channel_losses_dict['pca_total_loss'] = total_loss.item()
        channel_losses_dict['pca_n_components'] = n_components
        
        # 计算前几个重要channel的loss统计
        if n_components > 0:
            # 前5个channel的loss
            top_k = min(5, n_components)
            top_losses = [channel_losses_dict[f'pca_channel_{i:03d}_loss'] for i in range(top_k)]
            channel_losses_dict['pca_top5_avg_loss'] = sum(top_losses) / len(top_losses)
            channel_losses_dict['pca_top5_max_loss'] = max(top_losses)
            channel_losses_dict['pca_top5_min_loss'] = min(top_losses)
            
            # 所有channel的loss统计
            all_losses = [channel_losses_dict[f'pca_channel_{i:03d}_loss'] for i in range(n_components)]
            channel_losses_dict['pca_all_avg_loss'] = sum(all_losses) / len(all_losses)
            channel_losses_dict['pca_all_max_loss'] = max(all_losses)
            channel_losses_dict['pca_all_min_loss'] = min(all_losses)
        
        return total_loss, channel_losses_dict
    
    def get_explained_variance_ratio(self):
        """获取解释方差比例"""
        if self.explained_variance_ratio is not None:
            return self.explained_variance_ratio
        else:
            return None
    
    def get_principal_components_info(self):
        """获取主成分信息"""
        return {
            'n_components': self.principal_components.shape[1],
            'n_features': self.principal_components.shape[0],
            'explained_variance_ratio': self.explained_variance_ratio,
            'has_pca_object': self.pca_object is not None
        }
    
    def transform_data(self, data):
        """
        使用PCA转换数据（用于调试或分析）
        Args:
            data (torch.Tensor): 输入数据，形状为 (N, C)
        Returns:
            torch.Tensor: 转换后的PCA系数
        """
        data_centered = data - self.mean
        return torch.matmul(data_centered, self.principal_components)