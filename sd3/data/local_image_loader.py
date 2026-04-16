"""本地图像数据加载器：读取本地高质量图像，返回条件图像和输入图像

参考 comprehensive_evaluation.py 的实现，确保图像尺寸是32的倍数，
返回2x下采样的条件图像和原始输入图像。

Copyright (2024) Bytedance Ltd. and/or its affiliates
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import random


class LocalImageDataset(Dataset):
    """从本地文件夹读取图像的数据集，返回条件图像和输入图像"""
    
    def __init__(self, image_dir, max_samples=None, seed=42, use_precomputed_cond=True, target_size=1024, use_tensor_cache=False, downsample_factor=2, skip_cond_image=False):
        """
        初始化本地图像数据集
        
        Args:
            image_dir (str): 图像目录路径
            max_samples (int, optional): 最大样本数量，None表示使用所有图像
            seed (int): 随机种子，用于打乱图像顺序
            use_precomputed_cond (bool): 是否使用预计算的条件图像
            target_size (int): 目标图像尺寸，与S3数据加载器保持一致
            use_tensor_cache (bool): 是否使用tensor缓存（避免图像文件精度损失）
            downsample_factor (int): 下采样倍率，默认为2。图像尺寸会被调整到 downsample_factor * 16 的倍数
            skip_cond_image (bool): 是否跳过cond_image准备，让模型在tensor层面处理
        """
        self.image_dir = image_dir
        self.image_files = []
        self.cond_image_files = []
        self.cond_image_mapping = {}  # 存储原始图像到条件图像的映射
        self.seed = seed
        self.use_precomputed_cond = use_precomputed_cond
        self.target_size = target_size
        self.use_tensor_cache = use_tensor_cache
        self.downsample_factor = downsample_factor
        self.skip_cond_image = skip_cond_image
        
        # 如果使用tensor缓存，尝试加载缓存文件
        if self.use_tensor_cache:
            cache_file = os.path.join(os.path.dirname(image_dir), 'tensor_cache.pt')
            if os.path.exists(cache_file):
                print(f"加载tensor缓存: {cache_file}")
                cache_data = torch.load(cache_file)
                self.tensor_cache = cache_data
                self.use_tensor_cache = True
                print(f"缓存中包含 {len(self.tensor_cache)} 个样本")
                return
            else:
                print(f"警告: 找不到tensor缓存文件 {cache_file}，将使用图像文件加载")
                self.use_tensor_cache = False
        
        # 支持的图像格式
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        
        # 读取原始图像
        for ext in extensions:
            self.image_files.extend(glob.glob(os.path.join(image_dir, ext)))
            self.image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
        # 如果使用预计算的条件图像，尝试读取cond_images目录
        if self.use_precomputed_cond:
            cond_dir = os.path.join(os.path.dirname(image_dir), 'cond_images')
            if os.path.exists(cond_dir):
                for ext in extensions:
                    self.cond_image_files.extend(glob.glob(os.path.join(cond_dir, ext)))
                    self.cond_image_files.extend(glob.glob(os.path.join(cond_dir, ext.upper())))
                self.cond_image_files.sort()
                print(f"找到 {len(self.cond_image_files)} 个预计算的条件图像在 {cond_dir}")
                
                # 建立原始图像到条件图像的映射关系
                self._build_cond_image_mapping()
            else:
                print(f"警告: 找不到条件图像目录 {cond_dir}，将重新生成条件图像")
                self.use_precomputed_cond = False
        
        # 按文件名排序，确保顺序一致
        self.image_files.sort()
        
        # 设置随机种子并打乱顺序
        random.seed(seed)
        random.shuffle(self.image_files)
        
        # 限制样本数量
        if max_samples is not None:
            self.image_files = self.image_files[:max_samples]
        
        print(f"找到 {len(self.image_files)} 个图像文件在 {image_dir}")
        print(f"使用随机种子: {seed}")
        if self.use_precomputed_cond:
            print("使用预计算的条件图像")
        else:
            print("重新生成条件图像")
    
    def _build_cond_image_mapping(self):
        """建立原始图像到条件图像的映射关系
        
        使用特定的命名规则：
        - 原始图像: original_0000.png
        - 条件图像: cond_0000.png
        """
        # 建立映射关系：原始图像路径 -> 对应的条件图像路径
        for img_path in self.image_files:
            img_name = Path(img_path).stem
            
            # 检查是否是 original_ 开头的文件
            if img_name.startswith('original_'):
                # 提取数字部分
                number_part = img_name.replace('original_', '')
                # 构造对应的条件图像文件名
                cond_name = f'cond_{number_part}'
                
                # 查找对应的条件图像文件
                cond_image_path = None
                for cond_path in self.cond_image_files:
                    cond_path_stem = Path(cond_path).stem
                    if cond_path_stem == cond_name:
                        cond_image_path = cond_path
                        break
                
                if cond_image_path:
                    self.cond_image_mapping[img_path] = cond_image_path
                    print(f"映射: {img_name} -> {Path(cond_image_path).stem}")
                else:
                    print(f"警告: 找不到原始图像 {img_name} 对应的条件图像 {cond_name}")
            else:
                print(f"警告: 原始图像 {img_name} 不是 original_ 开头，跳过映射")
    
    def __len__(self):
        if self.use_tensor_cache:
            return len(self.tensor_cache)
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            dict: 包含以下键值对：
                - 'image': 原始输入图像 (tensor, CxHxW, 范围[0,1])
                - 'cond_image': 2x下采样的条件图像 (tensor, CxHxW, 范围[0,1])
                - 'image_path': 图像文件路径
                - '__key__': 图像文件名（不含扩展名）
        """
        # 如果使用tensor缓存，直接从缓存获取
        if self.use_tensor_cache:
            cache_item = self.tensor_cache[idx]
            result = {
                'image': cache_item['image'],
                'image_path': cache_item.get('image_path', f'cached_{idx}'),
                '__key__': cache_item.get('__key__', f'cached_{idx}')
            }
            # 如果跳过cond_image准备，返回None
            if self.skip_cond_image:
                result['cond_image'] = None
            else:
                result['cond_image'] = cache_item['cond_image']
            return result
        
        # 否则从图像文件加载
        image_path = self.image_files[idx]
        
        # 加载原始图像
        image = Image.open(image_path).convert('RGB')
        
        # 调整图像尺寸到 downsample_factor * 16 的倍数
        image = self._adjust_image_size(image)
        
        # 转换为tensor
        image_tensor = transforms.ToTensor()(image)
        
        # 获取图像文件名作为key
        image_key = Path(image_path).stem
        
        result = {
            'image': image_tensor,  # 原始输入图像
            'image_path': image_path,
            '__key__': image_key
        }
        
        # 如果跳过cond_image准备，返回None
        if self.skip_cond_image:
            result['cond_image'] = None
        else:
            # 获取条件图像
            if self.use_precomputed_cond and image_path in self.cond_image_mapping:
                # 使用预计算的条件图像
                cond_image_path = self.cond_image_mapping[image_path]
                cond_image = Image.open(cond_image_path).convert('RGB')
                cond_tensor = transforms.ToTensor()(cond_image)
            else:
                # 重新生成条件图像
                cond_tensor = self._create_cond_image(image)
            result['cond_image'] = cond_tensor  # 2x下采样的条件图像
        
        return result
    
    def _create_cond_image(self, pil_image):
        """
        创建可调节下采样倍率的条件图像
        
        Args:
            pil_image (PIL.Image): 原始PIL图像
            
        Returns:
            torch.Tensor: 下采样的条件图像tensor (CxHxW)
        """
        # 根据下采样倍率进行下采样
        cond_width = pil_image.width // self.downsample_factor
        cond_height = pil_image.height // self.downsample_factor
        cond_pil = transforms.Resize((cond_height, cond_width))(pil_image)
        # 转换回tensor
        cond_tensor = transforms.ToTensor()(cond_pil)
        
        return cond_tensor
    
    def _adjust_image_size(self, pil_image):
        """
        调整图像尺寸到 downsample_factor * 16 的倍数（与local_image_loader保持一致）
        
        Args:
            pil_image (PIL.Image): 原始PIL图像
            
        Returns:
            PIL.Image: 调整后的图像
        """
        # 计算目标尺寸的倍数
        multiple_of = 16 * self.downsample_factor
        
        # 获取当前尺寸
        current_width, current_height = pil_image.size
        
        # 计算调整后的尺寸（向下取整到最近的倍数）
        target_width = (current_width // multiple_of) * multiple_of
        target_height = (current_height // multiple_of) * multiple_of
        
        # 如果尺寸发生变化，进行调整
        if target_width != current_width or target_height != current_height:
            print(f"调整图像尺寸: {current_width}x{current_height} -> {target_width}x{target_height} (multiple_of: {multiple_of})")
            return transforms.Resize((target_height, target_width), antialias=True)(pil_image)
        
        return pil_image
    
    def _tensor_to_pil(self, tensor):
        """将tensor转换为PIL图像"""
        if tensor.dim() == 3:
            # 假设格式为CxHxW
            if tensor.shape[0] == 3:
                tensor = tensor.permute(1, 2, 0)
        # 确保值在[0,1]范围内
        tensor = torch.clamp(tensor, 0, 1)
        # 转换为0-255范围
        tensor = (tensor * 255).byte()
        return Image.fromarray(tensor.cpu().numpy())


def create_local_val_dataloader(image_dir, batch_size=1, num_workers=0, max_samples=None, seed=42, use_precomputed_cond=True, target_size=1024, downsample_factor=2):
    """
    创建本地验证数据加载器
    
    Args:
        image_dir (str): 图像目录路径
        batch_size (int): 批次大小，默认为1
        num_workers (int): 工作进程数，默认为0（避免多进程随机性）
        max_samples (int, optional): 最大样本数量
        seed (int): 随机种子
        use_precomputed_cond (bool): 是否使用预计算的条件图像
        target_size (int): 目标图像尺寸，与S3数据加载器保持一致
        downsample_factor (int): 下采样倍率，默认为2
        
    Returns:
        DataLoader: 本地图像数据加载器
    """
    dataset = LocalImageDataset(image_dir, max_samples=max_samples, seed=seed, use_precomputed_cond=use_precomputed_cond, target_size=target_size, downsample_factor=downsample_factor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 不打乱顺序，保持一致性
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return dataloader


def create_local_train_dataloader(image_dir, batch_size=1, num_workers=0, max_samples=None, seed=42, downsample_factor=2):
    """
    创建本地训练数据加载器（与验证加载器相同，但可以有不同的配置）
    
    Args:
        image_dir (str): 图像目录路径
        batch_size (int): 批次大小，默认为1
        num_workers (int): 工作进程数，默认为0（避免多进程随机性）
        max_samples (int, optional): 最大样本数量
        seed (int): 随机种子
        downsample_factor (int): 下采样倍率，默认为2
        
    Returns:
        DataLoader: 本地图像数据加载器
    """
    return create_local_val_dataloader(image_dir, batch_size, num_workers, max_samples, seed, downsample_factor=downsample_factor)


def create_val_dataloader_from_local(config, image_dir=None, max_samples=None, seed=42, downsample_factor=2):
    """
    Create validation dataloader from local image directory.
    
    Args:
        config: 配置对象，包含dataloader.val的设置
        image_dir (str): 图像目录路径，如果为None则使用config中的设置
        max_samples (int, optional): 最大样本数量
        seed (int): 随机种子
        downsample_factor (int): 下采样倍率，默认为2
        
    Returns:
        DataLoader: 本地图像数据加载器
    """
    if image_dir is None:
        # 尝试从config中获取image_dir
        if hasattr(config, 'dataloader') and hasattr(config.dataloader, 'val'):
            image_dir = getattr(config.dataloader.val, 'image_dir', None)
    
    if image_dir is None:
        raise ValueError("必须提供image_dir参数或config中包含image_dir设置")
    
    # 从config中获取其他参数
    batch_size = getattr(config.dataloader.val, 'batch_size', 1)
    num_workers = getattr(config.dataloader.val, 'num_workers', 0)
    
    return create_local_val_dataloader(
        image_dir=image_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
        seed=seed,
        downsample_factor=downsample_factor
    )


def create_train_dataloader_from_local(config, image_dir=None, max_samples=None, seed=42, downsample_factor=2):
    """
    Create training dataloader from local image directory.
    
    Args:
        config: 配置对象，包含dataloader.train的设置
        image_dir (str): 图像目录路径，如果为None则使用config中的设置
        max_samples (int, optional): 最大样本数量
        seed (int): 随机种子
        downsample_factor (int): 下采样倍率，默认为2
        
    Returns:
        DataLoader: 本地图像数据加载器
    """
    if image_dir is None:
        # 尝试从config中获取image_dir
        if hasattr(config, 'dataloader') and hasattr(config.dataloader.train, 'image_dir'):
            image_dir = getattr(config.dataloader.train, 'image_dir', None)
    
    if image_dir is None:
        raise ValueError("必须提供image_dir参数或config中包含image_dir设置")
    
    # 从config中获取其他参数
    batch_size = getattr(config.dataloader.train, 'batch_size', 1)
    num_workers = getattr(config.dataloader.train, 'num_workers', 0)
    
    return create_local_train_dataloader(
        image_dir=image_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
        seed=seed,
        downsample_factor=downsample_factor
    )


# 测试函数
def test_local_dataloader():
    """测试本地数据加载器"""
    import tempfile
    import shutil
    
    # 创建临时测试目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建一些测试图像，测试不同的尺寸调整场景
        test_images = [
            ('test1.jpg', (256, 256)),    # 256x256 (16*16) - 不需要调整
            ('test2.png', (300, 300)),    # 300x300 - 需要调整到 288x288 (18*16)
            ('test3.bmp', (1024, 1024)), # 1024x1024 (64*16) - 不需要调整
            ('test4.jpg', (1000, 1000))  # 1000x1000 - 需要调整到 992x992 (62*16)
        ]
        for img_name, size in test_images:
            img_path = os.path.join(temp_dir, img_name)
            # 创建一个简单的测试图像
            test_img = Image.new('RGB', size, color='red')
            test_img.save(img_path)
        
        print(f"创建测试图像在: {temp_dir}")
        
        # 测试数据加载器
        dataloader = create_local_val_dataloader(
            image_dir=temp_dir,
            batch_size=1,
            num_workers=0,
            max_samples=2,
            seed=42
        )
        
        print("测试数据加载器...")
        for i, batch in enumerate(dataloader):
            print(f"批次 {i}:")
            print(f"  图像形状: {batch['image'].shape}")
            print(f"  条件图像形状: {batch['cond_image'].shape}")
            print(f"  图像路径: {batch['image_path']}")
            print(f"  图像key: {batch['__key__']}")
            
            if i >= 2:  # 只测试前3个批次
                break
        
        print("本地数据加载器测试完成！")


if __name__ == "__main__":
    test_local_dataloader() 载器测试完成！")


if __name__ == "__main__":
    test_local_dataloader() 