"""
传统长边桶数据加载器 - 使用旧的桶系统，保持向后兼容
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from PIL import Image, ImageFilter
import torchvision.transforms as T
from typing import List, Tuple, Dict, Optional
import json
import hashlib


_COND_PREBLUR_FILTER = ImageFilter.GaussianBlur(radius=1.0)


def resize_long_edge_pil(pil_img, base_size, multiple_of=None, preserve_aspect_ratio=True, resample=Image.Resampling.LANCZOS):
    """
    将PIL图像的长边resize到指定尺寸，保持宽高比
    
    Args:
        pil_img: PIL图像
        base_size: 长边目标尺寸
        multiple_of: 尺寸必须是此数的倍数
        preserve_aspect_ratio: 是否保持宽高比（True）还是允许轻微偏差（False）
    
    Returns:
        resized PIL图像
    """
    w, h = pil_img.size
    
    # 确定长边
    if w >= h:
        # 宽度是长边
        new_w = base_size
        new_h = int(h * base_size / w)
    else:
        # 高度是长边
        new_h = base_size
        new_w = int(w * base_size / h)
    
    # 应用multiple_of约束
    if multiple_of is not None:
        if preserve_aspect_ratio:
            # 保持宽高比，向下取整到最近的倍数
            new_w = (new_w // multiple_of) * multiple_of
            new_h = (new_h // multiple_of) * multiple_of
        else:
            # 允许轻微偏差，向上取整到最近的倍数
            new_w = ((new_w + multiple_of - 1) // multiple_of) * multiple_of
            new_h = ((new_h + multiple_of - 1) // multiple_of) * multiple_of
    
    return pil_img.resize((new_w, new_h), resample=resample)



def resize_to_fill_and_crop(pil_img, target_dims):
    """
    Resizes an image to fill the target dimensions and then center-crops it.
    This is the standard approach to prevent distortion and black bars.

    Args:
        pil_img: The source PIL image.
        target_dims: A tuple (target_width, target_height).

    Returns:
        A PIL image cropped to the exact target dimensions.
    """
    w, h = pil_img.size
    target_w, target_h = target_dims

    # Get aspect ratios
    img_ratio = w / h
    target_ratio = target_w / target_h

    # Determine resize dimensions
    if img_ratio > target_ratio:
        # Image is wider than target: fit to target height
        new_h = target_h
        new_w = int(new_h * img_ratio)
    else:
        # Image is taller than or same as target: fit to target width
        new_w = target_w
        new_h = int(new_w / img_ratio)

    # Resize the image (it will be larger than target in one dimension)
    resized_img = pil_img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    # Calculate coordinates for center crop
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    # Crop and return
    return resized_img.crop((left, top, right, bottom))

def create_long_aspect_ratio_buckets(base_size=1024, multiple_of=32, max_ratio_error=0.1, lq_cond_size=None):
    """
    创建长边resize的宽高比桶（传统方法）
    
    Args:
        base_size: 基础尺寸 (长边尺寸)
        multiple_of: 倍数
        max_ratio_error: 最大比例误差
        lq_cond_size: 条件图像长边尺寸（如果设置则忽略）
    
    Returns:
        buckets: 桶列表
    """
    buckets = []
    
    # 生成各种宽高比的桶，确保长宽比不超过2
    for w in range(multiple_of, base_size + 1, multiple_of):
        for h in range(multiple_of, base_size + 1, multiple_of):
            # 确保长边等于base_size
            if max(w, h) == base_size:
                # 计算长宽比，确保不超过2
                aspect_ratio = max(w, h) / min(w, h)
                if aspect_ratio <= 2.0:
                    buckets.append((w, h))
    
    print(f"✅ 创建了 {len(buckets)} 个传统长边桶（长宽比≤2）")
    return buckets


def get_all_image_files(folder_path):
    """获取文件夹中所有图像文件"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIFF'}
    image_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return sorted(image_files)


class LegacyLongBucketDataset(Dataset):
    """传统长边桶数据集"""
    
    def __init__(self, image_folder_path, buckets, base_size, multiple_of, max_samples=None):
        """
        Args:
            image_folder_path: 图片文件夹路径
            buckets: 桶列表
            base_size: 基础尺寸 (长边尺寸)
            multiple_of: 倍数
            max_samples: 最大图像数量
        """
        self.image_folder_path = image_folder_path
        self.buckets = buckets
        self.base_size = base_size
        self.multiple_of = multiple_of
        self.max_samples = max_samples
        
        # 计算桶分配
        self._compute_bucket_assignments()
    
    def _compute_bucket_assignments(self):
        """计算桶分配 - 传统方法"""
        print(f"正在扫描文件夹: {self.image_folder_path}")
        self.image_files = get_all_image_files(self.image_folder_path)
        
        # 保存原始图像文件列表用于桶分配计算
        self.original_image_files = self.image_files.copy()
        
        if self.max_samples is not None and self.max_samples > 0:
            if len(self.image_files) >= self.max_samples:
                # 如果数据集大小 >= max_samples，只取前max_samples张
                self.image_files = self.image_files[:self.max_samples]
                print(f"🔒 Overfitting模式：限制使用前 {self.max_samples} 张图像")
            else:
                # 如果数据集大小 < max_samples，记录重复信息，但桶分配仍使用原始数据集
                self.original_size = len(self.image_files)
                self.target_size = self.max_samples
                print(f"🔄 数据集重复模式：原始 {self.original_size} 张图像，将重复到 {self.max_samples} 张图像")
        
        self.image_data = []
        bucket_stats = {}
        
        print(f"预计算图片长边resize后的宽高比并分配桶... 找到 {len(self.image_files)} 张图片")
        
        for filepath in self.image_files:
            try:
                with Image.open(filepath) as img:
                    original_width, original_height = img.size
                
                # 计算长边resize后的尺寸
                if original_width >= original_height:
                    resized_w = self.base_size
                    resized_h = int(original_height * self.base_size / original_width)
                    is_landscape = True  # 横图
                else:
                    resized_h = self.base_size
                    resized_w = int(original_width * self.base_size / original_height)
                    is_landscape = False  # 竖图
                
                aspect_ratio = resized_w / resized_h
                area = resized_w * resized_h
                
                # 找到最匹配的桶，确保方向匹配
                best_bucket_id = min(
                    range(len(self.buckets)), 
                    key=lambda i: (
                        # 首先检查方向是否匹配
                        abs((self.buckets[i][0] >= self.buckets[i][1]) - is_landscape),
                        # 然后比较宽高比
                        abs(aspect_ratio - (self.buckets[i][0] / self.buckets[i][1])),
                        # 最后比较面积
                        abs(area - (self.buckets[i][0] * self.buckets[i][1]))
                    )
                )
                
                self.image_data.append({
                    'filepath': filepath,
                    'bucket_id': best_bucket_id,
                    'original_size': (original_width, original_height),
                    'resized_size': (resized_w, resized_h)
                })
                
                bucket_stats[best_bucket_id] = bucket_stats.get(best_bucket_id, 0) + 1
                
            except Exception as e:
                print(f"警告: 无法处理图片 {filepath}: {e}")
                continue
        
        self._print_bucket_stats(bucket_stats)
    
    def _print_bucket_stats(self, bucket_stats):
        """打印桶统计信息"""
        print(f"\n📊 桶分配统计:")
        for bucket_id, count in sorted(bucket_stats.items()):
            bucket = self.buckets[bucket_id]
            print(f"  桶 {bucket_id}: {bucket[0]}x{bucket[1]} - {count} 张图片")
        print(f"总计: {sum(bucket_stats.values())} 张图片分配到 {len(bucket_stats)} 个桶")
    
    def __len__(self):
        if hasattr(self, 'target_size'):
            return self.target_size
        return len(self.image_data)
    
    def __getitem__(self, index):
        if hasattr(self, 'target_size'):
            # 重复模式：使用模运算来循环访问原始数据
            original_index = index % len(self.image_data)
            return self.image_data[original_index]
        else:
            return self.image_data[index]


class LegacyLongBucketBatchSampler:
    """传统长边桶批次采样器 - 按桶分组确保同一batch中的图片使用相同桶"""
    
    def __init__(self, sampler, batch_size, drop_last=False, shuffle=True):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # 获取数据集和图片数据
        if hasattr(self.sampler, 'dataset'):
            dataset = self.sampler.dataset
        else:
            dataset = self.sampler.data_source
        
        self.image_data = dataset.image_data
        
        # 按桶分组索引
        self._create_bucket_batches()
    
    def _create_bucket_batches(self):
        """创建按桶分组的批次"""
        indices = list(self.sampler)
        
        # 按桶ID分组
        buckets_to_indices = {}
        for idx in indices:
            # 如果数据集有重复模式，使用模运算来获取有效的索引
            dataset = self.sampler.dataset if hasattr(self.sampler, 'dataset') else self.sampler.data_source
            if hasattr(dataset, 'target_size'):
                effective_idx = idx % len(self.image_data)
            else:
                effective_idx = idx
            
            bucket_id = self.image_data[effective_idx]['bucket_id']
            if bucket_id not in buckets_to_indices:
                buckets_to_indices[bucket_id] = []
            buckets_to_indices[bucket_id].append(idx)  # 保持原始索引用于后续访问
        
        # 为每个桶创建批次
        self.batches = []
        for bucket_id, bucket_indices in buckets_to_indices.items():
            if self.shuffle:
                random.shuffle(bucket_indices)
            
            # 将桶内的索引分成批次
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                self.batches.append(batch)
        
        # 如果需要，打乱批次顺序
        if self.shuffle:
            random.shuffle(self.batches)
    
    def __iter__(self):
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)


def create_legacy_collate_fn(buckets, base_size, downsample_factor=2, upsample_factor=1, 
                           skip_cond_image=False, use_degraded_image=False, lq_cond_size=None, 
                           random_resize=False, multiple_of=32, cond_downsample_resample=Image.Resampling.BICUBIC, cond_upsample_resample=None, cond_preblur=False, cond_random_upsample_prob=0.0, load_text=False, caption_json_key="text"):
    """
    创建传统桶系统的collate_fn
    
    Args:
        buckets: 桶列表
        base_size: 基础尺寸
        downsample_factor: 下采样倍率
        upsample_factor: 上采样倍率
        skip_cond_image: 是否跳过cond_image准备
        use_degraded_image: 是否使用降质图像
        lq_cond_size: lq_cond长边尺寸
        random_resize: 是否随机resize
        multiple_of: 倍数约束
    """
    def collate_fn(batch):
        def _extract_caption_text(raw):
            if isinstance(raw, dict):
                value = raw.get(caption_json_key)
                if not value and caption_json_key != "text":
                    value = raw.get("text")
                if isinstance(value, (list, tuple)):
                    value = value[0] if value else ""
                return str(value) if value is not None else ""
            if isinstance(raw, (list, tuple)):
                return str(raw[0]) if raw else ""
            if raw is None:
                return ""
            return str(raw)

        # 现在batch中的所有图片都来自同一个桶，所以可以安全地使用第一个图片的bucket_id
        bucket_id = batch[0]['bucket_id']
        target_dims = buckets[bucket_id]

        processed_images = []
        processed_cond_images = []
        image_keys = []
        text_data = []
        
        to_tensor_transform = T.ToTensor()
        cond_upsample = cond_upsample_resample or cond_downsample_resample
        upsample_cond_to_final = cond_random_upsample_prob > 0.0 and random.random() < cond_random_upsample_prob

        # 如果启用随机resize
        if random_resize and lq_cond_size is not None:
            # 为整个batch选择一个统一的随机尺寸
            min_long_edge = lq_cond_size
            max_long_edge = base_size
            # 在符合multiple_of约束的尺寸中随机选择一个
            min_size = ((min_long_edge - 1) // multiple_of + 1) * multiple_of
            max_size = (max_long_edge // multiple_of) * multiple_of
            if min_size <= max_size:
                available_sizes = list(range(min_size, max_size + 1, multiple_of))
                batch_target_long_edge = random.choice(available_sizes)
            else:
                batch_target_long_edge = min_size
        else:
            batch_target_long_edge = None

        for item in batch:
            try:
                pil_img = Image.open(item['filepath']).convert('RGB')
                
                # 长边resize + 保持宽高比缩放
                final_img = resize_to_fill_and_crop(pil_img, target_dims)
                
                # 如果启用随机resize，对最终图像进行随机resize
                if random_resize and lq_cond_size is not None and batch_target_long_edge is not None:
                    final_img = resize_long_edge_pil(final_img, batch_target_long_edge, multiple_of)

                # 生成cond（低分）
                if lq_cond_size is not None:
                    # 使用lq_cond_size参数，将条件图像的长边resize到指定尺寸
                    cond_source_img = final_img.filter(_COND_PREBLUR_FILTER) if cond_preblur else final_img
                    cond_pil = resize_long_edge_pil(cond_source_img, lq_cond_size, multiple_of, resample=cond_downsample_resample)
                else:
                    # 使用传统的downsample_factor方式
                    cond_width = max(1, final_img.width // downsample_factor)
                    cond_height = max(1, final_img.height // downsample_factor)
                    cond_source_img = final_img.filter(_COND_PREBLUR_FILTER) if cond_preblur else final_img
                    cond_pil = cond_source_img.resize((cond_width, cond_height), resample=cond_downsample_resample)
                    if upsample_cond_to_final:
                        cond_pil = cond_pil.resize((final_img.width, final_img.height), resample=cond_upsample)

                if use_degraded_image:
                    # image 使用 下采样->上采样 回到目标分辨率
                    degraded_pil = cond_pil.resize((final_img.width, final_img.height), resample=cond_upsample)
                    tensor_img = to_tensor_transform(degraded_pil)
                else:
                    # 常规：image 为裁剪后的高分图
                    tensor_img = to_tensor_transform(final_img)
                
                image_key = os.path.splitext(os.path.basename(item['filepath']))[0]
                
                # 加载文本数据（如果启用）
                if load_text:
                    try:
                        json_path = os.path.splitext(item['filepath'])[0] + '.json'
                        if os.path.exists(json_path):
                            with open(json_path, 'r', encoding='utf-8') as f:
                                text_data_item = json.load(f)
                                text_data.append(text_data_item.get('text', ''))
                        else:
                            text_data.append('')
                    except Exception as e:
                        print(f"警告: 加载文本文件时出错 {json_path}: {e}")
                        text_data.append('')
                else:
                    text_data.append('')
                
                processed_images.append(tensor_img)
                image_keys.append(image_key)
                
                # 如果跳过cond_image准备，添加None
                if skip_cond_image:
                    processed_cond_images.append(None)
                else:
                    if use_degraded_image:
                        # degraded 模式下，cond 仅为低分，不做上采样
                        cond_tensor = torch.clamp(to_tensor_transform(cond_pil), 0.0, 1.0)
                    else:
                        if lq_cond_size is not None:
                            # 使用lq_cond_size时，直接使用resize后的cond_pil
                            cond_tensor = torch.clamp(to_tensor_transform(cond_pil), 0.0, 1.0)
                        else:
                            # 使用传统的downsample_factor + upsample_factor方式
                            cond_tensor = create_cond_image(
                                cond_source_img,
                                downsample_factor=downsample_factor,
                                upsample_factor=upsample_factor,
                                downsample_resample=cond_downsample_resample,
                                upsample_resample=cond_upsample
                            )
                    processed_cond_images.append(cond_tensor)
                
            except Exception as e:
                print(f"警告: 处理图片时出错 {item['filepath']}: {e}")
                default_img = torch.zeros(3, target_dims[1], target_dims[0])
                default_key = f"error_{len(processed_images)}"
                
                processed_images.append(default_img)
                image_keys.append(default_key)
                text_data.append('')  # 异常情况下添加空文本
                
                # 如果跳过cond_image准备，添加None
                if skip_cond_image:
                    processed_cond_images.append(None)
                else:
                    if upsample_cond_to_final:
                        default_cond_img = torch.zeros(3, target_dims[1], target_dims[0])
                    elif lq_cond_size is not None:
                        # 使用lq_cond_size时，计算默认条件图像尺寸
                        default_cond_img = torch.zeros(3, lq_cond_size, lq_cond_size)
                    else:
                        # 使用传统downsample_factor方式
                        default_cond_img = torch.zeros(3, target_dims[1] // downsample_factor, target_dims[0] // downsample_factor)
                    processed_cond_images.append(default_cond_img)
        
        result = {
            'image': torch.stack(processed_images),
            '__key__': image_keys
        }
        
        # 如果启用文本加载，添加文本数据
        if load_text:
            result['strText'] = text_data
        
        # 如果跳过cond_image准备，返回None
        if skip_cond_image:
            result['cond_image'] = None
        else:
            result['cond_image'] = torch.stack(processed_cond_images)
        
        return result
    
    return collate_fn


def _crop_to_target(pil_img, target_dims):
    """将图像裁剪到目标尺寸"""
    w, h = pil_img.size
    target_w, target_h = target_dims
    
    # 计算裁剪区域
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    
    return pil_img.crop((left, top, right, bottom))


def create_cond_image(pil_img, downsample_factor=2, upsample_factor=1, downsample_resample=Image.Resampling.BICUBIC, upsample_resample=None):
    """创建条件图像（传统方法）"""
    w, h = pil_img.size
    cond_w = max(1, w // downsample_factor)
    cond_h = max(1, h // downsample_factor)
    
    # 下采样
    cond_pil = pil_img.resize((cond_w, cond_h), resample=downsample_resample)
    
    # 上采样（如果upsample_factor > 1）
    final_upsample_resample = upsample_resample or downsample_resample
    if upsample_factor > 1:
        cond_w = max(1, cond_w * upsample_factor)
        cond_h = max(1, cond_h * upsample_factor)
        cond_pil = cond_pil.resize((cond_w, cond_h), resample=final_upsample_resample)
    
    return T.ToTensor()(cond_pil)


def create_legacy_long_dataloader(image_folder_path, base_size=1024, multiple_of=32, 
                                 batch_size=4, num_workers=4, drop_last=False, shuffle=True, 
                                 max_samples=None, downsample_factor=2, upsample_factor=1,
                                 skip_cond_image=False, use_degraded_image=False, 
                                 lq_cond_size=None, random_resize=False,
                                 cond_downsample_resample=Image.Resampling.BICUBIC, cond_upsample_resample=None,
                                 cond_preblur=False, cond_random_upsample_prob=0.0, load_text=False):
    """
    创建传统长边桶数据加载器
    
    Args:
        image_folder_path: 图像文件夹路径
        base_size: 基础尺寸（长边尺寸）
        multiple_of: 倍数约束
        batch_size: 批次大小
        num_workers: 工作进程数
        drop_last: 是否丢弃最后一个不完整的批次
        shuffle: 是否随机打乱
        max_samples: 最大图像数量
        downsample_factor: 下采样倍率
        upsample_factor: 上采样倍率
        skip_cond_image: 是否跳过cond_image准备
        use_degraded_image: 是否使用降质图像
        lq_cond_size: lq_cond长边尺寸
        random_resize: 是否随机resize
    
    Returns:
        DataLoader: 配置好的数据加载器
    """
    print("🚀 开始创建传统长边桶DataLoader...")
    if lq_cond_size is not None:
        resize_info = f", 随机resize: {random_resize}" if random_resize else ""
        print(f"长边尺寸: {base_size}, 条件图像长边: {lq_cond_size}, multiple_of: {multiple_of}, degraded: {use_degraded_image}{resize_info}")
    else:
        print(f"长边尺寸: {base_size}, 下采样倍率: {downsample_factor}, 上采样倍率: {upsample_factor}, multiple_of: {multiple_of}, degraded: {use_degraded_image}")
    print(f"随机化模式: {'训练模式（随机）' if shuffle else '验证模式（固定顺序）'}")
    if max_samples is not None:
        print(f"🔒 Overfitting模式：限制使用前 {max_samples} 张图像")
    
    # 1. 创建传统桶系统
    print("(1/4) 创建传统桶系统...")
    buckets = create_long_aspect_ratio_buckets(
        base_size=base_size,
        multiple_of=multiple_of,
        lq_cond_size=lq_cond_size
    )
    
    # 2. 创建数据集
    print("(2/4) 创建传统数据集...")
    dataset = LegacyLongBucketDataset(
        image_folder_path=image_folder_path,
        buckets=buckets,
        base_size=base_size,
        multiple_of=multiple_of,
        max_samples=max_samples
    )
    
    if len(dataset) == 0:
        raise ValueError(f"在 {image_folder_path} 中没有找到有效的图片文件")
    
    # 3. 创建采样器
    print("(3/4) 创建采样器...")
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    
    batch_sampler = LegacyLongBucketBatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle
    )
    
    # 4. 创建collate_fn
    print("(4/4) 创建collate函数...")
    collate_function = create_legacy_collate_fn(
        buckets=buckets,
        base_size=base_size,
        downsample_factor=downsample_factor,
        upsample_factor=upsample_factor,
        skip_cond_image=skip_cond_image,
        use_degraded_image=use_degraded_image,
        lq_cond_size=lq_cond_size,
        random_resize=random_resize,
        multiple_of=multiple_of,
        cond_downsample_resample=cond_downsample_resample,
        cond_upsample_resample=cond_upsample_resample,
        cond_preblur=cond_preblur,
        cond_random_upsample_prob=cond_random_upsample_prob,
        load_text=load_text
    )
    
    # 5. 创建DataLoader
    print("(5/5) 创建DataLoader...")
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_function,
        num_workers=num_workers
    )
    
    print(f"✅ 传统长边桶DataLoader创建完成！")
    print(f"数据集大小: {len(dataset)}, 批次大小: {batch_size}")
    print(f"桶数: {len(buckets)}")
    
    return dataloader


# 测试函数
def test_legacy_dataloader():
    """测试传统数据加载器"""
    print("🧪 测试传统长边桶数据加载器...")
    
    # 创建测试数据加载器
    try:
        dataloader = create_legacy_long_dataloader(
            image_folder_path="/tmp",  # 使用临时目录进行测试
            base_size=1024,
            multiple_of=32,
            batch_size=2,
            num_workers=0,
            max_samples=10
        )
        print("✅ 传统数据加载器创建成功！")
    except Exception as e:
        print(f"❌ 传统数据加载器创建失败: {e}")


if __name__ == "__main__":
    test_legacy_dataloader()
