"""
增强长边桶数据加载器 - 使用新的桶系统，确保lq_cond和最终图像比例完全一致
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


def create_lq_cond_buckets(lq_cond_size=512, multiple_of=16, max_aspect_ratio=2.0):
    """
    创建lq_cond桶，长边固定为lq_cond_size，短边为multiple_of的倍数
    
    Args:
        lq_cond_size: lq_cond的长边尺寸
        multiple_of: 尺寸必须是此数的倍数
        max_aspect_ratio: 最大宽高比，超过此比例的桶将被过滤
    
    Returns:
        lq_buckets: lq_cond桶列表
        lq_ratio_groups: 按比例分组的桶字典
    """
    lq_buckets = []
    lq_ratio_groups = {}
    filtered_count = 0
    
    # 生成横向桶 (长边为lq_cond_size)
    for short_edge in range(multiple_of, lq_cond_size + 1, multiple_of):
        w, h = lq_cond_size, short_edge
        aspect_ratio = w / h
        
        if aspect_ratio <= max_aspect_ratio:
            lq_buckets.append((w, h))
            ratio_key = round(aspect_ratio, 2)
            if ratio_key not in lq_ratio_groups:
                lq_ratio_groups[ratio_key] = []
            lq_ratio_groups[ratio_key].append((w, h))
        else:
            filtered_count += 1
        
        # 生成纵向桶（如果短边不等于长边）
        if short_edge != lq_cond_size:
            w, h = short_edge, lq_cond_size
            aspect_ratio = h / w
            
            if aspect_ratio <= max_aspect_ratio:
                lq_buckets.append((w, h))
                ratio_key = round(aspect_ratio, 2)
                if ratio_key not in lq_ratio_groups:
                    lq_ratio_groups[ratio_key] = []
                lq_ratio_groups[ratio_key].append((w, h))
            else:
                filtered_count += 1
    
    print(f"✅ 创建了 {len(lq_buckets)} 个lq_cond桶 (过滤了 {filtered_count} 个极端比例桶)")
    print(f"✅ lq_cond比例分组: {len(lq_ratio_groups)} 个不同的宽高比")
    
    return lq_buckets, lq_ratio_groups


def create_final_buckets_from_lq(lq_ratio_groups, base_size=1024, multiple_of=16, lq_cond_size=512):
    """
    基于lq_cond桶的比例创建最终图像桶
    
    Args:
        lq_ratio_groups: lq_cond桶的比例分组
        base_size: 最终图像的最大长边尺寸
        multiple_of: 尺寸必须是此数的倍数
        lq_cond_size: lq_cond的长边尺寸
    
    Returns:
        final_buckets: 最终图像桶列表
        bucket_pairs: lq_cond桶和最终图像桶的配对关系
    """
    final_buckets = []
    bucket_pairs = {}
    
    for ratio, lq_bucket_list in lq_ratio_groups.items():
        for lq_w, lq_h in lq_bucket_list:
            # 确定lq_cond桶的方向和比例
            if lq_w >= lq_h:
                lq_ratio = lq_w / lq_h
                is_landscape = True
            else:
                lq_ratio = lq_h / lq_w
                is_landscape = False
            
            final_bucket_list = []
            
            # 生成最终图像桶，长边必须大于lq_cond_size
            for long_edge in range(lq_cond_size + multiple_of, base_size + 1, multiple_of):
                if is_landscape:
                    # 横向：计算精确高度，然后取整到multiple_of
                    exact_h = long_edge / lq_ratio
                    final_h = round(exact_h / multiple_of) * multiple_of
                    final_w = long_edge
                else:
                    # 纵向：计算精确宽度，然后取整到multiple_of
                    exact_w = long_edge / lq_ratio
                    final_w = round(exact_w / multiple_of) * multiple_of
                    final_h = long_edge
                
                # 检查尺寸是否有效
                if max(final_w, final_h) <= base_size and min(final_w, final_h) >= multiple_of:
                    # 计算实际比例
                    if final_w >= final_h:
                        actual_ratio = final_w / final_h
                    else:
                        actual_ratio = final_h / final_w
                    
                    # 检查比例一致性（严格容差）
                    if abs(actual_ratio - lq_ratio) < 0.002:
                        final_bucket_list.append((final_w, final_h))
                        final_buckets.append((final_w, final_h))
            
            bucket_pairs[(lq_w, lq_h)] = final_bucket_list
    
    print(f"✅ 创建了 {len(final_buckets)} 个最终图像桶")
    print(f"✅ 桶配对数量: {len(bucket_pairs)}")
    
    return final_buckets, bucket_pairs


def create_enhanced_buckets(base_size=1024, multiple_of=16, lq_cond_size=512, max_aspect_ratio=2.0):
    """
    创建增强桶系统
    
    Args:
        base_size: 最终图像的最大长边尺寸
        multiple_of: 尺寸必须是此数的倍数
        lq_cond_size: lq_cond的长边尺寸
        max_aspect_ratio: 最大宽高比
    
    Returns:
        final_buckets: 最终图像桶列表
        lq_buckets: lq_cond桶列表
        bucket_pairs: 桶配对关系
    """
    # 1. 创建lq_cond桶
    lq_buckets, lq_ratio_groups = create_lq_cond_buckets(
        lq_cond_size=lq_cond_size,
        multiple_of=multiple_of,
        max_aspect_ratio=max_aspect_ratio
    )
    
    # 2. 基于lq_cond桶创建最终图像桶
    final_buckets, bucket_pairs = create_final_buckets_from_lq(
        lq_ratio_groups=lq_ratio_groups,
        base_size=base_size,
        multiple_of=multiple_of,
        lq_cond_size=lq_cond_size
    )
    
    return final_buckets, lq_buckets, bucket_pairs

def _resize_and_center_crop(pil_img, target_dims):
    """
    更稳健的缩放和居中裁剪，确保图像能覆盖目标尺寸。
    
    Args:
        pil_img: PIL图像
        target_dims: (宽度, 高度) 目标尺寸
    
    Returns:
        裁剪后的PIL图像
    """
    original_w, original_h = pil_img.size
    target_w, target_h = target_dims

    # 计算原始图像和目标尺寸的宽高比
    original_ratio = original_w / original_h
    target_ratio = target_w / target_h

    # 确定缩放尺寸
    if original_ratio > target_ratio:
        # 原始图像比目标更“宽”，以目标高度为基准进行缩放
        new_h = target_h
        new_w = int(new_h * original_ratio)
    else:
        # 原始图像比目标更“高”或比例相同，以目标宽度为基准进行缩放
        new_w = target_w
        new_h = int(new_w / original_ratio)
        
    # 缩放图像
    resized_img = pil_img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    # 居中裁剪
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    
    return resized_img.crop((left, top, right, bottom))

def get_all_image_files(folder_path):
    """获取文件夹中所有图像文件"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIFF'}
    image_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return sorted(image_files)


class EnhancedLongBucketDataset(Dataset):
    """增强长边桶数据集 - 使用新的桶系统"""
    
    def __init__(self, image_folder_path, lq_buckets, bucket_pairs, base_size, multiple_of, max_samples=None):
        """
        Args:
            image_folder_path: 图片文件夹路径
            lq_buckets: lq_cond桶列表
            bucket_pairs: lq_cond桶和最终图像桶的配对关系
            base_size: 基础尺寸 (长边尺寸)
            multiple_of: 倍数
            max_samples: 最大图像数量
        """
        self.image_folder_path = image_folder_path
        self.lq_buckets = lq_buckets
        self.bucket_pairs = bucket_pairs
        self.base_size = base_size
        self.multiple_of = multiple_of
        self.max_samples = max_samples
        
        # 计算桶分配
        self._compute_bucket_assignments()
    
    def _compute_bucket_assignments(self):
        """计算桶分配 - 根据lq_cond桶进行分配"""
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
                else:
                    resized_h = self.base_size
                    resized_w = int(original_width * self.base_size / original_height)
                
                # 计算宽高比
                if resized_w >= resized_h:
                    aspect_ratio = resized_w / resized_h
                else:
                    aspect_ratio = resized_h / resized_w
                
                # 找到最匹配的lq_cond桶
                best_lq_bucket = None
                min_ratio_diff = float('inf')
                
                for lq_bucket in self.lq_buckets:
                    lq_w, lq_h = lq_bucket
                    if lq_w >= lq_h:
                        lq_ratio = lq_w / lq_h
                    else:
                        lq_ratio = lq_h / lq_w
                    
                    ratio_diff = abs(aspect_ratio - lq_ratio)
                    if ratio_diff < min_ratio_diff:
                        min_ratio_diff = ratio_diff
                        best_lq_bucket = lq_bucket
                
                if best_lq_bucket is not None:
                    # 找到对应的lq_cond桶索引
                    lq_bucket_id = self.lq_buckets.index(best_lq_bucket)
                    best_bucket_id = lq_bucket_id
                else:
                    # 如果没有找到匹配的lq_cond桶，使用第一个桶
                    best_bucket_id = 0
                
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
            lq_bucket = self.lq_buckets[bucket_id]
            print(f"  lq_cond桶 {bucket_id}: {lq_bucket[0]}x{lq_bucket[1]} - {count} 张图片")
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


class EnhancedLongBucketBatchSampler:
    """增强长边桶批次采样器 - 按桶分组确保同一batch中的图片使用相同桶"""
    
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


def create_enhanced_collate_fn(lq_buckets, bucket_pairs, base_size, lq_cond_size, multiple_of, skip_cond_image=False, use_degraded_image=False, cond_downsample_resample=Image.Resampling.BICUBIC, cond_upsample_resample=None, cond_preblur=False, cond_random_upsample_prob=0.0, load_text=False, caption_json_key="text"):
    """
    创建增强桶系统的collate_fn
    
    Args:
        lq_buckets: lq_cond桶列表
        bucket_pairs: 桶配对关系
        base_size: 基础尺寸
        lq_cond_size: lq_cond长边尺寸
        multiple_of: 倍数约束
        skip_cond_image: 是否跳过cond_image准备
        use_degraded_image: 是否使用降质图像
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
        lq_bucket = lq_buckets[bucket_id]
        
        available_final_buckets = bucket_pairs[lq_bucket]
        if available_final_buckets:
            target_dims = random.choice(available_final_buckets)
        else:
            target_dims = lq_bucket
        
        processed_images = []
        processed_cond_images = []
        image_keys = []
        text_data = []
        
        to_tensor_transform = T.ToTensor()
        cond_upsample = cond_upsample_resample or cond_downsample_resample
        upsample_cond_to_final = cond_random_upsample_prob > 0.0 and random.random() < cond_random_upsample_prob
        
        for item in batch:
            try:
                pil_img = Image.open(item['filepath']).convert('RGB')
                
                # --- START: 修改部分 ---
                # 使用新的、更稳健的函数一步完成缩放和裁剪
                final_img = _resize_and_center_crop(pil_img, target_dims)
                # --- END: 修改部分 ---

                # 生成cond（低分），使用配置的插值模式
                cond_source_img = final_img.filter(_COND_PREBLUR_FILTER) if cond_preblur else final_img
                cond_pil = resize_long_edge_pil(cond_source_img, lq_cond_size, multiple_of, resample=cond_downsample_resample)
                
                if use_degraded_image:
                    degraded_pil = cond_pil.resize((final_img.width, final_img.height), resample=cond_upsample)
                    tensor_img = to_tensor_transform(degraded_pil)
                else:
                    tensor_img = to_tensor_transform(final_img)
                
                image_key = os.path.splitext(os.path.basename(item['filepath']))[0]
                
                # 加载文本数据（如果启用）
                if load_text:
                    try:
                        json_path = os.path.splitext(item['filepath'])[0] + '.json'
                        if os.path.exists(json_path):
                            with open(json_path, 'r', encoding='utf-8') as f:
                                text_data_item = json.load(f)
                                caption = _extract_caption_text(text_data_item)
                                text_data.append(caption)
                        else:
                            text_data.append('')
                    except Exception as e:
                        print(f"警告: 加载文本文件时出错 {json_path}: {e}")
                        text_data.append('')
                else:
                    text_data.append('')
                
                processed_images.append(tensor_img)
                image_keys.append(image_key)
                
                if skip_cond_image:
                    processed_cond_images.append(None)
                else:
                    cond_tensor = torch.clamp(to_tensor_transform(cond_pil), 0.0, 1.0)
                    if upsample_cond_to_final:
                        upsampled_pil = cond_pil.resize((final_img.width, final_img.height), resample=cond_upsample)
                        cond_tensor = torch.clamp(to_tensor_transform(upsampled_pil), 0.0, 1.0)
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
                    else:
                        default_cond_img = torch.zeros(3, lq_cond_size, lq_cond_size)
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


def create_enhanced_long_dataloader(image_folder_path, base_size=1024, multiple_of=16, lq_cond_size=512, 
                                   batch_size=4, num_workers=4, drop_last=False, shuffle=True, 
                                   max_samples=None, skip_cond_image=False, use_degraded_image=False, 
                                   cond_downsample_resample=Image.Resampling.BICUBIC, cond_upsample_resample=None, 
                                   cond_preblur=False, cond_random_upsample_prob=0.0, load_text=False, caption_json_key="text"):
    """
    创建增强长边桶数据加载器
    
    Args:
        image_folder_path: 图像文件夹路径
        base_size: 基础尺寸（长边尺寸）
        multiple_of: 倍数约束
        lq_cond_size: lq_cond长边尺寸
        batch_size: 批次大小
        num_workers: 工作进程数
        drop_last: 是否丢弃最后一个不完整的批次
        shuffle: 是否随机打乱
        max_samples: 最大图像数量
        skip_cond_image: 是否跳过cond_image准备
        use_degraded_image: 是否使用降质图像
        load_text: 是否加载文本数据（从对应的JSON文件）
        caption_json_key: 读取文本时优先使用的JSON键名
    
    Returns:
        DataLoader: 配置好的数据加载器
    """
    print("🚀 开始创建增强长边桶DataLoader...")
    print(f"长边尺寸: {base_size}, lq_cond长边: {lq_cond_size}, multiple_of: {multiple_of}")
    print(f"随机化模式: {'训练模式（随机）' if shuffle else '验证模式（固定顺序）'}")
    if max_samples is not None:
        print(f"🔒 Overfitting模式：限制使用前 {max_samples} 张图像")
    
    # 1. 创建增强桶系统
    print("(1/4) 创建增强桶系统...")
    final_buckets, lq_buckets, bucket_pairs = create_enhanced_buckets(
        base_size=base_size,
        multiple_of=multiple_of,
        lq_cond_size=lq_cond_size
    )
    
    # 2. 创建数据集
    print("(2/4) 创建增强数据集...")
    dataset = EnhancedLongBucketDataset(
        image_folder_path=image_folder_path,
        lq_buckets=lq_buckets,
        bucket_pairs=bucket_pairs,
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
    
    batch_sampler = EnhancedLongBucketBatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle
    )
    
    # 4. 创建collate_fn
    print("(4/4) 创建collate函数...")
    collate_function = create_enhanced_collate_fn(
        lq_buckets=lq_buckets,
        bucket_pairs=bucket_pairs,
        base_size=base_size,
        lq_cond_size=lq_cond_size,
        multiple_of=multiple_of,
        skip_cond_image=skip_cond_image,
        use_degraded_image=use_degraded_image,
        cond_downsample_resample=cond_downsample_resample,
        cond_upsample_resample=cond_upsample_resample,
        cond_preblur=cond_preblur,
        cond_random_upsample_prob=cond_random_upsample_prob,
        load_text=load_text,
        caption_json_key=caption_json_key
    )
    
    # 5. 创建DataLoader
    print("(5/5) 创建DataLoader...")
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_function,
        num_workers=num_workers
    )
    
    print(f"✅ 增强长边桶DataLoader创建完成！")
    print(f"数据集大小: {len(dataset)}, 批次大小: {batch_size}")
    print(f"lq_cond桶数: {len(lq_buckets)}, 最终图像桶数: {len(final_buckets)}")
    
    return dataloader


# 测试函数
def test_enhanced_dataloader():
    """测试增强数据加载器"""
    print("🧪 测试增强长边桶数据加载器...")
    
    # 创建测试数据加载器
    try:
        dataloader = create_enhanced_long_dataloader(
            image_folder_path="/tmp",  # 使用临时目录进行测试
            base_size=1024,
            multiple_of=16,
            lq_cond_size=512,
            batch_size=2,
            num_workers=0,
            max_samples=10
        )
        print("✅ 增强数据加载器创建成功！")
    except Exception as e:
        print(f"❌ 增强数据加载器创建失败: {e}")


if __name__ == "__main__":
    test_enhanced_dataloader()
