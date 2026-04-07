import math
import time
import os
import random
import torch
import pickle
import hashlib
import json
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, Sampler, DataLoader, BatchSampler, RandomSampler, SequentialSampler
import torchvision.transforms as T
import numpy as np
 


# Preblur filter used when cond_preblur is enabled
_COND_PREBLUR_FILTER = ImageFilter.GaussianBlur(radius=1.0)

def get_dataset_hash(image_folder_path, max_samples=None):
    """
    计算数据集的哈希值，用于缓存验证
    """
    image_files = get_all_image_files(image_folder_path)
    if max_samples is not None and max_samples > 0:
        image_files = image_files[:max_samples]
    
    hash_input = []
    for filepath in image_files:
        try:
            stat = os.stat(filepath)
            hash_input.append(f"{filepath}:{stat.st_mtime}:{stat.st_size}")
        except OSError:
            continue
    
    hash_str = hashlib.md5('|'.join(hash_input).encode()).hexdigest()
    return hash_str


def get_cache_path(image_folder_path, base_size, multiple_of, max_samples=None):
    """
    获取缓存文件路径
    """
    cache_dir = os.path.join(os.path.dirname(image_folder_path), '.bucket_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    dataset_hash = get_dataset_hash(image_folder_path, max_samples)
    cache_filename = f"bucket_cache_{base_size}_{multiple_of}_{dataset_hash}.pkl"
    return os.path.join(cache_dir, cache_filename)


def load_bucket_cache(cache_path):
    """
    加载bucket缓存
    """
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            print(f"✅ 从缓存加载bucket分配结果: {cache_path}")
            return cache_data
    except Exception as e:
        print(f"⚠️ 加载缓存失败: {e}")
    return None


def save_bucket_cache(
    cache_path,
    buckets,
    image_data,
    bucket_stats,
):
    """
    保存bucket缓存
    """
    try:
        cache_data = {
            'buckets': buckets,
            'image_data': image_data,
            'bucket_stats': bucket_stats
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"✅ 保存bucket分配结果到缓存: {cache_path}")
    except Exception as e:
        print(f"⚠️ 保存缓存失败: {e}")


def clear_bucket_cache(image_folder_path):
    """
    清理指定数据集的bucket缓存
    """
    try:
        cache_dir = os.path.join(os.path.dirname(image_folder_path), '.bucket_cache')
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print(f"✅ 已清理缓存目录: {cache_dir}")
        else:
            print(f"⚠️ 缓存目录不存在: {cache_dir}")
    except Exception as e:
        print(f"⚠️ 清理缓存失败: {e}")


def get_cache_info(image_folder_path):
    """
    获取缓存信息
    """
    cache_dir = os.path.join(os.path.dirname(image_folder_path), '.bucket_cache')
    if not os.path.exists(cache_dir):
        return {'exists': False, 'files': []}
    
    cache_files = []
    for filename in os.listdir(cache_dir):
        if filename.startswith('bucket_cache_') and filename.endswith('.pkl'):
            filepath = os.path.join(cache_dir, filename)
            stat = os.stat(filepath)
            cache_files.append({
                'filename': filename,
                'size': stat.st_size,
                'modified': stat.st_mtime
            })
    
    return {
        'exists': True,
        'cache_dir': cache_dir,
        'files': cache_files
    }


def create_aspect_ratio_buckets(base_size=1024, multiple_of=32, max_ratio_error=0.1):
    """
    生成基于宽高比的桶列表
    """
    buckets = []
    target_area = base_size * base_size
    
    for w in range(multiple_of, (base_size * 2) + 1, multiple_of):
        for h in range(multiple_of, (base_size * 2) + 1, multiple_of):
            area = w * h
            error = abs(area - target_area)
            buckets.append({'width': w, 'height': h, 'error': error})

    buckets.sort(key=lambda x: x['error'])
    
    max_allowed_error = buckets[int(len(buckets) * max_ratio_error)]['error']
    
    final_buckets = set()
    for bucket in buckets:
        if bucket['error'] <= max_allowed_error:
            final_buckets.add((bucket['width'], bucket['height']))

    sorted_buckets = sorted(list(final_buckets), key=lambda x: x[0] / x[1])
    
    print(f"✅ 创建了 {len(sorted_buckets)} 个桶")
    return sorted_buckets


def get_all_image_files(folder_path):
    """
    递归获取文件夹及其子文件夹中的所有图片文件
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIFF'}
    image_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                file_path = os.path.join(root, file)
                image_files.append(file_path)
    image_files.sort()
    return image_files


class AspectRatioBucketDataset(Dataset):
    """基于宽高比桶的数据集类"""
    
    ### <<< 修正/重点 >>> ###
    # 构造函数现在接收 base_size 和 multiple_of 以生成正确的缓存路径
    def __init__(
        self,
        image_folder_path,
        buckets,
        base_size,
        multiple_of,
        max_samples=None,
        use_cache=True,
    ):
        """
        Args:
            image_folder_path: 图片文件夹路径
            buckets: 桶列表
            base_size: 基础尺寸 (用于缓存)
            multiple_of: 倍数 (用于缓存)
            max_samples: 最大图像数量
            use_cache: 是否使用缓存
        """
        self.image_folder_path = image_folder_path
        self.buckets = buckets
        self.max_samples = max_samples
        self.use_cache = use_cache
        self.base_size = base_size
        self.multiple_of = multiple_of

        # 仅进行文件扫描与桶分配
        self.sharpness_settings = None
        self.sharpness_calibration = None
        
        if self.use_cache:
            cache_path = get_cache_path(image_folder_path, self.base_size, self.multiple_of, max_samples)
            cache_data = load_bucket_cache(cache_path)

            if cache_data is not None:
                cached_buckets = cache_data.get('buckets')
                if cached_buckets == buckets:
                    cached_image_data = cache_data.get('image_data', [])
                    cached_bucket_stats = cache_data.get('bucket_stats', {})
                    can_use_cache = True

                    if can_use_cache:
                        self.image_data = cached_image_data

                        # 即使从缓存加载，也需要根据 max_samples 决定是否进入重复模式或截断
                        if self.max_samples is not None and self.max_samples > 0:
                            if len(self.image_data) >= self.max_samples:
                                self.image_data = self.image_data[:self.max_samples]
                                print(f"🔒 Overfitting模式：限制使用前 {self.max_samples} 张图像（缓存）")
                            else:
                                self.original_size = len(self.image_data)
                                self.target_size = self.max_samples
                                print(f"🔄 数据集重复模式（缓存）：原始 {self.original_size} 张图像，将重复到 {self.max_samples} 张图像")

                        self._print_bucket_stats(cached_bucket_stats)
                        return
                else:
                    print("⚠️ 缓存中的buckets与当前buckets不匹配，重新计算...")
        
        self._compute_bucket_assignments()
    
    def _print_bucket_stats(self, bucket_stats):
        """打印桶的统计信息"""
        print(f"✅ 桶分配完成！共使用 {len(bucket_stats)} 个桶")
        if len(bucket_stats) <= 10:
            for bucket_id, count in sorted(bucket_stats.items()):
                bucket_w, bucket_h = self.buckets[bucket_id]
                aspect_ratio = bucket_w / bucket_h
                print(f"  桶 {bucket_id}: {bucket_w}x{bucket_h} (比例 {aspect_ratio:.2f}) - {count} 张图片")

    def _compute_bucket_assignments(self):
        """计算桶分配"""
        print(f"正在扫描文件夹: {self.image_folder_path}")
        self.image_files = get_all_image_files(self.image_folder_path)
        # 保存原始文件列表用于重复模式判断
        self.original_image_files = self.image_files.copy()
        
        if self.max_samples is not None and self.max_samples > 0:
            if len(self.image_files) >= self.max_samples:
                # 数据集足够大，截断到 max_samples
                self.image_files = self.image_files[:self.max_samples]
                print(f"🔒 Overfitting模式：限制使用前 {self.max_samples} 张图像")
            else:
                # 数据集不足，记录重复模式信息，但桶分配仍基于原始数据
                self.original_size = len(self.image_files)
                self.target_size = self.max_samples
                print(f"🔄 数据集重复模式：原始 {self.original_size} 张图像，将重复到 {self.max_samples} 张图像")
        
        self.image_data = []
        
        print(f"预计算图片宽高比并分配桶... 找到 {len(self.image_files)} 张图片")
        bucket_stats = {}
        total_files = len(self.image_files)
        start_t = time.perf_counter()
        next_progress = 10  # 每10%打印一次
        
        for idx, filepath in enumerate(self.image_files, 1):
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
                    aspect_ratio = width / height
                    area = width * height

                    best_bucket_id = min(
                        range(len(self.buckets)),
                        key=lambda i: (
                            abs(aspect_ratio - (self.buckets[i][0] / self.buckets[i][1])),
                            abs(area - (self.buckets[i][0] * self.buckets[i][1]))
                        )
                    )

                entry = {
                    'filepath': filepath,
                    'bucket_id': best_bucket_id
                }

                self.image_data.append(entry)

                bucket_stats[best_bucket_id] = bucket_stats.get(best_bucket_id, 0) + 1

            except Exception as e:
                print(f"警告: 无法处理图片 {filepath}: {e}")
                continue
            
            # 打印进度（每完成10%打印一次，并估算剩余时间）
            if total_files > 0:
                pct_int = int((idx * 100) / total_files)
                if pct_int >= next_progress or idx == total_files:
                    elapsed = max(1e-6, time.perf_counter() - start_t)
                    remaining_items = max(0, total_files - idx)
                    sec_per_item = elapsed / idx
                    remain_sec = remaining_items * sec_per_item
                    # 格式化剩余时间
                    if remain_sec >= 3600:
                        h = int(remain_sec // 3600)
                        m = int((remain_sec % 3600) // 60)
                        s = int(remain_sec % 60)
                        eta_str = f"{h}:{m:02d}:{s:02d}"
                    elif remain_sec >= 60:
                        m = int(remain_sec // 60)
                        s = int(remain_sec % 60)
                        eta_str = f"{m}:{s:02d}"
                    else:
                        eta_str = f"{int(remain_sec)}s"
                    print(f"  进度: {idx}/{total_files} ({pct_int}%) | 预计剩余: {eta_str}")
                    while next_progress <= pct_int:
                        next_progress += 10
        
        self._print_bucket_stats(bucket_stats)

        if self.use_cache:
            cache_path = get_cache_path(self.image_folder_path, self.base_size, self.multiple_of, self.max_samples)
            save_bucket_cache(
                cache_path,
                self.buckets,
                self.image_data,
                bucket_stats,
                # 精简后不再存储清晰度相关信息
            )
        # 无需处理预计算索引签名等
            
    def __len__(self):
        if hasattr(self, 'target_size'):
            return self.target_size
        return len(self.image_data)
        
    def __getitem__(self, index):
        if hasattr(self, 'target_size'):
            original_index = index % len(self.image_data)
            return self.image_data[original_index]
        return self.image_data[index]


### <<< 修正/重点 >>> ###
# 1. 继承自正确的 BatchSampler 基类
# 2. 修改 __init__ 签名以接收一个外部 sampler，从而兼容DDP
class AspectRatioBatchSampler(BatchSampler):
    """
    基于宽高比桶的批次采样器。
    它包装一个标准的 Sampler 对象，并根据桶信息将索引重新组合成批次。
    这确保了与分布式训练 (DDP) 的兼容性。
    """
    
    ### <<< MODIFICATION START >>> ###
    # The __init__ signature is simplified. We remove the 'image_data' argument.
    def __init__(self, sampler, batch_size, drop_last, shuffle=True):
        """
        Args:
            sampler (Sampler): A standard sampler instance (e.g., RandomSampler or DistributedSampler).
            batch_size (int): The batch size.
            drop_last (bool): Whether to drop the last incomplete batch.
            shuffle (bool): Whether to shuffle the batches (for training mode).
        """
        super().__init__(sampler, batch_size, drop_last)
        self.shuffle = shuffle
        
        ### <<< FINAL CORRECTION START >>> ###
        # Get the dataset object robustly from the sampler.
        # RandomSampler/SequentialSampler use 'data_source', while DistributedSampler uses 'dataset'.
        if hasattr(self.sampler, 'dataset'):
            dataset = self.sampler.dataset
        else:
            dataset = self.sampler.data_source
        
        # Now, retrieve image_data from the dataset
        self.image_data = dataset.image_data
        ### <<< FINAL CORRECTION END >>> ###
        ### <<< MODIFICATION END >>> ###
        
        # Detect repeat mode: dataset length is artificially expanded to target_size
        dataset = self.sampler.dataset if hasattr(self.sampler, 'dataset') else self.sampler.data_source
        self._repeat_mode = hasattr(dataset, 'target_size')
        
        if self._repeat_mode:
            # In repeat mode, avoid materializing an enormous list(self.sampler).
            # Build base bucket -> indices mapping from the original (non-repeated) indices.
            self._original_len = len(self.image_data)
            self._target_len = len(dataset)  # equals dataset.target_size
            
            self._base_buckets_to_indices = {}
            for idx in range(self._original_len):
                bucket_id = self.image_data[idx]['bucket_id']
                if bucket_id not in self._base_buckets_to_indices:
                    self._base_buckets_to_indices[bucket_id] = []
                self._base_buckets_to_indices[bucket_id].append(idx)
            
            # Precompute target number of batches to yield
            full_batches = self._target_len // self.batch_size
            has_partial = (self._target_len % self.batch_size) != 0
            self._num_batches_target = full_batches if self.drop_last else (full_batches + (1 if has_partial else 0))
            self._bucket_ids = list(self._base_buckets_to_indices.keys())
        else:
            # The rest of the logic remains the same for normal mode...
            indices = list(self.sampler)
            
            self.buckets_to_indices = {}
            for idx in indices:
                # 当数据集启用重复模式时，映射到有效的原始索引
                if hasattr(dataset, 'target_size'):
                    effective_idx = idx % len(self.image_data)
                else:
                    effective_idx = idx
                bucket_id = self.image_data[effective_idx]['bucket_id']
                if bucket_id not in self.buckets_to_indices:
                    self.buckets_to_indices[bucket_id] = []
                self.buckets_to_indices[bucket_id].append(idx)
                
            self.batches = []
            for bucket_id in self.buckets_to_indices:
                bucket_indices = self.buckets_to_indices[bucket_id]
                if self.shuffle:
                    random.shuffle(bucket_indices)
                
                for i in range(0, len(bucket_indices), self.batch_size):
                    batch = bucket_indices[i:i + self.batch_size]
                    if len(batch) < self.batch_size and self.drop_last:
                        continue
                    self.batches.append(batch)
            
            if self.shuffle:
                random.shuffle(self.batches)
        
    def __iter__(self):
        if not getattr(self, '_repeat_mode', False):
            # 迭代器直接返回预先计算好的批次列表
            yield from self.batches
            return
        
        # Lazy batch generation for repeat mode to avoid huge memory/time costs
        yielded = 0
        # To mix buckets across batches, randomize bucket traversal order per cycle if shuffle enabled
        while yielded < self._num_batches_target:
            bucket_order = self._bucket_ids[:]
            if self.shuffle:
                random.shuffle(bucket_order)
            for bucket_id in bucket_order:
                base_list = self._base_buckets_to_indices[bucket_id]
                if not base_list:
                    continue
                work_list = base_list[:]  # copy
                if self.shuffle:
                    random.shuffle(work_list)
                # Chunk into batches of size batch_size
                for i in range(0, len(work_list), self.batch_size):
                    if yielded >= self._num_batches_target:
                        return
                    batch = work_list[i:i + self.batch_size]
                    if len(batch) < self.batch_size and self.drop_last:
                        continue
                    # Map original small indices into dataset space; DataLoader will call dataset.__getitem__,
                    # which already supports modulo indexing in repeat mode, so we can yield original indices directly.
                    yield batch
                    yielded += 1
            # Loop continues until enough batches yielded
        
    def __len__(self):
        # 长度是批次的数量
        if getattr(self, '_repeat_mode', False):
            return self._num_batches_target
        return len(self.batches)


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
    resized_img = pil_img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

    # Calculate coordinates for center crop
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    # Crop and return
    return resized_img.crop((left, top, right, bottom))


def create_cond_image(pil_image, downsample_factor=2, upsample_factor=1, downsample_resample=Image.Resampling.BICUBIC, upsample_resample=None):
    """
    创建可调节下采样倍率的条件图像
    """
    cond_width = max(1, pil_image.width // downsample_factor)
    cond_height = max(1, pil_image.height // downsample_factor)
    cond_pil = pil_image.resize((cond_width, cond_height), resample=downsample_resample)

    final_upsample_resample = upsample_resample or downsample_resample
    if upsample_factor > 1:
        cond_width = max(1, cond_width * upsample_factor)
        cond_height = max(1, cond_height * upsample_factor)
        cond_pil = cond_pil.resize((cond_width, cond_height), resample=final_upsample_resample)

    cond_tensor = T.ToTensor()(cond_pil)
    return cond_tensor


def create_collate_fn(
    buckets,
    downsample_factor=2,
    upsample_factor=1,
    skip_cond_image=False,
    use_degraded_image=False,
    cond_downsample_resample=Image.Resampling.BICUBIC,
    cond_upsample_resample=None,
    cond_preblur=False,
    profile_collate=False,
    load_text=False,
    caption_json_key="text",
    enable_sharpness=False,
):
    """创建 collate_fn，整合 resize + crop 的完整流程。"""

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

        _profile = profile_collate or (os.environ.get('BUCKET_PROFILE', '0') == '1')
        bucket_id = batch[0]['bucket_id']
        target_dims = buckets[bucket_id]

        processed_images = []
        processed_cond_images = []
        image_keys = []
        text_needed = load_text
        text_data = [] if text_needed else None

        to_tensor_transform = T.ToTensor()
        cond_upsample = cond_upsample_resample or cond_downsample_resample

        if _profile:
            t_open = t_resize = t_cond = t_tensor = 0.0

        for item in batch:
            try:
                t0 = time.perf_counter() if _profile else None
                pil_img = Image.open(item['filepath']).convert('RGB')
                if _profile:
                    t_open += (time.perf_counter() - t0)

                t0 = time.perf_counter() if _profile else None
                final_img = resize_to_fill_and_crop(pil_img, target_dims)
                if _profile:
                    t_resize += (time.perf_counter() - t0)

                if use_degraded_image:
                    t0 = time.perf_counter() if _profile else None
                    cond_width = max(1, final_img.width // downsample_factor)
                    cond_height = max(1, final_img.height // downsample_factor)
                    cond_source_img = final_img.filter(_COND_PREBLUR_FILTER) if cond_preblur else final_img
                    cond_pil = cond_source_img.resize((cond_width, cond_height), resample=cond_downsample_resample)
                    degraded_pil = cond_pil.resize((final_img.width, final_img.height), resample=cond_upsample)
                    tensor_img = to_tensor_transform(degraded_pil)
                    if _profile:
                        t_cond += (time.perf_counter() - t0)
                else:
                    t0 = time.perf_counter() if _profile else None
                    tensor_img = to_tensor_transform(final_img)
                    if _profile:
                        t_tensor += (time.perf_counter() - t0)

                image_key = os.path.splitext(os.path.basename(item['filepath']))[0]
                processed_images.append(tensor_img)
                image_keys.append(image_key)

                caption = ""
                if load_text:
                    json_path = os.path.splitext(item['filepath'])[0] + '.json'
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                raw_caption = json.load(f)
                            caption = _extract_caption_text(raw_caption)
                        except Exception as e:
                            print(f"警告: 加载文本文件时出错 {json_path}: {e}")

                # 如果启用sharpness且需要文本数据，尝试从JSON读取sharpness并append
                if enable_sharpness and text_data is not None:
                    try:
                        json_path = os.path.splitext(item['filepath'])[0] + '.json'
                        if os.path.exists(json_path):
                            with open(json_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            if isinstance(data, dict) and 'sharpness' in data:
                                sharpness_val = data['sharpness']
                                # 尝试转换为整数
                                try:
                                    sharpness_val = int(float(sharpness_val))
                                    sharpness_text = f"sharpness: {sharpness_val}"
                                except (TypeError, ValueError):
                                    sharpness_text = f"sharpness: {sharpness_val}"
                                # Append到caption
                                caption = f"{caption.strip()} {sharpness_text}".strip() if caption else sharpness_text
                    except Exception as e:
                        # 静默处理，不影响正常流程
                        pass

                if text_data is not None:
                    text_data.append(caption)

                if skip_cond_image:
                    processed_cond_images.append(None)
                else:
                    if use_degraded_image:
                        t0 = time.perf_counter() if _profile else None
                        cond_width = max(1, final_img.width // downsample_factor)
                        cond_height = max(1, final_img.height // downsample_factor)
                        cond_source_img = final_img.filter(_COND_PREBLUR_FILTER) if cond_preblur else final_img
                        cond_pil = cond_source_img.resize((cond_width, cond_height), resample=cond_downsample_resample)
                        cond_tensor = torch.clamp(to_tensor_transform(cond_pil), 0.0, 1.0)
                        if _profile:
                            t_cond += (time.perf_counter() - t0)
                    else:
                        cond_source_img = final_img.filter(_COND_PREBLUR_FILTER) if cond_preblur else final_img
                        t0 = time.perf_counter() if _profile else None
                        cond_tensor = create_cond_image(
                            cond_source_img,
                            downsample_factor=downsample_factor,
                            upsample_factor=upsample_factor,
                            downsample_resample=cond_downsample_resample,
                            upsample_resample=cond_upsample
                        )
                        if _profile:
                            t_cond += (time.perf_counter() - t0)
                    processed_cond_images.append(cond_tensor)

            except Exception as e:
                print(f"警告: 处理图片时出错 {item['filepath']}: {e}")
                default_img = torch.zeros(3, target_dims[1], target_dims[0])
                default_key = f"error_{len(processed_images)}"

                processed_images.append(default_img)
                image_keys.append(default_key)
                if text_data is not None:
                    text_data.append("")

                if skip_cond_image:
                    processed_cond_images.append(None)
                else:
                    default_cond_img = torch.zeros(3, target_dims[1] // downsample_factor, target_dims[0] // downsample_factor)
                    processed_cond_images.append(default_cond_img)

        result = {
            'image': torch.stack(processed_images),
            '__key__': image_keys
        }

        if skip_cond_image:
            result['cond_image'] = None
        else:
            result['cond_image'] = torch.stack(processed_cond_images)

        if text_data is not None:
            result['strText'] = text_data

        if _profile:
            total = t_open + t_resize + t_cond + t_tensor
            try:
                print(f"[BUCKET_PROFILE] items={len(batch)} open={t_open:.3f}s resize={t_resize:.3f}s cond={t_cond:.3f}s tensor={t_tensor:.3f}s total={total:.3f}s")
            except Exception:
                pass
        return result

    text_output_flag = '开启' if load_text else '关闭'
    json_load_flag = '开启' if load_text else '关闭'
    sharpness_flag = '开启' if enable_sharpness else '关闭'
    print(
        f"✅ 已创建 Collate Function，下采样倍率: {downsample_factor}, 上采样倍率: {upsample_factor}, "
        f"degraded: {use_degraded_image}, 文本加载: {json_load_flag}, 文本输出: {text_output_flag}, 清晰度提示: {sharpness_flag}"
    )
    return collate_fn


def create_dataloader(
    image_folder_path,
    base_size=1024,
    multiple_of=32,
    batch_size=4,
    num_workers=4,
    drop_last=False,
    downsample_factor=2,
    upsample_factor=1,
    shuffle=True,
    max_samples=None,
    use_cache=True,
    skip_cond_image=False,
    use_degraded_image=False,
    cond_downsample_resample=Image.Resampling.BICUBIC,
    cond_upsample_resample=None,
    cond_preblur=False,
    profile_collate=False,
    load_text=False,
    caption_json_key="text",
    enable_sharpness=False,
):
    """创建完整的 DataLoader。"""

    print("🚀 开始创建 DataLoader...")
    print(f"下采样倍率: {downsample_factor}, 上采样倍率: {upsample_factor}, multiple_of: {multiple_of}, degraded: {use_degraded_image}")
    print(f"随机化模式: {'训练模式（随机）' if shuffle else '验证模式（固定顺序）'}")
    if load_text:
        print(f"文本加载: 启用，caption_json_key='{caption_json_key}'")
    if enable_sharpness:
        print(f"清晰度提示: 启用")
    if max_samples is not None:
        print(f"🔒 Overfitting模式：限制使用前 {max_samples} 张图像")
    
    print("(1/5) 创建宽高比桶...")
    buckets = create_aspect_ratio_buckets(base_size=base_size, multiple_of=multiple_of)
    
    print("(2/5) 创建数据集...")
    ### <<< 修正/重点 >>> ###
    # 将 base_size 和 multiple_of 传入 Dataset 以修复缓存路径问题
    dataset = AspectRatioBucketDataset(
        image_folder_path=image_folder_path, 
        buckets=buckets, 
        base_size=base_size, 
        multiple_of=multiple_of,
        max_samples=max_samples, 
        use_cache=use_cache,
    )
    
    if len(dataset) == 0:
        raise ValueError(f"在 {image_folder_path} 中没有找到有效的图片文件")
    
    print("(3/5) 创建采样器...")
    ### <<< 修正/重点 >>> ###
    # 这是使用自定义 BatchSampler 的正确流程
    # 1. 创建一个基础的 sampler (RandomSampler 或 SequentialSampler)
    #    在DDP环境中，这个 sampler 会被框架（如Lightning）自动替换为 DistributedSampler
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    
    # 2. 用我们修正后的 AspectRatioBatchSampler 包装基础 sampler
    batch_sampler = AspectRatioBatchSampler(
        sampler=sampler,
        batch_size=batch_size, 
        drop_last=drop_last, 
        shuffle=shuffle
    )
    
    print("(4/5) 创建 collate 函数...")
    collate_function = create_collate_fn(
        buckets,
        downsample_factor=downsample_factor,
        upsample_factor=upsample_factor,
        skip_cond_image=skip_cond_image,
        use_degraded_image=use_degraded_image,
        cond_downsample_resample=cond_downsample_resample,
        cond_upsample_resample=cond_upsample_resample,
        cond_preblur=cond_preblur,
        profile_collate=profile_collate or (os.environ.get('BUCKET_PROFILE', '0') == '1'),
        load_text=load_text,
        caption_json_key=caption_json_key,
        enable_sharpness=enable_sharpness,
    )
    
    print("(5/5) 创建 DataLoader...")
    ### <<< 修正/重点 >>> ###
    # 当使用 batch_sampler 时，必须移除 batch_size, shuffle, sampler, drop_last 参数
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_function,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print(f"✅ DataLoader 创建完成！数据集大小: {len(dataset)}, 批次数量: {len(batch_sampler)}")
    return dataloader


def test_dataloader(image_folder_path, num_epochs=2, max_batches=5, downsample_factor=2):
    """
    测试 DataLoader 的功能。
    """
    print(f"🧪 开始测试 DataLoader...")
    print(f"图片文件夹: {image_folder_path}")
    print(f"下采样倍率: {downsample_factor}")
    
    try:
        dataloader = create_dataloader(
            image_folder_path=image_folder_path,
            base_size=512,
            multiple_of=64,
            batch_size=2,
            num_workers=0,
            drop_last=True,
            downsample_factor=downsample_factor,
            shuffle=True,
            use_cache=False # 测试时建议禁用缓存以确保逻辑正确
        )
        
        for epoch in range(num_epochs):
            print(f"\n--- 第 {epoch+1} 轮测试 ---")
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                    
                images_tensor = batch['image']
                cond_images_tensor = batch['cond_image']
                print(f"批次 {i+1}:")
                print(f"  原始图像形状: {images_tensor.shape}")
                print(f"  条件图像形状: {cond_images_tensor.shape}")
                print(f"  数据类型: {images_tensor.dtype}")
                print(f"  数值范围: [{images_tensor.min():.3f}, {images_tensor.max():.3f}]")
                
                if torch.isnan(images_tensor).any():
                    print("  警告: 检测到 NaN 值!")
                if torch.isinf(images_tensor).any():
                    print("  警告: 检测到无穷大值!")
                    
        print("\n✅ DataLoader 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 配置参数
    BASE_SIZE = 512
    MULTIPLE_OF = 64
    BATCH_SIZE = 4
    DOWNSAMPLE_FACTOR = 4
    # --- 请确保修改为您的图片文件夹路径 ---
    IMAGE_FOLDER = './test_images' # 示例路径
    
    # 创建一个虚拟的图片文件夹用于测试
    if not os.path.exists(IMAGE_FOLDER):
        print(f"创建测试文件夹: {IMAGE_FOLDER}")
        os.makedirs(IMAGE_FOLDER, exist_ok=True)
        # 创建一些不同尺寸的虚拟图片
        try:
            Image.new('RGB', (1024, 512)).save(os.path.join(IMAGE_FOLDER, 'img1.png'))
            Image.new('RGB', (512, 1024)).save(os.path.join(IMAGE_FOLDER, 'img2.png'))
            Image.new('RGB', (768, 768)).save(os.path.join(IMAGE_FOLDER, 'img3.png'))
            Image.new('RGB', (640, 480)).save(os.path.join(IMAGE_FOLDER, 'img4.png'))
        except ImportError:
             print("请安装Pillow库: pip install Pillow")

    if not os.path.exists(IMAGE_FOLDER) or not os.listdir(IMAGE_FOLDER):
        print(f"❌ 图片文件夹不存在或为空: {IMAGE_FOLDER}")
        print("请修改 IMAGE_FOLDER 变量为有效的图片文件夹路径")
    else:
        # 清理一次旧缓存，确保测试的是新逻辑
        print("--- 清理旧缓存 ---")
        clear_bucket_cache(IMAGE_FOLDER)
        
        # 运行测试
        test_dataloader(IMAGE_FOLDER, num_epochs=2, max_batches=3, downsample_factor=DOWNSAMPLE_FACTOR)