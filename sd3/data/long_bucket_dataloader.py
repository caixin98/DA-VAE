import math
import os
import random
import torch
import pickle
import hashlib
from PIL import Image
from torch.utils.data import Dataset, Sampler, DataLoader, BatchSampler, RandomSampler, SequentialSampler
import torchvision.transforms as T


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
    cache_dir = os.path.join(os.path.dirname(image_folder_path), '.long_bucket_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    dataset_hash = get_dataset_hash(image_folder_path, max_samples)
    cache_filename = f"long_bucket_cache_{base_size}_{multiple_of}_{dataset_hash}.pkl"
    return os.path.join(cache_dir, cache_filename)


def load_bucket_cache(cache_path):
    """
    加载bucket缓存
    """
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            print(f"✅ 从缓存加载long bucket分配结果: {cache_path}")
            return cache_data
    except Exception as e:
        print(f"⚠️ 加载缓存失败: {e}")
    return None


def save_bucket_cache(cache_path, buckets, image_data, bucket_stats):
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
        print(f"✅ 保存long bucket分配结果到缓存: {cache_path}")
    except Exception as e:
        print(f"⚠️ 保存缓存失败: {e}")


def clear_bucket_cache(image_folder_path):
    """
    清理指定数据集的long bucket缓存
    """
    try:
        cache_dir = os.path.join(os.path.dirname(image_folder_path), '.long_bucket_cache')
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print(f"✅ 已清理long缓存目录: {cache_dir}")
        else:
            print(f"⚠️ long缓存目录不存在: {cache_dir}")
    except Exception as e:
        print(f"⚠️ 清理缓存失败: {e}")


def create_long_aspect_ratio_buckets(base_size=1024, multiple_of=32, max_ratio_error=0.1):
    """
    生成基于长边resize的宽高比桶列表
    与原始bucket不同，这里考虑的是长边resize后的宽高比
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
    
    print(f"✅ 创建了 {len(sorted_buckets)} 个long桶")
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


def resize_long_edge_pil(pil_img, base_size):
    """
    将长边resize到base_size，保持宽高比
    """
    original_w, original_h = pil_img.size
    
    if original_w >= original_h:
        # 宽度是长边
        new_w = base_size
        new_h = int(original_h * base_size / original_w)
    else:
        # 高度是长边
        new_h = base_size
        new_w = int(original_w * base_size / original_h)
    
    resized_img = pil_img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return resized_img


def random_resize_with_constraints(pil_img, min_long_edge, max_long_edge, multiple_of, lq_cond_size):
    """
    随机resize图像，保持长宽比，符合multiple_of约束，长边不小于lq_cond_size
    
    Args:
        pil_img: PIL图像
        min_long_edge: 最小长边尺寸
        max_long_edge: 最大长边尺寸
        multiple_of: 尺寸倍数约束
        lq_cond_size: 条件图像长边尺寸（最终长边不能小于此值）
    """
    original_w, original_h = pil_img.size
    aspect_ratio = original_w / original_h
    
    # 确保最小长边不小于lq_cond_size
    actual_min_long_edge = max(min_long_edge, lq_cond_size)
    
    # 在约束范围内随机选择长边尺寸
    # 确保尺寸符合multiple_of约束
    min_size = ((actual_min_long_edge - 1) // multiple_of + 1) * multiple_of
    max_size = (max_long_edge // multiple_of) * multiple_of
    
    if min_size > max_size:
        min_size = max_size
    
    if min_size == max_size:
        target_long_edge = min_size
    else:
        # 在符合multiple_of约束的尺寸中随机选择
        available_sizes = list(range(min_size, max_size + 1, multiple_of))
        target_long_edge = random.choice(available_sizes)
    
    # 根据长边resize
    if original_w >= original_h:
        # 宽度是长边
        new_w = target_long_edge
        new_h = int(original_h * target_long_edge / original_w)
    else:
        # 高度是长边
        new_h = target_long_edge
        new_w = int(original_w * target_long_edge / original_h)
    
    # 确保尺寸符合multiple_of约束
    new_w = (new_w // multiple_of) * multiple_of
    new_h = (new_h // multiple_of) * multiple_of
    
    resized_img = pil_img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return resized_img


class LongBucketDataset(Dataset):
    """基于长边resize的宽高比桶数据集类"""
    
    def __init__(self, image_folder_path, buckets, base_size, multiple_of, max_samples=None, use_cache=True):
        """
        Args:
            image_folder_path: 图片文件夹路径
            buckets: 桶列表
            base_size: 基础尺寸 (长边尺寸)
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
        
        if self.use_cache:
            cache_path = get_cache_path(image_folder_path, self.base_size, self.multiple_of, max_samples)
            cache_data = load_bucket_cache(cache_path)
            
            if cache_data is not None:
                if cache_data['buckets'] == buckets:
                    self.image_data = cache_data['image_data']
                    self._print_bucket_stats(cache_data['bucket_stats'])
                    return
                else:
                    print("⚠️ 缓存中的buckets与当前buckets不匹配，重新计算...")
        
        self._compute_bucket_assignments()
    
    def _print_bucket_stats(self, bucket_stats):
        """打印桶的统计信息"""
        print(f"✅ Long桶分配完成！共使用 {len(bucket_stats)} 个桶")
        if len(bucket_stats) <= 10:
            for bucket_id, count in sorted(bucket_stats.items()):
                bucket_w, bucket_h = self.buckets[bucket_id]
                aspect_ratio = bucket_w / bucket_h
                print(f"  桶 {bucket_id}: {bucket_w}x{bucket_h} (比例 {aspect_ratio:.2f}) - {count} 张图片")

    def _compute_bucket_assignments(self):
        """计算桶分配"""
        print(f"正在扫描文件夹: {self.image_folder_path}")
        self.image_files = get_all_image_files(self.image_folder_path)
        
        if self.max_samples is not None and self.max_samples > 0:
            self.image_files = self.image_files[:self.max_samples]
            print(f"🔒 Overfitting模式：限制使用前 {self.max_samples} 张图像")
        
        self.image_data = []
        
        print(f"预计算图片长边resize后的宽高比并分配桶... 找到 {len(self.image_files)} 张图片")
        bucket_stats = {}
        
        for filepath in self.image_files:
            try:
                with Image.open(filepath) as img:
                    original_width, original_height = img.size
                
                # 计算长边resize后的尺寸
                if original_width >= original_height:
                    # 宽度是长边
                    resized_w = self.base_size
                    resized_h = int(original_height * self.base_size / original_width)
                else:
                    # 高度是长边
                    resized_h = self.base_size
                    resized_w = int(original_width * self.base_size / original_height)
                
                aspect_ratio = resized_w / resized_h
                area = resized_w * resized_h
                
                # 找到最匹配的桶
                best_bucket_id = min(
                    range(len(self.buckets)), 
                    key=lambda i: (
                        abs(aspect_ratio - (self.buckets[i][0] / self.buckets[i][1])),
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
        
        if self.use_cache:
            cache_path = get_cache_path(self.image_folder_path, self.base_size, self.multiple_of, self.max_samples)
            save_bucket_cache(cache_path, self.buckets, self.image_data, bucket_stats)
            
    def __len__(self):
        return len(self.image_data)
        
    def __getitem__(self, index):
        return self.image_data[index]


class LongBucketBatchSampler(BatchSampler):
    """
    基于长边resize宽高比桶的批次采样器
    """
    
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
        
        # Get the dataset object robustly from the sampler
        if hasattr(self.sampler, 'dataset'):
            dataset = self.sampler.dataset
        else:
            dataset = self.sampler.data_source
        
        # Now, retrieve image_data from the dataset
        self.image_data = dataset.image_data
        
        # The rest of the logic remains the same...
        indices = list(self.sampler)
        
        self.buckets_to_indices = {}
        for idx in indices:
            bucket_id = self.image_data[idx]['bucket_id']
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
        # 迭代器直接返回预先计算好的批次列表
        return iter(self.batches)
        
    def __len__(self):
        # 长度是批次的数量
        return len(self.batches)


def resize_for_cropping_long_pil(pil_img, target_bucket_dims, base_size):
    """
    长边resize模式：先将长边resize到base_size，然后保持宽高比进行缩放
    """
    target_w, target_h = target_bucket_dims
    
    # 先进行长边resize
    resized_img = resize_long_edge_pil(pil_img, base_size)
    resized_w, resized_h = resized_img.size
    
    # 然后按照bucket尺寸进行缩放，保持宽高比
    base_size_for_scale = min(target_w, target_h)
    aspect_ratio = resized_w / resized_h

    new_w = int(round(max(base_size_for_scale * aspect_ratio, base_size_for_scale)))
    new_h = int(round(max(base_size_for_scale / aspect_ratio, base_size_for_scale)))

    final_resized_img = resized_img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return final_resized_img


def crop_pil(pil_img, target_dims):
    """
    中心裁剪逻辑
    """
    target_w, target_h = target_dims
    img_w, img_h = pil_img.size

    left = (img_w - target_w) // 2
    top = (img_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    
    cropped_img = pil_img.crop((left, top, right, bottom))
    return cropped_img


def create_cond_image(pil_image, downsample_factor=2, upsample_factor=1):
    """
    创建可调节下采样倍率的条件图像
    """
    cond_width = pil_image.width // downsample_factor
    cond_height = pil_image.height // downsample_factor
    cond_pil = pil_image.resize((cond_width, cond_height), resample=Image.Resampling.BICUBIC)
    if upsample_factor > 1:
        cond_width = cond_width * upsample_factor
        cond_height = cond_height * upsample_factor
        cond_pil = cond_pil.resize((cond_width, cond_height), resample=Image.Resampling.BICUBIC)
        
    cond_tensor = T.ToTensor()(cond_pil)
    return cond_tensor


def create_long_collate_fn(buckets, base_size, downsample_factor=2, upsample_factor=1, skip_cond_image=False, use_degraded_image=False, lq_cond_size=None, random_resize=False, multiple_of=32):
    """
    创建long模式的 collate_fn，整合长边resize + crop 的完整流程
    
    Args:
        buckets: 桶列表
        base_size: 基础尺寸（长边尺寸）
        downsample_factor: 下采样倍率（当lq_cond_size为None时使用）
        upsample_factor: 上采样倍率（当lq_cond_size为None时使用）
        skip_cond_image: 是否跳过cond_image准备
        use_degraded_image: 是否使用降质图像
        lq_cond_size: 条件图像长边尺寸（优先级最高，如果设置则忽略downsample_factor和upsample_factor）
        random_resize: 是否对最终输出图像进行随机resize
        multiple_of: 随机resize时尺寸的倍数约束
    """
    def collate_fn(batch):
        bucket_id = batch[0]['bucket_id']
        target_dims = buckets[bucket_id]

        processed_images = []
        processed_cond_images = []
        image_keys = []
        
        to_tensor_transform = T.ToTensor()

        for item in batch:
            try:
                pil_img = Image.open(item['filepath']).convert('RGB')
                
                # 长边resize + 保持宽高比缩放
                intermediate_img = resize_for_cropping_long_pil(pil_img, target_dims, base_size)
                final_img = crop_pil(intermediate_img, target_dims)
                
                # 如果启用随机resize，对最终图像进行随机resize
                if random_resize and lq_cond_size is not None:
                    # 随机resize范围：从lq_cond_size到base_size
                    min_long_edge = lq_cond_size
                    max_long_edge = base_size
                    final_img = random_resize_with_constraints(
                        final_img, min_long_edge, max_long_edge, multiple_of, lq_cond_size
                    )

                # 生成 cond（低分）
                if lq_cond_size is not None:
                    # 使用lq_cond_size参数，将条件图像的长边resize到指定尺寸
                    cond_pil = resize_long_edge_pil(final_img, lq_cond_size)
                else:
                    # 使用传统的downsample_factor方式
                    cond_width = max(1, final_img.width // downsample_factor)
                    cond_height = max(1, final_img.height // downsample_factor)
                    cond_pil = final_img.resize((cond_width, cond_height), resample=Image.Resampling.BICUBIC)

                if use_degraded_image:
                    # image 使用 下采样->上采样 回到目标分辨率
                    degraded_pil = cond_pil.resize((final_img.width, final_img.height), resample=Image.Resampling.BICUBIC)
                    tensor_img = to_tensor_transform(degraded_pil)
                else:
                    # 常规：image 为裁剪后的高分图
                    tensor_img = to_tensor_transform(final_img)
                image_key = os.path.splitext(os.path.basename(item['filepath']))[0]
                
                processed_images.append(tensor_img)
                image_keys.append(image_key)
                
                # 如果跳过cond_image准备，添加None
                if skip_cond_image:
                    processed_cond_images.append(None)
                else:
                    if use_degraded_image:
                        # degraded 模式下，cond 仅为低分，不做上采样
                        cond_tensor = T.ToTensor()(cond_pil)
                    else:
                        if lq_cond_size is not None:
                            # 使用lq_cond_size时，直接使用resize后的cond_pil
                            cond_tensor = T.ToTensor()(cond_pil)
                        else:
                            # 使用传统的downsample_factor + upsample_factor方式
                            cond_tensor = create_cond_image(final_img, downsample_factor=downsample_factor, upsample_factor=upsample_factor)
                    processed_cond_images.append(cond_tensor)
                
            except Exception as e:
                print(f"警告: 处理图片时出错 {item['filepath']}: {e}")
                default_img = torch.zeros(3, target_dims[1], target_dims[0])
                default_key = f"error_{len(processed_images)}"
                
                processed_images.append(default_img)
                image_keys.append(default_key)
                
                # 如果跳过cond_image准备，添加None
                if skip_cond_image:
                    processed_cond_images.append(None)
                else:
                    if lq_cond_size is not None:
                        # 使用lq_cond_size时，计算默认条件图像尺寸
                        # 假设默认条件图像为正方形，边长为lq_cond_size
                        default_cond_img = torch.zeros(3, lq_cond_size, lq_cond_size)
                    else:
                        # 使用传统downsample_factor方式
                        default_cond_img = torch.zeros(3, target_dims[1] // downsample_factor, target_dims[0] // downsample_factor)
                    processed_cond_images.append(default_cond_img)
            
        result = {
            'image': torch.stack(processed_images),
            '__key__': image_keys
        }
        
        # 如果跳过cond_image准备，返回None
        if skip_cond_image:
            result['cond_image'] = None
        else:
            result['cond_image'] = torch.stack(processed_cond_images)
        
        return result
        
    if lq_cond_size is not None:
        resize_info = f", 随机resize: {random_resize}" if random_resize else ""
        print(f"✅ 已创建Long Collate Function，长边尺寸: {base_size}, 条件图像长边: {lq_cond_size}, degraded: {use_degraded_image}{resize_info}")
    else:
        print(f"✅ 已创建Long Collate Function，长边尺寸: {base_size}, 下采样倍率: {downsample_factor}, 上采样倍率: {upsample_factor}, degraded: {use_degraded_image}")
    return collate_fn


def create_long_dataloader(image_folder_path, base_size=1024, multiple_of=32, batch_size=4, 
                          num_workers=4, drop_last=False, downsample_factor=2, upsample_factor=1, 
                          shuffle=True, max_samples=None, use_cache=True, skip_cond_image=False, 
                          use_degraded_image=False, lq_cond_size=None, random_resize=False):
    """
    创建完整的Long模式DataLoader
    
    Args:
        lq_cond_size: 条件图像长边尺寸（优先级最高，如果设置则忽略downsample_factor和upsample_factor）
        random_resize: 是否对最终输出图像进行随机resize（仅在训练时启用，测试时默认关闭）
    """
    print("🚀 开始创建Long模式DataLoader...")
    if lq_cond_size is not None:
        resize_info = f", 随机resize: {random_resize}" if random_resize else ""
        print(f"长边尺寸: {base_size}, 条件图像长边: {lq_cond_size}, multiple_of: {multiple_of}, degraded: {use_degraded_image}{resize_info}")
    else:
        print(f"长边尺寸: {base_size}, 下采样倍率: {downsample_factor}, 上采样倍率: {upsample_factor}, multiple_of: {multiple_of}, degraded: {use_degraded_image}")
    print(f"随机化模式: {'训练模式（随机）' if shuffle else '验证模式（固定顺序）'}")
    if max_samples is not None:
        print(f"🔒 Overfitting模式：限制使用前 {max_samples} 张图像")
    
    print("(1/5) 创建长边resize宽高比桶...")
    buckets = create_long_aspect_ratio_buckets(base_size=base_size, multiple_of=multiple_of)
    
    print("(2/5) 创建Long数据集...")
    dataset = LongBucketDataset(
        image_folder_path=image_folder_path, 
        buckets=buckets, 
        base_size=base_size, 
        multiple_of=multiple_of,
        max_samples=max_samples, 
        use_cache=use_cache
    )
    
    if len(dataset) == 0:
        raise ValueError(f"在 {image_folder_path} 中没有找到有效的图片文件")
    
    print("(3/5) 创建采样器...")
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    
    # 用LongBucketBatchSampler包装基础sampler
    batch_sampler = LongBucketBatchSampler(
        sampler=sampler,
        batch_size=batch_size, 
        drop_last=drop_last, 
        shuffle=shuffle
    )
    
    print("(4/5) 创建long collate函数...")
    collate_function = create_long_collate_fn(
        buckets, 
        base_size=base_size,
        downsample_factor=downsample_factor, 
        upsample_factor=upsample_factor, 
        skip_cond_image=skip_cond_image, 
        use_degraded_image=use_degraded_image,
        lq_cond_size=lq_cond_size,
        random_resize=random_resize,
        multiple_of=multiple_of
    )
    
    print("(5/5) 创建DataLoader...")
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_function,
        num_workers=num_workers
    )
    
    print(f"✅ Long模式DataLoader创建完成！数据集大小: {len(dataset)}, 批次数量: {len(batch_sampler)}")
    return dataloader


def test_long_dataloader(image_folder_path, num_epochs=2, max_batches=5, downsample_factor=2):
    """
    测试Long模式DataLoader的功能
    """
    print(f"🧪 开始测试Long模式DataLoader...")
    print(f"图片文件夹: {image_folder_path}")
    print(f"下采样倍率: {downsample_factor}")
    
    try:
        dataloader = create_long_dataloader(
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
                    
        print("\n✅ Long模式DataLoader测试完成！")
        
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
        test_long_dataloader(IMAGE_FOLDER, num_epochs=2, max_batches=3, downsample_factor=DOWNSAMPLE_FACTOR)
