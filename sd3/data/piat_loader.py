import torch
import math
import functools
import piat  # 假设 piat 库已安装
import json
import os
from PIL import Image, ImageFilter
import sys
import random
import numpy as np
from torchvision import transforms
import torch.distributed as dist

import piat, functools

_PIL_INTERPOLATION_MAP = {
    'nearest': Image.Resampling.NEAREST,
    'bilinear': Image.Resampling.BILINEAR,
    'bicubic': Image.Resampling.BICUBIC,
    'lanczos': Image.Resampling.LANCZOS,
    'box': Image.Resampling.BOX,
    'hamming': Image.Resampling.HAMMING,
}


_COND_PREBLUR_FILTER = ImageFilter.GaussianBlur(radius=1.0)


def _maybe_get(config_section, key, default=None):
    """Safely extract a key/attribute from OmegaConf sections or dicts."""
    if config_section is None:
        return default

    if isinstance(config_section, dict):
        return config_section.get(key, default)

    if hasattr(config_section, key):
        return getattr(config_section, key)

    if hasattr(config_section, 'get'):
        try:
            return config_section.get(key, default)
        except TypeError:
            # Some objects expose get without default parameter support
            try:
                return config_section.get(key)
            except Exception:
                return default

    return default


def _resolve_cond_interpolation_modes(config, override_section=None):
    """Return (downsample_mode, upsample_mode) strings with config fallbacks."""
    experiment_cfg = getattr(config, 'experiment', None) if config is not None else None

    downsample_mode = _maybe_get(override_section, 'cond_downsample_interpolation')
    if downsample_mode is None:
        downsample_mode = _maybe_get(experiment_cfg, 'cond_downsample_interpolation', 'lanczos')

    if downsample_mode is None:
        downsample_mode = 'lanczos'

    upsample_mode = _maybe_get(override_section, 'cond_upsample_interpolation')
    if upsample_mode is None:
        upsample_mode = _maybe_get(experiment_cfg, 'cond_upsample_interpolation')

    if upsample_mode is None:
        upsample_mode = downsample_mode

    return downsample_mode, upsample_mode


def _resolve_pil_resample(mode_name, default_resample):
    """Convert a config value into a PIL Image.Resampling enum with fallback."""
    if isinstance(mode_name, Image.Resampling):
        return mode_name

    if mode_name is None:
        return default_resample

    if isinstance(mode_name, str):
        lookup = _PIL_INTERPOLATION_MAP.get(mode_name.lower())
        if lookup is not None:
            return lookup

    print(f'⚠️ 未知的插值模式 {mode_name}，回退使用 {default_resample.name.lower()} 插值')
    return default_resample


def _resolve_cond_preblur(config, override_section=None):
    """Return bool flag indicating whether to blur before downsampling."""
    experiment_cfg = getattr(config, 'experiment', None) if config is not None else None

    value = _maybe_get(override_section, 'cond_preblur')
    if value is None:
        value = _maybe_get(experiment_cfg, 'cond_preblur', False)

    if isinstance(value, str):
        value = value.lower() in {'1', 'true', 'yes', 'on'}

    return bool(value)


def _resolve_cond_random_upsample_prob(config, override_section=None):
    """Resolve probability of upsampling LQ to HQ size."""
    experiment_cfg = getattr(config, 'experiment', None) if config is not None else None

    value = _maybe_get(override_section, 'cond_random_upsample_prob')
    if value is None:
        value = _maybe_get(experiment_cfg, 'cond_random_upsample_prob', 0.0)

    try:
        prob = float(value)
    except (TypeError, ValueError):
        prob = 0.0

    if prob < 0.0:
        prob = 0.0
    if prob > 1.0:
        prob = 1.0
    return prob


from data.piat_core import Dataloader
# 导入 bucket dataloader 用于本地图像训练
from data.bucket_dataloader import create_dataloader as create_bucket_dataloader
# 导入 long bucket dataloader 用于长边resize训练
from data.enhanced_long_dataloader import create_enhanced_long_dataloader
from data.legacy_long_dataloader import create_legacy_long_dataloader


# def degrade_image(pil_image, mode='2x'):
#     if mode == '2x':
#         # 先4倍下采样再上采样
#         w, h = pil_image.size
#         down_w, down_h = max(1, w // 2), max(1, h // 2)
#         img = pil_image.resize((down_w, down_h), resample=Image.Resampling.BICUBIC)
#         return img
  

def attach_images(objSettings, objScratch, objOutput):
    objSettings = objSettings.copy()
    downsample_factor = objSettings.get('downsample_factor', 2)
    upsample_factor = objSettings.get('upsample_factor', 1)
    skip_cond_image = objSettings.get('skip_cond_image', False)
    use_degraded_image = objSettings.get('use_degraded_image', False)
    cond_downsample_resample = objSettings.get('cond_downsample_resample', Image.Resampling.BICUBIC)
    cond_upsample_resample = objSettings.get('cond_upsample_resample', cond_downsample_resample)
    cond_preblur = objSettings.get('cond_preblur', False)

    if objOutput is None:
        return 'initialized'
    if 'npyImage' in objScratch:
        pil_img = Image.fromarray(objScratch['npyImage'].copy()).convert("RGB")
        objScratch['image'] = pil_img
        
        # 如果跳过cond_image准备，设置为None
        if skip_cond_image:
            objScratch['cond_image'] = None
        else:
            # 根据配置的插值模式进行下采样
            cond_width = pil_img.width // downsample_factor
            cond_height = pil_img.height // downsample_factor
            source_image = pil_img.filter(_COND_PREBLUR_FILTER) if cond_preblur else pil_img
            cond_pil = source_image.resize((cond_width, cond_height), resample=cond_downsample_resample)
            if use_degraded_image:
                # cond 保持为低分图；image 变为 低分图上采样回原尺寸
                degraded_pil = cond_pil.resize((pil_img.width, pil_img.height), resample=cond_upsample_resample)
                objScratch['image'] = degraded_pil
                objScratch['cond_image'] = cond_pil
            else:
                # 常规：cond 可选地上采样，image 为原图
                if upsample_factor > 1:
                    cond_width = cond_width * upsample_factor
                    cond_height = cond_height * upsample_factor
                    cond_pil = cond_pil.resize((cond_width, cond_height), resample=cond_upsample_resample)
                objScratch['cond_image'] = cond_pil
        

def load_internvl_caption(objScratch, objOutput):
        if objOutput is None:
            objScratch['objInternvlblob'] = piat.Blobfile('/sensei-fs/users/hyu/data/internvl_1p5_captions_stripe/internvl_captions_v108')
            return 'initialized'
        try:
            objScratch['strText'] = objScratch['objInternvlblob'][piat.meta_inthash(objScratch['strImagehash'])].decode('utf-8')
        except:
            return 'skip this sample'

def output_data(objScratch, objOutput):
    if objOutput is None:
        return 'initialized'
    if 'image' in objScratch:
        objOutput['image'] = transforms.ToTensor()(objScratch['image'])
    if 'cond_image' in objScratch:
        cond_img = objScratch['cond_image']
        if cond_img is None:
            objOutput['cond_image'] = None
        else:
            objOutput['cond_image'] = transforms.ToTensor()(cond_img)
    if 'strImagehash' in objScratch:
        objOutput['strImagehash'] = objScratch['strImagehash']
    if 'strText' in objScratch:
        objOutput['strText'] = objScratch['strText']

def collate_fn(batch):
    # 检查是否所有cond_image都是None（tensor downsample模式）
    cond_images = [x['cond_image'] for x in batch]
    if all(cond_img is None for cond_img in cond_images):
        # 如果所有cond_image都是None，返回None
        cond_image_tensor = None
    else:
        # 否则进行stack
        cond_image_tensor = torch.stack(cond_images)
    
    result = {
        'image': torch.stack([x['image'] for x in batch]),
        'cond_image': cond_image_tensor,
        '__key__': [x['strImagehash'] for x in batch],
    }
    
    # 检查是否有文本数据
    if batch and 'strText' in batch[0]:
        result['strText'] = [x['strText'] for x in batch]
    
    return result


def create_train_dataloader(config=None):
    """创建训练数据加载器（支持本地图像和S3数据）"""
    base_cond_down_mode, base_cond_up_mode = _resolve_cond_interpolation_modes(config)
    base_cond_down_resample = _resolve_pil_resample(base_cond_down_mode, Image.Resampling.BICUBIC)
    base_cond_up_resample = _resolve_pil_resample(base_cond_up_mode, base_cond_down_resample)

    base_cond_preblur = _resolve_cond_preblur(config)
    # 训练时的目标边长，可通过 config.base_size 控制，默认 512
    base_size = _maybe_get(config, 'base_size', 512)

    # 根据下采样倍率计算multiple_of（兼容无 experiment 的配置）
    experiment_cfg = getattr(config, 'experiment', None) if config is not None else None
    downsample_factor = _maybe_get(experiment_cfg, "downsample_factor", 2)
    upsample_factor = _maybe_get(experiment_cfg, "upsample_factor", 1)
    # skip_cond_image = config.model.get("use_tensor_downsample", False)
    skip_cond_image = _maybe_get(experiment_cfg, "skip_cond_image", False)
    use_degraded_image = _maybe_get(experiment_cfg, "use_degraded_image", False)
    multiple_of = 16 * downsample_factor if downsample_factor > 1 else 32

    if config is None:
        # 默认参数 - 使用S3数据
        downsample_factor = 2
        upsample_factor = 1
        skip_cond_image = False
        use_degraded_image = False
        multiple_of = 16 * downsample_factor if downsample_factor > 1 else 32
        base_cond_preblur = False
        batch_size = 5
        num_workers = 4
        num_threads = 8
        query_files = ['s3://sniklaus-clio-query/*/origin=si&offensiveness=0&genairemoval=v2&keywordblocklist=v1']
        
        return Dataloader(
            intBatchsize=batch_size,
            intWorkers=num_workers,
            intThreads=num_threads,
            strQueryfile=query_files,
            funcGroup=functools.partial(piat.meta_groupaspects, {'intSize': base_size, 'intMultiple': multiple_of, 'strResize': 'preserve-area'}),
            funcStages=[
                functools.partial(piat.filter_image, {'strCondition': 'fltPhoto > 0.8', 'strCondition': 'fltAesthscore > 6.0'}),
                functools.partial(piat.image_load, {'strSource': '1024-pil-antialias'}),
                functools.partial(piat.image_resize_antialias, {'intSize': base_size}),
                functools.partial(piat.image_crop_smart, {'intSize': base_size}),
                functools.partial(attach_images, {'downsample_factor': downsample_factor, 'upsample_factor': upsample_factor, 'skip_cond_image': skip_cond_image, 'use_degraded_image': use_degraded_image, 'cond_downsample_resample': base_cond_down_resample, 'cond_upsample_resample': base_cond_up_resample, 'cond_preblur': base_cond_preblur}),
                load_internvl_caption,
                output_data,
                functools.partial(piat.output_text, {}),

            ],
            intSeed=random.randint(0, 1000000),
            collate_fn=collate_fn,  
        )
    
    # 非 None 情况：读取配置（兼容无 experiment 的配置）
    experiment_cfg = getattr(config, 'experiment', None)
    downsample_factor = _maybe_get(experiment_cfg, "downsample_factor", 2)
    upsample_factor = _maybe_get(experiment_cfg, "upsample_factor", 1)
    # skip_cond_image = config.model.get("use_tensor_downsample", False)
    skip_cond_image = _maybe_get(experiment_cfg, "skip_cond_image", False)
    use_degraded_image = _maybe_get(experiment_cfg, "use_degraded_image", False)
    multiple_of = 16 * downsample_factor if downsample_factor > 1 else 32

    # 检查是否启用本地训练
    if hasattr(config, 'local_train') and config.local_train.enabled:
        print(f"使用本地图像训练，图像目录: {config.local_train.image_dir}")
        print(f"下采样倍率: {downsample_factor}, multiple_of: {multiple_of}")
        
        local_train_cfg = getattr(config, 'local_train', None)
        local_cond_down_mode, local_cond_up_mode = _resolve_cond_interpolation_modes(config, local_train_cfg)
        local_cond_down_resample = _resolve_pil_resample(local_cond_down_mode, base_cond_down_resample)
        local_cond_up_resample = _resolve_pil_resample(local_cond_up_mode, local_cond_down_resample)
        local_cond_preblur = _resolve_cond_preblur(config, local_train_cfg)
        local_cond_random_upsample_prob = _resolve_cond_random_upsample_prob(config, local_train_cfg)

        
        # 优先使用local_train中的专用参数；缺省时尝试全局dataloader；仍缺省则使用安全默认值
        batch_size = getattr(config.local_train, 'batch_size', None)
        if batch_size is None:
            batch_size = getattr(getattr(getattr(config, 'dataloader', None), 'train', None), 'batch_size', 5)
        num_workers = getattr(config.local_train, 'num_workers', None)
        if num_workers is None:
            num_workers = getattr(getattr(getattr(config, 'dataloader', None), 'train', None), 'num_workers', 4)
        
        # 获取max_samples参数，用于overfitting
        max_samples = getattr(config.local_train, 'max_samples', None)
        if max_samples is not None:
            print(f"🔒 Overfitting模式：限制使用前 {max_samples} 张图像")
        
        # 检查load_mode参数
        load_mode = getattr(config.local_train, 'load_mode', 'bucket')
        
        # 检查是否启用文本加载
        load_text = getattr(config.local_train, 'load_text', False)
        caption_json_key = _maybe_get(local_train_cfg, 'caption_json_key', None)
        if not caption_json_key:
            caption_json_key = 'text'
        
        # 检查是否启用sharpness提示
        enable_sharpness = getattr(config.local_train, 'enable_sharpness', False)
        
        if load_mode == 'simple':
            print(f"使用简单模式 - 直接random crop到固定尺寸")
            return create_local_simple_dataloader(
                image_folder_path=config.local_train.image_dir,
                target_size=config.get('base_size', 512),
                batch_size=batch_size,
                num_workers=num_workers,
                downsample_factor=downsample_factor,
                upsample_factor=upsample_factor,
                max_samples=max_samples,
                shuffle=True,
                skip_cond_image=skip_cond_image,
                use_degraded_image=use_degraded_image,
                cond_downsample_resample=local_cond_down_resample,
                cond_upsample_resample=local_cond_up_resample,
                cond_preblur=local_cond_preblur,
                load_text=load_text
            )
        elif load_mode == 'short':
            print(f"使用short模式 - 将最短边resize到base_size，然后center crop到正方形")
            return create_local_short_dataloader(
                image_folder_path=config.local_train.image_dir,
                base_size=config.get('base_size', 1024),
                batch_size=batch_size,
                num_workers=num_workers,
                downsample_factor=downsample_factor,
                upsample_factor=upsample_factor,
                max_samples=max_samples,
                shuffle=True,
                skip_cond_image=skip_cond_image,
                use_degraded_image=use_degraded_image,
                cond_downsample_resample=local_cond_down_resample,
                cond_upsample_resample=local_cond_up_resample,
                cond_preblur=local_cond_preblur,
                load_text=load_text
            )
        elif load_mode == 'long':
            print(f"使用long模式 - 将长边resize到base_size，然后基于宽高比桶的智能裁剪")
            # 获取lq_cond_size参数，优先级最高
            lq_cond_size = getattr(config.local_train, 'lq_cond_size', 512)
            # 获取random_resize参数，训练时默认启用，测试时默认关闭
            random_resize = getattr(config.local_train, 'random_resize', True)
            # 获取use_enhanced_buckets参数，默认使用增强桶系统
            use_enhanced_buckets = getattr(config.local_train, 'use_enhanced_buckets', True)
            
            if use_enhanced_buckets:
                print(f"使用增强桶系统 - 确保lq_cond和最终图像比例完全一致")
                return create_enhanced_long_dataloader(
                    image_folder_path=config.local_train.image_dir,
                    base_size=config.get('base_size', 1024),
                    multiple_of=16,   # long模式固定使用16的倍数
                    lq_cond_size=lq_cond_size,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=True,
                    shuffle=True,  # 训练模式：启用随机化
                    max_samples=max_samples,
                    skip_cond_image=skip_cond_image,
                    use_degraded_image=use_degraded_image,
                    cond_downsample_resample=local_cond_down_resample,
                    cond_upsample_resample=local_cond_up_resample,
                    cond_preblur=local_cond_preblur,
                    cond_random_upsample_prob=local_cond_random_upsample_prob,
                    load_text=load_text
                )
            else:
                print(f"使用传统桶系统 - 向后兼容模式")
                return create_legacy_long_dataloader(
                    image_folder_path=config.local_train.image_dir,
                    base_size=config.get('base_size', 1024),
                    multiple_of=multiple_of,   # long模式固定使用16的倍数
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=True,
                    downsample_factor=downsample_factor,
                    upsample_factor=upsample_factor,
                    shuffle=True,  # 训练模式：启用随机化
                    max_samples=max_samples,
                    skip_cond_image=skip_cond_image,
                    use_degraded_image=use_degraded_image,
                    cond_downsample_resample=local_cond_down_resample,
                    cond_upsample_resample=local_cond_up_resample,
                    cond_preblur=local_cond_preblur,
                    cond_random_upsample_prob=local_cond_random_upsample_prob,
                    lq_cond_size=lq_cond_size,
                    random_resize=random_resize,
                    load_text=load_text
                )
        else:  # bucket模式（默认）
            print(f"使用bucket模式 - 基于宽高比桶的智能裁剪")
            # 优先支持按步数控制 epoch 长度：steps_per_epoch / max_steps_epoch / epoch_steps
            steps_per_epoch = getattr(config.local_train, 'steps_per_epoch', None)
            if steps_per_epoch is None:
                steps_per_epoch = getattr(config.local_train, 'max_steps_epoch', None)
            if steps_per_epoch is None:
                steps_per_epoch = getattr(config.local_train, 'epoch_steps', None)

            adjusted_max_samples = max_samples  # 默认使用 max_samples
            try:
                world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
            except Exception:
                world_size = int(os.environ.get("WORLD_SIZE", "1"))
            global_batch = max(1, batch_size * world_size)

            if steps_per_epoch is not None:
                try:
                    steps_per_epoch = int(steps_per_epoch)
                except Exception:
                    steps_per_epoch = None

            if steps_per_epoch is not None and steps_per_epoch > 0:
                target_samples = steps_per_epoch * global_batch
                print(f"⚙️ bucket模式: 使用 steps_per_epoch={steps_per_epoch}，全局batch={global_batch} => 目标样本数={target_samples}（不足时将循环复用样本）")
                # 为了支持“步数优先”，这里不再裁剪到可用图像容量上限，改为后续通过 DataLoader 包装循环复用
                adjusted_max_samples = target_samples
            else:
                # 若未设置步数，则将 max_samples 按全局 batch 大小向下取整，确保只包含完整批次
                if adjusted_max_samples is not None and isinstance(adjusted_max_samples, int) and adjusted_max_samples > 0:
                    rounded = (adjusted_max_samples // global_batch) * global_batch
                    if rounded != adjusted_max_samples:
                        print(f"⚙️ bucket模式: max_samples {adjusted_max_samples} -> {rounded} (整除全局batch {global_batch})，避免最后一个不齐批次")
                    adjusted_max_samples = rounded
            base_loader = create_bucket_dataloader(
                image_folder_path=config.local_train.image_dir,
                base_size=config.get('base_size', 1024),
                multiple_of=multiple_of,   # 根据下采样倍率动态调整
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=True,
                downsample_factor=downsample_factor,
                upsample_factor=upsample_factor,
                shuffle=True,  # 训练模式：启用随机化
                max_samples=adjusted_max_samples,
                skip_cond_image=skip_cond_image,
                use_degraded_image=use_degraded_image,
                cond_downsample_resample=local_cond_down_resample,
                cond_upsample_resample=local_cond_up_resample,
                cond_preblur=local_cond_preblur,
                load_text=load_text,
                caption_json_key=caption_json_key,
                enable_sharpness=enable_sharpness,
            )
            # 当设置了 steps_per_epoch 时，用包装器确保每个 epoch 精确返回指定步数，数据不足时循环复用
            if steps_per_epoch is not None and steps_per_epoch > 0:
                class _RepeatOrTruncateDataLoader:
                    def __init__(self, base_loader, steps_per_epoch):
                        self._base = base_loader
                        self._steps = int(steps_per_epoch)
                    def __iter__(self):
                        yielded = 0
                        it = iter(self._base)
                        while yielded < self._steps:
                            try:
                                batch = next(it)
                            except StopIteration:
                                it = iter(self._base)
                                continue
                            yielded += 1
                            yield batch
                    def __len__(self):
                        return self._steps
                    def __getattr__(self, name):
                        return getattr(self._base, name)
                return _RepeatOrTruncateDataLoader(base_loader, steps_per_epoch)
            return base_loader
    else:
        # 使用S3数据
        # 从配置中读取参数
        dataloader_config = config.dataloader.train
        batch_size = dataloader_config.batch_size
        num_workers = dataloader_config.num_workers
        num_threads = dataloader_config.num_threads
        query_files = dataloader_config.query_files
        
        return Dataloader(
            intBatchsize=batch_size,
            intWorkers=num_workers,
            intThreads=num_threads,
            strQueryfile=query_files,
            funcGroup=functools.partial(piat.meta_groupaspects, {'intSize': base_size, 'intMultiple': multiple_of, 'strResize': 'preserve-area'}),
            funcStages=[
                functools.partial(piat.filter_image, {'strCondition': 'fltPhoto > 0.9', 'strCondition': 'fltAesthscore > 4.0', 'strCondition': 'fltGenai < 0.05'}),
                functools.partial(piat.image_load, {'strSource': '1024-pil-antialias'}),
                functools.partial(piat.image_resize_antialias, {'intSize': base_size}),
                functools.partial(piat.image_crop_smart, {'intSize': base_size}),
                functools.partial(attach_images, {'downsample_factor': downsample_factor, 'upsample_factor': upsample_factor, 'skip_cond_image': skip_cond_image, 'use_degraded_image': use_degraded_image, 'cond_downsample_resample': base_cond_down_resample, 'cond_upsample_resample': base_cond_up_resample, 'cond_preblur': base_cond_preblur}),
                load_internvl_caption,
                output_data,
                functools.partial(piat.output_text, {}),
            ],
            intSeed=random.randint(0, 1000000),
            collate_fn=collate_fn,
        )




def create_val_dataloader(config=None):
    """创建验证数据加载器（支持本地图像和S3数据）"""
    base_cond_down_mode, base_cond_up_mode = _resolve_cond_interpolation_modes(config)
    base_cond_down_resample = _resolve_pil_resample(base_cond_down_mode, Image.Resampling.BICUBIC)
    base_cond_up_resample = _resolve_pil_resample(base_cond_up_mode, base_cond_down_resample)

    if config is None:
        # 默认参数 - 使用S3数据
        downsample_factor = 2
        upsample_factor = 1
        skip_cond_image = False
        use_degraded_image = False
        multiple_of = 16 * downsample_factor if downsample_factor > 1 else 32
        base_cond_preblur = False
        batch_size = 4
        num_workers = 2
        num_threads = 4
        query_files = ['s3://sniklaus-clio-query/*/origin=sf&offensiveness=0&genairemoval=date&keywordblocklist=v1']
        
        # 验证默认路径：目标边长默认 512，可通过 config.base_size 覆盖
        base_size = _maybe_get(config, 'base_size', 512)
        return Dataloader(
            intBatchsize=batch_size,
            intWorkers=num_workers,
            intThreads=num_threads,
            strQueryfile=query_files,
            intSeed=0,
            funcGroup=functools.partial(piat.meta_groupaspects, {'intSize': base_size, 'intMultiple': multiple_of, 'strResize': 'preserve-area'}),
            funcStages=[
                functools.partial(piat.filter_image, {'strCondition': 'fltAesthscore > 5.0'}),
                functools.partial(piat.image_load, {'strSource': '1024-pil-antialias'}),
                functools.partial(piat.image_resize_antialias, {'intSize': base_size}),
                functools.partial(piat.image_crop_smart, {'intSize': base_size}),
                functools.partial(attach_images, {'downsample_factor': downsample_factor, 'upsample_factor': upsample_factor, 'skip_cond_image': skip_cond_image, 'use_degraded_image': use_degraded_image, 'cond_downsample_resample': base_cond_down_resample, 'cond_upsample_resample': base_cond_up_resample, 'cond_preblur': base_cond_preblur}),
                load_internvl_caption,
                functools.partial(piat.output_text, {}),
                output_data,
            ],
            collate_fn=collate_fn,
        )
    else:
        # 根据下采样倍率计算multiple_of（兼容无 experiment 的配置）
        experiment_cfg = getattr(config, 'experiment', None)
        downsample_factor = _maybe_get(experiment_cfg, "downsample_factor", 2)
        upsample_factor = _maybe_get(experiment_cfg, "upsample_factor", 1)
        # skip_cond_image = config.model.get("use_tensor_downsample", False)
        skip_cond_image = _maybe_get(experiment_cfg, "skip_cond_image", False)
        use_degraded_image = _maybe_get(experiment_cfg, "use_degraded_image", False)
        multiple_of = 16 * downsample_factor if downsample_factor > 1 else 32
    
    # 检查是否启用本地评估
    if hasattr(config, 'local_eval') and config.local_eval.enabled:
        print(f"使用本地图像评估，图像目录: {config.local_eval.image_dir}")
        print(f"下采样倍率: {downsample_factor}, multiple_of: {multiple_of}")
        
        local_eval_cfg = getattr(config, 'local_eval', None)
        local_cond_down_mode, local_cond_up_mode = _resolve_cond_interpolation_modes(config, local_eval_cfg)
        local_cond_down_resample = _resolve_pil_resample(local_cond_down_mode, base_cond_down_resample)
        local_cond_up_resample = _resolve_pil_resample(local_cond_up_mode, local_cond_down_resample)
        local_cond_preblur = _resolve_cond_preblur(config, local_eval_cfg)
        local_cond_random_upsample_prob = _resolve_cond_random_upsample_prob(config, local_eval_cfg)
        
        # 优先使用local_eval中的专用参数；缺省时尝试全局dataloader；仍缺省则使用安全默认值
        batch_size = getattr(config.local_eval, 'batch_size', None)
        if batch_size is None:
            batch_size = getattr(getattr(getattr(config, 'dataloader', None), 'val', None), 'batch_size', 4)
        num_workers = getattr(config.local_eval, 'num_workers', None)
        if num_workers is None:
            num_workers = getattr(getattr(getattr(config, 'dataloader', None), 'val', None), 'num_workers', 2)
        
        # 获取max_samples参数，用于overfitting
        max_samples = getattr(config.local_eval, 'max_samples', None)
        if max_samples is not None:
            print(f"🔒 Overfitting模式：限制使用前 {max_samples} 张图像")
        
        # 检查load_mode参数
        load_mode = getattr(config.local_eval, 'load_mode', 'bucket')
        
        # 检查是否启用文本加载
        load_text = getattr(config.local_eval, 'load_text', False)
        caption_json_key = _maybe_get(config.local_eval, 'caption_json_key', None)
        if caption_json_key is None and isinstance(local_eval_cfg, dict):
            caption_json_key = local_eval_cfg.get('caption_json_key')
        if not caption_json_key:
            caption_json_key = 'text'
        
        # 检查是否启用sharpness提示
        enable_sharpness = getattr(config.local_eval, 'enable_sharpness', False)
        
        if load_mode == 'simple':
            print(f"使用简单模式 - 直接random crop到固定尺寸")
            return create_local_simple_dataloader(
                image_folder_path=config.local_eval.image_dir,
                target_size=config.get('base_size', 512),
                batch_size=batch_size,
                num_workers=num_workers,
                downsample_factor=downsample_factor,
                upsample_factor=upsample_factor,
                max_samples=max_samples,
                shuffle=False,  # 验证集保持固定顺序
                skip_cond_image=skip_cond_image,
                use_degraded_image=use_degraded_image,
                cond_downsample_resample=local_cond_down_resample,
                cond_upsample_resample=local_cond_up_resample,
                cond_preblur=local_cond_preblur,
                load_text=load_text
            )
        elif load_mode == 'short':
            print(f"使用short模式 - 将最短边resize到base_size，然后center crop到正方形")
            return create_local_short_dataloader(
                image_folder_path=config.local_eval.image_dir,
                base_size=config.get('base_size', 1024),
                batch_size=batch_size,
                num_workers=num_workers,
                downsample_factor=downsample_factor,
                upsample_factor=upsample_factor,
                max_samples=max_samples,
                shuffle=False,  # 验证集保持固定顺序
                skip_cond_image=skip_cond_image,
                use_degraded_image=use_degraded_image,
                cond_downsample_resample=local_cond_down_resample,
                cond_upsample_resample=local_cond_up_resample,
                cond_preblur=local_cond_preblur,
                load_text=load_text
            )
        elif load_mode == 'long':
            print(f"使用long模式 - 将长边resize到base_size，然后基于宽高比桶的智能裁剪")
            # 获取lq_cond_size参数，优先级最高
            lq_cond_size = getattr(config.local_eval, 'lq_cond_size', 512)
            # 获取random_resize参数，测试时默认关闭
            random_resize = getattr(config.local_eval, 'random_resize', False)
            # 获取use_enhanced_buckets参数，默认使用增强桶系统
            use_enhanced_buckets = getattr(config.local_eval, 'use_enhanced_buckets', True)
            
            if use_enhanced_buckets:
                print(f"使用增强桶系统 - 确保lq_cond和最终图像比例完全一致")
                return create_enhanced_long_dataloader(
                    image_folder_path=config.local_eval.image_dir,
                    base_size=config.get('base_size', 1024),
                    multiple_of=16,   # long模式固定使用16的倍数
                    lq_cond_size=lq_cond_size,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=True,  # 验证时不丢弃最后一个批次
                    shuffle=False,  # 验证集保持固定顺序
                    max_samples=max_samples,
                    skip_cond_image=skip_cond_image,
                    use_degraded_image=use_degraded_image,
                    cond_downsample_resample=local_cond_down_resample,
                    cond_upsample_resample=local_cond_up_resample,
                    cond_preblur=local_cond_preblur,
                    cond_random_upsample_prob=local_cond_random_upsample_prob,
                    load_text=load_text
                )
            else:
                print(f"使用传统桶系统 - 向后兼容模式")
                return create_legacy_long_dataloader(
                    image_folder_path=config.local_eval.image_dir,
                    base_size=config.get('base_size', 1024),
                    multiple_of=multiple_of,  
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=True,  # 验证时不丢弃最后一个批次
                    downsample_factor=downsample_factor,
                    upsample_factor=upsample_factor,
                    shuffle=False,  # 验证集保持固定顺序
                    max_samples=max_samples,
                    skip_cond_image=skip_cond_image,
                    use_degraded_image=use_degraded_image,
                    cond_downsample_resample=local_cond_down_resample,
                    cond_upsample_resample=local_cond_up_resample,
                    cond_preblur=local_cond_preblur,
                    cond_random_upsample_prob=local_cond_random_upsample_prob,
                    lq_cond_size=lq_cond_size,
                    random_resize=random_resize,
                    load_text=load_text
                )
        else:  # bucket模式（默认）
            print(f"使用bucket模式 - 基于宽高比桶的智能裁剪")
            return create_bucket_dataloader(
                image_folder_path=config.local_eval.image_dir,
                base_size=config.get('base_size', 1024),
                multiple_of=multiple_of,   # 根据下采样倍率动态调整
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=True, 
                downsample_factor=downsample_factor,
                upsample_factor=upsample_factor,
                shuffle=False,  # 验证集保持固定顺序
                max_samples=max_samples,
                skip_cond_image=skip_cond_image,
                use_degraded_image=use_degraded_image,
                cond_downsample_resample=local_cond_down_resample,
                cond_upsample_resample=local_cond_up_resample,
                cond_preblur=local_cond_preblur,
                load_text=load_text,
                caption_json_key=caption_json_key,
                enable_sharpness=enable_sharpness,
            )
    else:
        # 使用S3数据
        # 从配置中读取参数
        dataloader_config = config.dataloader.val
        batch_size = dataloader_config.batch_size
        num_workers = dataloader_config.num_workers
        num_threads = dataloader_config.num_threads
        query_files = dataloader_config.query_files
        
        # 验证 S3 路径：目标边长默认 1024，可通过 config.base_size 覆盖
        base_size = _maybe_get(config, 'base_size', 1024)
        return Dataloader(
            intBatchsize=batch_size,
            intWorkers=num_workers,
            intThreads=num_threads,
            strQueryfile=query_files,
            intSeed=0,
            funcGroup=functools.partial(piat.meta_groupaspects, {'intSize': base_size, 'intMultiple': multiple_of, 'strResize': 'preserve-area'}),
            funcStages=[
                functools.partial(piat.filter_image, {'strCondition': 'fltPhoto > 0.9', 'strCondition': 'fltAesthscore > 5.0'}),
                functools.partial(piat.image_load, {'strSource': '1024-pil-antialias'}),
                functools.partial(piat.image_resize_antialias, {'intSize': base_size}),
                functools.partial(piat.image_crop_smart, {'intSize': base_size}),
                functools.partial(attach_images, {'downsample_factor': downsample_factor, 'upsample_factor': upsample_factor, 'skip_cond_image': skip_cond_image, 'use_degraded_image': use_degraded_image, 'cond_downsample_resample': base_cond_down_resample, 'cond_upsample_resample': base_cond_up_resample}),
                functools.partial(piat.text_load, {}),
                output_data,
                functools.partial(piat.output_text, {}),
            ],
            collate_fn=collate_fn,
        )


def create_local_simple_dataloader(image_folder_path, target_size=1024, batch_size=4, num_workers=4, 
                                 downsample_factor=2, upsample_factor=1, max_samples=None, shuffle=True, skip_cond_image=False, use_degraded_image=False, cond_downsample_resample=Image.Resampling.BICUBIC, cond_upsample_resample=None, cond_preblur=False, load_text=False):
    """
    创建一个简单的本地训练dataloader，直接从image_folder_path读取图像，
    进行random crop到固定尺寸，然后构造cond image
    
    Args:
        image_folder_path: 图像文件夹路径
        target_size: 目标图像尺寸（正方形）
        batch_size: 批次大小
        num_workers: 工作进程数
        downsample_factor: 下采样倍率
        upsample_factor: 上采样倍率
        max_samples: 最大图像数量（用于overfitting）
        shuffle: 是否随机打乱
        load_text: 是否加载文本数据（从对应的JSON文件）
    
    Returns:
        DataLoader: 配置好的数据加载器
    """
    import os
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    import random
    
    class SimpleLocalDataset(Dataset):
        def __init__(self, image_folder_path, target_size, downsample_factor, upsample_factor, max_samples, skip_cond_image=False, use_degraded_image=False, cond_downsample_resample=Image.Resampling.BICUBIC, cond_upsample_resample=None, cond_preblur=False, load_text=False):
            self.image_folder_path = image_folder_path
            self.target_size = target_size
            self.downsample_factor = downsample_factor
            self.upsample_factor = upsample_factor
            self.skip_cond_image = skip_cond_image
            self.use_degraded_image = use_degraded_image
            self.cond_downsample_resample = cond_downsample_resample
            self.cond_upsample_resample = cond_upsample_resample or cond_downsample_resample
            self.cond_preblur = cond_preblur
            self.load_text = load_text
            self.to_pil = transforms.ToPILImage()
            self.to_tensor = transforms.ToTensor()
            
            # 获取所有图像文件
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIFF'}
            self.image_files = []
            
            for root, dirs, files in os.walk(image_folder_path):
                for file in files:
                    if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                        file_path = os.path.join(root, file)
                        self.image_files.append(file_path)
            
            self.image_files.sort()
            
            # 如果指定了max_samples，则只使用前N张图像
            if max_samples is not None and max_samples > 0:
                self.image_files = self.image_files[:max_samples]
                print(f"🔒 Overfitting模式：限制使用前 {max_samples} 张图像")
            
            print(f"找到 {len(self.image_files)} 张图像")
            
            # 定义transforms：使用 RandomResizedCrop 提升对边界尺寸的鲁棒性
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    target_size,
                    scale=(0.8, 1.0),
                    ratio=(1.0, 1.0),
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
            ])
            
        def __len__(self):
            return len(self.image_files)
            
        def __getitem__(self, idx):
            try:
                # 加载图像
                image = Image.open(self.image_files[idx]).convert('RGB')
                
                # 如果图像小于目标尺寸，先resize
                if image.size[0] < self.target_size or image.size[1] < self.target_size:
                    # 计算需要的尺寸，保持宽高比（向上取整，避免 off-by-one 导致 RandomCrop 报错）
                    w, h = image.size
                    scale = max(self.target_size / w, self.target_size / h)
                    new_w = math.ceil(w * scale)
                    new_h = math.ceil(h * scale)
                    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # 应用transforms（包括random crop和to tensor）
                image_tensor = self.transform(image)
                cropped_pil = self.to_pil(image_tensor)
                
                # 生成图像键
                image_key = os.path.splitext(os.path.basename(self.image_files[idx]))[0]
                
                result = {
                    'image': image_tensor,
                    'strImagehash': image_key
                }
                
                # 加载文本数据（如果启用）
                if self.load_text:
                    try:
                        json_path = os.path.splitext(self.image_files[idx])[0] + '.json'
                        if os.path.exists(json_path):
                            with open(json_path, 'r', encoding='utf-8') as f:
                                text_data = json.load(f)
                                result['strText'] = text_data.get('text', '')
                        else:
                            result['strText'] = ''
                    except Exception as e:
                        print(f"警告: 加载文本文件时出错 {json_path}: {e}")
                        result['strText'] = ''
                
                # 如果跳过cond_image准备，返回None
                if self.skip_cond_image:
                    result['cond_image'] = None
                else:
                    # 创建cond image
                    cond_height = max(1, self.target_size // self.downsample_factor)
                    cond_width = max(1, self.target_size // self.downsample_factor)

                    cond_source = cropped_pil.filter(_COND_PREBLUR_FILTER) if self.cond_preblur else cropped_pil
                    cond_pil = cond_source.resize((cond_width, cond_height), resample=self.cond_downsample_resample)
                    cond_image_lq = torch.clamp(self.to_tensor(cond_pil), 0.0, 1.0)

                    if self.use_degraded_image:
                        # image 使用 低分 -> 上采样 回目标尺寸
                        degraded_pil = cond_pil.resize((self.target_size, self.target_size), resample=self.cond_upsample_resample)
                        result['image'] = torch.clamp(self.to_tensor(degraded_pil), 0.0, 1.0)
                        result['cond_image'] = cond_image_lq
                    else:
                        # 常规：image 为高分裁剪图；cond 可选上采样
                        cond_up_tensor = cond_image_lq
                        if self.upsample_factor > 1:
                            cond_up_h = max(1, cond_height * self.upsample_factor)
                            cond_up_w = max(1, cond_width * self.upsample_factor)
                            cond_up_pil = cond_pil.resize((cond_up_w, cond_up_h), resample=self.cond_upsample_resample)
                            cond_up_tensor = torch.clamp(self.to_tensor(cond_up_pil), 0.0, 1.0)
                        result['cond_image'] = cond_up_tensor
                
                return result
                
            except Exception as e:
                print(f"警告: 处理图像时出错 {self.image_files[idx]}: {e}")
                # 创建默认的黑色图像作为替代
                default_img = torch.zeros(3, self.target_size, self.target_size)
                default_cond_img = torch.zeros(3, self.target_size // self.downsample_factor, self.target_size // self.downsample_factor)
                default_key = f"error_{idx}"
                
                result = {
                    'image': default_img,
                    'cond_image': default_cond_img,
                    'strImagehash': default_key
                }
                
                # 如果启用文本加载，添加空文本
                if self.load_text:
                    result['strText'] = ''
                
                return result
    
    # 创建数据集
    dataset = SimpleLocalDataset(
        image_folder_path=image_folder_path,
        target_size=target_size,
        downsample_factor=downsample_factor,
        upsample_factor=upsample_factor,
        max_samples=max_samples,
        skip_cond_image=skip_cond_image,
        use_degraded_image=use_degraded_image,
        cond_downsample_resample=cond_downsample_resample,
        cond_upsample_resample=cond_upsample_resample,
        cond_preblur=cond_preblur,
        load_text=load_text
    )
    
    if len(dataset) == 0:
        raise ValueError(f"在 {image_folder_path} 中没有找到有效的图像文件")
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,  # 使用现有的collate_fn
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print(f"✅ 简单本地dataloader创建完成！")
    print(f"数据集大小: {len(dataset)}, 目标尺寸: {target_size}x{target_size}")
    print(f"下采样倍率: {downsample_factor}, 上采样倍率: {upsample_factor}")
    
    return dataloader


def create_local_short_dataloader(image_folder_path, base_size=1024, batch_size=4, num_workers=4, 
                                 downsample_factor=2, upsample_factor=1, max_samples=None, shuffle=True, skip_cond_image=False, use_degraded_image=False, cond_downsample_resample=Image.Resampling.BICUBIC, cond_upsample_resample=None, cond_preblur=False, load_text=False):
    """
    创建一个short模式的本地训练dataloader，将最短边resize到base_size，
    保持宽高比，然后进行center crop到正方形
    
    Args:
        image_folder_path: 图像文件夹路径
        base_size: 基础图像尺寸（最短边，最终正方形边长）
        batch_size: 批次大小
        num_workers: 工作进程数
        downsample_factor: 下采样倍率
        upsample_factor: 上采样倍率
        max_samples: 最大图像数量（用于overfitting）
        shuffle: 是否随机打乱
        skip_cond_image: 是否跳过cond_image准备
        use_degraded_image: 是否使用降质图像
        load_text: 是否加载文本数据（从对应的JSON文件）
    
    Returns:
        DataLoader: 配置好的数据加载器
    """
    import os
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    import random
    
    class ShortLocalDataset(Dataset):
        def __init__(self, image_folder_path, base_size, downsample_factor, upsample_factor, max_samples, skip_cond_image=False, use_degraded_image=False, cond_downsample_resample=Image.Resampling.BICUBIC, cond_upsample_resample=None, cond_preblur=False, load_text=False):
            self.image_folder_path = image_folder_path
            self.base_size = base_size
            self.downsample_factor = downsample_factor
            self.upsample_factor = upsample_factor
            self.skip_cond_image = skip_cond_image
            self.use_degraded_image = use_degraded_image
            self.cond_downsample_resample = cond_downsample_resample
            self.cond_upsample_resample = cond_upsample_resample or cond_downsample_resample
            self.cond_preblur = cond_preblur
            self.load_text = load_text
            self.to_tensor = transforms.ToTensor()
            
            # 获取所有图像文件
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIFF'}
            self.image_files = []
            
            for root, dirs, files in os.walk(image_folder_path):
                for file in files:
                    if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                        file_path = os.path.join(root, file)
                        self.image_files.append(file_path)
            
            self.image_files.sort()
            
            # 如果指定了max_samples，则只使用前N张图像
            if max_samples is not None and max_samples > 0:
                self.image_files = self.image_files[:max_samples]
                print(f"🔒 Overfitting模式：限制使用前 {max_samples} 张图像")
            
            print(f"找到 {len(self.image_files)} 张图像")
            
        def __len__(self):
            return len(self.image_files)
            
        def __getitem__(self, idx):
            try:
                # 加载图像
                image = Image.open(self.image_files[idx]).convert('RGB')
                original_w, original_h = image.size
                
                # 计算resize后的尺寸，保持宽高比，最短边为base_size
                if original_w > original_h:
                    # 高度更短
                    new_h = self.base_size
                    new_w = int(original_w * self.base_size / original_h)
                else:
                    # 宽度更短
                    new_w = self.base_size
                    new_h = int(original_h * self.base_size / original_w)
                
                # 先resize到计算出的尺寸
                image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # 然后center crop到正方形
                min_size = min(new_w, new_h)
                left = (new_w - min_size) // 2
                top = (new_h - min_size) // 2
                right = left + min_size
                bottom = top + min_size
                
                image = image.crop((left, top, right, bottom))
                
                # 转换为tensor
                image_tensor = self.to_tensor(image)
                
                # 生成图像键
                image_key = os.path.splitext(os.path.basename(self.image_files[idx]))[0]
                
                result = {
                    'image': image_tensor,
                    'strImagehash': image_key
                }
                
                # 加载文本数据（如果启用）
                if self.load_text:
                    try:
                        json_path = os.path.splitext(self.image_files[idx])[0] + '.json'
                        if os.path.exists(json_path):
                            with open(json_path, 'r', encoding='utf-8') as f:
                                text_data = json.load(f)
                                result['strText'] = text_data.get('text', '')
                        else:
                            result['strText'] = ''
                    except Exception as e:
                        print(f"警告: 加载文本文件时出错 {json_path}: {e}")
                        result['strText'] = ''
                
                # 如果跳过cond_image准备，返回None
                if self.skip_cond_image:
                    result['cond_image'] = None
                else:
                    # 创建cond image
                    cond_height = max(1, min_size // self.downsample_factor)
                    cond_width = max(1, min_size // self.downsample_factor)

                    cond_source = image.filter(_COND_PREBLUR_FILTER) if self.cond_preblur else image
                    cond_pil = cond_source.resize((cond_width, cond_height), resample=self.cond_downsample_resample)
                    cond_image_lq = torch.clamp(self.to_tensor(cond_pil), 0.0, 1.0)

                    if self.use_degraded_image:
                        # image 使用 低分 -> 上采样 回目标尺寸
                        degraded_pil = cond_pil.resize((min_size, min_size), resample=self.cond_upsample_resample)
                        result['image'] = torch.clamp(self.to_tensor(degraded_pil), 0.0, 1.0)
                        result['cond_image'] = cond_image_lq
                    else:
                        # 常规：image 为高分裁剪图；cond 可选上采样
                        cond_up_tensor = cond_image_lq
                        if self.upsample_factor > 1:
                            cond_up_h = max(1, cond_height * self.upsample_factor)
                            cond_up_w = max(1, cond_width * self.upsample_factor)
                            cond_up_pil = cond_pil.resize((cond_up_w, cond_up_h), resample=self.cond_upsample_resample)
                            cond_up_tensor = torch.clamp(self.to_tensor(cond_up_pil), 0.0, 1.0)
                        result['cond_image'] = cond_up_tensor
                
                return result
                
            except Exception as e:
                print(f"警告: 处理图像时出错 {self.image_files[idx]}: {e}")
                # 创建默认的黑色图像作为替代
                default_img = torch.zeros(3, self.base_size, self.base_size)
                default_cond_img = torch.zeros(3, self.base_size // self.downsample_factor, self.base_size // self.downsample_factor)
                default_key = f"error_{idx}"
                
                result = {
                    'image': default_img,
                    'cond_image': default_cond_img,
                    'strImagehash': default_key
                }
                
                # 如果启用文本加载，添加空文本
                if self.load_text:
                    result['strText'] = ''
                
                return result
    
    # 创建数据集
    dataset = ShortLocalDataset(
        image_folder_path=image_folder_path,
        base_size=base_size,
        downsample_factor=downsample_factor,
        upsample_factor=upsample_factor,
        max_samples=max_samples,
        skip_cond_image=skip_cond_image,
        use_degraded_image=use_degraded_image,
        cond_downsample_resample=cond_downsample_resample,
        cond_upsample_resample=cond_upsample_resample,
        cond_preblur=cond_preblur,
        load_text=load_text
    )
    
    if len(dataset) == 0:
        raise ValueError(f"在 {image_folder_path} 中没有找到有效的图像文件")
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,  # 使用现有的collate_fn
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print(f"✅ Short模式本地dataloader创建完成！")
    print(f"数据集大小: {len(dataset)}, 基础尺寸: {base_size}x{base_size}")
    print(f"下采样倍率: {downsample_factor}, 上采样倍率: {upsample_factor}")
    
    return dataloader


