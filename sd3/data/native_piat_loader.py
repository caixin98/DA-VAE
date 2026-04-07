import torch
import functools
import piat  # 假设 piat 库已安装
import json
import os
from PIL import Image
import sys
import random
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import hflip

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nit.data.piat_core import Dataloader

# ==============================================================================


def resize_arr(pil_image, height, width):
    pil_image = pil_image.resize((width, height), resample=Image.Resampling.BICUBIC)
    return pil_image

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

# --- Degradation function for cond_image ---
def degrade_image(pil_image, mode='4x'):
    """
    对输入PIL图片进行降质处理。
    支持不同的mode：
      - 'gaussian': 高斯模糊+随机hflip（原有逻辑）
      - '4x': 先4倍下采样再上采样回原始大小
    """
    from PIL import ImageFilter
    import numpy as np
    if mode == '4x':
        # 先4倍下采样再上采样
        w, h = pil_image.size
        down_w, down_h = max(1, w // 4), max(1, h // 4)
        img = pil_image.resize((down_w, down_h), resample=Image.Resampling.BICUBIC)
        img = img.resize((w, h), resample=Image.Resampling.BICUBIC)
        return img
    else:
        # 默认高斯模糊+随机hflip
        img = pil_image.filter(ImageFilter.GaussianBlur(radius=random.randint(1, 3)))
        if random.random() > 0.5:
            img = hflip(img)
        return img

# --- Stage: attach_images ---
def attach_images(objScratch, objOutput):
    if objOutput is None:
        return 'initialized'
    if 'npyImage' in objScratch:
        pil_img = Image.fromarray(objScratch['npyImage']).convert("RGB")
        objScratch['image'] = pil_img
        objScratch['cond_image'] = degrade_image(pil_img)
    # 其他情况直接return None

# --- Stage: attach_repa_images ---
def attach_repa_images(objScratch, objOutput):
    if objOutput is None:
        return 'initialized'
    if 'image' in objScratch and 'cond_image' in objScratch:
        h = objScratch.get('native_height', 256)
        w = objScratch.get('native_width', 256)
        # 这里和ImprovedPackedDataset保持一致
        preprocess = functools.partial(resize_arr, height=h, width=w)
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: preprocess(pil_image=pil_image)),
            transforms.Lambda(lambda pil_image: (pil_image, hflip(pil_image))),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        objScratch['repa_image'] = transform(objScratch['image'])
        objScratch['repa_cond_image'] = transform(objScratch['cond_image'])

def output_data(objScratch, objOutput):
    if objOutput is None:
        return 'initialized'
    if 'image' in objScratch:
        objOutput['image'] = objScratch['image']
    if 'cond_image' in objScratch:
        objOutput['cond_image'] = objScratch['cond_image']
    if 'repa_image' in objScratch:
        objOutput['repa_image'] = objScratch['repa_image']
    if 'repa_cond_image' in objScratch:
        objOutput['repa_cond_image'] = objScratch['repa_cond_image']
    if 'native_height' in objScratch:
        objOutput['native_height'] = objScratch['native_height']
    if 'native_width' in objScratch:
        objOutput['native_width'] = objScratch['native_width']

    # 其他情况直接return None

# --- Intercept 函数 (核心打包逻辑) ---
def greedy_token_packer(objIterator, target_token_count, patch_size):
    packed_batch = []
    tokens_in_batch = 0
    for objSample in objIterator():
        h = objSample.get('native_height', 0)
        w = objSample.get('native_width', 0)
        image_tokens = (h // patch_size) * (w // patch_size) if h > 0 and w > 0 else 0
        objSample['num_tokens'] = image_tokens
        # print("h, w:", h, w, "image_tokens:", image_tokens)
        if tokens_in_batch > 0 and (tokens_in_batch + image_tokens) > target_token_count:
            yield packed_batch
            packed_batch = [objSample]
            tokens_in_batch = image_tokens
        else:
            packed_batch.append(objSample)
            tokens_in_batch += image_tokens
    if packed_batch:
        yield packed_batch

# --- packed_i2i_collate_fn ---
def packed_i2i_collate_fn(batch):
    # batch: list of list of dict
    batch = [item for sublist in batch for item in sublist]
    image = []
    cond_image = []
    repa_image = []
    repa_cond_image = []
    for data in batch:
        image.append(transforms.ToTensor()(data['image']))
        cond_image.append(transforms.ToTensor()(data['cond_image']))
        repa_image.append(data['repa_image'])
        repa_cond_image.append(data['repa_cond_image'])
    return dict(image=image, cond_image=cond_image, repa_image=repa_image, repa_cond_image=repa_cond_image)

# 一个新的、更简洁的stage，它合并了加载和尺寸获取
def attach_size(objScratch, objOutput):
    if objOutput is None:
        return 'initialized'
    if 'npyImage' in objScratch:
        h, w = objScratch['npyImage'].shape[:2]
        objScratch['native_height'] = h
        objScratch['native_width'] = w

# ==============================================================================
# 步骤 3: 主函数 - 配置并运行 Dataloader
# ==============================================================================
if __name__ == '__main__':
    CONFIG = {
        'query_file': [
            's3://sniklaus-clio-query/*/origin=si',
        ],
        'batch_chunk_size': 1,
        'target_token_count': 216000,
        'patch_size': 16,
        'workers': 2,
        'threads': 4
    }

    training_stages = [
        functools.partial(piat.filter_image, {'strCondition': 'fltPhoto > 0.9', 'strCondition': 'fltAesthscore > 5.0'}),
        functools.partial(piat.image_load, {'strSource': '1024-pil-antialias'}),
        attach_size,
        # functools.partial(attach_images, {'degrade_mode': '4x'}),
        functools.partial(attach_images),
        attach_repa_images,
        output_data,
    ]

    dataloader = Dataloader(
        intBatchsize=CONFIG['batch_chunk_size'],
        intWorkers=CONFIG['workers'],
        intThreads=CONFIG['threads'],
        strQueryfile=CONFIG['query_file'],
        collate_fn=packed_i2i_collate_fn,
        funcStages=training_stages,
        funcIntercept=functools.partial(
            greedy_token_packer,
            target_token_count=CONFIG['target_token_count'],
            patch_size=CONFIG['patch_size']
        ),
    )

    print("开始从Dataloader中迭代打包好的批次...")
    save_dir = "tmp_vis"
    os.makedirs(save_dir, exist_ok=True)

    for i, packed_batch in enumerate(dataloader):
        if i >= 20:
            break
        print("packed_batch:", {k: type(v) for k, v in packed_batch.items()})
        print("packed_batch['image']:", len(packed_batch['image']))
        for j, image in enumerate(packed_batch['image']):
            # 如果image是Tensor，先转回PIL
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
            # 保存图片
            image.save(os.path.join(save_dir, f"batch{i}_img{j}.png"))
            print("image.shape:", image.size if hasattr(image, 'size') else image.shape)
   


