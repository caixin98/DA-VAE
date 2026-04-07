""" This file contains some utils functions for visualization.

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

import torch
import torchvision.transforms.functional as F
import torchvision.utils as vutils
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont

def make_viz_from_samples(
    original_images,
    reconstructed_images,
    lq_images=None
):
    """Generates visualization images from original images and reconstructed images.

    Args:
        original_images: A torch.Tensor, original images.
        reconstructed_images: A torch.Tensor, reconstructed images.
        lq_images: A torch.Tensor, low quality (condition) images. Optional.

    Returns:
        A tuple containing two lists - images_for_saving and images_for_logging.
    """
    reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
    reconstructed_images = reconstructed_images * 255.0
    reconstructed_images = reconstructed_images.cpu()
    
    original_images = torch.clamp(original_images, 0.0, 1.0)
    original_images *= 255.0
    original_images = original_images.cpu()

    # 准备要堆叠的图像列表
    to_stack = [original_images, reconstructed_images]
    
    # 如果有LQ图像，添加到可视化中
    if lq_images is not None:
        lq_images = lq_images.cpu()
        lq_images = torch.clamp(lq_images, 0.0, 1.0)
        # 将LQ图像上采样到与原始图像相同的尺寸
        if lq_images.shape[-2:] != original_images.shape[-2:]:
            import torch.nn.functional as F_interpolate
            # 保存原始数据类型
            original_dtype = lq_images.dtype
            # 临时转换为 float32 进行插值
            lq_images_float = lq_images.float()
            lq_images = F_interpolate.interpolate(
                lq_images_float, 
                size=original_images.shape[-2:], 
                mode='bicubic', 
                align_corners=False,
                antialias=True
            )
            # 转换回原始精度
            lq_images = lq_images.to(original_dtype)
        lq_images = torch.clamp(lq_images, 0.0, 1.0)
        lq_images = lq_images * 255.0
        
        to_stack.insert(1, lq_images)  # 在原始图像和重建图像之间插入LQ图像
    
    # 计算差异图像（原始图像和重建图像之间的差异）
    diff_img = torch.abs(original_images - reconstructed_images)
    to_stack.append(diff_img)

    # 根据是否有LQ图像调整布局
    if lq_images is not None:
        # 有LQ图像时：原始图像 | LQ图像 | 重建图像 | 差异图像
        images_for_logging = rearrange(
                torch.stack(to_stack),
                "(l1 l2) b c h w -> b c (l1 h) (l2 w)",
                l1=2).byte()
    else:
        # 没有LQ图像时：原始图像 | 重建图像 | 差异图像
        images_for_logging = rearrange(
                torch.stack(to_stack),
                "(l1 l2) b c h w -> b c (l1 h) (l2 w)",
                l1=1).byte()
    
    images_for_saving = [F.to_pil_image(image) for image in images_for_logging]

    return images_for_saving, images_for_logging


def make_viz_from_samples_generation(
    generated_images,
):
    generated = torch.clamp(generated_images, 0.0, 1.0) * 255.0
    images_for_logging = rearrange(
        generated, 
        "(l1 l2) c h w -> c (l1 h) (l2 w)",
        l1=2)

    images_for_logging = images_for_logging.cpu().byte()
    images_for_saving = F.to_pil_image(images_for_logging)

    return images_for_saving, images_for_logging


def make_viz_from_samples_t2i_generation(
    generated_images,
    captions,
):
    generated = torch.clamp(generated_images, 0.0, 1.0) * 255.0
    images_for_logging = rearrange(
        generated, 
        "(l1 l2) c h w -> c (l1 h) (l2 w)",
        l1=2)

    images_for_logging = images_for_logging.cpu().byte()
    images_for_saving = F.to_pil_image(images_for_logging)

    # Create a new image with space for captions
    width, height = images_for_saving.size
    caption_height = 20 * len(captions) + 10
    new_height = height + caption_height
    new_image = Image.new("RGB", (width, new_height), "black")
    new_image.paste(images_for_saving, (0, 0))

    # Adding captions below the image
    draw = ImageDraw.Draw(new_image)
    font = ImageFont.load_default()

    for i, caption in enumerate(captions):
        draw.text((10, height + 10 + i * 20), caption, fill="white", font=font)

    return new_image, images_for_logging


def save_image_grid(
    images: torch.Tensor,
    save_path,
    nrow: int = 8,
    normalize: bool = True,
    value_range=(0, 1)
):
    """将一组图像保存为网格图。

    Args:
        images: 形状为 [N, C, H, W] 或 [C, H, W] 的张量，数值范围通常为 [0, 1] 或 [0, 255]
        save_path: 保存路径（str 或 Path）
        nrow: 每行的图像数量
        normalize: 是否归一化到 [0, 1]
        value_range: 归一化时使用的 (min, max)
    """
    if images is None:
        raise ValueError("images 不能为空")

    # 保证为 4D 张量
    if images.dim() == 3:
        images = images.unsqueeze(0)
    elif images.dim() != 4:
        raise ValueError(f"images 维度应为 3 或 4，但收到 {images.dim()}")

    # 将数据移动到 CPU 并转换为浮点以便处理
    images_cpu = images.detach().to("cpu")

    # 使用 torchvision.utils.make_grid 生成网格
    grid = vutils.make_grid(
        images_cpu,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
        pad_value=1.0
    )

    # 保存为图片
    pil_img = F.to_pil_image(grid.clamp(0, 1))
    pil_img.save(str(save_path))