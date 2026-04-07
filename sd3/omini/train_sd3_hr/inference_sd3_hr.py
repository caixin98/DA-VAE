#!/usr/bin/env python3
"""
SD3 HR 推理脚本
支持加载训练好的HR模型进行推理
"""

import os
import argparse
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

import torch
import yaml
from PIL import Image

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from omini.train_sd3_srtitok.train_sd3_srtitok import get_config
from omini.train_sd3_hr.trainer import SD3HRModel


class SD3HRInference:
    """SD3 HR 推理类"""
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: Optional[str] = None,
        lora_checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        self.device = device
        self.dtype = getattr(torch, dtype)
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化模型
        self.model = self._init_model()
        
        # 加载检查点
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path, lora_checkpoint_path)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _init_model(self) -> SD3HRModel:
        """初始化SD3 HR模型"""
        print("[Inference] 初始化SD3 HR模型...")
        
        # 获取配置参数
        patch_embed_cfg = self.config.get("hr_patch_embed", {})
        patch_weights = self.config.get("hr_patch_weights")
        model_config = self.config.get("model", {})
        lora_paths = self.config.get("lora_paths", {})
        lora_config = self.config.get("train", {}).get("lora_config")
        
        # 创建模型
        model = SD3HRModel(
            sd3_pipe_id=self.config["sd3_pipe_id"],
            patch_embed_cfg=patch_embed_cfg,
            patch_embed_weights_path=patch_weights,
            lora_paths=lora_paths,
            lora_config=lora_config,
            device=self.device,
            dtype=self.dtype,
            model_config=model_config,
        )
        
        # 设置为评估模式
        model.eval()
        
        print("[Inference] 模型初始化完成")
        return model
    
    def _load_checkpoint(self, checkpoint_path: str, lora_checkpoint_path: Optional[str] = None):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"[Inference] 加载检查点: {checkpoint_path}")
        step = self.model.load_checkpoint(checkpoint_path, lora_checkpoint_path)
        print(f"[Inference] 成功加载检查点，步数: {step}")
    
    def generate(
        self,
        prompts: List[str],
        output_dir: str = "./inference_output",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        生成图像
        
        Args:
            prompts: 提示词列表
            output_dir: 输出目录
            num_inference_steps: 推理步数
            guidance_scale: 引导尺度
            height: 图像高度
            width: 图像宽度
            seed: 随机种子
        
        Returns:
            生成的图像列表
        """
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # 获取推理参数
        model_config = self.config.get("model", {})
        num_inference_steps = num_inference_steps or model_config.get("num_inference_steps", 28)
        guidance_scale = guidance_scale or model_config.get("guidance_scale", 3.5)
        height = height or model_config.get("sample_size", 1024)
        width = width or model_config.get("sample_size", 1024)
        # default negative prompt
        negative_prompt = model_config.get(
            "negative_prompt",
            " deformed, low quality & bad aesthetics & poor aesthetics ",
        )
        
        print(f"[Inference] 开始生成图像...")
        print(f"[Inference] 推理参数: steps={num_inference_steps}, guidance_scale={guidance_scale}, size={height}x{width}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        all_images = []
        
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                print(f"[Inference] 生成图像 {i+1}/{len(prompts)}: {prompt[:50]}...")
                
                try:
                    # 生成图像
                    # self.model.sd3_pipe.scheduler.config.use_dynamic_shifting = True
                    latents = self.model.sd3_pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        output_type="latent",
                    ).images
                    
                    images = self.model.sd3_pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        latents=latents,
                    ).images
                    
                    if images:
                        img = images[0]
                        all_images.append(img)
                        
                        # 保存图像
                        timestamp = int(time.time())
                        safe_prompt = self._sanitize_filename(prompt)
                        filename = f"inference_{timestamp}_{i:03d}_{safe_prompt}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        img.save(filepath)
                        print(f"[Inference] 保存图像: {filepath}")
                    else:
                        print(f"[Inference] 警告: 未生成图像 for prompt {i+1}")
                        
                except Exception as e:
                    print(f"[Inference] 错误: 生成图像失败 for prompt {i+1}: {e}")
                    continue
        
        print(f"[Inference] 完成! 生成了 {len(all_images)} 张图像")
        return all_images
    
    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """清理文件名"""
        # 取前几个词作为文件名
        words = text.split()[:5]
        filename = "_".join(words)
        
        # 替换不安全的字符
        unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in unsafe_chars:
            filename = filename.replace(char, '_')
        
        # 限制长度
        if len(filename) > max_length:
            filename = filename[:max_length]
        
        return filename


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SD3 HR 推理脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, help="检查点路径")
    parser.add_argument("--lora-checkpoint", type=str, help="LoRA检查点路径")
    parser.add_argument("--prompts", type=str, nargs="+", help="提示词列表")
    parser.add_argument("--prompt-file", type=str, help="提示词文件路径（每行一个提示词）")
    parser.add_argument("--output-dir", type=str, default="./inference_output", help="输出目录")
    parser.add_argument("--num-inference-steps", type=int, help="推理步数")
    parser.add_argument("--guidance-scale", type=float, help="引导尺度")
    parser.add_argument("--height", type=int, help="图像高度")
    parser.add_argument("--width", type=int, help="图像宽度")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
    
    args = parser.parse_args()
    
    # 获取提示词
    prompts = []
    if args.prompts:
        prompts.extend(args.prompts)
    if args.prompt_file:
        if not os.path.exists(args.prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            file_prompts = [line.strip() for line in f if line.strip()]
            prompts.extend(file_prompts)
    
    if not prompts:
        # 使用默认提示词
        prompts = [
            "A cat sitting on a windowsill, high detail photo",
            "A futuristic city at night in heavy rain, neon lights, 4k",
            "A scenic mountain lake at sunrise, ultra-detailed landscape",
            "Portrait of an astronaut in a forest, cinematic lighting, 85mm"
        ]
        print("[Inference] 使用默认提示词")
    
    print(f"[Inference] 将生成 {len(prompts)} 张图像")
    
    # 初始化推理器
    inference = SD3HRInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        lora_checkpoint_path=args.lora_checkpoint,
        device=args.device,
        dtype=args.dtype,
    )
    
    # 生成图像
    images = inference.generate(
        prompts=prompts,
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    
    print(f"[Inference] 推理完成! 生成了 {len(images)} 张图像")


if __name__ == "__main__":
    main()
