#!/usr/bin/env python3
"""
SD3 DAVAE 推理脚本
支持加载训练好的 Tokenizer/DA-VAE 模型进行推理
"""

import os
import argparse
import time
import math
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import re

import torch
import yaml
from PIL import Image

import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from omini.train_sd3_davae.tokenizer_trainer import SD3TokenizerModel
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline,
)


class SD3TokenizerInference:
    """SD3 DAVAE/Tokenizer 推理类"""
    
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

        self.project_root = project_root.resolve()
        self.config_path = Path(config_path).expanduser()
        if not self.config_path.is_absolute():
            self.config_path = (self.project_root / self.config_path).resolve()
        self.config_dir = self.config_path.parent

        # 加载配置
        self.config = self._load_config(str(self.config_path))

        # 解析默认的checkpoint路径
        self.run_dir: Optional[Path] = None
        self.checkpoint_path, self.lora_checkpoint_path = self._resolve_checkpoint_paths(
            checkpoint_path, lora_checkpoint_path
        )
        
        # 初始化模型
        self.model = self._init_model()
        
        # 加载检查点
        if self.checkpoint_path:
            print(f"[Inference] 使用 checkpoint: {self.checkpoint_path}")
        else:
            print("[Inference] 未提供 checkpoint，将使用基础模型参数")

        if self.lora_checkpoint_path:
            print(f"[Inference] 使用 LoRA checkpoint: {self.lora_checkpoint_path}")

        if self.checkpoint_path:
            self._load_checkpoint(self.checkpoint_path, self.lora_checkpoint_path)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _init_model(self) -> SD3TokenizerModel:
        """初始化SD3 Tokenizer/DAVAE模型"""
        print("[Inference] 初始化SD3 DAVAE模型...")
        
        # 获取配置参数（优先使用 tokenizer_* 配置项）
        patch_embed_cfg = (
            self.config.get("tokenizer_patch_embed")
            or self.config.get("hr_patch_embed", {})
            or {}
        )
        patch_weights = self.config.get("tokenizer_patch_weights") or self.config.get("hr_patch_weights")
        model_config = self.config.get("model", {})
        lora_paths = self.config.get("lora_paths", {})
        lora_config = self.config.get("train", {}).get("lora_config")
        
        # 创建模型
        model = SD3TokenizerModel(
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
        self._baseline_pipe: Optional[StableDiffusion3Pipeline] = None
        return model

    def _resolve_checkpoint_paths(
        self,
        checkpoint_path: Optional[str],
        lora_checkpoint_path: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        train_cfg = self.config.get("train", {}) or {}
        self.run_dir = self._resolve_run_dir(train_cfg)

        ckpt_request = checkpoint_path.strip().lower() if isinstance(checkpoint_path, str) else None
        lora_request = lora_checkpoint_path.strip().lower() if isinstance(lora_checkpoint_path, str) else None

        resolved_ckpt = None
        resolved_lora = None

        if ckpt_request not in {"", None} and ckpt_request not in {"auto", "latest"}:
            resolved_ckpt = self._normalize_path(checkpoint_path)
        if lora_request not in {"", None} and lora_request not in {"auto", "latest"}:
            resolved_lora = self._normalize_path(lora_checkpoint_path)

        if resolved_ckpt is None:
            candidate = train_cfg.get("resume_from_checkpoint") or train_cfg.get("checkpoint_path")
            if candidate:
                resolved_ckpt = self._normalize_existing(candidate)
            if resolved_ckpt is None and self.run_dir is not None:
                ckpt_dir = self.run_dir / "tokenizer_checkpoints"
                resolved_ckpt = self._find_latest_checkpoint(ckpt_dir)

        if resolved_lora is None:
            candidate = train_cfg.get("resume_from_lora_checkpoint") or train_cfg.get("lora_checkpoint_path")
            if candidate:
                resolved_lora = self._normalize_existing(candidate)
            if resolved_lora is None and self.run_dir is not None:
                lora_dir = self.run_dir / "lora_weights"
                resolved_lora = self._find_latest_lora(lora_dir)

        return resolved_ckpt, resolved_lora

    def _normalize_path(self, path_str: Optional[str]) -> Optional[str]:
        if not path_str:
            return None
        path_obj = Path(path_str).expanduser()
        if not path_obj.is_absolute():
            cfg_candidate = (self.config_dir / path_obj)
            if cfg_candidate.exists():
                path_obj = cfg_candidate
            else:
                path_obj = (self.project_root / path_obj)
        try:
            return str(path_obj.resolve())
        except FileNotFoundError:
            return str(path_obj)

    def _normalize_existing(self, path_str: str) -> Optional[str]:
        normalized = self._normalize_path(path_str)
        if normalized and os.path.exists(normalized):
            return normalized
        return None

    def _resolve_run_dir(self, train_cfg: Dict[str, Any]) -> Optional[Path]:
        run_name = train_cfg.get("run_name")
        if not run_name:
            return None
        save_path = train_cfg.get("save_path", "runs")
        save_path_obj = Path(save_path).expanduser()
        if not save_path_obj.is_absolute():
            cfg_candidate = (self.config_dir / save_path_obj)
            if cfg_candidate.exists():
                save_path_obj = cfg_candidate
            else:
                save_path_obj = (self.project_root / save_path_obj)
        run_dir = save_path_obj / run_name
        if run_dir.exists():
            try:
                return run_dir.resolve()
            except FileNotFoundError:
                return run_dir
        return None

    def _find_latest_checkpoint(self, directory: Path) -> Optional[str]:
        if not directory or not directory.exists():
            return None
        pattern = re.compile(r"step_(\d+)\.pt$")
        candidates = []
        for path in directory.glob("step_*.pt"):
            match = pattern.search(path.name)
            if match:
                candidates.append((int(match.group(1)), path))
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            return str(candidates[0][1].resolve())
        latest = directory / "latest.pt"
        if latest.exists():
            try:
                return str(latest.resolve())
            except FileNotFoundError:
                return str(latest)
        return None

    def _find_latest_lora(self, directory: Path) -> Optional[str]:
        if not directory or not directory.exists():
            return None
        dir_pattern = re.compile(r"lora_step_(\d+)")
        candidates = []
        for path in directory.iterdir():
            if path.is_dir():
                match = dir_pattern.search(path.name)
                if match:
                    candidates.append((int(match.group(1)), path))
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            return str(candidates[0][1].resolve())
        latest_dir = directory / "latest"
        if latest_dir.exists():
            try:
                return str(latest_dir.resolve())
            except FileNotFoundError:
                return str(latest_dir)
        weight_files = sorted(directory.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
        if weight_files:
            return str(weight_files[0].resolve())
        return None

    def _extract_scheduler_shift_value(self, scheduler) -> float:
        if scheduler is None:
            return 0.0

        attr_candidates = ["shift", "_shift", "shift_value", "shift_scalar", "shift_offset"]
        for attr in attr_candidates:
            value = getattr(scheduler, attr, None)
            if isinstance(value, (int, float)) and math.isfinite(value):
                return float(value)

        config = getattr(scheduler, "config", None)
        if config is not None:
            if hasattr(config, "get"):
                for key in attr_candidates:
                    try:
                        value = config.get(key)
                    except Exception:
                        value = None
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        return float(value)
            else:
                for key in attr_candidates:
                    value = getattr(config, key, None)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        return float(value)

        return 0.0

    def _format_shift_suffix(self, shift_value: float) -> str:
        try:
            if not math.isfinite(shift_value):
                return "_shift0"
            rounded = round(shift_value)
            if abs(shift_value - rounded) < 1e-6:
                shift_str = str(int(rounded))
            else:
                shift_str = f"{shift_value:.4f}".rstrip('0').rstrip('.')
            if not shift_str:
                shift_str = "0"
            return f"_shift{shift_str}"
        except Exception:
            return "_shift0"

    def _format_float_tag(self, prefix: str, value: float) -> str:
        try:
            if not math.isfinite(value):
                formatted = "0"
            else:
                rounded = round(value, 4)
                if abs(rounded - int(rounded)) < 1e-6:
                    formatted = str(int(rounded))
                else:
                    formatted = f"{rounded:.4f}".rstrip('0').rstrip('.')
                if not formatted:
                    formatted = "0"
            return f"_{prefix}{formatted}"
        except Exception:
            return f"_{prefix}0"
    
    def _load_checkpoint(self, checkpoint_path: str, lora_checkpoint_path: Optional[str] = None):
        """加载检查点（优先识别Lightning保存的.ckpt；若回退到自定义格式可选加载LoRA）"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"[Inference] 加载检查点: {checkpoint_path}")
        try:
            ckpt_obj = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            ckpt_obj = None
            print(f"[Inference] 警告: 无法预读检查点内容用于类型判断，将尝试按自定义格式加载 ({e})")

        # 如果是Lightning保存的ckpt（包含state_dict）则使用Lightning的load_from_checkpoint
        if isinstance(ckpt_obj, dict) and ("state_dict" in ckpt_obj):
            print("[Inference] 检测到Lightning格式ckpt，使用Lightning加载...")
            try:
                loaded = SD3TokenizerModel.load_from_checkpoint(
                    checkpoint_path,
                    map_location=self.device,
                    strict=False,
                )
                loaded.eval()
                # 将模型移动到指定设备
                try:
                    loaded.to(self.device)
                    if hasattr(loaded, "sd3_pipe"):
                        loaded.sd3_pipe.to(self.device)
                except Exception:
                    pass
                # 如Lightning ckpt中未携带EMA，则在权重加载完成后，用当前权重初始化EMA，以避免用到随机初始化的EMA
                try:
                    if getattr(loaded, "use_ema", False):
                        ema_shadow = getattr(loaded, "_ema_shadow", None)
                        if not isinstance(ema_shadow, dict) or len(ema_shadow) == 0:
                            if hasattr(loaded, "_init_ema"):
                                loaded._init_ema()
                                print("[Inference] Lightning加载后未检测到EMA，已用当前权重初始化EMA")
                except Exception:
                    pass
                self.model = loaded
                step = int((ckpt_obj or {}).get("global_step", 0))
                print(f"[Inference] 成功通过Lightning加载检查点，步数: {step}")
                return
            except Exception as e:
                print(f"[Inference] 使用Lightning加载失败，将尝试自定义格式: {e}")

        # 回退到自定义检查点格式
        step = self.model.load_checkpoint(checkpoint_path, lora_checkpoint_path)
        print(f"[Inference] 成功加载检查点（自定义格式），步数: {step}")
    
    def generate(
        self,
        prompts: List[str],
        output_dir: str = "./inference_output",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: Optional[int] = None,
        frames_dir: Optional[str] = None,
        save_interval: int = 1,
        save_latents: bool = False,
        sd3_baseline: bool = False,
        baseline_mode: Optional[str] = None,
        scheduler_shift: Optional[float] = None,
        save_frames: bool = True,
        image_format: str = "jpg",
        use_ema: Optional[bool] = None,
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
            save_frames: 是否保存每步 x_t 帧
        
        Returns:
            生成的图像列表
        """
        # 如果指定使用 EMA，则在生成期间切换到 EMA 权重
        did_swap_ema = False
        if use_ema is True:
            try:
                if getattr(self.model, "use_ema", False) and hasattr(self.model, "_ema_swap_to_shadow"):
                    self.model._ema_swap_to_shadow()
                    did_swap_ema = True
                    print("[Inference] 已切换至 EMA 权重进行推理")
            except Exception as exc:
                print(f"[Inference] EMA 切换失败，将继续使用当前权重: {exc}")
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
        negative_prompt = model_config.get(
            "negative_prompt",
            " deformed, low quality & bad aesthetics & poor aesthetics ",
        )
        
        print(f"[Inference] 开始生成图像...")
        print(f"[Inference] 推理参数: steps={num_inference_steps}, guidance_scale={guidance_scale}, size={height}x{width}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        frames_enabled = save_frames or save_latents
        if frames_enabled:
            frames_root = frames_dir or os.path.join(output_dir, "frames")
            os.makedirs(frames_root, exist_ok=True)
            print(f"[Inference] 每步 x_t 帧将保存到: {frames_root}")
        else:
            frames_root = None
        
        # 归一化输出图片格式
        image_ext = (image_format or "jpg").lower().strip()
        if image_ext in {"jpeg"}:
            image_ext = "jpg"
        if image_ext not in {"jpg", "png"}:
            print(f"[Inference] 警告: 不支持的图片格式 '{image_format}', 将回退为 'jpg'")
            image_ext = "jpg"

        all_images = []
        scheduler = self.model.sd3_pipe.scheduler
        if scheduler_shift is not None:
            try:
                scheduler.config.use_dynamic_shifting = False
                if hasattr(scheduler, "set_shift"):
                    scheduler.set_shift(float(scheduler_shift))
            except Exception as exc:
                print(f"[Inference] 调度器 set_shift({scheduler_shift}) 失败: {exc}")

        scheduler_shift_value = self._extract_scheduler_shift_value(scheduler)

        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                print(f"[Inference] 生成图像 {i+1}/{len(prompts)}: {prompt[:50]}...")
                
                try:
                    timestamp = int(time.time())
                    safe_prompt = self._sanitize_filename(prompt)
                    steps_suffix = f"_steps{int(num_inference_steps)}"
                    cfg_suffix = self._format_float_tag("cfg", float(guidance_scale))
                    shift_suffix = self._format_shift_suffix(scheduler_shift_value)
                    image_basename = (
                        f"inference_{timestamp}_{i:03d}_{safe_prompt}"
                        f"{steps_suffix}{cfg_suffix}{shift_suffix}"
                    )

                    # 为当前 prompt 创建独立帧目录，避免文件覆盖
                    frames_dir_i = None
                    if frames_enabled and frames_root is not None:
                        frames_dir_i = os.path.join(frames_root, image_basename)
                        os.makedirs(frames_dir_i, exist_ok=True)

                    callback_kwargs = {}
                    if frames_enabled:

                        def _save_step_callback(pipeline, step, timestep, callback_kwargs):
                            latents_step = callback_kwargs.get("latents")
                            if latents_step is None:
                                return {}
                            if save_interval > 1 and (step % int(save_interval)) != 0:
                                return {}

                            if save_frames and frames_dir_i is not None:
                                imgs_tensor = self.model.tokenizer_vae_wrapper.decode(latents_step.float())
                                imgs_tensor = imgs_tensor.detach().cpu().to(torch.float32)
                                imgs_tensor = torch.nan_to_num(imgs_tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

                                # 支持 batch（通常为1）
                                if imgs_tensor.dim() == 4:
                                    b = imgs_tensor.size(0)
                                    for bi in range(b):
                                        pil_img_step = Image.fromarray(
                                            (imgs_tensor[bi].permute(1, 2, 0).numpy() * 255).astype('uint8')
                                        )
                                        pil_img_step.save(os.path.join(frames_dir_i, f"step_{step:03d}_img_{bi}.png"))
                                else:
                                    pil_img_step = Image.fromarray(
                                        (imgs_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
                                    )
                                    pil_img_step.save(os.path.join(frames_dir_i, f"step_{step:03d}.png"))

                            if save_latents and frames_dir_i is not None:
                                torch.save(
                                    latents_step.detach().to(torch.float32).cpu(),
                                    os.path.join(frames_dir_i, f"step_{step:03d}_latents.pt"),
                                )

                            return {}

                        callback_kwargs = {
                            "callback_on_step_end": _save_step_callback,
                            "callback_on_step_end_tensor_inputs": ["latents"],
                        }

                    # 生成 latent 并使用 Tokenizer/DA-VAE 解码为像素
                    # Determine baseline behavior (new arg takes precedence; legacy flag maps to "low")
                    effective_baseline_mode = baseline_mode or ("low" if sd3_baseline else None)

                    if effective_baseline_mode == "low":
                        # 仅使用原生 SD3 路径，不再走自研 Tokenizer/DA-VAE 路线
                        self._generate_and_save_sd3_baseline_low_up(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            orig_height=height,
                            orig_width=width,
                            output_dir=output_dir,
                            image_basename=image_basename,
                            image_format=image_ext,
                        )
                    elif effective_baseline_mode == "original":
                        self._generate_and_save_sd3_baseline_original(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            height=height,
                            width=width,
                            output_dir=output_dir,
                            image_basename=image_basename,
                            image_format=image_ext,
                        )
                    else:
                        pipe_kwargs = dict(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            height=height,
                            width=width,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            output_type="latent",
                        )
                        pipe_kwargs.update(callback_kwargs)

                        latents = self.model.sd3_pipe(**pipe_kwargs).images
                        
                        # 使用 tokenizer_vae_wrapper 解码到像素
                        images_tensor = self.model.tokenizer_vae_wrapper.decode(latents.float())
                        images_tensor = images_tensor.detach().cpu().to(torch.float32)
                        images_tensor = torch.nan_to_num(images_tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                        
                        # 转为 PIL 并保存
                        pil_img = Image.fromarray((images_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype('uint8'))
                        all_images.append(pil_img)
                        
                        filename = f"{image_basename}.{image_ext}"
                        filepath = os.path.join(output_dir, filename)
                        pil_img.save(filepath)
                        print(f"[Inference] 保存图像: {filepath}")
                    
                except Exception as e:
                    print(f"[Inference] 错误: 生成图像失败 for prompt {i+1}: {e}")
                    continue
        
        print(f"[Inference] 完成! 生成了 {len(all_images)} 张图像")
        try:
            if did_swap_ema and hasattr(self.model, "_ema_restore"):
                self.model._ema_restore()
                print("[Inference] 已恢复为非 EMA 权重")
        except Exception:
            pass
        return all_images

    def _generate_and_save_sd3_baseline_low_up(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        guidance_scale: float,
        num_inference_steps: int,
        orig_height: int,
        orig_width: int,
        output_dir: str,
        image_basename: str,
        image_format: str = "jpg",
    ) -> None:
        try:
            baseline_pipe = self._get_baseline_pipe()
        except Exception as exc:
            print(f"[Inference][Baseline] 初始化原生SD3管线失败: {exc}")
            return

        if baseline_pipe is None:
            print("[Inference][Baseline] 原生SD3管线不可用，跳过")
            return

        down_height = max(orig_height // 2, 64)
        down_width = max(orig_width // 2, 64)
        down_height = max(64, (down_height // 8) * 8)
        down_width = max(64, (down_width // 8) * 8)

        try:
            result = baseline_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=down_height,
                width=down_width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            baseline_img = result.images[0]
        except Exception as exc:
            print(f"[Inference][Baseline] 生成下采样图失败: {exc}")
            return

        try:
            ext = (image_format or "jpg").lower().strip()
            if ext == "jpeg":
                ext = "jpg"
            if ext not in {"jpg", "png"}:
                ext = "jpg"
            baseline_low_path = os.path.join(output_dir, f"{image_basename}_sd3_low.{ext}")
            baseline_img.save(baseline_low_path)
            print(f"[Inference][Baseline] 保存下采样图: {baseline_low_path}")
        except Exception as exc:
            print(f"[Inference][Baseline] 保存下采样图失败: {exc}")

        try:
            upsampled = baseline_img.resize((orig_width, orig_height), resample=Image.BICUBIC)
            baseline_up_path = os.path.join(output_dir, f"{image_basename}_sd3_up.{ext}")
            upsampled.save(baseline_up_path)
            print(f"[Inference][Baseline] 保存上采样图: {baseline_up_path}")
        except Exception as exc:
            print(f"[Inference][Baseline] 保存上采样图失败: {exc}")

    def _generate_and_save_sd3_baseline_original(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        guidance_scale: float,
        num_inference_steps: int,
        height: int,
        width: int,
        output_dir: str,
        image_basename: str,
        image_format: str = "jpg",
    ) -> None:
        try:
            baseline_pipe = self._get_baseline_pipe()
        except Exception as exc:
            print(f"[Inference][Baseline] 初始化原生SD3管线失败: {exc}")
            return

        if baseline_pipe is None:
            print("[Inference][Baseline] 原生SD3管线不可用，跳过")
            return

        try:
            result = baseline_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            baseline_img = result.images[0]
        except Exception as exc:
            print(f"[Inference][Baseline] 生成原始分辨率图失败: {exc}")
            return

        try:
            ext = (image_format or "jpg").lower().strip()
            if ext == "jpeg":
                ext = "jpg"
            if ext not in {"jpg", "png"}:
                ext = "jpg"
            save_path = os.path.join(output_dir, f"{image_basename}.{ext}")
            baseline_img.save(save_path)
            print(f"[Inference][Baseline] 保存原始分辨率图: {save_path}")
        except Exception as exc:
            print(f"[Inference][Baseline] 保存原始分辨率图失败: {exc}")

    def _get_baseline_pipe(self) -> Optional[StableDiffusion3Pipeline]:
        if self._baseline_pipe is not None:
            return self._baseline_pipe

        try:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                self.config["sd3_pipe_id"], torch_dtype=self.dtype
            )
            pipe = pipe.to(self.device)
            pipe.set_progress_bar_config(disable=True)
            self._baseline_pipe = pipe
            print("[Inference][Baseline] 已加载原生SD3管线用于对照生成")
            return self._baseline_pipe
        except Exception as exc:
            print(f"[Inference][Baseline] 加载原生SD3管线失败: {exc}")
            self._baseline_pipe = None
            return None
    
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
    parser = argparse.ArgumentParser(description="SD3 DAVAE/Tokenizer 推理脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, help="检查点路径（默认自动查找最新）")
    parser.add_argument("--lora-checkpoint", type=str, help="LoRA检查点路径（默认自动查找最新）")
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
    parser.add_argument("--frames-dir", type=str, help="每步 x_t 帧保存目录（默认 <output-dir>/frames）")
    parser.add_argument("--save-interval", type=int, default=1, help="每 N 步保存一次帧（默认1）")
    parser.add_argument("--save-latents", action="store_true", help="是否保存每步原始 latent (.pt)")
    parser.add_argument("--sd3-baseline", action="store_true", help="同时生成原生SD3的2x下采样+上采样对照图")
    parser.add_argument("--shift", type=float, help="手动设置调度器 shift 值（默认使用模型原始配置）")
    parser.add_argument("--disable-frames", action="store_true", help="禁用每步 x_t 帧保存")
    
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
    inference = SD3TokenizerInference(
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
        frames_dir=args.frames_dir,
        save_interval=args.save_interval,
        save_latents=args.save_latents,
        sd3_baseline=args.sd3_baseline,
        scheduler_shift=args.shift,
        save_frames=not args.disable_frames,
    )
    
    print(f"[Inference] 推理完成! 生成了 {len(images)} 张图像")


if __name__ == "__main__":
    main()


