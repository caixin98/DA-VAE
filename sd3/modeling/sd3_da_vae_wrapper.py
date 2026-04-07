import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from accelerate import Accelerator

from modeling.modules.sd3_da_vae import SD3_DAAutoencoder


class SD3DAVAEWrapper(nn.Module):
    """
    SD3 DA-VAE 包装器

    - 加载配置与权重
    - 提供 encode / decode / forward 接口
    - 统一张量/ndarray 输入，自动放置到 device 与 dtype

    注意：
    - 输入图像应为 [0,1] 范围，形状为 (B, C, H, W) 或 (C, H, W)
    - encode 返回压缩潜变量 z（可选 sample 或 mode），也可选择返回 posterior
    - decode 返回像素空间图像 [0,1]
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        sample_posterior_default: bool = False,
        apply_latent_normalization: bool = False,
        latent_normalization_stats_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()

        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(requested_device)
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        self.sample_posterior_default = bool(sample_posterior_default)
        self.apply_latent_normalization = bool(apply_latent_normalization)

        self.logger = self._setup_logger()
        self.config = self._load_config(config_path) if config_path is not None else None

        self.model = self._create_model()
        self._maybe_load_pretrained(checkpoint_path)
        # 确保 original_vae_model 的 device 与 dtype 初始化对齐
        self._setup_original_vae_model()

        # inference defaults
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.latent_channels = self.model.latent_channels * self.model.da_factor
        # if self.model.da_mode == "diff":
        #     self.latent_channels = self.latent_channels * self.model.da_factor
        self.vae_scale_factor = 8 * self.model.da_factor
        
        # Latent normalization (optional)
        self._latent_norm_mean: Optional[torch.Tensor] = None
        self._latent_norm_std: Optional[torch.Tensor] = None
        self.checkpoint_dir = Path(checkpoint_path) if checkpoint_path is not None else None

        # Read normalization settings from config if available
        ln_cfg = getattr(self.config.model, "latent_normalization", None) if self.config is not None else None
        config_enable = bool(getattr(ln_cfg, "enable", False)) if ln_cfg is not None else False
        config_stats_path = getattr(ln_cfg, "stats_path", None) if ln_cfg is not None else None
        config_mean = getattr(ln_cfg, "mean", None) if ln_cfg is not None else None
        config_std = getattr(ln_cfg, "std", None) if ln_cfg is not None else None

        # If not explicitly enabled via arg, enable if config requests it
        if not self.apply_latent_normalization and config_enable:
            self.apply_latent_normalization = True

        # Prefer mean/std lists in config; else use stats path
        if config_mean is not None and config_std is not None:
            try:
                self._load_latent_norm_stats_from_values(config_mean, config_std)
                self.logger.info("✓ 已从配置加载 latent 归一化统计 (mean/std)")
            except Exception as exc:
                self.logger.warning(f"⚠️ 从配置加载 mean/std 失败: {exc}")
        else:
            try:
                preferred_path = None
                if latent_normalization_stats_path is not None:
                    preferred_path = Path(latent_normalization_stats_path)
                elif config_stats_path is not None:
                    preferred_path = Path(str(config_stats_path))
                elif self.checkpoint_dir is not None:
                    preferred_path = self.checkpoint_dir / "latent_norm_stats.json"

                if preferred_path is not None and preferred_path.exists():
                    self._load_latent_norm_stats(preferred_path)
                    self.logger.info(f"✓ 已加载 latent 归一化统计: {preferred_path}")
            except Exception as exc:
                self.logger.warning(f"⚠️ 加载 latent 归一化统计失败: {exc}")
    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("SD3-DAVAE-Wrapper")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _load_config(self, path_like: Union[str, Path]) -> OmegaConf:
        path = Path(path_like)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        cfg = OmegaConf.load(path)
        print(f"配置文件加载成功: {path}")
        return cfg

    def _create_model(self) -> SD3_DAAutoencoder:
        model = SD3_DAAutoencoder(config=self.config)
        model = model.to(self.device, dtype=self.dtype)
        return model

    def _maybe_load_pretrained(self, checkpoint_path: Optional[Union[str, Path]]) -> None:
        if checkpoint_path is None or checkpoint_path == "none":
            return
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"预训练权重不存在: {path}")
        # 当传入的是目录：优先尝试 accelerator 的 state；否则在目录中查找常见权重文件
        if path.is_dir():
            # # 优先在目录中寻找直接的权重文件（常见布局）
            # candidate_files = [
            #     path / "pytorch_model.bin",
            #     path / "model.safetensors",
            #     path / "unwrapped_model" / "pytorch_model.bin",
            #     path / "unwrapped_model" / "model.safetensors",
            # ]
            # file_found = None
            # for f in candidate_files:
            #     if f.exists():
            #         file_found = f
            #         break

            # if file_found is not None:
            #     try:
            #         self.model.load_pretrained(str(file_found), strict=False)
            #         self.logger.info(f"已从目录中的权重文件加载: {file_found}")
            #         return
            #     except Exception as exc:
            #         self.logger.warning(f"从目录内文件加载失败，尝试 accelerator state: {exc}")

            # 使用 Accelerator 尝试加载完整 state（与 sd3_tokenizer_2d_wrapper 类似）
            try:
                accel = Accelerator(
                    mixed_precision=getattr(getattr(self.config, 'training', None), 'mixed_precision', None) if self.config is not None else None,
                    gradient_accumulation_steps=getattr(getattr(self.config, 'training', None), 'gradient_accumulation_steps', 1) if self.config is not None else 1,
                    log_with=getattr(getattr(self.config, 'training', None), 'log_with', None) if self.config is not None else None,
                    project_dir=getattr(getattr(self.config, 'experiment', None), 'output_dir', None) if self.config is not None else None,
                )
                prepared = accel.prepare(self.model)
                accel.load_state(path, strict=False)
                # unwrap 并确保设备/精度
                self.model = accel.unwrap_model(prepared)
                self.model = self.model.to(self.device, dtype=self.dtype)
                self.logger.info(f"已通过 Accelerator 从目录加载 state: {path}")
                return
            except Exception as exc:
                self.logger.error(f"Accelerator 加载 state 失败: {exc}")
                raise
        else:
            # 直接文件加载
            try:
                self.model.load_pretrained(str(path), strict=False)
                self.logger.info(f"已加载预训练权重: {path}")
            except Exception as exc:
                self.logger.error(f"加载预训练权重失败: {exc}")
                raise

    def _setup_original_vae_model(self) -> None:
        """
        将 original_vae_model 移动到与 wrapper 相同的 device，并记录其权重 dtype，
        以便后续构建 lq_cond 时对输入进行一致的 dtype 铸造。
        """
        self.original_vae_dtype: Optional[torch.dtype] = None
        if not hasattr(self.model, "original_vae_model"):
            return
        vae_model = getattr(self.model, "original_vae_model", None)
        if vae_model is None:
            return
        # 直接对齐底层 VAE 模块（diffusers AutoencoderKL）
        inner_vae = getattr(vae_model, "vae", None)
        if inner_vae is None:
            raise RuntimeError("original_vae_model 不包含属性 'vae'，无法完成初始化设置")
        try:
            inner_vae = inner_vae.to(self.device)
            setattr(vae_model, "vae", inner_vae)
            # 同步外层跟踪信息（供 encode 使用）
            try:
                setattr(vae_model, "device", self.device)
                if hasattr(vae_model, "pipeline") and vae_model.pipeline is not None:
                    vae_model.pipeline.device = self.device
            except Exception:
                pass
        except Exception as exc:
            self.logger.warning(f"original_vae_model.vae 移动到设备失败: {exc}")

        # 记录 dtype，必须从底层 VAE 参数推断
        for p in inner_vae.parameters():
            self.original_vae_dtype = p.dtype
            break
        if self.original_vae_dtype is None:
            for b in inner_vae.buffers():
                self.original_vae_dtype = b.dtype
                break
        if self.original_vae_dtype is None:
            raise RuntimeError("无法确定 original_vae_model 的 dtype，请检查其权重或初始化逻辑")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(
        self,
        image: Union[torch.Tensor, np.ndarray],
        sample_posterior: Optional[bool] = None,
        return_posterior: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        编码图像到深度压缩潜变量。

        Args:
            image: (B, C, H, W) 或 (C, H, W)，范围 [0,1]
            sample_posterior: 是否从 posterior 采样；None 则使用默认设定
            return_posterior: 是否同时返回 posterior（DiagonalGaussianDistribution）

        Returns:
            - z
            - (z, posterior)
            - 对于 da_mode="diff"，返回拼接后的潜变量 [z, lq_cond]
              或 (concat, posterior)
        """
        tensor = self._to_tensor(image).to(self.device, dtype=self.dtype)
        posterior = self.model.encode(tensor)
        use_sample = self.sample_posterior_default if sample_posterior is None else bool(sample_posterior)
        z = posterior.sample() if use_sample else posterior.mode()
        if getattr(self.model, "da_mode", "full") == "diff":
            lq_cond = self._create_lq_cond_from_image(tensor).to(self.device, dtype=self.dtype)
            if lq_cond.shape[-2:] != z.shape[-2:]:
                raise ValueError(
                    f"lq_cond 形状 {tuple(lq_cond.shape)} 与 z 形状 {tuple(z.shape)} 的空间维度不匹配"
                )
            
            # Apply latent normalization if enabled (only on z, not lq_cond)
            if self.apply_latent_normalization:
                z = self._apply_latent_normalization(z, normalize_all_channels=True)
            
            concat = torch.cat([z, lq_cond], dim=1)
            
            if return_posterior:
                return concat, posterior
            return concat
        else:
            # Apply latent normalization if enabled
            if self.apply_latent_normalization:
                z = self._apply_latent_normalization(z, normalize_all_channels=True)
            
            if return_posterior:
                return z, posterior
            return z

    @torch.no_grad()
    def decode(self, z: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        解码潜变量到像素空间图像 [0,1]。
        - 对于 da_mode="diff"，传入应为已拼接好的潜变量 [z, lq_cond]
        """
        z_tensor = self._to_tensor(z)
        
        # Apply latent denormalization if enabled
        # For da_mode="diff", only denormalize the z part, keep lq_cond unchanged
        if self.apply_latent_normalization:
            if getattr(self.model, "da_mode", "full") == "diff":
                # Split into z and lq_cond
                z_channels = self.model.embed_dim_dc
                if z_tensor.shape[1] > z_channels:
                    z_part = z_tensor[:, :z_channels]
                    lq_cond_part = z_tensor[:, z_channels:]
                    # Denormalize only z part
                    z_part = self._apply_latent_denormalization(z_part, normalize_all_channels=True)
                    z_tensor = torch.cat([z_part, lq_cond_part], dim=1)
                else:
                    # Just z, no lq_cond concatenated
                    z_tensor = self._apply_latent_denormalization(z_tensor, normalize_all_channels=True)
            else:
                z_tensor = self._apply_latent_denormalization(z_tensor, normalize_all_channels=True)
        
        image = self.model.decode(z_tensor)
        return image

    @torch.no_grad()
    def forward(
        self,
        image: Union[torch.Tensor, np.ndarray],
        sample_posterior: Optional[bool] = None,
        return_posterior: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        直接重建：image -> z -> image。
        """
        tensor = self._to_tensor(image)
        # 复用内部 forward 以获得更高效的管线与对齐输出（如启用时）
        sample_flag = self.sample_posterior_default if sample_posterior is None else bool(sample_posterior)
        dec, result = self.model(tensor, sample_posterior=sample_flag)
        if return_posterior:
            posterior = result.get("posterior", None)
            if posterior is None:
                # 回落到重新编码以获取 posterior
                posterior = self.model.encode(tensor)
            return dec, posterior
        return dec

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    def _to_tensor(self, value: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value).to(torch.float32)
        elif isinstance(value, torch.Tensor):
            tensor = value
        else:
            raise TypeError(f"不支持的输入类型: {type(value)}")

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device, dtype=self.dtype)
        return tensor

    @torch.no_grad()
    def _create_lq_cond_from_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        基于输入图像创建用于 da_mode="diff" 的教师潜变量（与模型 forward 保持一致）。
        - 在像素空间按 da_factor 下采样，然后通过 original_vae_model.encode 得到潜变量。
        """
        if not hasattr(self.model, "original_vae_model") or self.model.original_vae_model is None:
            raise RuntimeError("original_vae_model 不可用，无法构建 lq_cond 潜变量")

        x = image_tensor
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = torch.clamp(x.to(self.device, dtype=self.dtype), 0.0, 1.0)
        dc = int(getattr(self.model, "da_factor", 2))
        if dc <= 1:
            x_down = x
        else:
            h, w = x.shape[-2:]
            if (h % dc != 0) or (w % dc != 0):
                raise ValueError(
                    f"输入图像空间尺寸 {tuple(x.shape[-2:])} 不能被 da_factor={dc} 整除"
                )
            target_hw = (h // dc, w // dc)
            x_down = F.interpolate(x, size=target_hw, mode="bicubic", align_corners=False)
        x_down = torch.clamp(x_down, 0.0, 1.0)
        with torch.no_grad():
            vae_model = self.model.original_vae_model
            vae_dtype = self.original_vae_dtype
            lq = vae_model.encode(x_down.to(self.device, dtype=vae_dtype))
        return lq.to(self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Latent Normalization Helpers
    # ------------------------------------------------------------------
    def _load_latent_norm_stats(self, stats_path: Path) -> None:
        """从 JSON 文件加载归一化统计量"""
        with stats_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        mean = data.get("mean")
        std = data.get("std")
        if mean is None or std is None:
            raise ValueError("统计文件缺少 mean 或 std 字段")
        mean_t = torch.tensor(mean, dtype=self.dtype, device=self.device).view(1, -1, 1, 1)
        std_arr = np.array(std, dtype=np.float64)
        std_arr[std_arr <= 0.0] = 1e-6
        std_t = torch.tensor(std_arr.tolist(), dtype=self.dtype, device=self.device).view(1, -1, 1, 1)
        self._latent_norm_mean = mean_t
        self._latent_norm_std = std_t

    def _load_latent_norm_stats_from_values(self, mean_list, std_list) -> None:
        """从列表加载归一化统计量"""
        if mean_list is None or std_list is None:
            raise ValueError("mean/std 不能为空")
        mean_t = torch.tensor(mean_list, dtype=self.dtype, device=self.device).view(1, -1, 1, 1)
        std_arr = np.array(std_list, dtype=np.float64)
        std_arr[std_arr <= 0.0] = 1e-6
        std_t = torch.tensor(std_arr.tolist(), dtype=self.dtype, device=self.device).view(1, -1, 1, 1)
        self._latent_norm_mean = mean_t
        self._latent_norm_std = std_t

    def _apply_latent_normalization(self, latents: torch.Tensor, normalize_all_channels: bool = True) -> torch.Tensor:
        """应用 latent 归一化
        
        Args:
            latents: 输入的 latent 张量
            normalize_all_channels: 是否归一化所有通道（True）
        """
        if self._latent_norm_mean is None or self._latent_norm_std is None:
            return latents
        if latents.dim() != 4:
            return latents
        
        c = latents.shape[1]
        stats_channels = int(self._latent_norm_mean.shape[1])
        
        # Check if stats match the number of channels
        if stats_channels == c:
            # Normalize all channels
            return (latents - self._latent_norm_mean) / self._latent_norm_std
        
        # Unknown alignment; skip normalization to be safe
        self.logger.warning(
            f"⚠️ 归一化统计通道数与输入不匹配: stats={stats_channels}, input={c}. 已跳过归一化。"
        )
        return latents

    def _apply_latent_denormalization(self, latents: torch.Tensor, normalize_all_channels: bool = True) -> torch.Tensor:
        """应用 latent 反归一化
        
        Args:
            latents: 输入的 latent 张量
            normalize_all_channels: 是否反归一化所有通道（True）
        """
        if self._latent_norm_mean is None or self._latent_norm_std is None:
            return latents
        if latents.dim() != 4:
            return latents
        
        c = latents.shape[1]
        stats_channels = int(self._latent_norm_mean.shape[1])
        
        # Check if stats match the number of channels
        if stats_channels == c:
            # Denormalize all channels
            return latents * self._latent_norm_std + self._latent_norm_mean
        
        # Unknown alignment; skip denormalization to be safe
        self.logger.warning(
            f"⚠️ 反归一化统计通道数与输入不匹配: stats={stats_channels}, input={c}. 已跳过反归一化。"
        )
        return latents


