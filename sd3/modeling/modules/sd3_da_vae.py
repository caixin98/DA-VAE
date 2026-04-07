import torch
import torch.nn as nn
import torch.nn.functional as F
from fnmatch import fnmatch
from typing import Optional, Tuple

from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from modeling.flux.sd3_vae import SD3_VAE
from torch.utils.checkpoint import checkpoint


def _ckpt(fn, *args, **kwargs):
    """Wrapper to support Torch versions without use_reentrant kw."""
    try:
        return checkpoint(fn, *args, use_reentrant=False, **kwargs)
    except TypeError:
        return checkpoint(fn, *args, **kwargs)


class DCDownBlock2d(nn.Module):
    """
    Down block with configurable spatial compression factor `r`:
      - 主干: conv 下采样（stride=r）或 conv 后 pixel_unshuffle 下采样（等价 stride=r）
      - shortcut: 先 pixel_unshuffle，把 H,W 对齐到 H/r,W/r
          * 若 (C_in * r^2) 能被 out_channels 整除 -> 分组均值到 out_channels（零参数）
          * 否则 -> 1x1 conv 投影到 out_channels（可学习）
    形状：
      输入:  [B, C_in, H, W] (H,W 必须被 r 整除)
      输出:  [B, out_channels, H/r, W/r]
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
        shortcut: bool = True,
        factor: int = 2,
    ) -> None:
        super().__init__()
        if factor <= 0:
            raise ValueError(f"factor must be a positive integer, got {factor}")

        self.downsample = downsample
        self.shortcut = shortcut
        self.factor = int(factor)  # r
        r2 = self.factor ** 2

        # ===== 主干 conv 设置（确保最终 x 是 [B, out_channels, H/r, W/r]）=====
        self.target_out_channels = out_channels  # 目标输出通道（相加后的通道）
        conv_out_channels = out_channels
        stride = self.factor
        if downsample:
            # conv 不降采样（stride=1），再用 pixel_unshuffle 做空间/r、通道*r^2
            assert out_channels % r2 == 0, f"downsample=True 时要求 out_channels 可被 {r2} 整除"
            conv_out_channels = out_channels // r2
            stride = 1

        self.conv = nn.Conv2d(
            in_channels,
            conv_out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

        # ===== shortcut 分支的“分组均值 or 1x1 投影”选择 =====
        total_after_unshuffle = in_channels * r2  # y 在 unshuffle 之后的通道数
        self._divisible = (total_after_unshuffle % out_channels == 0)

        if self.shortcut and not self._divisible:
            # 无法整除：使用 1x1 conv 直接把 total_after_unshuffle -> out_channels
            self.shortcut_proj = nn.Conv2d(total_after_unshuffle, out_channels, kernel_size=1, bias=False)
            # 可选：零初始化，初期等价只走主干，更稳
            nn.init.zeros_(self.shortcut_proj.weight)
        else:
            self.shortcut_proj = None  # 走分组均值，无参数
            if self.shortcut:
                self.group_size = total_after_unshuffle // out_channels  # 保证是整数（_divisible=True）
                assert self.group_size >= 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = hidden_states.shape
        # 基本检查（避免奇怪尺寸）
        assert H % self.factor == 0 and W % self.factor == 0, \
            f"H,W 必须能被 {self.factor} 整除，当前 {(H, W)}"

        # 主干
        x = self.conv(hidden_states)  # [B, conv_out, H/r or H, W/r or W]
        if self.downsample:
            x = F.pixel_unshuffle(x, self.factor)  # [B, out_channels, H/r, W/r]
        # 现在 x: [B, out_channels, H/r, W/r]

        if not self.shortcut:
            return x

        # shortcut：对齐空间到 H/r,W/r
        y = F.pixel_unshuffle(hidden_states, self.factor)  # [B, C_in * r^2, H/r, W/r]

        if self._divisible:
            # 分组均值：把通道按 group_size 分组到 out_channels
            # [B, out_channels, group_size, H/r, W/r] -> 对 group 维求均值
            y = y.unflatten(1, (-1, self.group_size)).mean(dim=2)
            # y: [B, out_channels, H/r, W/r]
        else:
            # 不能整除：用 1x1 conv 做通道投影
            y = self.shortcut_proj(y)  # [B, out_channels, H/r, W/r]

        return x + y


class DCUpBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        interpolate: bool = False,
        shortcut: bool = True,
        interpolation_mode: str = "nearest",
        factor: int = 2,
    ) -> None:
        super().__init__()

        if factor <= 0:
            raise ValueError(f"factor must be a positive integer, got {factor}")

        self.interpolate = interpolate
        self.interpolation_mode = interpolation_mode
        self.shortcut = shortcut
        self.factor = int(factor)

        out_ratio = self.factor**2
        if (out_channels * out_ratio) % in_channels != 0:
            raise ValueError(
                f"out_channels * factor^2 must be divisible by in_channels; got out_channels={out_channels},"
                f" in_channels={in_channels}, factor={self.factor}"
            )
        self.repeats = (out_channels * out_ratio) // in_channels

        if not interpolate:
            out_channels = out_channels * out_ratio

        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.interpolate:
            x = F.interpolate(hidden_states, scale_factor=self.factor, mode=self.interpolation_mode)
            x = self.conv(x)
        else:
            x = self.conv(hidden_states)
            x = F.pixel_shuffle(x, self.factor)

        if self.shortcut:
            y = hidden_states.repeat_interleave(
                self.repeats,
                dim=1,
                output_size=hidden_states.shape[1] * self.repeats,
            )
            y = F.pixel_shuffle(y, self.factor)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class SD3_DAAutoencoder(nn.Module):
    """
    Deep-compressed wrapper around SD3's diffusers AutoencoderKL.

    Strategy:
    - Encode: obtain Gaussian moments (mean, logvar) from SD3 VAE, stack along channel dim, then
      pass through DCDown to shrink spatial size by `da_factor` (default 2) while increasing channels.
      Build a DiagonalGaussianDistribution on the compressed moments.
    - Decode: sample z from the compressed posterior, apply DCUp with the same factor to recover
      original latent channels/spatial size, then decode using SD3 VAE.
    """

    def __init__(
        self,
        config=None,
        model_path: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        device: Optional[str] = None,
        enable_deep_compress: bool = True,
        upsample_interpolation: str = "nearest",
        dtype: Optional[torch.dtype] = None,
        freeze_vae: Optional[bool] = None,
        freeze_dc: Optional[bool] = None,
        da_factor: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # allow config to set model_path and precision
        if config is not None:
            model_path = getattr(config.model, 'sd3_model_path', model_path)
            mp = getattr(config.training, 'mixed_precision', None)
            if dtype is None and self.device == "cuda" and (mp == 'bf16' or mp == 'bf16-mixed'):
                dtype = torch.bfloat16
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)

        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=self.dtype).to(self.device)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self.vae, "config") else 8
        )

        # gradient checkpointing flag
        self.use_gradient_checkpointing = False
        if config is not None:
            try:
                self.use_gradient_checkpointing = bool(getattr(config.model, 'enable_gradient_checkpointing', False))
            except Exception:
                self.use_gradient_checkpointing = False

        # freeze options: allow separate encoder/decoder control, backward-compatible with freeze_vae
        freeze_all_default = False if (config is None and freeze_vae is None) else False
        freeze_all = bool(getattr(config.model, 'freeze_vae', False)) if (config is not None and freeze_vae is None) else bool(freeze_vae) if freeze_vae is not None else freeze_all_default

        enc_opt = getattr(config.model, 'freeze_vae_encoder', False) if config is not None else None
        dec_opt = getattr(config.model, 'freeze_vae_decoder', False) if config is not None else None

        self.freeze_vae_encoder = bool(enc_opt) if enc_opt is not None else freeze_all
        self.freeze_vae_decoder = bool(dec_opt) if dec_opt is not None else freeze_all

        # deep-compress module freeze option
        dc_opt = getattr(config.model, 'freeze_dc', False) if config is not None else None
        self.freeze_dc = bool(dc_opt) if dc_opt is not None else (bool(freeze_dc) if freeze_dc is not None else False)

        if self.freeze_vae_encoder:
            self.freeze_vae_encoder_parameters()
        else:
            self.unfreeze_vae_encoder_parameters()
        if self.freeze_vae_decoder:
            self.freeze_vae_decoder_parameters()
        else:
            self.unfreeze_vae_decoder_parameters()

        decoder_patterns = None
        if config is not None and hasattr(config, "model"):
            decoder_patterns = getattr(config.model, "vae_decoder_trainable_layers", None)

        if decoder_patterns:
            patterns = []
            if isinstance(decoder_patterns, str):
                stripped = decoder_patterns.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    stripped = stripped[1:-1]
                patterns = [p.strip().strip("\"'") for p in stripped.split(",") if p.strip().strip("\"'")]
            else:
                patterns = [str(p).strip() for p in decoder_patterns if str(p).strip()]

            if patterns and hasattr(self.vae, "decoder") and self.vae.decoder is not None:
                wildcard_tokens = ("*", "?", "[")

                def _matches(candidate: str, pattern: str) -> bool:
                    if not pattern:
                        return False
                    if any(token in pattern for token in wildcard_tokens):
                        return fnmatch(candidate, pattern)
                    if candidate.startswith(pattern):
                        return True
                    return pattern in candidate

                matched_params = []
                for name, param in self.vae.decoder.named_parameters():
                    should_train = any(_matches(name, pattern) for pattern in patterns)
                    param.requires_grad = should_train
                    if should_train:
                        matched_params.append(name)

                self.vae.decoder.train(bool(matched_params))

                print("Partial VAE decoder fine-tuning enabled - patterns:", patterns)
                if matched_params:
                    print(f"  Trainable decoder params ({len(matched_params)}):")
                    for name in matched_params:
                        print(f"    ✓ {name}")
                else:
                    print("  Warning: No decoder parameters matched the provided patterns; decoder remains frozen.")

        da_mode = getattr(config.model, 'da_mode', 'full')
        self.enable_deep_compress = enable_deep_compress
        self.da_mode = str(da_mode).lower()
        self.latent_channels = int(getattr(self.vae.config, "latent_channels", 4))

        model_cfg = getattr(config, "model", None) if config is not None else None
        da_factor_cfg = None
        if model_cfg is not None:
            if isinstance(model_cfg, dict):
                da_factor_cfg = model_cfg.get("da_factor", None)
            else:
                da_factor_cfg = getattr(model_cfg, "da_factor", None)
        da_factor_value = da_factor if da_factor is not None else da_factor_cfg
        self.da_factor = int(da_factor_value) if da_factor_value is not None else 2
        if self.da_factor <= 0:
            raise ValueError(f"da_factor must be a positive integer, got {self.da_factor}")

        # Channels right before encoder.conv_out
        self._enc_preconv_channels = int(self.vae.encoder.conv_norm_out.num_channels)
        # Desired channels right after decoder.conv_in (to bypass it)
        self._dec_block_in = int(self.vae.decoder.conv_in.out_channels)

        # embed_dim follows dc logic per mode
        # origin: z has 2*latent_channels; diff: z has latent_channels (concat with teacher makes 2*latent_channels)
        if self.da_mode == "diff":
            self.embed_dim_dc = self.latent_channels * (self.da_factor - 1)
        else:
            self.embed_dim_dc = self.latent_channels * self.da_factor
        if self.enable_deep_compress:
            # Compress encoder preconv features (produce mean+logvar for DiagonalGaussianDistribution)
            # Input to DiagonalGaussianDistribution must be 2 * latent_dim channels
            dc_down_out_channels = 2 * self.embed_dim_dc
            self.dc_down = DCDownBlock2d(
                in_channels=self._enc_preconv_channels,
                out_channels=dc_down_out_channels,
                downsample=True,
                shortcut=True,
                factor=self.da_factor,
            )
            # Expand decoder preconv input features (target decoder.conv_in.out_channels)
            # origin: in_channels = embed_dim_dc
            # diff: in_channels = 2 * embed_dim_dc (z + teacher)
            dc_up_in_channels = self.latent_channels * self.da_factor
            self.dc_up = DCUpBlock2d(
                in_channels=dc_up_in_channels,
                out_channels=self._dec_block_in,
                interpolate=False,
                shortcut=True,
                interpolation_mode=upsample_interpolation,
                factor=self.da_factor,
            )
            # ensure dc modules share the same device & dtype as VAE
            self.dc_down.to(self.device, dtype=self.dtype)
            self.dc_up.to(self.device, dtype=self.dtype)
            # apply dc freeze policy
            if self.freeze_dc:
                self.freeze_dc_parameters()
            else:
                self.unfreeze_dc_parameters()

        # alignment hidden settings (optional, compute only when alignment loss is enabled)
        self.encoder_alignment_proj = None
        self.encoder_alignment_enabled = False
        self._align_use_proj = False
        self._align_proj_channels = None
        self.encoder_alignment_method = "proj"

        # Robust nested config access supporting dict, OmegaConf, or attribute-like objects
        def _safe_get(obj, key, default=None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        align_cfg_root = _safe_get(config, 'losses', None)
        align_cfg = _safe_get(align_cfg_root, 'encoder_alignment_loss', {})
        weight = _safe_get(align_cfg, 'weight', 0.0)
        use_encoder_proj = _safe_get(align_cfg, 'use_encoder_proj', True)
        proj_channels = _safe_get(align_cfg, 'proj_channels', 16)
        align_method = _safe_get(align_cfg, 'align_method', 'proj')
        debug_flag = _safe_get(align_cfg, 'debug', False)
        # Remove noisy prints and asserts after debugging
        self.encoder_alignment_enabled = float(weight) > 0.0
        self.encoder_alignment_method = str(align_method).lower()
        # when align_method is 'mean', disable learnable projection
        self._align_use_proj = False if self.encoder_alignment_method == 'mean' else bool(use_encoder_proj)
        self._align_proj_channels = int(proj_channels)
        self.encoder_alignment_debug = bool(debug_flag)
        self.teacher_mode = getattr(config.model, 'teacher_mode', 'lq')

        # Keep silent by default after verification
     
        # Initialize projection layer at construction time if alignment is enabled and projection requested
        if self.encoder_alignment_enabled and self._align_use_proj:
            in_channels = self.embed_dim_dc
            proj_channels = self._align_proj_channels if self._align_proj_channels is not None else in_channels
            self.encoder_alignment_proj = nn.Conv2d(in_channels, proj_channels, kernel_size=1, bias=False)
            self.encoder_alignment_proj.to(self.device, dtype=self.dtype)

        # expose original_vae_model for training utilities (decode to pixel)
        try:
            self.original_vae_model = SD3_VAE(model_path=model_path, deterministic_sampling=True)
            # keep encoder off by default to save mem as in tokenizer
            self.original_vae_model.vae.eval()
            for p in self.original_vae_model.vae.parameters():
                p.requires_grad = False
        except Exception:
            self.original_vae_model = None
        # No init-time prints by default

    def freeze_vae_parameters(self) -> None:
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()

    def unfreeze_vae_parameters(self) -> None:
        for p in self.vae.parameters():
            p.requires_grad = True
        self.vae.train()

    def freeze_vae_encoder_parameters(self) -> None:
        if hasattr(self.vae, 'encoder') and self.vae.encoder is not None:
            for p in self.vae.encoder.parameters():
                p.requires_grad = False
            self.vae.encoder.eval()

    def unfreeze_vae_encoder_parameters(self) -> None:
        if hasattr(self.vae, 'encoder') and self.vae.encoder is not None:
            for p in self.vae.encoder.parameters():
                p.requires_grad = True
            self.vae.encoder.train()

    def freeze_vae_decoder_parameters(self) -> None:
        if hasattr(self.vae, 'decoder') and self.vae.decoder is not None:
            for p in self.vae.decoder.parameters():
                p.requires_grad = False
            self.vae.decoder.eval()

    def unfreeze_vae_decoder_parameters(self) -> None:
        if hasattr(self.vae, 'decoder') and self.vae.decoder is not None:
            for p in self.vae.decoder.parameters():
                p.requires_grad = True
            self.vae.decoder.train()

    def freeze_dc_parameters(self) -> None:
        modules = [getattr(self, 'dc_down', None), getattr(self, 'dc_up', None)]
        for m in modules:
            if m is None:
                continue
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

    def unfreeze_dc_parameters(self) -> None:
        modules = [getattr(self, 'dc_down', None), getattr(self, 'dc_up', None)]
        for m in modules:
            if m is None:
                continue
            for p in m.parameters():
                p.requires_grad = True
            m.train()

    @torch.no_grad()
    def _preprocess_image(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        images = self.image_processor.preprocess(x)
        return images.to(self.device).to(self.dtype)

    def _encode_preconv(self, x: torch.Tensor) -> torch.Tensor:
        images = self._preprocess_image(x)
        e = self.vae.encoder
        sample = e.conv_in(images)
        # Avoid checkpointing encoder down blocks to prevent shape mismatches in diffusers resnets
        for down_block in e.down_blocks:
            sample = down_block(sample)
        sample = e.mid_block(sample)
        sample = e.conv_norm_out(sample)
        sample = e.conv_act(sample)
        return sample

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = self._encode_preconv(x)
        if self.enable_deep_compress:
            h = self.dc_down(h)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def _decode_from_preconv(self, h: torch.Tensor) -> torch.Tensor:
        d = self.vae.decoder
        # skip d.conv_in; start from mid and go up
        sample = h
        def _decode_all(t: torch.Tensor) -> torch.Tensor:
            out = d.mid_block(t, None)
            out = out.to(next(iter(d.up_blocks.parameters())).dtype)
            for up_block in d.up_blocks:
                out = up_block(out, None)
            out = d.conv_norm_out(out)
            out = d.conv_act(out)
            out = d.conv_out(out)
            return out

        if self.use_gradient_checkpointing:
            sample = _ckpt(_decode_all, sample)
        else:
            sample = _decode_all(sample)
        # post-process to image space
        # map from [-1,1] to [0,1]
        image = (sample + 1) / 2
        image = torch.clamp(image, 0, 1)
        return image

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # DiagonalGaussianDistribution in LDM returns float32 by default; align dtype/device
        z = z.to(self.device, dtype=self.dtype)
        if self.enable_deep_compress:
            z = self.dc_up(z)
        image = self._decode_from_preconv(z)
        return image

    def _align_project_mean(self, x: torch.Tensor, out_channels: int) -> torch.Tensor:
        """Project channels by grouped mean to `out_channels` bins.
        If `x.shape[1]` is divisible by `out_channels`, uses equal groups; otherwise
        distributes channels as evenly as possible across bins.
        """
        if out_channels is None or out_channels <= 0:
            return x
        b, c, h, w = x.shape
        if c == out_channels:
            return x
        if c % out_channels == 0:
            group = c // out_channels
            return x.unflatten(1, (out_channels, group)).mean(dim=2)
        # Uneven grouping: compute near-equal splits
        # Build integer edges across channel axis
        edges = torch.linspace(0, c, steps=out_channels + 1, device=x.device)
        edges = torch.round(edges).to(torch.int64).tolist()
        chunks = [x[:, edges[i]:edges[i+1], :, :] for i in range(out_channels)]
        pooled = [
            (torch.mean(ch, dim=1, keepdim=True) if ch.shape[1] > 0 else torch.zeros((b, 1, h, w), device=x.device, dtype=x.dtype))
            for ch in chunks
        ]
        return torch.cat(pooled, dim=1)

    def forward(self, x: torch.Tensor, sample_posterior: bool = True):
        # Encode with access to preconv features for alignment tensors
        h_preconv = self._encode_preconv(x)  # [B, C_pre, H', W']
        h_for_posterior = h_preconv
        if self.enable_deep_compress:
            h_for_posterior = self.dc_down(h_preconv)  # [B, 2*embed_dim_dc, H'/r, W'/r]

        posterior = DiagonalGaussianDistribution(h_for_posterior)
        z = posterior.sample() if sample_posterior else posterior.mode()

        # Build alignment tensors only when alignment loss is enabled
        alignment_hidden = None
        lq_cond_spatial = None

        # if self.encoder_alignment_enabled:
        # align z with teacher latents from downsampled x
        alignment_hidden = z
        if self.encoder_alignment_method == 'proj':
            if self._align_use_proj and self.encoder_alignment_proj is not None:
                alignment_hidden = self.encoder_alignment_proj(alignment_hidden.to(self.encoder_alignment_proj.weight.dtype))
        elif self.encoder_alignment_method == 'mean':
            tmp_target_c = self._align_proj_channels if self._align_proj_channels is not None else alignment_hidden.shape[1]
            alignment_hidden = self._align_project_mean(alignment_hidden, int(tmp_target_c))

        if self.original_vae_model is not None:
            # Downsample input in pixel space so teacher latents match z spatial size (H/(8*da_factor), W/(8*da_factor) for SD3 VAE)
            x_img = x
            if x_img.dim() == 3:
                x_img = x_img.unsqueeze(0)
            x_img = x_img.to(self.device, dtype=self.dtype)
            x_img = torch.clamp(x_img, 0, 1)
            if self.da_factor == 1:
                x_down = x_img
            else:
                if any(dim % self.da_factor != 0 for dim in x_img.shape[-2:]):
                    raise ValueError(
                        f"Input spatial dims {tuple(x_img.shape[-2:])} must be divisible by da_factor={self.da_factor}"
                    )
                target_hw = tuple(dim // self.da_factor for dim in x_img.shape[-2:])
                x_down = F.interpolate(x_img, size=target_hw, mode="bicubic", align_corners=False)
                x_down = torch.clamp(x_down, 0, 1)
            if self.teacher_mode == "origin":
                with torch.no_grad():
                    teacher_latents = self.original_vae_model.encode(x_img)
                    if self.da_factor != 1:
                        if any(dim % self.da_factor != 0 for dim in teacher_latents.shape[-2:]):
                            raise ValueError(
                                f"Teacher latents spatial dims {tuple(teacher_latents.shape[-2:])} must be divisible by da_factor={self.da_factor}"
                            )
                        target_hw_latent = tuple(dim // self.da_factor for dim in teacher_latents.shape[-2:])
                        teacher_latents = F.interpolate(
                            teacher_latents,
                            size=target_hw_latent,
                            mode="bicubic",
                            align_corners=False,
                        )
            elif self.teacher_mode == "lq":
                with torch.no_grad():
                    teacher_latents = self.original_vae_model.encode(x_down)
    
            tgt_dtype = alignment_hidden.dtype if alignment_hidden is not None else self.dtype
            lq_cond_spatial = teacher_latents.to(self.device, dtype=tgt_dtype)
            if self.encoder_alignment_method == 'mean' and alignment_hidden is not None:
                target_c = int(lq_cond_spatial.shape[1])
                if int(alignment_hidden.shape[1]) != target_c:
                    alignment_hidden = self._align_project_mean(alignment_hidden, target_c)
        

        # Decode per mode
        if self.da_mode == "diff":
            assert lq_cond_spatial is not None
            # if lq_cond_spatial is None:
            #     # Fallback to zeros to keep channel contract
            #     lq_cond_spatial = torch.zeros_like(z)
            dec_latent = torch.cat([z, lq_cond_spatial], dim=1)
            dec = self.decode(dec_latent)
        else:
            dec = self.decode(z)

        result = {
            "posteriors": posterior,
            "encoder_hidden_spatial": alignment_hidden,
            "lq_cond_spatial": lq_cond_spatial,
            "encoder_alignment_method": self.encoder_alignment_method if self.encoder_alignment_enabled else None,
        }
        return dec, result

    def to(self, device: Optional[str] = None, dtype: Optional[torch.dtype] = None) -> "SD3_DAAutoencoder":
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        super().to(self.device, dtype=self.dtype)
        self.vae.to(self.device, dtype=self.dtype)
        if hasattr(self, "dc_down") and self.dc_down is not None:
            self.dc_down.to(self.device, dtype=self.dtype)
        if hasattr(self, "dc_up") and self.dc_up is not None:
            self.dc_up.to(self.device, dtype=self.dtype)
        return self

    def load_pretrained(self, path: str, strict: bool = False) -> None:
        obj = torch.load(path, map_location="cpu")
        sd = obj.get("state_dict", obj)
        missing, unexpected = self.load_state_dict(sd, strict=strict)
        print(f"Loaded SD3_DAAutoencoder: missing={len(missing)} unexpected={len(unexpected)}")

    def save_pretrained_weight(self, output_dir: str, save_function=None, state_dict=None) -> None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        sd = state_dict if state_dict is not None else self.state_dict()
        path = os.path.join(output_dir, "pytorch_model.bin")
        if save_function is not None:
            save_function(sd, path)
        else:
            torch.save(sd, path)


if __name__ == "__main__":
    # Lightweight shape sanity (may download weights on first run)
    model = SD3_DAAutoencoder(enable_deep_compress=True)
    # example input in [0,1]
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        dec, posterior = model(x, sample_posterior=False)
    print("encode compressed mean shape:", posterior.mean.shape)
    print("decode output shape:", dec.shape)


