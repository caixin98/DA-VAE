from __future__ import annotations

# Early global suppression for noisy resume RNG state warnings
import warnings as _early_warnings
try:
    _early_warnings.filterwarnings(
        "ignore",
        message=r"(?i).*has changed so we cannot load this state.*",
        category=Warning,
    )
    _early_warnings.filterwarnings(
        "ignore",
        message=r"(?i).*intSeed has changed so we cannot load this state.*",
        category=Warning,
    )
    # As a last resort, drop matching warnings at emission time
    import re as _re
    _orig_showwarning = _early_warnings.showwarning
    def _showwarning_no_rng_state(message, category, filename, lineno, file=None, line=None):
        try:
            text = str(message)
            if _re.search(r"(?i)has changed so we cannot load this state", text):
                return
        except Exception:
            pass
        return _orig_showwarning(message, category, filename, lineno, file=file, line=line)
    _early_warnings.showwarning = _showwarning_no_rng_state
except Exception:
    pass

from typing import Any, Dict, List, Optional, Union, Tuple
import os
import math
import time
import sys
import importlib
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import lightning as L
from torch.utils.data import DataLoader, Dataset, IterableDataset
from lightning.pytorch.loggers import CSVLogger
from PIL import Image
try:
    from lightning.pytorch.loggers import WandbLogger  # optional
except Exception:
    WandbLogger = None

from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.strategies import DDPStrategy
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from peft import LoraConfig
from peft import get_peft_model_state_dict  # type: ignore

from ..pipeline.sd3_transformer_wrapper_tokenizer import create_sd3_transformer_wrapper_tokenizer
from ..train_sd3_hr.simple_peft_lora_manager import SimplePEFTLoRAManager

# -------------------------------------------------------------------------
# 开源清理：移除硬编码的本地开发路径，改为动态推断仓库根目录（open_source/sd3）
# 这样在 bash 脚本里只需要把 open_source/sd3 加入 PYTHONPATH 即可。
# -------------------------------------------------------------------------
try:
    from pathlib import Path as _Path
    _SD3_ROOT = _Path(__file__).resolve().parents[3]  # .../open_source/sd3
    _sd3_root_str = str(_SD3_ROOT)
    if _sd3_root_str not in sys.path:
        sys.path.insert(0, _sd3_root_str)
except Exception:
    # 兜底：不阻断训练
    pass

import torchvision.transforms.functional as TF

# Silence verbose warnings from transformers and diffusers (e.g., CLIP 77-token notices)
try:
    from transformers.utils import logging as tlogging
    tlogging.set_verbosity_error()
except Exception:
    pass
try:
    from diffusers.utils import logging as dlogging
    dlogging.set_verbosity_error()
except Exception:
    pass

# Silence specific PIL warnings (EXIF corruption, palette transparency to RGBA)
try:
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r".*Corrupt EXIF data.*",
        category=UserWarning,
        module="PIL.TiffImagePlugin",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Palette images with Transparency expressed in bytes.*",
        category=UserWarning,
        module="PIL.Image",
    )
except Exception:
    pass

import warnings
warnings.filterwarnings(
    "ignore",
    message="Corrupt EXIF data",
    category=UserWarning,
    module="PIL.TiffImagePlugin",
)


# Silence NumPy RNG seed state compatibility warnings (resume may load different SeedSequence)
try:
    import warnings as _np_warnings
    _np_warnings.filterwarnings(
        "ignore",
        message=r".*SeedSequence has changed so we cannot load this state.*",
        category=UserWarning,
        module="numpy.random",
    )
    _np_warnings.filterwarnings(
        "ignore",
        message=r".*has changed so we cannot load this state.*",
        category=UserWarning,
        module="numpy.random",
    )
except Exception:
    pass


# Broaden suppression to any module emitting this resume RNG state warning
try:
    warnings.filterwarnings(
        "ignore",
        message=r"(?i).*has changed so we cannot load this state.*",
        category=UserWarning,
    )
except Exception:
    pass

try:
    # Catch any warning category, case-insensitive, to be extra safe
    warnings.filterwarnings(
        "ignore",
        message=r"(?i).*has changed so we cannot load this state.*",
        category=Warning,
    )
    # Also explicitly match the observed 'intSeed' variant from terminal output
    warnings.filterwarnings(
        "ignore",
        message=r"(?i).*intSeed has changed so we cannot load this state.*",
        category=Warning,
    )
except Exception:
    pass



def _as_float(v: Any, d: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


class SD3TokenizerModel(L.LightningModule):
    """
    Tokenizer-focused training module:
    - Replaces transformer's pos_embed/proj_out with tokenizer VAE patch embedding and scaled head
    - Uses standard SD3 forward path; trains LoRA and tokenizer heads
    """

    def __init__(
        self,
        sd3_pipe_id: str,
        patch_embed_cfg: dict,
        patch_embed_weights_path: Optional[str] = None,
        lora_paths: Optional[dict] = None,
        lora_config: Optional[dict] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        optimizer_config: Optional[dict] = None,
        model_config: Optional[dict] = None,
        train_dataset: Optional[Any] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config or {}
        self.optimizer_config = optimizer_config or {"type": "AdamW", "params": {"lr": 1e-4, "weight_decay": 1e-2}}
        
        patch_embed_cfg = dict(patch_embed_cfg or {})

        # Store training dataset for random sampling during validation
        self.train_dataset = train_dataset

        # Load SD3 pipeline
        self.sd3_pipe: StableDiffusion3Pipeline = StableDiffusion3Pipeline.from_pretrained(
            sd3_pipe_id, torch_dtype=dtype
        ).to(device)

        # Monkeypatch pipeline to skip CLIP untruncated length check (prevents 77-token warnings)
        try:
            from types import MethodType

            def _get_clip_prompt_embeds_no_warn(_self,
                prompt: Union[str, List[str]],
                num_images_per_prompt: int = 1,
                device: Optional[torch.device] = None,
                clip_skip: Optional[int] = None,
                clip_model_index: int = 0,
            ):
                device_local = device or _self._execution_device

                clip_tokenizers = [_self.tokenizer, _self.tokenizer_2]
                clip_text_encoders = [_self.text_encoder, _self.text_encoder_2]

                tokenizer = clip_tokenizers[clip_model_index]
                text_encoder = clip_text_encoders[clip_model_index]

                prompt_list = [prompt] if isinstance(prompt, str) else prompt
                batch_size = len(prompt_list)

                text_inputs = tokenizer(
                    prompt_list,
                    padding="max_length",
                    max_length=_self.tokenizer_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids

                outputs = text_encoder(text_input_ids.to(device_local), output_hidden_states=True)
                pooled_prompt_embeds_local = outputs[0]

                if clip_skip is None:
                    prompt_embeds_local = outputs.hidden_states[-2]
                else:
                    prompt_embeds_local = outputs.hidden_states[-(clip_skip + 2)]

                prompt_embeds_local = prompt_embeds_local.to(dtype=_self.text_encoder.dtype, device=device_local)

                _, seq_len, _ = prompt_embeds_local.shape
                prompt_embeds_local = prompt_embeds_local.repeat(1, num_images_per_prompt, 1)
                prompt_embeds_local = prompt_embeds_local.view(batch_size * num_images_per_prompt, seq_len, -1)

                pooled_prompt_embeds_local = pooled_prompt_embeds_local.repeat(1, num_images_per_prompt, 1)
                pooled_prompt_embeds_local = pooled_prompt_embeds_local.view(batch_size * num_images_per_prompt, -1)

                return prompt_embeds_local, pooled_prompt_embeds_local

            self.sd3_pipe._get_clip_prompt_embeds = MethodType(_get_clip_prompt_embeds_no_warn, self.sd3_pipe)
        except Exception:
            pass

        tokenizer_cfg: Dict[str, Any] = dict(self.model_config.get("tokenizer_vae", {}) or {})
        tokenizer_flag = tokenizer_cfg.get("enabled", None)
        # Default to True for tokenizer-focused trainer
        self.use_tokenizer_vae: bool = bool(
            self.model_config.get("use_tokenizer_vae", tokenizer_flag if tokenizer_flag is not None else True)
        )
        self.tokenizer_vae_wrapper: Optional[nn.Module] = None
        self.tokenizer_latent_channels: Optional[int] = None
        self.tokenizer_vae_device = device
        self.tokenizer_vae_dtype = dtype

        # Freeze encoders and VAE
        self.sd3_pipe.text_encoder.requires_grad_(False).eval().to(dtype=dtype)
        self.sd3_pipe.text_encoder_2.requires_grad_(False).eval().to(dtype=dtype)
        self.sd3_pipe.vae.requires_grad_(False).eval().to(dtype=dtype)

        # Tokenizer VAE path only
        if self.use_tokenizer_vae:
            self.tokenizer_vae_wrapper = self._init_tokenizer_vae(tokenizer_cfg, device=device, dtype=dtype)
            self.tokenizer_vae_device = getattr(self.tokenizer_vae_wrapper, "device", device)
            self.tokenizer_vae_dtype = getattr(self.tokenizer_vae_wrapper, "dtype", dtype)

            self.tokenizer_latent_channels = self.tokenizer_vae_wrapper.latent_channels
            if hasattr(self.tokenizer_vae_wrapper, "vae_scale_factor"):
                self.sd3_pipe.vae_scale_factor = self.tokenizer_vae_wrapper.vae_scale_factor

            tokenizer_patch_cfg = dict(patch_embed_cfg or {})
            tokenizer_patch_cfg["in_channels"] = self.tokenizer_latent_channels
            tokenizer_patch_cfg.setdefault("out_channels", self.tokenizer_latent_channels)

            self.transformer = create_sd3_transformer_wrapper_tokenizer(
                transformer=self.sd3_pipe.transformer,
                patch_embed_cfg=tokenizer_patch_cfg,
                patch_embed_weights_path=patch_embed_weights_path,
                device=device,
                dtype=dtype,
            )
            self.sd3_pipe.transformer = self.transformer
            # Optional random initialization for tokenizer patch embed
            try:
                init_mode = str(tokenizer_patch_cfg.get("init_mode", "")).lower()
            except Exception:
                init_mode = ""
            if init_mode == "random" and not patch_embed_weights_path:
                try:
                    self._randomly_initialize_tokenizer_patch_embed()
                    print("[TOK][Init] tokenizer_patch_embed initialized with random weights (init_mode=random)")
                except Exception as _e_init:
                    print(f"[TOK][Init] random init requested but failed: {_e_init}")
        else:
            # Fallback: no tokenizer wrapper
            self.transformer = self.sd3_pipe.transformer

        # Freeze all base transformer params; LoRA layers will be re-enabled below
        self.transformer.requires_grad_(False)

        # Optional: full finetune mode (disable LoRA, unfreeze all transformer params)
        self.full_finetune: bool = bool(self.model_config.get("full_finetune", False))
        if self.full_finetune:
            self.transformer.requires_grad_(True)

        # Head training flags (tokenizer-focused)
        if "train_pos_embed" not in self.model_config:
            self.model_config["train_pos_embed"] = bool(self.use_tokenizer_vae)
        if "train_proj_out" not in self.model_config:
            self.model_config["train_proj_out"] = bool(self.use_tokenizer_vae)
        if not self.use_tokenizer_vae:
            self.model_config["train_pos_embed"] = False
            self.model_config["train_proj_out"] = False

        # Initialize scheduler clone
        self.noise_scheduler = self.sd3_pipe.scheduler.__class__.from_config(self.sd3_pipe.scheduler.config)

        # LoRA setup (optional): add adapters / load existing (skip when full finetune)
        self.lora_manager = SimplePEFTLoRAManager(self.sd3_pipe, self.transformer)
        if not self.full_finetune:
            self._setup_lora(lora_paths=lora_paths, lora_config=lora_config)

        # Ensure at least one module requires grad before DDP wraps the model
        train_proj_out = bool(self.model_config.get("train_proj_out", True))
        train_pos_embed = bool(self.model_config.get("train_pos_embed", True))

        any_enabled = False
        if train_proj_out:
            proj_module = self._get_proj_out_module()
            if proj_module is not None:
                for p in proj_module.parameters():
                    p.requires_grad_(True)
                any_enabled = True

        if train_pos_embed:
            for patch_module in self._iter_patch_embed_modules():
                for p in patch_module.parameters():
                    p.requires_grad_(True)
                any_enabled = True

        # Final fallback: only when tokenizer wrapper is enabled
        if not any_enabled and self.use_tokenizer_vae:
            proj_module = self._get_proj_out_module()
            if proj_module is not None:
                for p in proj_module.parameters():
                    p.requires_grad_(True)

        # Sampling/validation configuration
        self.sample_interval: int = int(self.model_config.get("sample_interval", 1000))
        self.sample_prompts: List[str] = list(self.model_config.get("validation_prompts", []))
        # Fixed overfit prompt support
        self.fixed_train_prompt: Optional[str] = self.model_config.get("fixed_train_prompt")
        # Default negative prompt for sampling/inference
        self.default_negative_prompt: Optional[str] = self.model_config.get(
            "negative_prompt",
            " deformed, low quality & bad aesthetics & poor aesthetics ",
        )
        self.sample_size: int = int(self.model_config.get("sample_size", 1024))
        self.save_path: str = str(self.model_config.get("save_path", "./output/tokenizer"))
        self.print_every_n_steps: int = int(self.model_config.get("print_every_n_steps", 10))
        # Residual-guided scheduling config
        try:
            self.use_residual_schedule: bool = bool(self.model_config.get("use_residual_schedule", True))
        except Exception:
            self.use_residual_schedule = True
        try:
            self.residual_warmup_steps: int = int(self.model_config.get("residual_warmup_steps", 2000))
        except Exception:
            self.residual_warmup_steps = 2000
        try:
            residual_end_weight = float(self.model_config.get("residual_schedule_end_weight", 1.0))
        except Exception:
            residual_end_weight = 1.0
        self.residual_schedule_end_weight: float = max(0.0, residual_end_weight)
        # Gradient explosion guard configuration
        try:
            self.grad_skip_threshold: float = float(self.model_config.get("grad_skip_threshold", 10.0))
        except Exception:
            self.grad_skip_threshold = 0.0
        try:
            self.skip_on_nonfinite_grad: bool = bool(self.model_config.get("skip_on_nonfinite_grad", True))
        except Exception:
            self.skip_on_nonfinite_grad = True
        # Sigma clamp for weighting stability
        try:
            self.min_sigma_for_weighting: float = float(self.model_config.get("min_sigma_for_weighting", 1e-3))
        except Exception:
            self.min_sigma_for_weighting = 1e-3
        # Min-SNR-γ 策略参数
        try:
            self.use_min_snr_gamma: bool = bool(self.model_config.get("use_min_snr_gamma", False))
        except Exception:
            self.use_min_snr_gamma = False
        try:
            self.min_snr_gamma: float = float(self.model_config.get("min_snr_gamma", 5.0))
        except Exception:
            self.min_snr_gamma = 5.0
        # Mid loss 增强策略参数
        try:
            self.mid_loss_boost: bool = bool(self.model_config.get("mid_loss_boost", False))
        except Exception:
            self.mid_loss_boost = False
        try:
            self.mid_loss_weight: float = float(self.model_config.get("mid_loss_weight", 2.0))
        except Exception:
            self.mid_loss_weight = 2.0
        # Prompt dropout probability (for CFG-friendly finetuning); default disabled
        try:
            self.prompt_dropout_prob: float = float(self.model_config.get("prompt_dropout_prob", 0.1))
        except Exception:
            self.prompt_dropout_prob = 0.0
        # Checkpoint saving (configured later by train())
        self.save_interval: int = int(self.model_config.get("save_interval", -1))
        self.checkpoint_dir: str = str(self.model_config.get("checkpoint_dir", "./checkpoints"))

        # Ensure transformer runs in training mode (dropout, norms) even if base weights are frozen
        self.transformer.train()

        self.to(device).to(dtype)

        # EMA 配置
        try:
            self.use_ema: bool = bool(self.model_config.get("use_ema", True))
        except Exception:
            self.use_ema = True
        try:
            self.ema_decay: float = float(self.model_config.get("ema_decay", 0.9999))
        except Exception:
            self.ema_decay = 0.9999
        try:
            self.ema_update_every: int = int(self.model_config.get("ema_update_every", 1))
        except Exception:
            self.ema_update_every = 1
        try:
            self.use_ema_for_sampling: bool = bool(self.model_config.get("use_ema_for_sampling", True))
        except Exception:
            self.use_ema_for_sampling = True
        self._ema_shadow: Dict[str, torch.Tensor] = {}
        self._ema_backup: Optional[Dict[str, torch.Tensor]] = None
        self._ema_num_updates: int = 0
        if self.use_ema:
            try:
                self._init_ema()
            except Exception:
                self.use_ema = False
        # Runtime checks
        self._tokenizer_checked: bool = False
        # Dedup guards for per-step saves
        self._last_saved_lora_step: int = -1
        self._last_saved_tok_step: int = -1

        # Async S3 upload executor
        self._s3_executor = None
        try:
            max_workers = int(self.model_config.get("s3_max_workers", 2))
        except Exception:
            max_workers = 2
        try:
            self._s3_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="s3-uploader")
        except Exception:
            self._s3_executor = None

    # ----------------- numeric safety & diagnostics -----------------
    @staticmethod
    def _nan_to_num_(tensor: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(tensor).all():
            return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor

    def _log_tensor_stats(self, name: str, tensor: torch.Tensor) -> None:
        try:
            t = tensor.detach()
            finite = torch.isfinite(t)
            finite_ratio = finite.float().mean().item()
            self.log(f"train/{name}_finite_ratio", finite_ratio, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
            if finite.any():
                t_f = t[finite]
                self.log(f"train/{name}_mean", t_f.mean(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                self.log(f"train/{name}_max", t_f.max(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                self.log(f"train/{name}_min", t_f.min(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        except Exception:
            pass

    @torch.no_grad()
    def _pipeline_infer(
        self,
        prompt: str,
        height: int,
        width: int,
        decode_vae16: bool = True,
        use_ema_override: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[List["Image.Image"], Optional["Image.Image"]]:
        """
        Run SD3 pipeline for a prompt and optionally decode first 16 channels via Tokenizer VAE.
        Returns (images, vae16_image). images is a list of PIL images.
        """
        _did_swap_to_ema = False
        # Decide whether to use EMA for this inference
        try:
            if use_ema_override is None:
                do_ema = bool(getattr(self, "use_ema", False) and getattr(self, "use_ema_for_sampling", False))
            else:
                do_ema = bool(use_ema_override) and bool(getattr(self, "use_ema", False))
        except Exception:
            do_ema = False
        if do_ema:
            try:
                self._ema_swap_to_shadow()
                _did_swap_to_ema = True
            except Exception:
                pass
        try:
            vae16_pil: Optional["Image.Image"] = None
            if self.use_tokenizer_vae:
                latents = self.sd3_pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=int(self.model_config.get("num_inference_steps", 28)),
                    guidance_scale=_as_float(self.model_config.get("guidance_scale", 3.5), 3.5),
                    generator=generator,
                    output_type="latent",
                ).images
                # Decode to pixel image only; skip optional low-quality/lq outputs
                images_tensor = self.tokenizer_vae_wrapper.decode(latents.float())
                images_tensor = images_tensor.detach().cpu().to(torch.float32)
                images_tensor = torch.nan_to_num(images_tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                pil_img = TF.to_pil_image(images_tensor.squeeze(0))
                images: List["Image.Image"] = [pil_img]
                # Do not produce/return lq image for now
                vae16_pil = None
                return images, vae16_pil
            else:
                images = self.sd3_pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=int(self.model_config.get("num_inference_steps", 28)),
                    guidance_scale=_as_float(self.model_config.get("guidance_scale", 3.5), 3.5),
                    generator=generator,
                ).images
                return images, None
        finally:
            if _did_swap_to_ema:
                try:
                    self._ema_restore()
                except Exception:
                    pass

    def _init_tokenizer_vae(
        self, cfg: Dict[str, Any], device: Union[str, torch.device], dtype: torch.dtype
    ) -> nn.Module:
        repo_path = cfg.get("repo_path") or cfg.get("module_path") or os.environ.get("SR_TOKENIZER_PATH")
        if repo_path and str(repo_path) not in sys.path:
            sys.path.append(str(repo_path))
        class_path = cfg.get("wrapper_class", "modeling.sd3_da_vae_wrapper.SD3DAVAEWrapper")
        try:
            module_name, class_name = class_path.rsplit(".", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid tokenizer wrapper_class '{class_path}'. Expected '<module>.<ClassName>'."
            ) from exc

        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(f"Failed to import tokenizer VAE module '{module_name}': {exc}") from exc

        try:
            wrapper_cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ImportError(f"Module '{module_name}' does not define '{class_name}'.") from exc

        config_path = cfg.get("config_path")
        checkpoint_path = cfg.get("checkpoint_path")
        if not config_path or not checkpoint_path:
            raise ValueError("tokenizer_vae configuration requires 'config_path' and 'checkpoint_path'.")
        if not os.path.exists(str(config_path)):
            raise FileNotFoundError(f"Tokenizer config not found: {config_path}")
        if not os.path.exists(str(checkpoint_path)):
            raise FileNotFoundError(f"Tokenizer checkpoint not found: {checkpoint_path}")

        # Some wrappers (e.g., Tokenizer2D) accept 'auto_decode_to_pixel'; DA-VAE may not.
        try:
            wrapper = wrapper_cls(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                device=device,
                dtype=dtype,
                auto_decode_to_pixel=bool(cfg.get("auto_decode_to_pixel", False)),
                apply_latent_normalization=bool(cfg.get("apply_latent_normalization", False)),
            )
        except TypeError:
            wrapper = wrapper_cls(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                device=device,
                dtype=dtype,
                apply_latent_normalization=bool(cfg.get("apply_latent_normalization", False)),
            )
        if bool(cfg.get("switch_to_eval", True)):
            wrapper.eval()
        wrapper.requires_grad_(False)
        try:
            wrapper.to(device=device, dtype=dtype)
        except Exception:
            pass
        return wrapper

    def load_checkpoint(self, checkpoint_path: str, lora_checkpoint_path: Optional[str] = None) -> int:
        """Load checkpoint and return the step number"""
        try:
            if not os.path.exists(checkpoint_path):
                print(f"[TOK][Load] Checkpoint not found: {checkpoint_path}")
                return 0
            
            print(f"[TOK][Load] Loading checkpoint from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            
            # Decide upfront whether to reset EMA after all weights (incl. LoRA) are loaded
            try:
                _reset_ema_after_load = False
                train_cfg_local = getattr(self, "training_config", {}) or {}
                if isinstance(train_cfg_local, dict):
                    _reset_ema_after_load = bool(train_cfg_local.get("reset_ema_after_load", False))
                if not _reset_ema_after_load:
                    _reset_ema_after_load = bool(self.model_config.get("reset_ema_after_load", False))
            except Exception:
                _reset_ema_after_load = False
            
            # Load tokenizer patch embedding weights
            state_dict = ckpt.get("model_state_dict", {}) or {}
            tokenizer_patch = getattr(self.transformer, "tokenizer_patch_embed", None)
            if isinstance(tokenizer_patch, nn.Module):
                patch_state_dict = {k[len("tokenizer_patch_embed."):]: v for k, v in state_dict.items() if k.startswith("tokenizer_patch_embed.")}
                if patch_state_dict:
                    tokenizer_patch.load_state_dict(patch_state_dict, strict=False)
                    print(f"[TOK][Load] Loaded patch embedding weights for 'tokenizer_patch_embed'")

            # Load project_out head weights
            proj_module = self._get_proj_out_module()
            if proj_module is not None:
                proj_state_dict = {}
                for k, v in ckpt.get("model_state_dict", {}).items():
                    if k.startswith("project_out."):
                        new_key = k[len("project_out."):]
                        proj_state_dict[new_key] = v
                if proj_state_dict:
                    proj_module.load_state_dict(proj_state_dict)
                    print(f"[TOK][Load] Loaded project_out weights")

            step = int(ckpt.get("global_step", 0))
            print(f"[TOK][Load] Successfully loaded checkpoint from step {step}")
            # Load EMA shadow weights if available
            try:
                ema_state = ckpt.get("ema_state_dict", None)
                if isinstance(ema_state, dict) and getattr(self, "use_ema", False):
                    self._ema_shadow = {}
                    for name, tensor in ema_state.items():
                        try:
                            self._ema_shadow[name] = tensor.detach().to("cpu", dtype=torch.float32).clone()
                        except Exception:
                            pass
                    self._ema_backup = None
                    print(f"[TOK][Load] Loaded EMA shadow weights ({len(self._ema_shadow)})")
                else:
                    # Old checkpoints may not carry EMA; if EMA is enabled for this run,
                    # seed EMA from the just-loaded weights so updates start aligned.
                    if getattr(self, "use_ema", False):
                        try:
                            self._init_ema()
                            print("[TOK][Load] Initialized EMA from current weights (no EMA in checkpoint)")
                        except Exception:
                            pass
            except Exception:
                pass
            
            # Load LoRA checkpoint if provided
            if lora_checkpoint_path and os.path.exists(lora_checkpoint_path):
                self._load_lora_checkpoint(lora_checkpoint_path, step)
            
            # Optional: merge LoRA into base weights on resume
            try:
                _merge_lora_on_resume = False
                train_cfg_local = getattr(self, "training_config", {}) or {}
                if isinstance(train_cfg_local, dict):
                    _merge_lora_on_resume = bool(train_cfg_local.get("merge_lora_on_resume", False))
                if _merge_lora_on_resume:
                    print("[TOK][Load] merge_lora_on_resume=True, 正在尝试合并LoRA...")
                    merged = False
                    try:
                        merged = bool(self.lora_manager.merge_into_base())
                    except Exception:
                        merged = False
                    if merged:
                        print("[TOK][Load] Merged LoRA into base weights and unloaded adapters")
                    else:
                        print("[TOK][Load] merge_lora_on_resume requested but merge is not supported in this environment; keeping adapters loaded")
            except Exception:
                pass
            
            # Perform EMA reset after all weights (incl. LoRA) are fully loaded
            if _reset_ema_after_load and getattr(self, "use_ema", False):
                try:
                    self._init_ema()
                    print("[TOK][Load] Reset EMA to current model weights after LoRA load")
                except Exception:
                    pass
            
            return step
            
        except Exception as e:
            print(f"[TOK][Load] Error loading checkpoint: {e}")
            return 0

    def _load_lora_checkpoint(self, lora_checkpoint_path: str, step: int) -> None:
        """Load LoRA checkpoint using LoRA manager"""
        print(f"[TOK][Load] Loading LoRA checkpoint from: {lora_checkpoint_path}")
        
        success = self.lora_manager.load_lora_weights("default", lora_checkpoint_path, 1.0)
        if success:
            print(f"[TOK][Load] Successfully loaded LoRA checkpoint")
        else:
            print(f"[TOK][Load] Failed to load LoRA checkpoint")

    def _setup_lora(self, lora_paths: Optional[dict], lora_config: Optional[dict]):
        """Setup LoRA using the simplified LoRA manager"""
        print(f"[TOK][LoRA] 🚀 使用简化版LoRA管理器")
        
        # 添加LoRA配置
        if lora_config:
            adapter_name = lora_config.get("adapter_name", "default")
            cfg_kwargs = {k: v for k, v in lora_config.items() if k != "adapter_name"}
            weight = lora_config.get("weight", 1.0)
            
            success = self.lora_manager.add_lora(adapter_name, cfg_kwargs, weight)
            if success and os.environ.get("LOCAL_RANK", "0") == "0":
                print(f"[TOK][LoRA] ✅ 成功添加LoRA配置 '{adapter_name}'")

        # 加载LoRA权重
        if lora_paths:
            for adapter_name, path_info in lora_paths.items():
                if isinstance(path_info, (list, tuple)) and len(path_info) == 2:
                    path, weight = path_info
                else:
                    path = path_info
                    weight = 1.0
                
                success = self.lora_manager.load_lora_weights(adapter_name, path, weight)
                if success and os.environ.get("LOCAL_RANK", "0") == "0":
                    print(f"[TOK][LoRA] ✅ 成功加载LoRA权重 '{adapter_name}'")

    def configure_optimizers(self):
        # Full finetune: train all transformer params directly
        if bool(self.model_config.get("full_finetune", False)):
            params: List[torch.nn.Parameter] = [
                p for p in self.transformer.parameters() if p.requires_grad
            ]
            opt_conf = self.optimizer_config
            opt_type = opt_conf.get("type", "AdamW")
            opt_params = opt_conf.get("params", {})
            if opt_type == "AdamW":
                optimizer = torch.optim.AdamW(params, **opt_params)
            elif opt_type == "SGD":
                optimizer = torch.optim.SGD(params, **opt_params)
            else:
                raise NotImplementedError(f"Optimizer '{opt_type}' not implemented.")
            return optimizer

        # Default: Train only LoRA layers; if none, fallback to tokenizer heads
        params: List[torch.nn.Parameter] = []
        seen = set()

        # Collect LoRA parameters via simplified LoRA manager
        for p in self.lora_manager.get_all_parameters():
            if id(p) not in seen:
                params.append(p)
                seen.add(id(p))

        # Always include tokenizer-specific heads when enabled (defaults True)
        train_proj_out = bool(self.model_config.get("train_proj_out", True))
        train_pos_embed = bool(self.model_config.get("train_pos_embed", True))

        if train_proj_out:
            proj_module = self._get_proj_out_module()
            if proj_module is not None:
                for p in proj_module.parameters():
                    if id(p) not in seen:
                        p.requires_grad_(True)
                        params.append(p)
                        seen.add(id(p))

        if train_pos_embed:
            for patch_module in self._iter_patch_embed_modules():
                for p in patch_module.parameters():
                    if id(p) not in seen:
                        p.requires_grad_(True)
                        params.append(p)
                        seen.add(id(p))

        # Final safety: only fallback to proj_out when tokenizer training is enabled
        if not params:
            if bool(self.model_config.get("train_proj_out", False)):
                proj_module = self._get_proj_out_module()
                if proj_module is not None:
                    for p in proj_module.parameters():
                        if id(p) not in seen:
                            p.requires_grad_(True)
                            params.append(p)
                            seen.add(id(p))
            else:
                raise ValueError(
                    "No trainable parameters configured. Enable LoRA or set model_config.train_proj_out=True when using tokenizer wrapper."
                )

        opt_conf = self.optimizer_config
        opt_type = opt_conf.get("type", "AdamW")
        opt_params = opt_conf.get("params", {})
        if opt_type == "AdamW":
            optimizer = torch.optim.AdamW(params, **opt_params)
        elif opt_type == "SGD":
            optimizer = torch.optim.SGD(params, **opt_params)
        else:
            raise NotImplementedError(f"Optimizer '{opt_type}' not implemented.")
        return optimizer

    def _encode_prompts(self, prompts: Union[str, List[str]], device, dtype, B: int):
        prompt_embeds, _, pooled_prompt_embeds, _ = self.sd3_pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=None,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=False,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=self.model_config.get("max_sequence_length", 512),
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
        if prompt_embeds.shape[0] != B:
            prompt_embeds = prompt_embeds.expand(B, -1, -1)
        if pooled_prompt_embeds.shape[0] != B:
            pooled_prompt_embeds = pooled_prompt_embeds.expand(B, -1)
        return prompt_embeds, pooled_prompt_embeds

    def _get_random_train_prompts(self, num_prompts: int = 4) -> List[str]:
        """从训练集中随机选择prompt用于过拟合检测"""
        if not hasattr(self, '_cached_train_prompts'):
            self._cached_train_prompts = []
            self._cached_prompt_index = 0
        if len(self._cached_train_prompts) == 0 or self._cached_prompt_index >= len(self._cached_train_prompts):
            self._cached_train_prompts = self._sample_fresh_train_prompts(num_prompts * 2)
            self._cached_prompt_index = 0
        start_idx = self._cached_prompt_index
        end_idx = min(start_idx + num_prompts, len(self._cached_train_prompts))
        selected_prompts = self._cached_train_prompts[start_idx:end_idx]
        self._cached_prompt_index = end_idx
        print(f"[TOK][TrainSample] Using cached prompts {start_idx}-{end_idx}: {len(selected_prompts)} prompts")
        return selected_prompts

    def _sample_fresh_train_prompts(self, num_prompts: int = 8) -> List[str]:
        """从训练集中采样新的prompt"""
        if self.train_dataset is None:
            print(f"[TOK][TrainSample] train_dataset is None, trying to get samples from DataLoader")
            return self._get_prompts_from_dataloader(num_prompts)
        try:
            import random
            dataset_size = len(self.train_dataset)
            print(f"[TOK][TrainSample] Dataset size: {dataset_size}")
            if dataset_size == 0:
                print(f"[TOK][TrainSample] Dataset is empty")
                return []
            random_indices = random.sample(range(dataset_size), min(num_prompts, dataset_size))
            print(f"[TOK][TrainSample] Selected indices: {random_indices}")
            prompts = []
            for idx in random_indices:
                try:
                    sample = self.train_dataset[idx]
                    print(f"[TOK][TrainSample] Sample {idx} keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
                    prompt_text = sample.get("strText") or sample.get("description") or sample.get("text", "")
                    if prompt_text and isinstance(prompt_text, str):
                        prompts.append(prompt_text)
                        print(f"[TOK][TrainSample] Found prompt: {prompt_text[:50]}...")
                    else:
                        print(f"[TOK][TrainSample] No valid prompt found in sample {idx}")
                except Exception as e:
                    print(f"[TOK][TrainSample] Error getting prompt from index {idx}: {e}")
                    continue
            print(f"[TOK][TrainSample] Total prompts found: {len(prompts)}")
            return prompts
        except Exception as e:
            print(f"[TOK][TrainSample] Error sampling from training dataset: {e}")
            return []

    def _get_prompts_from_dataloader(self, num_prompts: int = 4) -> List[str]:
        """从DataLoader中顺序获取训练样本的prompt（不使用随机采样）"""
        try:
            if hasattr(self, 'trainer') and hasattr(self.trainer, 'train_dataloader'):
                dataloader = self.trainer.train_dataloader
            else:
                print(f"[TOK][TrainSample] No trainer or train_dataloader available")
                return []

            all_prompts = []
            batch_count = 0
            max_batches = min(20, num_prompts * 3)

            for batch in dataloader:
                batch_count += 1
                if batch_count > max_batches or len(all_prompts) >= num_prompts:
                    break
                if isinstance(batch, dict) and "strText" in batch:
                    batch_prompts = batch["strText"]
                    if isinstance(batch_prompts, list):
                        for prompt in batch_prompts:
                            if prompt and isinstance(prompt, str):
                                all_prompts.append(prompt)
                                if len(all_prompts) >= num_prompts:
                                    break

            selected_prompts = all_prompts[:num_prompts]
            return selected_prompts

        except Exception as e:
            print(f"[TOK][TrainSample] Error getting prompts from DataLoader: {e}")
            return []

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        device = self.device
        dtype = self.dtype

        # removed step_adjusted logging

        images = batch["image"].to(device=device, dtype=torch.float32)
        B, _, H, W = images.shape

        raw_prompts: Union[str, List[str]] = self.fixed_train_prompt or batch.get("strText") or batch.get("description") or ""
        prompts = [raw_prompts for _ in range(B)] if isinstance(raw_prompts, str) else list(raw_prompts)
        if len(prompts) < B:
            prompts = (prompts * ((B + len(prompts) - 1) // len(prompts)))[:B]

        # Apply prompt dropout
        p_drop = float(getattr(self, "prompt_dropout_prob", 0.0) or 0.0)
        if p_drop > 0.0:
            drop_mask = torch.rand(B).lt(p_drop).tolist()
            for i, do_drop in enumerate(drop_mask):
                if do_drop:
                    prompts[i] = ""

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = self._encode_prompts(prompts, device, dtype, B)
            latents = self._encode_images_to_latents(images, device=device, dtype=dtype)
            # numeric safety & stats
            latents = self._nan_to_num_(latents)
            self._log_tensor_stats("latents", latents)

            # one-time tokenizer VAE sanity check: encode->decode->stats
            if self.use_tokenizer_vae and not getattr(self, "_tokenizer_checked", False):
                try:
                    with torch.no_grad():
                        rec = self.tokenizer_vae_wrapper.decode(latents.float())
                        rec = rec.detach().to(torch.float32)
                        rec = torch.nan_to_num(rec, nan=0.0, posinf=1.0, neginf=0.0)
                        rec = rec.clamp(0.0, 1.0)
                        self._log_tensor_stats("vae_recon", rec)
                        # Save one reconstructed sample image on rank 0
                        try:
                            if str(os.environ.get("LOCAL_RANK", "0")) == "0":
                                os.makedirs(getattr(self, "save_path", "./output/tokenizer"), exist_ok=True)
                                base_step = getattr(self, '_resume_step', 0)
                                step = int(self.global_step) + base_step
                                pil_img = TF.to_pil_image(rec[0].cpu())
                                fn = os.path.join(self.save_path, f"tok_vae_recon_step_{step}.jpg")
                                pil_img.save(fn)
                                print(f"[TOK][Recon] Saved reconstructed image: {fn}")
                        except Exception:
                            pass
                        self._tokenizer_checked = True
                except Exception:
                    # do not crash if decode sanity fails
                    self._tokenizer_checked = True

        # Sample random timesteps using density-based sampling
        scheduler = self.noise_scheduler
        scheduler.config.use_dynamic_shifting = False
        scheduler.set_shift(4.0)
        # Conditionally set shift based on data.experiment.downsample_factor
        # try:
        #     # if int(getattr(self, "downsample_factor", 1)) == 4:
        #     #     scheduler.set_shift(2.0)
        # except Exception:
        #     pass
        num_train_timesteps = int(getattr(scheduler.config, "num_train_timesteps", scheduler.timesteps.shape[0]))
        scheduler.set_timesteps(num_inference_steps=num_train_timesteps, device=device)
        
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.model_config.get("weighting_scheme", "logit_normal"),
            batch_size=B,
            logit_mean=self.model_config.get("logit_mean", 0.0),
            logit_std=self.model_config.get("logit_std", 1.0),
            mode_scale=self.model_config.get("mode_scale", 1.29),
        )
        indices = (u * num_train_timesteps).long()
        indices_cpu = indices.detach().cpu()
        timesteps_tensor = scheduler.timesteps[indices].to(device=device, dtype=torch.long)
        scheduler_sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
        sigmas = scheduler_sigmas[indices_cpu].to(device=device, dtype=dtype)
        sigmas = sigmas.view(B, *([1] * (latents.ndim - 1)))
        
        noise = torch.randn_like(latents, device=device, dtype=dtype)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        noisy_latents = self._nan_to_num_(noisy_latents)
        # Residual-guided loss scheduling: compute omega_t (used in channel-wise loss weights)
        end_weight = float(getattr(self, "residual_schedule_end_weight", 1.0))
        omega = end_weight
        if getattr(self, "use_residual_schedule", True):
            try:
                base_step = int(getattr(self, "_resume_step", 0))
                cur_step = int(self.global_step) + base_step
                S_warm = max(1, int(getattr(self, "residual_warmup_steps", 2000)))
                s_clamped = cur_step if cur_step <= S_warm else S_warm
                progress = float(s_clamped) / float(S_warm)
                base_omega = 0.5 * (1.0 - math.cos(math.pi * progress))
                omega = end_weight * base_omega
                self.log("train/residual_weight", float(omega), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
            except Exception:
                omega = end_weight
    
        model_pred = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps_tensor,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        model_pred = self._nan_to_num_(model_pred)

        # Preconditioning as in SD3 example
        precondition_outputs = self.model_config.get("precondition_outputs", True)
        if precondition_outputs:
            model_pred = model_pred * (-sigmas) + noisy_latents

        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.model_config.get("weighting_scheme", "logit_normal"),
            sigmas=sigmas,
        )
        # flow matching loss with channel-wise residual mask
        target = latents if precondition_outputs else (noise - latents)
        B, C, Ht, Wt = model_pred.shape
        proj_module = self._get_proj_out_module()
        base_ch = None
        extra_ch = None
        reverse_concat = False
        try:
            base_ch = int(getattr(proj_module, "base_out_channels", None))
            extra_ch = int(getattr(proj_module, "extra_out_channels", None))
            reverse_concat = bool(getattr(proj_module, "reverse", False))
        except Exception:
            base_ch = None
            extra_ch = None
            reverse_concat = False
        # Build channel-wise weight mask [1, C, 1, 1]
        weight_mask = torch.ones(1, C, 1, 1, device=device, dtype=dtype)
        base_slice = None
        residual_slice = None
        if base_ch is not None and extra_ch is not None and base_ch > 0 and extra_ch > 0 and (base_ch + extra_ch) == C:
            if reverse_concat:
                # extras first then base
                residual_slice = slice(0, extra_ch)
                base_slice = slice(extra_ch, extra_ch + base_ch)
                weight_mask[:, residual_slice, :, :] = float(omega)
                weight_mask[:, base_slice, :, :] = 1.0
            else:
                # base first then extras
                base_slice = slice(0, base_ch)
                residual_slice = slice(base_ch, base_ch + extra_ch)
                weight_mask[:, base_slice, :, :] = 1.0
                weight_mask[:, residual_slice, :, :] = float(omega)
            denom_channels = float(base_ch) + float(omega) * float(extra_ch)
        else:
            # No branching head detected; default to uniform weighting
            denom_channels = float(C)

        sq_err = (model_pred.float() - target.float()) ** 2
        weighted_sq_err = sq_err * weight_mask
        num = (weighting.float() * weighted_sq_err).sum(dim=(1, 2, 3))
        denom = denom_channels * float(Ht * Wt)
        base_loss_per_sample = num / denom
        unweighted_loss_per_sample = torch.mean(
            ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        base_loss_per_sample = self._nan_to_num_(base_loss_per_sample)
        unweighted_loss_per_sample = self._nan_to_num_(unweighted_loss_per_sample)
        unweighted_base_loss_per_sample = None
        unweighted_residual_loss_per_sample = None
        if base_slice is not None and residual_slice is not None:
            base_sq_err = sq_err[:, base_slice, :, :].float()
            residual_sq_err = sq_err[:, residual_slice, :, :].float()
            unweighted_base_loss_per_sample = base_sq_err.mean(dim=(1, 2, 3))
            unweighted_residual_loss_per_sample = residual_sq_err.mean(dim=(1, 2, 3))
            unweighted_base_loss_per_sample = self._nan_to_num_(unweighted_base_loss_per_sample)
            unweighted_residual_loss_per_sample = self._nan_to_num_(unweighted_residual_loss_per_sample)
        final_weights = torch.ones_like(base_loss_per_sample)
        
        if self.use_min_snr_gamma:
            max_timestep = int(getattr(self.noise_scheduler.config, "num_train_timesteps", 1000))
            normalized_timesteps = timesteps_tensor.float() / max_timestep
            min_snr_weights = self.compute_min_snr_gamma_weights(normalized_timesteps)
            base_loss_per_sample = base_loss_per_sample * min_snr_weights
            final_weights = final_weights * min_snr_weights
        
        if self.mid_loss_boost:
            mid_boost_weights = self.compute_mid_loss_boost_weights(timesteps_tensor)
            base_loss_per_sample = base_loss_per_sample * mid_boost_weights
            final_weights = final_weights * mid_boost_weights
        
        if unweighted_base_loss_per_sample is not None and unweighted_residual_loss_per_sample is not None:
            base_mean = unweighted_base_loss_per_sample.mean()
            residual_mean = unweighted_residual_loss_per_sample.mean()
            if unweighted_base_loss_per_sample.numel() > 1:
                base_std = unweighted_base_loss_per_sample.std(unbiased=False)
                residual_std = unweighted_residual_loss_per_sample.std(unbiased=False)
            else:
                base_std = torch.zeros_like(base_mean)
                residual_std = torch.zeros_like(residual_mean)
            self.log(
                "train/unweighted_loss_base_mean",
                base_mean,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=False,
            )
            self.log(
                "train/unweighted_loss_residual_mean",
                residual_mean,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=False,
            )
            self.log(
                "train/unweighted_loss_base_std",
                base_std,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=False,
            )
            self.log(
                "train/unweighted_loss_residual_std",
                residual_std,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=False,
            )
        
        loss = base_loss_per_sample.mean()
        
        self._log_timestep_range_losses(
            timesteps_tensor,
            base_loss_per_sample,
            final_weights,
            weighting,
            unweighted_loss_per_sample,
            base_unweighted_loss_per_sample=unweighted_base_loss_per_sample,
            residual_unweighted_loss_per_sample=unweighted_residual_loss_per_sample,
        )
        
        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
        
        if self.use_min_snr_gamma:
            max_timestep = int(getattr(self.noise_scheduler.config, "num_train_timesteps", 1000))
            normalized_timesteps = timesteps_tensor.float() / max_timestep
            min_snr_weights = self.compute_min_snr_gamma_weights(normalized_timesteps)
            self.log("train/snr_weights_mean", min_snr_weights.mean(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
            self.log("train/timestep_mean", normalized_timesteps.mean(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
            unweighted_loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            ).mean()
            self.log("train/unweighted_loss", unweighted_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        
        if self.mid_loss_boost:
            mid_boost_weights = self.compute_mid_loss_boost_weights(timesteps_tensor)
            self.log("train/mid_boost_weights_mean", mid_boost_weights.mean(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
            self.log("train/mid_boost_weights_max", mid_boost_weights.max(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
            max_timestep = int(getattr(self.noise_scheduler.config, "num_train_timesteps", 1000))
            mid_start = max_timestep * 0.4
            mid_end = max_timestep * 0.6
            mid_mask = (timesteps_tensor >= mid_start) & (timesteps_tensor < mid_end)
            mid_count = mid_mask.sum().float()
            self.log("train/mid_timestep_count", mid_count, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        
        return loss

    def compute_snr(self, timesteps):
        timesteps_clamped = torch.clamp(timesteps, min=1e-8, max=1.0 - 1e-8)
        snr = (1.0 - timesteps_clamped) / timesteps_clamped
        return snr
    
    def compute_min_snr_gamma_weights(self, timesteps):
        if not self.use_min_snr_gamma:
            return torch.ones_like(timesteps)
        snr = self.compute_snr(timesteps)
        weights = torch.clamp(snr, max=self.min_snr_gamma)
        return weights

    def compute_mid_loss_boost_weights(self, timesteps):
        if not self.mid_loss_boost:
            return torch.ones_like(timesteps)
        max_timestep = int(getattr(self.noise_scheduler.config, "num_train_timesteps", 1000))
        mid_start = max_timestep * 0.4
        mid_end = max_timestep * 0.6
        mid_mask = (timesteps >= mid_start) & (timesteps < mid_end)
        weights = torch.ones_like(timesteps, dtype=torch.float32)
        weights[mid_mask] = self.mid_loss_weight
        return weights

    def _log_timestep_range_losses(
        self,
        timesteps,
        weighted_loss_per_sample,
        final_weights=None,
        base_weighting=None,
        unweighted_loss_per_sample=None,
        base_unweighted_loss_per_sample=None,
        residual_unweighted_loss_per_sample=None,
    ):
        max_timestep = int(getattr(self.noise_scheduler.config, "num_train_timesteps", 1000))
        ranges = [
            ("t_very_early", max_timestep * 0.95, max_timestep),
            ("t_early", max_timestep * 0.8, max_timestep * 0.95),
            ("t_mid_early", max_timestep * 0.6, max_timestep * 0.8),
            ("t_mid", max_timestep * 0.4, max_timestep * 0.6),
            ("t_mid_late", max_timestep * 0.2, max_timestep * 0.4),
            ("t_late", 0, max_timestep * 0.2),
        ]
        for range_name, t_min, t_max in ranges:
            mask = (timesteps >= t_min) & (timesteps < t_max)
            if mask.any():
                range_weighted_losses = weighted_loss_per_sample[mask]
                range_timesteps = timesteps[mask]
                sample_count = mask.sum().float()
                self.log(f"train/weighted_loss_{range_name}_mean", range_weighted_losses.mean(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                if sample_count > 1:
                    self.log(f"train/weighted_loss_{range_name}_std", range_weighted_losses.std(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                else:
                    self.log(f"train/weighted_loss_{range_name}_std", 0.0, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                if unweighted_loss_per_sample is not None:
                    range_unweighted_losses = unweighted_loss_per_sample[mask]
                    self.log(f"train/unweighted_loss_{range_name}_mean", range_unweighted_losses.mean(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                    if sample_count > 1:
                        self.log(f"train/unweighted_loss_{range_name}_std", range_unweighted_losses.std(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                    else:
                        self.log(f"train/unweighted_loss_{range_name}_std", 0.0, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                if final_weights is not None:
                    range_final_weights = final_weights[mask]
                    self.log(f"train/weights_{range_name}_mean", range_final_weights.mean(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                    self.log(f"train/weights_{range_name}_max", range_final_weights.max(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                    self.log(f"train/weights_{range_name}_min", range_final_weights.min(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                if base_weighting is not None:
                    range_base_weighting = base_weighting[mask]
                    self.log(f"train/base_weighting_{range_name}_mean", range_base_weighting.mean(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                self.log(f"train/loss_{range_name}_count", sample_count, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                self.log(f"train/timestep_{range_name}_mean", range_timesteps.float().mean(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                if base_unweighted_loss_per_sample is not None:
                    range_base_unweighted = base_unweighted_loss_per_sample[mask]
                    self.log(f"train/unweighted_base_loss_{range_name}_mean", range_base_unweighted.mean(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                    if sample_count > 1:
                        self.log(f"train/unweighted_base_loss_{range_name}_std", range_base_unweighted.std(unbiased=False), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                    else:
                        self.log(f"train/unweighted_base_loss_{range_name}_std", 0.0, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                if residual_unweighted_loss_per_sample is not None:
                    range_residual_unweighted = residual_unweighted_loss_per_sample[mask]
                    self.log(f"train/unweighted_residual_loss_{range_name}_mean", range_residual_unweighted.mean(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                    if sample_count > 1:
                        self.log(f"train/unweighted_residual_loss_{range_name}_std", range_residual_unweighted.std(unbiased=False), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                    else:
                        self.log(f"train/unweighted_residual_loss_{range_name}_std", 0.0, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
            else:
                self.log(f"train/weighted_loss_{range_name}_mean", 0.0, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                self.log(f"train/unweighted_loss_{range_name}_mean", 0.0, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                self.log(f"train/loss_{range_name}_count", 0.0, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                if base_unweighted_loss_per_sample is not None:
                    self.log(f"train/unweighted_base_loss_{range_name}_mean", 0.0, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                    self.log(f"train/unweighted_base_loss_{range_name}_std", 0.0, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                if residual_unweighted_loss_per_sample is not None:
                    self.log(f"train/unweighted_residual_loss_{range_name}_mean", 0.0, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                    self.log(f"train/unweighted_residual_loss_{range_name}_std", 0.0, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)

    def on_after_backward(self):
        """Log gradient norms of trainable parameters each step."""
        try:
            log_every = int(getattr(self, "log_every_n_steps", 1))
            upcoming_step = int(self.global_step) + 1
            if log_every > 0 and (upcoming_step % log_every) != 0:
                return
            total_sq = 0.0
            for param_name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if param.grad is None:
                    continue
                grad_tensor = param.grad.detach()
                grad_norm_l2 = grad_tensor.norm(2).item()
                total_sq += float(grad_norm_l2 ** 2)
                safe_name = param_name
                if safe_name.startswith("transformer.transformer."):
                    safe_name = "transformer." + safe_name[len("transformer.transformer."):]
                self.log(
                    f"train.grad_norm.{safe_name}",
                    grad_norm_l2,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    sync_dist=False,
                )
            if total_sq > 0.0:
                global_norm = float(total_sq) ** 0.5
                self.log(
                    "train.global_grad_norm",
                    global_norm,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    sync_dist=False,
                )

            grad_skip_threshold = float(getattr(self, "grad_skip_threshold", 0.0) or 0.0)
            skip_on_nonfinite = bool(getattr(self, "skip_on_nonfinite_grad", True))
            should_skip = False
            if skip_on_nonfinite:
                for p in self.parameters():
                    if p.requires_grad and p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            should_skip = True
                            break
            if grad_skip_threshold > 0.0:
                try:
                    if global_norm > grad_skip_threshold:
                        should_skip = True
                except Exception:
                    pass
            if should_skip:
                setattr(self, "_skip_step_due_to_grad", True)
        except Exception:
            pass

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if getattr(self, "_skip_step_due_to_grad", False):
            try:
                optimizer_closure()
            except Exception:
                pass
            try:
                self.log(
                    "train.skipped_step",
                    1.0,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    sync_dist=False,
                )
            except Exception:
                pass
            setattr(self, "_skip_step_due_to_grad", False)
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            return
        result = super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        try:
            self._ema_update()
        except Exception:
            pass
        try:
            lrs = []
            for i, group in enumerate(getattr(optimizer, "param_groups", [])):
                lr_val = float(group.get("lr", 0.0))
                lrs.append(lr_val)
                self.log(
                    f"train.lr/pg{i}",
                    lr_val,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    sync_dist=False,
                )
            if lrs:
                self.log(
                    "train.lr",
                    float(lrs[0]),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    sync_dist=False,
                )
        except Exception:
            pass
        return result

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        accumulate_grad_batches = getattr(self.trainer, 'accumulate_grad_batches', 1)
        if accumulate_grad_batches > 1:
            batch_in_accumulation = (batch_idx % accumulate_grad_batches) + 1
            is_last_batch_in_accumulation = batch_in_accumulation == accumulate_grad_batches
            if not is_last_batch_in_accumulation:
                return

        base_step = getattr(self, '_resume_step', 0)
        step = int(self.global_step) + base_step
        # Removed manual per-interval saving; rely on Lightning on_save_checkpoint

        if self.sample_interval <= 0:
            return
        base_step = getattr(self, '_resume_step', 0)
        step = int(self.global_step) + base_step
        if step == 0 or step % self.sample_interval != 0:
            return
        else:
            local_rank = os.environ.get("LOCAL_RANK", "0")
            if local_rank != "0":
                return
            try:
                os.makedirs(self.save_path, exist_ok=True)
            except Exception:
                pass
            was_training = self.training
            self.eval()
            try:
                orig_transformer = getattr(self.sd3_pipe, "transformer", None)
                unwrapped = orig_transformer.module if isinstance(orig_transformer, DDP) else orig_transformer
                if unwrapped is not None:
                    self.sd3_pipe.transformer = unwrapped
                
                self.sd3_pipe.scheduler.config.use_dynamic_shifting = False
                self.sd3_pipe.scheduler.set_shift(4.0)
                # Determine prompts for this validation sampling round
                try:
                    num_prompts = int(self.model_config.get("num_validation_samples", 4))
                except Exception:
                    num_prompts = 4
                # Always resample random prompts from training data each interval
                try:
                    prompts_to_use = self._get_random_train_prompts(num_prompts=num_prompts)
                except Exception:
                    prompts_to_use = []
                if (not prompts_to_use or len(prompts_to_use) == 0) and self.fixed_train_prompt:
                    prompts_to_use = [self.fixed_train_prompt]
                # Strip any existing sharpness tags and expand prompts with standardized sharpness values (80–100)
                try:
                    import re, random
                    def _strip_sharpness(t: str) -> str:
                        # remove occurrences like "sharpness: 92" or "sharpness: 92.5" (case-insensitive)
                        cleaned = re.sub(r"(?i)\\bsharpness\\s*:\\s*[-+]?\\d+(?:\\.\\d+)?", "", t)
                        # collapse extra whitespace
                        cleaned = re.sub(r"\\s+", " ", cleaned).strip()
                        return cleaned
                    base_prompts = [_strip_sharpness(p) for p in prompts_to_use]
                    # For each prompt, pick a random integer sharpness in [80, 100]
                    prompts_to_use = [f"{bp} sharpness: {random.randint(80, 100)}".strip() for bp in base_prompts]
                except Exception:
                    # If anything goes wrong, fall back to original prompts
                    pass
                if not prompts_to_use:
                    return
                # Build a safe, bounded-length filename prefix from prompts
                def _build_prefix(text: str, max_bytes: int = 140) -> str:
                    import hashlib, re
                    s = "_".join(text.split())
                    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
                    s = re.sub(r"_+", "_", s).strip("_")
                    b = s.encode("utf-8")
                    if len(b) <= max_bytes:
                        return s
                    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
                    keep = max(0, max_bytes - 9)
                    clipped = b[:keep]
                    # ensure valid utf-8 after clipping
                    while True:
                        try:
                            s_clip = clipped.decode("utf-8")
                            break
                        except UnicodeDecodeError:
                            if not clipped:
                                s_clip = ""
                                break
                            clipped = clipped[:-1]
                    return f"{s_clip}_{h}"

                def _save_prompt_json(text: str) -> None:
                    import hashlib, json
                    try:
                        h_full = hashlib.md5(text.encode("utf-8")).hexdigest()
                        json_path = os.path.join(self.save_path, f"{h_full}.json")
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump({"prompt": text, "hash": h_full}, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

                for i, prompt in enumerate(prompts_to_use):
                    prompt_prefix = _build_prefix(prompt)
                    _save_prompt_json(prompt)

                    sample_both = bool(self.model_config.get("sample_both_ema_and_current", False)) and bool(getattr(self, "use_ema", False))
                    if sample_both:
                        # Use the same random seed for cur vs ema for fair comparison
                        try:
                            import hashlib as _hashlib
                            seed_src = f"{prompt}|{step}|{i}"
                            seed = int(_hashlib.md5(seed_src.encode("utf-8")).hexdigest()[:8], 16)
                        except Exception:
                            seed = int(step * 1000 + i)
                        try:
                            gen_device = self.device
                        except Exception:
                            gen_device = torch.device("cpu")
                        gen_cur = torch.Generator(device=gen_device).manual_seed(seed)
                        gen_ema = torch.Generator(device=gen_device).manual_seed(seed)
                        # Current weights
                        images_cur, vae16_cur = self._pipeline_infer(
                            prompt=prompt,
                            height=self.sample_size,
                            width=self.sample_size,
                            decode_vae16=True,
                            use_ema_override=False,
                            generator=gen_cur,
                        )
                        if images_cur is not None:
                            img = images_cur[0]
                            fn = os.path.join(self.save_path, f"tok_sample_step_{step}_val_{i}_{prompt_prefix}_cur.jpg")
                            try:
                                img.save(fn)
                                print(f"[TOK][Sample] Saved validation image (cur): {fn}")
                            except Exception:
                                pass
                            if vae16_cur is not None:
                                fn_vae16 = os.path.join(self.save_path, f"tok_sample_step_{step}_val_{i}_{prompt_prefix}_cur_vae16.jpg")
                                try:
                                    vae16_cur.save(fn_vae16)
                                    print(f"[TOK][Sample] Saved validation vae16 image (cur): {fn_vae16}")
                                except Exception:
                                    pass

                        # EMA weights
                        images_ema, vae16_ema = self._pipeline_infer(
                            prompt=prompt,
                            height=self.sample_size,
                            width=self.sample_size,
                            decode_vae16=True,
                            use_ema_override=True,
                            generator=gen_ema,
                        )
                        if images_ema is not None:
                            img = images_ema[0]
                            fn = os.path.join(self.save_path, f"tok_sample_step_{step}_val_{i}_{prompt_prefix}_ema.jpg")
                            try:
                                img.save(fn)
                                print(f"[TOK][Sample] Saved validation image (ema): {fn}")
                            except Exception:
                                pass
                            if vae16_ema is not None:
                                fn_vae16 = os.path.join(self.save_path, f"tok_sample_step_{step}_val_{i}_{prompt_prefix}_ema_vae16.jpg")
                                try:
                                    vae16_ema.save(fn_vae16)
                                    print(f"[TOK][Sample] Saved validation vae16 image (ema): {fn_vae16}")
                                except Exception:
                                    pass
                    else:
                        images, vae16_pil = self._pipeline_infer(
                            prompt=prompt,
                            height=self.sample_size,
                            width=self.sample_size,
                            decode_vae16=True,
                        )
                        if images is not None:
                            img = images[0]
                            fn = os.path.join(self.save_path, f"tok_sample_step_{step}_val_{i}_{prompt_prefix}.jpg")
                            try:
                                img.save(fn)
                                print(f"[TOK][Sample] Saved validation image: {fn}")
                            except Exception:
                                pass
                            if vae16_pil is not None:
                                fn_vae16 = os.path.join(self.save_path, f"tok_sample_step_{step}_val_{i}_{prompt_prefix}_vae16.jpg")
                                try:
                                    vae16_pil.save(fn_vae16)
                                    print(f"[TOK][Sample] Saved validation vae16 image: {fn_vae16}")
                                except Exception:
                                    pass
                
                train_sample_count = int(self.model_config.get("train_sample_count", 1))
                if self.fixed_train_prompt:
                    train_prompts = [self.fixed_train_prompt][:train_sample_count]
                else:
                    train_prompts = self._get_random_train_prompts(num_prompts=train_sample_count)
                if train_prompts:
                    print(f"[TOK][Sample] Generating training images for overfitting detection (step {step})")
                    for i, prompt in enumerate(train_prompts):
                        print(f"[TOK][Sample] Training prompt {i}: {prompt}")
                        prompt_prefix = _build_prefix(prompt)
                        _save_prompt_json(prompt)
                        sample_both = bool(self.model_config.get("sample_both_ema_and_current", False)) and bool(getattr(self, "use_ema", False))
                        if sample_both:
                            # Use the same random seed for cur vs ema for fair comparison
                            try:
                                import hashlib as _hashlib
                                seed_src = f"{prompt}|{step}|train|{i}"
                                seed = int(_hashlib.md5(seed_src.encode("utf-8")).hexdigest()[:8], 16)
                            except Exception:
                                seed = int(step * 1000 + 100 + i)
                            try:
                                gen_device = self.device
                            except Exception:
                                gen_device = torch.device("cpu")
                            gen_cur = torch.Generator(device=gen_device).manual_seed(seed)
                            gen_ema = torch.Generator(device=gen_device).manual_seed(seed)
                            # Current weights
                            images_cur, vae16_cur = self._pipeline_infer(
                                prompt=prompt,
                                height=self.sample_size,
                                width=self.sample_size,
                                decode_vae16=True,
                                use_ema_override=False,
                                generator=gen_cur,
                            )
                            if images_cur is not None:
                                img = images_cur[0]
                                fn = os.path.join(self.save_path, f"tok_sample_step_{step}_train_{i}_{prompt_prefix}_cur.jpg")
                                try:
                                    img.save(fn)
                                    print(f"[TOK][Sample] Saved training image (cur): {fn}")
                                except Exception:
                                    print(f"[TOK][Sample] Failed to save training image (cur): {fn}")  
                                    pass
                                if vae16_cur is not None:
                                    fn_vae16 = os.path.join(self.save_path, f"tok_sample_step_{step}_train_{i}_{prompt_prefix}_cur_vae16.jpg")
                                    try:
                                        vae16_cur.save(fn_vae16)
                                        print(f"[TOK][Sample] Saved training vae16 image (cur): {fn_vae16}")
                                    except Exception:
                                        pass

                            # EMA weights
                            images_ema, vae16_ema = self._pipeline_infer(
                                prompt=prompt,
                                height=self.sample_size,
                                width=self.sample_size,
                                decode_vae16=True,
                                use_ema_override=True,
                                generator=gen_ema,
                            )
                            if images_ema is not None:
                                img = images_ema[0]
                                fn = os.path.join(self.save_path, f"tok_sample_step_{step}_train_{i}_{prompt_prefix}_ema.jpg")
                                try:
                                    img.save(fn)
                                    print(f"[TOK][Sample] Saved training image (ema): {fn}")
                                except Exception:
                                    print(f"[TOK][Sample] Failed to save training image (ema): {fn}")  
                                    pass
                                if vae16_ema is not None:
                                    fn_vae16 = os.path.join(self.save_path, f"tok_sample_step_{step}_train_{i}_{prompt_prefix}_ema_vae16.jpg")
                                    try:
                                        vae16_ema.save(fn_vae16)
                                        print(f"[TOK][Sample] Saved training vae16 image (ema): {fn_vae16}")
                                    except Exception:
                                        pass
                        else:
                            images, vae16_pil = self._pipeline_infer(
                                prompt=prompt,
                                height=self.sample_size,
                                width=self.sample_size,
                                decode_vae16=True,
                            )
                            if images is not None:
                                img = images[0]
                                fn = os.path.join(self.save_path, f"tok_sample_step_{step}_train_{i}_{prompt_prefix}.jpg")
                                try:
                                    img.save(fn)
                                    print(f"[TOK][Sample] Saved training image: {fn}")
                                except Exception:
                                    print(f"[TOK][Sample] Failed to save training image: {fn}")  
                                    pass
                                if vae16_pil is not None:
                                    fn_vae16 = os.path.join(self.save_path, f"tok_sample_step_{step}_train_{i}_{prompt_prefix}_vae16.jpg")
                                    try:
                                        vae16_pil.save(fn_vae16)
                                        print(f"[TOK][Sample] Saved training vae16 image: {fn_vae16}")
                                    except Exception:
                                        pass
                else:
                    print(f"[TOK][Sample] No training prompts available for overfitting detection")
            finally:
                try:
                    if orig_transformer is not None:
                        self.sd3_pipe.transformer = orig_transformer
                except Exception:
                    pass
                if was_training:
                    self.train()

    def _save_checkpoint(self, step: int) -> None:
        """Save tokenizer training state (Tokenizer PatchEmbed + ProjectOut) for resume/inference."""
        try:
            # Only global main process and dedupe per step
            try:
                if not self._is_global_main_process():
                    return
                if getattr(self, "_last_saved_tok_step", -1) == int(step):
                    return
            except Exception:
                pass
            ckpt: Dict[str, Any] = {
                "global_step": int(step),
                "timestamp": time.time(),
                "model_state_dict": {},
            }
            tokenizer_patch = getattr(self.transformer, "tokenizer_patch_embed", None)
            if isinstance(tokenizer_patch, nn.Module):
                for k, v in tokenizer_patch.state_dict().items():
                    ckpt["model_state_dict"][f"tokenizer_patch_embed.{k}"] = v

            proj_module = self._get_proj_out_module()
            if proj_module is not None:
                for k, v in proj_module.state_dict().items():
                    ckpt["model_state_dict"][f"project_out.{k}"] = v

            ckpt_dir = getattr(self, "checkpoint_dir", "./checkpoints")
            path = os.path.join(ckpt_dir, f"step_{step}.pt")
            latest = os.path.join(ckpt_dir, "latest.pt")
            try:
                if bool(self.model_config.get("save_ema", True)) and getattr(self, "use_ema", False) and self._ema_shadow:
                    ema_state: Dict[str, torch.Tensor] = {}
                    for name, tensor in self._ema_shadow.items():
                        try:
                            ema_state[name] = tensor.detach().to("cpu", dtype=torch.float32).clone()
                        except Exception:
                            pass
                    ckpt["ema_state_dict"] = ema_state
            except Exception:
                pass
            try:
                torch.save(ckpt, path)
                torch.save(ckpt, latest)
                print(f"[TOK][Save] Saved checkpoint: {path}")
            except Exception:
                pass
            try:
                self._submit_s3_upload(path)
            except Exception:
                pass
            try:
                self._submit_s3_upload(latest)
            except Exception:
                pass
        except Exception:
            pass

    def _encode_images_to_latents(
        self,
        images: torch.Tensor,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.use_tokenizer_vae and self.tokenizer_vae_wrapper is not None:
            pixel_inputs = images.to(device=self.tokenizer_vae_device, dtype=self.tokenizer_vae_dtype)
            latents = self.tokenizer_vae_wrapper.encode(pixel_inputs)
            if isinstance(latents, (tuple, list)):
                latents = latents[0]
            if not isinstance(latents, torch.Tensor):
                raise TypeError("Tokenizer VAE encode did not return a tensor")
            if latents.dim() != 4:
                raise ValueError(f"Tokenizer VAE expected 4D latents but received shape {tuple(latents.shape)}")
            latents = latents.to(device=device, dtype=dtype)
            latents = latents
            return latents

        vae = self.sd3_pipe.vae
        vae_dtype = getattr(vae, "dtype", None)
        if vae_dtype is None:
            try:
                vae_dtype = next(vae.parameters()).dtype
            except StopIteration:
                vae_dtype = dtype
        vae_in = (images * 2.0 - 1.0).to(device=device, dtype=vae_dtype)
        latent_dist = vae.encode(vae_in).latent_dist
        latents = latent_dist.sample()
        shift = getattr(vae.config, "shift_factor", 0.0)
        scale = getattr(vae.config, "scaling_factor", 1.0)
        latents = (latents - shift) * scale
        latents = latents.to(device=device, dtype=dtype)
        return latents

    def _iter_patch_embed_modules(self) -> List[nn.Module]:
        modules: List[nn.Module] = []
        tokenizer_patch = getattr(self.transformer, "tokenizer_patch_embed", None)
        if isinstance(tokenizer_patch, nn.Module):
            modules.append(tokenizer_patch)
        return modules

    def _randomly_initialize_tokenizer_patch_embed(self) -> None:
        """Randomly initialize weights of tokenizer_patch_embed and its children."""
        def _init_module(module: nn.Module) -> None:
            try:
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.InstanceNorm2d)):
                    if getattr(module, "weight", None) is not None:
                        nn.init.ones_(module.weight)
                    if getattr(module, "bias", None) is not None:
                        nn.init.zeros_(module.bias)
                else:
                    # Fallback to reset_parameters when available
                    reset = getattr(module, "reset_parameters", None)
                    if callable(reset):
                        reset()  # type: ignore[no-untyped-call]
            except Exception:
                # Never crash on init; continue best-effort
                pass

        for patch_module in self._iter_patch_embed_modules():
            try:
                patch_module.apply(_init_module)
            except Exception:
                pass

    # ----------------- helpers -----------------
    def _get_proj_out_module(self) -> Optional[nn.Module]:
        proj_module = getattr(self.transformer, "proj_out", None)
        if proj_module is None:
            proj_module = getattr(self.transformer, "project_out", None)
        return proj_module

    # ----------------- S3 helpers -----------------
    def _submit_s3_upload(self, local_path: Union[str, Path], use_sync: bool = False) -> None:
        """Submit S3 upload in background to avoid blocking training."""
        if not getattr(self, "s3_config", None):
            return
        try:
            if not self._is_global_main_process():
                return
        except Exception:
            pass
        try:
            if getattr(self, "_s3_executor", None) is not None:
                self._s3_executor.submit(self._upload_path_to_s3_mirror, local_path, use_sync)
            else:
                self._upload_path_to_s3_mirror(local_path, use_sync)
        except Exception:
            pass

    def _resolve_s3_config(self):
        s3_cfg: Dict[str, Any] = {}
        train_cfg = getattr(self, "training_config", {}) or {}
        if isinstance(train_cfg.get("s3"), dict):
            s3_cfg.update(train_cfg.get("s3"))

        enabled_flag = str(s3_cfg.get("enabled", True)).lower()
        if enabled_flag in {"false", "0", "no"}:
            return None

        bucket_name = (
            s3_cfg.get("bucket_name")
            or os.environ.get("TRAINING_S3_BUCKET")
            or os.environ.get("AWS_S3_BUCKET")
            or os.environ.get("AWS_BUCKET_NAME")
            or "nextcam-sharing"
        )

        base_path = (
            s3_cfg.get("base_path")
            or os.environ.get("TRAINING_S3_BASE_PATH")
            or "xcai/Ominicontrol"
        )

        if not bucket_name:
            return None

        bucket_name = str(bucket_name).strip()
        base_path = str(base_path).strip().strip("/") if base_path else ""
        if not base_path:
            return None

        config: Dict[str, Any] = {
            "bucket_name": bucket_name,
            "base_path": base_path,
            "use_aws_cli": str(s3_cfg.get("use_aws_cli", True)).lower() not in {"false", "0", "no"},
        }

        # Default prefix to 'runs' unless user provides something else
        if s3_cfg.get("prefix"):
            config["prefix"] = str(s3_cfg["prefix"]).strip("/")
        else:
            config["prefix"] = "runs"

        access_key = s3_cfg.get("aws_access_key_id") or os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = s3_cfg.get("aws_secret_access_key") or os.environ.get("AWS_SECRET_ACCESS_KEY")
        session_token = s3_cfg.get("aws_session_token") or os.environ.get("AWS_SESSION_TOKEN")
        region = (
            s3_cfg.get("region")
            or s3_cfg.get("aws_region")
            or os.environ.get("AWS_DEFAULT_REGION")
        )

        if access_key:
            config["aws_access_key_id"] = access_key
        if secret_key:
            config["aws_secret_access_key"] = secret_key
        if session_token:
            config["aws_session_token"] = session_token
        if region:
            config["aws_region"] = region

        return config

    # removed old artifact-type-based S3 upload helpers in favor of mirror mode

    def _upload_path_to_s3_mirror(self, local_path: Union[str, Path], use_sync: bool = False) -> None:
        """Mirror the local run directory structure exactly on S3.

        S3 key = base_path / prefix? / experiment_name? / <relative-path-from-run_dir>
        
        Args:
            local_path: Local file or directory path to upload
            use_sync: If True, use 'aws s3 sync' instead of 'aws s3 cp' for directories
        """
        if not getattr(self, "s3_config", None):
            return
        # Only global rank 0 performs uploads
        try:
            if not self._is_global_main_process():
                return
        except Exception:
            pass
        try:
            run_dir = getattr(self, "run_dir", None)
            if not run_dir:
                return
            path_obj = Path(local_path).resolve()
            if not path_obj.exists():
                return
            try:
                rel = path_obj.relative_to(Path(run_dir).resolve())
            except Exception:
                # If not under run_dir, don't upload
                return
            parts: List[str] = [self.s3_config["base_path"]]
            prefix = self.s3_config.get("prefix")
            if prefix:
                parts.append(prefix)
            experiment_name = getattr(self, "experiment_name", None)
            if experiment_name:
                parts.append(str(experiment_name))
            parts.append(rel.as_posix())
            key = "/".join(part.strip("/\\") for part in parts if part)
            bucket = self.s3_config["bucket_name"]

            uri = f"s3://{bucket}/{key}"
            if path_obj.is_file():
                # Prefer aws cli
                if self.s3_config.get("use_aws_cli", True) and shutil.which("aws"):
                    try:
                        result = subprocess.run(["aws", "s3", "cp", str(path_obj), uri], capture_output=True, text=True, timeout=300)
                        if result.returncode == 0:
                            return
                    except Exception:
                        pass
                try:
                    import boto3  # type: ignore
                    s3_client = boto3.client("s3")  # type: ignore
                    s3_client.upload_file(str(path_obj), bucket, key)
                except Exception as err:
                    pass
            else:
                # Directory upload
                if self.s3_config.get("use_aws_cli", True) and shutil.which("aws"):
                    try:
                        if use_sync:
                            # Use aws s3 sync for directories
                            result = subprocess.run(["aws", "s3", "sync", str(path_obj), uri], capture_output=True, text=True, timeout=600)
                        else:
                            # Use aws s3 cp --recursive
                            result = subprocess.run(["aws", "s3", "cp", str(path_obj), uri, "--recursive"], capture_output=True, text=True, timeout=600)
                        if result.returncode == 0:
                            return
                    except Exception:
                        pass
                try:
                    import boto3  # type: ignore
                    s3_client = boto3.client("s3")  # type: ignore
                    for file_path in path_obj.rglob("*"):
                        if file_path.is_file():
                            rel_key = file_path.relative_to(Path(run_dir)).as_posix()
                            full_key = "/".join(part.strip("/\\") for part in [self.s3_config["base_path"], self.s3_config.get("prefix", ""), getattr(self, "experiment_name", ""), rel_key] if part)
                            s3_client.upload_file(str(file_path), bucket, full_key)
                except Exception as err:
                    pass
        except Exception:
            pass

    # ----------------- EMA helpers -----------------
    def _iter_trainable_named_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name, param
    
    def reset_ema_to_model(self) -> None:
        """Reset EMA shadow weights to the current trainable model parameters."""
        try:
            if not getattr(self, "use_ema", False):
                return
            self._init_ema()
            print("[TOK][EMA] Reset EMA to current model weights")
        except Exception:
            pass

    def _init_ema(self) -> None:
        self._ema_shadow = {}
        self._ema_backup = None
        self._ema_num_updates = 0
        for name, param in self._iter_trainable_named_parameters():
            try:
                self._ema_shadow[name] = param.detach().data.to(torch.float32).clone()
            except Exception:
                pass

    def _ema_update(self) -> None:
        if not getattr(self, "use_ema", False):
            return
        if (self._ema_num_updates % max(1, int(self.ema_update_every))) != 0:
            self._ema_num_updates += 1
            return
        decay = float(self.ema_decay)
        one_minus = 1.0 - decay
        for name, param in self._iter_trainable_named_parameters():
            try:
                data_fp32 = param.detach().data.to(torch.float32)
                shadow = self._ema_shadow.get(name, None)
                if shadow is None:
                    self._ema_shadow[name] = data_fp32.clone()
                else:
                    shadow.mul_(decay).add_(data_fp32, alpha=one_minus)
            except Exception:
                pass
        self._ema_num_updates += 1

    def _ema_swap_to_shadow(self) -> None:
        if not getattr(self, "use_ema", False) or not self._ema_shadow:
            return
        if self._ema_backup is not None:
            return
        backup: Dict[str, torch.Tensor] = {}
        for name, param in self._iter_trainable_named_parameters():
            try:
                backup[name] = param.detach().data.clone()
                shadow = self._ema_shadow.get(name)
                if shadow is not None:
                    param.data.copy_(shadow.to(dtype=param.dtype, device=param.device))
            except Exception:
                pass
        self._ema_backup = backup

    def _ema_restore(self) -> None:
        if self._ema_backup is None:
            return
        for name, param in self._iter_trainable_named_parameters():
            try:
                if name in self._ema_backup:
                    param.data.copy_(self._ema_backup[name])
            except Exception:
                pass
        self._ema_backup = None


    def _save_lora_adapters(self, step: int) -> None:
        """Save LoRA adapters using simplified LoRA manager"""
        # Only global main process and dedupe per step
        try:
            if not self._is_global_main_process():
                return
            if getattr(self, "_last_saved_lora_step", -1) == int(step):
                return
        except Exception:
            pass
        base_dir = getattr(self, "lora_dir", "./lora_weights")
        save_dir = os.path.join(base_dir, f"lora_step_{step}")
        
        # print(f"[TOK][LoRA] 💾 保存LoRA到 {save_dir}")
        success = self.lora_manager.save_all(save_dir, step)
        
        if success:
            print(f"[TOK][LoRA] ✅ 成功保存LoRA到 {save_dir}")
            try:
                self._submit_s3_upload(save_dir)
            except Exception:
                pass
        else:
            print(f"[TOK][LoRA] ❌ 保存LoRA失败")
        try:
            self._last_saved_lora_step = int(step)
        except Exception:
            pass

    # ---- Distributed helpers ----
    def _is_global_main_process(self) -> bool:
        try:
            rank = os.environ.get("RANK")
            if rank is not None:
                return str(rank) == "0"
            # fallback: require node 0 and local rank 0
            node_rank = os.environ.get("NODE_RANK", "0")
            local_rank = os.environ.get("LOCAL_RANK", "0")
            return str(node_rank) == "0" and str(local_rank) == "0"
        except Exception:
            return True

    # ---- Lightning checkpoint integration ----
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:  # type: ignore[override]
        """Emit lightweight inference ckpt and LoRA when Lightning saves a checkpoint."""
        try:
            base_step = getattr(self, '_resume_step', 0)
            step = int(self.global_step) + base_step
            # Lightweight inference checkpoint
            self._save_checkpoint(step)
            # Save LoRA
            self._save_lora_adapters(step)
            # Store helpful pointers inside lightning ckpt
            checkpoint["tok_infer_ckpt"] = os.path.join(getattr(self, "checkpoint_dir", "./checkpoints"), f"step_{step}.pt")
            checkpoint["tok_lora_dir"] = os.path.join(getattr(self, "lora_dir", "./lora_weights"), f"lora_step_{step}")
            checkpoint["ema_num_updates"] = int(getattr(self, "_ema_num_updates", 0))
            # Also persist EMA shadow weights directly into Lightning checkpoint for proper resume
            try:
                if bool(self.model_config.get("save_ema", True)) and getattr(self, "use_ema", False) and self._ema_shadow:
                    ema_state: Dict[str, torch.Tensor] = {}
                    for name, tensor in self._ema_shadow.items():
                        try:
                            ema_state[name] = tensor.detach().to("cpu", dtype=torch.float32).clone()
                        except Exception:
                            pass
                    checkpoint["ema_state_dict"] = ema_state
            except Exception:
                pass
        except Exception:
            pass

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:  # type: ignore[override]
        """Restore extra runtime metadata and attempt LoRA reload from saved pointer."""
        try:
            if "ema_num_updates" in checkpoint:
                try:
                    self._ema_num_updates = int(checkpoint["ema_num_updates"])  # type: ignore[arg-type]
                except Exception:
                    pass
        except Exception:
            pass
        # Restore EMA shadow weights if present; otherwise, initialize from current params when EMA is enabled
        try:
            ema_state = checkpoint.get("ema_state_dict", None)
            if isinstance(ema_state, dict) and getattr(self, "use_ema", False):
                self._ema_shadow = {}
                for name, tensor in ema_state.items():
                    try:
                        self._ema_shadow[name] = tensor.detach().to("cpu", dtype=torch.float32).clone()
                    except Exception:
                        pass
            else:
                if getattr(self, "use_ema", False):
                    try:
                        self._init_ema()
                    except Exception:
                        pass
            # Optional: reset EMA on Lightning resume as well
            try:
                reset_flag = False
                train_cfg = getattr(self, "training_config", {}) or {}
                if isinstance(train_cfg, dict):
                    reset_flag = bool(train_cfg.get("reset_ema_after_load", False))
                if not reset_flag:
                    reset_flag = bool(self.model_config.get("reset_ema_after_load", False))
            except Exception:
                reset_flag = False
            if reset_flag and getattr(self, "use_ema", False):
                try:
                    self._init_ema()
                    print("[TOK][Resume] Reset EMA to current model weights as configured")
                except Exception:
                    pass
        except Exception:
            pass
        # Do not reload LoRA here: Lightning ckpt already contains adapter params when
        # save_weights_only=False. Avoid double-loading to prevent overrides.

    def on_fit_end(self):  # type: ignore[override]
        """Shutdown S3 upload executor without blocking training shutdown."""
        try:
            if getattr(self, "_s3_executor", None) is not None:
                self._s3_executor.shutdown(wait=False)
                self._s3_executor = None
        except Exception:
            pass


def train(dataloader_or_dataset, trainable_model: SD3TokenizerModel, config: dict, resume_from_checkpoint: Optional[str] = None, resume_from_lora_checkpoint: Optional[str] = None):
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    run_name = train_cfg.get("run_name") or os.environ.get("RUN_NAME") or time.strftime("%Y%m%d-%H%M%S")
    base_save_path = train_cfg.get("save_path", "./output")
    run_dir = os.path.join(base_save_path, run_name)
    run_dir_abs = os.path.abspath(run_dir)
    output_dir = os.path.join(run_dir_abs, "output")
    checkpoints_dir = os.path.join(run_dir_abs, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    full_ckpt_dir = os.path.join(run_dir_abs, "full_checkpoints")
    tok_ckpt_dir = os.path.join(run_dir_abs, "tokenizer_checkpoints")
    lora_dir = os.path.join(run_dir_abs, "lora_weights")
    os.makedirs(full_ckpt_dir, exist_ok=True)
    os.makedirs(tok_ckpt_dir, exist_ok=True)
    os.makedirs(lora_dir, exist_ok=True)
    checkpoints_dir = tok_ckpt_dir
    import yaml
    with open(os.path.join(run_dir_abs, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    prompts = list(data_cfg.get("validation_prompts", []))
    if prompts:
        trainable_model.sample_prompts = prompts

    trainable_model.sample_interval = int(train_cfg.get("sample_interval", getattr(trainable_model, "sample_interval", 1000)))
    trainable_model.save_path = output_dir

    trainable_model.save_interval = int(train_cfg.get("save_interval", getattr(trainable_model, "save_interval", -1)))
    trainable_model.checkpoint_dir = checkpoints_dir
    setattr(trainable_model, "lora_dir", lora_dir)

    # Set run/experiment info and S3 config
    experiment_cfg = config.get("experiment", {}) if isinstance(config, dict) else {}
    experiment_name = train_cfg.get("experiment_name") or (experiment_cfg.get("name") if isinstance(experiment_cfg, dict) else None) or run_name
    setattr(trainable_model, "run_name", run_name)
    setattr(trainable_model, "experiment_name", experiment_name)
    setattr(trainable_model, "training_config", train_cfg)
    setattr(trainable_model, "run_dir", run_dir_abs)
    try:
        trainable_model.s3_config = trainable_model._resolve_s3_config()
        if getattr(trainable_model, "s3_config", None):
            print(f"[S3] Checkpoint uploads enabled: bucket={trainable_model.s3_config['bucket_name']}, base_path={trainable_model.s3_config['base_path']}")
    except Exception:
        pass

    sample_size = int(data_cfg.get("base_size", getattr(trainable_model, "sample_size", 1024)))
    trainable_model.sample_size = sample_size

    # Expose downsample_factor from data.experiment to the model for runtime decisions
    try:
        ds_factor = int(data_cfg.get("experiment", {}).get("downsample_factor", 1))
        setattr(trainable_model, "downsample_factor", ds_factor)
    except Exception:
        pass

    trainable_model.print_every_n_steps = int(train_cfg.get("print_every_n_steps", getattr(trainable_model, "print_every_n_steps", 10)))
    trainable_model.log_every_n_steps = int(train_cfg.get("log_every_n_steps", getattr(trainable_model, "log_every_n_steps", 20)))

    if isinstance(dataloader_or_dataset, DataLoader):
        loader = dataloader_or_dataset
        train_dataset = None
    elif isinstance(dataloader_or_dataset, Dataset):
        train_dataset = dataloader_or_dataset
        loader = DataLoader(
            dataloader_or_dataset,
            batch_size=train_cfg.get("batch_size", 1),
            shuffle=True,
            num_workers=train_cfg.get("dataloader_workers", 0),
        )
    elif isinstance(dataloader_or_dataset, dict):
        raw_batch = dataloader_or_dataset

        class _RepeatBatch(IterableDataset):  # type: ignore
            def __iter__(self):  # noqa: D401
                while True:
                    yield raw_batch

        loader = DataLoader(_RepeatBatch(), batch_size=None, num_workers=0)
        train_dataset = None
    else:
        train_dataset = dataloader_or_dataset
        loader = DataLoader(
            dataloader_or_dataset,
            batch_size=train_cfg.get("batch_size", 1),
            shuffle=True,
            num_workers=train_cfg.get("dataloader_workers", 0),
        )
    
    if hasattr(trainable_model, 'train_dataset'):
        trainable_model.train_dataset = train_dataset
        if train_dataset is not None:
            print(f"[TOK][TrainSample] Set train_dataset: {type(train_dataset)} with size {len(train_dataset) if hasattr(train_dataset, '__len__') else 'unknown'}")
        else:
            print(f"[TOK][TrainSample] Using DataLoader for training samples (no direct dataset access)")
    else:
        print(f"[TOK][TrainSample] trainable_model has no train_dataset attribute")

    resume_step = 0
    auto_resume = train_cfg.get("auto_resume", True)
    use_lightning_resume = bool(train_cfg.get("use_lightning_resume", True))
    
    if not use_lightning_resume:
        if resume_from_checkpoint:
            if os.path.exists(resume_from_checkpoint):
                resume_step = trainable_model.load_checkpoint(resume_from_checkpoint, resume_from_lora_checkpoint)
            else:
                print(f"[TOK][Resume] Checkpoint not found: {resume_from_checkpoint}")
                # 即使基础checkpoint缺失，也尝试仅加载LoRA
                if resume_from_lora_checkpoint and os.path.exists(resume_from_lora_checkpoint):
                    print(f"[TOK][Resume] Base checkpoint missing; attempting LoRA-only load: {resume_from_lora_checkpoint}")
                    try:
                        trainable_model._load_lora_checkpoint(resume_from_lora_checkpoint, step=0)
                        # 可选：立即尝试合并
                        _merge_lora_on_resume = bool(train_cfg.get("merge_lora_on_resume", False))
                        if _merge_lora_on_resume:
                            try:
                                merged = bool(trainable_model.lora_manager.merge_into_base())
                            except Exception:
                                merged = False
                            if merged:
                                print("[TOK][Resume] LoRA-only load merged into base weights")
                            else:
                                print("[TOK][Resume] LoRA-only load; merge requested but not supported")
                    except Exception as e:
                        print(f"[TOK][Resume] LoRA-only load failed: {e}")
        elif auto_resume:
            latest_ckpt = os.path.join(checkpoints_dir, "latest.pt")
            if os.path.exists(latest_ckpt):
                print(f"[TOK][Resume] Auto-detected latest checkpoint: {latest_ckpt}")
                auto_lora_ckpt = None
                if resume_from_lora_checkpoint is None:
                    import glob
                    lora_ckpt_dir = os.path.join(lora_dir, "latest")
                    if os.path.exists(lora_ckpt_dir):
                        auto_lora_ckpt = lora_ckpt_dir
                        print(f"[TOK][Resume] Auto-detected latest LoRA checkpoint: {auto_lora_ckpt}")
                    else:
                        lora_pattern = os.path.join(lora_dir, "lora_step_*")
                        lora_dirs = glob.glob(lora_pattern)
                        if lora_dirs:
                            lora_dirs.sort(key=lambda x: int(x.split('_')[-1]))
                            auto_lora_ckpt = lora_dirs[-1]
                            print(f"[TOK][Resume] Auto-detected latest LoRA checkpoint: {auto_lora_ckpt}")
                resume_step = trainable_model.load_checkpoint(latest_ckpt, auto_lora_ckpt)
            else:
                print(f"[TOK][Resume] No checkpoint found, checking for LoRA-only auto resume")
                # 没有基础checkpoint时，尝试自动寻找最新LoRA并加载
                auto_lora_ckpt = None
                try:
                    lora_ckpt_dir = os.path.join(lora_dir, "latest")
                    if os.path.exists(lora_ckpt_dir):
                        auto_lora_ckpt = lora_ckpt_dir
                    else:
                        import glob
                        lora_pattern = os.path.join(lora_dir, "lora_step_*")
                        lora_dirs = glob.glob(lora_pattern)
                        if lora_dirs:
                            lora_dirs.sort(key=lambda x: int(x.split('_')[-1]))
                            auto_lora_ckpt = lora_dirs[-1]
                except Exception:
                    auto_lora_ckpt = None
                if auto_lora_ckpt and os.path.exists(auto_lora_ckpt):
                    print(f"[TOK][Resume] Auto-detected LoRA checkpoint (LoRA-only): {auto_lora_ckpt}")
                    try:
                        trainable_model._load_lora_checkpoint(auto_lora_ckpt, step=0)
                        _merge_lora_on_resume = bool(train_cfg.get("merge_lora_on_resume", False))
                        if _merge_lora_on_resume:
                            try:
                                merged = bool(trainable_model.lora_manager.merge_into_base())
                            except Exception:
                                merged = False
                            if merged:
                                print("[TOK][Resume] Auto LoRA-only merged into base weights")
                            else:
                                print("[TOK][Resume] Auto LoRA-only load; merge requested but not supported")
                    except Exception as e:
                        print(f"[TOK][Resume] Auto LoRA-only load failed: {e}")
                else:
                    print(f"[TOK][Resume] No checkpoint and no LoRA found; starting from scratch")
        else:
            print(f"[TOK][Resume] Auto-resume disabled, starting from scratch")
    else:
        # Lightning will handle resume via ckpt_path; keep internal adjustment off
        trainable_model._resume_step = 0
    
    # Control whether to reset step counter or continue from checkpoint step
    reset_step_on_resume = bool(train_cfg.get("resume_reset_step", False))
    if resume_step > 0:
        if reset_step_on_resume:
            print(f"[TOK][Resume] Loaded weights from step {resume_step}, but resetting step counter to 0 as configured")
            trainable_model._resume_step = 0
        else:
            print(f"[TOK][Resume] Resuming training from step {resume_step}")
            trainable_model._resume_step = resume_step

    if train_cfg.get("pretrain_sample", False):
        if os.environ.get("LOCAL_RANK", "0") == "0":
            try:
                was_training = trainable_model.training
                trainable_model.eval()
                val_count = int(train_cfg.get("validation_samples", 4))
                # Prefer random prompts from training data for pretrain sampling
                try:
                    prompts_list = trainable_model._get_random_train_prompts(num_prompts=val_count)
                except Exception:
                    prompts_list = []
                # Fallback to configured prompts or fixed prompt if random sampling unavailable (e.g., no trainer yet)
                if not prompts_list:
                    fallback_prompts = list(getattr(trainable_model, "sample_prompts", []))
                    if fallback_prompts:
                        prompts_list = fallback_prompts[:val_count]
                    elif getattr(trainable_model, "fixed_train_prompt", None):
                        prompts_list = [trainable_model.fixed_train_prompt]
                print(f"[TOK][Pretrain] pretrain_sample enabled: prompts={len(prompts_list)} save_dir={output_dir}")
                trainable_model.sd3_pipe.scheduler.config.use_dynamic_shifting = False
                trainable_model.sd3_pipe.scheduler.set_shift(4.0)
                for i, prompt in enumerate(prompts_list[: val_count]):
                    sample_both = bool(trainable_model.model_config.get("sample_both_ema_and_current", False)) and bool(getattr(trainable_model, "use_ema", False))
                    if sample_both:
                        # Use the same random seed for cur vs ema for fair comparison
                        try:
                            import hashlib as _hashlib
                            seed_src = f"{prompt}|pretrain|{i}"
                            seed = int(_hashlib.md5(seed_src.encode("utf-8")).hexdigest()[:8], 16)
                        except Exception:
                            seed = int(i + 12345)
                        try:
                            gen_device = trainable_model.device
                        except Exception:
                            gen_device = torch.device("cpu")
                        gen_cur = torch.Generator(device=gen_device).manual_seed(seed)
                        gen_ema = torch.Generator(device=gen_device).manual_seed(seed)
                        # Current
                        images_cur, vae16_cur = trainable_model._pipeline_infer(
                            prompt=prompt,
                            height=getattr(trainable_model, "sample_size", 1024),
                            width=getattr(trainable_model, "sample_size", 1024),
                            decode_vae16=True,
                            use_ema_override=False,
                            generator=gen_cur,
                        )
                        if images_cur is not None:
                            img = images_cur[0]
                            fn = os.path.join(output_dir, f"pretrain_{i}_cur.jpg")
                            try:
                                img.save(fn)
                                print(f"[TOK][Pretrain] saved (cur) -> {fn}")
                            except Exception:
                                print(f"[TOK][Pretrain] Failed to save training image (cur): {fn}")  
                                pass
                            if vae16_cur is not None:
                                fn_vae16 = os.path.join(output_dir, f"pretrain_{i}_cur_vae16.jpg")
                                try:
                                    vae16_cur.save(fn_vae16)
                                    print(f"[TOK][Pretrain] saved vae16 (cur) -> {fn_vae16}")
                                except Exception:
                                    pass

                        # EMA
                        images_ema, vae16_ema = trainable_model._pipeline_infer(
                            prompt=prompt,
                            height=getattr(trainable_model, "sample_size", 1024),
                            width=getattr(trainable_model, "sample_size", 1024),
                            decode_vae16=True,
                            use_ema_override=True,
                            generator=gen_ema,
                        )
                        if images_ema is not None:
                            img = images_ema[0]
                            fn = os.path.join(output_dir, f"pretrain_{i}_ema.jpg")
                            try:
                                img.save(fn)
                                print(f"[TOK][Pretrain] saved (ema) -> {fn}")
                            except Exception:
                                print(f"[TOK][Pretrain] Failed to save training image (ema): {fn}")  
                                pass
                            if vae16_ema is not None:
                                fn_vae16 = os.path.join(output_dir, f"pretrain_{i}_ema_vae16.jpg")
                                try:
                                    vae16_ema.save(fn_vae16)
                                    print(f"[TOK][Pretrain] saved vae16 (ema) -> {fn_vae16}")
                                except Exception:
                                    pass
                    else:
                        images, vae16_pil = trainable_model._pipeline_infer(
                            prompt=prompt,
                            height=getattr(trainable_model, "sample_size", 1024),
                            width=getattr(trainable_model, "sample_size", 1024),
                            decode_vae16=True,
                        )
                        if images is not None:
                            img = images[0]
                            fn = os.path.join(output_dir, f"pretrain_{i}.jpg")
                            try:
                                img.save(fn)
                                print(f"[TOK][Pretrain] saved -> {fn}")
                            except Exception:
                                print(f"[TOK][Pretrain] Failed to save training image: {fn}")  
                                pass
                            if vae16_pil is not None:
                                fn_vae16 = os.path.join(output_dir, f"pretrain_{i}_vae16.jpg")
                                try:
                                    vae16_pil.save(fn_vae16)
                                    print(f"[TOK][Pretrain] saved vae16 -> {fn_vae16}")
                                except Exception:
                                    pass
                        else:
                            print(f"[TOK][Pretrain] pipeline returned no images for prompt index {i}")
            finally:
                if was_training:
                    trainable_model.train()

    try:
        if bool(train_cfg.get("gradient_checkpointing", True)):
            try:
                trainable_model.transformer.enable_gradient_checkpointing()
                print("[TOK] Enabled gradient checkpointing on transformer")
            except Exception:
                pass
    except Exception:
        pass

    progress_bar = TQDMProgressBar(refresh_rate=train_cfg.get("progress_bar_refresh_rate", 10))
    loggers = None
    try:
        is_main_process = str(os.environ.get("LOCAL_RANK", "0")) == "0"
    except Exception:
        is_main_process = True
    if is_main_process:
        save_dir = train_cfg.get("save_path", "./output")
        run_name = train_cfg.get("run_name") or os.environ.get("RUN_NAME") or time.strftime("%Y%m%d-%H%M%S")
        csv_logger = CSVLogger(save_dir=save_dir, name=run_name)
        loggers = [csv_logger]
        wandb_cfg = train_cfg.get("wandb")
        if wandb_cfg is not None and WandbLogger is not None:
            try:
                wb = WandbLogger(project=wandb_cfg.get("project", "sd3-tokenizer"), name=run_name)
                loggers.append(wb)
            except Exception as e:
                print(f"WandB logger unavailable, using CSV only: {e}")

    trainer_kwargs = {
        "accumulate_grad_batches": train_cfg.get("accumulate_grad_batches", 1),
        "enable_checkpointing": True,
        "enable_progress_bar": True,
        "logger": loggers if loggers is not None else True,
        "log_every_n_steps": train_cfg.get("log_every_n_steps", 1),
        "max_steps": train_cfg.get("max_steps", -1),
        "max_epochs": train_cfg.get("max_epochs", -1),
        "gradient_clip_val": train_cfg.get("gradient_clip_val", 0.5),
        "callbacks": [progress_bar],
        "strategy": DDPStrategy(
            find_unused_parameters=True,
            process_group_backend="gloo",
            gradient_as_bucket_view=True,
        ),
    }

    # Configure Lightning ModelCheckpoint
    save_iv = int(train_cfg.get("save_interval", getattr(trainable_model, "save_interval", -1)))
    mc = ModelCheckpoint(
        dirpath=full_ckpt_dir,
        filename="{step}",
        auto_insert_metric_name=False,
        save_top_k=-1,
        save_last=True,
        save_weights_only=False,
        every_n_train_steps=save_iv if save_iv and save_iv > 0 else None,
        save_on_train_epoch_end=False,
    )
    trainer_kwargs["callbacks"].append(mc)

    class S3SyncLightningCheckpointCallback(Callback):  # type: ignore
        def __init__(self, target_dir: str):
            self.target_dir = target_dir
            self._last_sync_step: int = -1
            self._pending_sync_step: int = -1

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):  # type: ignore[override]
            # Only global rank 0 should mark for sync
            rank = os.environ.get("RANK")
            if rank is not None and str(rank) != "0":
                return
            if rank is None and str(os.environ.get("LOCAL_RANK", "0")) != "0":
                return

            mc_local = next((cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)), None)
            if mc_local is None:
                return

            current_step = int(trainer.global_step)
            if current_step != self._last_sync_step:
                # Mark this step for sync in the next training batch
                self._pending_sync_step = current_step
                print(f"[S3] Checkpoint saved at step {current_step}, will sync after next training step")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # type: ignore[override]
            # Only global rank 0 should upload
            rank = os.environ.get("RANK")
            if rank is not None and str(rank) != "0":
                return
            if rank is None and str(os.environ.get("LOCAL_RANK", "0")) != "0":
                return

            # Check if there's a pending sync
            if self._pending_sync_step > 0 and self._pending_sync_step != self._last_sync_step:
                # Checkpoint should be fully written to disk by now
                try:
                    if os.path.exists(self.target_dir):
                        pl_module._submit_s3_upload(self.target_dir, use_sync=True)
                        # Also upload generated images under output directory
                        output_dir = getattr(pl_module, "save_path", None)
                        if output_dir and os.path.exists(output_dir):
                            try:
                                pl_module._submit_s3_upload(output_dir, use_sync=True)
                                print(f"[S3] Synced output directory to S3 (step {self._pending_sync_step})")
                            except Exception as e:
                                print(f"[S3] Failed to sync output directory: {e}")
                        self._last_sync_step = self._pending_sync_step
                        self._pending_sync_step = -1
                        print(f"[S3] Synced full_checkpoints directory to S3 (checkpoint from step {self._last_sync_step})")
                except Exception as e:
                    print(f"[S3] Failed to sync full_checkpoints: {e}")
                    self._pending_sync_step = -1

    trainer_kwargs["callbacks"].append(S3SyncLightningCheckpointCallback(full_ckpt_dir))
    
    # Ensure Lightning's view of devices*num_nodes matches torch elastic WORLD_SIZE
    try:
        env_num_nodes = int(os.environ.get("NNODES", "1"))
    except Exception:
        env_num_nodes = 1
    try:
        # Prefer explicit env; fallback to detected GPU count on the current node
        env_gpus_per_node = int(os.environ.get("GPUS_PER_NODE", str(torch.cuda.device_count() or 1)))
    except Exception:
        env_gpus_per_node = torch.cuda.device_count() or 1

    trainer_kwargs.update({
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": env_gpus_per_node,
        "num_nodes": env_num_nodes,
    })

    if (not use_lightning_resume) and (resume_step > 0) and (not reset_step_on_resume):
        trainer_kwargs["min_steps"] = resume_step + 1
    
    trainer = L.Trainer(**trainer_kwargs)
    if use_lightning_resume and auto_resume:
        trainer.fit(trainable_model, loader, ckpt_path="last")
    else:
        trainer.fit(trainable_model, loader)


