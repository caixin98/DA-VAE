from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Tuple
import os
import math
import time
import sys
import importlib

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
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from peft import LoraConfig
from peft import get_peft_model_state_dict  # type: ignore

try:
    from ..pipeline.sd3_transformer_wrapper_hr import create_sd3_transformer_wrapper_hr
except ImportError:
    create_sd3_transformer_wrapper_hr = None
from ..pipeline.sd3_transformer_wrapper_tokenizer import create_sd3_transformer_wrapper_tokenizer
from .simple_peft_lora_manager import SimplePEFTLoRAManager
sd3_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if sd3_root not in sys.path:
    sys.path.insert(0, sd3_root)
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



def _as_float(v: Any, d: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


class SD3HRModel(L.LightningModule):
    """
    Minimal HR training module:
    - Replaces transformer's pos_embed/proj_out for HR with trained VAE patch embedding and scaled head
    - Uses standard SD3 forward path; trains only LoRA by default (per adapter names via external config)
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
        self.use_tokenizer_vae: bool = bool(
            self.model_config.get("use_tokenizer_vae", tokenizer_flag if tokenizer_flag is not None else False)
        )
        self.tokenizer_vae_wrapper: Optional[nn.Module] = None
        self.tokenizer_latent_channels: Optional[int] = None
        self.tokenizer_vae_device = device
        self.tokenizer_vae_dtype = dtype

        # Freeze encoders and VAE
        self.sd3_pipe.text_encoder.requires_grad_(False).eval().to(dtype=dtype)
        self.sd3_pipe.text_encoder_2.requires_grad_(False).eval().to(dtype=dtype)
        self.sd3_pipe.vae.requires_grad_(False).eval().to(dtype=dtype)

        # Optionally wrap transformer for HR; prefer YAML flag model.use_hr_transformer, fallback to env USE_HR_TRANSFORMER
        use_hr_cfg = self.model_config.get("use_hr_transformer", None)
        if use_hr_cfg is not None:
            use_hr = str(use_hr_cfg).lower() in {"1", "true", "yes", "on"}
        else:
            use_hr = str(os.environ.get("USE_HR_TRANSFORMER", "1")).lower() in {"1", "true", "yes", "on"}

        if self.use_tokenizer_vae:
            self.tokenizer_vae_wrapper = self._init_tokenizer_vae(tokenizer_cfg, device=device, dtype=dtype)
            self.tokenizer_vae_device = getattr(self.tokenizer_vae_wrapper, "device", device)
            self.tokenizer_vae_dtype = getattr(self.tokenizer_vae_wrapper, "dtype", dtype)

            self.tokenizer_latent_channels = self.tokenizer_vae_wrapper.latent_channels
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
            use_hr = False

        elif use_hr:
            if create_sd3_transformer_wrapper_hr is None:
                raise ImportError(
                    "HR transformer wrapper is not available. "
                    "Set model.use_hr_transformer=false or provide sd3_transformer_wrapper_hr module."
                )
            self.transformer = create_sd3_transformer_wrapper_hr(
                transformer=self.sd3_pipe.transformer,
                patch_embed_cfg=patch_embed_cfg,
                patch_embed_weights_path=patch_embed_weights_path,
                downsample_factor=int(patch_embed_cfg.get("downsample_factor", 2)),
                device=device,
                dtype=dtype,
            )
            self.sd3_pipe.transformer = self.transformer
        else:
            self.transformer = self.sd3_pipe.transformer

        # Freeze all base transformer params; LoRA layers will be re-enabled below
        self.transformer.requires_grad_(False)

        # Optional: full finetune mode (disable LoRA, unfreeze all transformer params)
        self.full_finetune: bool = bool(self.model_config.get("full_finetune", False))
        if self.full_finetune:
            self.transformer.requires_grad_(True)

        # Auto-enable HR training heads by default when HR is in use, unless explicitly overridden in YAML
        if "hr_train_pos_embed" not in self.model_config:
            self.model_config["hr_train_pos_embed"] = bool(use_hr or self.use_tokenizer_vae)
        if "hr_train_proj_out" not in self.model_config:
            self.model_config["hr_train_proj_out"] = bool(use_hr or self.use_tokenizer_vae)
        # Hard guard: if neither HR nor tokenizer wrapper is enabled, never train these heads
        if not (use_hr or self.use_tokenizer_vae):
            self.model_config["hr_train_pos_embed"] = False
            self.model_config["hr_train_proj_out"] = False

        # Initialize scheduler clone
        self.noise_scheduler = self.sd3_pipe.scheduler.__class__.from_config(self.sd3_pipe.scheduler.config)

        # LoRA setup (optional): add adapters / load existing (skip when full finetune)
        self.lora_manager = SimplePEFTLoRAManager(self.sd3_pipe, self.transformer)
        if not self.full_finetune:
            self._setup_lora(lora_paths=lora_paths, lora_config=lora_config)

        # Ensure at least one module requires grad before DDP wraps the model
        # When LoRA is disabled, explicitly unfreeze HR heads if configured
        train_proj_out = bool(self.model_config.get("hr_train_proj_out", True))
        train_pos_embed = bool(self.model_config.get("hr_train_pos_embed", True))

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

        # Final fallback: only when HR is enabled; do NOT auto-enable when use_hr is False
        if not any_enabled and (use_hr or self.use_tokenizer_vae):
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
        self.save_path: str = str(self.model_config.get("save_path", "./output/hr"))
        self.print_every_n_steps: int = int(self.model_config.get("print_every_n_steps", 10))
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
        # Parity check removed
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

    @torch.no_grad()
    def _pipeline_infer(
        self,
        prompt: str,
        height: int,
        width: int,
        decode_vae16: bool = True,
    ) -> Tuple[List["Image.Image"], Optional["Image.Image"]]:
        """
        Run SD3 pipeline for a prompt and optionally decode first 16 channels via SD3 VAE.
        Returns (images, vae16_image). images is a list of PIL images.
        """
        _did_swap_to_ema = False
        if getattr(self, "use_ema", False) and getattr(self, "use_ema_for_sampling", False):
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
                    output_type="latent",
                ).images
                if decode_vae16:
                    decoded = self.tokenizer_vae_wrapper.decode(latents, return_lq_image=True)
                    if isinstance(decoded, (tuple, list)) and len(decoded) == 2:
                        images_tensor, lq_tensor = decoded
                    else:
                        images_tensor, lq_tensor = decoded, None
                else:
                    images_tensor = self.tokenizer_vae_wrapper.decode(latents)
                    lq_tensor = None

                images_tensor = images_tensor.detach().cpu().to(torch.float32).clamp(0.0, 1.0)
                pil_img = TF.to_pil_image(images_tensor.squeeze(0))
                images: List["Image.Image"] = [pil_img]

                if decode_vae16 and lq_tensor is not None:
                    try:
                        lq_tensor = lq_tensor.detach().cpu().to(torch.float32).clamp(0.0, 1.0)
                        vae16_pil = TF.to_pil_image(lq_tensor.squeeze(0))
                    except Exception:
                        vae16_pil = None
                return images, vae16_pil
            else:
                images = self.sd3_pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=int(self.model_config.get("num_inference_steps", 28)),
                    guidance_scale=_as_float(self.model_config.get("guidance_scale", 3.5), 3.5),
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

        class_path = cfg.get("wrapper_class", "modeling.sd3_tokenizer_2d_wrapper.SD3Tokenizer2DWrapper")
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

        wrapper = wrapper_cls(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            auto_decode_to_pixel=bool(cfg.get("auto_decode_to_pixel", False)),
        )
        if bool(cfg.get("switch_to_eval", True)):
            wrapper.eval()
        wrapper.requires_grad_(False)
        return wrapper

    def load_checkpoint(self, checkpoint_path: str, lora_checkpoint_path: Optional[str] = None) -> int:
        """Load checkpoint and return the step number"""
        try:
            if not os.path.exists(checkpoint_path):
                print(f"[HR][Load] Checkpoint not found: {checkpoint_path}")
                return 0
            
            print(f"[HR][Load] Loading checkpoint from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            
            # Load HR patch embedding weights
            state_dict = ckpt.get("model_state_dict", {}) or {}
            patch_targets = []
            hr_patch = getattr(self.transformer, "hr_patch_embed", None)
            if hr_patch is not None and hasattr(hr_patch, "patch_embed"):
                patch_targets.append(("hr_patch_embed.patch_embed.", hr_patch.patch_embed))
            tokenizer_patch = getattr(self.transformer, "tokenizer_patch_embed", None)
            if isinstance(tokenizer_patch, nn.Module):
                patch_targets.append(("tokenizer_patch_embed.", tokenizer_patch))

            for prefix, module in patch_targets:
                patch_state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
                if patch_state_dict:
                    module.load_state_dict(patch_state_dict, strict=False)
                    print(f"[HR][Load] Loaded patch embedding weights for '{prefix}'")

            # Load HR project_out head weights
            proj_module = self._get_proj_out_module()
            if proj_module is not None:
                proj_state_dict = {}
                for k, v in ckpt.get("model_state_dict", {}).items():
                    if k.startswith("project_out."):
                        new_key = k[len("project_out."):]
                        proj_state_dict[new_key] = v
                if proj_state_dict:
                    proj_module.load_state_dict(proj_state_dict)
                    print(f"[HR][Load] Loaded HR project_out weights")

            step = int(ckpt.get("global_step", 0))
            print(f"[HR][Load] Successfully loaded checkpoint from step {step}")
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
                    print(f"[HR][Load] Loaded EMA shadow weights ({len(self._ema_shadow)})")
            except Exception:
                pass
            
            # Load LoRA checkpoint if provided
            if lora_checkpoint_path and os.path.exists(lora_checkpoint_path):
                self._load_lora_checkpoint(lora_checkpoint_path, step)
            
            return step
            
        except Exception as e:
            print(f"[HR][Load] Error loading checkpoint: {e}")
            return 0

    def _load_lora_checkpoint(self, lora_checkpoint_path: str, step: int) -> None:
        """Load LoRA checkpoint using LoRA manager"""
        print(f"[HR][Load] Loading LoRA checkpoint from: {lora_checkpoint_path}")
        
        # 使用LoRA管理器加载检查点
        success = self.lora_manager.load_lora_weights("default", lora_checkpoint_path, 1.0)
        if success:
            print(f"[HR][Load] Successfully loaded LoRA checkpoint")
        else:
            print(f"[HR][Load] Failed to load LoRA checkpoint")

    def _setup_lora(self, lora_paths: Optional[dict], lora_config: Optional[dict]):
        """Setup LoRA using the simplified LoRA manager"""
        print(f"[HR][LoRA] 🚀 使用简化版LoRA管理器")
        
        # 添加LoRA配置
        if lora_config:
            adapter_name = lora_config.get("adapter_name", "default")
            cfg_kwargs = {k: v for k, v in lora_config.items() if k != "adapter_name"}
            weight = lora_config.get("weight", 1.0)
            
            success = self.lora_manager.add_lora(adapter_name, cfg_kwargs, weight)
            if success and os.environ.get("LOCAL_RANK", "0") == "0":
                print(f"[HR][LoRA] ✅ 成功添加LoRA配置 '{adapter_name}'")

        # 加载LoRA权重
        if lora_paths:
            for adapter_name, path_info in lora_paths.items():
                # 从path_info中提取权重
                if isinstance(path_info, (list, tuple)) and len(path_info) == 2:
                    path, weight = path_info
                else:
                    path = path_info
                    weight = 1.0
                
                success = self.lora_manager.load_lora_weights(adapter_name, path, weight)
                if success and os.environ.get("LOCAL_RANK", "0") == "0":
                    print(f"[HR][LoRA] ✅ 成功加载LoRA权重 '{adapter_name}'")
        
        # # 打印状态
        # if os.environ.get("LOCAL_RANK", "0") == "0":
        #     self.lora_manager.print_status()


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

        # Default: Train only LoRA layers; if none, fallback to HR heads
        params: List[torch.nn.Parameter] = []
        seen = set()

        # Collect LoRA parameters via simplified LoRA manager
        for p in self.lora_manager.get_all_parameters():
            if id(p) not in seen:
                params.append(p)
                seen.add(id(p))

        # Always include HR-specific heads when enabled (defaults True)
        train_proj_out = bool(self.model_config.get("hr_train_proj_out", True))
        train_pos_embed = bool(self.model_config.get("hr_train_pos_embed", True))

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

        # Final safety: only fallback to proj_out when HR training is enabled
        if not params:
            if bool(self.model_config.get("hr_train_proj_out", False)):
                proj_module = self._get_proj_out_module()
                if proj_module is not None:
                    for p in proj_module.parameters():
                        if id(p) not in seen:
                            p.requires_grad_(True)
                            params.append(p)
                            seen.add(id(p))
            else:
                raise ValueError(
                    "No trainable parameters configured. Enable LoRA or set model_config.hr_train_proj_out=True when using HR wrapper."
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
        # 添加缓存机制，避免重复采样相同的prompt
        if not hasattr(self, '_cached_train_prompts'):
            self._cached_train_prompts = []
            self._cached_prompt_index = 0
        
        # 如果缓存为空或不够，重新采样
        if len(self._cached_train_prompts) == 0 or self._cached_prompt_index >= len(self._cached_train_prompts):
            self._cached_train_prompts = self._sample_fresh_train_prompts(num_prompts * 2)  # 采样更多作为缓存
            self._cached_prompt_index = 0
        
        # 从缓存中取prompt
        start_idx = self._cached_prompt_index
        end_idx = min(start_idx + num_prompts, len(self._cached_train_prompts))
        selected_prompts = self._cached_train_prompts[start_idx:end_idx]
        self._cached_prompt_index = end_idx
        
        print(f"[HR][TrainSample] Using cached prompts {start_idx}-{end_idx}: {len(selected_prompts)} prompts")
        return selected_prompts

    def _sample_fresh_train_prompts(self, num_prompts: int = 8) -> List[str]:
        """从训练集中采样新的prompt"""
        if self.train_dataset is None:
            print(f"[HR][TrainSample] train_dataset is None, trying to get samples from DataLoader")
            return self._get_prompts_from_dataloader(num_prompts)
        
        try:
            import random
            # 获取训练集大小
            dataset_size = len(self.train_dataset)
            print(f"[HR][TrainSample] Dataset size: {dataset_size}")
            if dataset_size == 0:
                print(f"[HR][TrainSample] Dataset is empty")
                return []
            
            # 随机选择索引
            random_indices = random.sample(range(dataset_size), min(num_prompts, dataset_size))
            print(f"[HR][TrainSample] Selected indices: {random_indices}")
            
            prompts = []
            for idx in random_indices:
                try:
                    # 获取训练集样本
                    sample = self.train_dataset[idx]
                    print(f"[HR][TrainSample] Sample {idx} keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
                    # 提取prompt文本
                    prompt_text = sample.get("strText") or sample.get("description") or sample.get("text", "")
                    if prompt_text and isinstance(prompt_text, str):
                        prompts.append(prompt_text)
                        print(f"[HR][TrainSample] Found prompt: {prompt_text[:50]}...")
                    else:
                        print(f"[HR][TrainSample] No valid prompt found in sample {idx}")
                except Exception as e:
                    print(f"[HR][TrainSample] Error getting prompt from index {idx}: {e}")
                    continue
            
            print(f"[HR][TrainSample] Total prompts found: {len(prompts)}")
            return prompts
        except Exception as e:
            print(f"[HR][TrainSample] Error sampling from training dataset: {e}")
            return []

    def _get_prompts_from_dataloader(self, num_prompts: int = 4) -> List[str]:
        """从DataLoader中顺序获取训练样本的prompt（不使用随机采样）"""
        try:
            # 获取当前训练器的DataLoader
            if hasattr(self, 'trainer') and hasattr(self.trainer, 'train_dataloader'):
                dataloader = self.trainer.train_dataloader
            else:
                print(f"[HR][TrainSample] No trainer or train_dataloader available")
                return []

            all_prompts = []
            batch_count = 0
            # 只取前max_batches批
            max_batches = min(20, num_prompts * 3)  # 可以适当减少覆盖范围

            for batch in dataloader:
                batch_count += 1
                if batch_count > max_batches or len(all_prompts) >= num_prompts:
                    break

                # 从batch中提取prompt
                if isinstance(batch, dict) and "strText" in batch:
                    batch_prompts = batch["strText"]
                    if isinstance(batch_prompts, list):
                        for prompt in batch_prompts:
                            if prompt and isinstance(prompt, str):
                                all_prompts.append(prompt)
                                # print(f"[HR][TrainSample] Found prompt from DataLoader: {prompt[:50]}...")
                                if len(all_prompts) >= num_prompts:
                                    break

            selected_prompts = all_prompts[:num_prompts]
            # print(f"[HR][TrainSample] Total prompts found from DataLoader: {len(selected_prompts)}")
            return selected_prompts

        except Exception as e:
            print(f"[HR][TrainSample] Error getting prompts from DataLoader: {e}")
            return []

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        device = self.device
        dtype = self.dtype

        # Handle resume step offset
        base_step = getattr(self, '_resume_step', 0)
        step = int(self.global_step) + base_step

        images = batch["image"].to(device=device, dtype=torch.float32)
        B, _, H, W = images.shape

        # Use fixed prompt if provided; otherwise use batch text
        raw_prompts: Union[str, List[str]] = self.fixed_train_prompt or batch.get("strText") or batch.get("description") or ""
        prompts = [raw_prompts for _ in range(B)] if isinstance(raw_prompts, str) else list(raw_prompts)
        if len(prompts) < B:
            prompts = (prompts * ((B + len(prompts) - 1) // len(prompts)))[:B]



        # Apply prompt dropout with probability p per sample by replacing with empty string
        p_drop = float(getattr(self, "prompt_dropout_prob", 0.0) or 0.0)
        if p_drop > 0.0:
            drop_mask = torch.rand(B).lt(p_drop).tolist()
            for i, do_drop in enumerate(drop_mask):
                if do_drop:
                    prompts[i] = ""

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = self._encode_prompts(prompts, device, dtype, B)
            latents = self._encode_images_to_latents(images, device=device, dtype=dtype)

        # Sample random timesteps using density-based sampling
        scheduler = self.noise_scheduler
        scheduler.config.use_dynamic_shifting = False
        scheduler.set_shift(2.0)
        num_train_timesteps = int(getattr(scheduler.config, "num_train_timesteps", scheduler.timesteps.shape[0]))
        scheduler.set_timesteps(num_inference_steps=num_train_timesteps, device=device)
        
        # for weighting schemes where we sample timesteps non-uniformly
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
        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        # Get sigmas for the sampled timesteps
        scheduler_sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
        sigmas = scheduler_sigmas[indices_cpu].to(device=device, dtype=dtype)
        sigmas = sigmas.view(B, *([1] * (latents.ndim - 1)))
        
        noise = torch.randn_like(latents, device=device, dtype=dtype)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
    

        # Native transformer forward
        model_pred = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps_tensor,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

        # Preconditioning as in SD3 example: move prediction towards clean target
        precondition_outputs = self.model_config.get("precondition_outputs", True)
        if precondition_outputs:
            model_pred = model_pred * (-sigmas) + noisy_latents

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.model_config.get("weighting_scheme", "logit_normal"),
            sigmas=sigmas,
        )
        # flow matching loss
        if precondition_outputs:
            target = latents
        else:
            target = noise - latents

        # Compute regular loss (基础loss，只应用diffusers的weighting)
        base_loss_per_sample = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        
        # 保存未加权的原始loss (只应用diffusers weighting，不应用Min-SNR-γ和Mid Loss权重)
        unweighted_loss_per_sample = torch.mean(
            ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        
        # 初始化最终权重
        final_weights = torch.ones_like(base_loss_per_sample)
        
        # 应用 Min-SNR-γ 权重（如果启用）
        if self.use_min_snr_gamma:
            # 将scheduler timesteps转换为[0,1]范围的归一化值
            max_timestep = int(getattr(self.noise_scheduler.config, "num_train_timesteps", 1000))
            normalized_timesteps = timesteps_tensor.float() / max_timestep
            min_snr_weights = self.compute_min_snr_gamma_weights(normalized_timesteps)
            base_loss_per_sample = base_loss_per_sample * min_snr_weights
            final_weights = final_weights * min_snr_weights
        
        # 应用 Mid Loss 增强权重（如果启用）
        if self.mid_loss_boost:
            mid_boost_weights = self.compute_mid_loss_boost_weights(timesteps_tensor)
            base_loss_per_sample = base_loss_per_sample * mid_boost_weights
            final_weights = final_weights * mid_boost_weights
        
        loss = base_loss_per_sample.mean()
        
        # Log timestep range losses for detailed analysis
        # 传入加权loss、未加权loss、最终权重和基础weighting
        self._log_timestep_range_losses(timesteps_tensor, base_loss_per_sample, final_weights, weighting, unweighted_loss_per_sample)
        
        # Log loss to progress bar every step; throttle logger (e.g., W&B/CSV) to every N optimizer steps
        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
        
        # 如果启用了 Min-SNR-γ 策略，记录相关指标
        if self.use_min_snr_gamma:
            max_timestep = int(getattr(self.noise_scheduler.config, "num_train_timesteps", 1000))
            normalized_timesteps = timesteps_tensor.float() / max_timestep
            min_snr_weights = self.compute_min_snr_gamma_weights(normalized_timesteps)
            self.log("train/snr_weights_mean", min_snr_weights.mean(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
            
            self.log("train/timestep_mean", normalized_timesteps.mean(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
            
            # 记录未加权的损失用于对比
            unweighted_loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            ).mean()
            self.log("train/unweighted_loss", unweighted_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        
        # 如果启用了 Mid Loss 增强策略，记录相关指标
        if self.mid_loss_boost:
            mid_boost_weights = self.compute_mid_loss_boost_weights(timesteps_tensor)
            self.log("train/mid_boost_weights_mean", mid_boost_weights.mean(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
            self.log("train/mid_boost_weights_max", mid_boost_weights.max(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
            
            # 统计mid timestep的样本数量
            max_timestep = int(getattr(self.noise_scheduler.config, "num_train_timesteps", 1000))
            mid_start = max_timestep * 0.4
            mid_end = max_timestep * 0.6
            mid_mask = (timesteps_tensor >= mid_start) & (timesteps_tensor < mid_end)
            mid_count = mid_mask.sum().float()
            self.log("train/mid_timestep_count", mid_count, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        
        return loss

    def compute_snr(self, timesteps):
        """
        计算给定时间步的信噪比 (SNR)
        
        Args:
            timesteps: 时间步 tensor, shape [B] (归一化到[0,1]范围)
            
        Returns:
            snr: 信噪比 tensor, shape [B]
        """
        # 对于扩散模型，SNR通常定义为 alpha_t^2 / sigma_t^2
        # 这里我们使用简化的计算方式：SNR(t) = (1-t) / t
        # 当 t 接近 0 时，SNR 很大（信号强，低噪声）
        # 当 t 接近 1 时，SNR 很小（噪声强，高噪声）
        
        # 避免除零错误
        timesteps_clamped = torch.clamp(timesteps, min=1e-8, max=1.0 - 1e-8)
        
        # 计算 SNR = (1-t) / t
        snr = (1.0 - timesteps_clamped) / timesteps_clamped
        
        return snr
    
    def compute_min_snr_gamma_weights(self, timesteps):
        """
        计算 Min-SNR-γ 损失权重
        
        Args:
            timesteps: 时间步 tensor, shape [B] (归一化到[0,1]范围)
            
        Returns:
            weights: 损失权重 tensor, shape [B]
        """
        if not self.use_min_snr_gamma:
            # 如果未启用 Min-SNR-γ 策略，返回全1权重
            return torch.ones_like(timesteps)
        
        # 计算 SNR
        snr = self.compute_snr(timesteps)
        
        # 应用 Min-SNR-γ 策略：weight = min(SNR, γ)
        weights = torch.clamp(snr, max=self.min_snr_gamma)
        
        return weights

    def compute_mid_loss_boost_weights(self, timesteps):
        """
        计算Mid Loss增强权重
        
        Args:
            timesteps: 时间步 tensor, shape [B] (scheduler的timesteps)
            
        Returns:
            weights: 损失权重 tensor, shape [B]
        """
        if not self.mid_loss_boost:
            return torch.ones_like(timesteps)
        
        # 获取scheduler的最大timestep
        max_timestep = int(getattr(self.noise_scheduler.config, "num_train_timesteps", 1000))
        
        # 定义mid timestep范围 (0.4-0.6)
        mid_start = max_timestep * 0.4
        mid_end = max_timestep * 0.6
        
        # 创建mid timestep的mask
        mid_mask = (timesteps >= mid_start) & (timesteps < mid_end)
        
        # 创建权重tensor
        weights = torch.ones_like(timesteps, dtype=torch.float32)
        weights[mid_mask] = self.mid_loss_weight
        
        return weights

    def _log_timestep_range_losses(self, timesteps, weighted_loss_per_sample, final_weights=None, base_weighting=None, unweighted_loss_per_sample=None):
        """
        按timestep范围记录loss统计
        
        Args:
            timesteps: 时间步 tensor, shape [B] (scheduler的timesteps)
            weighted_loss_per_sample: 每个样本的loss, shape [B] (已经应用了所有权重)
            final_weights: 最终权重 tensor, shape [B] (包含Min-SNR-γ和Mid Loss权重)
            base_weighting: 基础权重 tensor, shape [B] (diffusers的原始weighting)
            unweighted_loss_per_sample: 未加权的原始loss, shape [B] (只应用diffusers weighting)
        """
        # 定义timestep范围
        # 获取scheduler的最大timestep
        max_timestep = int(getattr(self.noise_scheduler.config, "num_train_timesteps", 1000))
        # 将范围映射到scheduler的timesteps
        ranges = [
            ("t_very_early", max_timestep * 0.95, max_timestep), # 非常早期timestep (高噪声，接近纯噪声)
            ("t_early", max_timestep * 0.8, max_timestep * 0.95),      # 早期timestep (高噪声，接近纯噪声)
            ("t_mid_early", max_timestep * 0.6, max_timestep * 0.8),  # 中早期timestep
            ("t_mid", max_timestep * 0.4, max_timestep * 0.6),  # 中期timestep
            ("t_mid_late", max_timestep * 0.2, max_timestep * 0.4),  # 中后期timestep
            ("t_late", 0, max_timestep * 0.2), # 后期timestep (低噪声，接近干净图像)
        ]
        
        # 为每个范围计算统计信息
        for range_name, t_min, t_max in ranges:
            # 创建mask来选择该范围内的样本
            mask = (timesteps >= t_min) & (timesteps < t_max)
            
            if mask.any():
                # 该范围内有样本
                range_weighted_losses = weighted_loss_per_sample[mask]  # 已经应用了所有权重的loss
                range_timesteps = timesteps[mask]
                sample_count = mask.sum().float()
                
                # 记录加权后的loss (这是实际用于训练的loss)
                self.log(f"train/weighted_loss_{range_name}_mean", range_weighted_losses.mean(), 
                        on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                # 只有当样本数量大于1时才计算标准差，避免警告
                if sample_count > 1:
                    self.log(f"train/weighted_loss_{range_name}_std", range_weighted_losses.std(), 
                            on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                else:
                    self.log(f"train/weighted_loss_{range_name}_std", 0.0, 
                            on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                
                # 记录未加权的原始loss (如果可用)
                if unweighted_loss_per_sample is not None:
                    range_unweighted_losses = unweighted_loss_per_sample[mask]
                    self.log(f"train/unweighted_loss_{range_name}_mean", range_unweighted_losses.mean(), 
                            on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                    if sample_count > 1:
                        self.log(f"train/unweighted_loss_{range_name}_std", range_unweighted_losses.std(), 
                                on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                    else:
                        self.log(f"train/unweighted_loss_{range_name}_std", 0.0, 
                                on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                
                # 记录权重统计
                if final_weights is not None:
                    range_final_weights = final_weights[mask]
                    self.log(f"train/weights_{range_name}_mean", range_final_weights.mean(), 
                            on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                    self.log(f"train/weights_{range_name}_max", range_final_weights.max(), 
                            on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                    self.log(f"train/weights_{range_name}_min", range_final_weights.min(), 
                            on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                
                # 记录基础weighting统计 (如果可用)
                if base_weighting is not None:
                    range_base_weighting = base_weighting[mask]
                    self.log(f"train/base_weighting_{range_name}_mean", range_base_weighting.mean(), 
                            on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                
                # 记录样本数量和timestep统计
                self.log(f"train/loss_{range_name}_count", sample_count, 
                        on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                self.log(f"train/timestep_{range_name}_mean", range_timesteps.float().mean(), 
                        on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
            else:
                # 该范围内没有样本，记录为0
                self.log(f"train/weighted_loss_{range_name}_mean", 0.0, 
                        on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                self.log(f"train/unweighted_loss_{range_name}_mean", 0.0, 
                        on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
                self.log(f"train/loss_{range_name}_count", 0.0, 
                        on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)

    def on_after_backward(self):
        """Log gradient norms of trainable parameters each step."""
        try:
            # Only log once every N optimizer steps. During gradient accumulation,
            # global_step increments after the optimizer step; use (global_step + 1)
            # to target the micro-batch that triggers the step.
            # Default to 1 so early steps are also recorded by default
            log_every = int(getattr(self, "log_every_n_steps", 1))
            upcoming_step = int(self.global_step) + 1
            if log_every > 0 and (upcoming_step % log_every) != 0:
                return
            # Compute global grad norm and per-parameter norms (only for trainable params)
            total_sq = 0.0
            for param_name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if param.grad is None:
                    continue
                grad_tensor = param.grad.detach()
                grad_norm_l2 = grad_tensor.norm(2).item()
                total_sq += float(grad_norm_l2 ** 2)
                # Sanitize metric key for loggers: keep dots, collapse duplicate prefixes
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
            # Global grad norm
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

            # Decide whether to skip this optimizer step due to exploding or non-finite grads
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
        # If previous hooks marked this step to be skipped, clear grads and return
        if getattr(self, "_skip_step_due_to_grad", False):
            # Execute the closure so Lightning's loop consumes the result even when skipping the step
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
        # Perform the optimizer step
        result = super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        # EMA update after optimizer step
        try:
            self._ema_update()
        except Exception:
            pass
        # Log learning rate(s) every step
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
        # 检查是否在梯度累积的最后一个batch
        # 当 accumulate_grad_batches > 1 时，只有在最后一个累积batch时才进行推理和保存
        accumulate_grad_batches = getattr(self.trainer, 'accumulate_grad_batches', 1)
        if accumulate_grad_batches > 1:
            # 计算当前batch在累积周期中的位置
            batch_in_accumulation = (batch_idx % accumulate_grad_batches) + 1
            is_last_batch_in_accumulation = batch_in_accumulation == accumulate_grad_batches
            if not is_last_batch_in_accumulation:
                return

        # Periodic checkpoint saving (只在最后一个累积batch时执行)
        base_step = getattr(self, '_resume_step', 0)
        step = int(self.global_step) + base_step
        save_iv = int(getattr(self, "save_interval", -1) or -1)
        if save_iv > 0 and step > 0 and step % save_iv == 0:
            # Only local rank 0 saves
            local_rank = os.environ.get("LOCAL_RANK", "0")
            
            if local_rank == "0":
                print(f"[HR][Save] Saving checkpoint on local_rank 0")
                os.makedirs(getattr(self, "checkpoint_dir", "./checkpoints"), exist_ok=True)
                # Save LoRA adapters if any are active
                self._save_lora_adapters(step)
                self._save_checkpoint(step)

        if self.sample_interval <= 0 or not self.sample_prompts:
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
            # Parity check removed
            was_training = self.training
            self.eval()
            try:
                # Avoid DDP collectives during generation: unwrap transformer if wrapped
                orig_transformer = getattr(self.sd3_pipe, "transformer", None)
                unwrapped = orig_transformer.module if isinstance(orig_transformer, DDP) else orig_transformer
                if unwrapped is not None:
                    self.sd3_pipe.transformer = unwrapped
                

                self.sd3_pipe.scheduler.config.use_dynamic_shifting = False
                self.sd3_pipe.scheduler.set_shift(2.0)
                # 1. 使用验证集prompt进行测试
                # print(f"[HR][Sample] Generating validation images (step {step})")
                # If fixed prompt set, override validation prompts to be the same
                prompts_to_use = self.sample_prompts
                if self.fixed_train_prompt and (not prompts_to_use or len(prompts_to_use) == 0):
                    prompts_to_use = [self.fixed_train_prompt]
                for i, prompt in enumerate(prompts_to_use):
                    images, vae16_pil = self._pipeline_infer(
                        prompt=prompt,
                        height=self.sample_size,
                        width=self.sample_size,
                        decode_vae16=True,
                    )
                    if images is not None:
                        img = images[0]
                        # 使用prompt的前N个词作为文件名的一部分
                        prompt_words = prompt.split()[:5]  # 取前5个词
                        prompt_prefix = "_".join(prompt_words).replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")
                        fn = os.path.join(self.save_path, f"hr_sample_step_{step}_val_{i}_{prompt_prefix}.jpg")
                        try:
                            img.save(fn)
                            print(f"[HR][Sample] Saved validation image: {fn}")
                        except Exception:
                            pass
                        if vae16_pil is not None:
                            fn_vae16 = os.path.join(self.save_path, f"hr_sample_step_{step}_val_{i}_{prompt_prefix}_vae16.jpg")
                            try:
                                vae16_pil.save(fn_vae16)
                                print(f"[HR][Sample] Saved validation vae16 image: {fn_vae16}")
                            except Exception:
                                pass
                
                # 2. 使用训练集prompt进行过拟合检测（单样本过拟合默认为1个prompt）
                train_sample_count = int(self.model_config.get("train_sample_count", 1))
                if self.fixed_train_prompt:
                    train_prompts = [self.fixed_train_prompt][:train_sample_count]
                else:
                    train_prompts = self._get_random_train_prompts(num_prompts=train_sample_count)
                if train_prompts:
                    print(f"[HR][Sample] Generating training images for overfitting detection (step {step})")
                    for i, prompt in enumerate(train_prompts):
                        print(f"[HR][Sample] Training prompt {i}: {prompt}")
                        images, vae16_pil = self._pipeline_infer(
                            prompt=prompt,
                            height=self.sample_size,
                            width=self.sample_size,
                            decode_vae16=True,
                        )
                        if images is not None:
                            img = images[0]
                            # 使用prompt的前N个词作为文件名的一部分
                            prompt_words = prompt.split()[:5]  # 取前5个词
                            prompt_prefix = "_".join(prompt_words).replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")
                            fn = os.path.join(self.save_path, f"hr_sample_step_{step}_train_{i}_{prompt_prefix}.jpg")
                            try:
                                img.save(fn)
                                print(f"[HR][Sample] Saved training image: {fn}")
                            except Exception:
                                print(f"[HR][Sample] Failed to save training image: {fn}")  
                                pass
                            if vae16_pil is not None:
                                fn_vae16 = os.path.join(self.save_path, f"hr_sample_step_{step}_train_{i}_{prompt_prefix}_vae16.jpg")
                                try:
                                    vae16_pil.save(fn_vae16)
                                    print(f"[HR][Sample] Saved training vae16 image: {fn_vae16}")
                                except Exception:
                                    pass
                else:
                    print(f"[HR][Sample] No training prompts available for overfitting detection")
            finally:
                # Restore original (potentially DDP-wrapped) transformer after sampling
                try:
                    if orig_transformer is not None:
                        self.sd3_pipe.transformer = orig_transformer
                except Exception:
                    pass
                if was_training:
                    self.train()

    def _save_checkpoint(self, step: int) -> None:
        """Save minimal HR training state (HR PatchEmbed + ProjectOut) for resume/inference."""
        try:
            ckpt: Dict[str, Any] = {
                "global_step": int(step),
                "timestamp": time.time(),
                "model_state_dict": {},
            }
            # Save HR patch embedding weights under a prefix friendly to the loader
            hr_patch = getattr(self.transformer, "hr_patch_embed", None)
            if hr_patch is not None and hasattr(hr_patch, "patch_embed"):
                for k, v in hr_patch.patch_embed.state_dict().items():
                    ckpt["model_state_dict"][f"hr_patch_embed.patch_embed.{k}"] = v

            tokenizer_patch = getattr(self.transformer, "tokenizer_patch_embed", None)
            if isinstance(tokenizer_patch, nn.Module):
                for k, v in tokenizer_patch.state_dict().items():
                    ckpt["model_state_dict"][f"tokenizer_patch_embed.{k}"] = v

            # Save project_out head whenever present (supports both proj_out/project_out)
            proj_module = self._get_proj_out_module()
            if proj_module is not None:
                for k, v in proj_module.state_dict().items():
                    ckpt["model_state_dict"][f"project_out.{k}"] = v

            # Write file(s)
            ckpt_dir = getattr(self, "checkpoint_dir", "./checkpoints")
            path = os.path.join(ckpt_dir, f"step_{step}.pt")
            latest = os.path.join(ckpt_dir, "latest.pt")
            # 可选：保存 EMA 权重
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
                print(f"[HR][Save] Saved checkpoint: {path}")
            except Exception:
                pass
        except Exception:
            pass

    def _encode_images_to_latents(
        self,
        images: torch.Tensor,
        device: Union[str, torch.device],
        dtype: torch.dtype,
        masked_channel_start: Optional[int] = None,
    ) -> torch.Tensor:
        if self.use_tokenizer_vae and self.tokenizer_vae_wrapper is not None:
            pixel_inputs = images
            latents = self.tokenizer_vae_wrapper.encode(
                pixel_inputs, masked_channel_start=masked_channel_start
            )
            if isinstance(latents, (tuple, list)):
                latents = latents[0]
            if not isinstance(latents, torch.Tensor):
                raise TypeError("Tokenizer VAE encode did not return a tensor")
            if latents.dim() != 4:
                raise ValueError(f"Tokenizer VAE expected 4D latents but received shape {tuple(latents.shape)}")
            latents = latents.to(device=device, dtype=dtype)
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
        hr_patch = getattr(self.transformer, "hr_patch_embed", None)
        if isinstance(hr_patch, nn.Module):
            modules.append(hr_patch)
        tokenizer_patch = getattr(self.transformer, "tokenizer_patch_embed", None)
        if isinstance(tokenizer_patch, nn.Module):
            modules.append(tokenizer_patch)
        return modules

    # ----------------- helpers -----------------
    def _get_proj_out_module(self) -> Optional[nn.Module]:
        proj_module = getattr(self.transformer, "proj_out", None)
        if proj_module is None:
            proj_module = getattr(self.transformer, "project_out", None)
        return proj_module

    # ----------------- EMA helpers -----------------
    def _iter_trainable_named_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name, param

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
        base_dir = getattr(self, "lora_dir", "./lora_weights")
        save_dir = os.path.join(base_dir, f"lora_step_{step}")
        
        print(f"[HR][LoRA] 💾 保存LoRA到 {save_dir}")
        success = self.lora_manager.save_all(save_dir, step)
        
        if success:
            print(f"[HR][LoRA] ✅ 成功保存LoRA到 {save_dir}")
        else:
            print(f"[HR][LoRA] ❌ 保存LoRA失败")

    # def _save_lora_adapters(self, step: int) -> None:
    #     """Persist active LoRA adapters for the transformer, if any."""
    #     # Query active adapters from pipeline if available
    #     active = list(self.transformer.get_active_adapters())  # type: ignore[attr-defined]
    #     print(f"[HR][LoRA] Active adapters: {active}")
    #     if not active:
    #         return

    #     # Directory for LoRA weights (prefer explicit lora_dir, fallback to checkpoints)
    #     base_dir = getattr(self, "lora_dir", None)
    #     if not base_dir:
    #         base_dir = getattr(self, "checkpoint_dir", "./checkpoints")
    #     save_dir = os.path.join(base_dir, f"lora_{step}")
    #     os.makedirs(save_dir, exist_ok=True)
        

    #     # Use peft helper to extract transformer LoRA layers
    #     try:
    #         from peft import get_peft_model_state_dict  # type: ignore
    #     except Exception:
    #         get_peft_model_state_dict = None  # type: ignore
    #     print(f"[HR][LoRA] Saving adapters to {save_dir}")
    #     for adapter in active:
    #         weight_name = f"{adapter}.safetensors"
    #         print(f"[HR][LoRA] Saving adapter '{adapter}' to {save_dir}")
    #         # try:
    #         state = get_peft_model_state_dict(self.transformer, adapter_name=adapter)
    #         self.sd3_pipe.save_lora_weights(
    #             save_directory=str(save_dir),
    #             weight_name=weight_name,
    #             transformer_lora_layers=state,
    #             safe_serialization=True,
    #         )
    #         print(f"[HR][LoRA] Saved adapter '{adapter}' to {save_dir}")
    #         # except Exception:
    #         #     pass
       


def train(dataloader_or_dataset, trainable_model: SD3HRModel, config: dict, resume_from_checkpoint: Optional[str] = None, resume_from_lora_checkpoint: Optional[str] = None):
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    # Resolve run directory similar to previous training flow
    run_name = train_cfg.get("run_name") or os.environ.get("RUN_NAME") or time.strftime("%Y%m%d-%H%M%S")
    base_save_path = train_cfg.get("save_path", "./output")
    run_dir = os.path.join(base_save_path, run_name)
    output_dir = os.path.join(run_dir, "output")
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    # Follow existing convention used in other trainers for full checkpoints and lora
    full_ckpt_dir = os.path.join(run_dir, "full_checkpoints")
    lora_dir = os.path.join(run_dir, "lora_weights")
    os.makedirs(full_ckpt_dir, exist_ok=True)
    os.makedirs(lora_dir, exist_ok=True)
    checkpoints_dir = full_ckpt_dir
    # Save full config for reproducibility
    import yaml
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    # Align sampling cadence and save locations with previous trainer
    prompts = list(data_cfg.get("validation_prompts", []))
    if prompts:
        trainable_model.sample_prompts = prompts

    trainable_model.sample_interval = int(train_cfg.get("sample_interval", getattr(trainable_model, "sample_interval", 1000)))
    trainable_model.save_path = output_dir

    # Configure checkpoint cadence and location
    trainable_model.save_interval = int(train_cfg.get("save_interval", getattr(trainable_model, "save_interval", -1)))
    trainable_model.checkpoint_dir = checkpoints_dir
    # Prefer dedicated lora directory for adapter weights
    setattr(trainable_model, "lora_dir", lora_dir)

    # Prefer data base_size for sample size if provided
    sample_size = int(data_cfg.get("base_size", getattr(trainable_model, "sample_size", 1024)))
    trainable_model.sample_size = sample_size

    # Configure loss print cadence from train config
    trainable_model.print_every_n_steps = int(train_cfg.get("print_every_n_steps", getattr(trainable_model, "print_every_n_steps", 10)))

    # Configure logger throttle interval (default 20)
    trainable_model.log_every_n_steps = int(train_cfg.get("log_every_n_steps", getattr(trainable_model, "log_every_n_steps", 20)))
    # Accept either:
    # - a prebuilt DataLoader
    # - a Dataset
    # - a raw batch dict (single-sample repeated), to avoid constructing any dataloader upstream
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
        # Wrap a raw batch dict into a tiny IterableDataset that repeats it indefinitely.
        raw_batch = dataloader_or_dataset

        class _RepeatBatch(IterableDataset):  # type: ignore
            def __iter__(self):  # noqa: D401
                while True:
                    yield raw_batch

        loader = DataLoader(_RepeatBatch(), batch_size=None, num_workers=0)
        train_dataset = None
    else:
        # Fallback: try to build a DataLoader from whatever was provided
        train_dataset = dataloader_or_dataset
        loader = DataLoader(
            dataloader_or_dataset,
            batch_size=train_cfg.get("batch_size", 1),
            shuffle=True,
            num_workers=train_cfg.get("dataloader_workers", 0),
        )
    
    # Pass training dataset to model for overfitting detection
    if hasattr(trainable_model, 'train_dataset'):
        trainable_model.train_dataset = train_dataset
        if train_dataset is not None:
            print(f"[HR][TrainSample] Set train_dataset: {type(train_dataset)} with size {len(train_dataset) if hasattr(train_dataset, '__len__') else 'unknown'}")
        else:
            print(f"[HR][TrainSample] Using DataLoader for training samples (no direct dataset access)")
    else:
        print(f"[HR][TrainSample] trainable_model has no train_dataset attribute")

    # Handle checkpoint resuming
    resume_step = 0
    auto_resume = train_cfg.get("auto_resume", True)  # Default to True for backward compatibility
    
    if resume_from_checkpoint:
        # Use provided checkpoint path
        if os.path.exists(resume_from_checkpoint):
            resume_step = trainable_model.load_checkpoint(resume_from_checkpoint, resume_from_lora_checkpoint)
        else:
            print(f"[HR][Resume] Checkpoint not found: {resume_from_checkpoint}")
    elif auto_resume:
        # Auto-detect latest checkpoint
        latest_ckpt = os.path.join(checkpoints_dir, "latest.pt")
        if os.path.exists(latest_ckpt):
            print(f"[HR][Resume] Auto-detected latest checkpoint: {latest_ckpt}")
            # Auto-detect corresponding LoRA checkpoint
            auto_lora_ckpt = None
            if resume_from_lora_checkpoint is None:
                # Try to find corresponding LoRA checkpoint
                lora_ckpt_dir = os.path.join(lora_dir, "latest")
                if os.path.exists(lora_ckpt_dir):
                    auto_lora_ckpt = lora_ckpt_dir
                    print(f"[HR][Resume] Auto-detected latest LoRA checkpoint: {auto_lora_ckpt}")
                else:
                    # Try to find the most recent LoRA checkpoint directory
                    import glob
                    lora_pattern = os.path.join(lora_dir, "lora_step_*")
                    lora_dirs = glob.glob(lora_pattern)
                    if lora_dirs:
                        # Sort by step number and get the latest
                        lora_dirs.sort(key=lambda x: int(x.split('_')[-1]))
                        auto_lora_ckpt = lora_dirs[-1]
                        print(f"[HR][Resume] Auto-detected latest LoRA checkpoint: {auto_lora_ckpt}")
            
            resume_step = trainable_model.load_checkpoint(latest_ckpt, auto_lora_ckpt)
        else:
            print(f"[HR][Resume] No checkpoint found, starting from scratch")
    else:
        print(f"[HR][Resume] Auto-resume disabled, starting from scratch")
    
    if resume_step > 0:
        print(f"[HR][Resume] Resuming training from step {resume_step}")
        # Store the resume step for later use in training
        trainable_model._resume_step = resume_step

    # Optional pretrain sampling (before training)
    if train_cfg.get("pretrain_sample", False):
        # Only rank 0 performs pretrain sample inference
        if os.environ.get("LOCAL_RANK", "0") == "0":
            try:
                was_training = trainable_model.training
                trainable_model.eval()
                prompts_list = list(getattr(trainable_model, "sample_prompts", []))
                print(f"[HR][Pretrain] pretrain_sample enabled: prompts={len(prompts_list)} save_dir={output_dir}")
                # Parity check removed
                trainable_model.sd3_pipe.scheduler.config.use_dynamic_shifting = False
                trainable_model.sd3_pipe.scheduler.set_shift(2.0)
                for i, prompt in enumerate(prompts_list[: int(train_cfg.get("validation_samples", 4))]):
                    images, vae16_pil = trainable_model._pipeline_infer(
                        prompt=prompt,
                        height=getattr(trainable_model, "sample_size", 1024),
                        width=getattr(trainable_model, "sample_size", 1024),
                        decode_vae16=True,
                    )
                    if images is not None:
                        img = images[0]  # PIL.Image
                        fn = os.path.join(output_dir, f"pretrain_{i}.jpg")
                        try:
                            img.save(fn)
                            print(f"[HR][Pretrain] saved -> {fn}")
                        except Exception:
                            print(f"[HR][Pretrain] Failed to save training image: {fn}")  
                            pass
                        if vae16_pil is not None:
                            fn_vae16 = os.path.join(output_dir, f"pretrain_{i}_vae16.jpg")
                            try:
                                vae16_pil.save(fn_vae16)
                                print(f"[HR][Pretrain] saved vae16 -> {fn_vae16}")
                            except Exception:
                                pass
                    else:
                        print(f"[HR][Pretrain] pipeline returned no images for prompt index {i}")
            finally:
                if was_training:
                    trainable_model.train()

    # Enable gradient checkpointing on transformer when requested by config
    try:
        if bool(train_cfg.get("gradient_checkpointing", True)):
            try:
                trainable_model.transformer.enable_gradient_checkpointing()
                print("[HR] Enabled gradient checkpointing on transformer")
            except Exception:
                pass
    except Exception:
        pass

    progress_bar = TQDMProgressBar(refresh_rate=train_cfg.get("progress_bar_refresh_rate", 10))
    # Configure loggers: CSV always on main, plus WandB if configured and available
    loggers = None
    try:
        # Treat LOCAL_RANK 0 as main process
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
                wb = WandbLogger(project=wandb_cfg.get("project", "sd3-hr"), name=run_name)
                loggers.append(wb)
            except Exception as e:
                print(f"WandB logger unavailable, using CSV only: {e}")

    # Configure trainer with resume support
    trainer_kwargs = {
        "accumulate_grad_batches": train_cfg.get("accumulate_grad_batches", 1),
        "enable_checkpointing": False,
        "enable_progress_bar": True,
        "logger": loggers if loggers is not None else True,
        "log_every_n_steps": train_cfg.get("log_every_n_steps", 1),
        "max_steps": train_cfg.get("max_steps", -1),
        "max_epochs": train_cfg.get("max_epochs", -1),
        "gradient_clip_val": train_cfg.get("gradient_clip_val", 0.5),
        "callbacks": [progress_bar],
        "strategy": 'ddp_find_unused_parameters_true',
    }
    
    # If resuming, set the starting step
    if resume_step > 0:
        trainer_kwargs["min_steps"] = resume_step + 1  # Start from next step
    
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(trainable_model, loader)


