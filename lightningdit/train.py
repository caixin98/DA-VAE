"""
Training Codes of LightningDiT together with VA-VAE.
It envolves advanced training methods, sampling methods, 
architecture design methods, computation methods. We achieve
state-of-the-art FID 1.35 on ImageNet 256x256.

by Maple (Jingfeng Yao) from HUST-VL
"""

import torch
import torch.distributed as dist
import torch.backends.cuda
import torch.backends.cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

import math
import yaml
import json
import numpy as np
import logging
import os
import argparse
import subprocess
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm

from diffusers.models import AutoencoderKL
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
from accelerate import Accelerator
from local_datasets.img_latent_dataset import ImgLatentDataset
import torchvision
from tokenizer.vavae import VA_VAE
from tokenizer.davae import DA_VAE

def load_patch_embed_from_vae_module(model, vae_model, rank: int = 0):
    """
    Pull PatchEmbed weights directly from a loaded VAE module (DAVAE)
    via its export_patch_embed_state(), and load into DiT x_embedder.proj.
    """
    if vae_model is None or not hasattr(vae_model, 'export_patch_embed_state'):
        raise AttributeError("VAE module does not support export_patch_embed_state()")
    sd = vae_model.export_patch_embed_state()
    if sd is None:
        raise RuntimeError("pe_student not initialized in VAE; enable PE alignment during VAE training")
    # unwrap DDP if needed on DiT model
    targ = getattr(model, 'module', model)
    proj = getattr(getattr(targ, 'x_embedder', None), 'proj', None)
    if proj is None:
        raise AttributeError("Model missing x_embedder.proj for patch embedding")
    # Avoid boolean evaluation on tensors by checking keys explicitly
    if 'x_embedder.proj.weight' in sd:
        w = sd['x_embedder.proj.weight']
    elif 'module.x_embedder.proj.weight' in sd:
        w = sd['module.x_embedder.proj.weight']
    else:
        w = sd.get('weight', None)
    if 'x_embedder.proj.bias' in sd:
        b = sd['x_embedder.proj.bias']
    elif 'module.x_embedder.proj.bias' in sd:
        b = sd['module.x_embedder.proj.bias']
    else:
        b = sd.get('bias', None)
    if w is None:
        raise KeyError("PatchEmbed weight not found in VAE export state")
    # Enforce exact shape match; no adaptation
    if tuple(w.shape) != tuple(proj.weight.shape):
        raise ValueError(
            f"PatchEmbed weight shape mismatch: source {tuple(w.shape)} vs target {tuple(proj.weight.shape)}"
        )
    target_has_bias = proj.bias is not None
    source_has_bias = b is not None
    if target_has_bias != source_has_bias:
        raise ValueError(
            f"PatchEmbed bias presence mismatch: source has_bias={source_has_bias} vs target has_bias={target_has_bias}"
        )
    if target_has_bias and tuple(b.shape) != tuple(proj.bias.shape):
        raise ValueError(
            f"PatchEmbed bias shape mismatch: source {tuple(b.shape)} vs target {tuple(proj.bias.shape)}"
        )

    with torch.no_grad():
        proj.weight.copy_(w)
        if target_has_bias:
            proj.bias.copy_(b)


def _sync_path_to_s3(local_path: str):
    """
    Mirror a local directory to S3 using a base-path mapping controlled by env vars.
    Environment:
      S3_MIRROR_LOCAL_BASE (default: current script directory)
      S3_MIRROR_S3_BASE    (default: disabled)
      S3_MIRROR_DISABLE    (set to '1' to disable)
    """
    try:
        if os.environ.get('S3_MIRROR_DISABLE', '0') == '1':
            return
        # Support multiple base prefixes via comma or ':' separated values; use the longest matching base
        _env_local_base = os.environ.get(
            'S3_MIRROR_LOCAL_BASE',
            ''
        )
        _parts = []
        for _chunk in _env_local_base.split(','):
            _parts.extend(_chunk.split(':'))
        candidate_bases = [os.path.abspath(p.strip()) for p in _parts if p.strip()]
        # also consider repo root as a candidate base
        try:
            _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
            if _repo_root not in candidate_bases:
                candidate_bases.append(_repo_root)
        except Exception:
            pass
        s3_base = os.environ.get('S3_MIRROR_S3_BASE', '').rstrip('/')
        if not s3_base:
            return
        local_path = os.path.abspath(local_path)
        local_real = os.path.realpath(local_path)
        if not os.path.isdir(local_real):
            return
        # choose the longest matching base (compare real paths to handle symlinks)
        best_base = None
        best_base_real = None
        for b in candidate_bases:
            b_real = os.path.realpath(b)
            if local_real == b_real or local_real.startswith(b_real + os.sep):
                if best_base is None or len(b_real) > len(best_base_real):
                    best_base = b
                    best_base_real = b_real
        if best_base is None:
            try:
                import logging as _logging
                _logging.getLogger(__name__).info(
                    f"S3 sync skip: {local_path} (real={local_real}) is not under any configured base: {candidate_bases}"
                )
            except Exception:
                print(f"S3 sync skip: {local_path} (real={local_real}) is not under any configured base: {candidate_bases}")
            return
        rel = os.path.relpath(local_real, best_base_real)
        s3_uri = f"{s3_base}/{rel}"
        res = subprocess.run(
            ['aws', 's3', 'sync', local_path, s3_uri, '--only-show-errors'],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res.returncode != 0:
            try:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    f"S3 sync failed for {local_path} -> {s3_uri} (code {res.returncode}). Stderr: {res.stderr.strip()[:500]}"
                )
            except Exception:
                print(f"S3 sync failed for {local_path} -> {s3_uri} (code {res.returncode}).")
                if res.stderr:
                    print(res.stderr.strip()[:500])
        else:
            try:
                import logging as _logging
                _logging.getLogger(__name__).info(f"S3 sync: {local_path} -> {s3_uri}")
            except Exception:
                print(f"S3 sync: {local_path} -> {s3_uri}")
    except Exception as e:
        # best-effort sync; never crash training, but surface why it failed
        try:
            import logging as _logging
            _logging.getLogger(__name__).warning(f"S3 sync exception for {local_path}: {e}")
        except Exception:
            print(f"S3 sync exception for {local_path}: {e}")


def _parse_dtype(value, default=torch.float32):
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        key = value.lower()
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if key in mapping:
            return mapping[key]
        if key == "auto":
            return default
    raise ValueError(f"Unsupported torch dtype specifier: {value}")


def _resolve_vae_config_path(name):
    if not name:
        return None
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if str(name).endswith((".yaml", ".yml")):
        candidates.append(str(name))
        candidates.append(os.path.join(base_dir, str(name)))
    else:
        candidates.append(os.path.join(base_dir, "tokenizer", "configs", f"{name}.yaml"))
        candidates.append(os.path.join(base_dir, "tokenizer", "configs", f"{name}.yml"))
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return None


def _create_vae(train_config, device):
    vae_cfg = train_config.get("vae") or {}
    data_cfg = train_config.get("data") or {}
    image_size = int(data_cfg.get("image_size", 256))
    model_name = vae_cfg.get("model_name", "da_f16d32_align_mean")
    variant = str(vae_cfg.get("variant", "")).lower()
    if not variant:
        variant = "davae" if model_name.lower().startswith("dc") else "vavae"
    variant = variant if variant in {"davae", "vavae"} else "davae"

    cfg_path = vae_cfg.get("config") or vae_cfg.get("config_path") or _resolve_vae_config_path(model_name)
    if cfg_path is None:
        raise FileNotFoundError(f"Unable to resolve VAE config path for '{model_name}'")
    fp16_flag = vae_cfg.get("fp16", True)

    if variant == "vavae":
        return (
            VA_VAE(
                cfg_path,
                img_size=image_size,
                fp16=bool(fp16_flag),
                device=device,
            ),
            "vavae",
        )

    return (
        DA_VAE(
            cfg_path,
            img_size=image_size,
            fp16=bool(fp16_flag),
            device=device,
        ),
        "davae",
    )


class ChannelLossScheduler:
    """Piecewise-linear scheduler that boosts loss weight on tail channels."""

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get('enabled', False))
        self.anchor_channel = max(int(cfg.get('anchor_channel', 32)), 0)
        self.start_step = int(cfg.get('start_step', 0))
        self.end_step = int(cfg.get('end_step', self.start_step))
        if self.end_step < self.start_step:
            self.end_step = self.start_step
        self.start_weight = float(cfg.get('start_weight', 0.0))
        self.end_weight = float(cfg.get('end_weight', 1.0))
        self.base_weight = float(cfg.get('base_weight', 1.0))
        self.schedule = str(cfg.get('schedule', 'linear')).lower()
        self._eps = float(cfg.get('eps', 1e-6))

    def _interp(self, step: int) -> float:
        if step <= self.start_step:
            return self.start_weight
        if step >= self.end_step:
            return self.end_weight
        span = max(1, self.end_step - self.start_step)
        progress = (step - self.start_step) / span
        if self.schedule == 'cosine':
            cos_term = 0.5 * (1 - math.cos(math.pi * progress))
            return self.start_weight + (self.end_weight - self.start_weight) * cos_term
        return self.start_weight + progress * (self.end_weight - self.start_weight)

    def is_active(self) -> bool:
        if not self.enabled:
            return False
        if abs(self.end_weight - self.base_weight) > self._eps:
            return True
        if abs(self.start_weight - self.base_weight) > self._eps:
            return True
        return False

    def get_mask(self, *, step: int, total_channels: int, ndim: int, device, dtype):
        if not self.is_active() or ndim < 2:
            return None
        has_tail = total_channels > self.anchor_channel
        if not has_tail and math.isclose(self.base_weight, 1.0, rel_tol=1e-6, abs_tol=self._eps):
            return None

        tail_weight = self._interp(step)
        if has_tail and math.isclose(tail_weight, 1.0, rel_tol=1e-6, abs_tol=self._eps) and math.isclose(self.base_weight, 1.0, rel_tol=1e-6, abs_tol=self._eps):
            return None

        mask_shape = [1, total_channels] + [1] * (ndim - 2)
        mask = torch.full(mask_shape, self.base_weight, device=device, dtype=dtype)
        if has_tail:
            mask[:, self.anchor_channel:, ...] = tail_weight

        if torch.allclose(mask, torch.ones_like(mask), atol=self._eps, rtol=0):
            return None

        return mask


def do_train(train_config, accelerator):
    """
    Trains a LightningDiT.
    """
    # Setup accelerator:
    device = accelerator.device
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Setup an experiment folder:
    if accelerator.is_main_process:
        # Resolve output base to absolute path anchored at repo root
        out_base = train_config['train']['output_dir']
        out_base_abs = out_base if os.path.isabs(out_base) else os.path.abspath(os.path.join(base_dir, out_base))
        os.makedirs(out_base_abs, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{out_base_abs}/*"))
        model_string_name = train_config['model']['model_type'].replace("/", "-")
        if train_config['train']['exp_name'] is None:
            exp_name = f'{experiment_index:03d}-{model_string_name}'
        else:
            exp_name = train_config['train']['exp_name']
        experiment_dir = f"{out_base_abs}/{exp_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        tb_base = "tensorboard_logs"
        tb_base_abs = tb_base if os.path.isabs(tb_base) else os.path.abspath(os.path.join(base_dir, tb_base))
        tensorboard_dir_log = f"{tb_base_abs}/{exp_name}"
        os.makedirs(tensorboard_dir_log, exist_ok=True)
        if SummaryWriter is not None:
            writer = SummaryWriter(log_dir=tensorboard_dir_log)
        else:
            class _NullWriter:
                def add_text(self, *args, **kwargs):
                    return
                def add_scalar(self, *args, **kwargs):
                    return
            writer = _NullWriter()

        # Initialize Weights & Biases by default (main process)
        try:
            import wandb  # lazy import to avoid hard dependency
            wandb_kwargs = {
                'project': 'lightningdit',
                'name': exp_name,
                'config': train_config,
            }
            if isinstance(train_config.get('wandb'), dict):
                cfg = train_config['wandb']
                if 'project' in cfg:
                    wandb_kwargs['project'] = cfg['project']
                if 'entity' in cfg:
                    wandb_kwargs['entity'] = cfg['entity']
                if 'name' in cfg:
                    wandb_kwargs['name'] = cfg['name']
            try:
                wandb.init(**wandb_kwargs)
                logger.info("Initialized Weights & Biases logging")
            except Exception as inner_e:
                # Fallback: retry without entity if entity appears invalid
                msg = str(inner_e).lower()
                if 'entity' in wandb_kwargs and ('entity' in msg or 'not found' in msg or 'upsertbucket' in msg):
                    bad_entity = wandb_kwargs.pop('entity')
                    logger.info(f"WandB entity '{bad_entity}' invalid, retrying without entity...")
                    wandb.init(**wandb_kwargs)
                    logger.info("Initialized Weights & Biases logging (fallback without entity)")
                else:
                    raise
        except Exception as e:
            logger.info(f"WandB not initialized: {e}")

        # add configs to tensorboard
        config_str=json.dumps(train_config, indent=4)
        writer.add_text('training configs', config_str, global_step=0)
        # initial sync to ensure remote tree exists
        _sync_path_to_s3(experiment_dir)
        _sync_path_to_s3(tensorboard_dir_log)
    out_base = train_config['train']['output_dir']
    out_base_abs = out_base if os.path.isabs(out_base) else os.path.abspath(os.path.join(base_dir, out_base))
    checkpoint_dir = f"{out_base_abs}/{train_config['train']['exp_name']}/checkpoints"

    # get rank
    rank = accelerator.local_process_index

    # Create model:
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 16
    assert train_config['data']['image_size'] % downsample_ratio == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = train_config['data']['image_size'] // downsample_ratio
    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        wo_shift=train_config['model']['wo_shift'] if 'wo_shift' in train_config['model'] else False,
        in_channels=train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4,
        use_checkpoint=train_config['model']['use_checkpoint'] if 'use_checkpoint' in train_config['model'] else False,
    )

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # load pretrained model
    if 'weight_init' in train_config['train']:
        checkpoint = torch.load(train_config['train']['weight_init'], map_location=lambda storage, loc: storage)
        # remove the prefix 'module.' from the keys
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        # Get adapt_mode from config: 'zero_pad' (default) or 'repeat_average'
        adapt_mode = train_config['train'].get('weight_init_adapt_mode', 'zero_pad')
        model = load_weights_with_shape_check(model, checkpoint, rank=rank, adapt_mode=adapt_mode)
        ema = load_weights_with_shape_check(ema, checkpoint, rank=rank, adapt_mode=adapt_mode)
        if accelerator.is_main_process:
            logger.info(f"Loaded pretrained model from {train_config['train']['weight_init']} (adapt_mode: {adapt_mode})")
    requires_grad(ema, False)
    
    # Optionally freeze DiT backbone: only train x_embedder and final_layer
    freeze_dit = train_config['train'].get('freeze_dit', False)
    if freeze_dit:
        requires_grad(model, False)
        requires_grad(model.x_embedder, True)
        requires_grad(model.final_layer, True)
        if accelerator.is_main_process:
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            num_total = sum(p.numel() for p in model.parameters())
            logger.info(f"Freeze DiT backbone enabled: training only x_embedder and final_layer | trainable params: {num_trainable:,} / {num_total:,}")
    
    # Optionally freeze only x_embedder
    freeze_x_embedder = train_config['train'].get('freeze_x_embedder', False)
    if freeze_x_embedder:
        requires_grad(model.x_embedder, False)
        if accelerator.is_main_process:
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            num_total = sum(p.numel() for p in model.parameters())
            logger.info(f"Freeze x_embedder enabled: trainable params: {num_trainable:,} / {num_total:,}")

    model = DDP(model.to(device), device_ids=[rank])
    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
    )  # default: velocity; 
    if accelerator.is_main_process:
        logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        logger.info(f'Use lognorm sampling: {train_config["transport"]["use_lognorm"]}')
        logger.info(f'Use cosine loss: {train_config["transport"]["use_cosine_loss"]}')
    # Optimizer only on trainable params
    opt_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(opt_params, lr=train_config['optimizer']['lr'], weight_decay=0, betas=(0.9, train_config['optimizer']['beta2']))
    
    # Setup data
    channel_loss_scheduler_cfg = train_config['train'].get('channel_loss_schedule', None)
    channel_loss_scheduler = ChannelLossScheduler(channel_loss_scheduler_cfg) if channel_loss_scheduler_cfg else None
    if channel_loss_scheduler is not None and not channel_loss_scheduler.is_active():
        channel_loss_scheduler = None
    if accelerator.is_main_process and channel_loss_scheduler is not None:
        logger.info(
            "Channel loss scheduler enabled | anchor_channel=%d | start_weight=%.4f -> end_weight=%.4f | steps=%d-%d",
            channel_loss_scheduler.anchor_channel,
            channel_loss_scheduler.start_weight,
            channel_loss_scheduler.end_weight,
            channel_loss_scheduler.start_step,
            channel_loss_scheduler.end_step,
        )
    latent_norm_channels_cfg = train_config['data'].get('latent_norm_channels', None)
    latent_norm_channels = None
    if latent_norm_channels_cfg is not None:
        try:
            latent_norm_channels = int(latent_norm_channels_cfg)
        except (TypeError, ValueError) as exc:
            raise ValueError("data.latent_norm_channels must be an integer when provided") from exc
        if latent_norm_channels <= 0:
            latent_norm_channels = None
    dataset = ImgLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
        latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
        latent_norm_channels=latent_norm_channels,
    )
    batch_size_per_gpu = int(np.round(train_config['train']['global_batch_size'] / accelerator.num_processes))
    global_batch_size = batch_size_per_gpu * accelerator.num_processes
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images {train_config['data']['data_path']}")
        logger.info(f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size")
    
    if 'valid_path' in train_config['data']:
        valid_dataset = ImgLatentDataset(
            data_dir=train_config['data']['valid_path'],
            latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
            latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
            latent_norm_channels=latent_norm_channels,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=True,
            num_workers=train_config['data']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        if accelerator.is_main_process:
            logger.info(f"Validation Dataset contains {len(valid_dataset):,} images {train_config['data']['valid_path']}")

    # ----------------------
    # VAE init and sanity check decode of GT latents
    # ----------------------
    vae = None
    vae_variant = None
    try:
        vae, vae_variant = _create_vae(train_config, device)
        if accelerator.is_main_process:
            logger.info(f"Loaded VAE variant '{vae_variant}' for decode checks and sampling")
    except Exception as e:
        vae = None
        if accelerator.is_main_process:
            logger.info(f"Failed to initialize VAE: {e}")

    vavae = VA_VAE("tokenizer/configs/vavae_f16d32.yaml")

    # latent stats for (de)normalization during decode
    latent_mean, latent_std = dataset.get_latent_stats() if hasattr(dataset, 'get_latent_stats') else (None, None)
    latent_multiplier = train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215
    use_latent_norm = bool(train_config['data'].get('latent_norm', False))
    if latent_mean is not None:
        latent_mean = latent_mean.to(device)
    if latent_std is not None:
        latent_std = latent_std.to(device)

    def _denormalize_latents(tensor: torch.Tensor) -> torch.Tensor:
        base = tensor / latent_multiplier
        if not (use_latent_norm and latent_mean is not None and latent_std is not None):
            return base
        if latent_norm_channels is None:
            return base * latent_std + latent_mean
        k = min(latent_norm_channels, base.shape[1])
        if k <= 0:
            return base
        base[:, :k] = base[:, :k] * latent_std[:, :k] + latent_mean[:, :k]
        return base

    # visualization config (smaller grids and optional fixed classes)
    vis_num_images = train_config['sample'].get('grid_num_images', 8)
    vis_nrow = train_config['sample'].get('grid_nrow', 4)
    default_vis_bs = 8
    vis_batch_size = train_config['sample'].get('vis_batch_size', default_vis_bs)
    # Use user's requested classes by default; can be overridden via config.sample.fixed_classes
    fixed_classes = train_config['sample'].get('fixed_classes', [975, 3, 207, 387, 388, 88, 979, 279])

    # save dirs
    exp_dir = f"{out_base_abs}/{train_config['train']['exp_name']}"
    images_dir = f"{exp_dir}/images"
    samples_dir = f"{exp_dir}/train_samples"
    if accelerator.is_main_process:
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)

    # helper to unwrap model across Accelerate and DDP wrappers
    def _unwrap(m):
        # try accelerate's unwrap first
        try:
            core = accelerator.unwrap_model(m)
        except Exception:
            core = m
        # then iteratively strip any nested .module wrappers (e.g., DDP -> model)
        obj = core
        seen = set()
        while hasattr(obj, 'module') and id(obj) not in seen:
            seen.add(id(obj))
            obj = obj.module
        return obj

    # Initialize DiT patch embed from the already loaded VAE (if enabled and DA_VAE with pe_student)
    init_pe_from_vae = bool(train_config['train'].get('init_x_embedder_from_vae', False))
    if init_pe_from_vae and vae is not None and hasattr(vae, 'model') and hasattr(vae.model, 'export_patch_embed_state'):
        load_patch_embed_from_vae_module(model, vae.model, rank=rank)
        if accelerator.is_main_process:
            logger.info("Loaded DiT patch embed from VAE module (pe_student)")

    # initial VAE decode sanity check on a small GT latent batch
    if accelerator.is_main_process and vae is not None and latent_mean is not None and latent_std is not None:
        try:
            with torch.no_grad():
                for x_gt, _ in loader:
                    x_gt = x_gt.to(device)
                    # only visualize a small number
                    k = min(vis_num_images, x_gt.shape[0])
                    x_gt = x_gt[:k]
                    # denormalize depending on data.latent_norm
                    denorm = _denormalize_latents(x_gt)
                    imgs = vae.decode_to_images(denorm)
                    imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
                    nrow = min(vis_nrow, k) if vis_nrow > 0 else k
                    grid = torchvision.utils.make_grid(imgs_t[:k], nrow=nrow)
                    torchvision.utils.save_image(grid, f"{images_dir}/vae_decode_check.png")
                    if SummaryWriter is not None:
                        writer.add_image('debug/vae_decode_check', grid, global_step=0)
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({'debug/vae_decode_check': [wandb.Image(f"{images_dir}/vae_decode_check.png")]}, step=0)
                    except Exception:
                        pass

    #                 # Compute pe_only-mode alignment loss as in da_autoencoder.py
    #                 try:
    #                     # Require teacher and student to exist
    #                     pe_teacher = getattr(getattr(vae.model, 'loss', None), 'pe_teacher', None)
    #                     pe_student = getattr(vae.model, 'pe_student', None)
    #                     if pe_teacher is not None and pe_student is not None:
    #                         imgs_in = imgs_t[:k].to(device)
    #                         imgs_in = imgs_in * 2.0 - 1.0  # to [-1, 1]
    #                         # dtype align to model params
    #                         model_dtype = next(vae.model.parameters()).dtype
    #                         imgs_in = imgs_in.to(dtype=model_dtype)
    #                         # encode base/DC
    #                         base_post = vae.model.encode_base(imgs_in)
    #                         z_b = base_post.mode().detach()
    #                         hc_post = vae.model.encode_hc(imgs_in)
    #                         z_hc = hc_post.mode()
    #                         # optional latent normalization for teacher path
    #                         if hasattr(vae.model.loss, 'pe_latent_norm_enable') and vae.model.loss.pe_latent_norm_enable and hasattr(vae.model.loss, '_apply_pe_latent_normalization'):
    #                             z_b = vae.model.loss._apply_pe_latent_normalization(z_b)
    #                         # dtype alignment to conv weights
    #                         z_b = z_b.to(dtype=pe_teacher.weight.dtype)
    #                         z_hc = z_hc.to(dtype=pe_student.weight.dtype)
    #                         teacher_proj = pe_teacher(z_b)
    #                         student_proj = pe_student(z_hc)
    #                         pe_loss_val = torch.mean((student_proj - teacher_proj.detach()) ** 2).float()
    #                         pe_loss_scalar = float(pe_loss_val.item())
    #                         logger.info(f"PE-only align loss (teacher vs student) = {pe_loss_scalar:.6e} on {k} samples")
    #                         if SummaryWriter is not None:
    #                             writer.add_scalar('debug/pe_only_align_loss', pe_loss_scalar, 0)
    #                         try:
    #                             import wandb as _wandb
    #                             if _wandb.run is not None:
    #                                 _wandb.log({'debug/pe_only_align_loss': pe_loss_scalar}, step=0)
    #                         except Exception:
    #                             pass
    #                     else:
    #                         logger.info("PE-only align loss skipped (missing pe_teacher or pe_student in VAE model)")
    #                 except Exception as e:
    #                     logger.info(f"PE-only align loss computation failed: {e}")

    #                 # Ensure we have the unwrapped current model for subsequent compares
    #                 cur_model = _unwrap(model).eval()

    #                 # ------------------------------------------------------------
    #                 # Debug: Per-block token MSE between ref-DiT(weight_init, uses z_b)
    #                 # and current-DiT(after VAE PE load, uses z_hc)
    #                 # ------------------------------------------------------------
    #                 try:
    #                     init_ckpt_path = train_config['train'].get('weight_init', None)
    #                     if init_ckpt_path is None or not os.path.isfile(init_ckpt_path):
    #                         logger.info("Block-wise token MSE skipped (no valid train.train.weight_init)")
    #                     else:
    #                         # Build reference DiT using in_channels inferred from init checkpoint
    #                         ckpt_raw = torch.load(init_ckpt_path, map_location='cpu')
    #                         sd = ckpt_raw.get('model', None)
    #                         if sd is None:
    #                             sd = ckpt_raw.get('state_dict', ckpt_raw)
    #                         key_pe_w = None
    #                         for kkey in list(sd.keys()):
    #                             if kkey.endswith('x_embedder.proj.weight'):
    #                                 key_pe_w = kkey
    #                                 break
    #                         if key_pe_w is None:
    #                             logger.info("Block-wise token MSE skipped (init ckpt missing x_embedder.proj.weight)")
    #                         else:
    #                             in_chans_ref = int(sd[key_pe_w].shape[1])
    #                             # Instantiate ref model
    #                             ref_ctor = LightningDiT_models[train_config['model']['model_type']]
    #                             ref_dit = ref_ctor(
    #                                 input_size=latent_size,
    #                                 num_classes=train_config['data']['num_classes'],
    #                                 use_qknorm=train_config['model']['use_qknorm'],
    #                                 use_swiglu=train_config['model'].get('use_swiglu', False),
    #                                 use_rope=train_config['model'].get('use_rope', False),
    #                                 use_rmsnorm=train_config['model'].get('use_rmsnorm', False),
    #                                 wo_shift=train_config['model'].get('wo_shift', False),
    #                                 in_channels=in_chans_ref,
    #                                 use_checkpoint=False,
    #                             ).to(device)
    #                             ref_dit.eval()
    #                             # Load weights with shape-aware helper (use same adapt_mode as main training)
    #                             ref_pack = {'model': {k.replace('module.', ''): v for k, v in sd.items()}}
    #                             adapt_mode = train_config['train'].get('weight_init_adapt_mode', 'zero_pad')
    #                             load_weights_with_shape_check(ref_dit, ref_pack, rank=rank, adapt_mode=adapt_mode)

    #                             # Prepare inputs and labels/timesteps
    #                             num = k  # use same k images
    #                             # labels: reuse fixed_classes subset
    #                             if fixed_classes is not None and len(fixed_classes) > 0:
    #                                 y_list = fixed_classes[:num]
    #                                 y_smpl = torch.tensor(y_list, device=device, dtype=torch.long)
    #                             else:
    #                                 y_smpl = torch.randint(0, train_config['data']['num_classes'], (num,), device=device)
    #                             # timesteps: use a fixed mid-range value
    #                             t_smpl = torch.full((num,), 0.5, device=device, dtype=torch.float32)

    #                             # Compute z_b and z_hc for the same images
    #                             imgs_in = imgs_t[:num].to(device) * 2.0 - 1.0
    #                             model_dtype = next(vae.model.parameters()).dtype
    #                             imgs_in = imgs_in.to(dtype=model_dtype)
    #                             base_post = vae.model.encode_base(imgs_in)
    #                             z_b = base_post.mode().detach()
    #                             z_b = vae.model.loss._apply_pe_latent_normalization(z_b)
    #                             hc_post = vae.model.encode_hc(imgs_in)
    #                             z_hc = hc_post.mode().detach()

    #                             # Compare raw patch embeddings: self.x_embedder(x) for ref(z_b) vs cur(z_hc)
    #                             try:
    #                                 with torch.no_grad():
    #                                     xb_ref = z_b.to(device=ref_dit.x_embedder.proj.weight.device, dtype=ref_dit.x_embedder.proj.weight.dtype)
    #                                     xb_cur = z_hc.to(device=cur_model.x_embedder.proj.weight.device, dtype=cur_model.x_embedder.proj.weight.dtype)
    #                                     pe_ref = ref_dit.x_embedder(xb_ref)
    #                                     pe_cur = cur_model.x_embedder(xb_cur)
    #                                     a = pe_ref.detach().float().cpu()
    #                                     b = pe_cur.detach().float().cpu()
    #                                     if a.shape == b.shape:
    #                                         diff = a - b
    #                                         pe_mse = float(torch.mean(diff * diff).item())
    #                                         pe_max = float(torch.max(torch.abs(diff)).item())
    #                                         logger.info(f"PatchEmbed token MSE (ref z_b vs cur z_hc): mse={pe_mse:.6e}, maxabs={pe_max:.3e}")
    #                                         if SummaryWriter is not None:
    #                                             writer.add_scalar('debug/patch_embed_mse', pe_mse, 0)
    #                                             writer.add_scalar('debug/patch_embed_maxabs', pe_max, 0)
    #                                         try:
    #                                             import wandb as _wandb
    #                                             if _wandb.run is not None:
    #                                                 _wandb.log({'debug/patch_embed_mse': pe_mse, 'debug/patch_embed_maxabs': pe_max}, step=0)
    #                                         except Exception:
    #                                             pass
    #                                     else:
    #                                         logger.info(f"PatchEmbed compare skipped due to shape mismatch: ref={tuple(a.shape)} cur={tuple(b.shape)}")
    #                             except Exception as e:
    #                                 logger.info(f"PatchEmbed compare failed: {e}")

    #                             # Compare ref.x_embedder(z_b) vs pe_teacher(z_b) on the same input
    #                             try:
    #                                 pe_teacher = getattr(getattr(vae.model, 'loss', None), 'pe_teacher', None)
    #                                 if pe_teacher is not None:
    #                                     zb_t = z_b
    #                                     if hasattr(vae.model.loss, 'pe_latent_norm_enable') and vae.model.loss.pe_latent_norm_enable and hasattr(vae.model.loss, '_apply_pe_latent_normalization'):
    #                                         zb_t = vae.model.loss._apply_pe_latent_normalization(zb_t)
    #                                     # dtype/device align per module
    #                                     xb_ref_t = zb_t.to(device=ref_dit.x_embedder.proj.weight.device, dtype=ref_dit.x_embedder.proj.weight.dtype)
    #                                     xb_teacher = zb_t.to(device=pe_teacher.weight.device, dtype=pe_teacher.weight.dtype)
    #                                     with torch.no_grad():
    #                                         toks_ref = ref_dit.x_embedder(xb_ref_t)  # (N, T, D)
    #                                         feat_teacher = pe_teacher(xb_teacher)   # (N, D, H, W)
    #                                         toks_teacher = feat_teacher.permute(0, 2, 3, 1).reshape(feat_teacher.shape[0], -1, feat_teacher.shape[1])
    #                                     A = toks_ref.detach().float().cpu()
    #                                     B = toks_teacher.detach().float().cpu()
    #                                     if A.shape == B.shape:
    #                                         diff = A - B
    #                                         mse = float(torch.mean(diff * diff).item())
    #                                         mx = float(torch.max(torch.abs(diff)).item())
    #                                         logger.info(f"PatchEmbed ref vs pe_teacher tokens (on z_b): mse={mse:.6e}, maxabs={mx:.3e}")
    #                                         if SummaryWriter is not None:
    #                                             writer.add_scalar('debug/patch_embed_ref_vs_teacher_mse', mse, 0)
    #                                             writer.add_scalar('debug/patch_embed_ref_vs_teacher_maxabs', mx, 0)
    #                                         try:
    #                                             import wandb as _wandb
    #                                             if _wandb.run is not None:
    #                                                 _wandb.log({'debug/patch_embed_ref_vs_teacher_mse': mse, 'debug/patch_embed_ref_vs_teacher_maxabs': mx}, step=0)
    #                                         except Exception:
    #                                             pass
    #                                     else:
    #                                         logger.info(f"PatchEmbed ref vs teacher skipped (shape mismatch): ref={tuple(A.shape)} teacher={tuple(B.shape)}")
    #                                 else:
    #                                     logger.info("PatchEmbed ref vs teacher skipped (no pe_teacher)")
    #                             except Exception as e:
    #                                 logger.info(f"PatchEmbed ref vs teacher failed: {e}")

    #                             # Compare cur.x_embedder(z_hc) vs pe_student(z_hc) on the same input
    #                             try:
    #                                 pe_student = getattr(vae.model, 'pe_student', None)
    #                                 if pe_student is not None:
    #                                     zh_s = z_hc
    #                                     xb_cur_s = zh_s.to(device=cur_model.x_embedder.proj.weight.device, dtype=cur_model.x_embedder.proj.weight.dtype)
    #                                     xb_student = zh_s.to(device=pe_student.weight.device, dtype=pe_student.weight.dtype)
    #                                     with torch.no_grad():
    #                                         toks_cur = cur_model.x_embedder(xb_cur_s)   # (N, T, D)
    #                                         feat_student = pe_student(xb_student)       # (N, D, H, W)
    #                                         toks_student = feat_student.permute(0, 2, 3, 1).reshape(feat_student.shape[0], -1, feat_student.shape[1])
    #                                     A = toks_cur.detach().float().cpu()
    #                                     B = toks_student.detach().float().cpu()
    #                                     if A.shape == B.shape:
    #                                         diff = A - B
    #                                         mse = float(torch.mean(diff * diff).item())
    #                                         mx = float(torch.max(torch.abs(diff)).item())
    #                                         logger.info(f"PatchEmbed cur vs pe_student tokens (on z_hc): mse={mse:.6e}, maxabs={mx:.3e}")
    #                                         if SummaryWriter is not None:
    #                                             writer.add_scalar('debug/patch_embed_cur_vs_student_mse', mse, 0)
    #                                             writer.add_scalar('debug/patch_embed_cur_vs_student_maxabs', mx, 0)
    #                                         try:
    #                                             import wandb as _wandb
    #                                             if _wandb.run is not None:
    #                                                 _wandb.log({'debug/patch_embed_cur_vs_student_mse': mse, 'debug/patch_embed_cur_vs_student_maxabs': mx}, step=0)
    #                                         except Exception:
    #                                             pass
    #                                     else:
    #                                         logger.info(f"PatchEmbed cur vs student skipped (shape mismatch): cur={tuple(A.shape)} student={tuple(B.shape)}")
    #                                 else:
    #                                     logger.info("PatchEmbed cur vs student skipped (no pe_student)")
    #                             except Exception as e:
    # #                                 logger.info(f"PatchEmbed cur vs student failed: {e}")

    #                             # Extract per-block token states
    #                             with torch.no_grad():
    #                                 out_ref = ref_dit.extract_block_outputs(z_b.to(device=ref_dit.x_embedder.proj.weight.device, dtype=ref_dit.x_embedder.proj.weight.dtype),
    #                                                                         t_smpl, y_smpl,
    #                                                                         return_token_states=True,
    #                                                                         return_image_predictions=False,
    #                                                                         detach=True, to_cpu=True)
    #                                 out_cur = cur_model.extract_block_outputs(z_hc.to(device=cur_model.x_embedder.proj.weight.device, dtype=cur_model.x_embedder.proj.weight.dtype),
    #                                                                           t_smpl, y_smpl,
    #                                                                           return_token_states=True,
    #                                                                           return_image_predictions=False,
    #                                                                           detach=True, to_cpu=True)

    #                             tok_ref = out_ref.get('token_states', [])
    #                             tok_cur = out_cur.get('token_states', [])
    #                             depth_cmp = min(len(tok_ref), len(tok_cur))
    #                             per_block_mse = []
    #                             for bi in range(depth_cmp):
    #                                 a = tok_ref[bi].float()
    #                                 b = tok_cur[bi].float()
    #                                 if a.shape == b.shape:
    #                                     diff = (a - b)
    #                                     mse = float(torch.mean(diff * diff).item())
    #                                 else:
    #                                     mse = float('nan')
    #                                 per_block_mse.append(mse)

    #                             # Log results
    #                             logger.info("Per-block token MSE (ref z_b vs cur z_hc): " + ", ".join([f"b{idx}={val:.3e}" for idx, val in enumerate(per_block_mse)]))
    #                             if SummaryWriter is not None:
    #                                 for idx, val in enumerate(per_block_mse):
    #                                     if not (val is None or (isinstance(val, float) and (val != val))):
    #                                         writer.add_scalar(f'debug/block_token_mse/b{idx}', val, 0)
    #                             try:
    #                                 import wandb as _wandb
    #                                 if _wandb.run is not None:
    #                                     log_dict = {f'debug/block_token_mse/b{idx}': val for idx, val in enumerate(per_block_mse) if not (val is None or (isinstance(val, float) and (val != val)))}
    #                                     if log_dict:
    #                                         _wandb.log(log_dict, step=0)
    #                             except Exception:
    #                                 pass
    #                 except Exception as e:
    #                     logger.info(f"Block-wise token MSE computation failed: {e}")
                    break
        except Exception as e:
            if accelerator.is_main_process:
                logger.info(f"Initial VAE decode check failed: {e}")

    # initial train sample at step 0 to visualize starting point
    if accelerator.is_main_process and vae is not None and latent_mean is not None and latent_std is not None:
        try:
            # Use current model (not EMA) at step 0 to avoid EMA lag producing noise
            cur_model = _unwrap(model)
            cur_model.eval()
            with torch.no_grad():
                sampler = Sampler(transport)
                mode = train_config['sample']['mode']
                if mode == "ODE":
                    sample_fn = sampler.sample_ode(
                        sampling_method=train_config['sample']['sampling_method'],
                        num_steps=train_config['sample']['num_sampling_steps'],
                        atol=train_config['sample']['atol'],
                        rtol=train_config['sample']['rtol'],
                        reverse=train_config['sample']['reverse'],
                        timestep_shift=train_config['sample'].get('timestep_shift', 0.0),
                    )
                else:
                    sample_fn = None
                if sample_fn is None:
                    raise RuntimeError(f"Unsupported sampling mode: {train_config['sample']['mode']}")

                num = min(vis_batch_size, vis_num_images)
                # build labels from fixed classes if provided
                if fixed_classes is not None and len(fixed_classes) > 0:
                    y_list = fixed_classes[:num]
                    y_smpl = torch.tensor(y_list, device=device, dtype=torch.long)
                else:
                    y_smpl = torch.randint(0, train_config['data']['num_classes'], (num,), device=device)
                # robustly determine in_channels from config (avoid DDP attribute access)
                in_chans_cfg = train_config['model'].get('in_chans', None)
                in_chans = in_chans_cfg if in_chans_cfg is not None else getattr(cur_model, 'in_channels', 4)
                z = torch.randn(len(y_smpl), in_chans, latent_size, latent_size, device=device)

                cfg_scale = train_config['sample']['cfg_scale']
                cfg_interval_start = train_config['sample'].get('cfg_interval_start', 0.0)
                using_cfg = cfg_scale > 1.0
                if using_cfg:
                    z = torch.cat([z, z], 0)
                    y_null = torch.tensor([1000] * len(y_smpl), device=device)
                    y_smpl = torch.cat([y_smpl, y_null], 0)
                    model_kwargs = dict(y=y_smpl, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=cfg_interval_start)
                    model_fn = cur_model.forward_with_cfg
                else:
                    model_kwargs = dict(y=y_smpl)
                    model_fn = cur_model.forward

                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                if using_cfg:
                    samples, _ = samples.chunk(2, dim=0)

                samples = _denormalize_latents(samples)
                vavae_imgs = vavae.decode_to_images(samples[:,:32])
                vavae_imgs_t = torch.from_numpy(vavae_imgs).permute(0, 3, 1, 2).float() / 255.0
                k = min(vis_num_images, vavae_imgs_t.shape[0])
                nrow = min(vis_nrow, k) if vis_nrow > 0 else k
                grid = torchvision.utils.make_grid(vavae_imgs_t[:k], nrow=nrow)
                out_path = f"{samples_dir}/train_step_{0:07d}_vavae.png"
                torchvision.utils.save_image(grid, out_path)
                if SummaryWriter is not None:
                    writer.add_image('samples/train_grid_vavae', grid, global_step=0)
                imgs = vae.decode_to_images(samples)
                imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
                k = min(vis_num_images, imgs_t.shape[0])
                nrow = min(vis_nrow, k) if vis_nrow > 0 else k
                grid = torchvision.utils.make_grid(imgs_t[:k], nrow=nrow)
                out_path = f"{samples_dir}/train_step_{0:07d}.png"
                torchvision.utils.save_image(grid, out_path)
                if SummaryWriter is not None:
                    writer.add_image('samples/train_grid', grid, global_step=0)
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({'samples/train_grid': [wandb.Image(out_path)]}, step=0)
                except Exception:
                    pass
        except Exception as e:
            if accelerator.is_main_process:
                logger.info(f"Initial sampling failed: {e}")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    train_config['train']['resume'] = train_config['train']['resume'] if 'resume' in train_config['train'] else False

    if train_config['train']['resume']:
        # check if the checkpoint exists
        checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: os.path.getsize(x))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'])
            # opt.load_state_dict(checkpoint['opt'])
            ema.load_state_dict(checkpoint['ema'])
            train_steps = int(latest_checkpoint.split('/')[-1].split('.')[0])
            if accelerator.is_main_process:
                logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        else:
            if accelerator.is_main_process:
                logger.info("No checkpoint found. Starting training from scratch.")
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    if not train_config['train']['resume']:
        train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    use_checkpoint = train_config['train']['use_checkpoint'] if 'use_checkpoint' in train_config['train'] else True
    ema_decay = float(train_config['train'].get('ema_decay', 0.999))
    if accelerator.is_main_process:
        logger.info(f"Using checkpointing: {use_checkpoint}")
        logger.info(f"EMA decay: {ema_decay}")

    while True:
        for x, y in loader:
            if accelerator.mixed_precision == 'no':
                x = x.to(device, dtype=torch.float32)
                y = y
            else:
                x = x.to(device)
                y = y.to(device)
            model_kwargs = dict(y=y)
            loss_weight_mask = None
            if channel_loss_scheduler is not None:
                loss_weight_mask = channel_loss_scheduler.get_mask(
                    step=train_steps,
                    total_channels=x.shape[1],
                    ndim=x.dim(),
                    device=x.device,
                    dtype=x.dtype,
                )
            loss_dict = transport.training_losses(model, x, model_kwargs, loss_weight_mask=loss_weight_mask)
            if 'cos_loss' in loss_dict:
                mse_loss = loss_dict["loss"].mean()
                loss = loss_dict["cos_loss"].mean() + mse_loss
            else:
                loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            if 'max_grad_norm' in train_config['optimizer']:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
            opt.step()
            update_ema(ema, model.module, decay=ema_decay)

            # Log loss values:
            if 'cos_loss' in loss_dict:
                running_loss += mse_loss.item()
            else:
                running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % train_config['train']['log_every'] == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    writer.add_scalar('Loss/train', avg_loss, train_steps)
                    # log to wandb if enabled
                    try:
                        import wandb  # local import in case rank!=0 didn't import
                        if wandb.run is not None:
                            wandb.log({
                                'Loss/train': avg_loss,
                                'train/steps_per_sec': steps_per_sec,
                            }, step=train_steps)
                    except Exception:
                        pass
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % train_config['train']['ckpt_every'] == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "config": train_config,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    if accelerator.is_main_process:
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                        logger.info(f"Syncing to S3 mirror (ckpt, exp, tb): {checkpoint_dir} | {experiment_dir} | {tensorboard_dir_log}")
                        _sync_path_to_s3(checkpoint_dir)
                        _sync_path_to_s3(experiment_dir)
                        _sync_path_to_s3(tensorboard_dir_log)
                dist.barrier()

                # Evaluate on validation set
                if 'valid_path' in train_config['data']:
                    if accelerator.is_main_process:
                        logger.info(f"Start evaluating at step {train_steps}")
                    val_loss = evaluate(
                        model,
                        valid_loader,
                        device,
                        transport,
                        sp_timesteps=(0.0, 1.0),
                        channel_loss_scheduler=channel_loss_scheduler,
                        current_step=train_steps,
                    )
                    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                    val_loss = val_loss.item() / dist.get_world_size()
                    if accelerator.is_main_process:
                        logger.info(f"Validation Loss: {val_loss:.4f}")
                        writer.add_scalar('Loss/validation', val_loss, train_steps)
                        # log to wandb if enabled
                        try:
                            import wandb
                            if wandb.run is not None:
                                wandb.log({'Loss/validation': val_loss}, step=train_steps)
                        except Exception:
                            pass
                    model.train()

            # Interval sampling: produce both current-model and EMA samples for comparison
            sample_every = train_config['train'].get('sample_every', None)
            if sample_every is None:
                # default: align with checkpoint interval
                sample_every = train_config['train']['ckpt_every']
            if (
                (train_steps % sample_every == 0)
                and train_steps > 0
                and accelerator.is_main_process
                and vae is not None
                and latent_mean is not None
                and latent_std is not None
            ):
                try:
                    with torch.no_grad():
                        # build sampler
                        sampler = Sampler(transport)
                        mode = train_config['sample']['mode']
                        if mode == "ODE":
                            sample_fn = sampler.sample_ode(
                                sampling_method=train_config['sample']['sampling_method'],
                                num_steps=train_config['sample']['num_sampling_steps'],
                                atol=train_config['sample']['atol'],
                                rtol=train_config['sample']['rtol'],
                                reverse=train_config['sample']['reverse'],
                                timestep_shift=train_config['sample'].get('timestep_shift', 0.0),
                            )
                        else:
                            sample_fn = None
                        if sample_fn is None:
                            raise RuntimeError(f"Unsupported sampling mode: {train_config['sample']['mode']}")

                        # shared labels and noise to compare fairly between current and EMA
                        num = min(vis_batch_size, vis_num_images)
                        if fixed_classes is not None and len(fixed_classes) > 0:
                            y_list = fixed_classes[:num]
                            y_smpl_base = torch.tensor(y_list, device=device, dtype=torch.long)
                        else:
                            y_smpl_base = torch.randint(0, train_config['data']['num_classes'], (num,), device=device)
                        in_chans_cfg = train_config['model'].get('in_chans', None)
                        in_chans = in_chans_cfg if in_chans_cfg is not None else getattr(_unwrap(model), 'in_channels', 4)
                        z_base = torch.randn(len(y_smpl_base), in_chans, latent_size, latent_size, device=device)

                        cfg_scale = train_config['sample']['cfg_scale']
                        cfg_interval_start = train_config['sample'].get('cfg_interval_start', 0.0)
                        using_cfg = cfg_scale > 1.0

                        # sample from both current model and EMA
                        for tag, sample_model in [("cur", _unwrap(model)), ("ema", _unwrap(ema))]:
                            sample_model.eval()
                            core_model = sample_model

                            # clone labels/noise per branch to avoid in-place changes
                            y_smpl = y_smpl_base.clone()
                            z = z_base.clone()
                            if using_cfg:
                                z = torch.cat([z, z], 0)
                                y_null = torch.tensor([1000] * len(y_smpl), device=device)
                                y_smpl = torch.cat([y_smpl, y_null], 0)
                                model_kwargs = dict(y=y_smpl, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=cfg_interval_start)
                                model_fn = core_model.forward_with_cfg
                            else:
                                model_kwargs = dict(y=y_smpl)
                                model_fn = core_model.forward

                            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                            if using_cfg:
                                samples, _ = samples.chunk(2, dim=0)

                            samples = _denormalize_latents(samples)
                            imgs = vae.decode_to_images(samples)
                            imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
                            k = min(vis_num_images, imgs_t.shape[0])
                            nrow = min(vis_nrow, k) if vis_nrow > 0 else k
                            grid = torchvision.utils.make_grid(imgs_t[:k], nrow=nrow)
                            out_path = f"{samples_dir}/train_step_{train_steps:07d}_{tag}.png"
                            torchvision.utils.save_image(grid, out_path)
                            if SummaryWriter is not None:
                                writer.add_image(f'samples/train_grid_{tag}', grid, global_step=train_steps)
                            try:
                                import wandb
                                if wandb.run is not None:
                                    wandb.log({f'samples/train_grid_{tag}': [wandb.Image(out_path)]}, step=train_steps)
                            except Exception:
                                pass

                        _sync_path_to_s3(experiment_dir)
                except Exception as e:
                    if accelerator.is_main_process:
                        logger.info(f"Sampling at step {train_steps} failed: {e}")
            if train_steps >= train_config['train']['max_steps']:
                break
        if train_steps >= train_config['train']['max_steps']:
            break

    if accelerator.is_main_process:
        logger.info("Done!")
        # Close wandb if running
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

    return accelerator

def load_weights_with_shape_check(model, checkpoint, rank=0, adapt_mode='zero_pad'):
    """
    Load weights with shape adaptation support.
    
    Args:
        model: target model
        checkpoint: checkpoint dict with 'model' key
        rank: process rank for logging
        adapt_mode: 'zero_pad', 'repeat_average', or 'random'
            - 'zero_pad': copy partial weights, pad rest with zeros
            - 'repeat_average': repeat-fill weights (x_embedder divided by repeat count, final_layer direct repeat)
            - 'random': randomly initialize the mismatched parts
    """
    model_state_dict = model.state_dict()
    # check shape and load weights
    for name, param in checkpoint['model'].items():
        if name not in model_state_dict:
            if rank == 0:
                print(f"Parameter '{name}' not found in model, skipping.")
            continue

        target = model_state_dict[name]

        # Exact match: direct copy
        if param.shape == target.shape:
            target.copy_(param)
            continue

        # Adapt input patch embed conv when in_channels or kernel size changes
        if name == 'x_embedder.proj.weight' and param.ndim == 4 and target.ndim == 4:
            # Shapes: [hidden, in_c, kh, kw]
            hidden_old, in_c_old, kh_old, kw_old = param.shape
            hidden_new, in_c_new, kh_new, kw_new = target.shape
            if hidden_old != hidden_new and rank == 0:
                print(f"Skipping '{name}' due to hidden dim mismatch: ckpt {param.shape} vs model {target.shape}")
            else:
                weight = torch.zeros_like(target)
                if adapt_mode == 'zero_pad':
                    # Special case: 1x1 conv -> larger kernel (e.g., 1x1 -> 2x2)
                    if kh_old == 1 and kw_old == 1 and (kh_new > 1 or kw_new > 1):
                        # Place 1x1 weight at center (or top-left) of larger kernel
                        # Center position for even kernels: kh_new//2, kw_new//2
                        center_h = kh_new // 2
                        center_w = kw_new // 2
                        c = min(in_c_old, in_c_new)
                        weight[:, :c, center_h:center_h+1, center_w:center_w+1] = param[:, :c, 0:1, 0:1]
                        if rank == 0:
                            print(f"Adapted '{name}' from {param.shape} to {target.shape} (zero_pad: 1x1 centered at [{center_h},{center_w}], in_c={c})")
                    else:
                        # Original: copy partial and pad zeros
                        c = min(in_c_old, in_c_new)
                        kh = min(kh_old, kh_new)
                        kw = min(kw_old, kw_new)
                        weight[:, :c, :kh, :kw] = param[:, :c, :kh, :kw]
                        if rank == 0:
                            print(f"Adapted '{name}' from {param.shape} to {target.shape} (zero_pad: copied in_c={c}, kh={kh}, kw={kw})")
                elif adapt_mode == 'repeat_average':
                    # Repeat-fill and average for x_embedder
                    # Special case: 1x1 conv -> larger kernel (e.g., 1x1 -> 2x2)
                    # This is common when adapting from pointwise to spatial convs
                    if kh_old == 1 and kw_old == 1 and (kh_new > 1 or kw_new > 1):
                        # Replicate 1x1 weights to all spatial positions
                        for i in range(hidden_new):
                            for c in range(in_c_new):
                                src_c = c % in_c_old
                                src_i = i % hidden_old
                                # Broadcast the 1x1 weight to all kernel positions
                                weight[i, c, :, :] = param[src_i, src_c, 0, 0]
                        
                        # Calculate repeat factors
                        c_repeat = (in_c_new + in_c_old - 1) // in_c_old
                        spatial_repeat = kh_new * kw_new  # number of spatial positions filled from one 1x1 weight
                        
                        # Average: divide by channel repeat (for channel expansion)
                        # and by spatial repeat (since 1x1 weight is copied to all positions)
                        if in_c_new > in_c_old:
                            weight = weight / c_repeat
                        # Divide by spatial expansion factor
                        weight = weight / spatial_repeat
                        
                        if rank == 0:
                            print(f"Adapted '{name}' from {param.shape} to {target.shape} (repeat_average: 1x1->kxk, c_repeat={c_repeat}, spatial_repeat={spatial_repeat})")
                    else:
                        # General case: tile using modulo
                        for i in range(hidden_new):
                            for c in range(in_c_new):
                                for h in range(kh_new):
                                    for w in range(kw_new):
                                        src_h = h % kh_old
                                        src_w = w % kw_old
                                        src_c = c % in_c_old
                                        src_i = i % hidden_old
                                        weight[i, c, h, w] = param[src_i, src_c, src_h, src_w]
                        
                        # Count how many times each output position is filled from different input channels
                        # For channel dimension: if in_c_new > in_c_old, we repeat; divide by repeat times
                        c_repeat = (in_c_new + in_c_old - 1) // in_c_old  # ceil division
                        kh_repeat = (kh_new + kh_old - 1) // kh_old
                        kw_repeat = (kw_new + kw_old - 1) // kw_old
                        
                        # Average by the repeat count (only for input channels)
                        # The key is to average over input channel repeats
                        if in_c_new > in_c_old:
                            weight = weight / c_repeat
                        
                        if rank == 0:
                            print(f"Adapted '{name}' from {param.shape} to {target.shape} (repeat_average: c_repeat={c_repeat}, kh_repeat={kh_repeat}, kw_repeat={kw_repeat})")
                elif adapt_mode == 'random':
                    # Random initialization for mismatched dimensions
                    # Copy the overlapping part, then randomly initialize the rest
                    c = min(in_c_old, in_c_new)
                    kh = min(kh_old, kh_new)
                    kw = min(kw_old, kw_new)
                    
                    # Copy overlapping part
                    weight[:, :c, :kh, :kw] = param[:, :c, :kh, :kw]
                    
                    # Randomly initialize the rest using Kaiming initialization (similar to conv layers)
                    # For extra channels
                    if in_c_new > c:
                        torch.nn.init.kaiming_uniform_(weight[:, c:, :, :], a=math.sqrt(5))
                    # For extra kernel dimensions
                    if kh_new > kh or kw_new > kw:
                        # Reinitialize the entire tensor to ensure uniform distribution
                        temp_weight = torch.empty_like(weight)
                        torch.nn.init.kaiming_uniform_(temp_weight, a=math.sqrt(5))
                        # Copy back the overlapping part
                        temp_weight[:, :c, :kh, :kw] = param[:, :c, :kh, :kw]
                        weight = temp_weight
                    
                    if rank == 0:
                        print(f"Adapted '{name}' from {param.shape} to {target.shape} (random: copied in_c={c}, kh={kh}, kw={kw}, randomly initialized rest)")
                else:
                    raise ValueError(f"Unknown adapt_mode: {adapt_mode}")
                model_state_dict[name] = weight
            continue

        # Adapt final head when out_channels (token size) changes
        if name.endswith('final_layer.linear.weight') and param.ndim == 2 and target.ndim == 2:
            r_old, c_old = param.shape
            r_new, c_new = target.shape
            weight = torch.zeros_like(target)
            if adapt_mode == 'zero_pad':
                # Original: copy partial and pad zeros
                r = min(r_old, r_new)
                c = min(c_old, c_new)
                weight[:r, :c] = param[:r, :c]
                if rank == 0:
                    print(f"Adapted '{name}' from {param.shape} to {target.shape} (zero_pad: copied rows={r}, cols={c})")
            elif adapt_mode == 'repeat_average':
                # Repeat-fill for final_layer (no averaging)
                for i in range(r_new):
                    for j in range(c_new):
                        weight[i, j] = param[i % r_old, j % c_old]
                if rank == 0:
                    print(f"Adapted '{name}' from {param.shape} to {target.shape} (repeat_average: tiled)")
            elif adapt_mode == 'random':
                # Random initialization for mismatched dimensions
                # Copy the overlapping part, then randomly initialize the rest
                r = min(r_old, r_new)
                c = min(c_old, c_new)
                weight[:r, :c] = param[:r, :c]
                
                # Randomly initialize the rest using Kaiming initialization
                if r_new > r:
                    fan_in = c_new
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    torch.nn.init.uniform_(weight[r:, :], -bound, bound)
                if c_new > c:
                    fan_in = c_new
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    torch.nn.init.uniform_(weight[:, c:], -bound, bound)
                
                if rank == 0:
                    print(f"Adapted '{name}' from {param.shape} to {target.shape} (random: copied rows={r}, cols={c}, randomly initialized rest)")
            else:
                raise ValueError(f"Unknown adapt_mode: {adapt_mode}")
            model_state_dict[name] = weight
            continue

        if name.endswith('final_layer.linear.bias') and param.ndim == 1 and target.ndim == 1:
            r_old = param.shape[0]
            r_new = target.shape[0]
            bias = torch.zeros_like(target)
            if adapt_mode == 'zero_pad':
                # Original: copy partial and pad zeros
                r = min(r_old, r_new)
                bias[:r] = param[:r]
                if rank == 0:
                    print(f"Adapted '{name}' from {param.shape} to {target.shape} (zero_pad: copied {r})")
            elif adapt_mode == 'repeat_average':
                # Repeat-fill for final_layer bias (no averaging)
                for i in range(r_new):
                    bias[i] = param[i % r_old]
                if rank == 0:
                    print(f"Adapted '{name}' from {param.shape} to {target.shape} (repeat_average: tiled)")
            elif adapt_mode == 'random':
                # Random initialization for mismatched dimensions
                # Copy the overlapping part, then randomly initialize the rest
                r = min(r_old, r_new)
                bias[:r] = param[:r]
                
                # Randomly initialize the rest (typically bias is initialized to zero, but we use small random values)
                if r_new > r:
                    torch.nn.init.uniform_(bias[r:], -0.01, 0.01)
                
                if rank == 0:
                    print(f"Adapted '{name}' from {param.shape} to {target.shape} (random: copied {r}, randomly initialized rest)")
            else:
                raise ValueError(f"Unknown adapt_mode: {adapt_mode}")
            model_state_dict[name] = bias
            continue

        # Fallback: skip mismatched params
        if rank == 0:
            print(
                f"Skipping loading parameter '{name}' due to shape mismatch: "
                f"checkpoint shape {param.shape}, model shape {target.shape}"
            )

    # load state dict
    model.load_state_dict(model_state_dict, strict=False)
    
    return model

@torch.no_grad()
def evaluate(model, loader, device, transport, sp_timesteps=None, channel_loss_scheduler=None, current_step=None):
    """
    Simple validation loop that mirrors the training loss.
    Returns a scalar tensor (mean loss over batches) on the given device.
    """
    was_training = model.training
    model.eval()

    total = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        loss_weight_mask = None
        if channel_loss_scheduler is not None:
            step = current_step if current_step is not None else channel_loss_scheduler.end_step
            loss_weight_mask = channel_loss_scheduler.get_mask(
                step=step,
                total_channels=x.shape[1],
                ndim=x.dim(),
                device=x.device,
                dtype=x.dtype,
            )
        loss_dict = transport.training_losses(
            model,
            x,
            dict(y=y),
            sp_timesteps=sp_timesteps,
            loss_weight_mask=loss_weight_mask,
        )
        loss = loss_dict["loss"].mean()
        total += float(loss.item())
        count += 1

    if was_training:
        model.train()

    if count == 0:
        return torch.tensor(0.0, device=device)
    return torch.tensor(total / count, device=device)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/debug.yaml')
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)
    do_train(train_config, accelerator)