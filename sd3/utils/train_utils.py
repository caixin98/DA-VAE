"""Training utils for DA-VAE.

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
import json
import os
import sys
import time
import math
from pathlib import Path
import pprint
import glob
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from omegaconf import OmegaConf
from torch.optim import AdamW
import gc

from utils.lr_schedulers import get_scheduler
from modeling.modules import (
    EMAModel, ReconstructionLoss_Single_Stage,
)

try:
    from evaluator import VQGANEvaluator
except ImportError:
    VQGANEvaluator = None

try:
    from utils.viz_utils import make_viz_from_samples
except ImportError:
    make_viz_from_samples = None


def safe_cuda_cleanup():
    """安全清理CUDA内存和缓存"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    except Exception as e:
        print(f"CUDA清理警告: {e}")


def log_memory_usage(logger, stage="", device=None):
    """记录GPU内存使用情况"""
    if not torch.cuda.is_available():
        return
    
    if device is None:
        device = torch.cuda.current_device()
    
    try:
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        
        logger.info(f"{stage} - GPU {device} 内存: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB, 总计 {total:.2f}GB")
        
        if allocated / total > 0.9:
            logger.warning(f"⚠️ GPU {device} 内存使用率过高: {allocated/total*100:.1f}%")
            
    except Exception as e:
        logger.warning(f"无法获取GPU {device} 内存信息: {e}")


def get_model_dtype(config):
    """获取模型的数据类型"""
    if config.training.mixed_precision == "fp16":
        return torch.float16
    elif config.training.mixed_precision == "bf16":
        return torch.bfloat16
    else:
        return torch.float32


def optimize_memory_settings(config, logger):
    """根据配置优化内存设置"""
    logger.info("=== 内存优化设置 ===")
    
    if config.training.mixed_precision == "fp16":
        logger.info("✅ 启用FP16混合精度训练 - 可节省约30-50%显存")
    elif config.training.mixed_precision == "bf16":
        logger.info("✅ 启用BF16混合精度训练 - 可节省约30-50%显存，数值稳定性更好")
    else:
        logger.info("⚠️ 未启用混合精度训练 - 建议启用以节省显存")
    
    if config.model.get("enable_gradient_checkpointing", False):
        logger.info("✅ 启用梯度检查点 - 可节省约20-30%显存，但会降低训练速度")
    else:
        logger.info("⚠️ 未启用梯度检查点 - 建议启用以节省显存")
    
    if config.training.get("enable_tf32", False):
        logger.info("✅ 启用TF32 - 在Ampere GPU上可提高性能并减少显存")
    else:
        logger.info("ℹ️ TF32未启用")
    
    current_batch_size = config.training.per_gpu_batch_size
    if current_batch_size > 8:
        logger.info(f"⚠️ 当前批次大小较大 ({current_batch_size}) - 如果显存不足，建议减小到8或4")
    else:
        logger.info(f"✅ 当前批次大小合理: {current_batch_size}")
    
    if config.training.gradient_accumulation_steps == 1:
        logger.info("💡 建议启用梯度累积 (gradient_accumulation_steps > 1) 来减少单步显存占用")
    else:
        logger.info(f"✅ 已启用梯度累积: {config.training.gradient_accumulation_steps} 步")


def load_weights_with_flexible_matching(model, model_weight, logger, weight_path=""):
    """
    Load weights with flexible parameter matching for shape changes.
    
    Args:
        model: Current model instance
        model_weight: Loaded weight dictionary
        logger: Logger instance
        weight_path: Path to the weight file for logging
    
    Returns:
        tuple: (load_message, loaded_keys_count, skipped_keys_count, skipped_keys_list)
    """
    current_state = model.state_dict()
    
    load_state = {}
    skipped_keys = []
    loaded_keys = []
    
    for key, value in model_weight.items():
        if key in current_state:
            current_shape = current_state[key].shape
            checkpoint_shape = value.shape
            
            if key.endswith("final_layer.linear.weight") and len(current_shape) == 2 and len(checkpoint_shape) == 2:
                if current_shape[0] != checkpoint_shape[0]:
                    logger.info(f"Adapting {key} from {checkpoint_shape} to {current_shape} due to token_size change")
                    if current_shape[0] > checkpoint_shape[0]:
                        new_weight = torch.zeros(current_shape, dtype=value.dtype)
                        new_weight[:checkpoint_shape[0], :] = value
                        load_state[key] = new_weight
                    else:
                        load_state[key] = value[:current_shape[0], :]
                    loaded_keys.append(key)
                    continue
            
            elif key.endswith("final_layer.linear.bias") and len(current_shape) == 1 and len(checkpoint_shape) == 1:
                if current_shape[0] != checkpoint_shape[0]:
                    logger.info(f"Adapting {key} from {checkpoint_shape} to {current_shape} due to token_size change")
                    if current_shape[0] > checkpoint_shape[0]:
                        new_bias = torch.zeros(current_shape, dtype=value.dtype)
                        new_bias[:checkpoint_shape[0]] = value
                        load_state[key] = new_bias
                    else:
                        load_state[key] = value[:current_shape[0]]
                    loaded_keys.append(key)
                    continue
            
            elif key.endswith("learnable_latent_tokens") and len(current_shape) == 3 and len(checkpoint_shape) == 3:
                if current_shape != checkpoint_shape:
                    logger.info(f"Adapting {key} from {checkpoint_shape} to {current_shape} due to shape change")
                    
                    current_batch, current_tokens, current_hidden = current_shape
                    checkpoint_batch, checkpoint_tokens, checkpoint_hidden = checkpoint_shape
                    
                    new_tokens = torch.zeros(current_shape, dtype=value.dtype, device=value.device)
                    
                    min_tokens = min(current_tokens, checkpoint_tokens)
                    min_hidden = min(current_hidden, checkpoint_hidden)
                    new_tokens[0, :min_tokens, :min_hidden] = value[0, :min_tokens, :min_hidden]
                    
                    if current_tokens > checkpoint_tokens:
                        logger.info(f"Expanding num_latent_tokens from {checkpoint_tokens} to {current_tokens}")
                        scale = current_hidden ** -0.5
                        for i in range(checkpoint_tokens, current_tokens):
                            new_tokens[0, i, :] = scale * torch.randn(current_hidden, dtype=value.dtype, device=value.device)
                    
                    if current_hidden > checkpoint_hidden:
                        logger.info(f"Expanding hidden_size from {checkpoint_hidden} to {current_hidden}")
                        scale = current_hidden ** -0.5
                        for i in range(min_tokens):
                            new_tokens[0, i, checkpoint_hidden:] = scale * torch.randn(
                                current_hidden - checkpoint_hidden, dtype=value.dtype, device=value.device
                            )
                        for i in range(checkpoint_tokens, current_tokens):
                            new_tokens[0, i, checkpoint_hidden:] = scale * torch.randn(
                                current_hidden - checkpoint_hidden, dtype=value.dtype, device=value.device
                            )
                    
                    load_state[key] = new_tokens
                    loaded_keys.append(key)
                    continue
            
            elif key == "latent_tokens" and len(current_shape) == 2 and len(checkpoint_shape) == 2:
                if current_shape[0] != checkpoint_shape[0] or current_shape[1] != checkpoint_shape[1]:
                    logger.info(f"Adapting legacy {key} from {checkpoint_shape} to {current_shape} due to shape change")
                    
                    if current_shape[0] > checkpoint_shape[0]:
                        if current_shape[1] > checkpoint_shape[1]:
                            new_tokens = torch.zeros(current_shape, dtype=value.dtype)
                            new_tokens[:checkpoint_shape[0], :checkpoint_shape[1]] = value
                            scale = current_shape[1] ** -0.5
                            for i in range(checkpoint_shape[0], current_shape[0]):
                                new_tokens[i, :] = scale * torch.randn(current_shape[1], dtype=value.dtype)
                        else:
                            new_tokens = torch.zeros(current_shape, dtype=value.dtype)
                            new_tokens[:checkpoint_shape[0], :] = value[:, :current_shape[1]]
                            scale = current_shape[1] ** -0.5
                            for i in range(checkpoint_shape[0], current_shape[0]):
                                new_tokens[i, :] = scale * torch.randn(current_shape[1], dtype=value.dtype)
                    else:
                        if current_shape[1] > checkpoint_shape[1]:
                            new_tokens = torch.zeros(current_shape, dtype=value.dtype)
                            new_tokens[:, :checkpoint_shape[1]] = value[:current_shape[0], :]
                        else:
                            new_tokens = value[:current_shape[0], :current_shape[1]]
                    
                    load_state[key] = new_tokens
                    loaded_keys.append(key)
                    continue
            
            elif key == "decoder.decoder_embed.weight" and len(current_shape) == 2 and len(checkpoint_shape) == 2:
                if current_shape[1] != checkpoint_shape[1]:
                    logger.info(f"Adapting {key} from {checkpoint_shape} to {current_shape} due to token_size change")
                    if current_shape[1] > checkpoint_shape[1]:
                        new_weight = torch.zeros(current_shape, dtype=value.dtype)
                        new_weight[:, :checkpoint_shape[1]] = value
                        load_state[key] = new_weight
                    else:
                        load_state[key] = value[:, :current_shape[1]]
                    loaded_keys.append(key)
                    continue
            
            elif current_shape != checkpoint_shape:
                logger.warning(f"Skipping {key} due to shape mismatch: {checkpoint_shape} vs {current_shape}")
                skipped_keys.append(key)
                continue
            
            load_state[key] = value
            loaded_keys.append(key)
        else:
            logger.warning(f"Skipping {key} - not found in current model")
            skipped_keys.append(key)
    
    msg = model.load_state_dict(load_state, strict=False)
    
    if weight_path:
        logger.info(f"loading weight from {weight_path}")
    logger.info(f"Loaded {len(loaded_keys)} parameters, skipped {len(skipped_keys)} parameters")
    
    if skipped_keys:
        logger.info(f"Skipped keys: {skipped_keys}")
    
    return msg, len(loaded_keys), len(skipped_keys), skipped_keys


def get_config():
    """Reads configs from a yaml file and terminal."""
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


class AverageMeter(object):
    """Computes and stores the average and current value.
    
    This class is borrowed from
    https://github.com/pytorch/examples/blob/main/imagenet/main.py#L423
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [_move_to_device(v, device) for v in obj]
        return type(obj)(t) if isinstance(obj, tuple) else t
    return obj


class CUDAPrefetcher:
    def __init__(self, dataloader, device):
        self.loader = iter(dataloader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch = None
        self._preload()

    def _preload(self):
        try:
            next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = _move_to_device(next_batch, self.device)

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch


def create_model_and_loss_module(config, logger, accelerator,
                                 model_type="sd3_da_vae"):
    """Creates SD3 DA-VAE model and loss module."""
    logger.info("Creating model and loss module.")

    from modeling.modules.sd3_da_vae import SD3_DAAutoencoder as SD3DAVAE
    model_cls = SD3DAVAE
    loss_cls = ReconstructionLoss_Single_Stage

    # 检查是否使用DMD损失类型
    if config.losses.get("use_dmd_loss", False):
        from modeling.modules.dmd_loss_type import DMDLossType
        loss_cls = DMDLossType

    model_dtype = torch.float32
    if config.training.mixed_precision == "fp16":
        model_dtype = torch.float16
        logger.info("Creating model with float16 precision")
    elif config.training.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
        logger.info("Creating model with bfloat16 precision")
    
    model = model_cls(config)

    log_memory_usage(logger, "模型创建后")

    if config.experiment.get("init_weight", ""):
        init_weight_path = config.experiment.init_weight
        
        if not os.path.exists(init_weight_path):
            logger.info(f"🔍 权重文件不存在: {init_weight_path}")
            logger.info("🔄 尝试从S3自动下载...")
            
            try:
                from utils.weight_loader import load_weight_with_auto_download
                init_weight_path = load_weight_with_auto_download(init_weight_path, logger=logger)
                logger.info(f"✅ 权重文件下载成功: {init_weight_path}")
            except Exception as e:
                logger.error(f"❌ 权重文件下载失败: {e}")
                raise FileNotFoundError(f"无法加载权重文件: {init_weight_path}")
        
        model_weight = torch.load(init_weight_path, map_location="cpu", weights_only=True)
        msg, loaded_keys, skipped_keys, skipped_keys_list = load_weights_with_flexible_matching(model, model_weight, logger, init_weight_path)

    # Create the EMA model.
    ema_model = None
    if config.training.use_ema:
        ema_model = EMAModel(model.parameters(), decay=0.999,
                            model_cls=model_cls, config=config)
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"),
                                                  model_cls=model_cls, config=config)
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                ema_model.save_pretrained(os.path.join(output_dir, "ema_model"))

        accelerator.register_load_state_pre_hook(load_model_hook)
        accelerator.register_save_state_pre_hook(save_model_hook)

    # Create loss module along with discriminator
    if loss_cls is not None:
        import inspect
        init_params = inspect.signature(loss_cls.__init__).parameters
        if 'model' in init_params:
            loss_module = loss_cls(config=config, model=model)
        else:
            loss_module = loss_cls(config=config)
    else:
        loss_module = None

    log_memory_usage(logger, "损失模块创建后")

    log_memory_usage(logger, "模型初始化完成")
    
    if accelerator.is_main_process:
        optimize_memory_settings(config, logger)
    
    return model, ema_model, loss_module


def create_optimizer(config, logger, model, loss_module,
                     model_type="sd3_da_vae", need_discrminator=True):
    """Creates optimizer for model and discriminator."""
    logger.info("Creating optimizers.")
    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate

    optimizer_cls = AdamW

    exclude = (lambda n, p: p.ndim < 2 or "ln" in n or "bias" in n or 'latent_tokens' in n 
               or 'mask_token' in n or 'embedding' in n or 'norm' in n or 'gamma' in n or 'embed' in n)
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    
    # 新增：检查是否需要包含VAE decoder参数
    vae_decoder_params = []
    if hasattr(model, 'train_vae_decoder') and model.train_vae_decoder:
        if hasattr(model, 'vae_model') and model.vae_model is not None:
            logger.info("检测到Flux VAE decoder训练模式，将VAE decoder参数添加到优化器")
            for name, param in model.vae_model.vae.named_parameters():
                if 'decoder' in name and param.requires_grad:
                    vae_decoder_params.append(param)
                    logger.debug(f"添加VAE decoder参数: {name}")
            
            if vae_decoder_params:
                logger.info(f"找到 {len(vae_decoder_params)} 个VAE decoder参数")
                for i, param in enumerate(vae_decoder_params):
                    named_parameters.append((f"vae_decoder.{i}", param))
            else:
                logger.warning("未找到可训练的VAE decoder参数")
    
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    optimizer = optimizer_cls(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": optimizer_config.weight_decay},
        ],
        lr=learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2)
    )
    
    optimizer.encoder_lr = learning_rate
    optimizer.decoder_lr = learning_rate
    optimizer.base_lr = learning_rate
    optimizer.use_separate_lr = False

    # 检查是否使用对抗损失
    use_adversarial_loss = config.losses.get("use_adversarial_loss", True)
    
    # 检查是否使用DMD损失
    use_dmd_loss = config.losses.get("use_dmd_loss", False)
    
    # 处理对抗损失（判别器）
    if config.model.vq_model.get("finetune_decoder", False) or use_adversarial_loss:
        discriminator_learning_rate = optimizer_config.get("discriminator_learning_rate", 1e-5)
        
        discriminator_named_parameters = list(loss_module.named_parameters())
        
        discriminator_gain_or_bias_params = [p for n, p in discriminator_named_parameters if exclude(n, p) and p.requires_grad]
        discriminator_rest_params = [p for n, p in discriminator_named_parameters if include(n, p) and p.requires_grad]

        discriminator_optimizer = optimizer_cls(
            [
                {"params": discriminator_gain_or_bias_params, "weight_decay": 0.},
                {"params": discriminator_rest_params, "weight_decay": optimizer_config.weight_decay},
            ],
            lr=discriminator_learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2)
        )
    else:
        discriminator_optimizer = None

    # 处理DMD损失（单独的优化器）
    dmd_optimizer = None
    if use_dmd_loss and hasattr(loss_module, 'get_dmd_guidance_parameters'):
        logger.info("为DMD损失创建单独的优化器")
        dmd_learning_rate = optimizer_config.get("dmd_learning_rate", 1e-5)
        
        dmd_guidance_parameters = list(loss_module.get_dmd_guidance_parameters())
        dmd_named_parameters = [(f"dmd_guidance.{name}", param) for name, param in loss_module.dmd_guidance.named_parameters()]
        
        dmd_gain_or_bias_params = [p for n, p in dmd_named_parameters if exclude(n, p) and p.requires_grad]
        dmd_rest_params = [p for n, p in dmd_named_parameters if include(n, p) and p.requires_grad]
        
        dmd_optimizer = optimizer_cls(
            [
                {"params": dmd_gain_or_bias_params, "weight_decay": 0.},
                {"params": dmd_rest_params, "weight_decay": optimizer_config.weight_decay},
            ],
            lr=dmd_learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2)
        )

    return optimizer, discriminator_optimizer, dmd_optimizer


def create_lr_scheduler(config, logger, accelerator, optimizer, discriminator_optimizer=None, dmd_optimizer=None):
    """Creates learning rate scheduler for model and discriminator."""
    logger.info("Creating lr_schedulers.")
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps * accelerator.num_processes,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
        base_lr=config.lr_scheduler.params.learning_rate,
        end_lr=config.lr_scheduler.params.end_lr,
    )
    if discriminator_optimizer is not None:
        discriminator_lr = config.optimizer.params.get("discriminator_learning_rate", 1e-5)
        discriminator_start = config.losses.get("discriminator_start", 0)
        if discriminator_lr > 0:
            discriminator_lr_scheduler = get_scheduler(
                config.lr_scheduler.scheduler,
                optimizer=discriminator_optimizer,
                num_training_steps=config.training.max_train_steps * accelerator.num_processes - discriminator_start,
                num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
                base_lr=discriminator_lr,
                end_lr=config.lr_scheduler.params.end_lr,
            )
        else:
            discriminator_lr_scheduler = None
            logger.info("Discriminator learning rate is 0, skipping discriminator lr scheduler")
    else:
        discriminator_lr_scheduler = None
    
    # 为DMD优化器创建学习率调度器
    if dmd_optimizer is not None:
        dmd_lr = config.optimizer.params.get("dmd_learning_rate", 1e-5)
        if dmd_lr > 0:
            dmd_lr_scheduler = get_scheduler(
                config.lr_scheduler.scheduler,
                optimizer=dmd_optimizer,
                num_training_steps=config.training.max_train_steps * accelerator.num_processes,
                num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
                base_lr=dmd_lr,
                end_lr=config.lr_scheduler.params.end_lr,
            )
        else:
            dmd_lr_scheduler = None
            logger.info("DMD learning rate is 0, skipping DMD lr scheduler")
    else:
        dmd_lr_scheduler = None
    
    return lr_scheduler, discriminator_lr_scheduler, dmd_lr_scheduler


def create_evaluator(config, logger, accelerator):
    """Creates evaluator."""
    logger.info("Creating evaluator.")
    use_vae = config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False)
    evaluator = VQGANEvaluator(
        device=accelerator.device,
        enable_rfid=True,
        enable_inception_score=True,
        enable_codebook_usage_measure=False,
        enable_codebook_entropy_measure=False,
        enable_reconstruction_metrics=True,
        enable_latent_mse=use_vae,
    )
    return evaluator


def auto_resume(config, logger, accelerator, ema_model,
                num_update_steps_per_epoch, strict=True):
    """Auto resuming the training."""
    global_step = 0
    first_epoch = 0
    if config.experiment.resume:            
        accelerator.wait_for_everyone()
        
        if hasattr(config.experiment, 'resume_checkpoint') and config.experiment.resume_checkpoint and Path(config.experiment.resume_checkpoint).exists():
            checkpoint_path = Path(config.experiment.resume_checkpoint)
            logger.info(f"Loading from specified checkpoint: {checkpoint_path}")
            global_step = load_checkpoint(
                checkpoint_path,
                accelerator,
                logger=logger,
                strict=strict
            )
            if config.training.use_ema:
                ema_model.set_step(global_step)
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            local_ckpt_list = list(glob.glob(os.path.join(
                config.experiment.output_dir, "checkpoint*")))
            logger.info(f"All globbed checkpoints are: {local_ckpt_list}")
            if len(local_ckpt_list) >= 1:
                if len(local_ckpt_list) > 1:
                    fn = lambda x: int(x.split('/')[-1].split('-')[-1])
                    checkpoint_paths = sorted(local_ckpt_list, key=fn, reverse=True)
                else:
                    checkpoint_paths = local_ckpt_list
                global_step = load_checkpoint(
                    Path(checkpoint_paths[0]),
                    accelerator,
                    logger=logger,
                    strict=strict
                )
                if config.training.use_ema:
                    ema_model.set_step(global_step)
                first_epoch = global_step // num_update_steps_per_epoch
            else:
                logger.info("Training from scratch.")
    return global_step, first_epoch


def train_one_epoch(config, logger, accelerator,
                    model, ema_model, loss_module,
                    optimizer, discriminator_optimizer,
                    lr_scheduler, discriminator_lr_scheduler,
                    train_dataloader, eval_dataloader,
                    evaluator,
                    global_step,
                    model_type="sd3_da_vae",
                    clip_tokenizer=None,
                    clip_encoder=None,
                    pretrained_tokenizer=None,
                    original_vae_model=None):
    """One epoch training for SD3 DA-VAE."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()

    loss_module_unwrapped = accelerator.unwrap_model(loss_module)
    store_original_pixel_inputs = (
        getattr(loss_module_unwrapped, "use_original_pixel_supervision", False)
        or getattr(config.training, "enable_gradient_visualization", False)
    )

    if accelerator.is_main_process and accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    if accelerator.is_main_process and global_step % 100 == 0:
        log_memory_usage(logger, f"训练开始 (Step {global_step})")

    # 使用传入的original_vae_model参数进行图像编码（生成ground truth）
    if original_vae_model is None and hasattr(accelerator.unwrap_model(model), 'original_vae_model'):
        original_vae_model = accelerator.unwrap_model(model).original_vae_model
        logger.info("train_one_epoch: 从模型中获取原始VAE模型进行图像编码")
    elif original_vae_model is not None:
        logger.info("train_one_epoch: 使用传入的原始VAE模型进行图像编码")

    autoencoder_logs = defaultdict(float)
    discriminator_logs = defaultdict(float)
    last_discriminator_logs = defaultdict(float)
    oom_batch_records = []
    for i, batch in enumerate(train_dataloader):
        model.train()
        if "image" in batch:
            images = batch["image"].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )

            original_pixel_inputs = images.detach()
            
            model_dtype = get_model_dtype(config)
            images = images.to(model_dtype)

        fnames = batch["__key__"]
        data_time_meter.update(time.time() - end)

        try:
            with accelerator.accumulate([model, loss_module]):
                # SD3 DA-VAE 自编码器：输入使用像素图像；GT可选用SD3 VAE重建结果
                gt_images = original_pixel_inputs
                use_vae = config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False)
                if original_vae_model is not None and use_vae:
                    with torch.no_grad():
                        encoded = original_vae_model.encode(original_pixel_inputs)
                        gt_images = original_vae_model.decode(encoded)
                    gt_images = gt_images.to(get_model_dtype(config))

                reconstructed_images, extra_results_dict = model(images)
                if store_original_pixel_inputs:
                    extra_results_dict["original_pixel_inputs"] = original_pixel_inputs
                autoencoder_loss, loss_dict = loss_module(
                    gt_images,
                    reconstructed_images,
                    extra_results_dict,
                    global_step,
                    mode="generator",
                )

                # Gather the losses across all processes for logging.
                autoencoder_logs = {}
                for k, v in loss_dict.items():
                    if k in ["discriminator_factor", "d_weight", "in_warmup"]:
                        if type(v) == torch.Tensor:
                            autoencoder_logs["train/" + k] = v.cpu().item()
                        else:
                            autoencoder_logs["train/" + k] = v
                    else:
                        if type(v) == torch.Tensor:
                            autoencoder_logs["train/" + k] = accelerator.gather(v).mean().item()
                        else:
                            autoencoder_logs["train/" + k] = v
    
                accelerator.backward(autoencoder_loss)
    
                if (not accelerator.unwrap_model(loss_module).is_discriminator_warmup_period(global_step) and 
                    config.training.max_grad_norm is not None and accelerator.sync_gradients):
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
    
                if not accelerator.unwrap_model(loss_module).is_discriminator_warmup_period(global_step):
                    optimizer.step()
                    lr_scheduler.step()
    
                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                    and not accelerator.unwrap_model(loss_module).is_discriminator_warmup_period(global_step)
                ):
                    log_grad_norm(model, accelerator, global_step + 1)
    
                optimizer.zero_grad(set_to_none=True)
    
                # Train discriminator.
                discriminator_logs = defaultdict(float)
                loss_module_unwrapped = accelerator.unwrap_model(loss_module)
                
                discriminator_frequency = loss_module_unwrapped.get_discriminator_training_frequency(global_step)
                
                should_train_discriminator = (
                    discriminator_frequency > 0 
                    and global_step % discriminator_frequency == 0
                )
                
                if should_train_discriminator:
                    discriminator_loss, loss_dict_discriminator = loss_module(
                        images,
                        reconstructed_images,
                        extra_results_dict,
                        global_step=global_step,
                        mode="discriminator",
                    )
    
                    for k, v in loss_dict_discriminator.items():
                        if type(v) == torch.Tensor:
                            discriminator_logs["train/" + k] = accelerator.gather(v).mean().item()
                        else:
                            discriminator_logs["train/" + k] = v
    
                    discriminator_logs["train/discriminator_frequency"] = discriminator_frequency
    
                    accelerator.backward(discriminator_loss)
    
                    if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                        if discriminator_optimizer is not None:
                            discriminator_lr = discriminator_optimizer.param_groups[0]['lr']
                            skip_when_zero = config.optimizer.params.get("skip_discriminator_optimizer_when_lr_zero", False)
                            if discriminator_lr > 0 or not skip_when_zero:
                                accelerator.clip_grad_norm_(loss_module.parameters(), config.training.max_grad_norm)
    
                    if discriminator_optimizer is not None:
                        discriminator_lr = discriminator_optimizer.param_groups[0]['lr']
                        skip_when_zero = config.optimizer.params.get("skip_discriminator_optimizer_when_lr_zero", False)
                        
                        if discriminator_lr > 0 or not skip_when_zero:
                            discriminator_optimizer.step()
                            if discriminator_lr_scheduler is not None:
                                discriminator_lr_scheduler.step()
                            
                            if hasattr(loss_module_unwrapped, 'update_discriminator_ema'):
                                loss_module_unwrapped.update_discriminator_ema(global_step)
                    else:
                        if global_step % config.experiment.log_every == 0 and accelerator.is_main_process:
                            logger.info(f"Skipping discriminator optimizer step due to lr=0")
            
                    if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                    ):
                        log_grad_norm(loss_module, accelerator, global_step + 1)
                    
                    if discriminator_optimizer is not None:
                        discriminator_optimizer.zero_grad(set_to_none=True)
                    
                    last_discriminator_logs.clear()
                    for k, v in discriminator_logs.items():
                        last_discriminator_logs[k] = v
    
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                batch_shape = None
                max_hw = None
                if isinstance(batch, dict) and "image" in batch:
                    batch_shape = tuple(batch["image"].shape)
                    max_hw = tuple(batch["image"].shape[-2:])
                logger.warning(
                    f"Skipping batch at dataloader index {i} (global_step={global_step}) due to OOM. "
                    f"image_shape={batch_shape} max_hw={max_hw}"
                )
                if accelerator.is_main_process:
                    oom_batch_records.append({
                        "iteration": i,
                        "step": global_step,
                        "shape": batch_shape,
                        "max_hw": max_hw,
                    })
                safe_cuda_cleanup()
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                if discriminator_optimizer is not None:
                    discriminator_optimizer.zero_grad(set_to_none=True)
                if getattr(accelerator.state, "deepspeed_plugin", None) is None and hasattr(accelerator, "free_memory"):
                    accelerator.free_memory()
                continue
            raise
        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]
                compute_time = max(0.0, batch_time_meter.val - data_time_meter.val)
                loader_pct = (data_time_meter.val / batch_time_meter.val) if batch_time_meter.val > 0 else 0.0
                
                current_discriminator_frequency = loss_module_unwrapped.get_discriminator_training_frequency(global_step)
                
                peak_memory_gb = None
                if accelerator.device.type == "cuda":
                    peak_memory_gb = torch.cuda.max_memory_allocated(accelerator.device) / (1024 ** 3)

                log_message = (
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"Compute (t): {compute_time:0.4f} "
                    f"Loader%: {loader_pct*100:0.1f}% "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Total Loss: {autoencoder_logs.get('train/total_loss', 0.0):0.4f} "
                    f"Recon Loss: {autoencoder_logs.get('train/reconstruction_loss', 0.0):0.4f} "
                    f"D Freq: {current_discriminator_frequency}"
                )
                if peak_memory_gb is not None:
                    log_message += f" PeakMem: {peak_memory_gb:.2f}GB"
                logger.info(log_message)
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                    "time/compute_time": compute_time,
                    "time/loader_pct": loader_pct,
                    "discriminator_frequency": current_discriminator_frequency,
                }

                if peak_memory_gb is not None:
                    logs["memory/peak_allocated_gb"] = peak_memory_gb
                    torch.cuda.reset_peak_memory_stats(accelerator.device)
                
                if discriminator_optimizer is not None:
                    discriminator_lr = discriminator_optimizer.param_groups[0]['lr']
                    logs["lr/discriminator"] = discriminator_lr
                
                logs.update(autoencoder_logs)
                if discriminator_logs:
                    logs.update(discriminator_logs)
                elif last_discriminator_logs:
                    logs.update(last_discriminator_logs)
                accelerator.log(logs, step=global_step + 1)

                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger, config=config, loss_module=loss_module)
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                reconstruct_images(
                    model,
                    images[:config.training.num_generated_images],
                    fnames[:config.training.num_generated_images],
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config,
                    model_type=model_type,
                    original_vae_model=original_vae_model,
                    skip_vae_encoding=True
                )

                if config.training.get("use_ema", False):
                    ema_model.restore(model.parameters())

            # Evaluate reconstruction.
            if eval_dataloader is not None and (global_step + 1) % config.experiment.eval_every == 0:
                if accelerator.is_main_process:
                    logger.info(f"Computing metrics on the validation set (main process only).")
                    if config.training.get("use_ema", False):
                        ema_model.store(model.parameters())
                        ema_model.copy_to(model.parameters())
                        eval_scores = eval_reconstruction(
                            model,
                            eval_dataloader,
                            accelerator,
                            evaluator,
                            model_type=model_type,
                            original_vae_model=original_vae_model,
                            config=config,
                            logger=logger
                        )
                        logger.info(
                            f"EMA EVALUATION "
                            f"Step: {global_step + 1} "
                        )
                        logger.info(pprint.pformat(eval_scores))
                        eval_log = {f'ema_eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)
                        if config.training.get("use_ema", False):
                            ema_model.restore(model.parameters())
                    else:
                        eval_scores = eval_reconstruction(
                            model,
                            eval_dataloader,
                            accelerator,
                            evaluator,
                            model_type=model_type,
                            original_vae_model=original_vae_model,
                            config=config,
                            logger=logger
                        )

                        logger.info(
                            f"Non-EMA EVALUATION "
                            f"Step: {global_step + 1} "
                        )
                        logger.info(pprint.pformat(eval_scores))
                        eval_log = {f'eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)
                else:
                    logger.info(f"Waiting for main process to complete evaluation...")

                accelerator.wait_for_everyone()

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break

    if accelerator.is_main_process:
        log_memory_usage(logger, f"训练结束 (Step {global_step})")
        if global_step % 1000 == 0:
            safe_cuda_cleanup()
            logger.info("已执行定期内存清理")

    if accelerator.is_main_process and oom_batch_records:
        logger.warning(f"本 epoch 共跳过 {len(oom_batch_records)} 个 batch（OOM）")
        for record in oom_batch_records:
            logger.warning(
                f"  step={record['step']} iter={record['iteration']} shape={record['shape']} max_hw={record['max_hw']}"
            )

    return global_step


@torch.no_grad()
def eval_reconstruction(
    model,
    eval_loader,
    accelerator,
    evaluator,
    model_type="sd3_da_vae",
    clip_tokenizer=None,
    clip_encoder=None,
    pretrained_tokenizer=None,
    original_vae_model=None,
    max_eval_images=1000,
    config=None,
    logger=None,
):
    model.eval()
    evaluator.reset_metrics()
    local_model = accelerator.unwrap_model(model)
    
    if original_vae_model is None and hasattr(local_model, 'original_vae_model'):
        original_vae_model = local_model.original_vae_model
        if logger:
            logger.info("eval_reconstruction: 从模型中获取原始VAE模型进行图像编码")
    elif original_vae_model is not None and logger:
        logger.info("eval_reconstruction: 使用传入的原始VAE模型进行图像编码")
    
    processed_images = 0

    # Optional: save a small subset of eval images (original/reconstructed)
    save_eval_images = False
    save_eval_max = 0
    save_eval_dir = None
    try:
        if config is not None and hasattr(config, 'experiment'):
            save_eval_images = bool(getattr(config.experiment, 'save_eval_images', False))
            save_eval_max = int(getattr(config.experiment, 'save_eval_max', 16))
            if save_eval_images and hasattr(config.experiment, 'output_dir'):
                ts_folder = f"run_{int(time.time())}"
                save_eval_dir = Path(config.experiment.output_dir) / "eval_images" / ts_folder
                save_eval_dir.mkdir(parents=True, exist_ok=True)
    except Exception as _:
        save_eval_images = False
        save_eval_dir = None

    for batch in eval_loader:
        images = batch["image"].to(
            accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
        )
        
        original_images = torch.clone(images)
        
        batch_size = images.shape[0]
        if processed_images + batch_size > max_eval_images:
            remaining = max_eval_images - processed_images
            images = images[:remaining]
            original_images = original_images[:remaining]

        # SD3 DA-VAE：直接前向得到像素重建
        reconstructed_images, model_dict = local_model(images, sample_posterior=False)

        # Pixel模型：不解码模型输出；将GT替换为VAE重建像素（如果使用VAE）
        if original_vae_model is not None:
            use_vae = config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False)
            if use_vae:
                with torch.no_grad():
                    encoded = original_vae_model.encode(original_images)
                    original_images = original_vae_model.decode(encoded)

        reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
        reconstructed_images = torch.round(reconstructed_images * 255.0) / 255.0
        original_images = torch.clamp(original_images, 0.0, 1.0)

        # Optionally save a subset of eval samples
        if save_eval_images and save_eval_dir is not None and processed_images < save_eval_max:
            try:
                remain = save_eval_max - processed_images
                n_save = min(remain, original_images.shape[0])
                orig_cpu = (
                    original_images[:n_save]
                    .detach()
                    .to(dtype=torch.float32, device="cpu")
                    .clamp(0.0, 1.0)
                )
                recon_cpu = (
                    reconstructed_images[:n_save]
                    .detach()
                    .to(dtype=torch.float32, device="cpu")
                    .clamp(0.0, 1.0)
                )
                for i in range(n_save):
                    idx = processed_images + i
                    from torchvision.transforms.functional import to_pil_image
                    to_pil_image(orig_cpu[i]).save(save_eval_dir / f"{idx:06d}_original.png")
                    to_pil_image(recon_cpu[i]).save(save_eval_dir / f"{idx:06d}_reconstructed.png")
            except Exception as e:
                if logger:
                    logger.warning(f"保存评估图像失败: {e}")

        evaluator.update(original_images, reconstructed_images.squeeze(2), None)
        
        processed_images += images.shape[0]
        
        if processed_images >= max_eval_images:
            break
            
    model.train()
    results = evaluator.result()
    return results


@torch.no_grad()
def reconstruct_images(model, original_images, fnames, accelerator, 
                    global_step, output_dir, logger, config=None,
                    model_type="sd3_da_vae", text_guidance=None, cond_images=None,   
                    pretrained_tokenizer=None, original_vae_model=None, skip_vae_encoding=False):
    logger.info("Reconstructing images...")
    original_images = torch.clone(original_images)
    original_images = original_images.to(dtype=next(model.parameters()).dtype, device=accelerator.device)
    
    local_model = accelerator.unwrap_model(model)
    if original_vae_model is None and hasattr(local_model, 'original_vae_model'):
        original_vae_model = local_model.original_vae_model
        logger.info("reconstruct_images: 从模型中获取原始VAE模型进行图像编码")
    elif original_vae_model is not None:
        logger.info("reconstruct_images: 使用传入的原始VAE模型进行图像编码")
    
    original_images_for_viz = torch.clone(original_images)

    model.eval()
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
        # Direct forward to get pixel reconstructions
        reconstructed_images, _ = accelerator.unwrap_model(model)(original_images, sample_posterior=False)

    images_for_saving, images_for_logging = make_viz_from_samples(
        original_images_for_viz,
        reconstructed_images,
        None
    )
    # Log images.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {f"Train Reconstruction": images_for_saving},
            step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Reconstruction": images_for_logging}, step=global_step
        )
    # Log locally.
    root = Path(output_dir) / "train_images"
    os.makedirs(root, exist_ok=True)
    for i,img in enumerate(images_for_saving):
        filename = f"{global_step:08}_s-{i:03}-{fnames[i]}.png"
        path = os.path.join(root, filename)
        img.save(path)

    model.train()


def save_checkpoint(model, output_dir, accelerator, global_step, logger, config=None, loss_module=None) -> Path:
    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    accelerator.wait_for_everyone()
    
    accelerator.save_state(save_path)
    
    if accelerator.is_main_process:
        state_dict = accelerator.get_state_dict(model)
        unwrapped_model = accelerator.unwrap_model(model)
        
        if loss_module is not None:
            unwrapped_loss_module = accelerator.unwrap_model(loss_module)
            if hasattr(unwrapped_loss_module, 'discriminator'):
                discriminator_state_dict = accelerator.get_state_dict(loss_module)
                discriminator_weights = {}
                for key, value in discriminator_state_dict.items():
                    if 'discriminator' in key:
                        discriminator_weights[key] = value
                
                if discriminator_weights:
                    state_dict.update(discriminator_weights)
                    logger.info(f"✅ 已保存判别器权重: {len(discriminator_weights)} 个参数")
        
        if hasattr(unwrapped_model, "save_pretrained_weight"):
            unwrapped_model.save_pretrained_weight(
                save_path / "unwrapped_model",
                save_function=accelerator.save,
                state_dict=state_dict,
            )
        else:
            (save_path / "unwrapped_model").mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, save_path / "unwrapped_model" / "pytorch_model.bin")
        
        metadata = {"global_step": global_step}
        if hasattr(unwrapped_model, "get_lora_info"):
            lora_info = unwrapped_model.get_lora_info()
            metadata["lora_info"] = lora_info
            logger.info(f"📊 LoRA信息: {lora_info}")
        
        json.dump(metadata, (save_path / "metadata.json").open("w+"))
        logger.info(f"✅ 已保存完整checkpoint到本地: {save_path}")
        
        # S3 upload
        try:
            if config and hasattr(config, 'experiment') and hasattr(config.experiment, 's3'):
                s3_config = config.experiment.s3
                bucket_name = s3_config.get('bucket_name', 'nextcam-sharing')
                base_path = s3_config.get('base_path', 'xcai/SRTokenizer')
                experiment_name = output_dir.split('/')[-1]
            else:
                bucket_name = "nextcam-sharing"
                base_path = "xcai/SRTokenizer"
                experiment_name = output_dir.split('/')[-1]
            
            s3_path = f"s3://{bucket_name}/{base_path}/{experiment_name}/checkpoint-{global_step}"
            
            try:
                import subprocess
                import shutil
                
                if shutil.which('aws'):
                    cmd = ['aws', 's3', 'cp', str(save_path), s3_path, '--recursive']
                    logger.info(f"🔄 使用 aws s3 cp 上传整个checkpoint到S3: {s3_path}")
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        logger.info(f"✅ 成功使用 aws s3 cp 上传整个checkpoint到S3: {s3_path}")
                    else:
                        logger.warning(f"⚠️ aws s3 cp 上传失败: {result.stderr}")
                        raise Exception("aws s3 cp failed")
                else:
                    raise Exception("aws command not found")
                    
            except Exception as e:
                logger.info(f"🔄 aws s3 cp 不可用，回退到Python boto3方式: {e}")
                try:
                    from utils.s3_utils import upload_checkpoint_to_s3, get_s3_config_from_env
                    
                    s3_config = get_s3_config_from_env()
                    
                    if config and hasattr(config, 'experiment') and hasattr(config.experiment, 's3'):
                        if hasattr(config.experiment.s3, 'aws_access_key_id') and config.experiment.s3.aws_access_key_id:
                            s3_config['aws_access_key_id'] = config.experiment.s3.aws_access_key_id
                        if hasattr(config.experiment.s3, 'aws_secret_access_key') and config.experiment.s3.aws_secret_access_key:
                            s3_config['aws_secret_access_key'] = config.experiment.s3.aws_secret_access_key
                        if hasattr(config.experiment.s3, 'region') and config.experiment.s3.region:
                            s3_config['aws_region'] = config.experiment.s3.region
                    
                    s3_success = upload_checkpoint_to_s3(
                        checkpoint_path=save_path,
                        s3_base_path=f"s3://{bucket_name}/{base_path}",
                        experiment_name=experiment_name,
                        global_step=global_step,
                        logger=logger,
                        **s3_config
                    )
                    
                    if s3_success:
                        logger.info(f"✅ 成功使用 boto3 上传整个checkpoint到S3: s3://{bucket_name}/{base_path}")
                    else:
                        logger.warning("⚠️ boto3上传失败，但本地保存成功")
                        
                except ImportError:
                    logger.warning("⚠️ boto3未安装，无法上传到S3。请运行: pip install boto3")
                except Exception as e2:
                    logger.warning(f"⚠️ boto3上传过程中发生错误: {e2}")
                    
        except Exception as e:
            logger.warning(f"⚠️ S3上传过程中发生错误: {e}，但本地保存成功")
        
        if config and hasattr(config, 's3_backup') and config.s3_backup.enabled:
            try:
                from utils.s3_utils import upload_checkpoint_to_s3, get_s3_config_from_env
                
                s3_config = get_s3_config_from_env()
                s3_success = upload_checkpoint_to_s3(
                    checkpoint_path=save_path,
                    s3_base_path=config.s3_backup.base_path,
                    experiment_name=config.experiment.name,
                    global_step=global_step,
                    logger=logger,
                    **s3_config
                )
                
                if s3_success:
                    logger.info(f"✅ 成功上传checkpoint到备用S3: {config.s3_backup.base_path}")
                else:
                    logger.warning("⚠️ 备用S3上传失败，但本地保存成功")
                    
            except ImportError:
                logger.warning("⚠️ boto3未安装，无法上传到备用S3。请运行: pip install boto3")
            except Exception as e:
                logger.warning(f"⚠️ 备用S3上传过程中发生错误: {e}")
        
        try:
            cleanup_old_checkpoints(output_dir, max_checkpoints=5, logger=logger)
        except Exception as e:
            logger.warning(f"⚠️ 清理旧checkpoint时发生错误: {e}")
    
    accelerator.wait_for_everyone()
    
    return save_path


def cleanup_old_checkpoints(output_dir, max_checkpoints=5, logger=None):
    """
    清理旧的checkpoint，只保留最新的max_checkpoints个
    
    Args:
        output_dir: 输出目录路径
        max_checkpoints: 最大保留的checkpoint数量，默认5个
        logger: 日志记录器
    """
    import shutil
    
    output_path = Path(output_dir)
    if not output_path.exists():
        return
    
    checkpoint_dirs = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step = int(item.name.split("-")[1])
                checkpoint_dirs.append((step, item))
            except (ValueError, IndexError):
                continue
    
    checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
    
    if len(checkpoint_dirs) > max_checkpoints:
        checkpoints_to_remove = checkpoint_dirs[max_checkpoints:]
        
        for step, checkpoint_dir in checkpoints_to_remove:
            try:
                shutil.rmtree(checkpoint_dir)
                if logger:
                    logger.info(f"🗑️ 已删除旧checkpoint: {checkpoint_dir.name} (步数: {step})")
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ 删除checkpoint {checkpoint_dir.name} 时发生错误: {e}")
        
        if logger:
            logger.info(f"✅ 清理完成，当前保留最新的 {min(len(checkpoint_dirs), max_checkpoints)} 个checkpoint")


def load_checkpoint(checkpoint_path: Path, accelerator, logger, strict=True):
    logger.info(f"Load checkpoint from {checkpoint_path}")

    try:
        accelerator.load_state(checkpoint_path, strict=strict)
    except RuntimeError as e:
        try:
            accelerator.load_state(checkpoint_path, strict=False)
        except Exception as e2:
            logger.warning(f"Could not load accelerator state: {e2}")
            logger.info("Continuing with model weights only...")
    try:
        with open(checkpoint_path / "metadata.json", "r") as f:
            metadata = json.load(f)
            global_step = int(metadata["global_step"])
            
            if hasattr(accelerator.unwrap_model(accelerator._models[0]), "validate_lora_parameters"):
                model = accelerator.unwrap_model(accelerator._models[0])
                is_valid, message = model.validate_lora_parameters()
                if is_valid:
                    logger.info(f"✅ {message}")
                else:
                    logger.warning(f"⚠️ {message}")
                
                if hasattr(model, "get_lora_info"):
                    lora_info = model.get_lora_info()
                    logger.info(f"📊 当前LoRA配置: {lora_info}")
                    
                    if "lora_info" in metadata:
                        saved_lora_info = metadata["lora_info"]
                        logger.info(f"📊 保存的LoRA配置: {saved_lora_info}")
                        
                        if lora_info.get("enabled") != saved_lora_info.get("enabled"):
                            logger.warning("⚠️ LoRA启用状态不匹配！")
                        elif lora_info.get("param_count") != saved_lora_info.get("param_count"):
                            logger.warning("⚠️ LoRA参数数量不匹配！")
            
        logger.info(f"Resuming at global_step {global_step}")
    except Exception as e:
        logger.warning(f"Could not load metadata.json: {e}")
        logger.info("Starting from global_step 0")
        global_step = 0
    
    return global_step


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


def inject_noise_to_lq_cond(lq_cond, config, training=True):
    """向lq_cond注入噪声
    
    Args:
        lq_cond: 低质量条件图像，shape (B, C, H, W)
        config: 配置对象，包含noise_injection配置
        training: 是否在训练模式下，训练时随机注入，评估时固定注入
    
    Returns:
        注入噪声后的lq_cond
    """
    if not hasattr(config, 'noise_injection') or not config.noise_injection.enabled:
        return lq_cond
    
    if lq_cond is None:
        return lq_cond
    
    noise_config = config.noise_injection
    strength = noise_config.strength
    noise_type = noise_config.type
    
    if strength <= 0.0:
        return lq_cond
    
    with torch.no_grad():
        image_std = lq_cond.std()
        noise_scale = strength * image_std
        
        if noise_type == "gaussian":
            if training:
                noise = torch.randn_like(lq_cond) * noise_scale
            else:
                torch.manual_seed(42)
                noise = torch.randn_like(lq_cond) * noise_scale
                torch.manual_seed(torch.initial_seed())
        elif noise_type == "uniform":
            if training:
                noise = (torch.rand_like(lq_cond) * 2 - 1) * noise_scale
            else:
                torch.manual_seed(42)
                noise = (torch.rand_like(lq_cond) * 2 - 1) * noise_scale
                torch.manual_seed(torch.initial_seed())
        else:
            return lq_cond
        
        noisy_lq_cond = lq_cond + noise
        
        if lq_cond.min() >= 0 and lq_cond.max() <= 1:
            noisy_lq_cond = torch.clamp(noisy_lq_cond, 0, 1)
        elif lq_cond.min() >= -1 and lq_cond.max() <= 1:
            noisy_lq_cond = torch.clamp(noisy_lq_cond, -1, 1)
    
    return noisy_lq_cond
