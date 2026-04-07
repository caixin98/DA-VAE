"""Training utils for TiTok.

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
import open_clip

from data import SimpleImageDataset, PretoeknizedDataSetJSONL, PretokenizedWebDataset
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from torch.optim import AdamW
try:
    from muon import MuonWithAuxAdam
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    print("Warning: Muon optimizer not available. Please install muon package to use Muon optimizer.")
from utils.lr_schedulers import get_scheduler
from modeling.modules import EMAModel, ReconstructionLoss_Stage1, ReconstructionLoss_Stage2, ReconstructionLoss_Single_Stage, ReconstructionLoss_FluxVAE, MLMLoss, ARLoss, SD3SRDistillationLoss
from utils.gradient_analyzer import GradientAnalyzer, GradientMonitor
from modeling.titok import TiTok, PretrainedTokenizer
from modeling.tatitok import TATiTok
from modeling.maskgit import ImageBert, UViTBert
from modeling.rar import RAR
from modeling.maskgen import MaskGen_VQ, MaskGen_KL, open_clip_text_encoding
from evaluator import VQGANEvaluator

from imagenet_classes import imagenet_idx2classname
from utils.viz_utils import make_viz_from_samples, make_viz_from_samples_generation, make_viz_from_samples_t2i_generation
from modeling.srtitok import SRTiTok
from torch.nn import functional as F
import gc
from modeling.modules.sd3_tokenizer import SD3Tokenizer, SD3SRTokenizer
from modeling.modules.flux_tokenizer import FluxTokenizer


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
        
        # 如果内存使用率过高，给出警告
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


def convert_tensor_to_model_dtype(tensor, config):
    """将张量转换为模型的数据类型"""
    target_dtype = get_model_dtype(config)
    if tensor.dtype != target_dtype:
        return tensor.to(target_dtype)
    return tensor


def optimize_memory_settings(config, logger):
    """根据配置优化内存设置"""
    logger.info("=== 内存优化设置 ===")
    
    # 混合精度设置
    if config.training.mixed_precision == "fp16":
        logger.info("✅ 启用FP16混合精度训练 - 可节省约30-50%显存")
    elif config.training.mixed_precision == "bf16":
        logger.info("✅ 启用BF16混合精度训练 - 可节省约30-50%显存，数值稳定性更好")
    else:
        logger.info("⚠️ 未启用混合精度训练 - 建议启用以节省显存")
    
    # 梯度检查点设置
    if config.model.get("enable_gradient_checkpointing", False):
        logger.info("✅ 启用梯度检查点 - 可节省约20-30%显存，但会降低训练速度")
    else:
        logger.info("⚠️ 未启用梯度检查点 - 建议启用以节省显存")
    
    # TF32设置
    if config.training.get("enable_tf32", False):
        logger.info("✅ 启用TF32 - 在Ampere GPU上可提高性能并减少显存")
    else:
        logger.info("ℹ️ TF32未启用")
    
    # 批次大小建议
    current_batch_size = config.training.per_gpu_batch_size
    if current_batch_size > 8:
        logger.info(f"⚠️ 当前批次大小较大 ({current_batch_size}) - 如果显存不足，建议减小到8或4")
    else:
        logger.info(f"✅ 当前批次大小合理: {current_batch_size}")
    
    # 梯度累积建议
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
    # Get the current model state dict
    current_state = model.state_dict()
    
    # Create a new state dict for loading with flexible parameter matching
    load_state = {}
    skipped_keys = []
    loaded_keys = []
    
    for key, value in model_weight.items():
        if key in current_state:
            current_shape = current_state[key].shape
            checkpoint_shape = value.shape
            
            # Handle token_size changes in final_layer.linear
            if key.endswith("final_layer.linear.weight") and len(current_shape) == 2 and len(checkpoint_shape) == 2:
                if current_shape[0] != checkpoint_shape[0]:
                    logger.info(f"Adapting {key} from {checkpoint_shape} to {current_shape} due to token_size change")
                    # Interpolate or pad the weight matrix
                    if current_shape[0] > checkpoint_shape[0]:
                        # Expand: pad with zeros
                        new_weight = torch.zeros(current_shape, dtype=value.dtype)
                        new_weight[:checkpoint_shape[0], :] = value
                        load_state[key] = new_weight
                    else:
                        # Shrink: take the first part
                        load_state[key] = value[:current_shape[0], :]
                    loaded_keys.append(key)
                    continue
            
            # Handle token_size changes in final_layer.linear.bias
            elif key.endswith("final_layer.linear.bias") and len(current_shape) == 1 and len(checkpoint_shape) == 1:
                if current_shape[0] != checkpoint_shape[0]:
                    logger.info(f"Adapting {key} from {checkpoint_shape} to {current_shape} due to token_size change")
                    if current_shape[0] > checkpoint_shape[0]:
                        # Expand: pad with zeros
                        new_bias = torch.zeros(current_shape, dtype=value.dtype)
                        new_bias[:checkpoint_shape[0]] = value
                        load_state[key] = new_bias
                    else:
                        # Shrink: take the first part
                        load_state[key] = value[:current_shape[0]]
                    loaded_keys.append(key)
                    continue
            
            # Handle learnable_latent_tokens with specialized logic
            elif key.endswith("learnable_latent_tokens") and len(current_shape) == 3 and len(checkpoint_shape) == 3:
                # learnable_latent_tokens shape: (1, num_latent_tokens, hidden_size)
                if current_shape != checkpoint_shape:
                    logger.info(f"Adapting {key} from {checkpoint_shape} to {current_shape} due to shape change")
                    
                    # Extract dimensions
                    current_batch, current_tokens, current_hidden = current_shape
                    checkpoint_batch, checkpoint_tokens, checkpoint_hidden = checkpoint_shape
                    
                    # Initialize new tokens with zeros
                    new_tokens = torch.zeros(current_shape, dtype=value.dtype, device=value.device)
                    
                    # Copy existing tokens (handle both token and hidden dimension changes)
                    min_tokens = min(current_tokens, checkpoint_tokens)
                    min_hidden = min(current_hidden, checkpoint_hidden)
                    new_tokens[0, :min_tokens, :min_hidden] = value[0, :min_tokens, :min_hidden]
                    
                    # Handle token dimension expansion
                    if current_tokens > checkpoint_tokens:
                        logger.info(f"Expanding num_latent_tokens from {checkpoint_tokens} to {current_tokens}")
                        # Use Xavier initialization for new tokens
                        scale = current_hidden ** -0.5
                        for i in range(checkpoint_tokens, current_tokens):
                            new_tokens[0, i, :] = scale * torch.randn(current_hidden, dtype=value.dtype, device=value.device)
                    
                    # Handle hidden dimension expansion
                    if current_hidden > checkpoint_hidden:
                        logger.info(f"Expanding hidden_size from {checkpoint_hidden} to {current_hidden}")
                        # For existing tokens, pad with small random values
                        scale = current_hidden ** -0.5
                        for i in range(min_tokens):
                            new_tokens[0, i, checkpoint_hidden:] = scale * torch.randn(
                                current_hidden - checkpoint_hidden, dtype=value.dtype, device=value.device
                            )
                        # For new tokens, initialize the entire hidden dimension
                        for i in range(checkpoint_tokens, current_tokens):
                            new_tokens[0, i, checkpoint_hidden:] = scale * torch.randn(
                                current_hidden - checkpoint_hidden, dtype=value.dtype, device=value.device
                            )
                    
                    load_state[key] = new_tokens
                    loaded_keys.append(key)
                    continue
            
            # Handle legacy latent_tokens (2D shape) for backward compatibility
            elif key == "latent_tokens" and len(current_shape) == 2 and len(checkpoint_shape) == 2:
                if current_shape[0] != checkpoint_shape[0] or current_shape[1] != checkpoint_shape[1]:
                    logger.info(f"Adapting legacy {key} from {checkpoint_shape} to {current_shape} due to shape change")
                    
                    # Handle both dimensions: num_latent_tokens and hidden_size
                    if current_shape[0] > checkpoint_shape[0]:
                        # Expand num_latent_tokens: pad with random initialization
                        if current_shape[1] > checkpoint_shape[1]:
                            # Expand hidden_size: pad with zeros
                            new_tokens = torch.zeros(current_shape, dtype=value.dtype)
                            new_tokens[:checkpoint_shape[0], :checkpoint_shape[1]] = value
                            # Fill expanded hidden_size with random initialization
                            scale = current_shape[1] ** -0.5
                            for i in range(checkpoint_shape[0], current_shape[0]):
                                new_tokens[i, :] = scale * torch.randn(current_shape[1], dtype=value.dtype)
                        else:
                            # Shrink hidden_size: take the first part
                            new_tokens = torch.zeros(current_shape, dtype=value.dtype)
                            new_tokens[:checkpoint_shape[0], :] = value[:, :current_shape[1]]
                            # Fill expanded num_latent_tokens with random initialization
                            scale = current_shape[1] ** -0.5
                            for i in range(checkpoint_shape[0], current_shape[0]):
                                new_tokens[i, :] = scale * torch.randn(current_shape[1], dtype=value.dtype)
                    else:
                        # Shrink num_latent_tokens: take the first part
                        if current_shape[1] > checkpoint_shape[1]:
                            # Expand hidden_size: pad with zeros
                            new_tokens = torch.zeros(current_shape, dtype=value.dtype)
                            new_tokens[:, :checkpoint_shape[1]] = value[:current_shape[0], :]
                        else:
                            # Shrink hidden_size: take the first part
                            new_tokens = value[:current_shape[0], :current_shape[1]]
                    
                    load_state[key] = new_tokens
                    loaded_keys.append(key)
                    continue
            
            # Handle decoder.decoder_embed.weight changes (token_size changes)
            elif key == "decoder.decoder_embed.weight" and len(current_shape) == 2 and len(checkpoint_shape) == 2:
                if current_shape[1] != checkpoint_shape[1]:
                    logger.info(f"Adapting {key} from {checkpoint_shape} to {current_shape} due to token_size change")
                    if current_shape[1] > checkpoint_shape[1]:
                        # Expand: pad with zeros
                        new_weight = torch.zeros(current_shape, dtype=value.dtype)
                        new_weight[:, :checkpoint_shape[1]] = value
                        load_state[key] = new_weight
                    else:
                        # Shrink: take the first part
                        load_state[key] = value[:, :current_shape[1]]
                    loaded_keys.append(key)
                    continue
            
            # Handle shape mismatches in other parameters
            elif current_shape != checkpoint_shape:
                logger.warning(f"Skipping {key} due to shape mismatch: {checkpoint_shape} vs {current_shape}")
                skipped_keys.append(key)
                continue
            
            # Normal case: shapes match
            load_state[key] = value
            loaded_keys.append(key)
        else:
            logger.warning(f"Skipping {key} - not found in current model")
            skipped_keys.append(key)
    
    # Load the adapted state dict to the model
    msg = model.load_state_dict(load_state, strict=False)
    
    if weight_path:
        logger.info(f"loading weight from {weight_path}")
    logger.info(f"Loaded {len(loaded_keys)} parameters, skipped {len(skipped_keys)} parameters")
    # logger.info(f"Load message: {msg}")
    
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


# --- CUDA prefetch utilities to overlap H2D copies with compute ---
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


def create_pretrained_tokenizer(config, accelerator=None):
    if config.model.vq_model.get("finetune_decoder", False):
        # No need of pretrained tokenizer at stage2
        pretrianed_tokenizer = None
    else:
        pretrianed_tokenizer = PretrainedTokenizer(config.model.vq_model.pretrained_tokenizer_weight)
        if accelerator is not None:
            pretrianed_tokenizer.to(accelerator.device)
    return pretrianed_tokenizer


def create_clip_model():
    clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
    del clip.visual
    tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
    clip.transformer.batch_first = False
    clip.eval()
    clip.requires_grad_(False)
    return clip, tokenizer


def create_model_and_loss_module(config, logger, accelerator,
                                 model_type="titok"):
    """Creates TiTok model and loss module."""
    logger.info("Creating model and loss module.")
    if model_type == "titok":
        model_cls = TiTok
        if config.model.get("use_flux_vae", False) or config.model.get("use_flux_sd3", False):
            loss_cls = ReconstructionLoss_FluxVAE
        else:
            loss_cls = ReconstructionLoss_Stage2 if config.model.vq_model.get("finetune_decoder", False) else ReconstructionLoss_Stage1
    elif model_type == "tatitok":
        model_cls = TATiTok
        if config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False):
            loss_cls = ReconstructionLoss_FluxVAE
        else:
            loss_cls = ReconstructionLoss_Single_Stage
    elif model_type == "srtitok":
        model_cls = SRTiTok
        if config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False):
            loss_cls = ReconstructionLoss_FluxVAE
        else:
            loss_cls = ReconstructionLoss_Single_Stage
    elif model_type in ("sd3_tokenizer", "sd3_tokenizer_sr"):
        if model_type == "sd3_tokenizer_sr":
            model_cls = SD3SRTokenizer
            loss_cls = SD3SRDistillationLoss
        else:
            loss_cls = ReconstructionLoss_Single_Stage
        
        # 检查是否使用DMD损失类型
        if config.losses.get("use_dmd_loss", False):
            from modeling.modules.dmd_loss_type import DMDLossType
            loss_cls = DMDLossType
    elif model_type in ("sd3_tokenizer_2d", "sd3_tokenizer_sr"):
        from modeling.modules.sd3_tokenizer_2d import SD3Tokenizer as SD3Tokenizer2D
        model_cls = SD3Tokenizer2D
        if config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False):
            loss_cls = ReconstructionLoss_FluxVAE
        else:
            loss_cls = ReconstructionLoss_Single_Stage
        
        # 检查是否使用DMD损失类型
        if config.losses.get("use_dmd_loss", False):
            from modeling.modules.dmd_loss_type import DMDLossType
            loss_cls = DMDLossType
    elif model_type == "sd3_da_vae":
        from modeling.modules.sd3_da_vae import SD3_DAAutoencoder as SD3DAVAE
        model_cls = SD3DAVAE
        # use image reconstruction loss in pixel space; can also use FluxVAE loss wrapper
        # if config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False):
        #     loss_cls = ReconstructionLoss_FluxVAE
        # else:
        loss_cls = ReconstructionLoss_Single_Stage

        # 检查是否使用DMD损失类型
        if config.losses.get("use_dmd_loss", False):
            from modeling.modules.dmd_loss_type import DMDLossType
            loss_cls = DMDLossType
    elif model_type == "cond_upsampler":
        from modeling.modules.cond_image_upsampler import CondImageUpsampler
        model_cls = CondImageUpsampler
        # 直接复用单阶段重建损失：输入为 target images，输出为 reconstructed
        loss_cls = ReconstructionLoss_FluxVAE
    elif model_type == "flux_tokenizer":
        model_cls = FluxTokenizer
        if config.model.get("use_flux_vae", False):
            loss_cls = ReconstructionLoss_FluxVAE
        else:
            loss_cls = ReconstructionLoss_Single_Stage
    elif model_type == "maskgit":
        if config.model.generator.model_type == "ViT":
            model_cls = ImageBert
        elif config.model.generator.model_type == "UViT":
            model_cls = UViTBert
        else:
            raise ValueError(f"Unsupported generator model_type {config.model.generator.model_type}")
        loss_cls = MLMLoss
    elif model_type == "rar":
        model_cls = RAR
        loss_cls = ARLoss
    elif model_type == "maskgen_vq":
        model_cls = MaskGen_VQ
        loss_cls = MLMLoss
    elif model_type == "maskgen_kl":
        model_cls = MaskGen_KL
        loss_cls = None
    else:
        raise ValueError(f"Unsupported model_type {model_type}")
    
    # # Set model dtype based on mixed precision configuration
    model_dtype = torch.float32
    if config.training.mixed_precision == "fp16":
        model_dtype = torch.float16
        logger.info("Creating model with float16 precision")
    elif config.training.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
        logger.info("Creating model with bfloat16 precision")
    
    model = model_cls(config)
    

    # 记录模型创建后的内存使用情况
    log_memory_usage(logger, "模型创建后")

    if config.experiment.get("init_weight", ""):
        # If loading a pretrained weight
        init_weight_path = config.experiment.init_weight
        
        # 检查权重文件是否存在，如果不存在则尝试从S3下载
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
        # if config.model.vq_model.finetune_decoder:
        #     # Add the MaskGIT-VQGAN's quantizer/decoder weight as well
        #     pretrained_tokenizer_weight = torch.load(
        #         config.model.vq_model.pretrained_tokenizer_weight, map_location="cpu", weights_only=True
        #     )
        #     # Only keep the quantize and decoder part
        #     pretrained_tokenizer_weight = {"pixel_" + k:v for k,v in pretrained_tokenizer_weight.items() if not "encoder." in k}
        #     model_weight.update(pretrained_tokenizer_weight)
        
        msg, loaded_keys, skipped_keys, skipped_keys_list = load_weights_with_flexible_matching(model, model_weight, logger, init_weight_path)

    # Create the EMA model.
    ema_model = None
    if config.training.use_ema:
        ema_model = EMAModel(model.parameters(), decay=0.999,
                            model_cls=model_cls, config=config)
        # Create custom saving and loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"),
                                                  model_cls=model_cls, config=config)
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                # 直接保存EMA状态
                ema_model.save_pretrained(os.path.join(output_dir, "ema_model"))


        accelerator.register_load_state_pre_hook(load_model_hook)
        accelerator.register_save_state_pre_hook(save_model_hook)

    # Create loss module along with discriminator - 在accelerator.prepare()之前创建
    if loss_cls is not None:
        import inspect
        init_params = inspect.signature(loss_cls.__init__).parameters
        if 'model' in init_params:
            loss_module = loss_cls(config=config, model=model)
        else:
            loss_module = loss_cls(config=config)
    else:
        loss_module = None
    
    # # 修复：让loss_module使用model的original_vae_model，确保一致性
    # if loss_module is not None and hasattr(loss_module, 'original_vae_model') and hasattr(model, 'original_vae_model'):
    #     print("🔧 修复：让loss_module使用model的original_vae_model")
    #     loss_module.original_vae_model = model.original_vae_model
    
 
    
    # 记录损失模块创建后的内存使用情况
    log_memory_usage(logger, "损失模块创建后")

    # # Print Model for sanity check.
    # if accelerator.is_main_process:
    #     if model_type in ["titok"]:
    #         input_size = (1, 3, config.dataset.preprocessing.crop_size, config.dataset.preprocessing.crop_size)
    #         model_summary_str = summary(model, input_size=input_size, depth=5,
    #         col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
    #         logger.info(model_summary_str)
    #     elif model_type in ["tatitok"]:
    #         input_image_size  = (1, 3, config.dataset.preprocessing.crop_size, config.dataset.preprocessing.crop_size)
    #         input_text_size = (1, 77, 768)
    #         input_size = [input_image_size, input_text_size]
    #         model_summary_str = summary(model, input_size=input_size, depth=5,
    #         col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
    #         logger.info(model_summary_str)
    #     elif model_type in ["srtitok"]:
    #         if config.model.get("use_flux_vae", False):
    #             input_size = (1, 64, 16, 16)
    #             input_cond_size = (1, 64, 8, 8)
    #         else:
    #             input_size = (1, 3, 256, 256)
    #             input_cond_size = (1, 3, 128, 128)
               
    #         input_size = [input_size, input_cond_size]
    #         model_summary_str = summary(model, input_size=input_size, depth=5,
    #         col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
    #         logger.info(model_summary_str)
    #     elif model_type in ["maskgit", "rar"]:
    #         input_size = (1, config.model.vq_model.num_latent_tokens)
    #         input_data = [
    #             torch.randint(0, config.model.vq_model.codebook_size, input_size),
    #             torch.ones(1, dtype=int)
    #         ]
    #         model_summary_str = summary(
    #             model, input_data=input_data, depth=7,
    #             col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
    #         logger.info(model_summary_str)
    #     elif model_type in ["maskgen_vq"]:
    #         x_size = (1, config.model.vq_model.num_latent_tokens)
    #         condition_size = (1, 77, 768)
    #         condition_pooled_size = (1, 1, 768)
    #         aes_size = (1,)
    #         input_size = [x_size, condition_size, condition_pooled_size, aes_size]
    #         model_summary_str = summary(model, input_size=input_size, depth=5, col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
    #         logger.info(model_summary_str)
    #     elif model_type in ["maskgen_kl"]:
    #         x_size = (1, config.model.vq_model.token_size * 2 * config.model.vq_model.num_latent_tokens)
    #         condition_size = (1, 77, 768)
    #         condition_pooled_size = (1, 1, 768)
    #         aes_size = (1,)
    #         input_size = [x_size, condition_size, condition_pooled_size, aes_size]
    #         model_summary_str = summary(model, input_size=input_size, depth=5, col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
    #         logger.info(model_summary_str)
    #     else:
    #         raise NotImplementedError

    # 在返回前记录最终内存使用情况
    log_memory_usage(logger, "模型初始化完成")
    
    # 提供内存优化建议
    if accelerator.is_main_process:
        optimize_memory_settings(config, logger)
    
    return model, ema_model, loss_module


def create_optimizer(config, logger, model, loss_module,
                     model_type="titok", need_discrminator=True):
    """Creates optimizer for TiTok and discrminator."""
    logger.info("Creating optimizers.")
    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer_cls = AdamW
    elif optimizer_type == "muon":
        if not MUON_AVAILABLE:
            raise ValueError("Muon optimizer is not available. Please install muon package.")
        optimizer_cls = MuonWithAuxAdam
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Exclude terms we may not want to apply weight decay.
    exclude = (lambda n, p: p.ndim < 2 or "ln" in n or "bias" in n or 'latent_tokens' in n 
               or 'mask_token' in n or 'embedding' in n or 'norm' in n or 'gamma' in n or 'embed' in n)
    include = lambda n, p: not exclude(n, p)
    train_compression_only = config.model.vq_model.get('train_compression_only', False)
    train_downsample_only = config.model.vq_model.get('train_downsample_only', False)
    freeze_downsampler = config.model.get('freeze_downsampler', False)
    # 检查是否使用Muon优化器
    if optimizer_type == "muon":
        logger.info("使用Muon优化器进行参数分组")
        
        # 根据Muon的要求进行参数分组
        # hidden_weights: 维度>=2的参数，使用Muon优化
        # hidden_gains_biases: 维度<2的参数，使用AdamW优化
        # nonhidden_params: 其他参数（如embeddings, classifier heads等），使用AdamW优化
        
        hidden_weights = []
        hidden_gains_biases = []
        nonhidden_params = []
        
        # 针对SD3Tokenizer的特殊处理
        if model_type in ("sd3_tokenizer", "sd3_tokenizer_2d", "sd3_tokenizer_sr"):
            logger.info(f"检测到{model_type}，使用专门的参数分组策略")
            
            # 检查是否只训练压缩/恢复网络层
            if train_compression_only:
                logger.info("启用压缩层专用训练模式 - 只训练 hidden_size_to_token_size 和 token_size_to_hidden_size")
            
            # 检查是否只训练下采样/上采样网络层
            if train_downsample_only:
                logger.info("启用下采样层专用训练模式 - 只训练 downsampler 和 upsampler")
            
            # 检查是否冻结下采样/上采样网络层
            if freeze_downsampler:
                logger.info("启用下采样层冻结模式 - 冻结 downsampler 和 upsampler")
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # 如果启用压缩层专用训练模式
                if train_compression_only:
                    # 只训练压缩和恢复网络层
                    if any(key in name for key in [
                        'hidden_size_to_token_size',  # encoder压缩网络
                        'token_size_to_hidden_size',  # decoder恢复网络
                    ]):
                        # 这些是压缩/恢复网络权重，使用Muon优化
                        hidden_weights.append(param)
                        logger.debug(f"添加压缩层参数到训练: {name}")
                    else:
                        # 其他所有参数都冻结
                        param.requires_grad = False
                        logger.debug(f"冻结参数: {name}")
                    continue
                
                # 如果启用下采样层专用训练模式
                if train_downsample_only:
                    # 只训练下采样和上采样网络层
                    if any(key in name for key in [
                        'downsampler',  # 下采样网络
                        'upsampler',    # 上采样网络
                    ]):
                        # 这些是下采样/上采样网络权重，使用Muon优化
                        hidden_weights.append(param)
                        logger.debug(f"添加下采样层参数到训练: {name}")
                    else:
                        # 其他所有参数都冻结
                        param.requires_grad = False
                        logger.debug(f"冻结参数: {name}")
                    continue
                
                # 如果启用下采样层冻结模式
                if freeze_downsampler:
                    # 冻结下采样和上采样网络层
                    if any(key in name for key in [
                        'downsampler',  # 下采样网络
                        'upsampler',    # 上采样网络
                    ]):
                        # 这些是下采样/上采样网络权重，需要冻结
                        param.requires_grad = False
                        logger.debug(f"冻结下采样层参数: {name}")
                        continue
                
                # 原有的参数分组逻辑（当 train_compression_only=False 时）
                if param.ndim >= 2:
                    # 检查是否是embedding层（应该使用AdamW优化）
                    if any(key in name for key in [
                        'transformer.pos_embed',  # 位置嵌入
                        'transformer.context_embedder',  # 上下文嵌入
                        'learnable_latent_tokens',  # 可学习的潜在tokens
              
                        'transformer.proj_out',  # 输出投影
                        'transformer.norm_out',  # 输出归一化
                    ]):
                        # embedding层使用AdamW优化
                        nonhidden_params.append(param)
                    # 检查是否是核心的网络层（应该使用Muon优化）
                    elif any(key in name for key in [
                        'transformer.transformer_blocks',  # transformer blocks
                        'hidden_size_to_token_size',  # encoder压缩网络
                        'token_size_to_hidden_size',  # decoder恢复网络
                    ]):
                        # 这些是核心的网络权重，使用Muon优化
                        hidden_weights.append(param)
                    else:
                        # 其他维度>=2的参数，使用AdamW优化
                        nonhidden_params.append(param)
                else:
                    # 维度<2的参数（bias, norm等），使用AdamW优化
                    hidden_gains_biases.append(param)
        
        # 针对FluxTokenizer的特殊处理
        elif model_type == "flux_tokenizer":
            logger.info("检测到FluxTokenizer，使用专门的参数分组策略")
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # FluxTokenizer特殊参数分组逻辑
                if param.ndim >= 2:
                    # 检查是否是embedding层（应该使用AdamW优化）
                    if any(key in name for key in [
                        'transformer.x_embedder',  # 输入嵌入
                        'transformer.context_embedder',  # 上下文嵌入
                        'transformer.pos_embed',  # 位置嵌入
                        'transformer.time_text_embed',  # 时间文本嵌入
                        'learnable_latent_tokens',  # 可学习的潜在tokens
                        'hidden_size_to_token_size',  # encoder压缩网络
                        'token_size_to_hidden_size',  # decoder恢复网络
                        'transformer.proj_out',  # 输出投影
                        'transformer.norm_out',  # 输出归一化
                        'transformer.encoder_hid_proj',  # 编码器隐藏状态投影
                    ]):
                        # embedding层使用AdamW优化
                        nonhidden_params.append(param)
                    # 检查是否是核心的网络层（应该使用Muon优化）
                    elif any(key in name for key in [
                        'transformer.transformer_blocks',  # transformer blocks
                        'transformer.single_transformer_blocks',  # single transformer blocks
                    ]):
                        # 这些是核心的网络权重，使用Muon优化
                        hidden_weights.append(param)
                    else:
                        # 其他维度>=2的参数，使用AdamW优化
                        nonhidden_params.append(param)
                else:
                    # 维度<2的参数（bias, norm等），使用AdamW优化
                    hidden_gains_biases.append(param)
        else:
            # 通用参数分组逻辑（适用于其他模型）
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                # 根据参数名称和维度进行分类
                if param.ndim >= 2:
                    # 维度>=2的参数使用Muon优化
                    hidden_weights.append(param)
                else:
                    # 维度<2的参数使用AdamW优化
                    hidden_gains_biases.append(param)
        
        # 创建参数组
        param_groups = []
        
        # Muon参数组（hidden_weights）
        if hidden_weights:
            muon_lr = optimizer_config.get("muon_learning_rate", 0.02)
            muon_weight_decay = optimizer_config.get("muon_weight_decay", 0.01)
            param_groups.append({
                "params": hidden_weights, 
                "use_muon": True,
                "lr": muon_lr, 
                "weight_decay": muon_weight_decay
            })
            logger.info(f"Muon参数组: {len(hidden_weights)}个参数，学习率: {muon_lr}")
        
        # AdamW参数组（hidden_gains_biases + nonhidden_params）
        if hidden_gains_biases:
            adamw_lr = optimizer_config.get("adamw_learning_rate", learning_rate)
            adamw_betas = (optimizer_config.get("adamw_beta1", 0.9), optimizer_config.get("adamw_beta2", 0.95))
            adamw_weight_decay = optimizer_config.get("adamw_weight_decay", optimizer_config.weight_decay)
            param_groups.append({
                "params": hidden_gains_biases, 
                "use_muon": False,
                "lr": adamw_lr, 
                "betas": adamw_betas,
                "weight_decay": adamw_weight_decay
            })
            logger.info(f"AdamW参数组(bias/norm): {len(hidden_gains_biases)}个参数，学习率: {adamw_lr}")
        
        # 非隐藏参数组（nonhidden_params，仅对SD3Tokenizer有效）
        if nonhidden_params:
            nonhidden_lr = optimizer_config.get("nonhidden_learning_rate", learning_rate)
            nonhidden_betas = (optimizer_config.get("nonhidden_beta1", 0.9), optimizer_config.get("nonhidden_beta2", 0.95))
            nonhidden_weight_decay = optimizer_config.get("nonhidden_weight_decay", optimizer_config.weight_decay)
            param_groups.append({
                "params": nonhidden_params, 
                "use_muon": False,
                "lr": nonhidden_lr, 
                "betas": nonhidden_betas,
                "weight_decay": nonhidden_weight_decay
            })
            logger.info(f"AdamW参数组(其他): {len(nonhidden_params)}个参数，学习率: {nonhidden_lr}")
        
        # 创建Muon优化器
        optimizer = MuonWithAuxAdam(param_groups)
        
        # 为优化器添加学习率信息
        adamw_lr = optimizer_config.get("adamw_learning_rate", learning_rate)
        optimizer.encoder_lr = adamw_lr if hidden_gains_biases else learning_rate
        optimizer.decoder_lr = adamw_lr if hidden_gains_biases else learning_rate
        optimizer.base_lr = learning_rate
        optimizer.use_separate_lr = False
        optimizer.optimizer_type = "muon"
        
    # 检查是否使用分离的学习率
    elif optimizer_config.get("use_separate_encoder_decoder_lr", False) and (model_type in ("srtitok", "sd3_tokenizer", "sd3_tokenizer_sr", "sd3_tokenizer_2d")):
        # 为encoder和decoder分别设置学习率
        encoder_lr = optimizer_config.get("encoder_learning_rate", learning_rate)
        decoder_lr = optimizer_config.get("decoder_learning_rate", learning_rate)
        
        logger.info(f"使用分离的学习率 - Encoder: {encoder_lr}, Decoder: {decoder_lr}")
        
        # 检查是否只训练压缩/恢复网络层（针对SD3Tokenizer）
        if model_type in ("sd3_tokenizer", "sd3_tokenizer_sr"):
            if train_compression_only:
                logger.info("启用压缩层专用训练模式 - 只训练 hidden_size_to_token_size 和 token_size_to_hidden_size")
                
                # 只保留压缩和恢复网络层的参数
                compression_params = []
                for name, param in model.named_parameters():
                    if param.requires_grad and any(key in name for key in [
                        'hidden_size_to_token_size',  # encoder压缩网络
                        'token_size_to_hidden_size',  # decoder恢复网络
                    ]):
                        compression_params.append({"params": param, "weight_decay": optimizer_config.weight_decay, "lr": learning_rate})
                        logger.debug(f"添加压缩层参数到训练: {name}")
                    else:
                        # 冻结其他所有参数
                        param.requires_grad = False
                        logger.debug(f"冻结参数: {name}")
                
                # 创建只包含压缩层参数的优化器
                optimizer = optimizer_cls(
                    compression_params,
                    betas=(optimizer_config.beta1, optimizer_config.beta2)
                )
                
                # 为优化器添加学习率信息
                optimizer.encoder_lr = learning_rate
                optimizer.decoder_lr = learning_rate
                optimizer.base_lr = learning_rate
                optimizer.use_separate_lr = False
            else:
                # 使用原有的分离学习率逻辑
                encoder_params = []
                decoder_params = []
                other_params = []
                
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                        
                    if name.startswith('encoder.'):
                        # encoder参数
                        if exclude(name, param):
                            encoder_params.append({"params": param, "weight_decay": 0., "lr": encoder_lr})
                        else:
                            encoder_params.append({"params": param, "weight_decay": optimizer_config.weight_decay, "lr": encoder_lr})
                    elif name.startswith('decoder.'):
                        # decoder参数
                        if exclude(name, param):
                            decoder_params.append({"params": param, "weight_decay": 0., "lr": decoder_lr})
                        else:
                            decoder_params.append({"params": param, "weight_decay": optimizer_config.weight_decay, "lr": decoder_lr})
                    else:
                        # 其他参数（如latent_tokens等）
                        if exclude(name, param):
                            other_params.append({"params": param, "weight_decay": 0., "lr": learning_rate})
                        else:
                            other_params.append({"params": param, "weight_decay": optimizer_config.weight_decay, "lr": learning_rate})
                
                # 创建优化器参数组
                param_groups = []
                if encoder_params:
                    param_groups.extend(encoder_params)
                    logger.info(f"Encoder参数组: {len(encoder_params)}个参数，学习率: {encoder_lr}")
                if decoder_params:
                    param_groups.extend(decoder_params)
                    logger.info(f"Decoder参数组: {len(decoder_params)}个参数，学习率: {decoder_lr}")
                if other_params:
                    param_groups.extend(other_params)
                    logger.info(f"其他参数组: {len(other_params)}个参数，学习率: {learning_rate}")
                
                optimizer = optimizer_cls(
                    param_groups,
                    betas=(optimizer_config.beta1, optimizer_config.beta2)
                )
                
                # 为优化器添加学习率信息，方便后续记录到WandB
                optimizer.encoder_lr = encoder_lr
                optimizer.decoder_lr = decoder_lr
                optimizer.base_lr = learning_rate
                optimizer.use_separate_lr = True
        else:
            # 原有的分离学习率逻辑（针对srtitok）
            encoder_params = []
            decoder_params = []
            other_params = []
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                if name.startswith('encoder.'):
                    # encoder参数
                    if exclude(name, param):
                        encoder_params.append({"params": param, "weight_decay": 0., "lr": encoder_lr})
                    else:
                        encoder_params.append({"params": param, "weight_decay": optimizer_config.weight_decay, "lr": encoder_lr})
                elif name.startswith('decoder.'):
                    # decoder参数
                    if exclude(name, param):
                        decoder_params.append({"params": param, "weight_decay": 0., "lr": decoder_lr})
                    else:
                        decoder_params.append({"params": param, "weight_decay": optimizer_config.weight_decay, "lr": decoder_lr})
                else:
                    # 其他参数（如latent_tokens等）
                    if exclude(name, param):
                        other_params.append({"params": param, "weight_decay": 0., "lr": learning_rate})
                    else:
                        other_params.append({"params": param, "weight_decay": optimizer_config.weight_decay, "lr": learning_rate})
            
            # 创建优化器参数组
            param_groups = []
            if encoder_params:
                param_groups.extend(encoder_params)
                logger.info(f"Encoder参数组: {len(encoder_params)}个参数，学习率: {encoder_lr}")
            if decoder_params:
                param_groups.extend(decoder_params)
                logger.info(f"Decoder参数组: {len(decoder_params)}个参数，学习率: {decoder_lr}")
            if other_params:
                param_groups.extend(other_params)
                logger.info(f"其他参数组: {len(other_params)}个参数，学习率: {learning_rate}")
            
            optimizer = optimizer_cls(
                param_groups,
                betas=(optimizer_config.beta1, optimizer_config.beta2)
            )
            
            # 为优化器添加学习率信息，方便后续记录到WandB
            optimizer.encoder_lr = encoder_lr
            optimizer.decoder_lr = decoder_lr
            optimizer.base_lr = learning_rate
            optimizer.use_separate_lr = True
        
    else:
        # 原有的优化器创建逻辑
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
                    # 将VAE decoder参数添加到named_parameters中
                    for i, param in enumerate(vae_decoder_params):
                        named_parameters.append((f"vae_decoder.{i}", param))
                else:
                    logger.warning("未找到可训练的VAE decoder参数")
        
        # 检查是否只训练压缩/恢复网络层（针对SD3Tokenizer）
        if model_type in ("sd3_tokenizer", "sd3_tokenizer_sr"):
            if train_compression_only:
                logger.info("启用压缩层专用训练模式 - 只训练 hidden_size_to_token_size 和 token_size_to_hidden_size")
                
                # 只保留压缩和恢复网络层的参数
                compression_params = []
                for name, param in named_parameters:
                    if param.requires_grad and any(key in name for key in [
                        'hidden_size_to_token_size',  # encoder压缩网络
                        'token_size_to_hidden_size',  # decoder恢复网络
                    ]):
                        compression_params.append(param)
                        logger.debug(f"添加压缩层参数到训练: {name}")
                    else:
                        # 冻结其他所有参数
                        param.requires_grad = False
                        logger.debug(f"冻结参数: {name}")
                
                # 创建只包含压缩层参数的优化器
                optimizer = optimizer_cls(
                    [{"params": compression_params, "weight_decay": optimizer_config.weight_decay}],
                    lr=learning_rate,
                    betas=(optimizer_config.beta1, optimizer_config.beta2)
                )
            else:
                # 如果启用下采样层冻结模式，需要先冻结相关参数
                if freeze_downsampler:
                    logger.info("启用下采样层冻结模式 - 冻结 downsampler 和 upsampler")
                    for name, param in named_parameters:
                        if any(key in name for key in [
                            'downsampler',  # 下采样网络
                            'upsampler',    # 上采样网络
                        ]):
                            param.requires_grad = False
                            logger.debug(f"冻结下采样层参数: {name}")
                
                # 使用原有的参数分组逻辑
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
        else:
            # 其他模型类型使用原有逻辑
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
        
        # 为优化器添加学习率信息
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

        if optimizer_type == "muon":
            # 为判别器也使用Muon优化器
            discriminator_hidden_weights = [p for p in discriminator_rest_params if p.ndim >= 2]
            discriminator_hidden_gains_biases = [p for p in discriminator_gain_or_bias_params if p.ndim < 2]
            
            discriminator_param_groups = []
            
            if discriminator_hidden_weights:
                discriminator_muon_lr = optimizer_config.get("discriminator_muon_learning_rate", discriminator_learning_rate)
                discriminator_muon_weight_decay = optimizer_config.get("discriminator_muon_weight_decay", optimizer_config.weight_decay)
                discriminator_param_groups.append({
                    "params": discriminator_hidden_weights,
                    "use_muon": True,
                    "lr": discriminator_muon_lr,
                    "weight_decay": discriminator_muon_weight_decay
                })
                logger.info(f"判别器Muon参数组: {len(discriminator_hidden_weights)}个参数，学习率: {discriminator_muon_lr}")
            
            if discriminator_hidden_gains_biases:
                discriminator_adamw_lr = optimizer_config.get("discriminator_adamw_learning_rate", discriminator_learning_rate)
                discriminator_adamw_betas = (optimizer_config.get("discriminator_adamw_beta1", optimizer_config.beta1), 
                                           optimizer_config.get("discriminator_adamw_beta2", optimizer_config.beta2))
                discriminator_adamw_weight_decay = optimizer_config.get("discriminator_adamw_weight_decay", optimizer_config.weight_decay)
                discriminator_param_groups.append({
                    "params": discriminator_hidden_gains_biases,
                    "use_muon": False,
                    "lr": discriminator_adamw_lr,
                    "betas": discriminator_adamw_betas,
                    "weight_decay": discriminator_adamw_weight_decay
                })
                logger.info(f"判别器AdamW参数组: {len(discriminator_hidden_gains_biases)}个参数，学习率: {discriminator_adamw_lr}")
            
            discriminator_optimizer = MuonWithAuxAdam(discriminator_param_groups)
        else:
            # 使用标准优化器
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
        
        # 获取DMD Guidance的参数
        dmd_guidance_parameters = list(loss_module.get_dmd_guidance_parameters())
        dmd_named_parameters = [(f"dmd_guidance.{name}", param) for name, param in loss_module.dmd_guidance.named_parameters()]
        
        dmd_gain_or_bias_params = [p for n, p in dmd_named_parameters if exclude(n, p) and p.requires_grad]
        dmd_rest_params = [p for n, p in dmd_named_parameters if include(n, p) and p.requires_grad]
        
        if optimizer_type == "muon":
            # 为DMD也使用Muon优化器
            dmd_hidden_weights = [p for p in dmd_rest_params if p.ndim >= 2]
            dmd_hidden_gains_biases = [p for p in dmd_gain_or_bias_params if p.ndim < 2]
            
            dmd_param_groups = []
            
            if dmd_hidden_weights:
                dmd_muon_lr = optimizer_config.get("dmd_muon_learning_rate", dmd_learning_rate)
                dmd_muon_weight_decay = optimizer_config.get("dmd_muon_weight_decay", optimizer_config.weight_decay)
                dmd_param_groups.append({
                    "params": dmd_hidden_weights,
                    "use_muon": True,
                    "lr": dmd_muon_lr,
                    "weight_decay": dmd_muon_weight_decay
                })
                logger.info(f"DMD Muon参数组: {len(dmd_hidden_weights)}个参数，学习率: {dmd_muon_lr}")
            
            if dmd_hidden_gains_biases:
                dmd_adamw_lr = optimizer_config.get("dmd_adamw_learning_rate", dmd_learning_rate)
                dmd_adamw_betas = (optimizer_config.get("dmd_adamw_beta1", optimizer_config.beta1), 
                                 optimizer_config.get("dmd_adamw_beta2", optimizer_config.beta2))
                dmd_adamw_weight_decay = optimizer_config.get("dmd_adamw_weight_decay", optimizer_config.weight_decay)
                dmd_param_groups.append({
                    "params": dmd_hidden_gains_biases,
                    "use_muon": False,
                    "lr": dmd_adamw_lr,
                    "betas": dmd_adamw_betas,
                    "weight_decay": dmd_adamw_weight_decay
                })
                logger.info(f"DMD AdamW参数组: {len(dmd_hidden_gains_biases)}个参数，学习率: {dmd_adamw_lr}")
            
            dmd_optimizer = MuonWithAuxAdam(dmd_param_groups)
        else:
            # 使用标准优化器
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
    """Creates learning rate scheduler for TiTok and discrminator."""
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
                base_lr=discriminator_lr,  # 使用判别器的学习率
                end_lr=config.lr_scheduler.params.end_lr,
            )
        else:
            # 当判别器学习率为0时，不创建调度器
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
                base_lr=dmd_lr,  # 使用DMD的学习率
                end_lr=config.lr_scheduler.params.end_lr,
            )
        else:
            dmd_lr_scheduler = None
            logger.info("DMD learning rate is 0, skipping DMD lr scheduler")
    else:
        dmd_lr_scheduler = None
    
    return lr_scheduler, discriminator_lr_scheduler, dmd_lr_scheduler


def create_dataloader(config, logger, accelerator):
    """Creates data loader for training and testing."""
    logger.info("Creating dataloaders.")
    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    total_batch_size = (
        config.training.per_gpu_batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
    )
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # T2I uses a pretokenized dataset for speed-up.
    if dataset_config.get("pretokenization", "") and dataset_config.get("dataset_with_text_label", False) is True:
        dataset = PretokenizedWebDataset(
            train_shards_path=dataset_config.train_shards_path_or_url,
            eval_shards_path=dataset_config.eval_shards_path_or_url,
            num_train_examples=config.experiment.max_train_examples,
            per_gpu_batch_size=config.training.per_gpu_batch_size,
            global_batch_size=total_batch_size_without_accum,
            num_workers_per_gpu=dataset_config.num_workers_per_gpu,
            resize_shorter_edge=preproc_config.resize_shorter_edge,
            crop_size=preproc_config.crop_size,
            random_crop=preproc_config.random_crop,
            random_flip=preproc_config.random_flip,
            normalize_mean=preproc_config.normalize_mean,
            normalize_std=preproc_config.normalize_std,
            process_recap=preproc_config.get("preproc_recap", True),
            use_recap_prob=preproc_config.get("use_recap_prob", 0.95)
        )
        train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader
    # SimpleImageDataset
    elif dataset_config.get("pretokenization", "") and dataset_config.get("dataset_with_text_label", False) is False:
        dataset = SimpleImageDataset(
            train_shards_path=dataset_config.train_shards_path_or_url,
            eval_shards_path=dataset_config.eval_shards_path_or_url,
            num_train_examples=config.experiment.max_train_examples,
            per_gpu_batch_size=config.training.per_gpu_batch_size,
            global_batch_size=total_batch_size_without_accum,
            num_workers_per_gpu=dataset_config.num_workers_per_gpu,
            resize_shorter_edge=preproc_config.resize_shorter_edge,
            crop_size=preproc_config.crop_size,
            random_crop=preproc_config.random_crop,
            random_flip=preproc_config.random_flip,
            dataset_with_class_label=dataset_config.get("dataset_with_class_label", True),
            dataset_with_text_label=dataset_config.get("dataset_with_text_label", False),
            res_ratio_filtering=preproc_config.get("res_ratio_filtering", False),
        )
        train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader
    # potentially, use a pretokenized dataset for ImageNet speed-up.
    else:
        if dataset_config.get("pretokenization", ""):
            train_dataloader = DataLoader(
                PretoeknizedDataSetJSONL(dataset_config.pretokenization),
                batch_size=config.training.per_gpu_batch_size,
                shuffle=True, drop_last=True, pin_memory=True)
            train_dataloader.num_batches = math.ceil(
                config.experiment.max_train_examples / total_batch_size_without_accum)
            eval_dataloader = None
        else:
            # Default case - create SimpleImageDataset
            dataset = SimpleImageDataset(
                train_shards_path=dataset_config.train_shards_path_or_url,
                eval_shards_path=dataset_config.eval_shards_path_or_url,
                num_train_examples=config.experiment.max_train_examples,
                per_gpu_batch_size=config.training.per_gpu_batch_size,
                global_batch_size=total_batch_size_without_accum,
                num_workers_per_gpu=dataset_config.num_workers_per_gpu,
                resize_shorter_edge=preproc_config.resize_shorter_edge,
                crop_size=preproc_config.crop_size,
                random_crop=preproc_config.random_crop,
                random_flip=preproc_config.random_flip,
                dataset_with_class_label=dataset_config.get("dataset_with_class_label", True),
                dataset_with_text_label=dataset_config.get("dataset_with_text_label", False),
                res_ratio_filtering=preproc_config.get("res_ratio_filtering", False),
            )
            train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader
    
    return train_dataloader, eval_dataloader


def create_evaluator(config, logger, accelerator):
    """Creates evaluator."""
    logger.info("Creating evaluator.")
    use_vae = config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False)
    # Always create an evaluator; toggle latent metrics based on VAE usage
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
    # If resuming training.
    if config.experiment.resume:            
        accelerator.wait_for_everyone()
        
        # 检查是否指定了具体的checkpoint路径
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
            # 原有的自动检测逻辑
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
                    model_type="titok",
                    clip_tokenizer=None,
                    clip_encoder=None,
                    pretrained_tokenizer=None,
                    original_vae_model=None):
    """One epoch training."""
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

    # 记录训练开始时的内存使用情况
    if accelerator.is_main_process and global_step % 100 == 0:
        log_memory_usage(logger, f"训练开始 (Step {global_step})")

    # 使用传入的original_vae_model参数进行图像编码（生成ground truth）
    # 如果没有传入original_vae_model，尝试从模型中获取
    if original_vae_model is None and hasattr(accelerator.unwrap_model(model), 'original_vae_model'):
        original_vae_model = accelerator.unwrap_model(model).original_vae_model
        logger.info("train_one_epoch: 从模型中获取原始VAE模型进行图像编码")
    elif original_vae_model is not None:
        logger.info("train_one_epoch: 使用传入的原始VAE模型进行图像编码")

    autoencoder_logs = defaultdict(float)
    discriminator_logs = defaultdict(float)
    # Store last discriminator logs for logging when discriminator is not trained
    last_discriminator_logs = defaultdict(float)
    oom_batch_records = []
    for i, batch in enumerate(train_dataloader):
        model.train()
        if "image" in batch:
            images = batch["image"].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )

            # extra_results_dict = {"original_pixel_inputs": images.clone()}
            original_pixel_inputs = images.detach()
            # Apply VAE encoding to inputs only for models that operate in latent space
            use_vae = config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False)
            if original_vae_model is not None and use_vae:
                # 仅对tokenizer类模型进行latent编码输入，其余模型保持像素输入
                should_encode_inputs = model_type in ("sd3_tokenizer", "sd3_tokenizer_sr", "sd3_tokenizer_2d", "flux_tokenizer")
                if should_encode_inputs:
                    with torch.no_grad():
                        images = original_vae_model.encode(images)
              
            
            # 确保输入数据与模型精度一致
            model_dtype = get_model_dtype(config)
            images = images.to(model_dtype)
            
                    
        if "text" in batch and model_type == "tatitok":
            text = batch["text"]
            with torch.no_grad():
                text_guidance = clip_tokenizer(text).to(accelerator.device)
                cast_dtype = clip_encoder.transformer.get_cast_dtype()
                text_guidance = clip_encoder.token_embedding(text_guidance).to(cast_dtype)  # [batch_size, n_ctx, d_model]
                text_guidance = text_guidance + clip_encoder.positional_embedding.to(cast_dtype)
                text_guidance = text_guidance.permute(1, 0, 2)  # LND -> NLD
                text_guidance = clip_encoder.transformer(text_guidance, attn_mask=clip_encoder.attn_mask)
                text_guidance = text_guidance.permute(1, 0, 2)  # LND -> NLD
                text_guidance = clip_encoder.ln_final(text_guidance)  # [batch_size, n_ctx, transformer.width]
                
                # 确保文本引导数据与模型精度一致
                model_dtype = get_model_dtype(config)
                text_guidance = text_guidance.to(model_dtype)
        
        # 检查是否需要处理cond_images
        use_tensor_downsample = config.model.get('use_tensor_downsample', False)
        if (model_type in ("srtitok", "sd3_tokenizer", "sd3_tokenizer_sr", "sd3_tokenizer_2d", "flux_tokenizer")) and (use_tensor_downsample or "cond_image" in batch):
            
            if use_tensor_downsample:
                # 在tensor层面downsample模式下，从原始images构造cond_images
                # 使用新的use_tensor_downsample_mode函数
                lq_cond_mode = config.model.get('lq_cond_mode', 'auto')
                
                # 保存原始数据类型
                original_dtype = original_pixel_inputs.dtype
                
                # 进行下采样
                if original_dtype == torch.bfloat16:
                    # 临时转换为 float32 进行插值
                    cond_images = use_tensor_downsample_mode(
                        original_pixel_inputs.to(torch.float32), 
                        config, 
                        lq_cond_mode
                    )
                    # 转换回原始精度
                    cond_images = cond_images.to(original_dtype)
                else:
                    cond_images = use_tensor_downsample_mode(
                        original_pixel_inputs, 
                        config, 
                        lq_cond_mode
                    )
            else:
                # 传统模式：使用数据加载器提供的cond_images
                cond_images = batch["cond_image"]
                if cond_images is not None:
                    cond_images = cond_images.to(
                        accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                    )
            
            # 统一的VAE编码逻辑（仅对latent空间模型编码cond）
            if cond_images is not None and original_vae_model is not None:
                use_vae = config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False)
                should_encode_cond = model_type in ("sd3_tokenizer", "sd3_tokenizer_sr", "sd3_tokenizer_2d", "flux_tokenizer")
                if use_vae and should_encode_cond:
                    with torch.no_grad():
                        cond_images = original_vae_model.encode(cond_images)
              
                
                # 确保条件图像数据与模型精度一致
                model_dtype = get_model_dtype(config)
                cond_images = cond_images.to(model_dtype)
            
            # 应用noise注入（如果启用）
            if cond_images is not None and hasattr(config, 'noise_injection') and config.noise_injection.enabled:
                log_noise_stats = (global_step % 100 == 0) and accelerator.is_main_process
                cond_images_original = None
                if log_noise_stats:
                    cond_images_original = cond_images.detach()

                # 注入噪声（训练模式）
                cond_images = inject_noise_to_lq_cond(cond_images, config, training=True)

                # 监控噪声注入统计（仅在需要时克隆）
                if log_noise_stats and cond_images_original is not None:
                    noise_stats = monitor_noise_injection_stats(
                        cond_images_original, cond_images, logger, global_step, config
                    )
                    # 将噪声统计信息添加到日志中
                    if noise_stats:
                        for key, value in noise_stats.items():
                            autoencoder_logs[key] = value

        fnames = batch["__key__"]
        data_time_meter.update(time.time() - end)

        # #### EXPERIMENTAL INPUT MODE ####
        # input_mode = config.model.vq_model.get("input_mode", "image")
        # if input_mode == "difference":
        #     images = images - F.interpolate(cond_images, size=images.shape[2:], mode="bicubic", align_corners=False, antialias=True)
        #     # images = (images + 1) / 2
        #     cond_images = torch.zeros_like(cond_images).to(images.device)
            # cond_images = images
        # if input_mode == "diff_identity":
        #     images = images - F.interpolate(cond_images, size=images.shape[2:], mode="bicubic", align_corners=False, antialias=True)
        #             cond_images = images
        #### EXPERIMENTAL INPUT MODE ####
        


        
        # Obtain proxy codes
        if pretrained_tokenizer is not None:
            pretrained_tokenizer.eval()
            proxy_codes = pretrained_tokenizer.encode(images)
        else:
            proxy_codes = None

        try:
            with accelerator.accumulate([model, loss_module]):
                if model_type == "titok":
                    reconstructed_images, extra_results_dict = model(images)
                    if proxy_codes is None:
                        autoencoder_loss, loss_dict = loss_module(
                            images,
                            reconstructed_images,
                            extra_results_dict,
                            global_step,
                            mode="generator",
                        )
                    else:
                        autoencoder_loss, loss_dict = loss_module(
                            proxy_codes,
                            reconstructed_images,
                            extra_results_dict
                        )    
                elif model_type == "tatitok":
                    reconstructed_images, extra_results_dict = model(images, text_guidance)
                    autoencoder_loss, loss_dict = loss_module(
                        images,
                        reconstructed_images,
                        extra_results_dict,
                        global_step,
                        mode="generator",
                    )
                elif model_type == "srtitok":
                    # 在id_based模式下，需要传递image_ids参数
                    if hasattr(config, 'model') and hasattr(config.model.vq_model, 'encoder_mode') and config.model.vq_model.encoder_mode == "id_based":
                        # 从batch中获取image ids
                        if "__key__" in batch:
                            image_ids = batch["__key__"]
                            if isinstance(image_ids, torch.Tensor):
                                image_ids = image_ids.tolist()
                            reconstructed_images, extra_results_dict = model(images, cond_images, image_ids)
                        else:
                            logger.warning("id_based模式下缺少__key__字段，无法获取image_ids")
                            reconstructed_images, extra_results_dict = model(images, cond_images)
                    else:
                        # 正常模式
                        reconstructed_images, extra_results_dict = model(images, cond_images)
                    
                    if store_original_pixel_inputs:
                        extra_results_dict["original_pixel_inputs"] = original_pixel_inputs
                    autoencoder_loss, loss_dict = loss_module(
                        images,
                        reconstructed_images,
                        extra_results_dict,
                        global_step,
                        mode="generator",
                    )
                elif model_type == "sd3_tokenizer":
                    reconstructed_images, extra_results_dict = model(images, cond_images)
                    
                    if store_original_pixel_inputs:
                        extra_results_dict["original_pixel_inputs"] = original_pixel_inputs
                    
                    # 检查是否使用DMD损失类型
                    if config.losses.get("use_dmd_loss", False):
                        autoencoder_loss, loss_dict = loss_module(
                            images,
                            reconstructed_images,
                            extra_results_dict,
                            global_step,
                            mode=loss_module.get_training_mode(),
                        )
                    else:
                        autoencoder_loss, loss_dict = loss_module(
                            images,
                            reconstructed_images,
                            extra_results_dict,
                            global_step,
                            mode="generator",
                        )
                elif model_type == "sd3_tokenizer_2d":
                    # SD3Tokenizer2D 使用与 sd3_tokenizer 相同的调用方式
                    reconstructed_images, extra_results_dict = model(images, cond_images)
                    
                    if store_original_pixel_inputs:
                        extra_results_dict["original_pixel_inputs"] = original_pixel_inputs
                    
                    # 检查是否使用DMD损失类型
                    if config.losses.get("use_dmd_loss", False):
                        # 使用DMD损失类型
                        autoencoder_loss, loss_dict = loss_module(
                            images,
                            reconstructed_images,
                            extra_results_dict,
                            global_step,
                            mode=loss_module.get_training_mode(),  # 获取当前训练模式
                        )
                    else:
                        # 使用传统损失
                        autoencoder_loss, loss_dict = loss_module(
                            images,
                            reconstructed_images,
                            extra_results_dict,
                            global_step,
                            mode="generator",
                        )
                elif model_type == "sd3_tokenizer_sr":
                    if cond_images is None:
                        raise ValueError("sd3_tokenizer_sr 模式需要 cond_images 作为学生输入")
                    distill_outputs = accelerator.unwrap_model(model).distill_forward(
                        teacher_inputs=images,
                        lq_cond=cond_images,
                        image_ids=batch.get('__key__') if isinstance(batch, dict) and '__key__' in batch else None
                    )
                    student_hidden = distill_outputs["student_hidden_states"]
                    teacher_hidden = distill_outputs["teacher_hidden_states"]
                    reconstructed_images = distill_outputs.get("student_reconstruction")
                    extra_results_dict = distill_outputs
                    autoencoder_loss, loss_dict = loss_module(
                        student_hidden,
                        teacher_hidden,
                        extra_result_dict=distill_outputs,
                        global_step=global_step
                    )
                elif model_type == "cond_upsampler":
                    # 仅依赖 cond_images 作为输入，输出对齐 images
                    if cond_images is None:
                        raise ValueError("cond_upsampler 训练需要 cond_image 或 use_tensor_downsample 模式提供条件输入")
                    reconstructed_images, extra_results_dict = model(cond_images)
                    if store_original_pixel_inputs:
                        extra_results_dict["original_pixel_inputs"] = original_pixel_inputs
                    autoencoder_loss, loss_dict = loss_module(
                        images,
                        reconstructed_images,
                        extra_results_dict,
                        global_step,
                        mode="generator",
                    )
                elif model_type == "flux_tokenizer":
                    # FluxTokenizer 需要低质条件输入
                    if cond_images is None:
                        raise ValueError("flux_tokenizer 训练需要 cond_image 或 use_tensor_downsample 模式提供条件输入")
                    reconstructed_images, extra_results_dict = model(images, cond_images)
                    if store_original_pixel_inputs:
                        extra_results_dict["original_pixel_inputs"] = original_pixel_inputs
                    if config.losses.get("use_dmd_loss", False):
                        autoencoder_loss, loss_dict = loss_module(
                            images,
                            reconstructed_images,
                            extra_results_dict,
                            global_step,
                            mode=loss_module.get_training_mode(),
                        )
                    else:
                        autoencoder_loss, loss_dict = loss_module(
                            images,
                            reconstructed_images,
                            extra_results_dict,
                            global_step,
                            mode="generator",
                        )
                elif model_type == "sd3_da_vae":
                    # SD3 DA-VAE 自编码器：输入使用像素图像；GT可选用SD3 VAE重建结果
                    # 准备GT（若启用use_sd3_vae/use_flux_vae，则用VAE重建像素作为监督）
                    gt_images = original_pixel_inputs
                    use_vae = config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False)
                    if original_vae_model is not None and use_vae:
                        with torch.no_grad():
                            encoded = original_vae_model.encode(original_pixel_inputs)
                            gt_images = original_vae_model.decode(encoded)
                        gt_images = gt_images.to(get_model_dtype(config))

                    # 前向与损失
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
                else:
                    raise NotImplementedError
    
                # 梯度可视化分析（在backward之前，避免计算图被释放）
                # 只在最后一次累积步骤时执行，避免重复执行
                if (config.training.get("enable_gradient_visualization", False) and 
                    accelerator.is_main_process and 
                    accelerator.sync_gradients and  # 只在同步梯度时执行（最后一次累积步骤）
                    global_step % config.training.get("gradient_vis_interval", 1000) == 0 and
                    reconstructed_images is not None):
                    
                    # 初始化梯度分析器（如果还没有初始化）
                    if not hasattr(train_one_epoch, 'gradient_analyzer'):
                        train_one_epoch.gradient_analyzer = GradientAnalyzer(
                            loss_module=accelerator.unwrap_model(loss_module),
                            log_interval=config.training.get("gradient_log_interval", 100),
                            vis_interval=config.training.get("gradient_vis_interval", 1000),
                            save_dir=os.path.join(config.experiment.output_dir, "gradient_visualizations"),
                            enable_logging=True,
                            enable_visualization=config.training.get("enable_gradient_visualization", False)
                        )
                    
                    # 分析梯度（在backward之前）
                    try:
                        gradient_stats = train_one_epoch.gradient_analyzer.analyze_step(
                            images, reconstructed_images, extra_results_dict, global_step
                        )
                        
                        # 将梯度统计信息添加到日志中
                        if gradient_stats:
                            for key, value in gradient_stats.items():
                                autoencoder_logs[f"gradients/{key}"] = value
                                
                    except Exception as e:
                        logger.warning(f"梯度分析失败: {e}")
    
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
    
                # Only apply gradient clipping if we're not in warmup period and there are gradients
                if (not accelerator.unwrap_model(loss_module).is_discriminator_warmup_period(global_step) and 
                    config.training.max_grad_norm is not None and accelerator.sync_gradients):
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
    
                # Skip generator optimizer step during discriminator warmup
                if not accelerator.unwrap_model(loss_module).is_discriminator_warmup_period(global_step):
                    optimizer.step()
                    lr_scheduler.step()
    
                # Log gradient norm before zeroing it.
                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                    and not accelerator.unwrap_model(loss_module).is_discriminator_warmup_period(global_step)
                ):
                    log_grad_norm(model, accelerator, global_step + 1)
    
                # 监控encoder梯度（在backward之后，optimizer.step之前）
                if (global_step + 1) % (config.experiment.log_grad_norm_every) == 0 and accelerator.is_main_process:
                    # 导入监控函数
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from scripts.train_srtitok import monitor_encoder_gradients, monitor_decoder_gradients, monitor_z_params_stats
                    monitor_encoder_gradients(model, logger, global_step + 1, model_type)
                    monitor_decoder_gradients(model, logger, global_step + 1, model_type)
                    monitor_z_params_stats(model, logger, global_step + 1, model_type)
    
                optimizer.zero_grad(set_to_none=True)
    
                # Train discriminator.
                discriminator_logs = defaultdict(float)
                loss_module_unwrapped = accelerator.unwrap_model(loss_module)
                
                # Get discriminator training frequency based on progressive training schedule
                discriminator_frequency = loss_module_unwrapped.get_discriminator_training_frequency(global_step)
                
                # Train discriminator based on frequency (0 means don't train, 1 means every step, 2 means every 2 steps, etc.)
                should_train_discriminator = (
                    discriminator_frequency > 0 
                    and global_step % discriminator_frequency == 0
                )
                # print(f"should_train_discriminator: {should_train_discriminator}")
                
                if should_train_discriminator:
                    # Store discriminator parameters before update for monitoring
                    # discriminator_logs = defaultdict(float)
                    discriminator_loss, loss_dict_discriminator = loss_module(
                        images,
                        reconstructed_images,
                        extra_results_dict,
                        global_step=global_step,
                        mode="discriminator",
                    )
    
                    # Gather the losses across all processes for logging.
                    for k, v in loss_dict_discriminator.items():
                        # Use accelerator.gather() for all metrics to properly aggregate across GPUs
                        if type(v) == torch.Tensor:
                            discriminator_logs["train/" + k] = accelerator.gather(v).mean().item()
                        else:
                            discriminator_logs["train/" + k] = v
    
                    # Add discriminator frequency to logs for monitoring
                    discriminator_logs["train/discriminator_frequency"] = discriminator_frequency
    
                    accelerator.backward(discriminator_loss)
    
                    # # Monitor discriminator gradients
                    # if global_step % config.experiment.log_every == 0 and accelerator.is_main_process:
                    #     discriminator_grad_norm = 0.0
                    #     total_grads = 0
                    #     for name, param in loss_module.named_parameters():
                    #         if 'discriminator' in name and param.grad is not None:
                    #             grad_norm = param.grad.norm().item()
                    #             discriminator_grad_norm += grad_norm ** 2
                    #             total_grads += 1
                        
                    #     if total_grads > 0:
                    #         discriminator_grad_norm = discriminator_grad_norm ** 0.5
                    #         discriminator_logs["train/discriminator_grad_norm"] = discriminator_grad_norm
                    #         logger.info(f"Discriminator grad norm: {discriminator_grad_norm:.2e}")
                    #     else:
                    #         discriminator_logs["train/discriminator_grad_norm"] = 0.0
                    #         logger.info("Discriminator has no gradients")
    
                    # Only clip grad norm if optimizer会step
                    if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                        if discriminator_optimizer is not None:
                            discriminator_lr = discriminator_optimizer.param_groups[0]['lr']
                            skip_when_zero = config.optimizer.params.get("skip_discriminator_optimizer_when_lr_zero", False)
                            if discriminator_lr > 0 or not skip_when_zero:
                                accelerator.clip_grad_norm_(loss_module.parameters(), config.training.max_grad_norm)
    
                    # Only update discriminator if learning rate is not 0
                    if discriminator_optimizer is not None:
                        discriminator_lr = discriminator_optimizer.param_groups[0]['lr']
                        skip_when_zero = config.optimizer.params.get("skip_discriminator_optimizer_when_lr_zero", False)
                        
                        if discriminator_lr > 0 or not skip_when_zero:
                            discriminator_optimizer.step()
                            if discriminator_lr_scheduler is not None:
                                discriminator_lr_scheduler.step()
                            
                            # Update discriminator EMA after optimizer step
                            if hasattr(loss_module_unwrapped, 'update_discriminator_ema'):
                                loss_module_unwrapped.update_discriminator_ema(global_step)
                    else:
                        # Skip optimizer step when learning rate is 0
                        if global_step % config.experiment.log_every == 0 and accelerator.is_main_process:
                            logger.info(f"Skipping discriminator optimizer step due to lr=0")
            
                    # Log gradient norm before zeroing it.
                    if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                    ):
                        log_grad_norm(loss_module, accelerator, global_step + 1)
                    
                    if discriminator_optimizer is not None:
                        discriminator_optimizer.zero_grad(set_to_none=True)
                    
                    # # Monitor discriminator parameter updates
                    # if global_step % config.experiment.log_every == 0 and accelerator.is_main_process:
                    #     discriminator_update_norm = 0.0
                    #     total_params = 0
                    #     for name, param in loss_module.named_parameters():
                    #         if 'discriminator' in name and name in discriminator_params_before:
                    #             param_diff = param.data - discriminator_params_before[name]
                    #             param_norm = param_diff.norm().item()
                    #             discriminator_update_norm += param_norm ** 2
                    #             total_params += 1
                        
                    #     if total_params > 0:
                    #         discriminator_update_norm = discriminator_update_norm ** 0.5
                    #         discriminator_logs["train/discriminator_param_update_norm"] = discriminator_update_norm
                    #         discriminator_logs["train/discriminator_lr"] = discriminator_optimizer.param_groups[0]['lr']
                            
                    #         # Log if discriminator is actually updating
                    #         discriminator_lr = discriminator_optimizer.param_groups[0]['lr']
                    #         if discriminator_lr > 0 and discriminator_update_norm > 1e-8:
                    #             logger.info(f"Discriminator IS updating: norm={discriminator_update_norm:.2e}, lr={discriminator_lr:.2e}")
                    #         elif discriminator_lr > 0 and discriminator_update_norm <= 1e-8:
                    #             logger.info(f"Discriminator NOT updating: norm={discriminator_update_norm:.2e}, lr={discriminator_lr:.2e}")
                    #         else:
                    #             logger.info(f"Discriminator SKIPPED (lr=0): norm={discriminator_update_norm:.2e}, lr={discriminator_lr:.2e}")
                    
                    # Store current discriminator logs for future use
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
                
                # Get current discriminator frequency for logging
                current_discriminator_frequency = loss_module_unwrapped.get_discriminator_training_frequency(global_step)
                
                # 记录分离学习率信息
                peak_memory_gb = None
                if accelerator.device.type == "cuda":
                    peak_memory_gb = torch.cuda.max_memory_allocated(accelerator.device) / (1024 ** 3)

                if hasattr(optimizer, 'use_separate_lr') and optimizer.use_separate_lr:
                    # 使用分离学习率时的日志
                    log_message = (
                        f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_meter.val:0.4f} "
                        f"Compute (t): {compute_time:0.4f} "
                        f"Loader%: {loader_pct*100:0.1f}% "
                        f"LR Base: {lr:0.6f} Encoder: {optimizer.encoder_lr:0.6f} Decoder: {optimizer.decoder_lr:0.6f} "
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
                        "lr/base": lr,
                        "lr/encoder": optimizer.encoder_lr,
                        "lr/decoder": optimizer.decoder_lr,
                        "lr/generator": lr,
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "time/data_time": data_time_meter.val,
                        "time/batch_time": batch_time_meter.val,
                        "time/compute_time": compute_time,
                        "time/loader_pct": loader_pct,
                        "discriminator_frequency": current_discriminator_frequency,
                    }
                else:
                    # 使用统一学习率时的日志
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
                
                # Add discriminator learning rate to logs
                if discriminator_optimizer is not None:
                    discriminator_lr = discriminator_optimizer.param_groups[0]['lr']
                    logs["lr/discriminator"] = discriminator_lr
                
                logs.update(autoencoder_logs)
                # Use current discriminator logs if available, otherwise use last stored logs
                if discriminator_logs:
                    logs.update(discriminator_logs)
                elif last_discriminator_logs:
                    logs.update(last_discriminator_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger, config=config, loss_module=loss_module)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
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
                    text_guidance=text_guidance[:config.training.num_generated_images] if model_type == "tatitok" else None,
                    cond_images=cond_images[:config.training.num_generated_images] if (model_type in ("srtitok", "sd3_tokenizer", "sd3_tokenizer_sr", "sd3_tokenizer_2d")) and cond_images is not None else None,
                    pretrained_tokenizer=pretrained_tokenizer,
                    original_vae_model=original_vae_model,
                    skip_vae_encoding=True
                ) # here we don't need to pass original_vae_model because we have already encoded the images in the data loading process

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())


            # Evaluate reconstruction.
            if eval_dataloader is not None and (global_step + 1) % config.experiment.eval_every == 0:
                # 只在主卡上运行评估，避免多卡数据不一致的问题
                if accelerator.is_main_process:
                    logger.info(f"Computing metrics on the validation set (main process only).")
                    if config.training.get("use_ema", False):
                        ema_model.store(model.parameters())
                        ema_model.copy_to(model.parameters())
                        # Eval for EMA.
                        eval_scores = eval_reconstruction(
                            model,
                            eval_dataloader,
                            accelerator,
                            evaluator,
                            model_type=model_type,
                            clip_tokenizer=clip_tokenizer,
                            clip_encoder=clip_encoder,
                            pretrained_tokenizer=pretrained_tokenizer,
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
                            # Switch back to the original model parameters for training.
                            ema_model.restore(model.parameters())
                    else:
                        # Eval for non-EMA.
                        eval_scores = eval_reconstruction(
                            model,
                            eval_dataloader,
                            accelerator,
                            evaluator,
                            model_type=model_type,
                            clip_tokenizer=clip_tokenizer,
                            clip_encoder=clip_encoder,
                            pretrained_tokenizer=pretrained_tokenizer,
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
                    # 非主卡等待主卡完成评估
                    logger.info(f"Waiting for main process to complete evaluation...")

                accelerator.wait_for_everyone()

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break

    # 训练循环结束，记录内存使用情况并清理缓存
    if accelerator.is_main_process:
        log_memory_usage(logger, f"训练结束 (Step {global_step})")
        # 定期清理内存缓存
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


def get_rar_random_ratio(config, cur_step):
    randomness_anneal_start = config.model.generator.randomness_anneal_start
    randomness_anneal_end = config.model.generator.randomness_anneal_end
    if cur_step < randomness_anneal_start:
        return 1.0
    elif cur_step > randomness_anneal_end:
        return 0.0
    else:
        return 1.0 - (cur_step - randomness_anneal_start) / (randomness_anneal_end - randomness_anneal_start)


def train_one_epoch_generator(
                    config, logger, accelerator,
                    model, ema_model, loss_module,
                    optimizer,
                    lr_scheduler,
                    train_dataloader,
                    tokenizer,
                    global_step,
                    model_type="maskgit"):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()

    for i, batch in enumerate(train_dataloader):
        model.train()
        if config.dataset.params.get("pretokenization", ""):
            # the data is already pre-tokenized
            conditions, input_tokens = batch
            input_tokens = input_tokens.to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
            conditions = conditions.to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        else:
            # tokenize on the fly
            if "image" in batch:
                images = batch["image"].to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )
                conditions = batch["class_id"].to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )

                # Encode images on the flight.
                with torch.no_grad():
                    tokenizer.eval()
                    input_tokens = tokenizer.encode(images)[1]["min_encoding_indices"].reshape(images.shape[0], -1)
            else:
                raise ValueError(f"Not found valid keys: {batch.keys()}")

        data_time_meter.update(time.time() - end)

        unwrap_model = accelerator.unwrap_model(model)


        if model_type == "maskgit":
            # Randomly masking out input tokens.
            masked_tokens, masks = unwrap_model.masking_input_tokens(
                input_tokens)
        elif model_type == "rar":
            unwrap_model.set_random_ratio(get_rar_random_ratio(config, global_step))
        else:
            raise NotImplementedError
            

        with accelerator.accumulate([model]):

            if model_type == "maskgit":
                logits = model(masked_tokens, conditions,
                            cond_drop_prob=config.model.generator.class_label_dropout)
                loss, loss_dict= loss_module(logits, input_tokens, weights=masks)
            elif model_type == "rar":
                condition = unwrap_model.preprocess_condition(
                    conditions, cond_drop_prob=config.model.generator.class_label_dropout
                )
                logits, labels = model(input_tokens, condition, return_labels=True)
                loss, loss_dict = loss_module(logits, labels)
            # Gather the losses across all processes for logging.
            gen_logs = {}
            for k, v in loss_dict.items():
                if type(v) == torch.Tensor:
                    gen_logs["train/" + k] = accelerator.gather(v).mean().item()
                else:
                    gen_logs["train/" + k] = v
            accelerator.backward(loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

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
                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Loss: {gen_logs['train/loss']:0.4f} "
                    f"Accuracy: {gen_logs['train/correct_tokens']:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(gen_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger, config=config, loss_module=loss_module)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                generate_images(
                    model,
                    tokenizer,
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config
                )

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break


    return global_step


def train_one_epoch_t2i_generator(
                    config, logger, accelerator,
                    model, ema_model, loss_module,
                    optimizer,
                    lr_scheduler,
                    train_dataloader,
                    tokenizer,
                    clip_tokenizer,
                    clip_encoder,
                    global_step,
                    model_type="maskgen_vq"):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()

    for i, batch in enumerate(train_dataloader):
        model.train()
        
        input_tokens = batch["tokens"].to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True)
        captions = batch["text"]
        if config.model.maskgen.micro_condition:
            aes_scores = batch["aes_score"].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        else:
            aes_scores = None

        data_time_meter.update(time.time() - end)

        unwrap_model = accelerator.unwrap_model(model)

        condition, condition_pooled = unwrap_model.preprocess_condition(
            captions, clip_tokenizer, clip_encoder,
        )

        if model_type == "maskgen_vq":
            with accelerator.accumulate([model]):
                logits, masks = model(input_tokens, condition, condition_pooled, aes_scores)
                t2i_gen_loss, loss_dict = loss_module(logits, input_tokens, weights=masks)
        elif model_type == "maskgen_kl":
            with accelerator.accumulate([model]):
                t2i_gen_loss, loss_dict = model(input_tokens, condition, condition_pooled, aes_scores)
        else:
            raise NotImplementedError

        with accelerator.accumulate([model]):
            # Gather the losses across all processes for logging.
            t2i_gen_logs = {}
            for k, v in loss_dict.items():
                if type(v) == torch.Tensor:
                    t2i_gen_logs["train/" + k] = accelerator.gather(v).mean().item()
                else:
                    t2i_gen_logs["train/" + k] = v
            accelerator.backward(t2i_gen_loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

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
                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Loss: {t2i_gen_logs['train/loss']:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(t2i_gen_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger, config=config, loss_module=loss_module)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                t2i_generate_images(
                    model,
                    tokenizer,
                    captions,
                    aes_scores,
                    clip_tokenizer,
                    clip_encoder,
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config,
                    model_type=model_type,
                )

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break


    return global_step


@torch.no_grad()
def eval_reconstruction(
    model,
    eval_loader,
    accelerator,
    evaluator,
    model_type="titok",
    clip_tokenizer=None,
    clip_encoder=None,
    pretrained_tokenizer=None,
    original_vae_model=None,
    max_eval_images=1000,
    config=None,
    logger=None,
):
    eval_use_training_mode = False
    if config is not None and hasattr(config, 'model'):
        eval_use_training_mode = config.model.get('eval_use_training_mode', False)
    force_training_mode = model_type in ('sd3_tokenizer', 'sd3_tokenizer_2d', 'sd3_tokenizer_sr') and eval_use_training_mode

    if force_training_mode:
        model.train()
        accelerator.unwrap_model(model).train()
    else:
        model.eval()
    evaluator.reset_metrics()
    local_model = accelerator.unwrap_model(model)
    if force_training_mode:
        local_model.train()
    
    # 如果没有传入original_vae_model，尝试从模型中获取
    if original_vae_model is None and hasattr(local_model, 'original_vae_model'):
        original_vae_model = local_model.original_vae_model
        if logger:
            logger.info("eval_reconstruction: 从模型中获取原始VAE模型进行图像编码")
    elif original_vae_model is not None and logger:
        logger.info("eval_reconstruction: 使用传入的原始VAE模型进行图像编码")
    
    processed_images = 0
    sr_alignment_total = 0.0
    sr_alignment_count = 0

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
        
        # Apply VAE encoding to inputs only for latent-space models
        original_images = torch.clone(images)
        if original_vae_model is not None:
            use_vae = config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False)
            latent_models = ("sd3_tokenizer", "sd3_tokenizer_2d", "sd3_tokenizer_sr", "flux_tokenizer")
            if use_vae and model_type in latent_models:
                with torch.no_grad():
                    images = original_vae_model.encode(images)
        
        # 检查是否已经处理了足够的图像
        batch_size = images.shape[0]
        if processed_images + batch_size > max_eval_images:
            # 只处理需要的部分
            remaining = max_eval_images - processed_images
            images = images[:remaining]
            original_images = original_images[:remaining]
        
        if model_type == "tatitok":
            conditions = batch["class_id"]
            if processed_images + batch_size > max_eval_images:
                remaining = max_eval_images - processed_images
                conditions = conditions[:remaining]
            text = [f"A photo of a {imagenet_idx2classname[condition.item()]}." for condition in conditions]
            text_guidance = clip_tokenizer(text).to(accelerator.device)
            cast_dtype = clip_encoder.transformer.get_cast_dtype()
            text_guidance = clip_encoder.token_embedding(text_guidance).to(cast_dtype)  # [batch_size, n_ctx, d_model]
            text_guidance = text_guidance + clip_encoder.positional_embedding.to(cast_dtype)
            text_guidance = text_guidance.permute(1, 0, 2)  # NLD -> LND
            text_guidance = clip_encoder.transformer(text_guidance, attn_mask=clip_encoder.attn_mask)
            text_guidance = text_guidance.permute(1, 0, 2)  # LND -> NLD
            text_guidance = clip_encoder.ln_final(text_guidance)  # [batch_size, n_ctx, transformer.width]

        # 检查是否需要处理cond_images
        use_tensor_downsample = config.model.get('use_tensor_downsample', False)
        if (model_type in ("srtitok", "sd3_tokenizer", "sd3_tokenizer_sr", "sd3_tokenizer_2d")) and (use_tensor_downsample or "cond_image" in batch):
            
            if use_tensor_downsample:
                # 在tensor层面downsample模式下，从原始images构造cond_images
                # 使用新的use_tensor_downsample_mode函数
                lq_cond_mode = config.model.get('lq_cond_mode', 'auto')
                
                # 保存原始数据类型
                original_dtype = original_images.dtype
                
                # 进行下采样
                if original_dtype == torch.bfloat16:
                    # 临时转换为 float32 进行插值
                    cond_images = use_tensor_downsample_mode(
                        original_images.to(torch.float32), 
                        config, 
                        lq_cond_mode
                    )
                    # 转换回原始精度
                    cond_images = cond_images.to(original_dtype)
                else:
                    cond_images = use_tensor_downsample_mode(
                        original_images, 
                        config, 
                        lq_cond_mode
                    )
                
                if processed_images + batch_size > max_eval_images:
                    remaining = max_eval_images - processed_images
                    cond_images = cond_images[:remaining]
            else:
                # 传统模式：处理cond_image
                cond_images = batch["cond_image"]
                
                # 检查是否有None值（skip_cond_image模式）
                if cond_images is not None:
                    cond_images = cond_images.to(
                        accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                    )
                    if processed_images + batch_size > max_eval_images:
                        remaining = max_eval_images - processed_images
                        cond_images = cond_images[:remaining]
            
            # 统一的VAE编码逻辑（仅对latent空间模型编码条件）
            if cond_images is not None and original_vae_model is not None:
                use_vae = config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False)
                latent_models = ("sd3_tokenizer", "sd3_tokenizer_2d", "sd3_tokenizer_sr", "flux_tokenizer")
                if use_vae and model_type in latent_models:
                    with torch.no_grad():
                        cond_images = original_vae_model.encode(cond_images)
            
            # 应用noise注入（如果启用，评估时使用固定种子）
            if cond_images is not None and hasattr(config, 'noise_injection') and config.noise_injection.enabled:
                cond_images = inject_noise_to_lq_cond(cond_images, config, training=False)
        
        if model_type == "titok":
            reconstructed_images, model_dict = local_model(images)
        elif model_type == "tatitok":
            reconstructed_images, model_dict = local_model(images, text_guidance)
        elif model_type == "srtitok":
            # 在id_based模式下，需要传递image_ids参数
            if hasattr(config, 'model') and hasattr(config.model.vq_model, 'encoder_mode') and config.model.vq_model.encoder_mode == "id_based":
                # 从batch中获取image ids
                if "__key__" in batch:
                    image_ids = batch["__key__"]
                    if isinstance(image_ids, torch.Tensor):
                        image_ids = image_ids.tolist()
                    reconstructed_images, model_dict = local_model(images, cond_images, image_ids)
                else:
                    logger.warning("id_based模式下缺少__key__字段，无法获取image_ids")
                    reconstructed_images, model_dict = local_model(images, cond_images)
            else:
                # 正常模式
                reconstructed_images, model_dict = local_model(images, cond_images)
        elif model_type == "sd3_tokenizer":
            # SD3Tokenizer 使用与 srtitok 相同的调用方式，可按需在评估时保持训练模式
            reconstructed_images, model_dict = local_model(images, cond_images, training=force_training_mode)
        elif model_type == "sd3_tokenizer_2d":
            # SD3Tokenizer2D 使用与 sd3_tokenizer 相同的调用方式
            reconstructed_images, model_dict = local_model(images, cond_images, training=force_training_mode)
        elif model_type == "sd3_da_vae":
            # SD3 DA-VAE：直接前向得到像素重建
            reconstructed_images, model_dict = local_model(images, sample_posterior=False)
        elif model_type == "sd3_tokenizer_sr":
            distill_outputs = local_model.distill_forward(images, cond_images)
            sr_mse = torch.nn.functional.mse_loss(
                distill_outputs["student_hidden_states"],
                distill_outputs["teacher_hidden_states"]
            )
            sr_alignment_total += sr_mse.detach().item() * images.shape[0]
            sr_alignment_count += images.shape[0]
            # reconstructed_images, model_dict = local_model(images, cond_images, training=force_training_mode)
            reconstructed_images = distill_outputs["student_reconstruction"]
            model_dict = {"sr_alignment_loss": sr_mse.detach(), "student_encoder_dict": distill_outputs["student_encoder_dict"], "teacher_encoder_dict": distill_outputs["teacher_encoder_dict"]}
            if isinstance(model_dict, dict):
                model_dict["sr_alignment_loss"] = sr_mse.detach()
        else:
            raise NotImplementedError

        # 根据模型类型决定是否需要解码/替换GT
        if original_vae_model is not None:
            use_vae = config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False)
            latent_models = ("sd3_tokenizer", "sd3_tokenizer_2d", "sd3_tokenizer_sr", "flux_tokenizer")
            if model_type in latent_models and use_vae:
                # 保存原始latent图像用于latent MSE计算
                original_latents = torch.clone(images)
                reconstructed_latents = torch.clone(reconstructed_images)
                
                vae_mode = config.model.get("flux_vae_mode", "latent_to_latent")
                use_original_pixel_supervision = config.losses.get("use_original_pixel_supervision", False)
                if not use_original_pixel_supervision:
                    original_images = original_vae_model.decode(images)
                if vae_mode.startswith("latent_to_latent"):
                    reconstructed_images = original_vae_model.decode(reconstructed_images)
            elif model_type == "sd3_da_vae" and use_vae:
                # Pixel模型：不解码模型输出；将GT替换为VAE重建像素
                with torch.no_grad():
                    encoded = original_vae_model.encode(original_images)
                    original_images = original_vae_model.decode(encoded)
            
     
    

        if pretrained_tokenizer is not None:
            reconstructed_images = pretrained_tokenizer.decode(reconstructed_images.argmax(1))
        reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
        # Quantize to uint8 (match train-time evaluator behavior)
        reconstructed_images = torch.round(reconstructed_images * 255.0) / 255.0
        original_images = torch.clamp(original_images, 0.0, 1.0)

        # Optionally save a subset of eval samples
        if save_eval_images and save_eval_dir is not None and processed_images < save_eval_max:
            try:
                # how many to save from this batch
                remain = save_eval_max - processed_images
                n_save = min(remain, original_images.shape[0])
                # move to cpu for saving
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
                    # save PNGs
                    from torchvision.transforms.functional import to_pil_image
                    to_pil_image(orig_cpu[i]).save(save_eval_dir / f"{idx:06d}_original.png")
                    to_pil_image(recon_cpu[i]).save(save_eval_dir / f"{idx:06d}_reconstructed.png")
            except Exception as e:
                if logger:
                    logger.warning(f"保存评估图像失败: {e}")
        

        # For VAE model.
        if original_vae_model is not None and model_type in ("sd3_tokenizer", "sd3_tokenizer_2d", "sd3_tokenizer_sr", "flux_tokenizer"):
            vae_mode = config.model.get("flux_vae_mode", "latent_to_latent")
            if vae_mode == "latent_to_pixel":
                evaluator.update(original_images, reconstructed_images.squeeze(2), None)
            else:
                evaluator.update(original_images, reconstructed_images.squeeze(2), None, 
                                original_latents, reconstructed_latents)
        else:
            evaluator.update(original_images, reconstructed_images.squeeze(2), None)
        
        # 更新已处理的图像数量
        processed_images += images.shape[0]
        
        # 如果已经处理了足够的图像，就停止
        if processed_images >= max_eval_images:
            break
            
    model.train()
    results = evaluator.result()
    if sr_alignment_count > 0:
        results["sr_alignment_mse"] = sr_alignment_total / sr_alignment_count
    return results


@torch.no_grad()
def reconstruct_images(model, original_images, fnames, accelerator, 
                    global_step, output_dir, logger, config=None,
                    model_type="titok", text_guidance=None, cond_images=None,   
                    pretrained_tokenizer=None, original_vae_model=None, skip_vae_encoding=False):
    logger.info("Reconstructing images...")
    original_images = torch.clone(original_images)
    original_images = original_images.to(dtype=next(model.parameters()).dtype, device=accelerator.device)
    
    # 如果没有传入original_vae_model，尝试从模型中获取
    local_model = accelerator.unwrap_model(model)
    if original_vae_model is None and hasattr(local_model, 'original_vae_model'):
        original_vae_model = local_model.original_vae_model
        logger.info("reconstruct_images: 从模型中获取原始VAE模型进行图像编码")
    elif original_vae_model is not None:
        logger.info("reconstruct_images: 使用传入的原始VAE模型进行图像编码")
    
    # Store original images for visualization
    
    # Apply VAE encoding for inputs only for latent-space models
    if original_vae_model is not None and not skip_vae_encoding:
        use_vae = config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False)
        latent_models = ("sd3_tokenizer", "sd3_tokenizer_2d", "sd3_tokenizer_sr", "flux_tokenizer")
        if use_vae and model_type in latent_models:
            with torch.no_grad():
                original_images = original_vae_model.encode(original_images)

    original_images_for_viz = torch.clone(original_images)
    
    
    if cond_images is not None:
        cond_images = cond_images.to(dtype=next(model.parameters()).dtype, device=accelerator.device)
        # Apply VAE encoding to condition images if enabled
        # 检查是否使用tensor downsample模式
        use_tensor_downsample = config.model.get('use_tensor_downsample', False)
        if original_vae_model is not None and not skip_vae_encoding:
            use_vae = config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False)
            latent_models = ("sd3_tokenizer", "sd3_tokenizer_2d", "sd3_tokenizer_sr", "flux_tokenizer")
            if use_vae and model_type in latent_models:
                with torch.no_grad():
                    cond_images = original_vae_model.encode(cond_images)
              
                
        # 应用noise注入（如果启用，图像重建时使用固定种子）
        if hasattr(config, 'noise_injection') and config.noise_injection.enabled:
            cond_images = inject_noise_to_lq_cond(cond_images, config, training=False)
    
    eval_use_training_mode = False
    if config is not None and hasattr(config, 'model'):
        eval_use_training_mode = config.model.get('eval_use_training_mode', False)
    force_training_mode = model_type in ('sd3_tokenizer', 'sd3_tokenizer_2d', 'sd3_tokenizer_sr') and eval_use_training_mode

    if force_training_mode:
        model.train()
        accelerator.unwrap_model(model).train()
    else:
        model.eval()
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
      
        if model_type == "srtitok":
            # 在id_based模式下，需要传递image_ids参数
            if hasattr(config.model.vq_model, 'encoder_mode') and config.model.vq_model.encoder_mode == "id_based":
                # 从fnames中获取image ids
                if isinstance(fnames, (list, tuple)):
                    image_ids = fnames
                elif isinstance(fnames, torch.Tensor):
                    image_ids = fnames.tolist()
                else:
                    image_ids = [str(fnames)] if fnames else []
                enc_tokens, encoder_dict = accelerator.unwrap_model(model).encode(original_images, cond_images, image_ids)
            else:
                enc_tokens, encoder_dict = accelerator.unwrap_model(model).encode(original_images, cond_images)
        elif model_type in ("sd3_tokenizer"):
            # SD3Tokenizer 需要 lq_cond，并且 encode 返回 (z, dict, lq_cond)
            encode_result = accelerator.unwrap_model(model).encode(original_images, cond_images)
            # 兼容老接口（二元返回）
            if isinstance(encode_result, tuple) and len(encode_result) == 3:
                enc_tokens, encoder_dict, lq_cond_from_encode = encode_result
            else:
                enc_tokens, encoder_dict = encode_result
                lq_cond_from_encode = cond_images
        elif model_type == "sd3_tokenizer_2d":
            # SD3Tokenizer2D 使用与 sd3_tokenizer 相同的调用方式
            encode_result = accelerator.unwrap_model(model).encode(original_images, cond_images)
            # 兼容老接口（二元返回）
            if isinstance(encode_result, tuple) and len(encode_result) == 3:
                enc_tokens, encoder_dict, lq_cond_from_encode = encode_result
            else:
                enc_tokens, encoder_dict = encode_result
                lq_cond_from_encode = cond_images
        elif model_type == "sd3_da_vae":
            # Pixel autoencoder; no token encoding step required
            pass
        else:
            enc_tokens, encoder_dict = accelerator.unwrap_model(model).encode(original_images)
    
        if model_type == "titok":
            reconstructed_images = accelerator.unwrap_model(model).decode(enc_tokens)
        elif model_type == "tatitok":
            reconstructed_images = accelerator.unwrap_model(model).decode(enc_tokens, text_guidance)
        elif model_type == "srtitok":
            # 对于srtitok，需要传递target_shape来正确计算mask token数量
            target_shape = (original_images.shape[2], original_images.shape[3])
            reconstructed_images = accelerator.unwrap_model(model).decode(enc_tokens, cond_images, target_shape)
        elif model_type in ("sd3_tokenizer"):
            # 对于SD3Tokenizer，使用与 srtitok 相同的调用方式，可选地保持训练模式以测试decode dropout
            target_shape = (original_images.shape[2], original_images.shape[3])
            reconstructed_images = accelerator.unwrap_model(model).decode(enc_tokens, lq_cond_from_encode, target_shape, training=force_training_mode)
        elif model_type == "sd3_tokenizer_2d":
            # 对于SD3Tokenizer2D，使用与 sd3_tokenizer 相同的调用方式
            target_shape = (original_images.shape[2], original_images.shape[3])
            reconstructed_images = accelerator.unwrap_model(model).decode(enc_tokens, lq_cond_from_encode, target_shape, training=force_training_mode)
        elif model_type == "sd3_da_vae":
            # Direct forward to get pixel reconstructions
            reconstructed_images, _ = accelerator.unwrap_model(model)(original_images, sample_posterior=False)
        
        if model_type == "sd3_tokenizer_sr":
            reconstructed_images = accelerator.unwrap_model(model).distill_forward(original_images, cond_images)["student_reconstruction"]
    # 根据vae_mode决定是否需要解码（仅对 latent 模型执行，像素自编码器不解码）
    if original_vae_model is not None:
        vae_mode = config.model.get("flux_vae_mode", "latent_to_pixel")
        latent_models = ("sd3_tokenizer", "sd3_tokenizer_2d", "sd3_tokenizer_sr", "flux_tokenizer")
        if model_type in latent_models:
            if vae_mode == "latent_to_pixel":
                # latent->pixel：原图需要解码用于可视化；重建已是pixel
                if config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False):
                    original_images_for_viz = original_vae_model.decode(original_images_for_viz)
            else:
                # latent->latent：需要将原图与重建都解码到pixel
                if config.model.get("use_flux_vae", False) or config.model.get("use_sd3_vae", False):
                    reconstructed_images = original_vae_model.decode(reconstructed_images)
                    original_images_for_viz = original_vae_model.decode(original_images_for_viz)

        
    if pretrained_tokenizer is not None:
        reconstructed_images = pretrained_tokenizer.decode(reconstructed_images.argmax(1))
    
    # 准备LQ图像用于可视化（如果存在）
    lq_images_for_viz = None
    if cond_images is not None:
        lq_images_for_viz = torch.clone(cond_images)
        # 如果使用了VAE，需要解码LQ图像用于可视化
        if original_vae_model is not None:
            vae_mode = config.model.get("flux_vae_mode", "latent_to_latent")
            # 在两种模式下都需要解码LQ图像，因为LQ图像总是latent格式
            if config.model.get("use_flux_vae", False):
                lq_images_for_viz = original_vae_model.decode(lq_images_for_viz)
            elif config.model.get("use_sd3_vae", False):
                lq_images_for_viz = original_vae_model.decode(lq_images_for_viz)
    
    images_for_saving, images_for_logging = make_viz_from_samples(
        original_images_for_viz,
        reconstructed_images,
        lq_images_for_viz
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


@torch.no_grad()
def generate_images(model, tokenizer, accelerator, 
                    global_step, output_dir, logger, config=None):
    model.eval()
    tokenizer.eval()
    logger.info("Generating images...")
    generated_image = sample_fn(
        accelerator.unwrap_model(model),
        tokenizer,
        guidance_scale=config.model.generator.get("guidance_scale", 3.0),
        guidance_decay=config.model.generator.get("guidance_decay", "constant"),
        guidance_scale_pow=config.model.generator.get("guidance_scale_pow", 3.0),
        randomize_temperature=config.model.generator.get("randomize_temperature", 2.0),
        softmax_temperature_annealing=config.model.generator.get("softmax_temperature_annealing", False),
        num_sample_steps=config.model.generator.get("num_steps", 8),
        device=accelerator.device,
        return_tensor=True
    )
    images_for_saving, images_for_logging = make_viz_from_samples_generation(
        generated_image)

    # Log images.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {"Train Generated": [images_for_saving]}, step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Generated": images_for_logging}, step=global_step
        )
    # Log locally.
    root = Path(output_dir) / "train_generated_images"
    os.makedirs(root, exist_ok=True)
    filename = f"{global_step:08}_s-generated.png"
    path = os.path.join(root, filename)
    images_for_saving.save(path)

    model.train()
    return


@torch.no_grad()
def t2i_generate_images(model, tokenizer, captions, aes_scores, clip_tokenizer, clip_encoder, accelerator, 
                    global_step, output_dir, logger, config=None, model_type="maskgen_kl"):
    model.eval()
    tokenizer.eval()
    local_model = accelerator.unwrap_model(model)
    logger.info("Generating images...")

    if model_type == "maskgen_vq":
        tokens = local_model.generate(
            captions=captions[:config.training.num_generated_images],
            sample_aesthetic_score=aes_scores[:config.training.num_generated_images] if config.model.maskgen.micro_condition else None,
            num_steps=config.model.maskgen.get("num_iter", 16),
            guidance_scale=config.model.maskgen.cfg,
            guidance_decay=config.model.maskgen.cfg_schedule,
            clip_tokenizer=clip_tokenizer,
            clip_encoder=clip_encoder,
            guidance_decay_scale_pow=config.model.maskgen.cfg_decay_scale_pow,
            randomize_temperature=config.model.maskgen.randomize_temperature,
            softmax_temperature_annealing=config.model.maskgen.get("softmax_temperature_annealing", True),
            prob_sorting=config.model.maskgen.get("prob_sorting", True)
        )
    elif model_type == "maskgen_kl":
        tokens = local_model.sample_tokens(config.training.num_generated_images, 
            clip_tokenizer=clip_tokenizer, clip_encoder=clip_encoder, 
            captions=captions[:config.training.num_generated_images], 
            aes_scores=aes_scores[:config.training.num_generated_images] if config.model.maskgen.micro_condition else None,
            num_iter=config.model.maskgen.num_iter, cfg_schedule=config.model.maskgen.cfg_schedule,
            cfg=config.model.maskgen.cfg, temperature=config.model.maskgen.temperature
        )
    else:
        raise NotImplementedError

    text_guidance = clip_tokenizer(captions[:config.training.num_generated_images]).to(accelerator.device)
    cast_dtype = clip_encoder.transformer.get_cast_dtype()
    text_guidance = clip_encoder.token_embedding(text_guidance).to(cast_dtype)  # [batch_size, n_ctx, d_model]
    text_guidance = text_guidance + clip_encoder.positional_embedding.to(cast_dtype)
    text_guidance = text_guidance.permute(1, 0, 2)  # NLD -> LND
    text_guidance = clip_encoder.transformer(text_guidance, attn_mask=clip_encoder.attn_mask)
    text_guidance = text_guidance.permute(1, 0, 2)  # LND -> NLD
    text_guidance = clip_encoder.ln_final(text_guidance)  # [batch_size, n_ctx, transformer.width]
    
    generated_image = tokenizer.decode_tokens(tokens, text_guidance=text_guidance)

    images_for_saving, images_for_logging = make_viz_from_samples_t2i_generation(generated_image, captions[:config.training.num_generated_images])

    # Log images.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {"Train Generated": [images_for_saving]}, step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Generated": images_for_logging}, step=global_step
        )
    # Log locally.
    root = Path(output_dir) / "train_generated_images"
    os.makedirs(root, exist_ok=True)
    filename = f"{global_step:08}_s-generated.png"
    path = os.path.join(root, filename)
    images_for_saving.save(path)

    model.train()
    return


def save_checkpoint(model, output_dir, accelerator, global_step, logger, config=None, loss_module=None) -> Path:
    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # 所有进程都需要等待，确保目录创建同步
    accelerator.wait_for_everyone()
    
    # 保存accelerator state（所有进程都需要）
    accelerator.save_state(save_path)
    
    # 只在主进程保存模型权重和元数据，然后上传到S3
    if accelerator.is_main_process:
        state_dict = accelerator.get_state_dict(model)
        unwrapped_model = accelerator.unwrap_model(model)
        
        # 如果提供了loss_module，也保存判别器权重
        if loss_module is not None:
            unwrapped_loss_module = accelerator.unwrap_model(loss_module)
            if hasattr(unwrapped_loss_module, 'discriminator'):
                discriminator_state_dict = accelerator.get_state_dict(loss_module)
                # 提取判别器相关的权重
                discriminator_weights = {}
                for key, value in discriminator_state_dict.items():
                    if 'discriminator' in key:
                        discriminator_weights[key] = value
                
                if discriminator_weights:
                    # 将判别器权重合并到主模型的state_dict中
                    state_dict.update(discriminator_weights)
                    logger.info(f"✅ 已保存判别器权重: {len(discriminator_weights)} 个参数")
        
        # 优先使用模型自带的保存方法；否则回退为直接保存state_dict
        if hasattr(unwrapped_model, "save_pretrained_weight"):
            unwrapped_model.save_pretrained_weight(
                save_path / "unwrapped_model",
                save_function=accelerator.save,
                state_dict=state_dict,
            )
        else:
            # 兼容普通nn.Module
            (save_path / "unwrapped_model").mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, save_path / "unwrapped_model" / "pytorch_model.bin")
        
        # 添加LoRA信息到metadata
        metadata = {"global_step": global_step}
        if hasattr(unwrapped_model, "get_lora_info"):
            lora_info = unwrapped_model.get_lora_info()
            metadata["lora_info"] = lora_info
            logger.info(f"📊 LoRA信息: {lora_info}")
        
        json.dump(metadata, (save_path / "metadata.json").open("w+"))
        logger.info(f"✅ 已保存完整checkpoint到本地: {save_path}")
        
        # 现在整个checkpoint目录都保存完整了，开始上传到S3
        # 默认开启S3备份 - 不需要config.experiment.s3.enabled
        # 优先使用配置中的S3设置，如果没有配置则使用默认值
        try:
            # 确定S3配置
            if config and hasattr(config, 'experiment') and hasattr(config.experiment, 's3'):
                # 使用配置中的S3设置
                s3_config = config.experiment.s3
                bucket_name = s3_config.get('bucket_name', 'nextcam-sharing')
                base_path = s3_config.get('base_path', 'xcai/SRTokenizer')
                experiment_name = output_dir.split('/')[-1]
            else:
                # 使用默认S3设置
                bucket_name = "nextcam-sharing"
                base_path = "xcai/SRTokenizer"
                experiment_name = output_dir.split('/')[-1]
            
            # 构建S3路径
            s3_path = f"s3://{bucket_name}/{base_path}/{experiment_name}/checkpoint-{global_step}"
            
            # 优先使用 aws s3 cp 命令
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
                # 回退到Python boto3方式
                try:
                    from utils.s3_utils import upload_checkpoint_to_s3, get_s3_config_from_env
                    
                    s3_config = get_s3_config_from_env()
                    
                    # 如果配置中提供了S3凭证，则使用配置中的值
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
        
        # 兼容旧的配置结构（如果存在）
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
        
        # 清理旧的checkpoint，只保留最新的5个
        try:
            cleanup_old_checkpoints(output_dir, max_checkpoints=5, logger=logger)
        except Exception as e:
            logger.warning(f"⚠️ 清理旧checkpoint时发生错误: {e}")
    
    # 所有进程等待主进程完成保存和上传
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
    import os
    import shutil
    from pathlib import Path
    
    output_path = Path(output_dir)
    if not output_path.exists():
        return
    
    # 查找所有checkpoint目录
    checkpoint_dirs = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                # 提取checkpoint的步数
                step = int(item.name.split("-")[1])
                checkpoint_dirs.append((step, item))
            except (ValueError, IndexError):
                # 如果无法解析步数，跳过这个目录
                continue
    
    # 按步数排序（最新的在前）
    checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
    
    # 如果checkpoint数量超过限制，删除多余的
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

    # Try to load with strict=True first
    try:
        accelerator.load_state(checkpoint_path, strict=strict)
    except RuntimeError as e:
        try:
            accelerator.load_state(checkpoint_path, strict=False)
        except Exception as e2:
            logger.warning(f"Could not load accelerator state: {e2}")
            logger.info("Continuing with model weights only...")
    # Always try to load the metadata to get the global_step
    try:
        with open(checkpoint_path / "metadata.json", "r") as f:
            metadata = json.load(f)
            global_step = int(metadata["global_step"])
            
            # 验证LoRA参数（如果模型支持）
            if hasattr(accelerator.unwrap_model(accelerator._models[0]), "validate_lora_parameters"):
                model = accelerator.unwrap_model(accelerator._models[0])
                is_valid, message = model.validate_lora_parameters()
                if is_valid:
                    logger.info(f"✅ {message}")
                else:
                    logger.warning(f"⚠️ {message}")
                
                # 显示LoRA信息
                if hasattr(model, "get_lora_info"):
                    lora_info = model.get_lora_info()
                    logger.info(f"📊 当前LoRA配置: {lora_info}")
                    
                    # 比较保存的LoRA信息
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
    
    # 计算噪声强度（基于图像的标准差）
    with torch.no_grad():
        # 计算图像的标准差作为噪声强度的参考
        image_std = lq_cond.std()
        noise_scale = strength * image_std
        
        if noise_type == "gaussian":
            # 高斯噪声
            if training:
                # 训练时：随机噪声
                noise = torch.randn_like(lq_cond) * noise_scale
            else:
                # 评估时：固定种子噪声，确保结果可重现
                torch.manual_seed(42)  # 固定种子
                noise = torch.randn_like(lq_cond) * noise_scale
                torch.manual_seed(torch.initial_seed())  # 恢复随机种子
        elif noise_type == "uniform":
            # 均匀分布噪声
            if training:
                # 训练时：随机噪声
                noise = (torch.rand_like(lq_cond) * 2 - 1) * noise_scale
            else:
                # 评估时：固定种子噪声
                torch.manual_seed(42)
                noise = (torch.rand_like(lq_cond) * 2 - 1) * noise_scale
                torch.manual_seed(torch.initial_seed())
        else:
            # 未知噪声类型，返回原始图像
            return lq_cond
        
        # 注入噪声
        noisy_lq_cond = lq_cond + noise
        
        # 可选：限制在合理范围内（0-1或-1到1，取决于图像范围）
        if lq_cond.min() >= 0 and lq_cond.max() <= 1:
            # 图像范围是0-1
            noisy_lq_cond = torch.clamp(noisy_lq_cond, 0, 1)
        elif lq_cond.min() >= -1 and lq_cond.max() <= 1:
            # 图像范围是-1到1
            noisy_lq_cond = torch.clamp(noisy_lq_cond, -1, 1)
    
    return noisy_lq_cond


def monitor_noise_injection_stats(lq_cond_original, lq_cond_noisy, logger, step, config):
    """监控噪声注入的统计信息
    
    Args:
        lq_cond_original: 原始的低质量条件图像
        lq_cond_noisy: 注入噪声后的低质量条件图像
        logger: 日志记录器
        step: 当前训练步数
        config: 配置对象
    """
    if not hasattr(config, 'noise_injection') or not config.noise_injection.enabled:
        return None
    
    if lq_cond_original is None or lq_cond_noisy is None:
        return None
    
    with torch.no_grad():
        # 计算噪声统计信息
        noise = lq_cond_noisy - lq_cond_original
        noise_mean = noise.mean().item()
        noise_std = noise.std().item()
        noise_norm = noise.norm().item()
        
        # 计算PSNR变化
        mse = ((lq_cond_original - lq_cond_noisy) ** 2).mean().item()
        if mse > 0:
            psnr_original = -10 * math.log10(mse)
        else:
            psnr_original = float('inf')
        
        # 计算SSIM变化（简化版本）
        # 这里使用结构相似性的简化计算
        mu_orig = lq_cond_original.mean()
        mu_noisy = lq_cond_noisy.mean()
        sigma_orig = lq_cond_original.std()
        sigma_noisy = lq_cond_noisy.std()
        sigma_cross = ((lq_cond_original - mu_orig) * (lq_cond_noisy - mu_noisy)).mean()
        
        c1 = 0.01 ** 2  # 避免除零
        c2 = 0.03 ** 2
        
        ssim_numerator = (2 * mu_orig * mu_noisy + c1) * (2 * sigma_cross + c2)
        ssim_denominator = (mu_orig ** 2 + mu_noisy ** 2 + c1) * (sigma_orig ** 2 + sigma_noisy ** 2 + c2)
        ssim = ssim_numerator / ssim_denominator
        
        # 记录统计信息
        if step % 100 == 0:  # 每100步记录一次，避免日志过多
            logger.info(f"Step {step}: Noise注入统计:")
            logger.info(f"  - 噪声均值: {noise_mean:.6f}")
            logger.info(f"  - 噪声标准差: {noise_std:.6f}")
            logger.info(f"  - 噪声范数: {noise_norm:.6f}")
            logger.info(f"  - PSNR变化: {psnr_original:.2f} dB")
            logger.info(f"  - SSIM变化: {ssim:.4f}")
            
            # 记录噪声强度配置
            noise_config = config.noise_injection
            logger.info(f"  - 配置强度: {noise_config.strength}")
            logger.info(f"  - 噪声类型: {noise_config.type}")
    
    # 返回统计信息字典，用于日志记录
    stats_dict = {
        "noise_injection/mean": noise_mean,
        "noise_injection/std": noise_std,
        "noise_injection/norm": noise_norm,
        "noise_injection/psnr_change": psnr_original,
        "noise_injection/ssim_change": ssim,
        "noise_injection/strength": config.noise_injection.strength,
    }
    
    return stats_dict


def validate_noise_injection_config(config, logger):
    """验证noise注入配置的有效性
    
    Args:
        config: 配置对象
        logger: 日志记录器
    
    Returns:
        bool: 配置是否有效
    """
    if not hasattr(config, 'noise_injection'):
        return True  # 没有配置也不算错误
    
    noise_config = config.noise_injection
    
    # 检查是否启用
    if not noise_config.enabled:
        return True  # 未启用，配置有效
    
    # 检查强度参数
    if not hasattr(noise_config, 'strength'):
        logger.error("Noise注入配置缺少strength参数")
        return False
    
    strength = noise_config.strength
    if not isinstance(strength, (int, float)) or strength < 0.0 or strength > 1.0:
        logger.error(f"Noise注入强度参数无效: {strength}，应该在[0.0, 1.0]范围内")
        return False
    
    # 检查噪声类型
    if not hasattr(noise_config, 'type'):
        logger.error("Noise注入配置缺少type参数")
        return False
    
    noise_type = noise_config.type
    if noise_type not in ["gaussian", "uniform"]:
        logger.error(f"不支持的噪声类型: {noise_type}，支持的类型: gaussian, uniform")
        return False
    
    # 记录配置信息
    logger.info(f"Noise注入配置验证通过:")
    logger.info(f"  - 启用状态: {noise_config.enabled}")
    logger.info(f"  - 噪声强度: {strength}")
    logger.info(f"  - 噪声类型: {noise_type}")
    
    return True


def use_tensor_downsample_mode(image: torch.Tensor, config, lq_cond_mode: str = "auto") -> torch.Tensor:
    """
    使用tensor下采样模式从输入图像构造lq_cond
    
    Args:
        image: 输入图像 tensor，形状为 (B, C, H, W)
        config: 配置对象
        lq_cond_mode: 下采样模式 ("long", "long_up", "traditional", "auto")
    
    Returns:
        lq_cond: 构造的低质量条件图像
    """
    import torch.nn.functional as F
    
    # 根据 lq_cond_mode 参数决定使用哪种模式
    if lq_cond_mode == "long":
        # 使用 long 模式：将 lq_cond resize 到长边最接近 lq_cond_size 且是 16 倍数的尺寸，不进行上采样
        lq_cond_size = getattr(config.local_eval, 'lq_cond_size', 512)
        return _create_lq_cond_long_mode_tensor(image, lq_cond_size, upsample_cond_full=False)
    elif lq_cond_mode == "long_up":
        # 使用 long_up 模式：将 lq_cond resize 到长边最接近 lq_cond_size 且是 16 倍数的尺寸，然后上采样到全尺寸
        lq_cond_size = getattr(config.local_eval, 'lq_cond_size', 512)
        return _create_lq_cond_long_mode_tensor(image, lq_cond_size, upsample_cond_full=True)
    elif lq_cond_mode == "traditional":
        # 使用传统模式：基于 downsample_factor 和 upsample_factor
        downsample_factor = config.experiment.get('downsample_factor', 2)
        upsample_factor = config.experiment.get('upsample_factor', 2)
        return _create_lq_cond_traditional_mode_tensor(image, downsample_factor, upsample_factor)
    else:  # "auto" 模式，根据配置自动选择
        # 检查是否使用 long 模式
        load_mode = getattr(config.local_eval, 'load_mode', None)
        if load_mode == "long":
            # 使用 long 模式：将 lq_cond resize 到长边最接近 lq_cond_size 且是 16 倍数的尺寸，不进行上采样
            lq_cond_size = getattr(config.local_eval, 'lq_cond_size', 512)
            return _create_lq_cond_long_mode_tensor(image, lq_cond_size, upsample_cond_full=False)
        else:
            # 使用传统模式：基于 downsample_factor 和 upsample_factor
            downsample_factor = config.experiment.get('downsample_factor', 2)
            upsample_factor = config.experiment.get('upsample_factor', 2)
            return _create_lq_cond_traditional_mode_tensor(image, downsample_factor, upsample_factor)


def _create_lq_cond_long_mode_tensor(image: torch.Tensor, lq_cond_size: int, upsample_cond_full: bool = False) -> torch.Tensor:
    """
    使用 long 模式创建 lq_cond（tensor 版本）：将长边 resize 到最接近 lq_cond_size 且是 16 倍数的尺寸
    
    Args:
        image: 输入图像 tensor，形状为 (B, C, H, W)
        lq_cond_size: 目标长边尺寸
        upsample_cond_full: 是否上采样到全尺寸
    
    Returns:
        lq_cond: 构造的低质量条件图像
    """
    import torch.nn.functional as F
    
    batch_size, channels, height, width = image.shape
    
    # 确定长边
    if width >= height:
        # 宽度是长边
        new_w = lq_cond_size
        new_h = int(height * lq_cond_size / width)
    else:
        # 高度是长边
        new_h = lq_cond_size
        new_w = int(width * lq_cond_size / height)
    
    # 应用 16 倍数约束，保持宽高比，向下取整到最近的倍数
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    
    # 确保至少有一个维度不为 0
    if new_w == 0:
        new_w = 16
    if new_h == 0:
        new_h = 16
    
    # 使用 F.interpolate 进行 resize
    lq_cond = F.interpolate(
        image, 
        size=(new_h, new_w), 
        mode='bicubic', 
        align_corners=False
    )
    lq_cond = torch.clamp(lq_cond, 0.0, 1.0)
    
    if upsample_cond_full:
        lq_cond = F.interpolate(
            lq_cond,
            size=(height, width),
            mode='bicubic',
            align_corners=False
        )
        lq_cond = torch.clamp(lq_cond, 0.0, 1.0)
    
    return lq_cond


def _create_lq_cond_traditional_mode_tensor(image: torch.Tensor, downsample_factor: int, upsample_factor: int) -> torch.Tensor:
    """
    使用传统模式创建 lq_cond（tensor 版本）：基于 downsample_factor 和 upsample_factor
    
    Args:
        image: 输入图像 tensor，形状为 (B, C, H, W)
        downsample_factor: 下采样倍率
        upsample_factor: 上采样倍率
    
    Returns:
        lq_cond: 构造的低质量条件图像
    """
    import torch.nn.functional as F
    
    batch_size, channels, height, width = image.shape
    
    # 计算下采样后的尺寸
    cond_height = height // downsample_factor
    cond_width = width // downsample_factor
    
    # 下采样
    lq_cond = F.interpolate(
        image, 
        size=(cond_height, cond_width), 
        mode='bicubic', 
        align_corners=False
    )
    
    # 如果需要进行上采样
    if upsample_factor > 1:
        final_height = cond_height * upsample_factor
        final_width = cond_width * upsample_factor
        lq_cond = F.interpolate(
            lq_cond, 
            size=(final_height, final_width), 
            mode='bicubic', 
            align_corners=False
        )
    
    lq_cond = torch.clamp(lq_cond, 0.0, 1.0)
    return lq_cond
