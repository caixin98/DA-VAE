"""Training script for SD3 Tokenizer.

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

Reference:
    https://github.com/huggingface/open-muse
"""
import math
import os
import argparse
from pathlib import Path
import json

from accelerate.utils import set_seed
from accelerate import Accelerator

import torch
from omegaconf import OmegaConf
from utils.logger import setup_logger
from data.piat_loader import create_train_dataloader, create_val_dataloader
from data.local_image_loader import create_local_val_dataloader
from utils.train_utils import (
    get_config, create_clip_model, 
    create_model_and_loss_module,
    create_optimizer, create_lr_scheduler, create_dataloader,
    create_evaluator, auto_resume, save_checkpoint, 
    train_one_epoch,
    reconstruct_images,
    validate_noise_injection_config,
    use_tensor_downsample_mode,
)
from modeling.modules.ema_model import EMAModel
from accelerate.utils import DistributedDataParallelKwargs

def monitor_encoder_gradients(model, logger, step, model_type="sd3_tokenizer"):
    """监控encoder的梯度状态"""
    if model_type not in ("sd3_tokenizer", "sd3_tokenizer_2d", "sd3_tokenizer_sr"):
        return
    
    # 获取模型（如果是分布式训练）
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model
    
    # 检查是否有encoder
    if not hasattr(actual_model, 'encoder') or actual_model.encoder is None:
        logger.info(f"Step {step}: 跳过encoder训练 (skip_encoder=True)")
        return
    
    # 检查encoder参数
    encoder_params = list(actual_model.encoder.parameters())
    if not encoder_params:
        logger.info(f"Step {step}: Encoder没有可训练参数")
        return
    
    # 检查梯度状态
    total_grad_norm = 0.0
    param_count = 0
    has_grad = False
    
    for param in encoder_params:
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            param_count += 1
    
    if has_grad:
        avg_grad_norm = (total_grad_norm ** 0.5) / max(param_count, 1)
        logger.info(f"Step {step}: Encoder梯度正常 - 参数数: {param_count}, 平均梯度范数: {avg_grad_norm:.6f}")
    else:
        logger.warning(f"Step {step}: ⚠️ Encoder没有梯度！")
    
    # 检查encoder是否处于训练模式
    if actual_model.encoder.training:
        logger.info(f"Step {step}: Encoder处于训练模式")
    else:
        logger.warning(f"Step {step}: ⚠️ Encoder处于评估模式！")


def monitor_decoder_gradients(model, logger, step, model_type="sd3_tokenizer"):
    """监控decoder的梯度状态"""
    if model_type not in ("sd3_tokenizer", "sd3_tokenizer_2d", "sd3_tokenizer_sr"):
        return
    
    # 获取模型（如果是分布式训练）
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model
    
    # 检查是否有decoder
    if not hasattr(actual_model, 'decoder') or actual_model.decoder is None:
        logger.error(f"Step {step}: 模型没有decoder！")
        return
    
    # 检查decoder参数
    decoder_params = list(actual_model.decoder.parameters())
    if not decoder_params:
        logger.info(f"Step {step}: Decoder没有可训练参数")
        return
    
    # 检查梯度状态
    total_grad_norm = 0.0
    param_count = 0
    has_grad = False
    
    for param in decoder_params:
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            param_count += 1
    
    if has_grad:
        avg_grad_norm = (total_grad_norm ** 0.5) / max(param_count, 1)
        logger.info(f"Step {step}: Decoder梯度正常 - 参数数: {param_count}, 平均梯度范数: {avg_grad_norm:.6f}")
    else:
        logger.warning(f"Step {step}: ⚠️ Decoder没有梯度！")
    
    # 检查decoder是否处于训练模式
    if actual_model.decoder.training:
        logger.info(f"Step {step}: Decoder处于训练模式")
    else:
        logger.warning(f"Step {step}: ⚠️ Decoder处于评估模式！")
    
    # 检查decoder类型信息
    if hasattr(actual_model, 'decoder_type'):
        logger.info(f"Step {step}: Decoder类型: {actual_model.decoder_type}")
    
    # 检查是否只微调decoder
    if hasattr(actual_model, 'finetune_decoder'):
        if actual_model.finetune_decoder:
            logger.info(f"Step {step}: 仅微调decoder模式")
        else:
            logger.info(f"Step {step}: 正常训练模式")


def monitor_z_params_stats(model, logger, step, model_type="sd3_tokenizer"):
    """监控z params的统计量"""
    if model_type not in ("sd3_tokenizer", "sd3_tokenizer_2d", "sd3_tokenizer_sr"):
        return None
    
    # 获取模型（如果是分布式训练）
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model
    
    # 检查是否有encoder和image_z_quantized_params
    if not hasattr(actual_model, 'encoder') or actual_model.encoder is None:
        logger.info(f"Step {step}: 跳过z params监控 (没有encoder)")
        return None
    
    if not hasattr(actual_model.encoder, 'image_z_quantized_params'):
        logger.info(f"Step {step}: 跳过z params监控 (没有image_z_quantized_params)")
        return None
    
    # 获取z params
    z_params = actual_model.encoder.image_z_quantized_params
    
    if z_params is None:
        logger.warning(f"Step {step}: ⚠️ image_z_quantized_params为None")
        return None
    
    # 计算统计量
    with torch.no_grad():
        # 基本统计量
        z_mean = z_params.mean().item()
        z_std = z_params.std().item()
        z_min = z_params.min().item()
        z_max = z_params.max().item()
        z_norm = z_params.norm().item()
        
        # 计算非零参数的比例
        z_nonzero = (z_params != 0).float().mean().item()
        
        # # 计算参数分布（分位数）
        # z_25 = torch.quantile(z_params, 0.25).item()
        # z_50 = torch.quantile(z_params, 0.50).item()
        # z_75 = torch.quantile(z_params, 0.75).item()
        
        # 计算参数变化（如果之前有记录）
        if not hasattr(monitor_z_params_stats, 'prev_z_params'):
            monitor_z_params_stats.prev_z_params = z_params.clone()
            z_change = 0.0
        else:
            z_change = (z_params - monitor_z_params_stats.prev_z_params).norm().item()
            monitor_z_params_stats.prev_z_params = z_params.clone()
    
    # 记录统计信息
    logger.info(f"Step {step}: Z Params统计量:")
    logger.info(f"  - 形状: {z_params.shape}")
    logger.info(f"  - 均值: {z_mean:.6f}, 标准差: {z_std:.6f}")
    logger.info(f"  - 范围: [{z_min:.6f}, {z_max:.6f}]")
    logger.info(f"  - 范数: {z_norm:.6f}")
    logger.info(f"  - 非零比例: {z_nonzero:.4f}")
    logger.info(f"  - 参数变化: {z_change:.6f}")
    
    # 检查是否有异常值
    if z_std > 10.0:
        logger.warning(f"Step {step}: ⚠️ Z params标准差过大: {z_std:.6f}")
    if z_norm > 1000.0:
        logger.warning(f"Step {step}: ⚠️ Z params范数过大: {z_norm:.6f}")
    if z_nonzero < 0.1:
        logger.warning(f"Step {step}: ⚠️ Z params非零比例过低: {z_nonzero:.4f}")
    
    # 如果支持，获取更多详细信息
    if hasattr(actual_model, 'get_image_parameter_stats'):
        try:
            detailed_stats = actual_model.get_image_parameter_stats()
            logger.info(f"Step {step}: 详细参数统计: {detailed_stats}")
        except Exception as e:
            logger.warning(f"Step {step}: 获取详细参数统计失败: {e}")
    
    # 返回统计信息字典，用于日志记录
    stats_dict = {
        "z_params/mean": z_mean,
        "z_params/std": z_std,
        "z_params/min": z_min,
        "z_params/max": z_max,
        "z_params/norm": z_norm,
        "z_params/nonzero_ratio": z_nonzero,
        "z_params/change": z_change,
        "z_params/shape": str(z_params.shape),
    }
    
    return stats_dict


def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="SD3 Tokenizer训练脚本")
    parser.add_argument("--local_image_dir", type=str, default=None, help="本地图像目录路径（用于评估）")
    parser.add_argument("--local_eval_samples", type=int, default=100, help="本地评估样本数量")
    parser.add_argument("--local_train_image_dir", type=str, default=None, help="本地训练图像目录路径")
    parser.add_argument("--local_train_samples", type=int, default=None, help="本地训练样本数量")
    # 添加noise注入相关的命令行参数
    parser.add_argument("--enable_noise_injection", action="store_true", help="启用noise注入机制")
    parser.add_argument("--noise_strength", type=float, default=0.1, help="noise注入强度 (0.0-1.0)")
    parser.add_argument("--noise_type", type=str, default="gaussian", choices=["gaussian", "uniform"], help="noise类型")
    # 添加FLUX VAE unpacked模式相关的命令行参数
    args, unknown = parser.parse_known_args()
    
    workspace = os.environ.get('WORKSPACE', '')
    if workspace:
        torch.hub.set_dir(workspace + "/models/hub")

    config = get_config()
    
    # 如果提供了命令行参数，覆盖配置中的设置
    if args.local_train_image_dir:
        if not hasattr(config, 'local_train'):
            config.local_train = {}
        config.local_train.enabled = True
        config.local_train.image_dir = args.local_train_image_dir
        config.local_train.max_samples = args.local_train_samples
    
    # 处理noise注入配置
    if args.enable_noise_injection:
        if not hasattr(config, 'noise_injection'):
            config.noise_injection = {}
        config.noise_injection.enabled = True
        config.noise_injection.strength = args.noise_strength
        config.noise_injection.type = args.noise_type
        logger.info(f"启用noise注入机制 - 强度: {args.noise_strength}, 类型: {args.noise_type}")
    elif not hasattr(config, 'noise_injection'):
        # 如果没有命令行参数，设置默认配置
        config.noise_injection = {
            'enabled': False,
            'strength': 0.1,
            'type': 'gaussian'
        }
    

    
    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    # Whether logging to Wandb or Tensorboard.
    tracker = "tensorboard"
    if config.training.enable_wandb:
        tracker = "wandb"
    # print(f"Mixed precision: {config.training.mixed_precision}")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision="no" if not config.training.mixed_precision else config.training.mixed_precision,
        log_with=tracker,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )

    logger = setup_logger(name="SD3 Tokenizer", log_level="INFO",
     output_file=f"{output_dir}/log{accelerator.process_index}.txt")

    # 验证noise注入配置（在logger创建之后）
    if not validate_noise_injection_config(config, logger):
        logger.error("Noise注入配置验证失败，退出训练")
        return

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(config.experiment.name)
        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)
        
    accelerator.wait_for_everyone()

    model_type = config.model.get("tokenizer_type", "sd3_tokenizer")
    model, ema_model, loss_module = create_model_and_loss_module(
        config, logger, accelerator, model_type=model_type)
    
    original_vae_model = model.original_vae_model
    # # 验证修复结果
    # if accelerator.is_main_process:
    #     is_same_object = model.original_vae_model is loss_module.original_vae_model
    #     print(f"✅ 验证结果：两个original_vae_model是同一个对象: {is_same_object}")
    
    # 新增：如果启用了Flux VAE decoder训练，应用模型冻结
    if hasattr(loss_module, 'train_vae_decoder') and loss_module.train_vae_decoder:
        logger.info("检测到Flux VAE decoder训练模式，应用模型冻结")
        loss_module.apply_model_freeze(model)

    # # 在id_based模式下，需要先获取训练集的id list并更新模型
    # if hasattr(model, 'encoder_mode') and model.encoder_mode == "id_based":
    #     if accelerator.is_main_process:
    #         logger.info("检测到id_based模式，使用预定义的image参数配置")
        
    #     # 等待所有进程同步
    #     accelerator.wait_for_everyone()
        
    #     # 检查配置文件中是否指定了id_list_file
    #     config_id_list_file = config.model.vq_model.get("id_list_file", None)
    #     if config_id_list_file and os.path.exists(config_id_list_file):
    #         if accelerator.is_main_process:
    #             logger.info(f"配置文件中指定了id_list_file: {config_id_list_file}")
    #             logger.info(f"模型已自动加载image_id_list，包含{len(model.get_image_id_list())}个id")
    #     else:
    #         if accelerator.is_main_process:
    #             logger.warning("配置文件中未指定id_list_file，模型将使用空的image_id_list")
    #             logger.info(f"当前模型支持的最大image数量: {config.model.vq_model.get('max_image_count', 10000)}")
        
        # 显示参数统计信息
        if hasattr(model, 'get_image_parameter_stats'):
            stats = model.get_image_parameter_stats()
            logger.info(f"模型参数统计: {stats}")
        
        # 等待所有进程同步
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            logger.info("id_based模式初始化完成")

    optimizer, discriminator_optimizer, dmd_optimizer = create_optimizer(config, logger, model, loss_module, model_type=model_type)

    lr_scheduler, discriminator_lr_scheduler, dmd_lr_scheduler = create_lr_scheduler(
        config, logger, accelerator, optimizer, discriminator_optimizer, dmd_optimizer)

    # 创建训练数据加载器（支持本地图像）

    
    if hasattr(config, 'local_train') and config.local_train.enabled:
        logger.info(f"使用本地图像训练，图像目录: {config.local_train.image_dir}")
        train_dataloader = create_train_dataloader(config)  # 传递下采样倍率参数
        logger.info(f"本地训练样本数: {config.local_train.max_samples if config.local_train.max_samples else '全部'}")
        
        # 显示实际使用的参数
        local_batch_size = getattr(config.local_train, 'batch_size', config.dataloader.train.batch_size)
        local_num_workers = getattr(config.local_train, 'num_workers', config.dataloader.train.num_workers)
        logger.info(f"本地训练参数 - batch_size: {local_batch_size}, num_workers: {local_num_workers}")
    else:
        logger.info("使用S3训练数据加载器")
        train_dataloader = create_train_dataloader(config)
    
    # 创建评估数据加载器
    if hasattr(model, 'encoder_mode') and model.encoder_mode == "id_based":
        # 在id_based模式下，使用训练集的一部分进行评估
        logger.info("id_based模式：使用训练集的一部分进行评估")
        
        # 创建训练集的子集用于评估
        if hasattr(config, 'local_train') and config.local_train.enabled:
            # 使用本地训练配置创建评估数据加载器
            eval_config = config.copy()
            # 设置评估样本数量，默认使用训练集的20%
            eval_samples = getattr(config.local_eval, 'max_samples', 
                                 getattr(config.local_train, 'max_samples', None))
            if eval_samples is None:
                # 如果没有设置eval_samples，则使用训练集的20%
                eval_samples = int(getattr(config.local_train, 'max_samples', 10000) * 0.2)
            
            eval_config.local_train.max_samples = eval_samples
            eval_config.local_train.batch_size = getattr(config.local_train, 'eval_batch_size', 
                                                      config.local_train.batch_size)
            eval_config.local_train.num_workers = getattr(config.local_train, 'eval_num_workers', 
                                                       config.local_train.num_workers)
            
            eval_dataloader = create_train_dataloader(eval_config)
            
            if accelerator.is_main_process:
                logger.info(f"评估时将使用训练集的前{eval_samples}个样本")
                logger.info(f"评估batch_size: {eval_config.local_train.batch_size}")
                logger.info(f"评估num_workers: {eval_config.local_train.num_workers}")
        else:
            # 使用S3训练数据，创建子集用于评估
            eval_config = config.copy()
            # 设置评估样本数量
            eval_samples = getattr(config, 'eval_samples', 1000)
            eval_config.experiment.max_train_examples = eval_samples
            
            eval_dataloader = create_train_dataloader(eval_config)
            
            if accelerator.is_main_process:
                logger.info(f"评估时将使用训练集的前{eval_samples}个样本")
    elif config.local_eval.enabled:
        logger.info(f"使用本地图像评估，图像目录: {config.local_eval.image_dir}")
        eval_dataloader = create_val_dataloader(config)  # 传递下采样倍率参数
        
        # 显示实际使用的参数
        local_eval_batch_size = getattr(config.local_eval, 'batch_size', config.dataloader.val.batch_size)
        local_eval_num_workers = getattr(config.local_eval, 'num_workers', config.dataloader.val.num_workers)
        logger.info(f"本地评估参数 - batch_size: {local_eval_batch_size}, num_workers: {local_eval_num_workers}")
    else:
        logger.info("使用S3验证数据加载器")
        eval_dataloader = create_val_dataloader(config)

    # 用 CUDA 预取包装以重叠 H2D 与计算（仅 CUDA）
    if torch.cuda.is_available():
        try:
            from utils.train_utils import CUDAPrefetcher
            train_dataloader = CUDAPrefetcher(train_dataloader, accelerator.device)
        except Exception as e:
            logger.warning(f"CUDAPrefetcher 未启用: {e}")

    # Set up evaluator.
    evaluator = create_evaluator(config, logger, accelerator)

    # # VAE模型现在从loss_module中获取，避免重复创建
    # vae_model = None
    # if hasattr(loss_module, 'vae_model') and loss_module.vae_model is not None:
    #     vae_model = loss_module.vae_model
    #     logger.info("使用loss_module中的VAE模型")
    # else:
    #     logger.info("未检测到VAE模型，将使用pixel空间训练")
    # Prepare everything with accelerator.
    logger.info("Preparing model, optimizer and dataloaders")
    
    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler = accelerator.prepare(
        model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler
    )
    

    if config.training.use_ema:
        ema_model.to(accelerator.device)


    # === Sanity check: run one evaluation before training ===
    run_sanity_check = getattr(config.training, "run_initial_sanity_check", False)
    if accelerator.is_main_process and run_sanity_check:
        logger.info("Running sanity check evaluation before training...")
        batch = next(iter(train_dataloader))
        
        # 处理cond_images生成（与训练循环中的逻辑一致）
        use_tensor_downsample = config.model.get('use_tensor_downsample', False)
        if use_tensor_downsample:
            # 在tensor层面downsample模式下，从原始images构造cond_images
            # 使用新的use_tensor_downsample_mode函数
            images = batch['image']
            lq_cond_mode = config.model.get('lq_cond_mode', 'auto')
            
            # 保存原始数据类型
            original_dtype = images.dtype
            
            # 进行下采样
            if original_dtype == torch.bfloat16:
                # 临时转换为 float32 进行插值
                cond_images = use_tensor_downsample_mode(
                    images.to(torch.float32), 
                    config, 
                    lq_cond_mode
                )
                # 转换回原始精度
                cond_images = cond_images.to(original_dtype)
            else:
                cond_images = use_tensor_downsample_mode(
                    images, 
                    config, 
                    lq_cond_mode
                )
        else:
            # 传统模式：使用数据加载器提供的cond_images
            cond_images = batch.get('cond_image', None)
        
        if cond_images is not None:
            cond_images = cond_images[:config.training.num_generated_images]
        
        reconstruct_images(
            model,
            batch['image'][:config.training.num_generated_images],
        batch['__key__'][:config.training.num_generated_images],
        accelerator,
        0,  # global_step
        config.experiment.output_dir,
            logger=logger,
            config=config,
            model_type=model_type,
            cond_images=cond_images,
            original_vae_model=original_vae_model
        )
        logger.info("Sanity check evaluation done.")
    elif accelerator.is_main_process:
        logger.info("Skipping sanity check before training (run_initial_sanity_check=False)")
    # === End sanity check ===

    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    num_batches = math.ceil(
        config.experiment.max_train_examples / total_batch_size_without_accum)
    num_update_steps_per_epoch = math.ceil(num_batches / config.training.gradient_accumulation_steps)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # Start training.
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Instantaneous batch size per gpu = { config.training.per_gpu_batch_size}")
    logger.info(f"""  Total train batch size (w. parallel, distributed & accumulation) = {(
        config.training.per_gpu_batch_size *
        accelerator.num_processes *
        config.training.gradient_accumulation_steps)}""")
    if hasattr(config, 'local_train') and config.local_train.enabled:
        logger.info(f"  本地图像训练: 启用，图像目录: {config.local_train.image_dir}")
        logger.info(f"  本地训练样本数: {config.local_train.max_samples if config.local_train.max_samples else '全部'}")
    if config.local_eval.enabled:
        logger.info(f"  本地图像评估: 启用，图像目录: {config.local_eval.image_dir}")
        logger.info(f"  本地评估样本数: {config.local_eval.max_samples}")
    
    # 显示noise注入配置信息
    if config.noise_injection.enabled:
        logger.info(f"  🔊 Noise注入: 启用，强度: {config.noise_injection.strength}, 类型: {config.noise_injection.type}")
    else:
        logger.info(f"  🔇 Noise注入: 禁用")
    

    # 显示encoder跳过信息
    if config.model.vq_model.get("skip_encoder", False):
        logger.info(f"  ⚠️ 跳过encoder训练: 只训练decoder")
    else:
        logger.info(f"  ✅ 正常训练: encoder + decoder")
    global_step = 0
    first_epoch = 0

    global_step, first_epoch = auto_resume(
        config, logger, accelerator, ema_model, num_update_steps_per_epoch,
        strict=True)

    # Reset discriminator learning rate after loading checkpoint
    if discriminator_optimizer is not None:
        old_lr = discriminator_optimizer.param_groups[0]['lr']
        discriminator_optimizer.param_groups[0]['lr'] = config.optimizer.params.discriminator_learning_rate
        logger.info(f"Reset discriminator learning rate from {old_lr} to {discriminator_optimizer.param_groups[0]['lr']}")
        
        # # Also reset discriminator lr scheduler if it exists
        # if discriminator_lr_scheduler is not None:
        #     logger.info("Discriminator lr scheduler exists, but will be skipped due to lr=0")

    for current_epoch in range(first_epoch, num_train_epochs):
        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs-1} started.")
        global_step = train_one_epoch(config, logger, accelerator,
                            model, ema_model, loss_module,
                            optimizer, discriminator_optimizer,
                            lr_scheduler, discriminator_lr_scheduler,
                            train_dataloader, eval_dataloader,
                            evaluator,
                            global_step,
                            model_type=model_type,
                            original_vae_model=original_vae_model
                            )
        # Stop training if max steps is reached.
        if global_step >= config.training.max_train_steps:
            accelerator.print(
                f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
            )
            break

    accelerator.wait_for_everyone()
    # Save checkpoint at the end of training.
    save_checkpoint(model, output_dir, accelerator, global_step, logger=logger, config=config, loss_module=loss_module)
    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained_weight(output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()
