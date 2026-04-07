"""Training script for DA-VAE (Detail-Aligned VAE).

Trains the DA-VAE tokenizer on top of a pretrained SD3 VAE, adding detail
channels with alignment loss for high-resolution image reconstruction.

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

from accelerate.utils import set_seed
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

import torch
from omegaconf import OmegaConf
from utils.logger import setup_logger
from data.piat_loader import create_train_dataloader, create_val_dataloader
from data.local_image_loader import create_local_val_dataloader
from utils.train_utils import (
    get_config,
    create_model_and_loss_module,
    create_optimizer, create_lr_scheduler,
    create_evaluator, auto_resume, save_checkpoint,
    train_one_epoch,
    reconstruct_images,
)
from modeling.modules.ema_model import EMAModel


def main():
    parser = argparse.ArgumentParser(description="DA-VAE Training")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to config YAML (alternative to TRAIN_CONFIG env var)")
    args, _ = parser.parse_known_args()

    if args.config_path:
        os.environ["TRAIN_CONFIG"] = args.config_path

    workspace = os.environ.get('WORKSPACE', '')
    if workspace:
        torch.hub.set_dir(workspace + "/models/hub")

    config = get_config()

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    tracker = "wandb" if config.training.enable_wandb else "tensorboard"
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision="no" if not config.training.mixed_precision else config.training.mixed_precision,
        log_with=tracker,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )

    logger = setup_logger(
        name="DA-VAE", log_level="INFO",
        output_file=f"{output_dir}/log{accelerator.process_index}.txt")

    if accelerator.is_main_process:
        accelerator.init_trackers(config.experiment.name)
        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    accelerator.wait_for_everyone()

    # Create model, EMA, and loss module
    model_type = config.model.get("tokenizer_type", "sd3_da_vae")
    model, ema_model, loss_module = create_model_and_loss_module(
        config, logger, accelerator, model_type=model_type)

    original_vae_model = model.original_vae_model

    # Create optimizers and schedulers
    optimizer, discriminator_optimizer, dmd_optimizer = create_optimizer(
        config, logger, model, loss_module, model_type=model_type)

    lr_scheduler, discriminator_lr_scheduler, dmd_lr_scheduler = create_lr_scheduler(
        config, logger, accelerator, optimizer, discriminator_optimizer, dmd_optimizer)

    # Create dataloaders
    if hasattr(config, 'local_train') and config.local_train.enabled:
        logger.info(f"Using local training data: {config.local_train.image_dir}")
        train_dataloader = create_train_dataloader(config)
    else:
        train_dataloader = create_train_dataloader(config)

    if hasattr(config, 'local_eval') and config.local_eval.enabled:
        logger.info(f"Using local eval data: {config.local_eval.image_dir}")
        eval_dataloader = create_val_dataloader(config)
    else:
        eval_dataloader = create_val_dataloader(config)

    # Optional CUDA prefetching
    if torch.cuda.is_available():
        try:
            from utils.train_utils import CUDAPrefetcher
            train_dataloader = CUDAPrefetcher(train_dataloader, accelerator.device)
        except Exception:
            pass

    evaluator = create_evaluator(config, logger, accelerator)

    # Prepare with accelerator
    logger.info("Preparing model, optimizer and dataloaders")
    model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler = accelerator.prepare(
        model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler
    )

    if config.training.use_ema:
        ema_model.to(accelerator.device)

    # Compute training schedule
    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    num_batches = math.ceil(config.experiment.max_train_examples / total_batch_size_without_accum)
    num_update_steps_per_epoch = math.ceil(num_batches / config.training.gradient_accumulation_steps)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Instantaneous batch size per gpu = {config.training.per_gpu_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = "
                f"{config.training.per_gpu_batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps}")

    global_step = 0
    first_epoch = 0

    global_step, first_epoch = auto_resume(
        config, logger, accelerator, ema_model, num_update_steps_per_epoch, strict=True)

    if discriminator_optimizer is not None:
        discriminator_optimizer.param_groups[0]['lr'] = config.optimizer.params.discriminator_learning_rate

    for current_epoch in range(first_epoch, num_train_epochs):
        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs-1} started.")
        global_step = train_one_epoch(
            config, logger, accelerator,
            model, ema_model, loss_module,
            optimizer, discriminator_optimizer,
            lr_scheduler, discriminator_lr_scheduler,
            train_dataloader, eval_dataloader,
            evaluator,
            global_step,
            model_type=model_type,
            original_vae_model=original_vae_model
        )
        if global_step >= config.training.max_train_steps:
            accelerator.print(
                f"Finishing training: Global step is >= Max train steps: "
                f"{global_step} >= {config.training.max_train_steps}")
            break

    accelerator.wait_for_everyone()
    save_checkpoint(model, output_dir, accelerator, global_step,
                    logger=logger, config=config, loss_module=loss_module)
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained_weight(output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()
