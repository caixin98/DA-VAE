import os
import time
from typing import Dict

import torch

from .tokenizer_trainer import SD3TokenizerModel, train
from ..train_sd3_text_token_image.hf_loader import create_hf_text2image_dataloader


def get_config() -> Dict:
    """
    读取 OMINI_CONFIG 指向的 YAML 配置。
    这里做成轻量实现，避免仅为读取配置就强依赖 datasets/pandas/lightning 等训练依赖。
    """
    config_path = os.environ.get("OMINI_CONFIG")
    assert config_path is not None, "Please set the OMINI_CONFIG environment variable"

    # 优先使用 ruamel.yaml（更稳），否则回退到 PyYAML
    try:
        from ruamel.yaml import YAML  # type: ignore
        yaml_parser = YAML(typ="safe")
        yaml_parser.preserve_quotes = True
        with open(config_path, "r") as f:
            config = yaml_parser.load(f)
    except Exception:
        import yaml  # type: ignore
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    return config


class SimpleConfig:
    """简单的配置适配器，将 dict 转换为可点访问的对象。"""

    def __init__(self, config_dict):
        for key, value in (config_dict or {}).items():
            if isinstance(value, dict):
                setattr(self, key, SimpleConfig(value))
            else:
                setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)


def main():
    config = get_config()
    training_config = config["train"]
    data_config = config["data"]

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    # Optional: override a single fixed prompt for overfitting via env MODEL_FIXED_PROMPT
    fixed_prompt = os.environ.get("MODEL_FIXED_PROMPT")
    if fixed_prompt:
        model_conf = config.get("model", {})
        model_conf["fixed_train_prompt"] = fixed_prompt
        data_val_prompts = [fixed_prompt]
        data_conf = config.get("data", {})
        data_conf["validation_prompts"] = data_val_prompts
        config["model"] = model_conf
        config["data"] = data_conf

    # Prefer tokenizer_* keys; fallback to legacy hr_* if present
    patch_embed_cfg = (
        config.get("tokenizer_patch_embed")
        or config.get("hr_patch_embed", {})
        or {}
    )
    patch_weights = config.get("tokenizer_patch_weights") or config.get("hr_patch_weights")

    model_config = config.get("model", {})

    model = SD3TokenizerModel(
        sd3_pipe_id=config["sd3_pipe_id"],
        patch_embed_cfg=patch_embed_cfg,
        patch_embed_weights_path=patch_weights,
        lora_paths=config.get("lora_paths"),
        lora_config=training_config.get("lora_config"),
        device="cuda",
        dtype=getattr(torch, config.get("dtype", "float32")),
        optimizer_config=training_config.get("optimizer"),
        model_config=model_config,
    )

    hf_conf = data_config.get("hf_dataset")
    if hf_conf:
        train_loader = create_hf_text2image_dataloader(
            dataset_name=hf_conf.get("name"),
            split=hf_conf.get("split", "train"),
            image_key=hf_conf.get("image_key", "image"),
            caption_key=hf_conf.get("caption_key", "text"),
            batch_size=data_config.get("batch_size", training_config.get("batch_size", 1)),
            num_workers=data_config.get("num_workers", training_config.get("num_workers", 0)),
            image_size=data_config.get("base_size", 512),
            shuffle=hf_conf.get("shuffle", True),
            streaming=hf_conf.get("streaming", False),
            max_samples=hf_conf.get("max_samples"),
        )
    else:
        lt = data_config.get("local_train", {}) or {}
        image_dir = str(lt.get("image_dir", "datasets/overfit-one/images"))
        overfit_env = str(os.environ.get("OVERFIT_ONE", "")).lower() in {"1", "true", "yes"}
        overfit_cfg = lt.get("max_samples", None) == 1
        use_overfit = overfit_env or overfit_cfg

        if use_overfit:
            image_path = os.path.join(image_dir, "sample.png")
            from PIL import Image
            from torchvision import transforms as T
            target_size = int(data_config.get("base_size", 1024))
            pil = Image.open(image_path).convert("RGB")
            tfm = T.Compose([
                T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(target_size),
                T.ToTensor(),
            ])
            image_tensor = tfm(pil).unsqueeze(0)
            fixed_prompt = os.environ.get("MODEL_FIXED_PROMPT") or model_config.get("fixed_train_prompt", "")
            batch = {"image": image_tensor, "strText": [fixed_prompt]}
            train_loader = batch
            print(f"[TOK][Data] Using single-sample batch from {image_path} (size={target_size})")
        else:
            from data.local_image_loader import create_local_train_dataloader
            train_loader = create_local_train_dataloader(
                image_dir=image_dir,
                batch_size=int(lt.get("batch_size", 1)),
                num_workers=int(lt.get("num_workers", 0)),
                max_samples=lt.get("max_samples"),
                seed=42,
                downsample_factor=int(lt.get("downsample_factor", 2)),
            )
            print(f"[TOK][Data] Using local image dataloader from {image_dir}")

    # Check for resume checkpoint
    resume_from_checkpoint = training_config.get("resume_from_checkpoint")
    resume_from_lora_checkpoint = training_config.get("resume_from_lora_checkpoint")
    
    if resume_from_checkpoint:
        print(f"[TOK][Resume] Resume checkpoint specified: {resume_from_checkpoint}")
    if resume_from_lora_checkpoint:
        print(f"[TOK][Resume] Resume LoRA checkpoint specified: {resume_from_lora_checkpoint}")
    
    train(dataloader_or_dataset=train_loader, trainable_model=model, config=config, 
          resume_from_checkpoint=resume_from_checkpoint, 
          resume_from_lora_checkpoint=resume_from_lora_checkpoint)


if __name__ == "__main__":
    main()


