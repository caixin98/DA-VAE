import os
import time
from typing import Dict

import torch

from ..config import SimpleConfig, get_config
from ..train_sd3_hr.trainer import SD3HRModel, train
from ..train_sd3_text_token_image.hf_loader import create_hf_text2image_dataloader


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
        # also use the same for validation sampling
        data_val_prompts = [fixed_prompt]
        data_conf = config.get("data", {})
        data_conf["validation_prompts"] = data_val_prompts
        config["model"] = model_conf
        config["data"] = data_conf

    patch_embed_cfg = config.get("hr_patch_embed", {})
    patch_weights = config.get("hr_patch_weights")

    model_config = config.get("model", {})

    model = SD3HRModel(
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

    # Choose dataloader based on config: prefer HF or PIAT; only use single-sample overfit when explicitly requested
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
        # Overfit mode is enabled only when explicitly requested via env flag or config
        overfit_env = str(os.environ.get("OVERFIT_ONE", "")).lower() in {"1", "true", "yes"}
        overfit_cfg = lt.get("max_samples", None) == 1
        use_overfit = overfit_env or overfit_cfg

        if use_overfit:
            # Minimal no-dataloader path for 1-sample overfit: load a single image and build a raw batch dict
            image_path = os.path.join(image_dir, "sample.png")
            try:
                from PIL import Image
                from torchvision import transforms as T
                target_size = int(data_config.get("base_size", 1024))
                pil = Image.open(image_path).convert("RGB")
                # center-crop/resize to target
                tfm = T.Compose([
                    T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(target_size),
                    T.ToTensor(),
                ])
                image_tensor = tfm(pil).unsqueeze(0)  # [1, C, H, W]
                # fixed prompt from env or config
                fixed_prompt = os.environ.get("MODEL_FIXED_PROMPT") or model_config.get("fixed_train_prompt", "")
                batch = {"image": image_tensor, "strText": [fixed_prompt]}
                train_loader = batch  # pass raw batch; trainer wraps it as infinite iterable
                print(f"[HR][Data] Using single-sample batch from {image_path} (size={target_size})")
            except Exception as e:
                print(f"[HR][Data] Single-sample path failed ({e}); falling back to local simple dataloader")
                try:
                    from data.piat_loader import create_local_simple_dataloader  # type: ignore
                    train_loader = create_local_simple_dataloader(
                        image_folder_path=image_dir,
                        target_size=int(data_config.get("base_size", 1024)),
                        batch_size=1,
                        num_workers=0,
                        max_samples=1,
                        shuffle=True,
                        load_text=False,
                    )
                except Exception:
                    from data.piat_loader import create_train_dataloader  # type: ignore
                    train_loader = create_train_dataloader(SimpleConfig(data_config))
        else:
            # Standard local training: use PIAT dataloader
            # try:
            from data.piat_loader import create_train_dataloader  # type: ignore
            train_loader = create_train_dataloader(SimpleConfig(data_config))
            print("[HR][Data] Using PIAT train dataloader for local training")
            # except Exception as e:
            #     print(f"[HR][Data] PIAT dataloader unavailable ({e}); falling back to local simple dataloader")
            #     try:
            #         from data.piat_loader import create_local_simple_dataloader  # type: ignore
            #         train_loader = create_local_simple_dataloader(
            #             image_folder_path=image_dir,
            #             target_size=int(data_config.get("base_size", 1024)),
            #             batch_size=int(lt.get("batch_size", 1)),
            #             num_workers=int(lt.get("num_workers", 0)),
            #             max_samples=lt.get("max_samples"),
            #             shuffle=True,
            #             load_text=bool(lt.get("load_text", False)),
            #         )
            #     except Exception as e2:
            #         raise RuntimeError(
            #             f"Failed to construct any local dataloader. PIAT error: {e}; simple loader error: {e2}"
            #         )

    # Check for resume checkpoint
    resume_from_checkpoint = training_config.get("resume_from_checkpoint")
    resume_from_lora_checkpoint = training_config.get("resume_from_lora_checkpoint")
    
    if resume_from_checkpoint:
        print(f"[HR][Resume] Resume checkpoint specified: {resume_from_checkpoint}")
    if resume_from_lora_checkpoint:
        print(f"[HR][Resume] Resume LoRA checkpoint specified: {resume_from_lora_checkpoint}")
    
    # run training loop (pass DataLoader directly to preserve collate_fn and batch keys)
    train(dataloader_or_dataset=train_loader, trainable_model=model, config=config, 
          resume_from_checkpoint=resume_from_checkpoint, 
          resume_from_lora_checkpoint=resume_from_lora_checkpoint)


if __name__ == "__main__":
    main()


