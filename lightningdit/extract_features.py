import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import argparse
import os
from safetensors.torch import save_file
from omegaconf import OmegaConf
from datetime import datetime
from local_datasets.img_latent_dataset import ImgLatentDataset
from tokenizer.vavae import VA_VAE
from tokenizer.davae import DA_VAE


def _detect_vae_variant(config_path: str) -> str:
    """
    Detect whether the configuration describes DA-VAE or VA-VAE.
    Fallback to 'davae' when uncertain because DA-VAE is the CVPR2026 target.
    """
    if not config_path:
        return "davae"

    basename = os.path.basename(str(config_path)).lower()
    if "dc" in basename:
        return "davae"
    if "vavae" in basename or "vae" in basename:
        return "vavae"

    try:
        cfg = OmegaConf.load(config_path)
        model_cfg = cfg.get("model", {})
        target = model_cfg.get("target", "")
        if isinstance(target, str) and "da_autoencoder.DAVAE" in target:
            return "davae"
    except Exception:
        pass

    return "davae"


def main(args):
    """
    Run a tokenizer on full dataset and save the features.
    """
    assert torch.cuda.is_available(), "Extract features currently requires at least one GPU."

    # Setup DDP:
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        seed = args.seed + rank
        if rank == 0:
            print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    except:
        print("Failed to initialize DDP. Running in local mode.")
        rank = 0
        device = 0
        world_size = 1
        seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Create model (auto-detect if requested):
    variant = args.vae
    if variant == "auto":
        variant = _detect_vae_variant(args.config)

    default_run_name = os.path.splitext(os.path.basename(args.config))[0] if args.config else variant
    run_name = default_run_name

    if variant == "davae":
        tokenizer = DA_VAE(args.config, img_size=args.image_size)
    else:
        tokenizer = VA_VAE(args.config, img_size=args.image_size)

    # Setup feature folders:
    output_dir = os.path.join(args.output_path, run_name, f'{args.data_split}_{args.image_size}')
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # Setup data:
    datasets = [
        ImageFolder(args.data_path, transform=tokenizer.img_transform(p_hflip=0.0)),
        ImageFolder(args.data_path, transform=tokenizer.img_transform(p_hflip=1.0))
    ]
    samplers = [
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=args.seed
        ) for dataset in datasets
    ]
    loaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        ) for dataset, sampler in zip(datasets, samplers)
    ]
    total_data_in_loop = len(loaders[0].dataset)
    if rank == 0:
        print(f"Total data in one loop: {total_data_in_loop}")

    run_images = 0
    saved_files = 0
    latents = []
    latents_flip = []
    labels = []
    for batch_idx, batch_data in enumerate(zip(*loaders)):
        run_images += batch_data[0][0].shape[0]
        if run_images % 100 == 0 and rank == 0:
            print(f'{datetime.now()} processing {run_images} of {total_data_in_loop} images')
        
        for loader_idx, data in enumerate(batch_data):
            x = data[0]
            y = data[1]  # (N,)
            
            z = tokenizer.encode_images(x).detach().cpu()  # (N, C, H, W)
            # print('z', z.shape, 'dtype', z.dtype)
            if batch_idx == 0 and rank == 0:
                print('latent shape', z.shape, 'dtype', z.dtype)
            
            if loader_idx == 0:
                latents.append(z)
                labels.append(y)
            else:
                latents_flip.append(z)

        if len(latents) == 10000 // args.batch_size:
            latents = torch.cat(latents, dim=0).contiguous()
            latents_flip = torch.cat(latents_flip, dim=0).contiguous()
            labels = torch.cat(labels, dim=0).contiguous()
            save_dict = {
                'latents': latents,
                'latents_flip': latents_flip,
                'labels': labels
            }
            for key in save_dict:
                if rank == 0:
                    print(key, save_dict[key].shape)
            save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
            save_file(
                save_dict,
                save_filename,
                metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
            )
            if rank == 0:
                print(f'Saved {save_filename}')
            
            latents = []
            latents_flip = []
            labels = []
            saved_files += 1

    # save remainder latents that are fewer than 10000 images
    if len(latents) > 0:
        latents = torch.cat(latents, dim=0).contiguous()
        latents_flip = torch.cat(latents_flip, dim=0).contiguous()
        labels = torch.cat(labels, dim=0).contiguous()
        save_dict = {
            'latents': latents,
            'latents_flip': latents_flip,
            'labels': labels
        }
        for key in save_dict:
            if rank == 0:
                print(key, save_dict[key].shape)
        save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
        save_file(
            save_dict,
            save_filename,
            metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
        )
        if rank == 0:
            print(f'Saved {save_filename}')

    # Calculate latents stats
    dist.barrier()
    if rank == 0:
        # Remove previously cached latent normalization stats to force recomputation
        stats_cache = os.path.join(output_dir, "latents_stats.pt")
        if os.path.exists(stats_cache):
            try:
                os.remove(stats_cache)
                print(f"Removed existing latent norm stats: {stats_cache}")
            except Exception as e:
                print(f"Warning: failed to remove latent norm stats at {stats_cache}: {e}")
        dataset = ImgLatentDataset(output_dir, latent_norm=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/path/to/imagenet/ILSVRC2012_train/data')
    parser.add_argument("--data_split", type=str, default='imagenet_train')
    parser.add_argument("--output_path", type=str, default="/path/to/imagenet/latents")
    parser.add_argument("--config", type=str, default="config_details.yaml")
    parser.add_argument("--vae", type=str, default="auto", choices=["auto", "vavae", "davae"], help="Select VAE variant; 'auto' infers from config")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()
    main(args)