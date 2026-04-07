"""
Sampling Scripts of LightningDiT.

by Maple (Jingfeng Yao) from HUST-VL
"""

import os, math, json, pickle, logging, argparse, yaml, torch
import csv
from time import time, strftime
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torchvision
# local imports
from tokenizer.vavae import VA_VAE
from tokenizer.davae import DA_VAE
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
from local_datasets.img_latent_dataset import ImgLatentDataset

# sample function
def do_sample(train_config, accelerator, ckpt_path=None, cfg_scale=None, model=None, vae=None, demo_sample_mode=False, demo_samples_per_class=100):
    """
    Run sampling.
    """

    folder_name = f"{train_config['model']['model_type'].replace('/', '-')}-ckpt-{ckpt_path.split('/')[-1].split('.')[0]}-{train_config['sample']['sampling_method']}-{train_config['sample']['num_sampling_steps']}".lower()
    if cfg_scale is None:
        cfg_scale = train_config['sample']['cfg_scale']
    cfg_interval_start = train_config['sample']['cfg_interval_start'] if 'cfg_interval_start' in train_config['sample'] else 0
    timestep_shift = train_config['sample']['timestep_shift'] if 'timestep_shift' in train_config['sample'] else 0
    if cfg_scale > 1.0:
        folder_name += f"-interval{cfg_interval_start:.2f}"+f"-cfg{cfg_scale:.2f}"
        folder_name += f"-shift{timestep_shift:.2f}"

    # if demo_sample_mode:
    #     cfg_interval_start = 0
    #     timestep_shift = 0
    #     cfg_scale = 9.0

    sample_folder_dir = os.path.join(train_config['train']['output_dir'], train_config['train']['exp_name'], folder_name)
    if accelerator.process_index == 0:
        if not demo_sample_mode:
            print_with_prefix('Sample_folder_dir=', sample_folder_dir)
        print_with_prefix('ckpt_path=', ckpt_path)
        print_with_prefix('cfg_scale=', cfg_scale)
        print_with_prefix('cfg_interval_start=', cfg_interval_start)
        print_with_prefix('timestep_shift=', timestep_shift)

    existing_samples = 0
    next_index = 0
    fid_target = train_config['sample']['fid_num']
    fid_resize_option = train_config['sample'].get('fid_resize_to', None)
    fid_resize_size = None
    if fid_resize_option is not None:
        if isinstance(fid_resize_option, int):
            fid_resize_size = (fid_resize_option, fid_resize_option)
        else:
            fid_resize_size = tuple(fid_resize_option)
            assert len(fid_resize_size) == 2, "fid_resize_to must be an int or a sequence of length 2"

    if not os.path.exists(sample_folder_dir):
        if accelerator.process_index == 0:
            os.makedirs(sample_folder_dir, exist_ok=True)
    else:
        png_files = sorted([f for f in os.listdir(sample_folder_dir) if f.endswith('.png')])
        existing_samples = len(png_files)
        if not demo_sample_mode and existing_samples >= fid_target:
            if fid_resize_size is not None and accelerator.process_index == 0:
                converted = 0
                for name in png_files:
                    image_path = os.path.join(sample_folder_dir, name)
                    with Image.open(image_path) as img:
                        if img.size != fid_resize_size:
                            img = img.resize(fid_resize_size, resample=Image.LANCZOS)
                            img.save(image_path)
                            converted += 1
                if converted > 0:
                    print_with_prefix(f"Resized {converted} existing samples in place to match fid_resize_to={fid_resize_size}.")
                else:
                    print_with_prefix("Existing samples already match fid_resize_to, skipping sampling.")
            accelerator.wait_for_everyone()
            if accelerator.process_index == 0 and fid_resize_size is None:
                print_with_prefix(f"Found {existing_samples} PNG files in {sample_folder_dir}, skip sampling.")
            return sample_folder_dir
        indexed_png = []
        for name in png_files:
            stem, _ = os.path.splitext(name)
            if stem.isdigit():
                indexed_png.append(int(stem))
        if indexed_png:
            next_index = max(indexed_png) + 1
        else:
            next_index = existing_samples

    torch.backends.cuda.matmul.allow_tf32 = True  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup accelerator:
    device = accelerator.device

    # Setup DDP:
    device = accelerator.device
    seed = train_config['train']['global_seed'] * accelerator.num_processes + accelerator.process_index
    torch.manual_seed(seed)
    # torch.cuda.set_device(device)
    print_with_prefix(f"Starting rank={accelerator.local_process_index}, seed={seed}, world_size={accelerator.num_processes}.")
    rank = accelerator.local_process_index

    # Load model:
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 16
    latent_size = train_config['data']['image_size'] // downsample_ratio

    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    model.load_state_dict(checkpoint)
    model.eval()  # important!
    model.to(device)

    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
    )  # default: velocity;
    sampler = Sampler(transport)
    mode = train_config['sample']['mode']
    if mode == "ODE":
        sample_fn = sampler.sample_ode(
            sampling_method=train_config['sample']['sampling_method'],
            num_steps=train_config['sample']['num_sampling_steps'],
            atol=train_config['sample']['atol'],
            rtol=train_config['sample']['rtol'],
            reverse=train_config['sample']['reverse'],
            timestep_shift=timestep_shift,
        )
    else:
        raise NotImplementedError(f"Sampling mode {mode} is not supported.")
    
    if vae is None:
        vae_cfg = train_config.get("vae", {})
        data_cfg = train_config.get("data", {})
        image_size = int(data_cfg.get("image_size", 256))
        model_name = vae_cfg.get("model_name", "da_f16d32_piat_align_mean")
        variant = str(vae_cfg.get("variant", "")).lower()
        if not variant:
            variant = "davae" if model_name.lower().startswith("dc") else "vavae"
        variant = variant if variant in {"davae", "vavae"} else "davae"

        cfg_path = vae_cfg.get("config") or vae_cfg.get("config_path") or _resolve_vae_config_path(model_name)
        if cfg_path is None:
            raise FileNotFoundError(f"Unable to resolve VAE config path for '{model_name}'")

        if variant == "vavae":
            vae = VA_VAE(cfg_path, img_size=image_size, device=device)
            if accelerator.process_index == 0:
                print_with_prefix(f'Loaded VA-VAE model from {cfg_path}')
        else:
            vae = DA_VAE(cfg_path, img_size=image_size, device=device)
            if accelerator.process_index == 0:
                print_with_prefix(f'Loaded DA-VAE model from {cfg_path}')

    using_cfg = cfg_scale > 1.0
    if using_cfg:
        if accelerator.process_index == 0:
            print_with_prefix('Using cfg:', using_cfg)

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        if accelerator.process_index == 0 and not demo_sample_mode:
            print_with_prefix(f"Saving .png samples at {sample_folder_dir}")
    accelerator.wait_for_everyone()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = train_config['sample']['per_proc_batch_size']
    global_batch_size = n * accelerator.num_processes
    samples_remaining = max(fid_target - existing_samples, 0)
    if not demo_sample_mode and samples_remaining == 0:
        if accelerator.process_index == 0:
            print_with_prefix(f"Found {existing_samples} samples, nothing to generate.")
        return sample_folder_dir

    total_samples = int(math.ceil(samples_remaining / global_batch_size) * global_batch_size)
    if rank == 0 and accelerator.process_index == 0:
        if existing_samples > 0:
            print_with_prefix(f"Found {existing_samples} existing samples, generating {samples_remaining} more to reach {fid_target}.")
        print_with_prefix(f"Sampling will produce {total_samples} images this run (may exceed target to match batch size).")

    assert total_samples % accelerator.num_processes == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // accelerator.num_processes)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    iteration_range = range(iterations)
    if not demo_sample_mode and rank == 0:
        total_iterations = int(math.ceil(fid_target / global_batch_size))
        initial_iterations = existing_samples // global_batch_size
        pbar = tqdm(iteration_range, initial=initial_iterations, total=total_iterations)
    else:
        pbar = iteration_range
    
    if accelerator.process_index == 0:
        print_with_prefix("Using latent normalization")
    dataset = ImgLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
        latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
    )
    latent_mean, latent_std = dataset.get_latent_stats()
    latent_multiplier = train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215
    # move to device
    latent_mean = latent_mean.clone().detach().to(device)
    latent_std = latent_std.clone().detach().to(device)


    max_index_exclusive = next_index + samples_remaining

    if demo_sample_mode:
        # demo_labels = train_config['sample'].get('demo_labels', [975, 3, 207, 387, 388, 88, 979, 279])
        demo_labels = train_config['sample'].get('demo_labels', [864, 865, 23, 8, 309])

        samples_per_label = demo_samples_per_class
        timestamp = strftime("%Y%m%d-%H%M%S")
        demo_root_dir = Path(train_config['train']['output_dir']) / train_config['train']['exp_name'] / f"{folder_name}-demo" / timestamp
        if accelerator.process_index == 0:
            print_with_prefix(f"Demo sampling enabled: {len(demo_labels)} classes, {samples_per_label} samples each.")
            print_with_prefix(f"Saving demo samples to {demo_root_dir}")
            os.makedirs(demo_root_dir, exist_ok=True)
        for label in demo_labels:
            class_dir = demo_root_dir / f"class_{label:04d}"
            if accelerator.process_index == 0:
                class_dir.mkdir(parents=True, exist_ok=True)
                print_with_prefix(f"Sampling class {label} into {class_dir}")

            generated = 0
            while generated < samples_per_label:
                current_batch_size = min(n, samples_per_label - generated)
                z = torch.randn(current_batch_size, model.in_channels, latent_size, latent_size, device=device)
                y = torch.full((current_batch_size,), label, device=device)

                if using_cfg:
                    z = torch.cat([z, z], 0)
                    y_null = torch.full((current_batch_size,), 1000, device=device)
                    y = torch.cat([y, y_null], 0)
                    model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=False, cfg_interval_start=cfg_interval_start)
                    model_fn = model.forward_with_cfg
                else:
                    model_kwargs = dict(y=y)
                    model_fn = model.forward

                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                if using_cfg:
                    samples, _ = samples.chunk(2, dim=0)

                samples = (samples * latent_std) / latent_multiplier + latent_mean
                samples = vae.decode_to_images(samples)

                if accelerator.process_index == 0:
                    for idx_in_batch, sample in enumerate(samples):
                        image_index = generated + idx_in_batch
                        image = Image.fromarray(sample)
                        image.save(class_dir / f"{image_index:04d}.png")

                generated += current_batch_size

        if accelerator.process_index == 0:
            print_with_prefix(f"Demo sampling finished. Files saved to {demo_root_dir}")
            return str(demo_root_dir)
        return None
    else:
        for iteration_idx in pbar:
            # Sample inputs:
            z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
            y = torch.randint(0, train_config['data']['num_classes'], (n,), device=device)
            
            # Setup classifier-free guidance:
            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=cfg_interval_start)
                model_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                model_fn = model.forward

            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            samples = (samples * latent_std) / latent_multiplier + latent_mean
            samples = vae.decode_to_images(samples)

            # Save samples to disk as individual .png files
            base_index = next_index + iteration_idx * global_batch_size
            for local_idx, sample in enumerate(samples):
                index = base_index + local_idx * accelerator.num_processes + accelerator.process_index
                if index >= max_index_exclusive:
                    continue
                image = Image.fromarray(sample)
                if fid_resize_size is not None:
                    image = image.resize(fid_resize_size, resample=Image.LANCZOS)
                image.save(f"{sample_folder_dir}/{index:06d}.png")
            accelerator.wait_for_everyone()

    return sample_folder_dir

# some utils
def print_with_prefix(*messages):
    prefix = f"\033[34m[LightningDiT-Sampling {strftime('%Y-%m-%d %H:%M:%S')}]\033[0m"
    combined_message = ' '.join(map(str, messages))
    print(f"{prefix}: {combined_message}")


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


def record_fid_result(train_config, config_path, sample_folder_dir, fid_value):
    exp_root = Path(train_config['train']['output_dir']) / train_config['train']['exp_name']
    exp_root.mkdir(parents=True, exist_ok=True)

    csv_path = exp_root / "fid_results.csv"
    header = [
        "timestamp",
        "config_path",
        "sample_dir",
        "cfg_scale",
        "cfg_interval_start",
        "timestep_shift",
        "sampling_method",
        "num_sampling_steps",
        "fid",
    ]

    sample_cfg = train_config.get('sample', {})
    row = [
        strftime("%Y-%m-%d %H:%M:%S"),
        os.path.abspath(config_path),
        str(sample_folder_dir) if sample_folder_dir is not None else "",
        sample_cfg.get('cfg_scale', ''),
        sample_cfg.get('cfg_interval_start', ''),
        sample_cfg.get('timestep_shift', ''),
        sample_cfg.get('sampling_method', ''),
        sample_cfg.get('num_sampling_steps', ''),
        f"{fid_value:.6f}",
    ]

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    if sample_folder_dir:
        summary_path = Path(sample_folder_dir) / "fid.txt"
        summary_lines = [
            f"timestamp: {row[0]}",
            f"config_path: {row[1]}",
            f"cfg_scale: {row[3]}",
            f"cfg_interval_start: {row[4]}",
            f"timestep_shift: {row[5]}",
            f"sampling_method: {row[6]}",
            f"num_sampling_steps: {row[7]}",
            f"fid: {row[8]}",
        ]
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return csv_path


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":

    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lightningdit_xl_davae_f32d128_align_mean.yaml')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--demo-samples-per-class', type=int, default=10)
    args = parser.parse_args()
    accelerator = Accelerator()
    train_config = load_config(args.config)

    # get ckpt_dir
    assert 'ckpt_path' in train_config, "ckpt_path must be specified in config"
    if accelerator.process_index == 0:
        print_with_prefix('Using ckpt:', train_config['ckpt_path'])
    ckpt_dir = train_config['ckpt_path']

    if 'downsample_ratio' in train_config['vae']:
        latent_size = train_config['data']['image_size'] // train_config['vae']['downsample_ratio']
    else:
        latent_size = train_config['data']['image_size'] // 16
    print_with_prefix('latent_size=', latent_size)
    # get model
    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        wo_shift=train_config['model']['wo_shift'] if 'wo_shift' in train_config['model'] else False,
        in_channels=train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4,
        learn_sigma=train_config['model']['learn_sigma'] if 'learn_sigma' in train_config['model'] else False,
    )

    # naive sample
    sample_folder_dir = do_sample(
        train_config,
        accelerator,
        ckpt_path=ckpt_dir,
        model=model,
        demo_sample_mode=args.demo,
        demo_samples_per_class=args.demo_samples_per_class,
    )
    
    if not args.demo:
        # calculate FID
        # Important: FID is only for reference, please use ADM evaluation for paper reporting
        if accelerator.process_index == 0:
            from tools.calculate_fid import calculate_fid_given_paths
            print_with_prefix('Calculating FID with {} number of samples'.format(train_config['sample']['fid_num']))
            assert 'fid_reference_file' in train_config['data'], "fid_reference_file must be specified in config"
            fid_reference_file = train_config['data']['fid_reference_file']
            fid = calculate_fid_given_paths(
                [fid_reference_file, sample_folder_dir],
                batch_size=50,
                dims=2048,
                device='cuda',
                num_workers=8,
                sp_len = train_config['sample']['fid_num']
            )
            print_with_prefix('fid=',fid)
            csv_path = record_fid_result(train_config, args.config, sample_folder_dir, fid)
            if sample_folder_dir:
                summary_path = Path(sample_folder_dir) / "fid.txt"
                print_with_prefix('Saved FID metrics to', csv_path, 'and', summary_path)
            else:
                print_with_prefix('Saved FID metrics to', csv_path)
