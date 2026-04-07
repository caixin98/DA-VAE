"""
Training Scripts of LDM & VA-VAE.

We mainly follow the training script of LDM. <https://github.com/CompVis/latent-diffusion>
Only slight modifications are applied to the meet the requirements of torch 2.x and 
pytorch-lightning 2.x. Thanks for their great code!

Main codes related to VA-VAE are:
- ldm/models/foundation_models.py
- ldm/models/autoencoder.py
- ldm/modules/losses/contperceptual.py

by Jingfeng Yao
from HUST-VL
"""
from diffusers import AutoencoderDC
import re


import argparse, os, sys, datetime, glob, importlib, csv
import subprocess
import shutil
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

# -----------------------------------------------------------------------------
# Make repo-root modules importable under torchrun/srun.
#
# In many cluster launchers, the working directory and/or PYTHONPATH may not be
# preserved across ranks. This project keeps `local_datasets/` at the repo root,
# while this script lives under `davae/`, so we explicitly add the repo root to
# `sys.path` for stable imports (e.g. `local_datasets.ms_data`).
# -----------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TAMING_TRANSFORMERS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "taming-transformers"))
if _TAMING_TRANSFORMERS_ROOT not in sys.path:
    # IMPORTANT: do not prepend. This directory contains a top-level `main.py`
    # which would shadow this training entrypoint module name (`main`).
    sys.path.append(_TAMING_TRANSFORMERS_ROOT)

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

# S3 mirror helper
def _sync_path_to_s3(local_path: str):
    """
    Mirror a local directory to S3 using a base-path mapping controlled by env vars.
    Environment:
      S3_MIRROR_LOCAL_BASE (default: script parent directory)
      S3_MIRROR_S3_BASE    (default: )
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
            _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if _repo_root not in candidate_bases:
                candidate_bases.append(_repo_root)
        except Exception:
            pass
        s3_base = os.environ.get('S3_MIRROR_S3_BASE', '').rstrip('/')
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

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "--no-resume",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable auto-resume from latest checkpoint",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs at provided absolute/relative paths (no '/output' prefixing)
            os.makedirs(self.logdir, exist_ok=True)
            print(f"logdir exists: {self.logdir}")
            os.makedirs(self.ckptdir, exist_ok=True)
            print(f"ckptdir exists: {self.ckptdir}")
            os.makedirs(self.cfgdir, exist_ok=True)
            print(f"cfgdir exists: {self.cfgdir}")

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            OmegaConf.save(self.config,
                            os.path.join(self.cfgdir, "project.yaml"))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "lightning.yaml"))

        # else:
        #     # ModelCheckpoint callback created log directory --- remove it
        #     if not self.resume and os.path.exists(self.logdir):
        #         dst, name = os.path.split(self.logdir)
        #         dst = os.path.join(dst, "child_runs", name)
        #         os.makedirs(os.path.split(dst)[0], exist_ok=True)
        #         try:
        #             os.rename(self.logdir, dst)
        #         except FileNotFoundError:
        #             pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            # pl.loggers.TestTubeLogger: self._testtube,
            pl.loggers.TensorBoardLogger: self._testtube,
            # Support Wandb image logging
            pl.loggers.WandbLogger: self._wandb,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def _tensorboard(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        try:
            import wandb  # lazy import to avoid hard dependency when not used
        except Exception:
            return
        run = getattr(pl_module.logger, "experiment", None)
        if run is None:
            return
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            # map from [-1, 1] to [0, 1]
            grid = (grid + 1.0) / 2.0
            tag = f"{split}/{k}"
            # log a single image per key as a grid
            try:
                run.log({tag: [wandb.Image(grid, caption=tag)]}, step=pl_module.global_step)
            except Exception:
                # best-effort: skip on failure to avoid breaking training
                pass

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            # Make PIL-compatible uint8 HWC array.
            # NOTE: Under bf16-mixed training, `grid` may be bfloat16. Converting such tensors
            # to numpy can yield dtype=object, which PIL cannot handle. Force float32 first.
            grid = grid.detach().cpu()
            if grid.dtype != torch.float32:
                grid = grid.float()
            grid = grid.permute(1, 2, 0)  # HWC
            grid = (grid * 255.0).clamp(0.0, 255.0).to(torch.uint8).numpy()
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            # Pop only when consuming a scheduled step to avoid IndexError noise
            if check_idx in self.log_steps and self.log_steps:
                self.log_steps.pop(0)
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            # rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            # rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


class S3SyncCallback(Callback):
    def __init__(self, paths=None, every_n_steps=500):
        super().__init__()
        self.paths = paths or []
        self.every_n_steps = every_n_steps

    @rank_zero_only
    def on_pretrain_routine_start(self, trainer, pl_module):
        try:
            base_logdir = getattr(trainer, 'logdir', None)
            if base_logdir:
                _sync_path_to_s3(base_logdir)
        except Exception:
            pass

    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        try:
            base_logdir = getattr(trainer, 'logdir', None)
            if base_logdir:
                _sync_path_to_s3(base_logdir)
                _sync_path_to_s3(os.path.join(base_logdir, 'checkpoints'))
        except Exception:
            pass

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        try:
            if self.every_n_steps and trainer.global_step > 0 and trainer.global_step % self.every_n_steps == 0:
                for p in self.paths:
                    _sync_path_to_s3(p)
        except Exception:
            pass

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        try:
            base_logdir = getattr(trainer, 'logdir', None)
            if base_logdir:
                _sync_path_to_s3(base_logdir)
        except Exception:
            pass

if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    # parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    cfg_fname = os.path.split(opt.base[0])[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    name = "_" + cfg_name
    nowname = now + name + opt.postfix
    # logdir = os.path.join(opt.logdir, nowname)
    # Resolve repo root and make logdir absolute under it (if not already absolute)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    _raw_logdir = os.path.join(opt.logdir, cfg_name)
    logdir = _raw_logdir if os.path.isabs(_raw_logdir) else os.path.abspath(os.path.join(repo_root, _raw_logdir))

    # auto resume from the latest checkpoint
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)
    print(f"logdir: {logdir}")

    if opt.no_resume:
        print("Auto-resume disabled by --no-resume; starting from scratch.")
        ckpt = None
    else:
        print(f"Try to resume from {logdir}")
        ckpt_files = glob.glob(os.path.join(logdir, "checkpoints", "epoch=*.ckpt"))
        if not ckpt_files:
            print(f"Warning: No checkpoint files found in {os.path.join(logdir, 'checkpoints')}, training from scratch")
            ckpt = None
        else:
            def _epoch_key(path):
                base = os.path.basename(path)
                m = re.search(r"epoch=(\d+)", base)
                if m:
                    try:
                        return int(m.group(1))
                    except Exception:
                        pass
                # Fallback: order by modification time if epoch not parseable
                return os.path.getmtime(path)

            ckpt_files.sort(key=_epoch_key)
            ckpt = ckpt_files[-1]
    opt.resume_from_checkpoint = ckpt

    trainer = None

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_cfg = lightning_config.get("trainer", OmegaConf.create())
        trainer_container = OmegaConf.to_container(trainer_cfg, resolve=True) or {}
        trainer_dict = dict(trainer_container)
        # Allow specifying ckpt_path in config for resume; don't pass it into Trainer constructor
        ckpt_path_from_cfg = trainer_dict.pop("ckpt_path", None)
        if "accelerator" not in trainer_dict:
            trainer_dict["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"
        if "devices" not in trainer_dict:
            trainer_dict["devices"] = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if "num_nodes" not in trainer_dict:
            trainer_dict["num_nodes"] = 1
        if "strategy" not in trainer_dict:
            trainer_dict["strategy"] = None
        if "max_epochs" not in trainer_dict:
            trainer_dict["max_epochs"] = None
        if "precision" not in trainer_dict:
            trainer_dict["precision"] = 32
        trainer_opt = argparse.Namespace(**trainer_dict)
        lightning_config.trainer = OmegaConf.create(trainer_dict)

        # Honor top-level ckpt_path in config for resume (if provided)
        try:
            if "ckpt_path" in config and str(config.ckpt_path).strip():
                top_ckpt_path = str(config.ckpt_path).strip()
                if os.path.exists(top_ckpt_path):
                    print(f"Resume from top-level ckpt_path in config: {top_ckpt_path}")
                else:
                    print(f"Warning: top-level ckpt_path not found: {top_ckpt_path}")
                opt.resume_from_checkpoint = top_ckpt_path
        except Exception:
            pass

        # model
        model = instantiate_from_config(config.model)

        # if config.init_weight is not None, load the weights
        try:
            print(f"Loading initial weights from {config.init_weight}")
            model.load_state_dict(torch.load(config.init_weight)['state_dict'], strict=False)
        except:
            print(f"There is no initial weights to load.")

        # trainer and callbacks
        trainer_kwargs = dict()
        # Ensure Trainer has a concrete root dir for logs/checkpoints on all ranks
        # to avoid distributed broadcast issues with logger-derived paths
        trainer_kwargs["default_root_dir"] = logdir

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
            "tensorboard": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "name": "tensorboard",
                    "save_dir": logdir,
                }
            }
        }
        default_logger_cfg = default_logger_cfgs["wandb"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)

        # Backward-compatible W&B config:
        # Some configs specify `project/entity/version` at `lightning.logger` top-level (instead of under `params`).
        # WandbLogger only receives kwargs from `params`, so we mirror them into `params` if needed.
        try:
            if "params" not in logger_cfg or logger_cfg.params is None:
                logger_cfg.params = OmegaConf.create()
            for k in ("project", "entity", "version", "group", "tags", "notes"):
                if k in logger_cfg and k not in logger_cfg.params:
                    logger_cfg.params[k] = logger_cfg[k]
        except Exception:
            # Never fail training due to logger config massaging
            pass

        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = -1

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
            "s3_sync": {
                "target": "main.S3SyncCallback",
                "params": {
                    "paths": [logdir, ckptdir, cfgdir],
                    "every_n_steps": 500
                }
            },
        }
        # Disable S3 sync by default if awscli is unavailable (avoid noisy exceptions).
        if shutil.which("aws") is None:
            default_callbacks_cfg.pop("s3_sync", None)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 10000,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        # if opt.resume_from_checkpoint:
        #     trainer_kwargs["resume_from_checkpoint"] = opt.resume_from_checkpoint
        #     print(f"Resuming from checkpoint: {opt.resume_from_checkpoint}")

        # Ensure Lightning's distributed topology matches torch elastic environment variables
        env_num_nodes_override = None
        world_size_env_value = os.environ.get("WORLD_SIZE")
        world_size_int = None
        if world_size_env_value is not None:
            try:
                world_size_int = int(world_size_env_value)
            except ValueError:
                world_size_int = None

        env_num_nodes_from_env = False
        for _env_key in ("NNODES", "NUM_NODES"):
            _env_val = os.environ.get(_env_key)
            if _env_val is not None:
                try:
                    env_num_nodes_override = int(_env_val)
                    env_num_nodes_from_env = True
                    break
                except ValueError:
                    pass
        if env_num_nodes_override is None:
            env_num_nodes_override = getattr(trainer_opt, "num_nodes", None) or 1
        if env_num_nodes_override <= 0:
            env_num_nodes_override = 1

        env_devices_override = None
        for _env_key in ("GPUS_PER_NODE", "LOCAL_WORLD_SIZE", "NPROC_PER_NODE"):
            _env_val = os.environ.get(_env_key)
            if _env_val is not None:
                try:
                    env_devices_override = int(_env_val)
                    break
                except ValueError:
                    pass
        if env_devices_override is None and world_size_int is not None:
            if env_num_nodes_override > 0 and world_size_int % env_num_nodes_override == 0:
                env_devices_override = world_size_int // env_num_nodes_override
        if env_devices_override is None:
            env_devices_override = getattr(trainer_opt, "devices", None)
            if env_devices_override in (None, "auto"):
                env_devices_override = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if isinstance(env_devices_override, str):
            try:
                env_devices_override = int(env_devices_override)
            except ValueError:
                env_devices_override = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if not env_devices_override or env_devices_override <= 0:
            env_devices_override = 1 if not torch.cuda.is_available() else torch.cuda.device_count() or 1

        if (not env_num_nodes_from_env) and world_size_int is not None and env_devices_override > 0:
            inferred_nodes = world_size_int // env_devices_override
            if inferred_nodes > 0:
                env_num_nodes_override = inferred_nodes

        accelerator_value = "gpu" if torch.cuda.is_available() else "cpu"

        trainer_kwargs.update({
            "accelerator": accelerator_value,
            "devices": env_devices_override,
            "num_nodes": env_num_nodes_override,
        })

        trainer_opt.accelerator = accelerator_value
        trainer_opt.devices = env_devices_override
        trainer_opt.num_nodes = env_num_nodes_override
        if hasattr(lightning_config, "trainer"):
            lightning_config.trainer.accelerator = accelerator_value
            lightning_config.trainer.devices = env_devices_override
            lightning_config.trainer.num_nodes = env_num_nodes_override

        trainer_opt_kwargs = {
            k: v for k, v in vars(trainer_opt).items()
            if v is not None and k not in trainer_kwargs
        }

        trainer = Trainer(**trainer_opt_kwargs, **trainer_kwargs)
        trainer.logdir = logdir  ###

        # data: 使用配置中的 datamodule（已去除 PIAT 分支）
        data = instantiate_from_config(config.data)
        # NOTE: Do NOT eagerly call `prepare_data()/setup()` under multi-process launch.
        # In torchrun with WORLD_SIZE>1, calling `setup()` here would load/validate the parquet
        # dataset *once per rank* before PL initializes distributed, which can look like a hang.
        # Let PyTorch Lightning invoke these hooks at the right time.
        try:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except Exception:
            world_size = 1
        if world_size <= 1:
            data.prepare_data()
            data.setup()
            print("#### Data #####")
            if hasattr(data, "datasets"):
                for k in data.datasets:
                    try:
                        length = len(data.datasets[k])
                    except TypeError:
                        length = 'iterable'
                    print(f"{k}, {data.datasets[k].__class__.__name__}, {length}")
            else:
                # best-effort: try common attributes from MsParquetDataModule
                for name in ("train_dataset", "val_dataset", "test_dataset"):
                    ds_obj = getattr(data, name, None)
                    if ds_obj is None:
                        continue
                    try:
                        length = len(ds_obj)
                    except TypeError:
                        length = 'iterable'
                    print(f"{name.replace('_dataset','')}, {ds_obj.__class__.__name__}, {length}")

        # configure learning rate
        bs = config.data.params.batch_size
        base_lr = config.model.base_learning_rate
        # if not cpu:
        #     ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        # else:
        #     ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        # if opt.scale_lr:
        #     model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        #     print(
        #         "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
        #             model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        # else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        # If ckpt_path is provided in YAML and not null, prefer it to override resume logic
        if ckpt_path_from_cfg is not None and str(ckpt_path_from_cfg).strip():
            opt.resume_from_checkpoint = str(ckpt_path_from_cfg).strip()

        if opt.train:
            try:
                # Decide upfront: if the checkpoint looks weights-only (e.g., trainstep_checkpoints),
                # do manual weights load and don't pass ckpt_path to Lightning to avoid optimizer restore.
                resume_path = opt.resume_from_checkpoint
                should_manual_load = False
                if isinstance(resume_path, str) and resume_path:
                    if "/trainstep_checkpoints/" in resume_path:
                        should_manual_load = True
                    else:
                        try:
                            _ckpt_obj = torch.load(resume_path, map_location="cpu")
                            # Heuristic: weights-only if optimizer-related keys are missing
                            has_opt = any(k in _ckpt_obj for k in ("optimizer_states", "lr_schedulers"))
                            # Some PL versions embed under 'loops', still fail in restore if minimal
                            if not has_opt:
                                should_manual_load = True
                        except Exception as _e:
                            print(f"Warning: failed to inspect ckpt ({resume_path}): {_e}")

                if should_manual_load:
                    try:
                        _ckpt_obj = torch.load(resume_path, map_location="cpu")
                        _sd_only = _ckpt_obj.get("state_dict", _ckpt_obj)
                        missing, unexpected = model.load_state_dict(_sd_only, strict=False)
                        print(f"Loaded weights-only checkpoint: missing={len(missing)} unexpected={len(unexpected)}; proceeding without optimizer state.")
                    except Exception as ee:
                        print(f"Manual weights load failed: {ee}")
                    # Run without ckpt_path (fresh optimizer state)
                    trainer.fit(model, data)
                else:
                    # Full resume path
                    trainer.fit(model, data, ckpt_path=resume_path)
            except Exception as e:
                # Avoid hanging in emergency checkpoint saving under DDP; surface the real error first.
                # Users can re-enable emergency ckpt via env var if desired.
                import traceback
                print(f"[ERROR] trainer.fit failed: {e}", flush=True)
                traceback.print_exc()
                if os.environ.get("ENABLE_EMERGENCY_CKPT", "0") == "1":
                    print("Summoning checkpoint.", flush=True)
                    melk()
                raise
        # if not opt.no_test and not trainer.interrupted:
        #     trainer.test(model, data)
    except Exception:
        if opt.debug and trainer is not None and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer is not None and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer is not None and trainer.global_rank == 0:
            print(trainer.profiler.summary())
