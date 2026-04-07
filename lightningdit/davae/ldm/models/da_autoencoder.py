import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
import pytorch_lightning as pl
from typing import Optional

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.da_model import DAAutoencoder as DAModel
from ldm.modules.diffusionmodules.da_model import DASampleAutoencoder as DASimpleModel
from ldm.modules.diffusionmodules.da_model import DADownBlock2d


"""
This module provides a LightningModule that implements Detail-Aligned VAE (DA-VAE).
It uses a dual-path architecture:
- Base path: frozen encoder producing posterior z_b (for alignment only)
- DA path: trainable DA encoder/decoder producing z_hc and reconstruction
"""

class DAVAE(pl.LightningModule):
    """
    Detail-Aligned VAE (DA-VAE): Dual-path KL autoencoder
    - Base path: frozen encoder producing posterior z_b (for alignment only)
    - DA path: trainable DA encoder/decoder producing z_hc and reconstruction

    This is the core implementation of the Detail-Aligned VAE from the CVPR2026 paper.
    """
    def __init__(
        self,
        ddconfig_base,
        ddconfig_da,
        lossconfig,
        base_embed_dim: int,
        base_ckpt_path: Optional[str] = None,
        da_ckpt_path: Optional[str] = None,
        pe_ckpt_path: Optional[str] = None,
        ignore_keys=None,
        image_key: str = "image",
        monitor: Optional[str] = None,
        enable_deep_compress: bool = True,
        upsample_interpolation: str = "nearest",
        freeze_da_encoder: bool = False,
        freeze_da_decoder: bool = False,
        pe_only_mode: bool = False,
        pe_only_weight: float = 1.0,
        da_mode: str = "full",
        da_factor: int = 2,
        align_method: str = "proj",
        z_hc_dropout_p: float = 0.0,
        z_hc_dropout_mode: str = "sample",
    ):
        super().__init__()
        if ignore_keys is None:
            ignore_keys = []

        self.image_key = image_key
        self.monitor = monitor
        self.base_embed_dim = base_embed_dim
        self.align_method = align_method
        assert self.align_method in ["proj", "mean"], \
            f"align_method must be 'proj' or 'mean', got {self.align_method}"
        self.z_hc_dropout_p = float(z_hc_dropout_p)
        self.z_hc_dropout_mode = str(z_hc_dropout_mode)
        assert self.z_hc_dropout_mode in ["element", "channel", "sample"], \
            f"z_hc_dropout_mode must be one of ['element','channel','sample'], got {self.z_hc_dropout_mode}"

        # Base encoder (frozen)
        self.base_encoder = Encoder(**ddconfig_base)
        assert ddconfig_base["double_z"], "Base encoder must output double_z for KL."
        self.base_quant_conv = nn.Conv2d(2 * ddconfig_base["z_channels"], 2 * base_embed_dim, 1)

        # DA path (trainable): choose based on da_mode
        self.da_mode = str(da_mode)
        self.da_factor = int(da_factor)
        if self.da_mode == "simple":
            self.da = DASimpleModel(ddconfig=ddconfig_da, factor=self.da_factor)
        elif self.da_mode == "full":
            self.da = DAModel(ddconfig=ddconfig_da, enable_deep_compress=enable_deep_compress, upsample_interpolation=upsample_interpolation)
        elif self.da_mode == "detail":
            # Use the full DA model; adjust da_down to output only student (residual) moments
            self.da = DAModel(ddconfig=ddconfig_da, enable_deep_compress=enable_deep_compress, upsample_interpolation=upsample_interpolation)
        else:
            raise ValueError(f"Unsupported da_mode: {self.da_mode}. Use 'full', 'simple', or 'detail'.")

        # Freeze flags
        self.freeze_da_encoder = bool(freeze_da_encoder)
        self.freeze_da_decoder = bool(freeze_da_decoder)
        # Simple PE-only training mode
        self.pe_only_mode = bool(pe_only_mode)
        self.pe_only_weight = float(pe_only_weight)

        # Loss
        self.loss = instantiate_from_config(lossconfig)

        # Determine student latent channels (detail mode uses residual channels only)
        if self.da_mode == "detail":
            assert int(self.da.embed_dim) > int(self.base_embed_dim), \
                f"da.embed_dim ({int(self.da.embed_dim)}) must be > base_embed_dim ({int(self.base_embed_dim)}) in detail mode"
            self.embed_dim_student = int(self.da.embed_dim) - int(self.base_embed_dim)
        else:
            self.embed_dim_student = int(self.da.embed_dim)

        # If detail mode with deep compress, reconfigure da_down to output 2 * student channels
        if self.da_mode == "detail" and getattr(self.da, "enable_deep_compress", False):
            in_ch = int(self.da._enc_preconv_channels)
            out_ch = 2 * int(self.embed_dim_student)
            self.da.da_down = DADownBlock2d(
                in_channels=in_ch,
                out_channels=out_ch,
                downsample=True,
                shortcut=True,
            )

        # Optional: host PatchEmbed student in the model so it is saved/loaded with checkpoints
        self.pe_student = None
        if getattr(self.loss, 'pe_align_enable', False):
            # Build pe_student using the teacher weights present in loss.pe_teacher
            teacher = getattr(self.loss, 'pe_teacher', None)
            if teacher is not None and self.pe_student is None and hasattr(teacher, 'weight'):
                tw = teacher.weight.data
                tb = teacher.bias.data if teacher.bias is not None else None
                out_c, in_c_teacher, kh, kw = tw.shape
                in_c_student = int(self.da.embed_dim)
                pe = nn.Conv2d(in_channels=in_c_student, out_channels=int(out_c), kernel_size=(int(kh), int(kw)), stride=(int(kh), int(kw)), bias=True)
                with torch.no_grad():
                    pe.weight.zero_()
                    c = min(in_c_student, int(in_c_teacher))
                    kh_c = min(int(kh), int(tw.shape[2]))
                    kw_c = min(int(kw), int(tw.shape[3]))
                    pe.weight[:, :c, :kh_c, :kw_c] = tw[:, :c, :kh_c, :kw_c]
                    if tb is not None:
                        r = min(int(out_c), int(tb.shape[0]))
                        pe.bias.zero_()
                        pe.bias[:r] = tb[:r]
                    else:
                        nn.init.zeros_(pe.bias)
                self.pe_student = pe
                # Wire into loss for joint use
                self.loss.pe_student = self.pe_student

        # Optionally load and freeze the base path from checkpoint
        if base_ckpt_path is not None:
            self._init_base_from_ckpt(base_ckpt_path, ignore_keys)
        self._freeze_module(self.base_encoder)
        self._freeze_module(self.base_quant_conv)

        # Optionally load DA model checkpoint

        if da_ckpt_path is not None:
            self.init_da_from_ckpt(da_ckpt_path, ignore_keys=ignore_keys or [])
        # Optionally load PatchEmbed weights (default to da_ckpt_path if not provided)
        pe_path = pe_ckpt_path if pe_ckpt_path is not None else da_ckpt_path
        if pe_path is not None:
            self._load_pe_from_ckpt(pe_path)

        # Alignment projection: map student latent channels -> Base latent channels
        # Initialize AFTER loading dc so in_channels matches self.embed_dim_student
        if self.align_method == "proj":
            self.align_proj = nn.Conv2d(self.embed_dim_student, base_embed_dim, kernel_size=1, bias=True)
        elif self.align_method == "mean":
            # No learnable projection; will use group mean instead
            self.align_proj = None
            assert int(self.embed_dim_student) % int(base_embed_dim) == 0, \
                f"For align_method='mean', embed_dim_student ({int(self.embed_dim_student)}) must be divisible by base_embed_dim ({int(base_embed_dim)})"


        # Optionally freeze DA encoder/decoder components (da_down/up remain trainable)
        if self.freeze_da_encoder:
            self._freeze_module(self.da.encoder)
            # Also freeze encoder-side conv heads if present (e.g., quant_conv)
            if hasattr(self.da, "quant_conv") and self.da.quant_conv is not None:
                self._freeze_module(self.da.quant_conv)
            if hasattr(self.da, "da_down") and self.da.da_down is not None:
                self._freeze_module(self.da.da_down)
        if self.freeze_da_decoder:
            if hasattr(self.da, "da_up") and self.da.da_up is not None:
                self._freeze_module(self.da.da_up)
            self._freeze_module(self.da.decoder)
            # Also freeze decoder-side conv heads if present (e.g., post_quant_conv)
            if hasattr(self.da, "post_quant_conv") and self.da.post_quant_conv is not None:
                self._freeze_module(self.da.post_quant_conv)

        # Manual optimization (two optimizers)
        self.automatic_optimization = False

    # ---------------------- Checkpoints ----------------------
    def _init_base_from_ckpt(self, path, ignore_keys):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        cleaned = {}
        for k, v in sd.items():
            if any(k.startswith(ik) for ik in ignore_keys):
                continue
            if k.startswith("encoder."):
                cleaned[k.replace("encoder.", "base_encoder.")] = v
            elif k.startswith("quant_conv."):
                cleaned[k.replace("quant_conv.", "base_quant_conv.")] = v
        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            print(f"[DAVAE] base init missing={len(missing)} unexpected={len(unexpected)}")

    def init_da_from_ckpt(self, path, ignore_keys=list(), strict_encoder=False, strict_decoder=False):
        self.da.load_pretrained(path, ignore_keys=ignore_keys, strict_encoder=strict_encoder, strict_decoder=strict_decoder)

    def _load_pe_from_ckpt(self, path: str):
        # Load PatchEmbed student weights from checkpoint if present
        try:
            sd_raw = torch.load(path, map_location="cpu")
            sd = sd_raw.get("state_dict", sd_raw)
            # Find candidate keys for PE weights/bias
            k_w = None
            k_b = None
            for k in list(sd.keys()):
                if k.endswith("x_embedder.proj.weight") or k.endswith("pe_student.weight"):
                    k_w = k
                if k.endswith("x_embedder.proj.bias") or k.endswith("pe_student.bias"):
                    k_b = k
            if k_w is None:
                return
            w = sd[k_w]
            b = sd.get(k_b, None)
            pe_module = getattr(self, "pe_student", None)
            if pe_module is None:
                return
            # Direct copy (assumes exactly matching shapes)
            with torch.no_grad():
                if w.dtype != pe_module.weight.dtype:
                    w = w.to(dtype=pe_module.weight.dtype)
                pe_module.weight.copy_(w)
                if pe_module.bias is not None and b is not None:
                    if b.dtype != pe_module.bias.dtype:
                        b = b.to(dtype=pe_module.bias.dtype)
                    pe_module.bias.copy_(b)
            print(f"Loaded PatchEmbed weights from checkpoint")
        except Exception as e:
            print(f"[DAVAE] _load_pe_from_ckpt: skipped pe_student load ({e})")

    @staticmethod
    def _freeze_module(module: nn.Module):
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()

    def _maybe_dropout_latent(self, z: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return z
        p = float(getattr(self, "z_hc_dropout_p", 0.0))
        if p <= 0.0:
            return z
        mode = getattr(self, "z_hc_dropout_mode", "element")
        if mode == "element":
            return F.dropout(z, p=p, training=True)
        if mode == "channel":
            # Zero out entire channels per sample
            return F.dropout2d(z, p=p, training=True)
        if mode == "sample":
            # With prob p, zero the entire latent for a sample
            keep_prob = 1.0 - p
            if keep_prob <= 0.0:
                return torch.zeros_like(z)
            mask = (torch.rand(z.shape[0], 1, 1, 1, device=z.device, dtype=z.dtype) < keep_prob).to(z.dtype)
            return z * (mask / keep_prob)
        return z

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_encoder.eval()
        self.base_quant_conv.eval()
        # Keep frozen DA parts in eval mode during training (da_down/up stay trainable)
        if getattr(self, "freeze_da_encoder", False):
            self.da.encoder.eval()
            if hasattr(self.da, "quant_conv") and self.da.quant_conv is not None:
                self.da.quant_conv.eval()
        if getattr(self, "freeze_da_decoder", False):
            self.da.decoder.eval()
            if hasattr(self.da, "post_quant_conv") and self.da.post_quant_conv is not None:
                self.da.post_quant_conv.eval()
        return self

    # ---------------------- Encode/Decode ----------------------
    def _compute_align_proj(self, z_hc):
        """
        Non-learned alignment projection via group mean.
        Groups z_hc channels and averages them to match base_embed_dim.
        
        Args:
            z_hc: (N, self.dc.embed_dim, H, W)
        Returns:
            z_hc_mapped: (N, base_embed_dim, H, W)
        """
        N, C, H, W = z_hc.shape
        num_groups = C // self.base_embed_dim
        # Reshape to (N, base_embed_dim, num_groups, H, W)
        z_hc_grouped = z_hc.view(N, self.base_embed_dim, num_groups, H, W)
        # Mean over the group dimension
        z_hc_mapped = z_hc_grouped.mean(dim=2)
        return z_hc_mapped
    
    def encode_base(self, x):
        x_ds = F.interpolate(x, scale_factor=0.5, mode="bicubic", align_corners=False)
        h = self.base_encoder(x_ds)
        moments = self.base_quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def encode_hc(self, x):
        return self.da.encode(x)

    def decode(self, z_hc):
        return self.da.decode(z_hc)

    def forward(self, input, sample_posterior: bool = True):
        with torch.no_grad():
            base_post = self.encode_base(input)
            z_b = base_post.mode().detach()

        hc_post = self.encode_hc(input)
        z_hc_raw = hc_post.sample() if sample_posterior else hc_post.mode()
        # Keep a clean copy for VF/PE losses
        z_hc = z_hc_raw

        if self.align_method == "proj":
            z_hc_mapped = self.align_proj(z_hc)
        elif self.align_method == "mean":
            z_hc_mapped = self._compute_align_proj(z_hc)

        # In detail mode, concatenate student and teacher latents before decode
        # Apply dropout only to the decoder input
        z_hc_for_decode = self._maybe_dropout_latent(z_hc_raw)
        if self.da_mode == "detail":
            z_for_decode = torch.cat([z_b, z_hc_for_decode], dim=1)
        else:
            z_for_decode = z_hc_for_decode

        dec = self.decode(z_for_decode)
        return dec, (base_post, hc_post), (z_b, z_hc, z_hc_mapped)

    # ---------------------- Lightning utils ----------------------
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if not x.shape[1] in (1, 3):
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)

        # If in PE-only mode, optimize only projection alignment
        if getattr(self, "pe_only_mode", False):
            with torch.no_grad():
                base_post = self.encode_base(inputs)
                z_b = base_post.mode().detach()
            hc_post = self.encode_hc(inputs)
            z_hc = hc_post.sample()

            # Build student if hosted; fall back to loss.pe_student
            pe_student = getattr(self, 'pe_student', None)
            if pe_student is None and hasattr(self, 'loss') and hasattr(self.loss, 'pe_student'):
                pe_student = self.loss.pe_student
            pe_teacher = getattr(self.loss, 'pe_teacher', None) if hasattr(self, 'loss') else None

            if pe_student is None or pe_teacher is None:
                raise RuntimeError("pe_only_mode requires both pe_student and pe_teacher to be initialized via loss config.")

            zb = z_b
            zh = z_hc  # no dropout: PE/VF alignment should use clean latent
            # Optional latent normalization on z_b before teacher
            if hasattr(self.loss, 'pe_latent_norm_enable') and self.loss.pe_latent_norm_enable:
                if hasattr(self.loss, '_apply_pe_latent_normalization'):
                    zb = self.loss._apply_pe_latent_normalization(zb)

            # dtype alignment
            zb = zb.to(dtype=pe_teacher.weight.dtype)
            zh = zh.to(dtype=pe_student.weight.dtype)

            with torch.no_grad():
                teacher_proj = pe_teacher(zb)
            student_proj = pe_student(zh)
            pe_loss = F.mse_loss(student_proj, teacher_proj.detach(), reduction="mean")
            pe_loss = self.pe_only_weight * pe_loss

            ae_opt = self.optimizers()
            # In case Lightning returns a tuple/list
            if isinstance(ae_opt, (list, tuple)):
                ae_opt = ae_opt[0]

            self._log_rank0("pe_only/loss", pe_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=inputs.shape[0])

            ae_opt.zero_grad()
            self.manual_backward(pe_loss)
            self._log_optimizer_grad_norms(ae_opt, prefix="ae/")
            ae_opt.step()
            return

        # Default: full loss path
        reconstructions, (base_post, hc_post), (z_b, z_hc, z_hc_mapped) = self(inputs)

        ae_opt, disc_opt = self.optimizers()

        # Choose an encoder parameter that actually participates in both
        # reconstruction and VF loss graphs. The DA encode path bypasses
        # `encoder.conv_out`, so prefer the DA downsample conv when enabled;
        # otherwise fall back to the encoder's norm_out weight.
        if getattr(self.da, "enable_deep_compress", False) and hasattr(self.da, "da_down"):
            enc_last_layer = self.da.da_down.conv.weight
        else:
            enc_last_layer = self.da.encoder.norm_out.weight

        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            hc_post,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
            z=z_hc_mapped,
            aux_feature=z_b,
            enc_last_layer=enc_last_layer,
            z_pe=z_hc,
            align_method=self.align_method,
        )

        self._log_rank0("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=inputs.shape[0])
        for k, v in log_dict_ae.items():
            self._log_rank0(k, v, prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=inputs.shape[0])

        

        ae_opt.zero_grad()
        self.manual_backward(aeloss)
        self._log_optimizer_grad_norms(ae_opt, prefix="ae/")
        ae_opt.step()

        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            hc_post,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
            enc_last_layer=enc_last_layer,
            align_method=self.align_method,
        )
        self._log_rank0("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=inputs.shape[0])
        for k, v in log_dict_disc.items():
            self._log_rank0(k, v, prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=inputs.shape[0])

        disc_opt.zero_grad()
        self.manual_backward(discloss)
        self._log_optimizer_grad_norms(disc_opt, prefix="disc/")
        disc_opt.step()

    def validation_step(self, batch, batch_idx, dataloader_idx=0, data_type=None):
        inputs = self.get_input(batch, self.image_key)

        if getattr(self, "pe_only_mode", False):
            with torch.no_grad():
                base_post = self.encode_base(inputs)
                z_b = base_post.mode().detach()
                hc_post = self.encode_hc(inputs)
                z_hc = hc_post.mode()

                pe_student = getattr(self, 'pe_student', None)
                if pe_student is None and hasattr(self, 'loss') and hasattr(self.loss, 'pe_student'):
                    pe_student = self.loss.pe_student
                pe_teacher = getattr(self.loss, 'pe_teacher', None) if hasattr(self, 'loss') else None

                if pe_student is None or pe_teacher is None:
                    raise RuntimeError("pe_only_mode requires both pe_student and pe_teacher to be initialized via loss config.")

                zb = z_b
                zh = z_hc
                if hasattr(self.loss, 'pe_latent_norm_enable') and self.loss.pe_latent_norm_enable and hasattr(self.loss, '_apply_pe_latent_normalization'):
                    zb = self.loss._apply_pe_latent_normalization(zb)

                zb = zb.to(dtype=pe_teacher.weight.dtype)
                zh = zh.to(dtype=pe_student.weight.dtype)
                teacher_proj = pe_teacher(zb)
                student_proj = pe_student(zh)
                pe_loss = F.mse_loss(student_proj, teacher_proj.detach(), reduction="mean")
                pe_loss = self.pe_only_weight * pe_loss

                # Rely on global ImageLogger callback for visualization

            self._log_sync("val/pe_only_loss", pe_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=inputs.shape[0])
            return

        reconstructions, (base_post, hc_post), (z_b, z_hc, z_hc_mapped) = self(inputs)
        if getattr(self.da, "enable_deep_compress", False) and hasattr(self.da, "da_down"):
            enc_last_layer = self.da.da_down.conv.weight
        else:
            enc_last_layer = self.da.encoder.norm_out.weight

        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            hc_post,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
            z=z_hc_mapped,
            aux_feature=z_b,
            enc_last_layer=enc_last_layer,
            z_pe=z_hc,
            align_method=self.align_method,
        )

        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            hc_post,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
            enc_last_layer=enc_last_layer,
            align_method=self.align_method,
        )

        if "val/rec_loss" in log_dict_ae:
            self._log_sync("val/rec_loss", log_dict_ae["val/rec_loss"], on_step=False, on_epoch=True, sync_dist=True, batch_size=inputs.shape[0])
        for k, v in {**log_dict_ae, **log_dict_disc}.items():
            if isinstance(k, str) and k.startswith("val/"):
                self._log_sync(k, v, on_step=False, on_epoch=True, sync_dist=True, batch_size=inputs.shape[0])
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.da.parameters())
        # Add align_proj parameters only if using learned projection
        if self.align_method == "proj" and self.align_proj is not None:
            params += list(self.align_proj.parameters())
        # filter only trainable parameters
        params = [p for p in params if p.requires_grad]
        # include trainable PatchEmbed student hosted in this module (preferred)
        if getattr(self, 'pe_student', None) is not None:
            params += [p for p in self.pe_student.parameters() if p.requires_grad]
        else:
            # fallback: include loss-side student if present
            if hasattr(self, 'loss') and hasattr(self.loss, 'pe_student') and self.loss.pe_student is not None:
                loss_trainable = [p for p in self.loss.pe_student.parameters() if p.requires_grad]
                if len(loss_trainable) > 0:
                    params += loss_trainable
        opt_ae = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))
        # In pe-only mode, skip discriminator optimizer entirely
        if getattr(self, 'pe_only_mode', False):
            return [opt_ae], []
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    # ---------------------- Export utilities ----------------------
    @torch.no_grad()
    def export_patch_embed_state(self):
        if getattr(self, 'pe_student', None) is None:
            return None
        return {
            'x_embedder.proj.weight': self.pe_student.weight.detach().cpu(),
            'x_embedder.proj.bias': self.pe_student.bias.detach().cpu(),
        }

    @torch.no_grad()
    def save_patch_embed(self, path: str):
        sd = self.export_patch_embed_state()
        if sd is None:
            raise RuntimeError("pe_student not initialized; enable PE alignment to train it.")
        torch.save(sd, path)
        return path

    def get_last_layer(self):
        return self.da.decoder.conv_out.weight

    def _log_sync(self, key, value, **kwargs):
        if isinstance(value, torch.Tensor):
            value = value.to(self.device)
        else:
            value = torch.tensor(value, device=self.device)
        self.log(key, value, **kwargs)

    def _log_rank0(self, key, value, **kwargs):
        if getattr(self, "global_rank", 0) != 0:
            return
        if isinstance(value, torch.Tensor):
            value = value
        else:
            value = torch.tensor(value, device=self.device)
        self.log(key, value, **kwargs)

    def _log_optimizer_grad_norms(self, optimizer, prefix=""):
        if getattr(self, "global_rank", 0) != 0:
            return
        param_id_to_name = {id(param): name for name, param in self.named_parameters()}
        for group in optimizer.param_groups:
            for param in group.get("params", []):
                if param is None or param.grad is None:
                    continue
                name = param_id_to_name.get(id(param))
                if not name:
                    continue
                grads = param.grad.detach()
                grad_norm = (grads.norm(p=2) / grads.numel()).item()
                key = f"grad_norm.{prefix}{name}" if prefix else f"grad_norm/{name}"
                self.log(key, grad_norm, prog_bar=False, logger=True, on_step=True, on_epoch=False)

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, _, _ = self(x)
            if x.shape[1] > 3:
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["reconstructions"] = xrec
        # Also provide PE-only visualization helper when batch is raw images
        log["inputs"] = x
        return log

    # Removed: rely purely on ImageLogger callback for image logging in pe_only mode

    def to_rgb(self, x):
        return 2.0 * (x - x.min()) / (x.max() - x.min() + 1e-6) - 1.0

    # ---------------------- Checkpoint utils ----------------------
    def load_pretrained(self, path, ignore_keys=list(), strict_encoder=False, strict_decoder=False):
        self.init_da_from_ckpt(path, ignore_keys=ignore_keys, strict_encoder=strict_encoder, strict_decoder=strict_decoder)
        self._load_pe_from_ckpt(path)
        return self

# 兼容旧命名
DAAutoencoderKL = DAVAE

