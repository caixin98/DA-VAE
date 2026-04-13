import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from taming.modules.losses.vqperceptual import *
except ImportError:
    raise ImportError(
        "taming-transformers is required. "
        "Install via: pip install taming-transformers-rom1504"
    )


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", pp_style=False, vf_weight=1e2, adaptive_vf=False,
                cos_margin=0, distmat_margin=0, distmat_weight=1.0, cos_weight=1.0,
                 # Additional projection-level alignment settings
                 pe_align_enable: bool = False,
                 pe_align_weight: float = 0.0,
                 pe_teacher_ckpt_path: str = None,
                 pe_latent_in_channels: int = None,
                 pe_patch_size: int = None,
                 # latent normalization for z_b before teacher PatchEmbed
                 pe_latent_norm_enable: bool = False,
                 pe_latent_norm_stats_path: str = None,
                pe_latent_norm_start_channel: int = 16,
                # When align_method=="proj", optionally use MSE instead of cosine/distmat VF
                vf_proj_use_mse: bool = False):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.distmat_weight = distmat_weight
        self.cos_weight = cos_weight
        self.vf_proj_use_mse = bool(vf_proj_use_mse)
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.pp_style = pp_style
        if pp_style:
            print("Using pp_style for nll loss")
        self.vf_weight = vf_weight
        self.adaptive_vf = adaptive_vf
        self.cos_margin = cos_margin
        self.distmat_margin = distmat_margin

        # ---- Projection-level alignment (PatchEmbed) ----
        self.pe_align_enable = bool(pe_align_enable)
        self.pe_align_weight = float(pe_align_weight) if pe_align_weight is not None else 0.0
        self.pe_teacher_ckpt_path = pe_teacher_ckpt_path
        self.pe_latent_in_channels = pe_latent_in_channels
        self.pe_patch_size = pe_patch_size

        # teacher and student PatchEmbed convs (created lazily after loading ckpt)
        self.pe_teacher = None  # frozen
        self.pe_student = None  # trainable

        # latent normalization stats for z_b path
        self.pe_latent_norm_enable = bool(pe_latent_norm_enable)
        self.register_buffer("pe_latent_norm_mean", None, persistent=False)
        self.register_buffer("pe_latent_norm_std", None, persistent=False)
        self.pe_latent_norm_start_channel = int(pe_latent_norm_start_channel) if pe_latent_norm_start_channel is not None else 16

        if self.pe_align_enable and self.pe_teacher_ckpt_path is not None:
            self._init_patch_embed_from_dit_ckpt(self.pe_teacher_ckpt_path)
        if self.pe_latent_norm_enable and pe_latent_norm_stats_path is not None:
            self._load_pe_latent_norm_stats(Path(pe_latent_norm_stats_path))

    # ---------------- Projection-level alignment helpers ----------------
    def _init_patch_embed_from_dit_ckpt(self, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        w_key = "x_embedder.proj.weight"
        b_key = "x_embedder.proj.bias"
        # Support checkpoints saved with DistributedDataParallel: keys prefixed by 'module.'
        if w_key not in sd:
            alt_w_key = f"module.{w_key}"
            alt_b_key = f"module.{b_key}"
            if alt_w_key in sd:
                w = sd[alt_w_key]
                b = sd.get(alt_b_key, None)
            else:
                raise KeyError(f"[LPIPSWithDiscriminator][PE] Missing key '{w_key}' (and '{alt_w_key}') in teacher checkpoint: {ckpt_path}")
        else:
            w = sd[w_key]
            b = sd.get(b_key, None)
        if w.ndim != 4:
            raise RuntimeError(f"[LPIPSWithDiscriminator][PE] Unexpected PatchEmbed weight shape: {w.shape}")
        out_c, in_c, kh, kw = w.shape
        # Create teacher conv (frozen) exactly as in the ckpt (do not adapt)
        teacher = nn.Conv2d(in_channels=int(in_c), out_channels=int(out_c), kernel_size=(int(kh), int(kw)), stride=(int(kh), int(kw)), bias=b is not None)
        teacher.weight.data.copy_(w)
        if b is not None:
            teacher.bias.data.copy_(b)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        self.pe_teacher = teacher
        # pe_student is provided by the host model (e.g., DAVAE)
        self.pe_student = getattr(self, 'pe_student', None)

    def _load_pe_latent_norm_stats(self, stats_path: Path) -> None:
        data = torch.load(str(stats_path), map_location="cpu")
        if isinstance(data, dict) and ("mean" in data and "std" in data):
            mean_t = data["mean"]
            std_t = data["std"]
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            mean_t, std_t = data
        else:
            raise ValueError(f"[LPIPSWithDiscriminator][PE] Unsupported stats content in {stats_path}: type={type(data)} keys={list(data.keys()) if isinstance(data, dict) else None}")

        if not isinstance(mean_t, torch.Tensor):
            mean_t = torch.tensor(mean_t, dtype=torch.float32)
        if not isinstance(std_t, torch.Tensor):
            std_t = torch.tensor(std_t, dtype=torch.float32)
        mean = mean_t.to(dtype=torch.float32).view(1, -1, 1, 1)
        std = std_t.to(dtype=torch.float32).view(1, -1, 1, 1)
        self.pe_latent_norm_mean = mean
        self.pe_latent_norm_std = std
        

    def _apply_pe_latent_normalization(self, latents: torch.Tensor) -> torch.Tensor:
        if (self.pe_latent_norm_mean is None) or (self.pe_latent_norm_std is None):
            return latents
        if latents.ndim != 4:
            return latents
        mean = self.pe_latent_norm_mean.to(device=latents.device, dtype=latents.dtype)
        std = self.pe_latent_norm_std.to(device=latents.device, dtype=latents.dtype)
        start = int(self.pe_latent_norm_start_channel)
        c = latents.shape[1]
        if start >= c:
            return latents
        head = latents[:, :start]
        tail = latents[:, start:]
        stats_c = int(mean.shape[1])
        if stats_c == c:
            tail_norm = (tail - mean[:, start:]) / torch.clamp(std[:, start:], min=1e-6)
        else:
            tail_norm = (tail - mean) / torch.clamp(std, min=1e-6)
        return torch.cat([head, tail_norm], dim=1)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def calculate_adaptive_weight_vf(self, nll_loss, vf_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            vf_grads = torch.autograd.grad(vf_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            vf_grads = torch.autograd.grad(vf_loss, self.last_layer[0], retain_graph=True)[0]

        vf_weight = torch.norm(nll_grads) / (torch.norm(vf_grads) + 1e-4)
        vf_weight = torch.clamp(vf_weight, 0.0, 1e8).detach()
        vf_weight = vf_weight * self.vf_weight
        return vf_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None, z=None, aux_feature=None, enc_last_layer=None,
                z_pe=None, align_method="proj"):
        if not self.pp_style:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                rec_loss = rec_loss + self.perceptual_weight * p_loss
            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss
            if weights is not None:
                weighted_nll_loss = weights*nll_loss
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            kl_loss = posteriors.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        else:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                rec_loss = rec_loss + self.perceptual_weight * p_loss
            nll_loss = rec_loss
            weighted_nll_loss = nll_loss
            if weights is not None:
                weighted_nll_loss = weights*nll_loss
            # weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            # kl_loss = posteriors.kl()
            # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            weighted_nll_loss = torch.mean(weighted_nll_loss)
            nll_loss = torch.mean(nll_loss)
            kl_loss = posteriors.kl(no_sum=True)
            kl_loss = torch.mean(kl_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            # vf loss
            if z is not None and aux_feature is not None:
                use_mse_align = (align_method == "mean") or (align_method == "proj" and getattr(self, "vf_proj_use_mse", True))
                if use_mse_align:
                    # Simple MSE alignment
                    vf_loss = F.mse_loss(z, aux_feature, reduction="mean")
                else:
                    # Original VF loss with cosine similarity + distance matrix
                    z_flat = rearrange(z, 'b c h w -> b c (h w)')
                    aux_feature_flat = rearrange(aux_feature, 'b c h w -> b c (h w)')
                    z_norm = torch.nn.functional.normalize(z_flat, dim=1)
                    aux_feature_norm = torch.nn.functional.normalize(aux_feature_flat, dim=1)
                    z_cos_sim = torch.einsum('bci,bcj->bij', z_norm, z_norm)
                    aux_feature_cos_sim = torch.einsum('bci,bcj->bij', aux_feature_norm, aux_feature_norm)
                    diff = torch.abs(z_cos_sim - aux_feature_cos_sim)
                    vf_loss_1 = torch.nn.functional.relu(diff-self.distmat_margin).mean()
                    vf_loss_2 = torch.nn.functional.relu(1 - self.cos_margin - torch.nn.functional.cosine_similarity(aux_feature, z)).mean()
                    vf_loss = vf_loss_1*self.distmat_weight + vf_loss_2*self.cos_weight
            else:
                vf_loss = None

            # New: projection-level alignment loss via PatchEmbed
            pe_loss = None
            if self.pe_align_enable and self.pe_align_weight > 0.0 and (aux_feature is not None):
                zb = aux_feature
                zh = z_pe if z_pe is not None else z
                # latent norm on z_b before teacher PE if enabled
                if self.pe_latent_norm_enable:
                    zb = self._apply_pe_latent_normalization(zb)
                # both teacher and student must exist; student managed by host model
                if (self.pe_teacher is None) or (self.pe_student is None):
                    pe_loss = None
                else:
                    # Ensure dtype alignment with weights
                    zb = zb.to(dtype=self.pe_teacher.weight.dtype)
                    zh = zh.to(dtype=self.pe_student.weight.dtype)
                    with torch.no_grad():
                        teacher_proj = self.pe_teacher(zb)
                    student_proj = self.pe_student(zh)
                    # L2 between projections (mean)
                    pe_loss = F.mse_loss(student_proj, teacher_proj.detach(), reduction="mean")

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            if vf_loss is not None:
                if self.adaptive_vf:
                    try:
                        vf_weight = self.calculate_adaptive_weight_vf(nll_loss, vf_loss, last_layer=enc_last_layer)
                    except RuntimeError:
                        assert not self.training
                        vf_weight = torch.tensor(0.0)
                else:
                    vf_weight = self.vf_weight
                loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss + vf_weight * vf_loss
            else:
                loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            # add projection-level alignment loss
            if pe_loss is not None and self.pe_align_weight > 0.0:
                loss = loss + (self.pe_align_weight * pe_loss)

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            if vf_loss is not None:
                log["{}/vf_loss".format(split)] = vf_loss.detach().mean()
                if not isinstance(vf_weight, float):
                    log["{}/vf_weight".format(split)] = vf_weight.detach()
                else:
                    log["{}/vf_weight".format(split)] = torch.tensor(vf_weight)
            if pe_loss is not None and self.pe_align_weight > 0.0:
                log["{}/pe_align_loss".format(split)] = (self.pe_align_weight * pe_loss).detach().mean()
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

