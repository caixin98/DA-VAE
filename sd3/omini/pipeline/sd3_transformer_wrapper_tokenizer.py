"""
Tokenizer-aware SD3 transformer wrapper.

Aligns the SD3 transformer with custom tokenizer/latent channel counts
without adding additional pixel-shuffle logic. Patch embedding and
projection heads are re-instantiated when the requested channel counts
mismatch the stock SD3 configuration, while preserving module interfaces
for LoRA and downstream tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import copy
import os

import torch
import torch.nn as nn

from diffusers.models.embeddings import PatchEmbed
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel


@dataclass
class TokenizerWrapperConfig:
    in_channels: int
    embed_dim: int
    out_channels: int
    patch_size: int
    init_mode: str = "zero_pad"
    augment_mode: str = "none"  # "none" or "branch" (conv add, linear concat)
    load_weights_path: Optional[str] = None
    reverse: bool = False


class SplitAddConv2d(nn.Module):
    """Wrap two convs to process original and extra input channels and sum outputs.

    The input tensor is split along channel dimension into the first
    `base_in_channels` channels (routed to `base_conv`) and the remaining
    channels (routed to `extra_conv`). The two conv outputs must have the
    same spatial and channel dimensions and are added element-wise.
    """

    def __init__(self, base_conv: nn.Conv2d, extra_conv: nn.Conv2d, base_in_channels: int, reverse: bool = False):
        super().__init__()
        self.base_conv = base_conv
        self.extra_conv = extra_conv
        self.base_in_channels = int(base_in_channels)
        self.reverse = bool(reverse)
        # Expose common Conv2d-like attributes for downstream compatibility
        self.in_channels = int(base_conv.in_channels + extra_conv.in_channels)
        self.out_channels = int(base_conv.out_channels)
        self.kernel_size = base_conv.kernel_size
        self.stride = base_conv.stride
        self.padding = base_conv.padding if hasattr(base_conv, "padding") else (0, 0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total_channels = x.shape[1]
        if self.reverse:
            # Extras at the front, base as the last base_in_channels
            extra_channels = total_channels - self.base_in_channels
            x_extra = x[:, :extra_channels, :, :]
            x_base = x[:, total_channels - self.base_in_channels :, :, :]
        else:
            # Base at the front, extras after
            x_base = x[:, : self.base_in_channels, :, :]
            x_extra = x[:, self.base_in_channels :, :, :]
        y = self.base_conv(x_base)
        if x_extra.shape[1] > 0:
                y = y + self.extra_conv(x_extra)
        return y


class ConcatLinearHead(nn.Module):
    """Branching proj_out that preserves per-patch channel ordering.

    For SD3, `proj_out` maps tokens [B, N, D] -> [B, N, p^2*C]. The transformer then unpatchifies by
    reshaping features into [p, p, C] with channels as the innermost dimension. To ensure that base
    channels occupy indices 0..C_base-1 within each patch (and extras follow without disturbing base),
    we must concatenate along the channel axis per patch, not along the flattened feature axis.
    """

    def __init__(self, base_linear: nn.Linear, extra_linear: nn.Linear, patch_size: int, reverse: bool = False):
        super().__init__()
        assert base_linear.in_features == extra_linear.in_features
        self.base_linear = base_linear
        self.extra_linear = extra_linear
        self.in_features = int(base_linear.in_features)
        self.patch_size = int(patch_size)
        self.p2 = int(self.patch_size * self.patch_size)
        self.reverse = bool(reverse)
        # Derive per-patch channel counts from out_features
        assert base_linear.out_features % self.p2 == 0, "base_linear.out_features must be divisible by p^2"
        assert extra_linear.out_features % self.p2 == 0, "extra_linear.out_features must be divisible by p^2"
        self.base_out_channels = int(base_linear.out_features // self.p2)
        self.extra_out_channels = int(extra_linear.out_features // self.p2)
        self.out_features = int((self.base_out_channels + self.extra_out_channels) * self.p2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        y_base = self.base_linear(x)   # [B, N, p^2 * C_base]
        y_extra = self.extra_linear(x) # [B, N, p^2 * C_extra]
        b, n, _ = y_base.shape
        # Reshape to explicit patch/channel axes
        yb = y_base.view(b, n, self.p2, self.base_out_channels)
        ye = y_extra.view(b, n, self.p2, self.extra_out_channels)
        if self.reverse:
            y = torch.cat([ye, yb], dim=-1)  # [B, N, p^2, C_base + C_extra]
        else:
            y = torch.cat([yb, ye], dim=-1)  # [B, N, p^2, C_base + C_extra]
        return y.view(b, n, self.p2 * (self.base_out_channels + self.extra_out_channels))
        
class SD3TransformerWrapperTokenizer(nn.Module):
    """Replace SD3 patch/project heads to match custom tokenizer channels."""

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        patch_embed_cfg: Optional[dict] = None,
        patch_embed_weights_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.transformer = transformer
        cfg_dict = patch_embed_cfg or {}
        reverse_value = bool(cfg_dict.get("reverse", False))
        self.cfg = TokenizerWrapperConfig(
            in_channels=int(cfg_dict.get("in_channels", getattr(transformer.config, "in_channels", 16))),
            embed_dim=int(cfg_dict.get("embed_dim", getattr(transformer.config, "hidden_size", 1536))),
            out_channels=int(cfg_dict.get("out_channels", getattr(transformer.config, "out_channels", 16))),
            patch_size=int(cfg_dict.get("patch_size", getattr(transformer.config, "patch_size", 2))),
            init_mode=str(cfg_dict.get("init_mode", "zero_pad")).lower(),
            augment_mode=str(cfg_dict.get("augment_mode", "none")).lower(),
            load_weights_path=patch_embed_weights_path,
            reverse=reverse_value,
        )

        self._init_patch_embed()
        self._init_project_out()

        
        if self.cfg.load_weights_path:
            self.load_patch_embedding_weights(self.cfg.load_weights_path)

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _init_patch_embed(self) -> None:
        cfg = self.cfg
        original = getattr(self.transformer, "pos_embed", None)

        if original is not None and isinstance(original, PatchEmbed):
            patch_embed = copy.deepcopy(original)
        else:
            patch_embed = PatchEmbed(
                patch_size=cfg.patch_size,
                in_channels=cfg.in_channels,
                embed_dim=cfg.embed_dim,
            )

        proj = getattr(patch_embed, "proj", None)
        if isinstance(proj, nn.Conv2d) and proj.in_channels != cfg.in_channels:
            base_in = int(proj.in_channels)
            # When augment_mode == "branch" and increasing channels, use split-add wrapper.
            # Otherwise (including augment_mode == "none"), re-instantiate a single conv and
            # copy/expand weights via _copy_expand_in_channels.
            if cfg.augment_mode == "branch" and cfg.in_channels > base_in:
                extra_in = int(cfg.in_channels - base_in)
                extra_conv = nn.Conv2d(
                    in_channels=extra_in,
                    out_channels=proj.out_channels,
                    kernel_size=proj.kernel_size,
                    stride=proj.stride,
                    padding=getattr(proj, "padding", 0),
                    dilation=getattr(proj, "dilation", 1),
                    groups=getattr(proj, "groups", 1),
                    bias=(proj.bias is not None),
                )
                with torch.no_grad():
                    extra_conv.weight.zero_()
                    if extra_conv.bias is not None:
                        extra_conv.bias.zero_()
                wrapper = SplitAddConv2d(
                    base_conv=proj,
                    extra_conv=extra_conv,
                    base_in_channels=base_in,
                    reverse=cfg.reverse,
                )
                patch_embed.proj = wrapper
            else:
                # Re-instantiate conv and copy/expand overlapping weights
                new_proj = nn.Conv2d(
                    in_channels=cfg.in_channels,
                    out_channels=proj.out_channels,
                    kernel_size=proj.kernel_size,
                    stride=proj.stride,
                    padding=getattr(proj, "padding", 0),
                    dilation=getattr(proj, "dilation", 1),
                    groups=getattr(proj, "groups", 1),
                    bias=(proj.bias is not None),
                )
                with torch.no_grad():
                    self._copy_expand_in_channels(new_proj, proj, mode=cfg.init_mode)
                patch_embed.proj = new_proj

        self.tokenizer_patch_embed = patch_embed
        self.transformer.pos_embed = patch_embed
        # self.transformer.tokenizer_patch_embed = patch_embed

    def _init_project_out(self) -> None:
        cfg = self.cfg
        original_proj = getattr(self.transformer, "proj_out", None)

        desired_out_features = (cfg.patch_size ** 2) * cfg.out_channels
        desired_in_features = None

        if isinstance(original_proj, nn.Linear):
            desired_in_features = original_proj.in_features
            if cfg.augment_mode == "branch" and desired_out_features > original_proj.out_features:
                extra_out = int(desired_out_features - original_proj.out_features)
                extra_linear = nn.Linear(
                    in_features=original_proj.in_features,
                    out_features=extra_out,
                    bias=(original_proj.bias is not None),
                )
                # Zero-init extra branch to preserve baseline behavior
                with torch.no_grad():
                    extra_linear.weight.zero_()
                    if extra_linear.bias is not None:
                        extra_linear.bias.zero_()
                project_out = ConcatLinearHead(
                    base_linear=original_proj,
                    extra_linear=extra_linear,
                    patch_size=cfg.patch_size,
                    reverse=cfg.reverse,
                )
            else:
                if original_proj.out_features == desired_out_features:
                    project_out = copy.deepcopy(original_proj)
                else:
                    project_out = nn.Linear(
                        in_features=desired_in_features,
                        out_features=desired_out_features,
                        bias=(original_proj.bias is not None),
                    )
                    with torch.no_grad():
                        self._copy_resize_linear(project_out, original_proj, mode=cfg.init_mode)
        else:
            # Fallback when no original or non-linear head: initialize a new compatible Linear
            hidden_dim = cfg.embed_dim if original_proj is None else getattr(original_proj, "in_features", cfg.embed_dim)
            desired_in_features = hidden_dim
            project_out = nn.Linear(hidden_dim, desired_out_features, bias=True)

        self.project_out = project_out
        self.transformer.proj_out = self.project_out
        # self.transformer.project_out = project_out

        # Update config metadata for downstream utilities
        if hasattr(self.transformer, "config"):
            self.transformer.config.in_channels = cfg.in_channels
            self.transformer.config.out_channels = cfg.out_channels
            self.transformer.config.patch_size = cfg.patch_size
        self.transformer.in_channels = cfg.in_channels
        self.transformer.out_channels = cfg.out_channels

    # ----------------------------
    # Augment helper modules
    # ----------------------------
    


    @staticmethod
    @torch.no_grad()
    def _copy_expand_in_channels(dst: nn.Conv2d, src: nn.Conv2d, mode: str = "zero_pad") -> None:
        mode = mode if mode in {"tile_divide", "zero_pad"} else "zero_pad"
        src_w = src.weight.to(dtype=dst.weight.dtype, device=dst.weight.device)

        if mode == "tile_divide":
            repeats = max(dst.in_channels // max(src.in_channels, 1), 1)
            expanded = src_w.repeat(1, repeats, 1, 1)
            expanded = expanded[:, : dst.in_channels, :, :]
            dst.weight.copy_(expanded / float(repeats))
        else:
            dst.weight.zero_()
            cin = min(src_w.shape[1], dst.in_channels)
            dst.weight[:, :cin, :, :].copy_(src_w[:, :cin, :, :])

        if dst.bias is not None:
            if src.bias is None:
                dst.bias.zero_()
            else:
                dst.bias.copy_(src.bias.to(dtype=dst.bias.dtype, device=dst.bias.device))

    @staticmethod
    @torch.no_grad()
    def _copy_resize_linear(dst: nn.Linear, src: nn.Linear, mode: str = "zero_pad") -> None:
        """Resize/copy Linear weights from src -> dst, handling in/out feature changes.

        When mode == "tile_divide", weights are repeated along input/output dimensions as
        needed and scaled by the input-repeat factor to preserve activation scale.
        When mode == "zero_pad", the destination is zero-initialized and the overlapping
        submatrix is copied.
        """
        mode = mode if mode in {"tile_divide", "zero_pad"} else "zero_pad"
        src_w = src.weight.to(dtype=dst.weight.dtype, device=dst.weight.device)

        if mode == "tile_divide":
            in_repeats = max(dst.in_features // max(src_w.shape[1], 1), 1)
            out_repeats = max(dst.out_features // max(src_w.shape[0], 1), 1)
            expanded = src_w.repeat(out_repeats, in_repeats)
            expanded = expanded[: dst.out_features, : dst.in_features]
            # Divide by input repeats to stabilize magnitude similar to conv channel tiling
            dst.weight.copy_(expanded / float(in_repeats))
        else:
            dst.weight.zero_()
            rows = min(src_w.shape[0], dst.out_features)
            cols = min(src_w.shape[1], dst.in_features)
            dst.weight[:rows, :cols].copy_(src_w[:rows, :cols])

        if dst.bias is not None:
            if src.bias is None:
                dst.bias.zero_()
            else:
                src_b = src.bias.to(dtype=dst.weight.dtype, device=dst.weight.device)
                if mode == "tile_divide":
                    out_repeats = max(dst.out_features // max(src_b.shape[0], 1), 1)
                    expanded_b = src_b.repeat(out_repeats)[: dst.out_features]
                    dst.bias.copy_(expanded_b)
                else:
                    dst.bias.zero_()
                    rows = min(src_b.shape[0], dst.out_features)
                    dst.bias[:rows].copy_(src_b[:rows])

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def load_patch_embedding_weights(self, weights_path: str) -> None:
        if not isinstance(weights_path, str) or not os.path.exists(weights_path):
            print(f"[TokenizerWrapper] patch embedding weights not found: {weights_path}")
            return

        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            print(f"[TokenizerWrapper] failed to load weights from {weights_path}: {exc}")
            return

        if isinstance(state, dict):
            state = state.get("model_state_dict", state)
        if not isinstance(state, dict):
            print(f"[TokenizerWrapper] unexpected state format when loading {weights_path}")
            return

        patch_keys = {k[len("tokenizer_patch_embed."):]: v for k, v in state.items() if k.startswith("tokenizer_patch_embed.")}
        if patch_keys:
            self.tokenizer_patch_embed.load_state_dict(patch_keys, strict=False)

        proj_keys = {k[len("project_out."):]: v for k, v in state.items() if k.startswith("project_out.")}
        if proj_keys:
            self.project_out.load_state_dict(proj_keys, strict=False)

    def enable_gradient_checkpointing(self):
        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        if hasattr(self.transformer, "disable_gradient_checkpointing"):
            self.transformer.disable_gradient_checkpointing()

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        self.transformer.train(mode)
        return self

    def eval(self):
        super().eval()
        self.transformer.eval()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.transformer.to(*args, **kwargs)
        return self

    def parameters(self, recurse: bool = True):
        yield from super().parameters(recurse=recurse)
        yield from self.transformer.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        yield from super().named_parameters(prefix=prefix, recurse=recurse)
        yield from self.transformer.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        state.update(self.transformer.state_dict(prefix=f"{prefix}transformer.", keep_vars=keep_vars))
        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        transformer_prefix = "transformer."
        wrapper_state = {k: v for k, v in state_dict.items() if not k.startswith(transformer_prefix)}
        transformer_state = {
            k[len(transformer_prefix):]: v for k, v in state_dict.items() if k.startswith(transformer_prefix)
        }
        super().load_state_dict(wrapper_state, strict=False)
        self.transformer.load_state_dict(transformer_state, strict=strict)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)


def create_sd3_transformer_wrapper_tokenizer(
    transformer: SD3Transformer2DModel,
    patch_embed_cfg: Optional[dict] = None,
    patch_embed_weights_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> SD3TransformerWrapperTokenizer:
    return SD3TransformerWrapperTokenizer(
        transformer=transformer,
        patch_embed_cfg=patch_embed_cfg,
        patch_embed_weights_path=patch_embed_weights_path,
        device=device,
        dtype=dtype,
    )
