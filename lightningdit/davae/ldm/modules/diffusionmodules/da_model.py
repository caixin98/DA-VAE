import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution


# class DADownBlock2d(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, downsample: bool = False, shortcut: bool = True) -> None:
#         super().__init__()

#         self.downsample = downsample
#         self.factor = 2
#         self.stride = 1 if downsample else 2
#         self.group_size = in_channels * self.factor**2 // out_channels
#         self.shortcut = shortcut

#         out_ratio = self.factor**2
#         if downsample:
#             assert out_channels % out_ratio == 0
#             out_channels = out_channels // out_ratio

#         self.conv = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=3,
#             stride=self.stride,
#             padding=1,
#         )

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         x = self.conv(hidden_states)
#         if self.downsample:
#             x = F.pixel_unshuffle(x, self.factor)

#         if self.shortcut:
#             y = F.pixel_unshuffle(hidden_states, self.factor)
#             y = y.unflatten(1, (-1, self.group_size))
#             y = y.mean(dim=2)
#             hidden_states = x + y
#         else:
#             hidden_states = x

#         return hidden_states


class DADownBlock2d(nn.Module):
    """
    Down block:
      - 主干: conv 下采样（stride=2）或 conv 后 pixel_unshuffle 下采样（等价 stride=2）
      - shortcut: 先 pixel_unshuffle，把 H,W 对齐到 H/2,W/2
          * 若 (C_in * r^2) 能被 out_channels 整除 -> 分组均值到 out_channels（零参数）
          * 否则 -> 1x1 conv 投影到 out_channels（可学习）
    形状：
      输入:  [B, C_in, H, W] (H,W 必须被 r 整除)
      输出:  [B, out_channels, H/2, W/2]
    """
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False, shortcut: bool = True) -> None:
        super().__init__()
        self.downsample = downsample
        self.shortcut = shortcut
        self.factor = 2  # r
        r2 = self.factor ** 2

        # ===== 主干 conv 设置（确保最终 x 是 [B, out_channels, H/2, W/2]）=====
        self.target_out_channels = out_channels  # 目标输出通道（相加后的通道）
        conv_out_channels = out_channels
        stride = 2
        if downsample:
            # conv 不降采样（stride=1），再用 pixel_unshuffle 做空间/2、通道*4
            assert out_channels % r2 == 0, f"downsample=True 时要求 out_channels 可被 {r2} 整除"
            conv_out_channels = out_channels // r2
            stride = 1

        self.conv = nn.Conv2d(
            in_channels,
            conv_out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

        # ===== shortcut 分支的“分组均值 or 1x1 投影”选择 =====
        total_after_unshuffle = in_channels * r2  # y 在 unshuffle 之后的通道数
        self._divisible = (total_after_unshuffle % out_channels == 0)

        if self.shortcut and not self._divisible:
            # 无法整除：使用 1x1 conv 直接把 total_after_unshuffle -> out_channels
            self.shortcut_proj = nn.Conv2d(total_after_unshuffle, out_channels, kernel_size=1, bias=False)
            # 可选：零初始化，初期等价只走主干，更稳
            nn.init.zeros_(self.shortcut_proj.weight)
        else:
            self.shortcut_proj = None  # 走分组均值，无参数
            if self.shortcut:
                self.group_size = total_after_unshuffle // out_channels  # 保证是整数（_divisible=True）
                assert self.group_size >= 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = hidden_states.shape
        # 基本检查（避免奇怪尺寸）
        assert H % self.factor == 0 and W % self.factor == 0, \
            f"H,W 必须能被 {self.factor} 整除，当前 {(H, W)}"

        # 主干
        x = self.conv(hidden_states)  # [B, conv_out, H/2 or H, W/2 or W]
        if self.downsample:
            x = F.pixel_unshuffle(x, self.factor)  # [B, out_channels, H/2, W/2]
        # 现在 x: [B, out_channels, H/2, W/2]

        if not self.shortcut:
            return x

        # shortcut：对齐空间到 H/2,W/2
        y = F.pixel_unshuffle(hidden_states, self.factor)  # [B, C_in * r^2, H/2, W/2]

        if self._divisible:
            # 分组均值：把通道按 group_size 分组到 out_channels
            # [B, out_channels, group_size, H/2, W/2] -> 对 group 维求均值
            y = y.unflatten(1, (-1, self.group_size)).mean(dim=2)
            # y: [B, out_channels, H/2, W/2]
        else:
            # 不能整除：用 1x1 conv 做通道投影
            y = self.shortcut_proj(y)  # [B, out_channels, H/2, W/2]

        return x + y



class DAUpBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        interpolate: bool = False,
        shortcut: bool = True,
        interpolation_mode: str = "nearest",
    ) -> None:
        super().__init__()

        self.interpolate = interpolate
        self.interpolation_mode = interpolation_mode
        self.shortcut = shortcut
        self.factor = 2
        self.repeats = out_channels * self.factor**2 // in_channels

        out_ratio = self.factor**2

        if not interpolate:
            out_channels = out_channels * out_ratio

        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.interpolate:
            x = F.interpolate(hidden_states, scale_factor=self.factor, mode=self.interpolation_mode)
            x = self.conv(x)
        else:
            x = self.conv(hidden_states)
            x = F.pixel_shuffle(x, self.factor)

        if self.shortcut:
            y = hidden_states.repeat_interleave(self.repeats, dim=1, output_size=hidden_states.shape[1] * self.repeats)
            y = F.pixel_shuffle(y, self.factor)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class DAAutoencoder(nn.Module):
    def __init__(self,
                 ddconfig,
                 enable_deep_compress: bool = True,
                 upsample_interpolation: str = "nearest"):
        super().__init__()

        ddconfig = dict(ddconfig)
        embed_dim = ddconfig.pop("embed_dim", None)

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"], "ddconfig.double_z must be True for KL VAE"

        z_channels = ddconfig["z_channels"]

        if embed_dim is None:
            embed_dim = z_channels * 2 * 2
        self.enable_deep_compress = enable_deep_compress
        self._enc_preconv_channels = self.encoder.ch * self.encoder.in_ch_mult[-1]
        self._dec_block_in = self.decoder.conv_in.out_channels

        if self.enable_deep_compress:
            self.da_down = DADownBlock2d(
                in_channels=self._enc_preconv_channels,
                out_channels=2 * embed_dim,
                downsample=True,
                shortcut=True,
            )
            self.da_up = DAUpBlock2d(
                in_channels=embed_dim,
                out_channels=self._dec_block_in,
                interpolate=False,
                shortcut=True,
                interpolation_mode=upsample_interpolation,
            )

        self.embed_dim = embed_dim

    def load_pretrained(self, path, ignore_keys=list(), strict_encoder=False, strict_decoder=False, strict_dc=False):
        obj = torch.load(path, map_location="cpu")
        sd = obj.get("state_dict", obj)

        # Strip potential 'module.' prefix
        def strip_prefix(k, prefix):
            return k[len(prefix):] if k.startswith(prefix) else None

        enc_sd = {}
        dec_sd = {}
        down_sd = {}
        up_sd = {}

        # Common heads seen in checkpoints: direct, 'dc.', and sometimes 'model.dc.'
        heads = ("", "dc.", "model.dc.")

        for k, v in sd.items():
            if any(k.startswith(ik) for ik in ignore_keys):
                continue
            k_wo_module = k[7:] if k.startswith("module.") else k

            matched = False
            for head in heads:
                enc_k = strip_prefix(k_wo_module, f"{head}encoder.")
                if enc_k is not None:
                    enc_sd[enc_k] = v
                    matched = True
                    break
                dec_k = strip_prefix(k_wo_module, f"{head}decoder.")
                if dec_k is not None:
                    dec_sd[dec_k] = v
                    matched = True
                    break
                down_k = strip_prefix(k_wo_module, f"{head}dc_down.")
                if down_k is not None:
                    down_sd[down_k] = v
                    matched = True
                    break
                up_k = strip_prefix(k_wo_module, f"{head}dc_up.")
                if up_k is not None:
                    up_sd[up_k] = v
                    matched = True
                    break
            if matched:
                continue

        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=strict_encoder)
        print(f"Loaded encoder: missing={len(missing)} unexpected={len(unexpected)}")
        missing, unexpected = self.decoder.load_state_dict(dec_sd, strict=strict_decoder)
        print(f"Loaded decoder: missing={len(missing)} unexpected={len(unexpected)}")

     
        # Optionally load da_down / da_up if present and the module is enabled
        # Helper to filter out keys with shape mismatch to avoid RuntimeError
        def _filter_by_shape(module: nn.Module, candidate_sd: dict):
            current = module.state_dict()
            filtered = {}
            skipped = []
            for k, v in candidate_sd.items():
                if k not in current:
                    continue
                if current[k].shape != v.shape:
                    skipped.append((k, tuple(v.shape), tuple(current[k].shape)))
                    continue
                filtered[k] = v
            return filtered, skipped

        if getattr(self, "enable_deep_compress", True) and hasattr(self, "da_down") and down_sd:
            filtered_down_sd, skipped_down = _filter_by_shape(self.da_down, down_sd)
            missing, unexpected = self.da_down.load_state_dict(filtered_down_sd, strict=False)
            print(
                f"Loaded da_down: missing={len(missing)} unexpected={len(unexpected)} skipped_shape={len(skipped_down)}"
            )
        if getattr(self, "enable_deep_compress", True) and hasattr(self, "da_up") and up_sd:
            filtered_up_sd, skipped_up = _filter_by_shape(self.da_up, up_sd)
            missing, unexpected = self.da_up.load_state_dict(filtered_up_sd, strict=False)
            print(
                f"Loaded da_up: missing={len(missing)} unexpected={len(unexpected)} skipped_shape={len(skipped_up)}"
            )

    def _encode_preconv(self, x: torch.Tensor) -> torch.Tensor:
        e = self.encoder
        temb = None
        hs = [e.conv_in(x)]
        for i_level in range(e.num_resolutions):
            for i_block in range(e.num_res_blocks):
                h = e.down[i_level].block[i_block](hs[-1], temb)
                if len(e.down[i_level].attn) > 0:
                    h = e.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != e.num_resolutions - 1:
                hs.append(e.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = e.mid.block_1(h, temb)
        h = e.mid.attn_1(h)
        h = e.mid.block_2(h, temb)

        h = e.norm_out(h)
        h = F.silu(h)
        return h

    def encode(self, x):
        # take pre-conv features, bypassing encoder.conv_out
        h = self._encode_preconv(x)
        if self.enable_deep_compress:
            h = self.da_down(h)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z):
        if self.enable_deep_compress:
            z = self.da_up(z)
        dec = self._decode_from_preconv(z)
        return dec

    def _decode_from_preconv(self, h: torch.Tensor) -> torch.Tensor:
        d = self.decoder
        temb = None
        # middle
        h = d.mid.block_1(h, temb)
        h = d.mid.attn_1(h)
        h = d.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(d.num_resolutions)):
            for i_block in range(d.num_res_blocks+1):
                h = d.up[i_level].block[i_block](h, temb)
                if len(d.up[i_level].attn) > 0:
                    h = d.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = d.up[i_level].upsample(h)

        # end
        if getattr(d, 'give_pre_end', False):
            return h
        h = d.norm_out(h)
        h = F.silu(h)
        h = d.conv_out(h)
        if getattr(d, 'tanh_out', False):
            h = torch.tanh(h)
        return h

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    # No training utilities; this is a plain nn.Module wrapper


class DASampleAutoencoder(nn.Module):
    def __init__(self, ddconfig, factor: int = 2):
        super().__init__()

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # Optional KL heads (mirroring AutoencoderKL modules); created lazily if ckpt provides them
        self.quant_conv = None
        self.post_quant_conv = None
        self.factor = factor
        self.embed_dim = ddconfig["z_channels"] * self.factor * self.factor
     

    def load_pretrained(self, path, ignore_keys=list(), strict_encoder=False, strict_decoder=False):
        obj = torch.load(path, map_location="cpu")
        sd = obj.get("state_dict", obj)

        def strip_prefix(k, prefix):
            return k[len(prefix):] if k.startswith(prefix) else None

        enc_sd = {}
        dec_sd = {}
        quant_sd = {}
        post_quant_sd = {}

        # Try common heads; include empty so bare 'encoder.' works
        heads = ("", "dc.", "model.dc.")

        for k, v in sd.items():
            if any(k.startswith(ik) for ik in ignore_keys):
                continue
            k_wo_module = k[7:] if k.startswith("module.") else k

            matched = False
            for head in heads:
                enc_k = strip_prefix(k_wo_module, f"{head}encoder.")
                if enc_k is not None:
                    enc_sd[enc_k] = v
                    matched = True
                    break
                dec_k = strip_prefix(k_wo_module, f"{head}decoder.")
                if dec_k is not None:
                    dec_sd[dec_k] = v
                    matched = True
                    break
                q_k = strip_prefix(k_wo_module, f"{head}quant_conv.")
                if q_k is not None:
                    quant_sd[q_k] = v
                    matched = True
                    break
                pq_k = strip_prefix(k_wo_module, f"{head}post_quant_conv.")
                if pq_k is not None:
                    post_quant_sd[pq_k] = v
                    matched = True
                    break
            if matched:
                continue

        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=strict_encoder)
        print(f"[sample] Loaded encoder: missing={len(missing)} unexpected={len(unexpected)}")
        missing, unexpected = self.decoder.load_state_dict(dec_sd, strict=strict_decoder)
        print(f"[sample] Loaded decoder: missing={len(missing)} unexpected={len(unexpected)}")

        # Lazily instantiate KL heads if present in checkpoint
        if len(quant_sd) > 0 and self.quant_conv is None:
            # Infer in/out from weights in sd
            for name, tensor in quant_sd.items():
                if name.endswith("weight") and tensor.ndim == 4:
                    in_ch = tensor.size(1)
                    out_ch = tensor.size(0)
                    self.quant_conv = torch.nn.Conv2d(in_ch, out_ch, 1)
                    break
        if len(post_quant_sd) > 0 and self.post_quant_conv is None:
            for name, tensor in post_quant_sd.items():
                if name.endswith("weight") and tensor.ndim == 4:
                    in_ch = tensor.size(1)
                    out_ch = tensor.size(0)
                    self.post_quant_conv = torch.nn.Conv2d(in_ch, out_ch, 1)
                    break

        if self.quant_conv is not None and len(quant_sd) > 0:
            missing, unexpected = self.quant_conv.load_state_dict(quant_sd, strict=False)
            print(f"[sample] Loaded quant_conv: missing={len(missing)} unexpected={len(unexpected)}")
        if self.post_quant_conv is not None and len(post_quant_sd) > 0:
            missing, unexpected = self.post_quant_conv.load_state_dict(post_quant_sd, strict=False)
            print(f"[sample] Loaded post_quant_conv: missing={len(missing)} unexpected={len(unexpected)}")

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        # AutoencoderKL path with extra shuffle: encoder -> (quant_conv) -> pixel_unshuffle -> posterior
        h = self.encoder(x)
        moments_in = h
        if getattr(self, "quant_conv", None) is not None:
            moments_in = self.quant_conv(moments_in)
        else:
            print("[sample] Warning: quant_conv not found; using encoder output as moments.")
        posterior = DiagonalGaussianDistribution(moments_in)
        moments_deep = posterior.mode()
        moments_deep = F.pixel_unshuffle(moments_deep, self.factor)
        moments_deep = torch.cat((moments_deep, moments_deep), 1)
        posterior = DiagonalGaussianDistribution(moments_deep, deterministic=True)
        return posterior

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Invert shuffle, then follow AutoencoderKL path: pixel_shuffle -> (post_quant_conv) -> decoder
        h = F.pixel_shuffle(z, self.factor)
        z_post = h
        if getattr(self, "post_quant_conv", None) is not None:
            z_post = self.post_quant_conv(z_post)
        else:
            print("[sample] Warning: post_quant_conv not found; passing shuffled features to decoder directly.")
        dec = self.decoder(z_post)
        return dec

    def forward(self, input: torch.Tensor, sample_posterior: bool = True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

if __name__ == "__main__":
    # Simple shape test for encoder and decoder
    ddconfig = dict(
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=4,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
    )
    ddconfig["embed_dim"] = 16
    model = DAAutoencoder(ddconfig=ddconfig, enable_deep_compress=True)
    x = torch.randn(2, 3, 256, 256)
    posterior = model.encode(x)
    print("encode moments shape:", posterior.mean.shape)
    z = posterior.sample()
    print("latent z shape:", z.shape)
    xrec = model.decode(z)
    print("decode output shape:", xrec.shape)


