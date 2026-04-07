import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import PatchEmbed, Mlp
from einops import rearrange
from modeling.flux.flux_rope import FluxRope
from typing import Optional, Tuple, Dict, Any
from torch.nn import functional as F
from modeling.modules.blocks_old import *
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
from diffusers.models.normalization import RMSNorm

"""
Attention模式配置说明:
- attn_mode="standard": 标准自注意力，x作为Q、K、V
- attn_mode="cross": 交叉注意力，x作为Q，latent_tokens作为K、V
- attn_mode="native": 原生注意力，用于flash attention

配置示例:
config.model.vq_model.attn_mode = "cross"  # 设置为交叉注意力模式
"""


# --- 工具函数 ---
def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def rotate_half(x):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)




# --- Timestep Embedding ---
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(float(max_period))) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.positional_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


# here we use the standard attention for batch size > 1, rather than the native seq-len attention
class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = RMSNorm,
            attn_mode: str = "standard",
            add_latent_proj: bool = False
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 分别的Q、K、V投影层
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.add_latent_proj = add_latent_proj
        if add_latent_proj:
            self.to_q_latent = nn.Linear(dim, dim, bias=qkv_bias)
            self.to_k_latent = nn.Linear(dim, dim, bias=qkv_bias)
            self.to_v_latent = nn.Linear(dim, dim, bias=qkv_bias)
            self.q_norm_latent = norm_layer(self.head_dim) if qk_norm else nn.Identity()
            self.k_norm_latent = norm_layer(self.head_dim) if qk_norm else nn.Identity()
            self.proj_latent = nn.Linear(dim, dim, bias=proj_bias)
            self.norm_latent = norm_layer(dim) if scale_norm else nn.Identity()
            self.proj_drop_latent = nn.Dropout(proj_drop)

        if attn_mode == "standard":
            self.forward = self.forward_standard
        elif attn_mode == "native":
            self.forward = self.forward_native
        elif attn_mode == "cross":
            self.forward = self.forward_cross
        else:
            raise ValueError(f"Invalid attention mode: {attn_mode}")
        
        # 保存attention模式作为实例属性
        self.attn_mode = attn_mode

    def forward_cross(
            self,
            x: torch.Tensor,
            freqs_cos: torch.Tensor,
            freqs_sin: torch.Tensor,
            lq_cond: Optional[torch.Tensor] = None,
            latent_tokens: Optional[torch.Tensor] = None,
            drop_latent_tokens: bool = False,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross attention mode: x as query, latent_tokens as key and value
        """
        if latent_tokens is None:
            raise ValueError("latent_tokens must be provided for cross attention mode")
            
        B, N, C = x.shape
        # print(f"x.shape: {x.shape}")
        # print(f"latent_tokens.shape: {latent_tokens.shape}")    
        # print(f"freqs_cos.shape: {freqs_cos.shape}")
        # print(f"freqs_sin.shape: {freqs_sin.shape}")
        # print(f"lq_cond.shape: {lq_cond.shape}")
        
        # x作为query
        q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        # latent_tokens作为key和value
        B_latent, N_latent, C_latent = latent_tokens.shape

        if lq_cond is not None:
            B_lq, N_lq, C_lq = lq_cond.shape
            k_lq = self.to_k(lq_cond).reshape(B_lq, N_lq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v_lq = self.to_v(lq_cond).reshape(B_lq, N_lq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k_lq = self.k_norm(k_lq)
        else:
            k_lq = None
            v_lq = None

        if self.add_latent_proj:
            k_latent = self.to_k_latent(latent_tokens).reshape(B_latent, N_latent, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v_latent = self.to_v_latent(latent_tokens).reshape(B_latent, N_latent, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k_latent = self.k_norm_latent(k_latent)
        else:
            k_latent = self.to_k(latent_tokens).reshape(B_latent, N_latent, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v_latent = self.to_v(latent_tokens).reshape(B_latent, N_latent, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k_latent = self.k_norm(k_latent)
        if lq_cond is not None: 
            if drop_latent_tokens:
                k = k_lq
                v = v_lq
            else:
                k = torch.cat([k_lq, k_latent], dim=2)
                v = torch.cat([v_lq, v_latent], dim=2)
        else:
            k = k_latent
            v = v_latent
        

        img_freqs_cos, img_freqs_sin = freqs_cos[:,:, :N], freqs_sin[:,:, :N]
        latent_freqs_cos, latent_freqs_sin = freqs_cos[:,:, N:], freqs_sin[:,:, N:]
        if drop_latent_tokens:
            latent_freqs_cos = freqs_cos[:,:, N:-N_latent]
            latent_freqs_sin = freqs_sin[:,:, N:-N_latent]
        else:
            latent_freqs_cos = freqs_cos[:,:, N:]
            latent_freqs_sin = freqs_sin[:,:, N:]
        
        # 应用旋转位置编码
        q = q * img_freqs_cos + rotate_half(q) * img_freqs_sin
        k = k * latent_freqs_cos + rotate_half(k) * latent_freqs_sin
        
        # 计算attention
        attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        
        # 重塑并应用输出投影
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.norm(attn_output)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



    def forward_standard(
            self,
            x: torch.Tensor,
            freqs_cos: torch.Tensor,
            freqs_sin: torch.Tensor,
            latent_tokens: Optional[torch.Tensor] = None,
            drop_latent_tokens: bool = False,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
    
        # 记录原始输入的形状，用于后续分离
        original_x_shape = x.shape[1]

        if not self.add_latent_proj and latent_tokens is not None:
            x = torch.cat([x, latent_tokens], dim=1)

        B, N, C = x.shape
        
        q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
        q, k = self.q_norm(q), self.k_norm(k)
        if latent_tokens is not None and self.add_latent_proj:
            N_latent = latent_tokens.shape[1]
            q_latent = self.to_q_latent(latent_tokens).reshape(B, N_latent, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k_latent = self.to_k_latent(latent_tokens).reshape(B, N_latent, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v_latent = self.to_v_latent(latent_tokens).reshape(B, N_latent, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q_latent, k_latent = self.q_norm_latent(q_latent), self.k_norm_latent(k_latent)
            q = torch.cat([q, q_latent], dim=2)
            k = torch.cat([k, k_latent], dim=2)
            v = torch.cat([v, v_latent], dim=2)
        
        q = q * freqs_cos + rotate_half(q) * freqs_sin
        k = k * freqs_cos + rotate_half(k) * freqs_sin
        attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        attn_output = attn_output.transpose(1, 2).reshape(B, -1, C)
        
        if latent_tokens is not None and self.add_latent_proj:
            # 分离x和latent_tokens
            x = attn_output[:, :original_x_shape]
            latent_tokens = attn_output[:, original_x_shape:]
            latent_tokens = self.norm_latent(latent_tokens)
            latent_tokens = self.proj_latent(latent_tokens)
            latent_tokens = self.proj_drop_latent(latent_tokens)
            return x, latent_tokens
        else:
            x = attn_output
            x = self.norm(x)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
    
    def forward_native(self, x, freqs_cos, freqs_sin, cu_seqlens = None):
        N, C = x.shape
        q = self.to_q(x).reshape(N, self.num_heads, self.head_dim).permute(1, 0, 2)
        k = self.to_k(x).reshape(N, self.num_heads, self.head_dim).permute(1, 0, 2)
        v = self.to_v(x).reshape(N, self.num_heads, self.head_dim).permute(1, 0, 2)
        
        ori_dtype = q.dtype
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * freqs_cos + rotate_half(q) * freqs_sin
        k = k * freqs_cos + rotate_half(k) * freqs_sin
        q, k = q.to(ori_dtype), k.to(ori_dtype)
        # 这里假设 flash_attn_varlen_func 已经导入
        from flash_attn import flash_attn_varlen_func
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        x = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
        ).reshape(N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# --- NiTBlock ---
class NiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_norm=False, use_adaln_lora=False, adaln_lora_dim=None, add_latent_proj=False, attn_mode="standard"):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.add_latent_proj = add_latent_proj
        self.attn_mode = attn_mode  # 保存attention模式作为实例属性
        
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm, add_latent_proj=add_latent_proj, attn_mode=attn_mode
        )
        if attn_mode == "cross":
            self.self_attn = Attention(
                hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm, add_latent_proj=False, attn_mode="standard"
            )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        # if add_latent_proj:
        self.norm1_latent = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2_latent = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_latent = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )

        if use_adaln_lora:
            assert adaln_lora_dim is not None
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=True),
                nn.Linear(adaln_lora_dim, 6 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )

    def forward(self, x, c, freqs_cos, freqs_sin, 
                lq_cond = None, latent_tokens = None, drop_latent_tokens = False, cu_seqlens = None):
        """
        简化的forward方法，位置编码通过freqs_cos和freqs_sin传递
        freqs_cos/freqs_sin应该包含所有需要的序列位置编码
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        
        if self.attn_mode == "cross":
            B, N, C = x.shape
            attn_output_x = self.self_attn(self.norm1(x), freqs_cos[:,:,:N], freqs_sin[:,:,:N])
            x = x + gate_msa * attn_output_x
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            
            attn_output_x = self.attn.forward_cross(modulate(self.norm1(x), shift_msa, scale_msa), freqs_cos, freqs_sin, 
            self.norm1(lq_cond), self.norm1_latent(latent_tokens), drop_latent_tokens)
            x = x + gate_msa * attn_output_x
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            return x
        else:
            if self.add_latent_proj:
                attn_output_x, attn_output_latent_tokens = self.attn(
                    modulate(self.norm1(x), shift_msa, scale_msa), 
                    freqs_cos, freqs_sin, self.norm1_latent(latent_tokens), drop_latent_tokens
                )
                x = x + gate_msa * attn_output_x
                latent_tokens = latent_tokens + attn_output_latent_tokens
                latent_tokens = latent_tokens + self.mlp_latent(self.norm2_latent(latent_tokens))
                x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
                return x, latent_tokens
            else:
                attn_output_x = self.attn(
                    modulate(self.norm1(x), shift_msa, scale_msa), 
                    freqs_cos, freqs_sin
                )
                x = x + gate_msa * attn_output_x
                x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
                return x



# --- Final Layer ---
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def _expand_token(token, batch_size: int):
    # 如果token是2维 (num_latent_tokens, hidden_size)，先添加batch维度
    if token.dim() == 2:
        token = token.unsqueeze(0)  # (1, num_latent_tokens, hidden_size)
    # 如果token已经是3维 (1, num_latent_tokens, hidden_size)，直接扩展
    elif token.dim() == 3:
        pass  # 已经是正确形状
    # 如果token是其他维度，先确保是3维再扩展
    elif token.dim() == 4:
        # 如果是 (1, hidden_size, 1, num_latent_tokens)，reshape为 (1, num_latent_tokens, hidden_size)
        if token.shape[0] == 1 and token.shape[2] == 1:
            token = token.squeeze(0).squeeze(1).unsqueeze(0)  # (1, num_latent_tokens, hidden_size)
        else:
            raise ValueError(f"Unexpected 4D token shape: {token.shape}")
    else:
        raise ValueError(f"Unexpected token dimensions: {token.dim()}, shape: {token.shape}")
    
    return token.expand(batch_size, -1, -1)


# --- PatchEmbedEncoder ---
class PatchEmbedEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        vq_model = config.model.vq_model
        input_size = 256
        patch_size = vq_model.get('vit_enc_patch_size', 16)
        in_channels = vq_model.get('in_channels', 3)
        token_size = vq_model.get('token_size', 64)
        if config.model.vq_model.get("quantize_mode", "vq") == "vae":
            token_size = token_size * 2 # needs to split into mean and std
        self.hidden_size = token_size
        self.patch_embed = PatchEmbed(
            input_size, patch_size, in_channels, token_size, bias=True, strict_img_size=False
        )
    def forward(self, x, latent_tokens, lq_cond=None):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        x = x.reshape(batch_size, self.hidden_size, 1, -1) # (B, hidden_size, 1, num_latent_tokens)
        return x





# --- NiTEncoder ---
class NiTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        vq_model = config.model.vq_model
        input_size = 256
        # --- 支持字符串配置 ---
        model_size = vq_model.get('vit_enc_model_size', 'base')
        width_map = {"small": 512, "base": 768, "large": 1024}
        depth_map = {"small": 8, "base": 12, "large": 24}
        num_heads_map = {"small": 8, "base": 12, "large": 16}
        hidden_size = vq_model.get('hidden_size', width_map.get(model_size, 768))
        depth = vq_model.get('vit_enc_depth', depth_map.get(model_size, 12))
        num_heads = vq_model.get('vit_enc_num_heads', num_heads_map.get(model_size, 12))
        # --- 其他参数 ---
        patch_size = vq_model.get('vit_enc_patch_size', 16)
        in_channels = vq_model.get('in_channels', 3)
        token_size = vq_model.get('token_size', 64)
        num_latent_tokens = vq_model.get('num_latent_tokens', 32)
        mlp_ratio = vq_model.get('mlp_ratio', 4.0)
        theta = vq_model.get('theta', 10000)
        block_kwargs = vq_model.get('block_kwargs', {})
        grad_checkpointing = vq_model.get('grad_checkpointing', True)
        
        self.patch_size = patch_size
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=False
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        head_dim = hidden_size // num_heads
        axes_dim = [head_dim // 8, head_dim * 7 // 16, head_dim * 7 // 16]
        self.blocks = nn.ModuleList([
            NiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])
        self.hidden_size = hidden_size
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size
        self.grad_checkpointing = grad_checkpointing
        if config.model.vq_model.get("quantize_mode", "vq") == "vae":
            self.token_size = self.token_size * 2 # needs to split into mean and std
        self.final_layer = FinalLayer(hidden_size, 1, self.token_size)
        
        # 添加lq条件支持
        self.use_lq_condition = vq_model.get('use_lq_condition', False)
        if self.use_lq_condition:
            self.rope = FluxRope(theta=theta, axes_dim=axes_dim, type="encode_with_lq")
            self.lq_embedder = PatchEmbed(
                input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=False
            )
        else:
            self.rope = FluxRope(theta=theta, axes_dim=axes_dim, type="encode")
        
        # 保存rope类型信息，用于动态切换
        self.rope_type = "encode_with_lq" if self.use_lq_condition else "encode"
        
        # 添加downsample_factor配置支持
        self.downsample_factor = config.experiment.get("downsample_factor", 2) / config.experiment.get("upsample_factor", 1)

        
        self.initialize_weights()
    # --- 权重初始化 ---
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.normal_(self.final_layer.adaLN_modulation[-1].weight, std=0.02)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        if self.use_lq_condition:
            nn.init.normal_(self.lq_embedder.proj.weight, std=0.02)
            nn.init.constant_(self.lq_embedder.proj.bias, 0)

    # --- 前向传播 ---
    def forward(self, x, latent_tokens, lq_cond=None):
        batch_size = x.shape[0] # x.shape = [batch_size, in_channels, height, hidden_size]
        h, w = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        x = self.x_embedder(x)  # (B, D, H, W) -> (B, C, H // patch_size, W // patch_size)
        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        # 根据是否实际使用lq条件来决定rope调用方式
        actual_use_lq = self.use_lq_condition and lq_cond is not None
        if self.use_lq_condition:
            assert lq_cond is not None, "lq_cond is required when use_lq_condition is True"
        if actual_use_lq:
            # 使用lq条件
            h_lq, w_lq = lq_cond.shape[2] // self.patch_size, lq_cond.shape[3] // self.patch_size
            lq_tokens = self.lq_embedder(lq_cond)  # (B, C, H_lq // patch_size, W_lq // patch_size)
            lq_tokens = lq_tokens.reshape(batch_size, -1, self.hidden_size)  # (B, H_lq*W_lq, C)
            x = x.reshape(batch_size, -1, self.hidden_size)  # (B, H*W, C)
            x = torch.cat([x, lq_tokens, latent_tokens], dim=1)
            
            # 使用三段位置编码：hq/lq/latent
            freqs_cos, freqs_sin = self.rope([h, w], [h_lq, w_lq], latent_tokens.shape[1], batch_size, x.device, x.dtype, self.downsample_factor)
        else:
            # 原始的两段位置编码：hq/latent
            x = x.reshape(batch_size, -1, self.hidden_size)  # (B, H*W, C)
            x = torch.cat([x, latent_tokens], dim=1)
            # 如果rope是encode_with_lq类型，需要创建一个临时的encode rope
            if self.rope_type == "encode_with_lq":
                # 创建临时的encode rope
                temp_rope = FluxRope(theta=self.rope.theta, axes_dim=self.rope.axes_dim, type="encode")
                freqs_cos, freqs_sin = temp_rope([h, w], latent_tokens.shape[1], batch_size, x.device, x.dtype, self.downsample_factor)
            else:
                freqs_cos, freqs_sin = self.rope([h, w], latent_tokens.shape[1], batch_size, x.device, x.dtype, self.downsample_factor)
        
        t = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        c = self.t_embedder(t)
        c = c.unsqueeze(1).expand(-1, x.shape[1], -1)
        
        for block in self.blocks:
            # assert block.attn_mode == "standard", "NiTEncoder is not supported for cross attention mode"
            # assert block.add_latent_proj == False, "NiTEncoder is not supported for add_latent_proj mode"
                # if self.add_latent_proj:
                #     x, latent_tokens = checkpoint(block, x, c, freqs_cos, freqs_sin, latent_tokens, use_reentrant=False)
                # else:
                #     # 对于cross attention模式，需要传递latent_tokens
                #     if hasattr(block.attn, 'attn_mode') and block.attn.attn_mode == "cross":
                #         x = checkpoint(block, x, c, freqs_cos, freqs_sin, latent_tokens, use_reentrant=False)
                #     else:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x, c, freqs_cos, freqs_sin, use_reentrant=False)
            else:
                # if self.add_latent_proj:
                #     x, latent_tokens = block(x, c, freqs_cos, freqs_sin, latent_tokens)
                # else:
                #     # 对于cross attention模式，需要传递latent_tokens
                #     if hasattr(block.attn, 'attn_mode') and block.attn.attn_mode == "cross":
                #         x = block(x, c, freqs_cos, freqs_sin, latent_tokens)
                #     else:
                x = block(x, c, freqs_cos, freqs_sin)
                
        if actual_use_lq:
            # 取出latent tokens部分（在lq tokens之后）
            x = x[:, h * w + h_lq * w_lq:]
            c = c[:, h * w + h_lq * w_lq:]
        else:
            # 取出latent tokens部分
            x = x[:, h * w:]
            c = c[:, h * w:]
            
        x = self.final_layer(x, c)    # B, num_latent_tokens, token_size 
        x = x.reshape(batch_size, self.token_size, 1, self.num_latent_tokens)
        return x


class NiTCrossEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        vq_model = config.model.vq_model
        input_size = 256
        # --- 支持字符串配置 ---
        model_size = vq_model.get('vit_enc_model_size', 'base')
        width_map = {"small": 512, "base": 768, "large": 1024}
        depth_map = {"small": 8, "base": 12, "large": 24}
        num_heads_map = {"small": 8, "base": 12, "large": 16}
        hidden_size = vq_model.get('hidden_size', width_map.get(model_size, 768))
        depth = vq_model.get('vit_enc_depth', depth_map.get(model_size, 12))
        num_heads = vq_model.get('vit_enc_num_heads', num_heads_map.get(model_size, 12))
        # --- 其他参数 ---
        patch_size = vq_model.get('vit_enc_patch_size', 16)
        in_channels = vq_model.get('in_channels', 3)
        token_size = vq_model.get('token_size', 64)
        num_latent_tokens = vq_model.get('num_latent_tokens', 32)
        mlp_ratio = vq_model.get('mlp_ratio', 4.0)
        theta = vq_model.get('theta', 10000)
        block_kwargs = vq_model.get('block_kwargs', {})
        grad_checkpointing = vq_model.get('grad_checkpointing', True)
        
        self.patch_size = patch_size
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=False
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        head_dim = hidden_size // num_heads
        axes_dim = [head_dim // 8, head_dim * 7 // 16, head_dim * 7 // 16]
        self.blocks = nn.ModuleList([
            NiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_mode="cross", add_latent_proj=True, **block_kwargs) for _ in range(depth)
        ])
        self.hidden_size = hidden_size
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size
        self.grad_checkpointing = grad_checkpointing
        if config.model.vq_model.get("quantize_mode", "vq") == "vae":
            self.token_size = self.token_size * 2 # needs to split into mean and std
        self.final_layer = FinalLayer(hidden_size, 1, self.token_size)
        
        # 添加lq条件支持
        self.use_lq_condition = vq_model.get('use_lq_condition', False)
        if self.use_lq_condition:
            self.rope = FluxRope(theta=theta, axes_dim=axes_dim, type="encode_with_lq")
            self.lq_embedder = PatchEmbed(
                input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=False
            )
        else:
            self.rope = FluxRope(theta=theta, axes_dim=axes_dim, type="encode")
        
        # 保存rope类型信息，用于动态切换
        self.rope_type = "encode_with_lq" if self.use_lq_condition else "encode"
        
        # 添加downsample_factor配置支持
        self.downsample_factor = config.experiment.get("downsample_factor", 2) / config.experiment.get("upsample_factor", 1)

        
        self.initialize_weights()
    # --- 权重初始化 ---
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.normal_(self.final_layer.adaLN_modulation[-1].weight, std=0.02)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        if self.use_lq_condition:
            nn.init.normal_(self.lq_embedder.proj.weight, std=0.02)
            nn.init.constant_(self.lq_embedder.proj.bias, 0)

    # --- 前向传播 ---
    def forward(self, x, latent_tokens, lq_cond=None):
        batch_size = x.shape[0] # x.shape = [batch_size, in_channels, height, hidden_size]
        h, w = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        x = self.x_embedder(x)  # (B, D, H, W) -> (B, C, H // patch_size, W // patch_size)
        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        # 根据是否实际使用lq条件来决定rope调用方式
        actual_use_lq = self.use_lq_condition and lq_cond is not None
        if self.use_lq_condition:
            assert lq_cond is not None, "lq_cond is required when use_lq_condition is True"
        if actual_use_lq:
            # 使用lq条件
            h_lq, w_lq = lq_cond.shape[2] // self.patch_size, lq_cond.shape[3] // self.patch_size
            lq_tokens = self.lq_embedder(lq_cond)  # (B, C, H_lq // patch_size, W_lq // patch_size)
            lq_tokens = lq_tokens.reshape(batch_size, -1, self.hidden_size)  # (B, H_lq*W_lq, C)
            x = x.reshape(batch_size, -1, self.hidden_size)  # (B, H*W, C)
            x = torch.cat([x, lq_tokens], dim=1)
            
            # 使用三段位置编码：hq/lq/latent
            freqs_cos, freqs_sin = self.rope([h, w], [h_lq, w_lq], latent_tokens.shape[1], batch_size, x.device, x.dtype, self.downsample_factor, diff_first=True)
        else:
            # 原始的两段位置编码：hq/latent
            x = x.reshape(batch_size, -1, self.hidden_size)  # (B, H*W, C)
            # 如果rope是encode_with_lq类型，需要创建一个临时的encode rope
            if self.rope_type == "encode_with_lq":
                # 创建临时的encode rope
                temp_rope = FluxRope(theta=self.rope.theta, axes_dim=self.rope.axes_dim, type="encode")
                freqs_cos, freqs_sin = temp_rope([h, w], latent_tokens.shape[1], batch_size, x.device, x.dtype, self.downsample_factor, diff_first=True)
            else:
                freqs_cos, freqs_sin = self.rope([h, w], latent_tokens.shape[1], batch_size, x.device, x.dtype, self.downsample_factor, diff_first=True)
        
        t = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        c = self.t_embedder(t)
        c = c.unsqueeze(1).expand(-1, latent_tokens.shape[1], -1)
        
        for block in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                latent_tokens = checkpoint(block, latent_tokens, c, freqs_cos, freqs_sin, None, x, use_reentrant=False)
            else:
                latent_tokens = block(latent_tokens, c, freqs_cos, freqs_sin, None, x)
            
        latent_tokens = self.final_layer(latent_tokens, c)    # B, num_latent_tokens, token_size 
        latent_tokens = latent_tokens.reshape(batch_size, self.token_size, 1, self.num_latent_tokens)
        return latent_tokens

# --- QwenEncoder ---
from utils.tensor_image_processor import TensorImageProcessor, smart_resize
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from copy import deepcopy


class QwenEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        vq_model = config.model.vq_model
        
        # 获取配置参数
        token_size = vq_model.get('token_size', 64)
        num_latent_tokens = vq_model.get('num_latent_tokens', 32)
        theta = vq_model.get('theta', 10000)
        grad_checkpointing = vq_model.get('grad_checkpointing', True)
        # 添加处理模式配置
        self.process_mode = vq_model.get('process_mode', 'batch')  # 'native' 或 'batch'
        
        # 添加lq条件支持
        self.use_lq_condition = vq_model.get('use_lq_condition', False)
        
        # 使用新的tensor image processor，支持from_pretrained
        try:
            # 尝试从预训练模型加载配置
            self.processor = TensorImageProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        except:
            # 如果失败，使用默认配置
            self.processor = TensorImageProcessor(
                patch_size=14,
                temporal_patch_size=2,
                merge_size=2,
                do_resize=True,
                do_rescale=True,
                do_normalize=True,
                min_pixels=56 * 56,
                max_pixels=28 * 28 * 1280
            )
        
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        self.model = deepcopy(qwen_model.visual)
        self.patch_size = self.model.patch_size * 2
        self.hidden_size = 1280  # Qwen2.5-VL 的隐藏维度
       
        del qwen_model
        del self.model.merger
        torch.cuda.empty_cache()
            
        # 添加 NiTEncoder 风格的组件
        self.num_latent_tokens = num_latent_tokens
        self.token_size = token_size
        self.grad_checkpointing = grad_checkpointing
        if config.model.vq_model.get("quantize_mode", "vq") == "vae":
            self.token_size = self.token_size * 2 # needs to split into mean and std
        self.final_layer = nn.Linear(self.hidden_size, self.token_size, bias=True)
        # 添加 latent_tokens 投影层
        
        
        self.initialize_weights()
    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.final_layer.weight, std=0.02)
        nn.init.constant_(self.final_layer.bias, 0)

    def forward(self, x, latent_tokens, lq_cond=None):
        B, C, H, W = x.shape
        h_bar, w_bar = smart_resize(H, W, self.patch_size)
        output_h = h_bar // self.patch_size
        output_w = w_bar // self.patch_size
        # 使用新的tensor image processor，直接处理tensor
        processed = self.processor(images=x, do_rescale=False)
        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)

        x = self.model.patch_embed(processed["pixel_values"]) # after patch_embed, x.shape = [B * seq_len, hidden_size]
        image_grid_thw = processed["image_grid_thw"]
        
        # 处理lq条件
        if self.use_lq_condition and lq_cond is not None:
            # 处理lq条件图像
            processed_lq = self.processor(images=lq_cond, do_rescale=False)
            lq_tokens = self.model.patch_embed(processed_lq["pixel_values"])  # [B * seq_len_lq, hidden_size]
            lq_grid_thw = processed_lq["image_grid_thw"]
            
            # 分析lq的序列长度
            lq_seq_lens = []
            for i in range(lq_grid_thw.shape[0]):
                t, h, w = lq_grid_thw[i]
                lq_seq_lens.append(t * h * w)

        # 分析每个batch的序列长度
        image_seq_lens = []
        for i in range(image_grid_thw.shape[0]):
            t, h, w = image_grid_thw[i]
            image_seq_lens.append(t * h * w)
        
        if self.use_lq_condition and lq_cond is not None:
            # 使用lq条件的三段拼接：hq/lq/latent
            if self.process_mode == 'native':
                # native resolution style process
                x_list = []
                lq_start_idx = 0
                start_idx = 0
                for i, (seq_len, lq_seq_len) in enumerate(zip(image_seq_lens, lq_seq_lens)):
                    # 添加当前batch的hq图像tokens
                    end_idx = start_idx + seq_len
                    x_list.append(x[start_idx:end_idx])
                    # 添加当前batch的lq图像tokens
                    lq_end_idx = lq_start_idx + lq_seq_len
                    x_list.append(lq_tokens[lq_start_idx:lq_end_idx])
                    # 添加当前batch的latent tokens
                    x_list.append(latent_tokens[i])
                    start_idx = end_idx
                    lq_start_idx = lq_end_idx
                x = torch.cat(x_list, dim=0)
            else:
                # batch style process
                x = x.reshape(B, -1, self.hidden_size)
                lq_tokens = lq_tokens.reshape(B, -1, self.hidden_size)
                x = torch.cat([x, lq_tokens, latent_tokens], dim=1)
                x = x.reshape(-1, self.hidden_size)
        else:
            # 原始的两段拼接：hq/latent
            if self.process_mode == 'native':
                # native resolution style process
                # 重新组织x和latent_tokens，按batch交替拼接
                x_list = []
                start_idx = 0
                for i, seq_len in enumerate(image_seq_lens):
                    # 添加当前batch的图像tokens
                    end_idx = start_idx + seq_len
                    x_list.append(x[start_idx:end_idx])
                    # 添加当前batch的latent tokens
                    x_list.append(latent_tokens[i])
                    start_idx = end_idx
                x = torch.cat(x_list, dim=0)
            else:
                # batch style process
                # 假设所有batch的序列长度相同，reshape为batch形式
                x = x.reshape(B, -1, self.hidden_size)
                x = torch.cat([x, latent_tokens], dim=1)
                x = x.reshape(-1, self.hidden_size)
        
        if self.use_lq_condition and lq_cond is not None:
            # 使用lq条件的三段位置编码：hq/lq/latent
            if self.process_mode == 'native':
                # 获取hq图像的位置编码
                image_rotary_pos_emb = self.model.rot_pos_emb(image_grid_thw)
                # 获取lq图像的位置编码
                lq_rotary_pos_emb = self.model.rot_pos_emb(lq_grid_thw)
                
                # 为latent tokens创建位置编码
                grid_size = max(2, (self.num_latent_tokens + 1) // 2 * 2)
                latent_grid_thw = torch.tensor([[1, grid_size, grid_size]], device=x.device, dtype=torch.int32)
                latent_grid_thw = latent_grid_thw.repeat(B, 1)
                latent_rotary_pos_emb = self.model.rot_pos_emb(latent_grid_thw)
                # 需要为每个batch生成num_latent_tokens个位置编码
                latent_rotary_pos_emb_per_batch = []
                for i in range(B):
                    start_idx = i * grid_size * grid_size
                    end_idx = start_idx + self.num_latent_tokens
                    latent_rotary_pos_emb_per_batch.append(latent_rotary_pos_emb[start_idx:end_idx])
                latent_rotary_pos_emb = torch.cat(latent_rotary_pos_emb_per_batch, dim=0)

                # 重新组织位置编码：hq/lq/latent
                emb_list = []
                hq_start_idx = 0
                lq_start_idx = 0
                for i, (hq_seq_len, lq_seq_len) in enumerate(zip(image_seq_lens, lq_seq_lens)):
                    # 添加当前batch的hq图像tokens位置编码
                    hq_end_idx = hq_start_idx + hq_seq_len
                    emb_list.append(image_rotary_pos_emb[hq_start_idx:hq_end_idx])
                    # 添加当前batch的lq图像tokens位置编码
                    lq_end_idx = lq_start_idx + lq_seq_len
                    emb_list.append(lq_rotary_pos_emb[lq_start_idx:lq_end_idx])
                    # 添加当前batch的latent tokens位置编码
                    emb_list.append(latent_rotary_pos_emb[i*self.num_latent_tokens:(i+1)*self.num_latent_tokens])
                    hq_start_idx = hq_end_idx
                    lq_start_idx = lq_end_idx
                
                emb = torch.cat(emb_list, dim=0)
            else:
                # batch style position encoding
                # 获取hq图像的位置编码
                image_rotary_pos_emb = self.model.rot_pos_emb(image_grid_thw)
                # 获取lq图像的位置编码
                lq_rotary_pos_emb = self.model.rot_pos_emb(lq_grid_thw)
                
                # 为latent tokens创建位置编码
                grid_size = max(2, (self.num_latent_tokens + 1) // 2 * 2)
                latent_grid_thw = torch.tensor([[1, grid_size, grid_size]], device=x.device, dtype=torch.int32)
                latent_rotary_pos_emb_single = self.model.rot_pos_emb(latent_grid_thw)
                latent_rotary_pos_emb_single = latent_rotary_pos_emb_single[:self.num_latent_tokens]
                latent_rotary_pos_emb = latent_rotary_pos_emb_single.repeat(B, 1)
                
                # 将位置编码reshape为batch形式，然后cat
                image_seq_len = image_seq_lens[0]
                lq_seq_len = lq_seq_lens[0]
                image_rotary_pos_emb = image_rotary_pos_emb.reshape(B, image_seq_len, -1)
                lq_rotary_pos_emb = lq_rotary_pos_emb.reshape(B, lq_seq_len, -1)
                latent_rotary_pos_emb = latent_rotary_pos_emb.reshape(B, self.num_latent_tokens, -1)
                
                # 在序列维度上cat：hq/lq/latent
                emb = torch.cat([image_rotary_pos_emb, lq_rotary_pos_emb, latent_rotary_pos_emb], dim=1)
                emb = emb.reshape(-1, emb.shape[-1])
        else:
            # 原始的两段位置编码：hq/latent
            if self.process_mode == 'native':
                # 获取图像的位置编码
                image_rotary_pos_emb = self.model.rot_pos_emb(image_grid_thw) # the shape of image_rotary_pos_emb is [B * seq_len * 3, head_dim // 2]
                
                # 为latent tokens创建位置编码
                # 为latent tokens创建位置编码 - 使用正确的网格大小
                grid_size = max(2, (self.num_latent_tokens + 1) // 2 * 2)  # 确保是2的倍数且>=2
                latent_grid_thw = torch.tensor([[1, grid_size, grid_size]], device=x.device, dtype=torch.int32) # t = 2 for latent tokens
                latent_grid_thw = latent_grid_thw.repeat(B, 1)
                latent_rotary_pos_emb = self.model.rot_pos_emb(latent_grid_thw)
                # 需要为每个batch生成num_latent_tokens个位置编码
                latent_rotary_pos_emb_per_batch = []
                for i in range(B):
                    start_idx = i * grid_size * grid_size
                    end_idx = start_idx + self.num_latent_tokens
                    latent_rotary_pos_emb_per_batch.append(latent_rotary_pos_emb[start_idx:end_idx])
                latent_rotary_pos_emb = torch.cat(latent_rotary_pos_emb_per_batch, dim=0)

                # 拼接位置编码 - 需要按batch交替拼接
                # 当前: [所有batch的图像tokens, 所有batch的latent tokens]
                # 正确: [batch0_image, batch0_latent, batch1_image, batch1_latent, ...]
                
                # 重新组织位置编码
                emb_list = []
                start_idx = 0
                for i, seq_len in enumerate(image_seq_lens):
                    # 添加当前batch的图像tokens
                    end_idx = start_idx + seq_len
                    emb_list.append(image_rotary_pos_emb[start_idx:end_idx])
                    # 添加当前batch的latent tokens
                    emb_list.append(latent_rotary_pos_emb[i*self.num_latent_tokens:(i+1)*self.num_latent_tokens])
                    start_idx = end_idx
                
                emb = torch.cat(emb_list, dim=0)
            else:
                # batch style position encoding
                # 获取图像的位置编码
                image_rotary_pos_emb = self.model.rot_pos_emb(image_grid_thw) # [B * seq_len * 3, head_dim // 2]
                
                # 为latent tokens创建位置编码 - 只需要一个batch的，然后重复
                grid_size = max(2, (self.num_latent_tokens + 1) // 2 * 2)
                latent_grid_thw = torch.tensor([[1, grid_size, grid_size]], device=x.device, dtype=torch.int32)
                latent_rotary_pos_emb_single = self.model.rot_pos_emb(latent_grid_thw)
                # 只取前num_latent_tokens个位置编码
                latent_rotary_pos_emb_single = latent_rotary_pos_emb_single[:self.num_latent_tokens]
                # 重复B次
                latent_rotary_pos_emb = latent_rotary_pos_emb_single.repeat(B, 1)
                
                # 将位置编码reshape为batch形式，然后cat
                image_seq_len = image_seq_lens[0]  # 假设所有batch的序列长度相同
                image_rotary_pos_emb = image_rotary_pos_emb.reshape(B, image_seq_len, -1)  # [B, L_img, C]
                latent_rotary_pos_emb = latent_rotary_pos_emb.reshape(B, self.num_latent_tokens, -1)  # [B, L_latent, C]
                
                # 在序列维度上cat
                emb = torch.cat([image_rotary_pos_emb, latent_rotary_pos_emb], dim=1)  # [B, L_img + L_latent, C]
                emb = emb.reshape(-1, emb.shape[-1])  # [B * (L_img + L_latent), C]
        
        # 位置编码需要被重复2次以匹配attention head维度
        # 原始维度是40，attention head维度是80
        emb = emb.repeat(1, 2)
        position_embeddings = (emb.cos(), emb.sin())
        

    
        # 计算cu_seqlens用于flash attention
        if self.use_lq_condition and lq_cond is not None:
            # 使用lq条件的三段序列：hq/lq/latent
            if self.process_mode == 'native':
                seq_lens = []
                for i, (hq_seq_len, lq_seq_len) in enumerate(zip(image_seq_lens, lq_seq_lens)):
                    # 每个batch的序列长度 = hq tokens + lq tokens + latent tokens
                    batch_seq_len = hq_seq_len + lq_seq_len + self.num_latent_tokens
                    seq_lens.append(batch_seq_len)
                
                seq_lens = torch.tensor(seq_lens, device=x.device, dtype=torch.int32)
                cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=x.device), seq_lens.cumsum(0)])
            else:
                # batch style: 所有batch的序列长度相同
                t, h, w = image_grid_thw[0]
                hq_seq_len = t * h * w
                t_lq, h_lq, w_lq = lq_grid_thw[0]
                lq_seq_len = t_lq * h_lq * w_lq
                batch_seq_len = hq_seq_len + lq_seq_len + self.num_latent_tokens
                
                seq_lens = torch.full((B,), batch_seq_len, device=x.device, dtype=torch.int32)
                cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=x.device), seq_lens.cumsum(0)])
        else:
            # 原始的两段序列：hq/latent
            if self.process_mode == 'native':
                # 每个batch是一个独立的序列，序列长度是 image_seq_len + latent_seq_len
                seq_lens = []
                for i, seq_len in enumerate(image_seq_lens):
                    # 每个batch的序列长度 = 图像tokens + latent tokens
                    batch_seq_len = seq_len + self.num_latent_tokens
                    seq_lens.append(batch_seq_len)
                
                seq_lens = torch.tensor(seq_lens, device=x.device, dtype=torch.int32)
                cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=x.device), seq_lens.cumsum(0)])
            else:
                # batch style: 所有batch的序列长度相同
                # 计算单个batch的序列长度
                t, h, w = image_grid_thw[0]  # 假设所有batch的尺寸相同
                image_seq_len = t * h * w
                batch_seq_len = image_seq_len + self.num_latent_tokens
                
                # 所有batch的序列长度相同
                seq_lens = torch.full((B,), batch_seq_len, device=x.device, dtype=torch.int32)
                cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=x.device), seq_lens.cumsum(0)])
        
        for layer_num, blk in enumerate(self.model.blocks):
            if self.grad_checkpointing and self.training:
                x = checkpoint(
                    blk.__call__, x, cu_seqlens, None, position_embeddings, use_reentrant=False
                )
            else:
                x = blk(x, cu_seqlens, position_embeddings=position_embeddings)
    
        # 从每个batch中取出对应的latent tokens
        if self.use_lq_condition and lq_cond is not None:
            # 使用lq条件：latent tokens在lq tokens之后
            if self.process_mode == 'native':
                latent_tokens_list = []
                for i in range(len(seq_lens)):
                    batch_seq_len = seq_lens[i]
                    hq_seq_len = image_seq_lens[i]
                    lq_seq_len = lq_seq_lens[i]
                    
                    # latent tokens在每个batch的末尾（在lq tokens之后）
                    start_idx = cu_seqlens[i] + hq_seq_len + lq_seq_len
                    end_idx = cu_seqlens[i + 1]
                    
                    batch_latent_tokens = x[start_idx:end_idx]
                    latent_tokens_list.append(batch_latent_tokens)
                
                x = torch.cat(latent_tokens_list, dim=0)
            else:
                x = x.reshape(B, -1, self.hidden_size)
                # latent tokens在每个batch的末尾（在lq tokens之后）
                x = x[:, -self.num_latent_tokens:, :]
                x = x.reshape(-1, self.hidden_size)
        else:
            # 原始方式：latent tokens在hq tokens之后
            if self.process_mode == 'native':
                # 根据cu_seqlens确定每个batch的latent tokens位置
                latent_tokens_list = []
                for i in range(len(seq_lens)):
                    # 每个batch的序列长度 = image_seq_len + latent_seq_len
                    batch_seq_len = seq_lens[i]
                    image_seq_len = image_seq_lens[i]
                    
                    # latent tokens在每个batch的末尾
                    start_idx = cu_seqlens[i] + image_seq_len
                    end_idx = cu_seqlens[i + 1]
                    
                    # 取出当前batch的latent tokens
                    batch_latent_tokens = x[start_idx:end_idx]
                    latent_tokens_list.append(batch_latent_tokens)
                
                # 将所有batch的latent tokens拼接成一个batch
                x = torch.cat(latent_tokens_list, dim=0)  # [B * num_latent_tokens, hidden_size]
            else:
                # batch style: 直接reshape为batch形式，然后取出latent tokens部分
                x = x.reshape(B, -1, self.hidden_size)  # [B, seq_len, hidden_size]
                # latent tokens在每个batch的末尾
                x = x[:, -self.num_latent_tokens:, :]  # [B, num_latent_tokens, hidden_size]
                x = x.reshape(-1, self.hidden_size)  # [B * num_latent_tokens, hidden_size]
        
        x = self.final_layer(x)    # [B * num_latent_tokens, token_size]
      
        # 使用reshape而不是view，因为tensor可能不是连续的
        x = x.reshape(B, self.token_size, 1, self.num_latent_tokens)
        return x




# --- NiTDecoder ---
class NiTDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        vq_model = config.model.vq_model
     
        # --- 支持字符串配置 ---
        model_size = vq_model.get('vit_dec_model_size', 'large')
        width_map = {"small": 512, "base": 768, "large": 1024}
        depth_map = {"small": 8, "base": 12, "large": 24}
        num_heads_map = {"small": 8, "base": 12, "large": 16}
        hidden_size = vq_model.get('hidden_size', width_map.get(model_size, 1024))
        depth = vq_model.get('vit_dec_depth', depth_map.get(model_size, 24))
        num_heads = vq_model.get('vit_dec_num_heads', num_heads_map.get(model_size, 16))
        self.depth = depth
        self.num_heads = num_heads
        # --- 其他参数 ---
        self.num_latent_tokens = vq_model.get('num_latent_tokens', 32)
        self.token_size = vq_model.get('token_size', 64)
        patch_size = vq_model.get('vit_dec_patch_size', 16)
        out_channels = vq_model.get('out_channels', 3)
        token_size = vq_model.get('token_size', 64)
        mlp_ratio = vq_model.get('mlp_ratio', 4.0)
        theta = vq_model.get('theta', 10000)
        # 添加add_latent_proj配置支持
        add_latent_proj = vq_model.get('add_latent_proj', False)
        attn_mode = vq_model.get('attn_mode', 'cross')
        self.add_latent_proj = add_latent_proj  # 保存配置

        # 添加attention模式配置
        self.attn_mode = attn_mode
        block_kwargs = vq_model.get('block_kwargs', {})
        grad_checkpointing = vq_model.get('grad_checkpointing', True)
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.grad_checkpointing = grad_checkpointing
        self.t_embedder = TimestepEmbedder(hidden_size)
        head_dim = hidden_size // num_heads
        axes_dim = [head_dim // 8, head_dim * 7 // 16, head_dim * 7 // 16]
        self.head_dim = head_dim
        self.theta = theta
        self.axes_dim = axes_dim
        self.rope = FluxRope(theta=theta, axes_dim=axes_dim)
        self.blocks = nn.ModuleList([
            NiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, add_latent_proj=add_latent_proj, attn_mode=attn_mode, **block_kwargs) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.patch_size, out_channels)
        self.decoder_embed = nn.Linear(token_size, hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.initialize_weights()
    
    def freeze_except_latent_proj(self):
        """
        冻结除了add_latent_proj相关参数以外的所有参数
        这样只训练latent projection相关的参数，其他参数保持冻结状态
        """
        # if not self.add_latent_proj:
        #     print("Warning: add_latent_proj is False, no latent projection parameters to unfreeze")
        #     return
        
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
        
        # 解冻latent projection相关的参数
        for block in self.blocks:
            if hasattr(block.attn, 'to_q_latent'):
                block.attn.to_q_latent.weight.requires_grad = True
                if block.attn.to_q_latent.bias is not None:
                    block.attn.to_q_latent.bias.requires_grad = True
            if hasattr(block.attn, 'to_k_latent'):
                block.attn.to_k_latent.weight.requires_grad = True
                if block.attn.to_k_latent.bias is not None:
                    block.attn.to_k_latent.bias.requires_grad = True
            if hasattr(block.attn, 'to_v_latent'):
                block.attn.to_v_latent.weight.requires_grad = True
                if block.attn.to_v_latent.bias is not None:
                    block.attn.to_v_latent.bias.requires_grad = True
            if hasattr(block.attn, 'q_norm_latent'):
                if hasattr(block.attn.q_norm_latent, 'weight'):
                    block.attn.q_norm_latent.weight.requires_grad = True
                if hasattr(block.attn.q_norm_latent, 'bias'):
                    block.attn.q_norm_latent.bias.requires_grad = True
            if hasattr(block.attn, 'k_norm_latent'):
                if hasattr(block.attn.k_norm_latent, 'weight'):
                    block.attn.k_norm_latent.weight.requires_grad = True
                if hasattr(block.attn.k_norm_latent, 'bias'):
                    block.attn.k_norm_latent.bias.requires_grad = True
            if hasattr(block.attn, 'proj_latent'):
                block.attn.proj_latent.weight.requires_grad = True
                if block.attn.proj_latent.bias is not None:
                    block.attn.proj_latent.bias.requires_grad = True
            if hasattr(block.attn, 'norm_latent'):
                if hasattr(block.attn.norm_latent, 'weight'):
                    block.attn.norm_latent.weight.requires_grad = True
                if hasattr(block.attn.norm_latent, 'bias'):
                    block.attn.norm_latent.bias.requires_grad = True
            if hasattr(block.attn, 'proj_drop_latent'):
                # Dropout层没有可训练参数，跳过
                pass
        
        print(f"Frozen all parameters except latent projection parameters in {len(self.blocks)} blocks")
        
        # 打印可训练参数数量
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100.0 * trainable_params / total_params:.2f}%)")
    
    def unfreeze_all(self):
        """
        解冻所有参数
        """
        for param in self.parameters():
            param.requires_grad = True
        print("Unfroze all parameters")
    
    # --- 权重初始化 ---
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.normal_(self.final_layer.adaLN_modulation[-1].weight, std=0.02)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    # --- 前向传播 ---
    def forward(self, z_quantized):
        x = self.decoder_embed(z_quantized)
        batch_size = x.shape[0]
        t = torch.zeros(batch_size, x.shape[1], device=x.device, dtype=x.dtype)
        c = self.t_embedder(t)
        
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.blocks:
                x = checkpoint(block, x, c, None, None, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x, c, None, None)
                
        x = self.final_layer(x, c)
        return x



class SRTiTokDecoder(NiTDecoder):
    def __init__(self, config):
        super().__init__(config)
        scale = self.hidden_size ** -0.5
        self.scale = scale
        # self.lq_token_size = config.model.vq_model.get("lq_token_size", self.token_size)
        self.sr_h = config.experiment.get("downsample_factor", 2) / config.experiment.get("upsample_factor", 1)
        self.sr_w = config.experiment.get("downsample_factor", 2) / config.experiment.get("upsample_factor", 1)
        self.downsample_factor = config.experiment.get("downsample_factor", 2) / config.experiment.get("upsample_factor", 1)
        vq_model = config.model.vq_model
        input_size = 256
        encode_patch_size = vq_model.get('vit_enc_patch_size', 16)
        decode_patch_size = vq_model.get('vit_dec_patch_size', 16)
        self.drop_latent_tokens = vq_model.get('drop_latent_tokens', False)
        in_channels = vq_model.get('in_channels', 3)
        self.patch_size = decode_patch_size
        self.encode_patch_size = encode_patch_size
        self.lq_decoder_embed = PatchEmbed(
            input_size, encode_patch_size, in_channels, self.hidden_size, bias=True, strict_img_size=False
        )
        # c 条件来源配置："t_embedder" 或 "latent"
        vq_model = config.model.vq_model
        self.decoder_c_mode = vq_model.get('decoder_c_mode', 'latent')
        if self.decoder_c_mode not in ["t_embedder", "latent"]:
            raise ValueError(f"Invalid decoder_c_mode: {self.decoder_c_mode}. Must be 't_embedder' or 'latent'.")
        # 当使用 latent 作为条件时，需要把 (B, L_latent * C) 投影到 (B, C)
        if self.decoder_c_mode == "latent":
            self.c_from_latent_proj = nn.Linear(self.num_latent_tokens * self.hidden_size, self.hidden_size, bias=True)
     
    def forward(self, latent_tokens, lq_cond, target_shape=None):
        # 检查lq_cond是否为None
        if lq_cond is None:
            raise ValueError("SRTiTokDecoder requires lq_cond to be provided. Cannot be None.")
        
        H_lq = lq_cond.shape[2] // self.encode_patch_size
        W_lq = lq_cond.shape[3] // self.encode_patch_size
        lq_tokens = self.lq_decoder_embed(lq_cond)
        
        B, C, H, W = latent_tokens.shape
        # assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        L_latent = H * W
        latent_tokens = latent_tokens.reshape(B, C*H, W).permute(0, 2, 1) # B, L_latent, C
        
        latent_tokens = self.decoder_embed(latent_tokens)
        
        # 如果提供了target_shape，使用它来计算mask token数量
        if target_shape is not None:
            H_hq, W_hq = target_shape[0] // self.encode_patch_size, target_shape[1] // self.encode_patch_size
        else:
            # 否则回退到原来的逻辑，根据lq_cond和sr_h/sr_w计算
            H_hq = int(H_lq * self.sr_h)
            W_hq = int(W_lq * self.sr_w)
     
        num_mask_tokens = H_hq * W_hq

        mask_tokens = self.mask_token.repeat(B, num_mask_tokens, 1).to(lq_tokens.dtype)
        x = mask_tokens
        # 计算downsample_factor用于ROPE位置编码
        downsample_factor = self.sr_h  # 使用配置中的下采样倍率
        freqs_cos, freqs_sin = self.rope([H_hq, W_hq], [H_lq, W_lq], L_latent, B, lq_tokens.device, lq_tokens.dtype, downsample_factor)
        # 生成条件 c：根据 decoder_c_mode 选择
        if self.decoder_c_mode == "t_embedder":
            c = self.t_embedder(torch.zeros(B, device=x.device, dtype=x.dtype))
        else:
            # 使用 latent_tokens 生成 c
            # 当前 latent_tokens 形状为 [B, L_latent, hidden_size]
            c_flat = latent_tokens.reshape(B, -1)
            c = self.c_from_latent_proj(c_flat).to(x.dtype)
        
        if self.attn_mode != "cross":
            if self.add_latent_proj:
                x = torch.cat([x, lq_tokens], dim=1)
            else:
                if self.drop_latent_tokens:
                    N_latent = latent_tokens.shape[1]
                    x = torch.cat([x, lq_tokens], dim=1)
                    freqs_cos = freqs_cos[:, :, :-N_latent]
                    freqs_sin = freqs_sin[:, :, :-N_latent]
                else:
                    x = torch.cat([x, lq_tokens, latent_tokens], dim=1)
        c = c.unsqueeze(1).expand(-1, x.shape[1], -1)


        for block in self.blocks:
            # assert block.attn_mode == "cross", "SRTiTokDecoder is not supported for cross attention mode"
            # assert block.add_latent_proj == True, "SRTiTokDecoder is not supported for add_latent_proj mode"
            if self.grad_checkpointing:
                if self.attn_mode == "cross":
                    result = checkpoint(block, x, c, freqs_cos, freqs_sin, lq_tokens, latent_tokens, self.drop_latent_tokens, use_reentrant=False)
                elif self.add_latent_proj:
                    result = checkpoint(block, x, c, freqs_cos, freqs_sin, latent_tokens, self.drop_latent_tokens, use_reentrant=False)
                else:
                    result = checkpoint(block, x, c, freqs_cos, freqs_sin, use_reentrant=False)
            else:
                if self.attn_mode == "cross":
                    result = block(x, c, freqs_cos, freqs_sin, lq_tokens, latent_tokens, self.drop_latent_tokens)
                elif self.add_latent_proj:
                    result = block(x, c, freqs_cos, freqs_sin, latent_tokens, self.drop_latent_tokens)
                else:
                    result = block(x, c, freqs_cos, freqs_sin)
            
            # 统一处理返回值：根据返回类型自动解包
            if isinstance(result, tuple):
                if len(result) == 3:
                    x, lq_tokens, latent_tokens = result
                elif len(result) == 2:
                    x, latent_tokens = result
                else:
                    x = result[0]  # 取第一个元素作为x
            else:
                x = result
     
                
        x = x[:, :num_mask_tokens]
        c = c[:, :num_mask_tokens]
        x = self.final_layer(x, c)
        x = x.reshape(B, H_hq , W_hq, self.out_channels, self.patch_size, self.patch_size).permute(0, 3, 1, 4, 2, 5).reshape(B, self.out_channels, H_hq * self.patch_size, W_hq * self.patch_size)
        return x

class SRMMDITEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        vq_model = config.model.vq_model
        model_size = vq_model.get('vit_dec_model_size', 'large')
        width_map = {"small": 512, "base": 768, "large": 1024}
        depth_map = {"small": 8, "base": 12, "large": 24}
        num_heads_map = {"small": 8, "base": 12, "large": 16}
        hidden_size = vq_model.get('hidden_size', width_map.get(model_size, 1024))
        depth = vq_model.get('vit_dec_depth', depth_map.get(model_size, 24))
        num_heads = vq_model.get('vit_dec_num_heads', num_heads_map.get(model_size, 16))
        # --- 其他参数 ---
        patch_size = vq_model.get('vit_enc_patch_size', 16)
        in_channels = vq_model.get('in_channels', 3)
        token_size = vq_model.get('token_size', 64)

        num_latent_tokens = vq_model.get('num_latent_tokens', 32)
        theta = vq_model.get('theta', 10000)
        block_kwargs = vq_model.get('block_kwargs', {})
        grad_checkpointing = vq_model.get('grad_checkpointing', True)
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.grad_checkpointing = grad_checkpointing
        self.t_embedder = TimestepEmbedder(hidden_size)
        head_dim = hidden_size // num_heads
        input_size = 256
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=False
        )
        self.num_latent_tokens = num_latent_tokens
        self.blocks = nn.ModuleList([
            FluxTransformerBlock(hidden_size, num_heads, head_dim) 
            for _ in range(depth)
        ])
        self.theta = theta
        axes_dim = [head_dim // 8, head_dim * 7 // 16, head_dim * 7 // 16]
        self.token_size = token_size
        if config.model.vq_model.get("quantize_mode", "vq") == "vae":
            self.token_size = self.token_size * 2 # needs to split into mean and std
        self.use_lq_condition = vq_model.get('use_lq_condition', False)
        self.rope_type = "encode_with_lq" if self.use_lq_condition else "encode"
        self.rope = FluxRope(theta=theta, axes_dim=axes_dim, type=self.rope_type)
        self.axes_dim = axes_dim
       
        self.final_layer = FinalLayer(hidden_size, 1, self.token_size)
        self.decoder_embed = nn.Linear(token_size, hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        scale = self.hidden_size ** -0.5
        self.scale = scale
        # self.lq_token_size = config.model.vq_model.get("lq_token_size", self.token_size)
        self.downsample_factor = config.experiment.get("downsample_factor", 2) / config.experiment.get("upsample_factor", 1)
        input_size = 256
        self.lq_embedder = PatchEmbed(
                input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=False
            )
        self.initialize_weights()
    

    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 权重初始化
        for block in self.blocks:
            # AdaLayerNormZero的权重初始化
            if hasattr(block.norm1, 'weight') and block.norm1.weight is not None:
                nn.init.constant_(block.norm1.weight, 0)
            if hasattr(block.norm1_context, 'weight') and block.norm1_context.weight is not None:
                nn.init.constant_(block.norm1_context.weight, 0)
            if hasattr(block.norm2, 'weight') and block.norm2.weight is not None:
                nn.init.constant_(block.norm2.weight, 0)
            if hasattr(block.norm2_context, 'weight') and block.norm2_context.weight is not None:
                nn.init.constant_(block.norm2_context.weight, 0)

        nn.init.normal_(self.final_layer.adaLN_modulation[-1].weight, std=0.02)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(self, x, latent_tokens, lq_cond=None):
        batch_size = x.shape[0] # x.shape = [batch_size, in_channels, height, hidden_size]
        h, w = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        x = self.x_embedder(x)  # (B, D, H, W) -> (B, C, H // patch_size, W // patch_size)
        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        num_x_tokens = h * w
        # 根据是否实际使用lq条件来决定rope调用方式
        actual_use_lq = self.use_lq_condition and lq_cond is not None
        if self.use_lq_condition:
            assert lq_cond is not None, "lq_cond is required when use_lq_condition is True"
        if actual_use_lq:
            # 使用lq条件
            h_lq, w_lq = lq_cond.shape[2] // self.patch_size, lq_cond.shape[3] // self.patch_size
            lq_tokens = self.lq_embedder(lq_cond)  # (B, C, H_lq // patch_size, W_lq // patch_size)
            lq_tokens = lq_tokens.reshape(batch_size, -1, self.hidden_size)  # (B, H_lq*W_lq, C)
            x = x.reshape(batch_size, -1, self.hidden_size)  # (B, H*W, C)
            x = torch.cat([x, lq_tokens], dim=1)
            freqs_cos, freqs_sin = self.rope([h, w], [h_lq, w_lq], latent_tokens.shape[1], batch_size, x.device, x.dtype, self.downsample_factor)
        else:
            # 原始的两段位置编码：hq/latent
            x = x.reshape(batch_size, -1, self.hidden_size)  # (B, H*W, C)
            # 如果rope是encode_with_lq类型，需要创建一个临时的encode rope
            if self.rope_type == "encode_with_lq":
                # 创建临时的encode rope
                temp_rope = FluxRope(theta=self.rope.theta, axes_dim=self.rope.axes_dim, type="encode")
                freqs_cos, freqs_sin = temp_rope([h, w], latent_tokens.shape[1], batch_size, x.device, x.dtype, self.downsample_factor)
            else:
                freqs_cos, freqs_sin = self.rope([h, w], latent_tokens.shape[1], batch_size, x.device, x.dtype, self.downsample_factor)
        # 计算downsample_factor用于ROPE位置编码
        rotary_emb = (freqs_cos[0, 0, :], freqs_sin[0, 0, :]) # [L, C]
        c = self.t_embedder(torch.zeros(batch_size, device=x.device, dtype=x.dtype))
        c_final = c.unsqueeze(1).expand(-1, latent_tokens.shape[1], -1) # [B, L, C]
        for block in self.blocks:
            if self.grad_checkpointing:
                x, latent_tokens = checkpoint(block, latent_tokens, x, c, rotary_emb, use_reentrant=False)
            else:
                x, latent_tokens = block(latent_tokens, x, c, rotary_emb)

        latent_tokens = self.final_layer(latent_tokens, c_final).permute(0, 2, 1)
        latent_tokens = latent_tokens.reshape(batch_size, self.token_size, 1, self.num_latent_tokens)
        return latent_tokens






class SRMMDITDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        vq_model = config.model.vq_model
        model_size = vq_model.get('vit_dec_model_size', 'large')
        width_map = {"small": 512, "base": 768, "large": 1024}
        depth_map = {"small": 8, "base": 12, "large": 24}
        num_heads_map = {"small": 8, "base": 12, "large": 16}
        hidden_size = vq_model.get('hidden_size', width_map.get(model_size, 1024))
        depth = vq_model.get('vit_dec_depth', depth_map.get(model_size, 24))
        num_heads = vq_model.get('vit_dec_num_heads', num_heads_map.get(model_size, 16))
        # --- 其他参数 ---
        self.num_latent_tokens = vq_model.get('num_latent_tokens', 32)
        self.token_size = vq_model.get('token_size', 64)
        patch_size = vq_model.get('vit_dec_patch_size', 16)
        out_channels = vq_model.get('out_channels', 3)
        token_size = vq_model.get('token_size', 64)
        theta = vq_model.get('theta', 10000)
        grad_checkpointing = vq_model.get('grad_checkpointing', True)
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.grad_checkpointing = grad_checkpointing
        self.t_embedder = TimestepEmbedder(hidden_size)
        head_dim = hidden_size // num_heads
        
        self.blocks = nn.ModuleList([
            FluxTransformerBlock(hidden_size, num_heads, head_dim) 
            for _ in range(depth)
        ])
        self.theta = theta
        axes_dim = [head_dim // 8, head_dim * 7 // 16, head_dim * 7 // 16]
        self.rope = FluxRope(theta=theta, axes_dim=axes_dim, type="decode_wo_lq")
        self.axes_dim = axes_dim
       
        self.final_layer = FinalLayer(hidden_size, self.patch_size, out_channels)
        self.decoder_embed = nn.Linear(token_size, hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        scale = self.hidden_size ** -0.5
        self.scale = scale
        # self.lq_token_size = config.model.vq_model.get("lq_token_size", self.token_size)
        self.sr_h = config.experiment.get("downsample_factor", 2) / config.experiment.get("upsample_factor", 1)
        self.sr_w = config.experiment.get("downsample_factor", 2) / config.experiment.get("upsample_factor", 1)
        input_size = 256
        self.lq_decoder_embed = PatchEmbed(
            input_size, self.patch_size, self.out_channels, self.hidden_size, bias=True, strict_img_size=False
        )
        self.initialize_weights()
    

    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 权重初始化
        for block in self.blocks:
            # AdaLayerNormZero的权重初始化
            if hasattr(block.norm1, 'weight') and block.norm1.weight is not None:
                nn.init.constant_(block.norm1.weight, 0)
            if hasattr(block.norm1_context, 'weight') and block.norm1_context.weight is not None:
                nn.init.constant_(block.norm1_context.weight, 0)
            if hasattr(block.norm2, 'weight') and block.norm2.weight is not None:
                nn.init.constant_(block.norm2.weight, 0)
            if hasattr(block.norm2_context, 'weight') and block.norm2_context.weight is not None:
                nn.init.constant_(block.norm2_context.weight, 0)

        nn.init.normal_(self.final_layer.adaLN_modulation[-1].weight, std=0.02)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(self, latent_tokens, lq_cond, target_shape=None):
        # 检查lq_cond是否为None
        if lq_cond is None:
            raise ValueError("SRTiTokDecoder requires lq_cond to be provided. Cannot be None.")
        
        H_lq = lq_cond.shape[2] // self.patch_size
        W_lq = lq_cond.shape[3] // self.patch_size
        lq_tokens = self.lq_decoder_embed(lq_cond)
        latent_tokens = latent_tokens.to(lq_tokens.dtype)
    
        B, C, H, W = latent_tokens.shape # B, C, 1, L
        # assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        L_latent = H * W
        latent_tokens = latent_tokens.reshape(B, C*H, W).permute(0, 2, 1) # B, L_latent, C
        latent_tokens = self.decoder_embed(latent_tokens)
        
        # 如果提供了target_shape，使用它来计算mask token数量
        if target_shape is not None:
            H_hq, W_hq = target_shape[0] // self.patch_size, target_shape[1] // self.patch_size
        else:
            # 否则回退到原来的逻辑，根据lq_cond和sr_h/sr_w计算
            H_hq = H_lq * self.sr_h
            W_hq = W_lq * self.sr_w
            
        num_mask_tokens = H_hq * W_hq
        # mask_tokens = self.mask_token.repeat(B, num_mask_tokens, 1).to(lq_tokens.dtype)
        # 计算downsample_factor用于ROPE位置编码
        freqs_cos, freqs_sin = self.rope([H_lq, W_lq], L_latent, B, lq_tokens.device, lq_tokens.dtype, diff_first=True)
        rotary_emb = (freqs_cos[0, 0, :], freqs_sin[0, 0, :]) # [L, C]
        # x = torch.cat([mask_tokens, lq_tokens], dim=1)
        x = lq_tokens
        c = self.t_embedder(torch.zeros(B, device=x.device, dtype=x.dtype))
        c_final = c.unsqueeze(1).expand(-1, x.shape[1], -1) # [B, L, C]
        for block in self.blocks:
            if self.grad_checkpointing:
                latent_tokens, x = checkpoint(block, x, latent_tokens, c, rotary_emb, use_reentrant=False)
            else:
                latent_tokens, x = block(x, latent_tokens, c, rotary_emb)

        x = x[:, :num_mask_tokens]
        c_final = c_final[:, :num_mask_tokens]
        x = self.final_layer(x, c_final)
        x = x.reshape(B, H_hq , W_hq, self.out_channels, self.patch_size, self.patch_size).permute(0, 3, 1, 4, 2, 5).reshape(B, self.out_channels, H_hq * self.patch_size, W_hq * self.patch_size)
        return x

class DiffSRTiTokDecoder(SRTiTokDecoder):
    def __init__(self, config):
        super().__init__(config)
        self.rope = FluxRope(theta=self.theta, axes_dim=self.axes_dim, type="decode_wo_lq")
    
    def forward(self, latent_tokens, lq_cond, target_shape=None):
               # 检查lq_cond是否为None
        if lq_cond is None:
            raise ValueError("SRTiTokDecoder requires lq_cond to be provided. Cannot be None.")
        
        latent_tokens = latent_tokens.to(lq_cond.dtype)
    
        B, C, H, W = latent_tokens.shape
        # assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        L_latent = H * W
        latent_tokens = latent_tokens.reshape(B, C*H, W).permute(0, 2, 1) # B, L_latent, C
        latent_tokens = self.decoder_embed(latent_tokens)
        
        # 如果提供了target_shape，使用它来计算mask token数量
        H_hq, W_hq = target_shape[0] // self.patch_size, target_shape[1] // self.patch_size
        
            
        num_mask_tokens = H_hq * W_hq
        mask_tokens = self.mask_token.repeat(B, num_mask_tokens, 1).to(lq_cond.dtype)
        x = mask_tokens
        # 计算downsample_factor用于ROPE位置编码
        downsample_factor = self.sr_h  # 使用配置中的下采样倍率
        freqs_cos, freqs_sin = self.rope([H_hq, W_hq], L_latent, B, lq_cond.device, lq_cond.dtype)
   
        c = self.t_embedder(torch.zeros(B, device=x.device, dtype=x.dtype))
        c = c.unsqueeze(1).expand(-1, x.shape[1], -1)
        
        for block in self.blocks:
            # assert block.attn_mode == "cross", "SRTiTokDecoder is not supported for cross attention mode"
            # assert block.add_latent_proj == True, "SRTiTokDecoder is not supported for add_latent_proj mode"
            if self.grad_checkpointing:
                if self.attn_mode == "cross":
                    result = checkpoint(block, x, c, freqs_cos, freqs_sin, None, latent_tokens, use_reentrant=False)
                elif self.add_latent_proj:
                    result = checkpoint(block, x, c, freqs_cos, freqs_sin, latent_tokens, use_reentrant=False)
                else:
                    x = torch.cat([x, latent_tokens], dim=1)
                    result = checkpoint(block, x, c, freqs_cos, freqs_sin, use_reentrant=False)
            else:
                if self.attn_mode == "cross":
                    result = block(x, c, freqs_cos, freqs_sin, None, latent_tokens)
                elif self.add_latent_proj:
                    result = block(x, c, freqs_cos, freqs_sin, latent_tokens)
                else:
                    result = block(x, c, freqs_cos, freqs_sin)
            
            # 统一处理返回值：根据返回类型自动解包
            if isinstance(result, tuple):
                x, latent_tokens = result
            else:
                x = result
     
                
        x = x[:, :num_mask_tokens]
        c = c[:, :num_mask_tokens]
        x = self.final_layer(x, c)
        x = x.reshape(B, H_hq , W_hq, self.out_channels, self.patch_size, self.patch_size).permute(0, 3, 1, 4, 2, 5).reshape(B, self.out_channels, H_hq * self.patch_size, W_hq * self.patch_size)
        # lq_cond_up = F.interpolate(lq_cond, size=(H_hq * self.patch_size, W_hq * self.patch_size), mode='bilinear', align_corners=False, antialias=True)
        if self.downsample_factor > 1:
            lq_cond_up = F.interpolate(lq_cond, size=(H_hq * self.patch_size, W_hq * self.patch_size), mode='bilinear', align_corners=False, antialias=True)
            x = x + lq_cond_up
        else:
            x = x + lq_cond
        
        return x



class DiffMMDITDecoder(SRMMDITDecoder):
    def __init__(self, config):
        super().__init__(config)
        self.rope = FluxRope(theta=self.theta, axes_dim=self.axes_dim, type="decode_wo_lq")
    
  
    def forward(self, latent_tokens, lq_cond, target_shape=None):
        B, C, H, W = latent_tokens.shape
        # assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        L_latent = H * W
        latent_tokens = latent_tokens.reshape(B, C*H, W).permute(0, 2, 1) # B, L_latent, C
        latent_tokens = self.decoder_embed(latent_tokens)
        H_hq = target_shape[0] // self.patch_size
        W_hq = target_shape[1] // self.patch_size
        num_mask_tokens = H_hq * W_hq
        mask_tokens = self.mask_token.repeat(B, num_mask_tokens, 1).to(latent_tokens.dtype)
        freqs_cos, freqs_sin = self.rope([H_hq, W_hq], L_latent, B, latent_tokens.device, latent_tokens.dtype, diff_first=True)
        rotary_emb = (freqs_cos[0, 0, :], freqs_sin[0, 0, :]) # [L, C]
        x = mask_tokens
        c = self.t_embedder(torch.zeros(B, device=x.device, dtype=x.dtype))
        c_final = c.unsqueeze(1).expand(-1, x.shape[1], -1)
        for block in self.blocks:
            if self.grad_checkpointing:
                latent_tokens, x = checkpoint(block, x, latent_tokens, c, rotary_emb, use_reentrant=False)
            else:
                latent_tokens, x = block(x, latent_tokens, c, rotary_emb)
        
        x = x[:, :num_mask_tokens]
        c_final = c_final[:, :num_mask_tokens]
        x = self.final_layer(x, c_final)
        x = x.reshape(B, H_hq , W_hq, self.out_channels, self.patch_size, self.patch_size).permute(0, 3, 1, 4, 2, 5).reshape(B, self.out_channels, H_hq * self.patch_size, W_hq * self.patch_size)
        lq_cond_up = F.interpolate(lq_cond, size=(H_hq * self.patch_size, W_hq * self.patch_size), mode='bilinear', align_corners=False, antialias=True)
        x = x + lq_cond_up
        return x

from diffusers import SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig
class SD3Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = SD3Transformer2DModel.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium",
            subfolder="transformer")
        self.transformer.enable_gradient_checkpointing()
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="scheduler")
        self.scheduler.timesteps = self.scheduler.timesteps.cuda()
        self.scheduler.sigmas = self.scheduler.sigmas.cuda()
        
        # 检查是否启用LoRA
        self.use_lora = getattr(config.model.vq_model, 'use_lora', False)
        if self.use_lora:
            target_modules = [
                # 核心注意力模块
                "to_q", "to_k", "to_v", "to_out.0",
                
                # SD3 特有的额外注意力模块
                "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
                
                # 第二个注意力块（文本条件）
                "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
                
                # 前馈网络模块
                "ff.net.0.proj", "ff.net.2",
                "ff_context.net.0.proj", "ff_context.net.2",
                
                # 嵌入和投影模块（类似 Flux 的 x_embedder）
                "pos_embed.proj", "context_embedder", "proj_out"
            ]
            
            # 获取LoRA配置参数，如果没有配置则使用默认值
            lora_rank = getattr(config.model.vq_model, 'lora_rank', 512)
            lora_alpha = getattr(config.model.vq_model, 'lora_alpha', 512)
            
            G_lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            self.transformer.add_adapter(G_lora_cfg)
            lora_params = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))
            assert lora_params, "Failed to find lora parameters"
            for p in lora_params:
                p.data = p.to(torch.float32)
            print(f"LoRA enabled with rank={lora_rank}, alpha={lora_alpha}")
        else:
            print("LoRA disabled")
            
        self.token_size = config.model.vq_model.get('token_size', 64)
        self.num_latent_tokens = config.model.vq_model.get('num_latent_tokens', 1024)
        self.hidden_size = 4096
        self.decoder_embed = torch.nn.Linear(self.token_size, self.hidden_size)
        self.empty_pooled_embed = torch.zeros(1, 2048)
    
    def freeze_except_latent_proj(self):
        """
        冻结除了add_latent_proj相关参数以外的所有参数
        这样只训练latent projection相关的参数，其他参数保持冻结状态
        """
        for param in self.parameters():
            param.requires_grad = False
        self.decoder_embed.requires_grad = True

    def forward(self, latent_tokens, lq_cond, target_shape=None):
        B, C, H, W = latent_tokens.shape
        latent_tokens = latent_tokens.reshape(B, C*H, W).permute(0, 2, 1) # B, L_latent, C
        latent_tokens = self.decoder_embed(latent_tokens)
        x = lq_cond
        # current_timestep = torch.zeros(B, device=x.device, dtype=x.dtype)
        current_timestep = torch.ones(B, device=x.device, dtype=x.dtype) * 500
        # try:
        # 将batch timestep转换为标量值，因为scheduler.index_for_timestep期望标量输入
        timestep_scalar = current_timestep[0].item()
        sigma_idx = self.scheduler.index_for_timestep(timestep_scalar)
        current_sigma = self.scheduler.sigmas[sigma_idx]
        # except Exception as e:
        #     # 如果index_for_timestep失败，使用最近邻查找
        #     distances = torch.abs(self.scheduler.timesteps - current_timestep)
        #     sigma_idx = torch.argmin(distances)
        #     current_sigma = self.scheduler.sigmas[sigma_idx]
        pooled_text_embed = self.empty_pooled_embed.repeat(B, 1).to(x.device).to(x.dtype)
        eps = self.transformer(
            hidden_states=x,
            timestep=current_timestep,
            encoder_hidden_states=latent_tokens,
            pooled_projections=pooled_text_embed,
        ).sample

              # 从scheduler中获取sigma_0（最小的sigma值，对应无噪声状态）
        sigma_0 = self.scheduler.sigma_min
        
        # 计算噪声差值
        sigma_diff = sigma_0 - current_sigma
        
        # 应用一步去噪公式: z = z_in + (sigma_0 - sigma_t) * eps
        x = lq_cond + sigma_diff * eps
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, 1, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y, use_reentrant=False)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

