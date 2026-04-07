# sd3_transformer_wrapper.py
# -*- coding: utf-8 -*-

import copy
from typing import Optional

import torch
import torch.nn as nn

from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import PatchEmbed


class SD3TransformerWrapper(nn.Module):
    """
    SD3 Transformer Wrapper for Latent Token Processing

    这个 wrapper 扩展现有的 SD3Transformer2DModel，添加对 latent token 处理的支持。
    它保持与原始模型的兼容性，同时提供三路分支（text / latent / image）的处理能力。
    - 为每个 block 添加 latent 分支的 norm/ff 参数：norm1_latent、norm2_latent、ff_latent
    - 为 Attention 添加 latent 分支的 QKV 与输出投影：latent_q_proj / latent_k_proj / latent_v_proj / latent_out_proj
    - 在 Transformer 级别添加：latent_pos_embed、latent_norm_out、latent_proj_out
    """

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        add_latent_params: bool = True,
        latent_dim: Optional[int] = None,
        token_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            transformer: 原始的 SD3Transformer2DModel 实例
            add_latent_params: 是否添加 latent 分支的专用参数
            latent_dim: latent 特征维度；默认取 transformer.inner_dim
            token_size: SRTokenizer 的 token 通道数（等价于 C）；默认 64（请与上游 SRTokenizerIntegration 一致）
            device: 期望放置的设备
            dtype: 期望的 dtype
        """
        super().__init__()
        self.transformer = transformer

        # 推断维度
        self.latent_dim = latent_dim or getattr(self.transformer, "inner_dim", None)
        if self.latent_dim is None:
            # 兜底：尝试从第一个 block 的 FFN 推断
            first_block = getattr(self.transformer, "transformer_blocks", [None])[0]
            if first_block is None or not hasattr(first_block, "ff"):
                raise ValueError("无法从 transformer 推断 latent_dim，请显式传入 latent_dim。")
            # ff is typically a GEGLU/SiLU MLP;其最后一层 out features = inner_dim
            self.latent_dim = first_block.ff.net[-1].out_features

        # 记录 token_size（对应 latent token 的通道数）
        self.token_size = token_size or 64

        # 添加 latent 分支的专用参数
        if add_latent_params:
            self._add_latent_parameters()

        # 设备与 dtype 迁移（一次性、安全写法）
        if (device is not None) or (dtype is not None):
            self.to(device=device, dtype=dtype)

    # --------------------------- private helpers ---------------------------

    def _add_latent_parameters(self):
        """为每个 transformer block 及 transformer 顶层添加 latent 分支相关参数。"""

        # 逐 block 添加
        for i, block in enumerate(self.transformer.transformer_blocks):
            # 1) latent 分支的 norm1 / norm2 / ff
            if not hasattr(block, "norm1_latent"):
                # norm1_latent 的结构/参数与 norm1 一致
                block.norm1_latent = copy.deepcopy(block.norm1)

            if not hasattr(block, "norm2_latent"):
                # norm2_latent 的结构/参数与 norm2 一致
                block.norm2_latent = copy.deepcopy(block.norm2)

            if not hasattr(block, "ff_latent"):
                # ff_latent 的结构/参数与 ff 一致
                block.ff_latent = copy.deepcopy(block.ff)

            # 2) 注意力层：为 attn（单路注意力）添加 latent QKV & 输出投影
            if hasattr(block, "attn") and isinstance(block.attn, Attention):
                self._add_attention_latent_params(block.attn)

            # 说明：
            # 早先有人会在 block 里添加 block.latent_out_proj（拷 FFN 的末层），
            # 但我们的主干实现里使用的是 attn.latent_out_proj（拷 to_out），
            # 因此这里不再额外加 block.latent_out_proj 以免混淆。

        # 顶层：为 latent token 的「位置编码器/输入嵌入」、输出归一化与投影添加模块
        if not hasattr(self.transformer, "latent_pos_embed"):
            # 针对 token 格式（B, C=token_size, H=1, W=T）使用 patch_size=1 的 PatchEmbed
            self.transformer.latent_pos_embed = PatchEmbed(
                patch_size=1,
                in_channels=self.token_size,
                embed_dim=self.latent_dim,
            )

        if not hasattr(self.transformer, "latent_norm_out"):
            # 与原始 norm_out 一致
            if hasattr(self.transformer, "norm_out"):
                self.transformer.latent_norm_out = copy.deepcopy(self.transformer.norm_out)
            else:
                raise ValueError("transformer 缺少 norm_out，无法创建 latent_norm_out。")

        if not hasattr(self.transformer, "latent_proj_out"):
            # 末尾把 hidden_dim=latent_dim 映射回 token_size 通道
            self.transformer.latent_proj_out = nn.Linear(self.latent_dim, self.token_size, bias=True)

    def _add_attention_latent_params(self, attn: Attention):
        """
        为 Attention 添加 latent 分支的 Q/K/V 投影与输出投影。
        这些模块的结构参数与原始 to_q/to_k/to_v/to_out 保持一致。
        """
        # Q/K/V
        if not hasattr(attn, "latent_q_proj"):
            if hasattr(attn, "to_q"):
                attn.latent_q_proj = copy.deepcopy(attn.to_q)
            else:
                raise ValueError("Attention 缺少 to_q，无法创建 latent_q_proj。")

        if not hasattr(attn, "latent_k_proj"):
            if hasattr(attn, "to_k"):
                attn.latent_k_proj = copy.deepcopy(attn.to_k)
            else:
                raise ValueError("Attention 缺少 to_k，无法创建 latent_k_proj。")

        if not hasattr(attn, "latent_v_proj"):
            if hasattr(attn, "to_v"):
                attn.latent_v_proj = copy.deepcopy(attn.to_v)
            else:
                raise ValueError("Attention 缺少 to_v，无法创建 latent_v_proj。")

        # 输出投影（注意：主文件里会按 attn.latent_out_proj[0]/[1] 调用）
        if not hasattr(attn, "latent_out_proj"):
            if hasattr(attn, "to_out"):
                attn.latent_out_proj = copy.deepcopy(attn.to_out)
            else:
                raise ValueError("Attention 缺少 to_out，无法创建 latent_out_proj。")

        # 可选：如果希望与 text/image 分支的归一化策略一致，可拷贝 norm_q/k 作为 norm_latent_q/k
        # 主干实现中对此有 fallback（优先用 norm_latent_q/k，否则退回 norm_q/k），因此这一步不是必需。
        if not hasattr(attn, "norm_latent_q") and hasattr(attn, "norm_q"):
            try:
                attn.norm_latent_q = copy.deepcopy(attn.norm_q)
            except Exception:
                # 有些实现的 norm_q 不是独立模块，跳过即可（主逻辑会 fallback 到 norm_q）
                pass

        if not hasattr(attn, "norm_latent_k") and hasattr(attn, "norm_k"):
            try:
                attn.norm_latent_k = copy.deepcopy(attn.norm_k)
            except Exception:
                pass

    
    # def forward(
    #     self,
    #     text_states: Optional[List[torch.FloatTensor]] = None,
    #     latent_states: Optional[List[torch.FloatTensor]] = None,
    #     image_states: Optional[List[torch.FloatTensor]] = None,
    #     pooled_projections: Optional[List[torch.FloatTensor]] = None,
    #     timesteps: Optional[List[torch.LongTensor]] = None,
    #     adapters: Optional[List[Union[str, List[str]]]] = None,
    #     attn_forward: Optional[Callable] = None,
    #     **kwargs
    # ):
    #     """
    #     前向传播，支持三路分支处理
        
    #     Args:
    #         text_states: 文本分支的输入状态
    #         latent_states: latent分支的输入状态（SRTokenizer编码）
    #         image_states: 图像分支的输入状态（条件图像）
    #         pooled_projections: 池化投影
    #         timesteps: 时间步
    #         adapters: 适配器名称
    #         attn_forward: 自定义注意力前向函数
    #         **kwargs: 其他参数
            
    #     Returns:
    #         tuple: (text_outputs, latent_outputs, image_outputs)
    #     """
        
    #     # 计算各分支的数量
    #     txt_n = len(text_states) if text_states is not None else 0
    #     latent_n = len(latent_states) if latent_states is not None else 0
    #     image_n = len(image_states) if image_states is not None else 0
    #     total_branches = txt_n + latent_n + image_n
        
    #     # 设置默认值
    #     if adapters is None:
    #         adapters = [None] * total_branches
    #     if timesteps is None:
    #         timesteps = [torch.zeros(1, dtype=torch.long, device=self.device)] * total_branches
    #     if pooled_projections is None:
    #         pooled_projections = [None] * total_branches
        
    #     # 预处理各分支的输入
    #     text_hidden_states = []
    #     if text_states is not None and len(text_states) > 0:
    #         for text_feature in text_states:
    #             text_hidden_states.append(self.transformer.context_embedder(text_feature))
        
    #     latent_hidden_states = []
    #     if latent_states is not None and len(latent_states) > 0:
    #         latent_embedder = getattr(self.transformer, "latent_pos_embed", self.transformer.pos_embed)
    #         for latent_feature in latent_states:
    #             latent_hidden_states.append(latent_embedder(latent_feature))
        
    #     image_hidden_states = []
    #     if image_states is not None and len(image_states) > 0:
    #         for image_feature in image_states:
    #             image_hidden_states.append(self.transformer.pos_embed(image_feature))
        
    #     # 准备时间嵌入
    #     def get_temb(timestep, pooled_projection):
    #         ref_dtype = None
    #         if text_hidden_states:
    #             ref_dtype = text_hidden_states[0].dtype
    #         elif latent_hidden_states:
    #             ref_dtype = latent_hidden_states[0].dtype
    #         elif image_hidden_states:
    #             ref_dtype = image_hidden_states[0].dtype
    #         else:
    #             ref_dtype = torch.float32
            
    #         timestep = timestep.to(ref_dtype)
    #         if timestep.dim() == 0:
    #             timestep = timestep.unsqueeze(0)
    #         return self.transformer.time_text_embed(timestep, pooled_projection)
        
    #     tembs = [get_temb(*each) for each in zip(timesteps, pooled_projections)]
        
    #     # 通过 transformer blocks
    #     for i, block in enumerate(self.transformer.transformer_blocks):
    #         # 准备 block 参数
    #         block_kwargs = {
    #             "self": block,
    #             "text_states": text_hidden_states,
    #             "latent_states": latent_hidden_states,
    #             "image_states": image_hidden_states,
    #             "tembs": tembs,
    #             "adapters": adapters,
    #             "attn_forward": attn_forward,
    #             **kwargs,
    #         }
            
    #         # 使用梯度检查点（如果启用）
    #         if self.training and self.transformer.gradient_checkpointing:
    #             from torch.utils.checkpoint import checkpoint
    #             gckpt_kwargs = {"use_reentrant": False} if torch.__version__ >= "1.11.0" else {}
    #             text_hidden_states, latent_hidden_states, image_hidden_states = checkpoint(
    #                 self._block_forward, **block_kwargs, **gckpt_kwargs
    #             )
    #         else:
    #             text_hidden_states, latent_hidden_states, image_hidden_states = self._block_forward(**block_kwargs)
        
    #     return text_hidden_states, latent_hidden_states, image_hidden_states
    
    # def _block_forward(
    #     self,
    #     text_states: Optional[List[torch.FloatTensor]] = None,
    #     latent_states: Optional[List[torch.FloatTensor]] = None,
    #     image_states: Optional[List[torch.FloatTensor]] = None,
    #     tembs: Optional[List[torch.FloatTensor]] = None,
    #     adapters: Optional[List[Union[str, List[str]]]] = None,
    #     attn_forward: Optional[Callable] = None,
    #     **kwargs
    # ):
    #     """
    #     单个 transformer block 的前向传播
        
    #     这个方法处理三路分支的注意力计算和融合
    #     """
        
    #     # 计算各分支数量
    #     txt_n = len(text_states) if text_states is not None else 0
    #     latent_n = len(latent_states) if latent_states is not None else 0
    #     image_n = len(image_states) if image_states is not None else 0
        
    #     # 准备输出列表
    #     text_out = []
    #     latent_out = []
    #     image_out = []
        
    #     # 处理各分支的 AdaNorm 和注意力
    #     text_variables = []
    #     latent_variables = []
    #     image_variables = []
        
    #     # Text 分支处理
    #     for i in range(txt_n):
    #         if text_states[i] is not None:
    #             # AdaNorm 计算
    #             norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._compute_adnorm(
    #                 text_states[i], tembs[i], self.norm1_context
    #             )
    #             text_variables.append((norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp))
        
    #     # Latent 分支处理
    #     for i in range(latent_n):
    #         if latent_states[i] is not None:
    #             # AdaNorm 计算
    #             norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._compute_adnorm(
    #                 latent_states[i], tembs[i + txt_n], self.norm1_latent
    #             )
    #             latent_variables.append((norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp))
        
    #     # Image 分支处理
    #     for i in range(image_n):
    #         if image_states[i] is not None:
    #             # AdaNorm 计算
    #             norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._compute_adnorm(
    #                 image_states[i], tembs[i + txt_n + latent_n], self.norm1
    #             )
    #             image_variables.append((norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp))
        
    #     # 注意力计算
    #     if attn_forward is not None:
    #         text_attn_out, latent_attn_out, image_attn_out = attn_forward(
    #             self.attn1,
    #             text_states=text_states,
    #             latent_states=latent_states,
    #             image_states=image_states,
    #             adapters=adapters,
    #             **kwargs
    #         )
    #     else:
    #         # 使用默认的注意力计算
    #         text_attn_out, latent_attn_out, image_attn_out = self._default_attention_forward(
    #             text_states, latent_states, image_states, adapters, **kwargs
    #         )
        
    #     # 融合注意力输出
    #     # Text 分支融合
    #     for i in range(txt_n):
    #         if text_states[i] is not None and i < len(text_attn_out):
    #             _, gate_msa, shift_mlp, scale_mlp, gate_mlp = text_variables[i]
    #             h = text_states[i] + text_attn_out[i] * gate_msa.unsqueeze(1)
    #             norm_h = self.norm2(h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    #             text_out.append(norm_h)
        
    #     # Latent 分支融合
    #     for i in range(latent_n):
    #         if latent_states[i] is not None and i < len(latent_attn_out):
    #             _, gate_msa, shift_mlp, scale_mlp, gate_mlp = latent_variables[i]
    #             h = latent_states[i] + latent_attn_out[i] * gate_msa.unsqueeze(1)
    #             norm_h = self.norm2(h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    #             latent_out.append(norm_h)
        
    #     # Image 分支融合
    #     for i in range(image_n):
    #         if image_states[i] is not None and i < len(image_attn_out):
    #             _, gate_msa, shift_mlp, scale_mlp, gate_mlp = image_variables[i]
    #             h = image_states[i] + image_attn_out[i] * gate_msa.unsqueeze(1)
    #             norm_h = self.norm2(h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    #             image_out.append(norm_h)
        
    #     return text_out, latent_out, image_out
    
    # def _compute_adnorm(self, hidden_states, temb, norm_layer):
    #     """计算 AdaNorm"""
    #     # 这里需要根据实际的 AdaNorm 实现来调整
    #     # 这是一个简化的实现
    #     norm_hidden_states = norm_layer(hidden_states)
    #     gate_msa = torch.ones(hidden_states.shape[0], device=hidden_states.device)
    #     shift_mlp = torch.zeros(hidden_states.shape[0], device=hidden_states.device)
    #     scale_mlp = torch.ones(hidden_states.shape[0], device=hidden_states.device)
    #     gate_mlp = torch.ones(hidden_states.shape[0], device=hidden_states.device)
    #     return norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp
    
    # def _default_attention_forward(self, text_states, latent_states, image_states, adapters, **kwargs):
    #     """默认的注意力前向传播"""
    #     # 这里需要根据实际的注意力实现来调整
    #     text_out = text_states if text_states is not None else []
    #     latent_out = latent_states if latent_states is not None else []
    #     image_out = image_states if image_states is not None else []
    #     return text_out, latent_out, image_out
    
    def enable_gradient_checkpointing(self):
        """启用梯度检查点"""
        self.transformer.enable_gradient_checkpointing()
    
    def disable_gradient_checkpointing(self):
        """禁用梯度检查点"""
        self.transformer.disable_gradient_checkpointing()
    
    def train(self, mode=True):
        """设置训练模式"""
        super().train(mode)
        self.transformer.train(mode)
        return self
    
    def eval(self):
        """设置评估模式"""
        super().eval()
        self.transformer.eval()
        return self
    
    def to(self, *args, **kwargs):
        """移动到指定设备或数据类型；兼容 torch.nn.Module.to 的多种调用形式"""
        super().to(*args, **kwargs)
        self.transformer.to(*args, **kwargs)
        return self
    
    def parameters(self, recurse=True):
        """返回所有参数"""
        for param in super().parameters(recurse):
            yield param
        for param in self.transformer.parameters(recurse):
            yield param
    
    def named_parameters(self, prefix='', recurse=True):
        """返回所有命名参数"""
        for name, param in super().named_parameters(prefix, recurse):
            yield name, param
        for name, param in self.transformer.named_parameters(prefix, recurse):
            yield name, param
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """返回状态字典"""
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        transformer_state = self.transformer.state_dict(prefix='transformer.', keep_vars=keep_vars)
        state_dict.update(transformer_state)
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """加载状态字典"""
        # 分离 transformer 的状态
        transformer_state = {k[12:]: v for k, v in state_dict.items() if k.startswith('transformer.')}
        wrapper_state = {k: v for k, v in state_dict.items() if not k.startswith('transformer.')}
        
        
        # 加载 wrapper 状态
        super().load_state_dict(wrapper_state, strict=False)
        
        # 加载 transformer 状态
        self.transformer.load_state_dict(transformer_state, strict=strict)
    
    def __getattr__(self, name):
        """代理属性访问到原始 transformer"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)


def create_sd3_transformer_wrapper(
    transformer: SD3Transformer2DModel,
    add_latent_params: bool = True,
    token_size: Optional[int] = 64,
    **kwargs
) -> SD3TransformerWrapper:
    """
    创建 SD3 Transformer Wrapper 的便捷函数
    
    Args:
        transformer: 原始的 SD3Transformer2DModel
        add_latent_params: 是否添加 latent 分支参数
        token_size: SRTokenizer 的 token 大小，如果为 None 则使用默认值 16
        **kwargs: 其他参数
        
    Returns:
        SD3TransformerWrapper: 包装后的 transformer
    """
    return SD3TransformerWrapper(
        transformer=transformer,
        add_latent_params=add_latent_params,
        token_size=token_size,
        **kwargs
    )


# 为了保持兼容性，提供别名
SD3TransformerWithLatent = SD3TransformerWrapper
