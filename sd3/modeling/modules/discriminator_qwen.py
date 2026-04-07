from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_vl import Qwen2VLImageProcessor
import torch
from PIL import Image
import numpy as np
import torch.nn as nn
from copy import deepcopy
from .maskgit_vqgan import Conv2dSame
import functools
from utils.tensor_image_processor import TensorImageProcessor, smart_resize
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]

    return lora_module_names

def forward_qwen_visual(model, hidden_states, grid_thw):
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

    Returns:
        `torch.Tensor`: hidden_states.
    """
    hidden_states = model.patch_embed(hidden_states)
    rotary_pos_emb = model.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = model.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // model.spatial_merge_unit, model.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // model.spatial_merge_unit, model.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for layer_num, blk in enumerate(model.blocks):
        if layer_num in model.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens
        if model.gradient_checkpointing and model.training:
            hidden_states = model._gradient_checkpointing_func(
                blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
            )
        else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

    # hidden_states = model.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    return hidden_states
        
class QwenVLDiscriminator(nn.Module):
    def __init__(self, use_checkpoint=True, use_lora=False):
        super().__init__()
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
        out_channels = 1280
        activation = functools.partial(torch.nn.LeakyReLU, negative_slope=0.1)
        self.to_logits = torch.nn.Sequential(
            Conv2dSame(out_channels, out_channels // 2, 1),
            activation(),
            Conv2dSame(out_channels // 2, 1, kernel_size=1)
        )   
        del qwen_model
        del self.model.merger
        # 应用LoRA配置
        self.use_lora = use_lora
        if use_lora:
                # 默认LoRA配置
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=32,  # LoRA rank
                lora_alpha=32,  # LoRA alpha parameter
                lora_dropout=0.1,  # LoRA dropout
                target_modules=find_target_linear_names(self.model, num_lora_modules=-1, lora_namespan_exclude=["merger", "to_logits"])
            )
            
            # 冻结原始模型参数
            for param in self.model.parameters():
                param.requires_grad = False
            
            # 应用LoRA到模型
            self.model = get_peft_model(self.model, lora_config)
            print(f"Applied LoRA with config: {lora_config}")
            
        self.use_checkpoint = use_checkpoint
        self.print_trainable_parameters()

    def forward(self, x):
        B, C, H, W = x.shape
        h_bar, w_bar = smart_resize(H, W, self.patch_size)
        output_h = h_bar // self.patch_size
        output_w = w_bar // self.patch_size
        
        # 使用新的tensor image processor，直接处理tensor
        processed = self.processor(images=x, do_rescale=False)
        if self.use_checkpoint:
            output = checkpoint(forward_qwen_visual, self.model, processed["pixel_values"], processed["image_grid_thw"], use_reentrant=False)
        else:
            output = forward_qwen_visual(self.model, processed["pixel_values"], processed["image_grid_thw"])
        
        if output is not None:
            output = output.view(B, output_h, output_w, -1).permute(0, 3, 1, 2).contiguous()
            output = self.to_logits(output)
        else:
            # 处理output为None的情况
            raise ValueError("Model output is None, check the forward pass")
            
        return output

    def get_trainable_parameters(self):
        """获取可训练参数，用于LoRA训练时的参数统计"""
        if self.use_lora:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        if self.use_lora:
            print("LoRA训练参数:")
            self.model.print_trainable_parameters()
        else:
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.parameters())
            print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.2f}%")

