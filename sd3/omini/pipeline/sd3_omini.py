import torch
from typing import List, Union, Optional, Dict, Any, Callable, Type, Tuple
import math
import time
import functools
import numpy as np
from diffusers.models.attention_processor import Attention
import torch.nn.functional as F
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline, calculate_shift, retrieve_timesteps
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.embeddings import apply_rotary_emb
from transformers.pipelines import pipeline

from peft.tuners.tuners_utils import BaseTunerLayer
from accelerate.utils import is_torch_version
from contextlib import contextmanager

import cv2
from PIL import Image, ImageFilter

import inspect


def seed_everything(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)


def clip_hidden_states(hidden_states: torch.FloatTensor) -> torch.FloatTensor:
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return hidden_states


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


def encode_images(pipeline: StableDiffusion3Pipeline, images: torch.Tensor):
    """
    Encodes the images into tokens and ids for FLUX pipeline.
    """
    images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_ids = _prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )   
    return images, images_ids


depth_pipe = None


def convert_to_condition(
    condition_type: str,
    raw_img: Union[Image.Image, torch.Tensor],
    blur_radius: Optional[float] = 5.0,
) -> Union[Image.Image, torch.Tensor]:
    if condition_type == "depth":
        global depth_pipe
        depth_pipe = depth_pipe or pipeline(
            task="depth-estimation",
            model="LiheYoung/depth-anything-small-hf",
            device="cpu",
        )
        source_image = raw_img.convert("RGB")
        condition_img = depth_pipe(source_image)["depth"].convert("RGB")
        return condition_img
    elif condition_type == "canny":
        img = np.array(raw_img)
        edges = cv2.Canny(img, 100, 200)
        edges = Image.fromarray(edges).convert("RGB")
        return edges
    elif condition_type == "coloring":
        return raw_img.convert("L").convert("RGB")
    elif condition_type == "deblurring":
        radius = blur_radius if blur_radius is not None else 5.0
        condition_image = (
            raw_img.convert("RGB")
            .filter(ImageFilter.GaussianBlur(float(radius)))
            .convert("RGB")
        )
        return condition_image
    else:
        print("Warning: Returning the raw image.")
        return raw_img.convert("RGB")


class Condition:
    def __init__(
        self,
        condition: Union[Image.Image, torch.Tensor],
        adapter_setting: Union[str, dict, List[Union[str, dict]]],
        position_delta=None,
        position_scale=1.0,
        latent_mask=None,
        is_complement=False,
    ):
        self.condition = condition
        self.adapter = adapter_setting if isinstance(adapter_setting, list) else [adapter_setting]
        self.position_delta = position_delta
        self.position_scale = position_scale
        self.latent_mask = (
            latent_mask.T.reshape(-1) if latent_mask is not None else None
        )
        self.is_complement = is_complement

    def encode(
        self, pipe: StableDiffusion3Pipeline, empty: bool = False
    ):
        condition_empty = Image.new("RGB", self.condition.size, (0, 0, 0))
        tokens, ids = encode_images(pipe, condition_empty if empty else self.condition)
        
        if self.position_delta is not None:
            ids[:, 1] += self.position_delta[0]
            ids[:, 2] += self.position_delta[1]
        if self.position_scale != 1.0:
            scale_bias = (self.position_scale - 1.0) / 2
            ids[:, 1:] *= self.position_scale
            ids[:, 1:] += scale_bias
        if self.latent_mask is not None:
            tokens = tokens[:, self.latent_mask]
            ids = ids[self.latent_mask]

        return tokens, ids


@contextmanager
def specify_lora(lora_modules: List[BaseTunerLayer], specified_lora: Union[str, List[str]]):
    # 这里假设lora_modules为SD3 transformer的LoRA层
    valid_lora_modules = [m for m in lora_modules if isinstance(m, BaseTunerLayer)]
    original_scales = [
        {
            adapter: module.scaling[adapter]
            for adapter in getattr(module, 'active_adapters', [])
            if hasattr(module, 'scaling') and adapter in module.scaling
        }
        for module in valid_lora_modules
    ]
    if not isinstance(specified_lora, list):
        specified_lora = [specified_lora]
    for module in valid_lora_modules:
        for adapter in getattr(module, 'active_adapters', []):
            if hasattr(module, 'scaling') and adapter in module.scaling:
                module.scaling[adapter] = module.scaling[adapter] if adapter in specified_lora else 0
    try:
        yield
    finally:
        for module, scales in zip(valid_lora_modules, original_scales):
            for adapter in getattr(module, 'active_adapters', []):
                if hasattr(module, 'scaling') and adapter in module.scaling:
                    module.scaling[adapter] = scales[adapter]


def attn_forward(
    attn: Attention,
    hidden_states: List[torch.FloatTensor],
    adapters: List[Union[str, List[str]]],
    encoder_hidden_states: Optional[List[torch.FloatTensor]] = None,
    position_embs: Optional[List[torch.Tensor]] = None,
    group_mask: Optional[torch.Tensor] = None,
    cache_mode: Optional[str] = None,
    to_cache: Optional[List[torch.Tensor]] = None,
    cache_storage: Optional[List[torch.Tensor]] = None,
    **kwargs: dict,
):
    # 兼容text+image分支，encoder_hidden_states为text分支
    bs, _, _ = hidden_states[0].shape
    queries, keys, values = [], [], []
    # 处理text分支
    if encoder_hidden_states is not None:
        for i, hidden_state in enumerate(encoder_hidden_states):
            query = attn.add_q_proj(hidden_state)
            key = attn.add_k_proj(hidden_state)
            value = attn.add_v_proj(hidden_state)
            head_dim = key.shape[-1] // attn.heads
            reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
            query, key, value = map(reshape_fn, (query, key, value))
            query, key = attn.norm_added_q(query), attn.norm_added_k(key)
            queries.append(query)
            keys.append(key)
            values.append(value)
    txt_n = len(encoder_hidden_states) if encoder_hidden_states is not None else 0
    # 处理image/condition分支
    for i, hidden_state in enumerate(hidden_states):
        current_adapters = adapters[i + txt_n]
        if not isinstance(current_adapters, list):
            current_adapters = [current_adapters]
        with specify_lora([getattr(attn, 'to_q', None), getattr(attn, 'to_k', None), getattr(attn, 'to_v', None)], current_adapters):
            query = attn.to_q(hidden_state)
            key = attn.to_k(hidden_state)
            value = attn.to_v(hidden_state)
        head_dim = key.shape[-1] // attn.heads
        reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
        query, key, value = map(reshape_fn, (query, key, value))
        query, key = attn.norm_q(query), attn.norm_k(key)
        queries.append(query)
        keys.append(key)
        values.append(value)
    # 缓存写入
    if cache_mode == "write":
        for i, (k, v) in enumerate(zip(keys, values)):
            if to_cache[i]:
                cache_storage[attn.cache_idx][0].append(k)
                cache_storage[attn.cache_idx][1].append(v)
    # 注意力计算
    attn_outputs = []
    for i, query in enumerate(queries):
        keys_i, values_i = [keys[i]], [values[i]]
        if group_mask is not None:
            for j in range(len(keys)):
                if group_mask[i][j].item():
                    keys_i.append(keys[j])
                    values_i.append(values[j])
        if cache_mode == "read":
            keys_i.extend(cache_storage[attn.cache_idx][0])
            values_i.extend(cache_storage[attn.cache_idx][1])
        if keys_i:
            attn_output = F.scaled_dot_product_attention(
                query, torch.cat(keys_i, dim=2), torch.cat(values_i, dim=2)
            ).to(query.dtype)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bs, -1, attn.heads * head_dim)
            attn_outputs.append(attn_output)
        else:
            attn_outputs.append(torch.zeros_like(query.transpose(1, 2).reshape(bs, -1, attn.heads * head_dim)))
    h_out = []
    for i, hidden_state in enumerate(queries):
        h = attn_outputs[i]
        if getattr(attn, "to_out", None) is not None:
            with specify_lora((attn.to_out[0],), adapters[i] if i < len(adapters) else [None]):
                h = attn.to_out[0](h)
        h_out.append(h)
    return h_out


@torch.no_grad()
def generate(
    pipeline: StableDiffusion3Pipeline,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    # Condition Parameters (Optional)
    main_adapter: Optional[List[str]] = None,
    conditions: List[Condition] = [],
    image_guidance_scale: float = 1.0,
    transformer_kwargs: Optional[Dict[str, Any]] = {},
    kv_cache=False,
    attn_type: str = "standard",  # 'standard' or 'swin'
    window_size: int = 8,
    shift_size: int = 4,
    latent_mask: Optional[torch.Tensor] = None,
    **params: dict,
):
    self = pipeline

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    height = height // 2 * 2
    width = width // 2 * 2

    # 1. 检查输入
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    # 2. 文本编码
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=True if guidance_scale > 1 else False,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=self._execution_device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )

    # 3. batch_size
    if pooled_prompt_embeds is not None:
        batch_size = pooled_prompt_embeds.shape[0]
    else:
        batch_size = 1
    device = self._execution_device

    # 4. latent变量准备
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        pooled_prompt_embeds.dtype if pooled_prompt_embeds is not None else torch.float32,
        device,
        generator,
        latents,
    )

    latent_image_ids = _prepare_latent_image_ids(batch_size, int(height) // self.vae_scale_factor, int(width) // self.vae_scale_factor, device, latents.dtype)

    # 5. 条件准备
    c_latents, uc_latents, c_ids, c_timesteps = ([], [], [], [])
    c_projections, c_guidances, c_adapters = ([], [], [])
    complement_cond = None
    for condition in conditions:
        tokens, ids = condition.encode(self)
        c_latents.append(tokens)
        if image_guidance_scale != 1.0:
            uc_latents.append(condition.encode(self, empty=True)[0])
        c_ids.append(ids)
        c_timesteps.append(torch.zeros([1], device=device))
        c_projections.append(pooled_prompt_embeds)
        c_guidances.append(torch.ones([1], device=device))
        c_adapters.append(condition.adapter)
        if condition.is_complement:
            complement_cond = (tokens, ids)

    # 6. 步长准备
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, timesteps, sigmas, mu=mu
    )
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * self.scheduler.order, 0
    )
    self._num_timesteps = len(timesteps)

    if kv_cache:
        attn_counter = 0
        for module in self.transformer.modules():
            if isinstance(module, Attention):
                setattr(module, "cache_idx", attn_counter)
                attn_counter += 1
        kv_cond = [[[], []] for _ in range(attn_counter)]
        kv_uncond = [[[], []] for _ in range(attn_counter)]
        def clear_cache():
            for storage in [kv_cond, kv_uncond]:
                for kesy, values in storage:
                    kesy.clear()
                    values.clear()

    # 7. 组装分支参数（text, image, condition）
    branch_n = len(conditions) + 2
    group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool)
    group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(conditions)))
    if kv_cache:
        group_mask[2:, :2] = False

    # text分支参数
    text_features = [prompt_embeds] if prompt_embeds is not None else []
    txt_ids = [prompt_embeds] if prompt_embeds is not None else []
    text_adapters = [main_adapter] if main_adapter is not None else [None]
    text_timesteps = [timesteps[0]]
    text_projections = [pooled_prompt_embeds]

    # image分支参数
    image_features = [latents]
    img_ids = [latent_image_ids]
    image_adapters = [main_adapter]
    image_timesteps = [timesteps[0]]
    image_projections = [pooled_prompt_embeds]

    # condition分支参数
    cond_features = c_latents if c_latents else []
    cond_ids = c_ids if c_ids else []
    cond_adapters = c_adapters if c_adapters else []
    cond_timesteps = c_timesteps if c_timesteps else []
    cond_projections = c_projections if c_projections else []

    # 合并所有分支参数
    all_features = text_features + image_features + cond_features
    all_ids = txt_ids + img_ids + cond_ids
    all_adapters = text_adapters + image_adapters + cond_adapters
    all_timesteps = text_timesteps + image_timesteps + cond_timesteps
    all_projections = text_projections + image_projections + cond_projections

    # 去噪循环
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype) / 1000
            if kv_cache:
                mode = "write" if i == 0 else "read"
                if mode == "write":
                    clear_cache()
            use_cond = not (kv_cache) or mode == "write"
            selected_attn_forward = attn_forward
            transformer_kwargs["latent_height"] = height // self.vae_scale_factor
            transformer_kwargs["latent_width"] = width // self.vae_scale_factor
            noise_pred = transformer_forward(
                self.transformer,
                hidden_states=all_features[1:] + (cond_features if use_cond else []),
                encoder_hidden_states=all_features[:1],
                img_ids=all_ids[1:] + (cond_ids if use_cond else []),
                txt_ids=all_ids[:1],
                timesteps=all_timesteps,
                pooled_projections=all_projections,
                adapters=all_adapters,
                cache_mode=mode if kv_cache else None,
                cache_storage=kv_cond if kv_cache else None,
                to_cache=[False, False, *[True] * len(c_latents)],
                group_mask=group_mask,
                attn_forward=selected_attn_forward,
                **transformer_kwargs,
            )[0]
            if image_guidance_scale != 1.0:
                unc_pred = transformer_forward(
                    self.transformer,
                    hidden_states=all_features[1:] + (uc_latents if use_cond else []),
                    encoder_hidden_states=all_features[:1],
                    img_ids=all_ids[1:] + (cond_ids if use_cond else []),
                    txt_ids=all_ids[:1],
                    timesteps=all_timesteps,
                    pooled_projections=all_projections,
                    adapters=all_adapters,
                    cache_mode=mode if kv_cache else None,
                    cache_storage=kv_uncond if kv_cache else None,
                    to_cache=[False, False, *[True] * len(c_latents)],
                    attn_forward=selected_attn_forward,
                    group_mask=group_mask,
                    **transformer_kwargs,
                )[0]
                noise_pred = unc_pred + image_guidance_scale * (noise_pred - unc_pred)
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents)[0]
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
    if latent_mask is not None:
        assert complement_cond is not None
        comp_latent, comp_ids = complement_cond
        all_ids = torch.cat([latent_image_ids, comp_ids], dim=0)  # (Ta+Tc,3)
        shape = (all_ids.max(dim=0).values + 1).to(torch.long)  # (3,)
        H, W = shape[1].item(), shape[2].item()
        B, _, C = latents.shape
        canvas = latents.new_zeros(B, H * W, C)  # (B,H*W,C)
        def _stash(canvas, tokens, ids, H, W) -> None:
            B, T, C = tokens.shape
            ids = ids.to(torch.long)
            flat_idx = (ids[:, 1] * W + ids[:, 2]).to(torch.long)
            canvas.view(B, -1, C).index_copy_(1, flat_idx, tokens)
        _stash(canvas, latents, latent_image_ids, H, W)
        _stash(canvas, comp_latent, comp_ids, H, W)
        latents = canvas.view(B, H * W, C)
    if output_type == "latent":
        image = latents
    else:
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
    self.maybe_free_model_hooks()
    if not return_dict:
        return (image,)
    return StableDiffusion3PipelineOutput(images=image)


def block_forward(
    self,
    hidden_states: List[torch.FloatTensor],  # image分支
    encoder_hidden_states: Optional[List[torch.FloatTensor]],  # text分支
    tembs: List[torch.FloatTensor],
    adapters: List[str],
    position_embs=None,
    attn_forward=attn_forward,
    group_mask=None,
    joint_attention_kwargs=None,
    **kwargs,
):
    joint_attention_kwargs = joint_attention_kwargs or {}
    img_variables = []
    for i, image_h in enumerate(hidden_states):
        with specify_lora((self.norm1.linear,), adapters[i + len(encoder_hidden_states) if encoder_hidden_states is not None else 0]):
            img_variables.append(self.norm1(image_h, emb=tembs[i + len(encoder_hidden_states) if encoder_hidden_states is not None else 0]))
    txt_variables = []
    if encoder_hidden_states is not None:
        for i, text_h in enumerate(encoder_hidden_states):
            txt_variables.append(self.norm1_context(text_h, emb=tembs[i]))

    image_out = []
    text_out = []
    for i in range(len(hidden_states)):
        if getattr(self, "use_dual_attention", False):
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = img_variables[i]
            norm_encoder_hidden_states = txt_variables[0][0] if txt_variables else None
            # joint attention
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                **joint_attention_kwargs,
            )
            # dual attention
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2)
            h = hidden_states[i] + attn_output * gate_msa.unsqueeze(1) + attn_output2 * gate_msa2.unsqueeze(1)
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = img_variables[i]
            norm_encoder_hidden_states = txt_variables[0][0] if txt_variables else None
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                **joint_attention_kwargs,
            )
            h = hidden_states[i] + attn_output * gate_msa.unsqueeze(1)
        norm_h = self.norm2(h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        with specify_lora((self.ff.net[2],), adapters[i]):
            h = h + self.ff(norm_h) * gate_mlp.unsqueeze(1)
        image_out.append(clip_hidden_states(h))
        # text分支后处理（仅第一个text分支）
        if txt_variables:
            norm_text, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = txt_variables[0]
            text_h = encoder_hidden_states[0] + context_attn_output * c_gate_msa.unsqueeze(1)
            norm_text_h = self.norm2_context(text_h) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            text_h = text_h + self.ff_context(norm_text_h) * c_gate_mlp.unsqueeze(1)
            text_out.append(clip_hidden_states(text_h))
    return image_out, text_out


def transformer_forward(
    transformer: SD3Transformer2DModel,
    hidden_states: List[torch.Tensor],
    encoder_hidden_states: Optional[List[torch.Tensor]] = None,
    img_ids: List[torch.Tensor] = None,
    txt_ids: List[torch.Tensor] = None,
    pooled_projections: List[torch.Tensor] = None,
    timesteps: List[torch.LongTensor] = None,
    adapters: List[str] = None,
    block_forward=block_forward,
    attn_forward=attn_forward,
    group_mask=None,
    **kwargs: dict,
):
    self = transformer
    adapters = adapters or [None] * (len(hidden_states))
    assert len(adapters) == len(timesteps)
    # 预处理image_features
    image_hidden_states = []
    txt_n = len(encoder_hidden_states) if encoder_hidden_states is not None else 0
    for i, image_feature in enumerate(hidden_states):
        with specify_lora((self.x_embedder,), adapters[i + txt_n]):
            image_hidden_states.append(self.x_embedder(image_feature))
    # 预处理text_features
    text_hidden_states = []
    if encoder_hidden_states is not None:
        for text_feature in encoder_hidden_states:
            text_hidden_states.append(self.context_embedder(text_feature))
    def get_temb(timestep, pooled_projection):
        timestep = timestep.to(image_hidden_states[0].dtype) * 1000
        return self.time_text_embed(timestep, pooled_projection)
    tembs = [get_temb(*each) for each in zip(timesteps, pooled_projections)]
    # 只为图像tokens准备位置编码
    # position_embs = [self.pos_embed(each) for each in img_ids]
    # 梯度检查点参数
    gckpt_kwargs: Dict[str, Any] = (
        {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
    )
    for i, block in enumerate(self.transformer_blocks):
        block_kwargs = {
            "self": block,
            "hidden_states": image_hidden_states,
            "encoder_hidden_states": text_hidden_states,
            "tembs": tembs,
            # "position_embs": position_embs,
            "adapters": adapters,
            "attn_forward": attn_forward,
            "group_mask": group_mask,
            **kwargs,
        }
        if self.training and self.gradient_checkpointing:
            image_hidden_states = torch.utils.checkpoint.checkpoint(
                block_forward, **block_kwargs, **gckpt_kwargs
            )
        else:
            image_hidden_states = block_forward(**block_kwargs)
    hidden_state = image_hidden_states[0]
    with specify_lora((self.norm_out.linear, self.proj_out), adapters[0]):
        hidden_state = self.norm_out(hidden_state, tembs[0])
        hidden_state = self.proj_out(hidden_state)
    return (hidden_state,)