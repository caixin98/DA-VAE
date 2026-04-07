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
    encoder_hidden_states: List[torch.FloatTensor] = None,
    position_embs: Optional[List[torch.Tensor]] = None,
    group_mask: Optional[torch.Tensor] = None,
    cache_mode: Optional[str] = None,
    to_cache: Optional[List[torch.Tensor]] = None,
    cache_storage: Optional[List[torch.Tensor]] = None,
    **kwargs: dict,
):
    # 这里只是示例，实际应根据SD3 Attention结构调整
    bs, _, _ = hidden_states[0].shape
    queries, keys, values = [], [], []
    for i, hidden_state in enumerate(hidden_states):
        current_adapters = adapters[i]
        if not isinstance(current_adapters, list):
            current_adapters = [current_adapters]
        # 这里假设attn有to_q/to_k/to_v等LoRA层
        with specify_lora([getattr(attn, 'to_q', None), getattr(attn, 'to_k', None), getattr(attn, 'to_v', None)], current_adapters):
            query = attn.to_q(hidden_state)
            key = attn.to_k(hidden_state)
            value = attn.to_v(hidden_state)
        head_dim = key.shape[-1] // attn.heads
        reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
        query, key, value = map(reshape_fn, (query, key, value))
        query, key = attn.norm_added_q(query), attn.norm_added_k(key)
        queries.append(query)
        keys.append(key)
        values.append(value)

    # # 测量位置编码时间
    # if position_embs is not None:
    #     queries = [apply_rotary_emb(q, position_embs[i]) for i, q in enumerate(queries)]
    #     keys = [apply_rotary_emb(k, position_embs[i]) for i, k in enumerate(keys)]

    # 测量缓存写入时间
    if cache_mode == "write":
        for i, (k, v) in enumerate(zip(keys, values)):
            if to_cache[i]:
                cache_storage[attn.cache_idx][0].append(k)
                cache_storage[attn.cache_idx][1].append(v)

    # 测量注意力计算时间
    attn_outputs = []
    for  i, query in enumerate(queries):
        img_idx = i
        keys_i, values_i = [keys[img_idx]], [values[img_idx]]

        if group_mask is not None:
            for j in range(len(keys)):
                if group_mask[img_idx][j].item():
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
    for i, hidden_state in enumerate(hidden_states):
        h = attn_outputs[i]
        if getattr(attn, "to_out", None) is not None:
            with specify_lora((attn.to_out[0],), adapters[i]):
                h = attn.to_out[0](h)
        h_out.append(h)
    return h_out


def swin_attn_forward(
    attn: Attention,
    hidden_states: List[torch.FloatTensor],  # image部分
    adapters: List[Union[str, List[str]]],
    position_embs: Optional[List[torch.Tensor]] = None,
    group_mask: Optional[torch.Tensor] = None,
    cache_mode: Optional[str] = None,
    to_cache: Optional[List[torch.Tensor]] = None,
    cache_storage: Optional[List[torch.Tensor]] = None,
    window_size: int = 8,
    shift_size: int = 4,
    **kwargs: dict,
) -> torch.FloatTensor:
    raise NotImplementedError("Swin Attention is not implemented for SD3")

def block_forward(
    self,
    hidden_states: List[torch.FloatTensor],
    encoder_hidden_states: List[torch.FloatTensor],
    tembs: List[torch.FloatTensor],
    adapters: List[str],
    position_embs=None,
    attn_forward=attn_forward,
    **kwargs: dict,
):
    
    # 归一化
    img_variables = []
    for i, image_h in enumerate(hidden_states):
        with specify_lora((self.norm1.linear,), adapters[i]):
            img_variables.append(self.norm1(image_h, emb=tembs[i]))

    attn_output = attn_forward(
        self.attn,
        hidden_states=[each[0] for each in img_variables],
        encoder_hidden_states=encoder_hidden_states,
        adapters=adapters,
        position_embs=position_embs,
        **kwargs,
    )
    image_out = []
    for i in range(len(hidden_states)):
        if self.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = img_variables[i]
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = img_variables[i]
        image_h = (
            hidden_states[i] + attn_output[i] * gate_msa.unsqueeze(1)
        ).to(hidden_states[i].dtype)

        if self.use_dual_attention and i == 0:
            with specify_lora([getattr(self.attn2, 'to_q', None), getattr(self.attn2, 'to_k', None), getattr(self.attn2, 'to_v', None), getattr(self.attn2, 'to_out', None)], adapters[i]):
                attn_output2 = self.attn2(hidden_states=norm_hidden_states2)
                attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
                image_h = image_h + attn_output2

        norm_h = self.norm2(image_h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        with specify_lora((self.ff.net[2],), adapters[i]):
            image_h = image_h + self.ff(norm_h) * gate_mlp.unsqueeze(1)
        image_out.append(clip_hidden_states(image_h))
    # skip the context pre-only block
    return image_out



def transformer_forward(
    transformer: SD3Transformer2DModel,
    hidden_states: List[torch.Tensor],
    encoder_hidden_states: List[torch.Tensor] = None,
    img_ids: List[torch.Tensor] = None,
    pooled_projections: List[torch.Tensor] = None,
    timesteps: List[torch.LongTensor] = None,
    adapters: List[str] = None,
    block_forward=block_forward,
    attn_forward=attn_forward,
    latent_height: Optional[int] = None,
    latent_width: Optional[int] = None,
    **kwargs: dict,
):
    self = transformer
    img_n = len(hidden_states)

    adapters = adapters or [None] * img_n
    assert len(adapters) == len(timesteps)
    
    # 检查timesteps长度
    assert len(timesteps) == img_n

    def get_temb(timestep, pooled_projection):
        timestep = timestep.to(hidden_states[0].dtype) * 1000
        return self.time_text_embed(timestep, pooled_projection)

    tembs = [get_temb(*each) for each in zip(timesteps, pooled_projections)]
    # print(f"tembs: {tembs[0].shape}")
    # 只为图像tokens准备位置编码
    # position_embs = [self.pos_embed(each) for each in img_ids]
    hidden_states = [self.pos_embed(each) for each in hidden_states]
    # 梯度检查点参数
    gckpt_kwargs: Dict[str, Any] = (
        {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
    )

    # 只经过图像分支的transformer blocks
    for i, block in enumerate(self.transformer_blocks):
  
        block_kwargs = {
            "self": block,
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "tembs": tembs,
            # "position_embs": position_embs,
            "adapters": adapters,
            "attn_forward": attn_forward,
            **kwargs,
        }
        if self.training and self.gradient_checkpointing:
            hidden_states = torch.utils.checkpoint.checkpoint(
                block_forward, **block_kwargs, **gckpt_kwargs
            )
        else:
            hidden_states = block_forward(**block_kwargs)
    hidden_state = hidden_states[0]
    with specify_lora((self.norm_out.linear, self.proj_out), adapters[0]):
        hidden_state = self.norm_out(hidden_state, tembs[0])
        hidden_state = self.proj_out(hidden_state)

        # unpatchify
        patch_size = self.config.patch_size
        height = latent_height // patch_size
        width = latent_width // patch_size

        hidden_state = hidden_state.reshape(
            shape=(hidden_state.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_state = torch.einsum("nhwpqc->nchpwq", hidden_state)
        hidden_state = hidden_state.reshape(
            shape=(hidden_state.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )
    return hidden_state


@torch.no_grad()
def generate(
    pipeline: StableDiffusion3Pipeline,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
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
    # 检查输入
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    # 只考虑图像分支，batch_size 由 pooled_prompt_embeds 推断
    if pooled_prompt_embeds is not None:
        batch_size = pooled_prompt_embeds.shape[0]
    else:
        # fallback
        batch_size = 1

    device = self._execution_device

    if pooled_prompt_embeds is None:
        pooled_prompt_embeds = torch.randn(batch_size, 2048, device=device).to(self.dtype)
    # 不再处理prompt相关内容，直接使用pooled_prompt_embeds
    # latent变量准备
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

    print(f"latents: {latents.shape}")

    latent_image_ids = _prepare_latent_image_ids(batch_size, int(height) // self.vae_scale_factor, int(width) // self.vae_scale_factor, device, latents.dtype)




    # 条件准备
    c_latents, uc_latents, c_ids, c_timesteps = ([], [], [], [])
    c_projections, c_guidances, c_adapters = ([], [], [])
    complement_cond = None
    for condition in conditions:
        tokens, ids = condition.encode(self)
        c_latents.append(tokens)  # [batch_size, token_n, token_dim]
        if image_guidance_scale != 1.0:
            uc_latents.append(condition.encode(self, empty=True)[0])
        c_ids.append(ids)  # [token_n, id_dim(3)]
        c_timesteps.append(torch.zeros([1], device=device))
        c_projections.append(pooled_prompt_embeds)
        c_guidances.append(torch.ones([1], device=device))
        c_adapters.append(condition.adapter)
        if condition.is_complement:
            complement_cond = (tokens, ids)

    # 步长准备
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
                     
    branch_n = len(conditions) + 1
    group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool)
    group_mask[1:, 1:] = torch.diag(torch.tensor([1] * len(conditions)))
    if kv_cache:
        group_mask[1:, 1:] = False
    # 去噪循环
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype) / 1000

   

            if kv_cache:
                mode = "write" if i == 0 else "read"
                if mode == "write":
                    clear_cache()
            use_cond = not (kv_cache) or mode == "write"
            
            # 选择注意力前向函数
            selected_attn_forward = attn_forward
            # if attn_type == "swin":
            #     selected_attn_forward = swin_attn_forward
            #     transformer_kwargs["window_size"] = window_size
            #     transformer_kwargs["shift_size"] = shift_size
            transformer_kwargs["latent_height"] = height // self.vae_scale_factor
            transformer_kwargs["latent_width"] = width // self.vae_scale_factor
            noise_pred = transformer_forward(
                self.transformer,
                hidden_states=[latents] + (c_latents if use_cond else []),
                encoder_hidden_states=None,
                img_ids=[latent_image_ids] + (c_ids if use_cond else []),
                txt_ids=None,
                timesteps=[timestep] + (c_timesteps if use_cond else []),
                pooled_projections=[pooled_prompt_embeds] + (c_projections if use_cond else []),
                return_dict=False,
                adapters=[main_adapter] + (c_adapters if use_cond else []),
                cache_mode=mode if kv_cache else None,
                cache_storage=kv_cond if kv_cache else None,
                to_cache=[False, *[True] * len(c_latents)],
                group_mask=group_mask,
                attn_forward=selected_attn_forward,

                **transformer_kwargs,
            )[0]

            if image_guidance_scale != 1.0:
                unc_pred = transformer_forward(
                    self.transformer,
                    hidden_states=[latents] + (uc_latents if use_cond else []),
                    encoder_hidden_states=None,
                    img_ids=[latent_image_ids] + (c_ids if use_cond else []),
                    txt_ids=None,
                    timesteps=[timestep] + (c_timesteps if use_cond else []),
                    pooled_projections=[pooled_prompt_embeds] + (c_projections if use_cond else []),
                    return_dict=False,
                    adapters=[main_adapter] + (c_adapters if use_cond else []),
                    cache_mode=mode if kv_cache else None,
                    cache_storage=kv_uncond if kv_cache else None,
                    to_cache=[False, *[True] * len(c_latents)],
                    attn_forward=selected_attn_forward,
                    group_mask=group_mask,
                    **transformer_kwargs,
                )[0]

                noise_pred = unc_pred + image_guidance_scale * (noise_pred - unc_pred)

            # 计算上一步噪声
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
        # latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return StableDiffusion3PipelineOutput(images=image)