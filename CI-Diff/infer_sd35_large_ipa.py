# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput

from models.resampler import TimeResampler
from models.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.normalization import RMSNorm
from einops import rearrange


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusion3Pipeline

        >>> pipe = StableDiffusion3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        >>> image.save("sd3.png")
        ```
"""


class AdaLayerNorm(nn.Module):
    """
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, time_embedding_dim=None, mode='normal'):
        super().__init__()

        self.silu = nn.SiLU()
        num_params_dict = dict(
            zero=6,
            normal=2,
        )
        num_params = num_params_dict[mode]
        self.linear = nn.Linear(time_embedding_dim or embedding_dim, num_params * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        self.mode = mode

    def forward(
        self,
        x,
        hidden_dtype = None,
        emb = None,
    ):
        emb = self.linear(self.silu(emb))
        if self.mode == 'normal':
            shift_msa, scale_msa = emb.chunk(2, dim=1)
            x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
            return x

        elif self.mode == 'zero':
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
            x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class JointIPAttnProcessor(torch.nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        ip_hidden_states_dim=None,
        ip_encoder_hidden_states_dim=None,
        head_dim=None,
        timesteps_emb_dim=1280,
    ):
        super().__init__()

        self.norm_ip = AdaLayerNorm(ip_hidden_states_dim, time_embedding_dim=timesteps_emb_dim)
        self.to_k_ip = nn.Linear(ip_hidden_states_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(ip_hidden_states_dim, hidden_size, bias=False)
        self.norm_q = RMSNorm(head_dim, 1e-6)
        self.norm_k = RMSNorm(head_dim, 1e-6)
        self.norm_ip_k = RMSNorm(head_dim, 1e-6)


    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        emb_dict=None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        img_query = query
        img_key = key
        img_value = value

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)


        # IPadapter
        ip_hidden_states = emb_dict.get('ip_hidden_states', None)
        ip_hidden_states = self.get_ip_hidden_states(
            attn,
            img_query,
            ip_hidden_states,
            img_key,
            img_value,
            None,
            None,
            emb_dict['temb'],
        )
        if ip_hidden_states is not None:
            hidden_states = hidden_states + ip_hidden_states * emb_dict.get('scale', 1.0)


        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


    def get_ip_hidden_states(self, attn, query, ip_hidden_states, img_key=None, img_value=None, text_key=None, text_value=None, temb=None):
        if ip_hidden_states is None:
            return None
        
        if not hasattr(self, 'to_k_ip') or not hasattr(self, 'to_v_ip'):
            return None

        # norm ip input
        norm_ip_hidden_states = self.norm_ip(ip_hidden_states, emb=temb)

        # to k and v
        ip_key = self.to_k_ip(norm_ip_hidden_states)
        ip_value = self.to_v_ip(norm_ip_hidden_states)

        # reshape
        query = rearrange(query, 'b l (h d) -> b h l d', h=attn.heads)
        img_key = rearrange(img_key, 'b l (h d) -> b h l d', h=attn.heads)
        img_value = rearrange(img_value, 'b l (h d) -> b h l d', h=attn.heads)
        ip_key = rearrange(ip_key, 'b l (h d) -> b h l d', h=attn.heads)
        ip_value = rearrange(ip_value, 'b l (h d) -> b h l d', h=attn.heads)

        # norm
        query = self.norm_q(query)
        img_key = self.norm_k(img_key)
        ip_key = self.norm_ip_k(ip_key)

        # cat img
        key = torch.cat([img_key, ip_key], dim=2)
        value = torch.cat([img_value, ip_value], dim=2)

        # 
        ip_hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        ip_hidden_states = rearrange(ip_hidden_states, 'b h l d -> b l (h d)')
        ip_hidden_states = ip_hidden_states.to(query.dtype)
        return ip_hidden_states


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusion3Pipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin):
    r"""
    Args:
        transformer ([`SD3Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant,
            with an additional added projection layer that is initialized with a diagonal matrix with the `hidden_size`
            as its dimension.
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        text_encoder_3 ([`T5EncoderModel`]):
            Frozen text-encoder. Stable Diffusion 3 uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_3 (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        if self.text_encoder is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def check_inputs(
        self,
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_3 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_3`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")
        elif prompt_3 is not None and (not isinstance(prompt_3, str) and not isinstance(prompt_3, list)):
            raise ValueError(f"`prompt_3` has to be of type `str` or `list` but is {type(prompt_3)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_3 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_3`: {negative_prompt_3} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt


    @torch.inference_mode()
    def init_ipadapter(self, ip_adapter_path, image_encoder_path, nb_token, output_dim=2432):
        from transformers import SiglipVisionModel, SiglipImageProcessor
        state_dict = torch.load(ip_adapter_path, map_location="cpu")

        device, dtype = self.transformer.device, self.transformer.dtype
        image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path)
        image_processor = SiglipImageProcessor.from_pretrained(image_encoder_path)
        image_encoder.eval()
        image_encoder.to(device, dtype=dtype)
        self.image_encoder = image_encoder
        self.clip_image_processor = image_processor

        sample_class = TimeResampler
        image_proj_model = sample_class(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=nb_token,
            embedding_dim=1152,
            output_dim=output_dim,
            ff_mult=4,
            timestep_in_dim=320,
            timestep_flip_sin_to_cos=True,
            timestep_freq_shift=0,
        )
        image_proj_model.eval()
        image_proj_model.to(device, dtype=dtype)
        key_name = image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)
        print(f"=> loading image_proj_model: {key_name}")

        self.image_proj_model = image_proj_model


        attn_procs = {}
        transformer = self.transformer
        for idx_name, name in enumerate(transformer.attn_processors.keys()):
            hidden_size = transformer.config.attention_head_dim * transformer.config.num_attention_heads
            ip_hidden_states_dim = transformer.config.attention_head_dim * transformer.config.num_attention_heads
            ip_encoder_hidden_states_dim = transformer.config.caption_projection_dim
            
            attn_procs[name] = JointIPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=transformer.config.caption_projection_dim,
                ip_hidden_states_dim=ip_hidden_states_dim,
                ip_encoder_hidden_states_dim=ip_encoder_hidden_states_dim,
                head_dim=transformer.config.attention_head_dim,
                timesteps_emb_dim=1280,
            ).to(device, dtype=dtype)

        self.transformer.set_attn_processor(attn_procs)
        tmp_ip_layers = torch.nn.ModuleList(self.transformer.attn_processors.values())

        key_name = tmp_ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)
        print(f"=> loading ip_adapter: {key_name}")


    @torch.inference_mode()
    def encode_clip_image_emb(self, clip_image, device, dtype):

        # clip
        clip_image_tensor = self.clip_image_processor(images=clip_image, return_tensors="pt").pixel_values
        clip_image_tensor = clip_image_tensor.to(device, dtype=dtype)
        clip_image_embeds = self.image_encoder(clip_image_tensor, output_hidden_states=True).hidden_states[-2]
        clip_image_embeds = torch.cat([torch.zeros_like(clip_image_embeds), clip_image_embeds], dim=0)

        return clip_image_embeds



    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_subject: Union[str, List[str]] = None,  # New: Subject Text (e.g., "A frog")
        subject_image_path: str = None,  # New: Local Subject Image Path
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale_rare: float = 5.0,  # New: Rare Text Guidance Strength
        guidance_scale_subject: float = 4.0,  # New: Subject Text Guidance Strength
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_subject: Optional[Union[str, List[str]]] = None,  # New: Subject Text Negative Prompt
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds_rare: Optional[torch.FloatTensor] = None,  # New: Subject Text Negative Prompt
        prompt_embeds_subject: Optional[torch.FloatTensor] = None,  # New: Subject Text Pre-generated Embedding
        negative_prompt_embeds_rare: Optional[torch.FloatTensor] = None,  # New: Rare Text Negative Embedding
        negative_prompt_embeds_subject: Optional[torch.FloatTensor] = None,  # New: Subject Text Negative Embedding
        pooled_prompt_embeds_rare: Optional[torch.FloatTensor] = None,  # New: Rare Text Pooled Embedding
        pooled_prompt_embeds_subject: Optional[torch.FloatTensor] = None,  # New: Subject Text Pooled Embedding
        negative_pooled_prompt_embeds_rare: Optional[torch.FloatTensor] = None,  # New: Rare Text Negative Pooled Embedding
        negative_pooled_prompt_embeds_subject: Optional[torch.FloatTensor] = None,  # New: Subject Text Negative Pooled Embedding
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        clip_image=None,  
        ipadapter_scale=1.0,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        # height = height or self.default_sample_size * self.vae_scale_factor
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
    
        # 2. Input Validity Check (Extended Check for Dual Text Parameters)
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds_rare,  # Perform Basic Check Using Rare Text Embedding
            negative_prompt_embeds=negative_prompt_embeds_rare,
            pooled_prompt_embeds=pooled_prompt_embeds_rare,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds_rare,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
        # Additional Check for Subject Text and Image Path
        if prompt_subject is None and prompt_embeds_subject is None:
            raise ValueError("Subject text (prompt_subject) or pre-generated subject text embedding (prompt_embeds_subject) is required to be passed in")
        if subject_image_path is None and clip_image is None:
            raise ValueError("Either the local subject image path (subject_image_path) or clip_image needs to be passed in")
    
        # 3. Global Parameter Assignment
        self._guidance_scale = max(guidance_scale_rare, guidance_scale_subject)  # Compatible with the original guidance_scale logic
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs or {}
        self._interrupt = False
    
        # 4. Determine Batch Size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds_rare.shape[0] if prompt_embeds_rare is not None else 1
    
        device = self._execution_device
        dtype = self.transformer.dtype
        lora_scale = self._joint_attention_kwargs.get("scale", None)
    
        # 5. Load local subject image (prioritize subject_image_path, adapt to SigLIP 384x384 input)
        # if subject_image_path is not None:
        #     try:
        #         from PIL import Image
        #         clip_image = Image.open(subject_image_path).convert("RGB")
        #         clip_image = clip_image.resize((384, 384), Image.Resampling.LANCZOS)  
        #         print(f"Successfully loaded subject image: {subject_image_path} (resized to 384x384)")
        #     except Exception as e:
        #         raise FileNotFoundError(f"Failed to load subject image: {e}, please check if the path is correct")
        # else:
        #     # If using the original clip_image, force resize the dimensions
        #     clip_image = clip_image.resize((384, 384), Image.Resampling.LANCZOS)
    
        # 6. Dual text encoding: Process "rare text" and "subject text" separately
        # 6.1 Encode rare text (original prompt logic)
        # prompt_2=None,
        # prompt_3=None,
        # negative_prompt_2=None
        # negative_prompt_3=None
        
        (
            prompt_embeds_rare,
            negative_prompt_embeds_rare,
            pooled_prompt_embeds_rare,
            negative_pooled_prompt_embeds_rare,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds_rare,
            negative_prompt_embeds=negative_prompt_embeds_rare,
            pooled_prompt_embeds=pooled_prompt_embeds_rare,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds_rare,
            device=device,
            clip_skip=self._clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
    
        # 6.2 Encode subject text (new logic)
        # prompt_2/prompt_3 for subject text reuse itself (to avoid mixing with other text)
        prompt_2_subject = prompt_subject
        prompt_3_subject = prompt_subject
        # prompt_2=None,
        # prompt_3=None,
        # negative_prompt_2=None
        # negative_prompt_3=None
        # When not provided by the user, the negative prompt for subject text defaults to being the same as the negative prompt for rare text.
        negative_prompt_subject = negative_prompt_subject or negative_prompt
        negative_prompt_2_subject = negative_prompt_subject
        negative_prompt_3_subject = negative_prompt_subject
    
        (
            prompt_embeds_subject,
            negative_prompt_embeds_subject,
            pooled_prompt_embeds_subject,
            negative_pooled_prompt_embeds_subject,
        ) = self.encode_prompt(
            prompt=prompt_subject,
            prompt_2=prompt_2_subject,
            prompt_3=prompt_3_subject,
            negative_prompt=negative_prompt_subject,
            negative_prompt_2=negative_prompt_2_subject,
            negative_prompt_3=negative_prompt_3_subject,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds_subject,
            negative_prompt_embeds=negative_prompt_embeds_subject,
            pooled_prompt_embeds=pooled_prompt_embeds_subject,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds_subject,
            device=device,
            clip_skip=self._clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
    
        # 6.3 Process Classifier-Free Guidance: Concatenate negative embeddings + positive embeddings
        if self.do_classifier_free_guidance:
            # Rare text embedding: Negative + Positive
            prompt_embeds_rare = torch.cat([negative_prompt_embeds_rare, prompt_embeds_rare], dim=0)
            pooled_prompt_embeds_rare = torch.cat([negative_pooled_prompt_embeds_rare, pooled_prompt_embeds_rare], dim=0)
            # Subject text embedding: Negative + Positive
            prompt_embeds_subject = torch.cat([negative_prompt_embeds_subject, prompt_embeds_subject], dim=0)
            pooled_prompt_embeds_subject = torch.cat([negative_pooled_prompt_embeds_subject, pooled_prompt_embeds_subject], dim=0)
    
        # 7. Image encoding: Generate shared image embedding (reused by both texts)
        clip_image = clip_image.resize((max(clip_image.size), max(clip_image.size)))
        clip_image_embeds = self.encode_clip_image_emb(clip_image, device, dtype)
    
        # 8. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        # Control timesteps for image injection
        inject_start_ratio = 0.1
        inject_end_ratio = 0.9
        inject_start_idx = int(num_inference_steps * inject_start_ratio)  # Start index (at 30% position)
        inject_end_idx = int(num_inference_steps * inject_end_ratio)    # End index (at 70% position)
        print(f"IPAdapter injection effective range: Step {inject_start_idx} ~ {inject_end_idx} (total {num_inference_steps} steps)")

        
    
        # 9. Prepare Latent
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds_rare.dtype,
            device,
            generator,
            latents,
        )
    
        # 10. Core: Denoising loop (dual noise generation + weighted fusion)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    break
                
                is_inject_step = inject_start_idx <= i < inject_end_idx
                # Print again (log reflects actual values)
                # print(f"inject_start_idx：{inject_start_idx}")
                # print(f"inject_end_idx：{inject_end_idx}")
                # print(f"i：{i}")  # It is recommended to add a print statement for i to facilitate troubleshooting interval judgment issues.
                # print(f"is_inject_step：{is_inject_step}")
                ipadapter_scale1 = ipadapter_scale if is_inject_step else 0.0  # If not within the interval, set scale=0 (disable injection)
                # print(f"ipadapter_scale1为：{ipadapter_scale1}")
    
                # 10.1 Expand Latent (adapt for CFG)
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0])
    
                # 10.2 Generate image projection embedding (updated once per timestep)
                image_prompt_embeds, timestep_emb = self.image_proj_model(
                    clip_image_embeds,
                    timestep.to(dtype=latents.dtype),
                    need_temb=True
                )
                # 10.3 Generate noise a: Rare text + Image embedding
                joint_kwargs_rare = dict(
                    emb_dict=dict(
                        ip_hidden_states=image_prompt_embeds,
                        temb=timestep_emb,
                        scale=ipadapter_scale1,
                        text_type="rare"  # Debug flag, optional
                    )
                )
                noise_a = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds_rare,
                    pooled_projections=pooled_prompt_embeds_rare,
                    joint_attention_kwargs=joint_kwargs_rare,
                    return_dict=False,
                )[0]
    
                # 10.4 Generate noise b: Subject text + Image embedding
                joint_kwargs_subject = dict(
                    emb_dict=dict(
                        ip_hidden_states=image_prompt_embeds,
                        temb=timestep_emb,
                        scale=ipadapter_scale1,
                        text_type="subject"  # Debug flag, optional
                    )
                )
                noise_b = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds_subject,
                    pooled_projections=pooled_prompt_embeds_subject,
                    joint_attention_kwargs=joint_kwargs_subject,
                    return_dict=False,
                )[0]
    
                # 10.5 Weighted fusion of noises (counterfactual logic: balancing rare attributes and subject morphology)
                if self.do_classifier_free_guidance:
                    # Split noise a into unconditional (negative prompt) / conditional (positive prompt) parts
                    noise_uncond_a, noise_cond_a = noise_a.chunk(2)
                    # Split the conditional part of noise b (no unconditional needed for subject text, as the subject serves as the morphological baseline)
                    _, noise_cond_b = noise_b.chunk(2)

                    noise_pred = noise_uncond_a + 5*(noise_cond_a-noise_uncond_a) + 5*(noise_cond_a-noise_cond_b)

                    # noise_pred = (guidance_scale_rare + guidance_scale_subject - 1) * noise_cond_a \
                    #              + (1 - guidance_scale_rare) * noise_uncond_a \
                    #              + (1 - guidance_scale_subject) * noise_cond_b
                else:
                    # Direct weighting when CFG is disabled
                    noise_pred = (guidance_scale_rare * noise_a + guidance_scale_subject * noise_b) / (guidance_scale_rare + guidance_scale_subject)
    
                # 10.6 Update Latent (denoising step)
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)
    
                # 10.7 Callback function (retain original logic)
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    # Update parameters returned by the callback
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds_rare = callback_outputs.pop("prompt_embeds_rare", prompt_embeds_rare)
                    prompt_embeds_subject = callback_outputs.pop("prompt_embeds_subject", prompt_embeds_subject)
                    negative_prompt_embeds_rare = callback_outputs.pop("negative_prompt_embeds_rare", negative_prompt_embeds_rare)
                    negative_prompt_embeds_subject = callback_outputs.pop("negative_prompt_embeds_subject", negative_prompt_embeds_subject)
    
                # 10.8 Progress update
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
    
                # XLA platform compatibility
                if XLA_AVAILABLE:
                    xm.mark_step()
    
        # 11. Image post-processing (retain original logic)
        if output_type == "latent":
            image = latents
        else:
            # VAE decoding: adjust scale and shift
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            # Image post-processing (convert to PIL/NDArray)
            image = self.image_processor.postprocess(image, output_type=output_type)
    
        # 12. Release model hooks
        self.maybe_free_model_hooks()
    
        # 13. Return results
        if not return_dict:
            return (image,)
        return StableDiffusion3PipelineOutput(images=image)
