"""One-reference Krea 2 Edit inference nodes for ComfyUI.

The implementation mirrors diffusion-pipe's ``krea2_edit`` training contract:

* the reference image is encoded by the VAE and appended after the noisy
  target as clean tokens at model timestep zero and RoPE frame one;
* the same reference image is shown to Qwen3-VL for both positive and negative
  conditioning, so CFG changes the instruction rather than the reference;
* only target tokens are returned by the patched diffusion model.

The recommended setup node keeps the image, both prompts, VAE latent and model
patch in one graph operation. The modular nodes expose the same pieces for
advanced workflows without encoding the VAE reference twice.
"""

from dataclasses import dataclass
import math
import re

import torch
from einops import rearrange

import comfy.conds
import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_sampling
import comfy.patcher_extension
import comfy.utils
import folder_paths
import node_helpers
from comfy.ldm.flux.layers import timestep_embedding
from comfy.text_encoders.krea2 import KREA2_TEMPLATE


REFERENCE_TYPE = "CTXRUSH_KREA2_REFERENCE"
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
DEFAULT_VL_MAX_PIXELS = 384 * 384
PUBLIC_REFERENCE_MAX_PIXELS = 1024 * 1024
REFERENCE_SNAP = 16

# Krea 2 Raw official mu interpolation in image-token space: 256px -> mu 0.5,
# 1280px -> mu 1.15 (same constants as tools/krea2_sampling.py). ComfyUI's
# stock Krea 2 config uses a fixed mu of 1.15, which matches Turbo but roughly
# doubles the effective shift for Raw at 672x384.
KREA2_MU_TOKENS_MIN = 256
KREA2_MU_TOKENS_MAX = 6400
KREA2_MU_MIN = 0.5
KREA2_MU_MAX = 1.15
KREA2_TURBO_MU = 1.15


def _krea2_raw_mu(width, height):
    tokens = (width // 16) * (height // 16)
    slope = (KREA2_MU_MAX - KREA2_MU_MIN) / (KREA2_MU_TOKENS_MAX - KREA2_MU_TOKENS_MIN)
    return slope * tokens + (KREA2_MU_MIN - slope * KREA2_MU_TOKENS_MIN)


def _apply_krea2_sampling(patched_model, mu):
    """Set the model's flow shift to ``mu``, mirroring the validated runner."""

    class _Krea2ModelSampling(
        comfy.model_sampling.ModelSamplingFlux, comfy.model_sampling.CONST
    ):
        pass

    model_sampling = _Krea2ModelSampling(patched_model.model.model_config)
    model_sampling.set_parameters(shift=mu)
    patched_model.add_object_patch("model_sampling", model_sampling)
    return patched_model


@dataclass(frozen=True)
class Krea2Reference:
    """A paired visual/Qwen reference encoded once for both CFG branches."""

    vl_image: torch.Tensor
    latent: torch.Tensor
    fit_mode: str
    target_width: int
    target_height: int


def _require_single_image(image):
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(
            "CtxRush Krea 2 Edit expects one ComfyUI IMAGE in BHWC layout."
        )
    if image.shape[0] != 1:
        raise ValueError(
            "CtxRush Krea 2 Edit supports exactly one reference image. "
            "Use a batch size of one on Load Image."
        )
    return image[..., :3]


def _fit_area(image, max_pixels, snap=1):
    """Downscale only, preserving aspect ratio and optionally snapping size."""
    samples = image.movedim(-1, 1)
    height, width = samples.shape[-2:]
    scale = min(1.0, math.sqrt(max_pixels / (height * width)))
    new_width = max(round(width * scale / snap) * snap, snap)
    new_height = max(round(height * scale / snap) * snap, snap)
    if (new_height, new_width) == (height, width):
        return image
    samples = comfy.utils.common_upscale(
        samples, new_width, new_height, "area", "disabled"
    )
    return samples.movedim(1, -1)


def _fit_vl(image, max_pixels):
    """Resize the Qwen3-VL copy exactly like training's prepare_vl_image:
    aspect-preserving downscale-only, bicubic with antialias, 28px floor per
    side (models/krea2_edit.py). The generic _fit_area (area kernel, 1px
    floor) stays for the VAE branch only."""
    samples = image.movedim(-1, 1)
    height, width = samples.shape[-2:]
    scale = min(1.0, math.sqrt(max_pixels / (height * width)))
    new_height = max(round(height * scale), 28)
    new_width = max(round(width * scale), 28)
    if (new_height, new_width) == (height, width):
        return image
    samples = torch.nn.functional.interpolate(
        samples.float(),
        size=(new_height, new_width),
        mode="bicubic",
        antialias=True,
    ).clamp(0.0, 1.0).to(image.dtype)
    return samples.movedim(1, -1)


def _crop_fit(image, width, height):
    """Match diffusion-pipe's same-bucket center-crop reference contract."""
    samples = image.movedim(-1, 1)
    source_height, source_width = samples.shape[-2:]
    scale = max(width / source_width, height / source_height)
    resized_width = max(round(source_width * scale), width)
    resized_height = max(round(source_height * scale), height)
    samples = comfy.utils.common_upscale(
        samples, resized_width, resized_height, "lanczos", "disabled"
    )
    top = (resized_height - height) // 2
    left = (resized_width - width) // 2
    return samples[:, :, top : top + height, left : left + width].movedim(1, -1)


def _fit_vl_longest(image, longest_side):
    """conradlocke grounding_px semantics: cap the LONGEST side, area resample."""
    samples = image.movedim(-1, 1)
    height, width = samples.shape[-2:]
    if longest_side and max(height, width) > longest_side:
        scale = longest_side / max(height, width)
        new_h = max(round(height * scale), 28)
        new_w = max(round(width * scale), 28)
        samples = comfy.utils.common_upscale(samples, new_w, new_h, "area", "disabled")
    return samples.movedim(1, -1)


def _build_reference(
    vae,
    image,
    width,
    height,
    fit_mode="training_crop",
    vl_image_max_pixels=DEFAULT_VL_MAX_PIXELS,
    vl_longest_side=0,
):
    image = _require_single_image(image)
    if width % REFERENCE_SNAP or height % REFERENCE_SNAP:
        raise ValueError("Target width and height must be multiples of 16.")

    if vl_longest_side:
        vl_image = _fit_vl_longest(image, vl_longest_side)
    else:
        vl_image = _fit_vl(image, vl_image_max_pixels)
    if fit_mode == "training_crop":
        vae_image = _crop_fit(image, width, height)
    elif fit_mode == "preserve_aspect_1mp":
        vae_image = _fit_area(
            image, PUBLIC_REFERENCE_MAX_PIXELS, snap=REFERENCE_SNAP
        )
    else:
        raise ValueError(f"Unknown reference fit mode: {fit_mode}")

    latent = vae.encode(vae_image)
    return Krea2Reference(
        vl_image=vl_image,
        latent=latent,
        fit_mode=fit_mode,
        target_width=width,
        target_height=height,
    )


def _encode_conditioning(clip, prompt, reference, vl_prompt_style="picture_n"):
    if vl_prompt_style == "plain":
        # conradlocke layout: bare vision block, no "Picture 1:" prefix.
        text = f"{VISION_BLOCK}{prompt}"
    else:
        text = f"Picture 1: {VISION_BLOCK}{prompt}"
    try:
        tokens = clip.tokenize(
            text,
            images=[reference.vl_image],
            llama_template=KREA2_TEMPLATE,
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)
    except Exception as error:
        raise RuntimeError(
            "Krea 2 visual conditioning failed. Use a Krea 2 CLIP/text encoder "
            "checkpoint that includes the Qwen3-VL visual.* weights."
        ) from error

    return node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents": [reference.latent]},
        append=True,
    )


def _empty_krea_latent(width, height, batch_size):
    latent = torch.zeros(
        [batch_size, 16, height // 8, width // 8],
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    )
    return {"samples": latent, "downscale_ratio_spacial": 8}


def _pack_reference(dit, reference, batch_size, device, dtype):
    """Patchify one processed reference and assign fixed RoPE frame one."""
    if reference.ndim == 5:
        ref_batch, channels, frames, height, width = reference.shape
        if frames != 1:
            raise ValueError(
                "CtxRush Krea 2 Edit supports one reference frame, "
                f"but received {frames}."
            )
        reference = reference.reshape(
            ref_batch * frames, channels, height, width
        )
    if reference.ndim != 4:
        raise ValueError(
            "The processed Krea 2 reference latent must be BCHW or BCTHW."
        )

    reference = comfy.ldm.common_dit.pad_to_patch_size(
        reference.to(device=device, dtype=dtype), (dit.patch, dit.patch)
    )
    reference = comfy.utils.repeat_to_batch_size(reference, batch_size)
    grid_height = reference.shape[-2] // dit.patch
    grid_width = reference.shape[-1] // dit.patch
    tokens = rearrange(
        reference,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=dit.patch,
        pw=dit.patch,
    )

    positions = torch.zeros(
        grid_height,
        grid_width,
        3,
        device=device,
        dtype=torch.float32,
    )
    positions[..., 0] = 1.0
    positions[..., 1] = torch.arange(
        grid_height, device=device, dtype=torch.float32
    )[:, None]
    positions[..., 2] = torch.arange(
        grid_width, device=device, dtype=torch.float32
    )[None, :]
    positions = positions.reshape(1, grid_height * grid_width, 3).repeat(
        batch_size, 1, 1
    )
    return tokens, positions


def _block_with_clean_reference(
    block,
    hidden_states,
    target_timestep,
    clean_timestep,
    reference_start,
    frequencies,
    transformer_options,
):
    """Apply Krea modulation at sampled t to target/text and t=0 to reference."""
    target_mod = block.mod(target_timestep)
    reference_mod = block.mod(clean_timestep)

    def modulate(states, scale_index, shift_index):
        return torch.cat(
            (
                (1 + target_mod[scale_index]) * states[:, :reference_start]
                + target_mod[shift_index],
                (1 + reference_mod[scale_index]) * states[:, reference_start:]
                + reference_mod[shift_index],
            ),
            dim=1,
        )

    def gate(states, gate_index):
        return torch.cat(
            (
                target_mod[gate_index] * states[:, :reference_start],
                reference_mod[gate_index] * states[:, reference_start:],
            ),
            dim=1,
        )

    attention_input = modulate(block.prenorm(hidden_states), 0, 1)
    attention_output = block.attn(
        attention_input,
        frequencies,
        None,
        transformer_options=transformer_options,
    )
    hidden_states = hidden_states + gate(attention_output, 2)
    mlp_input = modulate(block.postnorm(hidden_states), 3, 4)
    hidden_states = hidden_states + gate(block.mlp(mlp_input), 5)
    return hidden_states


def _forward_with_reference(
    dit,
    x,
    timesteps,
    context,
    reference_latents,
    transformer_options,
    reference_timestep="zero",
):
    if len(reference_latents) != 1:
        raise ValueError(
            "This CtxRush adapter was trained with exactly one reference image."
        )

    temporal = x.ndim == 5
    if temporal:
        batch_5d, channels_5d, frames_5d, height_5d, width_5d = x.shape
        x = x.reshape(batch_5d * frames_5d, channels_5d, height_5d, width_5d)

    batch_size, _, original_height, original_width = x.shape
    patch = dit.patch
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    grid_height = x.shape[-2] // patch
    grid_width = x.shape[-1] // patch

    context = dit._unpack_context(context)
    target_tokens = rearrange(
        x,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=patch,
        pw=patch,
    )
    reference_tokens, reference_positions = _pack_reference(
        dit,
        reference_latents[0],
        batch_size,
        x.device,
        x.dtype,
    )
    image_tokens = dit.first(torch.cat((target_tokens, reference_tokens), dim=1))

    target_features = dit.tmlp(
        timestep_embedding(timesteps, dit.tdim).unsqueeze(1).to(image_tokens.dtype)
    )
    target_timestep = dit.tproj(target_features)
    if reference_timestep == "target":
        # conradlocke convention: one modulation timestep for the whole
        # sequence (adapters trained with reference_timestep='target').
        clean_timestep = target_timestep
    else:
        clean_features = dit.tmlp(
            timestep_embedding(torch.zeros_like(timesteps), dit.tdim)
            .unsqueeze(1)
            .to(image_tokens.dtype)
        )
        clean_timestep = dit.tproj(clean_features)

    context = dit.txtfusion(
        context, mask=None, transformer_options=transformer_options
    )
    context = dit.txtmlp(context)
    text_length = context.shape[1]
    target_length = target_tokens.shape[1]
    reference_start = text_length + target_length
    hidden_states = torch.cat((context, image_tokens), dim=1)

    text_positions = torch.zeros(
        batch_size,
        text_length,
        3,
        device=x.device,
        dtype=torch.float32,
    )
    target_positions = torch.zeros(
        grid_height,
        grid_width,
        3,
        device=x.device,
        dtype=torch.float32,
    )
    target_positions[..., 1] = torch.arange(
        grid_height, device=x.device, dtype=torch.float32
    )[:, None]
    target_positions[..., 2] = torch.arange(
        grid_width, device=x.device, dtype=torch.float32
    )[None, :]
    target_positions = target_positions.reshape(
        1, grid_height * grid_width, 3
    ).repeat(batch_size, 1, 1)
    frequencies = dit.pe_embedder(
        torch.cat((text_positions, target_positions, reference_positions), dim=1)
    )

    for block in dit.blocks:
        hidden_states = _block_with_clean_reference(
            block,
            hidden_states,
            target_timestep,
            clean_timestep,
            reference_start,
            frequencies,
            transformer_options,
        )

    output = dit.last(hidden_states, target_features)
    output = output[:, text_length:reference_start]
    output = rearrange(
        output,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=grid_height,
        w=grid_width,
        ph=patch,
        pw=patch,
        c=dit.channels,
    )
    output = output[:, :, :original_height, :original_width]
    if temporal:
        output = output.reshape(
            batch_5d, frames_5d, dit.channels, original_height, original_width
        ).movedim(1, 2)
    return output


def _patch_model(model, reference_timestep="zero", lora_path=None, lora_strength=1.0):
    patched = model.clone()
    base_model = patched.model
    dit = patched.get_model_object("diffusion_model")
    required = (
        "patch",
        "channels",
        "blocks",
        "txtfusion",
        "txtmlp",
        "pe_embedder",
        "_unpack_context",
    )
    missing = [name for name in required if not hasattr(dit, name)]
    if dit.__class__.__name__ != "SingleStreamDiT" or missing:
        detail = f"; missing attributes: {', '.join(missing)}" if missing else ""
        raise ValueError(
            "CtxRush Krea 2 Edit Model Patch requires the ComfyUI Krea 2 "
            f"SingleStreamDiT, got {dit.__class__.__name__}{detail}."
        )

    lora_entries = []
    if lora_path:
        pairs = _load_omini_lora(lora_path)
        named = dict(dit.named_modules())
        missing = []
        for path, (a, b) in pairs.items():
            module = named.get(path)
            if module is None:
                missing.append(path)
                continue
            lora_entries.append((module, a.cuda(), b.cuda()))
        if missing:
            print(f'[ctxrush_edit] WARNING: {len(missing)} LoRA keys not matched (e.g. {missing[:3]})')
        print(f'[ctxrush_edit] runtime LoRA: {len(lora_entries)} linears @ strength {lora_strength}')

    original_extra_conds = base_model.extra_conds
    original_extra_conds_shapes = base_model.extra_conds_shapes
    original_forward = dit.forward

    def extra_conds(**kwargs):
        output = original_extra_conds(**kwargs)
        references = kwargs.get("reference_latents")
        if references:
            output["ctxrush_reference_latents"] = comfy.conds.CONDList(
                [base_model.process_latent_in(latent) for latent in references]
            )
        return output

    def extra_conds_shapes(**kwargs):
        output = original_extra_conds_shapes(**kwargs)
        references = kwargs.get("reference_latents")
        if references:
            total_elements = sum(math.prod(reference.size()) for reference in references)
            output["ctxrush_reference_latents"] = [1, 16, total_elements // 16]
        return output

    def forward(
        x,
        timesteps,
        context,
        attention_mask=None,
        transformer_options=None,
        ctxrush_reference_latents=None,
        **kwargs,
    ):
        options = {} if transformer_options is None else transformer_options
        if not ctxrush_reference_latents:
            return original_forward(
                x,
                timesteps,
                context,
                attention_mask=attention_mask,
                transformer_options=options,
                **kwargs,
            )
        if lora_entries:
            with _FullLoraScope(lora_entries, lora_strength):
                return _forward_with_reference(
                    dit, x, timesteps, context, ctxrush_reference_latents,
                    options, reference_timestep=reference_timestep,
                )
        return _forward_with_reference(
            dit,
            x,
            timesteps,
            context,
            ctxrush_reference_latents,
            options,
            reference_timestep=reference_timestep,
        )

    patched.add_object_patch("extra_conds", extra_conds)
    patched.add_object_patch("extra_conds_shapes", extra_conds_shapes)
    patched.add_object_patch("diffusion_model.forward", forward)
    return patched


REFERENCE_INPUTS = {
    "vae": ("VAE", {"tooltip": "Qwen Image VAE used by Krea 2."}),
    "reference": (
        "IMAGE",
        {
            "tooltip": (
                "The single source/control image. It is used by both the VAE "
                "detail path and the Qwen3-VL semantic path."
            )
        },
    ),
    "width": (
        "INT",
        {
            "default": 672,
            "min": 64,
            "max": 8192,
            "step": 16,
            "tooltip": "Target generation width. Use the evaluated training bucket first.",
        },
    ),
    "height": (
        "INT",
        {
            "default": 384,
            "min": 64,
            "max": 8192,
            "step": 16,
            "tooltip": "Target generation height. Use the evaluated training bucket first.",
        },
    ),
    "reference_fit": (
        ["training_crop", "preserve_aspect_1mp"],
        {
            "default": "training_crop",
            "tooltip": (
                "training_crop center-crops the reference to the target bucket and "
                "matches this diffusion-pipe LoRA. preserve_aspect_1mp matches public "
                "Ostris/ai-toolkit edit LoRAs but is out of distribution for this run."
            ),
        },
    ),
    "vl_image_max_pixels": (
        "INT",
        {
            "default": DEFAULT_VL_MAX_PIXELS,
            "min": 784,
            "max": 1048576,
            "step": 784,
            "tooltip": (
                "Pixel budget shown to Qwen3-VL. 147456 (384x384 area) is the "
                "training value; high-resolution detail comes from the VAE path."
            ),
        },
    ),
}


class CtxRushKrea2ReferenceEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": dict(REFERENCE_INPUTS)}

    RETURN_TYPES = (REFERENCE_TYPE,)
    RETURN_NAMES = ("reference",)
    FUNCTION = "encode"
    CATEGORY = "CtxRush/Krea 2 Edit"
    DESCRIPTION = (
        "Encode one Krea 2 Edit reference once for both CFG branches. The "
        "output contains the VAE latent and the downscaled Qwen3-VL image."
    )

    def encode(
        self,
        vae,
        reference,
        width,
        height,
        reference_fit,
        vl_image_max_pixels,
    ):
        return (
            _build_reference(
                vae,
                reference,
                width,
                height,
                reference_fit,
                vl_image_max_pixels,
            ),
        )


class CtxRushKrea2EditCFGEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": (
                    "CLIP",
                    {
                        "tooltip": (
                            "Krea 2 Qwen3-VL text encoder with visual.* weights."
                        )
                    },
                ),
                "reference": (
                    REFERENCE_TYPE,
                    {
                        "tooltip": (
                            "Reference produced by CtxRush Krea 2 Reference Encode."
                        )
                    },
                ),
                "positive_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": (
                            "Describe the target/next scene, not the source image."
                        ),
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": (
                            "Negative instruction. The reference remains grounded in "
                            "this branch so CFG contrasts text only."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "CtxRush/Krea 2 Edit"
    DESCRIPTION = (
        "Encode positive and negative Krea 2 conditioning with the same visual "
        "reference attached to both branches."
    )

    def encode(self, clip, reference, positive_prompt, negative_prompt=""):
        return (
            _encode_conditioning(clip, positive_prompt, reference),
            _encode_conditioning(clip, negative_prompt, reference),
        )


class CtxRushKrea2EditModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "Krea 2 model after loading the edit LoRA. The patch is a "
                            "no-op when conditioning contains no reference."
                        )
                    },
                ),
                "model_variant": (
                    ["raw", "turbo"],
                    {
                        "default": "raw",
                        "tooltip": (
                            "Raw derives the flow shift (mu) from the output "
                            "resolution like the official Krea sampler; Turbo "
                            "keeps the fixed mu 1.15."
                        ),
                    },
                ),
                "width": (
                    "INT",
                    {"default": 672, "min": 64, "max": 4096, "step": 16,
                     "tooltip": "Output width, used to derive the Raw flow shift."},
                ),
                "height": (
                    "INT",
                    {"default": 384, "min": 64, "max": 4096, "step": 16,
                     "tooltip": "Output height, used to derive the Raw flow shift."},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "CtxRush/Krea 2 Edit"
    DESCRIPTION = (
        "Enable the clean-reference sequence used by diffusion-pipe krea2_edit: "
        "text, noisy target, clean reference at t=0/RoPE frame 1. Also sets the "
        "resolution-dependent Raw flow shift (ComfyUI's stock Krea 2 config uses "
        "Turbo's fixed mu 1.15, which over-shifts Raw)."
    )

    def patch(self, model, model_variant="raw", width=672, height=384):
        mu = _krea2_raw_mu(width, height) if model_variant == "raw" else KREA2_TURBO_MU
        return (_apply_krea2_sampling(_patch_model(model), mu),)


class CtxRushKrea2EditSetup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "Krea 2 model after loading the edit LoRA."},
                ),
                "clip": (
                    "CLIP",
                    {
                        "tooltip": (
                            "Krea 2 Qwen3-VL text encoder with visual.* weights."
                        )
                    },
                ),
                **REFERENCE_INPUTS,
                "positive_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Describe the desired target/next scene.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": (
                            "Negative instruction; the visual reference is retained."
                        ),
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Number of targets generated from the same reference.",
                    },
                ),
                "model_variant": (
                    ["raw", "turbo"],
                    {
                        "default": "raw",
                        "tooltip": (
                            "Returns safe sampler defaults: Raw=28 steps/CFG 5.5; "
                            "Turbo=8 steps/CFG 1.0."
                        ),
                    },
                ),
                "adapter_contract": (
                    ["ostris_t0_picture", "conrad_target_plain"],
                    {
                        "default": "ostris_t0_picture",
                        "tooltip": (
                            "Contrato com que o adapter foi TREINADO. "
                            "ostris_t0_picture: refs a t=0, template 'Picture 1:', grounding área 384² "
                            "(adapters v1/v2 ostris-style). "
                            "conrad_target_plain: refs no timestep do target, vision block sem prefixo, "
                            "grounding maior-lado 768 (adapters conrad-style, ex. ctxrush_conrad750)."
                        ),
                    },
                ),
            },
            "optional": {
                "lora_name": (
                    ["none"] + folder_paths.get_filename_list("loras"),
                    {
                        "default": "none",
                        "tooltip": (
                            "APLICAÇÃO RUNTIME (recomendado p/ base fp8): o delta do LoRA é "
                            "somado em bf16 a cada forward, sem fundir nos pesos — o merge do "
                            "Load LoRA padrão requantiza W+ΔW para fp8 e afoga o delta de "
                            "adapters jovens. Se usar isto, NÃO use Load LoRA no model."
                        ),
                    },
                ),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "INT",
        "FLOAT",
    )
    RETURN_NAMES = ("model", "positive", "negative", "latent", "steps", "cfg")
    FUNCTION = "setup"
    CATEGORY = "CtxRush/Krea 2 Edit"
    DESCRIPTION = (
        "Recommended all-in-one Krea 2 Edit setup. Encodes one reference once, "
        "grounds both CFG branches, patches the model and creates the target latent."
    )

    def setup(
        self,
        model,
        clip,
        vae,
        reference,
        width,
        height,
        reference_fit,
        vl_image_max_pixels,
        positive_prompt,
        negative_prompt,
        batch_size,
        model_variant,
        adapter_contract="ostris_t0_picture",
        lora_name="none",
        lora_strength=1.0,
    ):
        conrad = adapter_contract == "conrad_target_plain"
        encoded_reference = _build_reference(
            vae,
            reference,
            width,
            height,
            reference_fit,
            vl_image_max_pixels,
            vl_longest_side=768 if conrad else 0,
        )
        style = "plain" if conrad else "picture_n"
        positive = _encode_conditioning(
            clip, positive_prompt, encoded_reference, vl_prompt_style=style
        )
        negative = _encode_conditioning(
            clip, negative_prompt, encoded_reference, vl_prompt_style=style
        )
        steps, cfg = (28, 5.5) if model_variant == "raw" else (8, 1.0)
        mu = _krea2_raw_mu(width, height) if model_variant == "raw" else KREA2_TURBO_MU
        lora_path = None
        if lora_name and lora_name != "none":
            lora_path = folder_paths.get_full_path("loras", lora_name)
        patched_model = _apply_krea2_sampling(
            _patch_model(
                model,
                reference_timestep="target" if conrad else "zero",
                lora_path=lora_path,
                lora_strength=lora_strength,
            ),
            mu,
        )
        return (
            patched_model,
            positive,
            negative,
            _empty_krea_latent(width, height, batch_size),
            steps,
            cfg,
        )


def _load_omini_lora(lora_path):
    """Parse a fork-format adapter (diffusion_model.blocks.*.lora_{A,B}.weight)
    into {module_path: (A, B)} bf16 tensors. alpha==rank in the fork, so the
    LoRA scale is exactly the user strength."""
    state = comfy.utils.load_torch_file(lora_path, safe_load=True)
    pairs = {}
    for key, value in state.items():
        if '.lora_A.' not in key and '.lora_B.' not in key:
            continue
        path = key.replace('diffusion_model.', '', 1)
        which = 'A' if '.lora_A.' in key else 'B'
        path = path.split('.lora_')[0]
        pairs.setdefault(path, {})[which] = value.to(torch.bfloat16)
    out = {p: (v['A'], v['B']) for p, v in pairs.items() if 'A' in v and 'B' in v}
    if not out:
        raise ValueError(f'No fork-format LoRA pairs found in {lora_path}')
    return out


class _FullLoraScope:
    """Runtime bf16 LoRA delta on every call of the targeted Linears (exact
    PEFT semantics), instead of merging into the weights. Merging into
    fp8-SCALED checkpoints requantizes W+ΔW to e4m3, drowning the small delta
    of young adapters — runtime application preserves it."""

    def __init__(self, entries, scale):
        self.entries = entries
        self.scale = scale
        self._originals = []

    def __enter__(self):
        scale = self.scale
        for module, lora_a, lora_b in self.entries:
            orig = module.forward

            def wrapped(x, *args, _orig=orig, _a=lora_a, _b=lora_b, **kwargs):
                out = _orig(x, *args, **kwargs)
                delta = torch.nn.functional.linear(
                    torch.nn.functional.linear(x.to(_a.dtype), _a), _b
                )
                return out + (delta * scale).to(out.dtype)

            self._originals.append((module, orig))
            module.forward = wrapped
        return self

    def __exit__(self, *exc):
        for module, orig in self._originals:
            module.forward = orig
        self._originals.clear()
        return False


class _MaskedLoraScope:
    """Temporarily wrap targeted Linears so the LoRA delta lands ONLY on the
    reference span — OminiControl's condition-only routing. ComfyUI's stock
    LoRA loader merges deltas into the weights (all rows), which is NOT
    equivalent for adapters trained with condition-only routing."""

    def __init__(self, entries, span_start, span_end, seq_len, scale):
        self.entries = entries
        self.span = (span_start, span_end)
        self.seq_len = seq_len
        self.scale = scale
        self._originals = []

    def __enter__(self):
        s, e = self.span
        seq_len, scale = self.seq_len, self.scale
        for entry in self.entries:
            module, lora_a, lora_b = entry[:3]
            entry_scale = entry[3] if len(entry) > 3 else 1.0
            orig = module.forward

            def wrapped(
                x,
                *args,
                _orig=orig,
                _a=lora_a,
                _b=lora_b,
                _entry_scale=entry_scale,
                **kwargs,
            ):
                out = _orig(x, *args, **kwargs)
                if x.ndim >= 3 and x.shape[-2] == seq_len:
                    piece = x[..., s:e, :].to(_a.dtype)
                    delta = torch.nn.functional.linear(
                        torch.nn.functional.linear(piece, _a), _b
                    )
                    out[..., s:e, :] += (delta * scale * _entry_scale).to(out.dtype)
                return out

            self._originals.append((module, orig))
            module.forward = wrapped
        return self

    def __exit__(self, *exc):
        for module, orig in self._originals:
            module.forward = orig
        self._originals.clear()
        return False


def _krea2_omini_forward(m, x, timesteps, context, src_latent, lora_state, strength, transformer_options,
                         reference_timestep='zero'):
    """OminiControl-true forward: [text | noisy target | clean ref], UNIFORM
    timestep modulation (training reference_timestep='target'), reference on
    width-shifted RoPE positions (frame axis 0, w += target grid width), LoRA
    delta masked to the reference span, output sliced to the target."""
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)
    bs, c, H_orig, W_orig = x.shape
    patch = m.patch
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2:]
    h_, w_ = H // patch, W // patch

    src = src_latent
    if src.ndim == 5:
        src = src.reshape(src.shape[0] * src.shape[2], src.shape[1], *src.shape[-2:])
    src = src.to(x.device, x.dtype)
    if src.shape[0] != bs:
        src = src[:1].expand(bs, *src.shape[1:])
    if src.shape[-2:] != (H, W):
        src = torch.nn.functional.interpolate(src.float(), size=(H, W), mode='bilinear').to(x.dtype)
    src = comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch))

    context = m._unpack_context(context)
    tgt = m.first(rearrange(x, 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=patch, pw=patch))
    ref = m.first(rearrange(src, 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=patch, pw=patch))

    t = m.tmlp(timestep_embedding(timesteps, m.tdim).unsqueeze(1).to(tgt.dtype))
    tvec = m.tproj(t)

    context = m.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = m.txtmlp(context)

    txtlen, tgtlen, reflen = context.shape[1], tgt.shape[1], ref.shape[1]
    combined = torch.cat([context, tgt, ref], dim=1)

    if reference_timestep == 'zero':
        # Per-token modulation: text+target at the sampled t, reference at 0
        # (the fork's clean-reference convention).
        t0 = m.tmlp(timestep_embedding(torch.zeros_like(timesteps), m.tdim).unsqueeze(1).to(tgt.dtype))
        tv0 = m.tproj(t0)
        tvec = torch.cat([
            tvec.expand(-1, txtlen + tgtlen, -1),
            tv0.expand(-1, reflen, -1),
        ], dim=1)

    device = combined.device
    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    grid = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    grid[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    grid[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    tgtpos = grid.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
    refpos = tgtpos.clone()
    refpos[..., 2] = refpos[..., 2] + float(w_)  # width-shift, frame axis 0
    freqs = m.pe_embedder(torch.cat([txtpos, tgtpos, refpos], dim=1))

    entries = lora_state['entries']
    if lora_state.get('device') != device:
        entries = [(mod, a.to(device), b.to(device)) for mod, a, b in entries]
        lora_state['entries'] = entries
        lora_state['device'] = device

    seq_len = txtlen + tgtlen + reflen
    with _MaskedLoraScope(entries, txtlen + tgtlen, seq_len, seq_len, strength):
        for block in m.blocks:
            combined = block(combined, tvec, freqs, None, transformer_options=transformer_options)

    final = m.last(combined, t)
    out = final[:, txtlen:txtlen + tgtlen, :]
    out = rearrange(out, 'b (h w) (c ph pw) -> b c (h ph) (w pw)', h=h_, w=w_, ph=patch, pw=patch, c=m.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, m.channels, H_orig, W_orig).movedim(1, 2)
    return out


class CtxRushKrea2OminiApply:
    """OminiControl-true inference for adapters trained by the fork with
    type=krea2_ominicontrol, position_mode=width_shift, reference_timestep=target."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL', {'tooltip': 'Krea 2 model WITHOUT any LoRA loader (the node applies the adapter itself, masked to the condition span).'}),
                'image': ('IMAGE', {'tooltip': 'Reference image (previous panel). The node crop-fits it to width/height and VAE-encodes at native resolution — the training geometry. Do NOT pre-encode with VAEEncode: resizing in latent space washes out the reference signal.'}),
                'vae': ('VAE',),
                'lora_name': (folder_paths.get_filename_list('loras'),),
                'strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.05}),
                'model_variant': (['raw', 'turbo'], {'default': 'raw', 'tooltip': 'Raw deriva o flow shift (mu) da resolução como o sampler oficial; Turbo usa mu fixo 1.15.'}),
                'reference_timestep': (['zero', 'target'], {'default': 'zero', 'tooltip': "Modulação dos tokens da referência. Adapters krea2_ominicontrol treinados ANTES do fix do to_layers (2026-07-17) usaram t=0 na prática, independente da metadata — use 'zero' para eles."}),
                'width': ('INT', {'default': 672, 'min': 64, 'max': 4096, 'step': 16}),
                'height': ('INT', {'default': 384, 'min': 64, 'max': 4096, 'step': 16}),
            }
        }

    RETURN_TYPES = ('MODEL',)
    RETURN_NAMES = ('model',)
    FUNCTION = 'apply'
    CATEGORY = 'CtxRush/Krea 2 Edit'
    DESCRIPTION = (
        'OminiControl-true for Krea 2: clean reference on width-shifted RoPE '
        'positions, uniform timestep modulation, and the LoRA delta applied '
        'ONLY to the reference tokens (stock LoRA loaders merge into weights, '
        'which is wrong for condition-only adapters). Use plain CLIP Text '
        'Encode conditioning; recommended Raw 28 steps / CFG 5.5.'
    )

    def apply(self, model, image, vae, lora_name, strength, model_variant='raw',
              reference_timestep='zero', width=672, height=384):
        lora_path = folder_paths.get_full_path('loras', lora_name)
        pairs = _load_omini_lora(lora_path)
        patched = model.clone()
        dit = patched.get_model_object('diffusion_model')
        named = dict(dit.named_modules())
        entries = []
        missing = []
        for path, (a, b) in pairs.items():
            module = named.get(path)
            if module is None:
                missing.append(path)
                continue
            entries.append((module, a, b))
        if not entries:
            raise ValueError('No LoRA target modules matched the diffusion model')
        if missing:
            print(f'[CtxRushKrea2OminiApply] WARNING: {len(missing)} LoRA keys not matched (e.g. {missing[:3]})')
        # Train-matched reference geometry: pixel-space center-crop to the
        # OUTPUT size, then a native VAE encode (the adapter was trained with
        # bucket-cropped references; bilinear latent downsampling of a large
        # source instead blurs the channels and weakens the conditioning).
        pixels = _crop_fit(image, width, height)
        latent = vae.encode(pixels[:, :, :, :3])
        src = patched.model.process_latent_in(latent)
        print(f'[CtxRushKrea2OminiApply] reference encoded at {width}x{height}, latent {tuple(latent.shape)}')
        lora_state = {'entries': entries, 'device': None}

        def wrapper(executor, x, timesteps, context, *args, **kwargs):
            transformer_options = kwargs.get('transformer_options')
            if transformer_options is None:
                transformer_options = next((a for a in args if isinstance(a, dict)), {})
            return _krea2_omini_forward(
                executor.class_obj, x, timesteps, context, src, lora_state, strength,
                transformer_options,
                reference_timestep=reference_timestep,
            )

        to = patched.model_options.setdefault('transformer_options', {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, 'ctxrush_omini', wrapper, to
        )
        mu = _krea2_raw_mu(width, height) if model_variant == 'raw' else KREA2_TURBO_MU
        return (_apply_krea2_sampling(patched, mu),)


def _krea2_omini_grounded_forward(m, x, timesteps, context, src_latent, blocks_state,
                                  fusion_entries, strength, transformer_options,
                                  fusion_strength=None, reference_timestep='zero'):
    """Omini-Grounded forward: identical geometry/timestep to the omini node
    (width-shift, refs at t=0 per-token, masked block deltas) but the context
    is GROUNDED (encoded with the reference through Qwen3-VL) and the
    txtfusion LoRA applies globally on the text stream."""
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)
    bs, c, H_orig, W_orig = x.shape
    patch = m.patch
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2:]
    h_, w_ = H // patch, W // patch

    src = src_latent
    if src.ndim == 5:
        src = src.reshape(src.shape[0] * src.shape[2], src.shape[1], *src.shape[-2:])
    src = src.to(x.device, x.dtype)
    if src.shape[0] != bs:
        src = src[:1].expand(bs, *src.shape[1:])
    if src.shape[-2:] != (H, W):
        src = torch.nn.functional.interpolate(src.float(), size=(H, W), mode='bilinear').to(x.dtype)
    src = comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch))

    context = m._unpack_context(context)
    tgt = m.first(rearrange(x, 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=patch, pw=patch))
    ref = m.first(rearrange(src, 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=patch, pw=patch))

    t = m.tmlp(timestep_embedding(timesteps, m.tdim).unsqueeze(1).to(tgt.dtype))
    tvec_t = m.tproj(t)

    fusion_scale = strength if fusion_strength is None else fusion_strength
    use_fusion = bool(fusion_entries) and fusion_scale > 0
    with _FullLoraScope(fusion_entries, fusion_scale) if use_fusion else _NullScope():
        context = m.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = m.txtmlp(context)

    txtlen, tgtlen, reflen = context.shape[1], tgt.shape[1], ref.shape[1]
    combined = torch.cat([context, tgt, ref], dim=1)

    if reference_timestep == 'target':
        ref_tvec = tvec_t
    else:
        t0 = m.tmlp(timestep_embedding(torch.zeros_like(timesteps), m.tdim).unsqueeze(1).to(tgt.dtype))
        ref_tvec = m.tproj(t0)
    tvec = torch.cat([
        tvec_t.expand(-1, txtlen + tgtlen, -1),
        ref_tvec.expand(-1, reflen, -1),
    ], dim=1)

    device = combined.device
    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    grid = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    grid[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    grid[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    tgtpos = grid.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
    refpos = tgtpos.clone()
    refpos[..., 2] = refpos[..., 2] + float(w_)
    freqs = m.pe_embedder(torch.cat([txtpos, tgtpos, refpos], dim=1))

    entries = blocks_state['entries']
    if blocks_state.get('device') != device:
        entries = [
            (entry[0], entry[1].to(device), entry[2].to(device), *entry[3:])
            for entry in entries
        ]
        blocks_state['entries'] = entries
        blocks_state['device'] = device

    seq_len = txtlen + tgtlen + reflen
    with _MaskedLoraScope(entries, txtlen + tgtlen, seq_len, seq_len, strength):
        for block in m.blocks:
            combined = block(combined, tvec, freqs, None, transformer_options=transformer_options)

    final = m.last(combined, t)
    out = final[:, txtlen:txtlen + tgtlen, :]
    out = rearrange(out, 'b (h w) (c ph pw) -> b c (h ph) (w pw)', h=h_, w=w_, ph=patch, pw=patch, c=m.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, m.channels, H_orig, W_orig).movedim(1, 2)
    return out


class _NullScope:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _ctxrush_schedule_mult(sigma, start_percent, end_percent, curve, power):
    """Return the strength multiplier for the current denoise progress."""
    progress = 1.0 - max(0.0, min(1.0, float(sigma)))
    span = max(1e-6, end_percent - start_percent)
    if progress < start_percent or progress > end_percent:
        return 0.0
    u = (progress - start_percent) / span
    if curve == 'constant':
        value = 1.0
    elif curve == 'fade_out':
        value = 1.0 - u
    elif curve == 'fade_in':
        value = u
    else:  # fade_in_out
        value = 1.0 - abs(2.0 * u - 1.0)
    return max(0.0, value) ** max(0.05, float(power))


def _ctxrush_layer_scales(entries_with_path, layers, layer_taper):
    """Filter LoRA entries by block range and attach a per-layer scale."""
    if layers.strip().lower() in ('', 'all'):
        low, high = 0, 10**9
    else:
        try:
            low, high = (int(value) for value in layers.replace(' ', '').split('-'))
        except Exception as error:
            raise ValueError(
                f"layers deve ser 'all' ou 'inicio-fim' (ex. 4-24), recebi: {layers!r}"
            ) from error

    output = []
    for module, lora_a, lora_b, path in entries_with_path:
        match = re.search(r'blocks\.(\d+)\.', path)
        index = int(match.group(1)) if match else 0
        if not (low <= index <= high):
            continue
        if layer_taper == 'flat' or high <= low:
            scale = 1.0
        else:
            position = (index - low) / max(1, min(high, 27) - low)
            scale = (
                1.0 - 0.8 * position
                if layer_taper == 'fade_deep'
                else 0.2 + 0.8 * position
            )
        output.append((module, lora_a, lora_b, scale))
    if not output:
        raise ValueError(f'Nenhum block na faixa de layers {layers!r}')
    return output


class CtxRushKrea2OminiGroundedApply:
    """All-in-one for adapters trained with type=krea2_omini_grounded: omini
    core (width-shift, refs t=0, condition-only block LoRA) + Qwen3-VL
    grounded conditioning + global txtfusion LoRA, applied at runtime in bf16."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL', {'tooltip': 'Krea 2 model SEM Load LoRA (o node aplica o adapter em runtime).'}),
                'clip': ('CLIP', {'tooltip': 'Krea 2 Qwen3-VL com torre visual.'}),
                'vae': ('VAE',),
                'image': ('IMAGE', {'tooltip': 'Referência (painel anterior).'}),
                'positive_prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
                'negative_prompt': ('STRING', {'default': '', 'multiline': True, 'dynamicPrompts': True}),
                'lora_name': (folder_paths.get_filename_list('loras'),),
                'block_strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.05,
                                             'tooltip': 'Escala dos deltas ROUTADOS nos blocks (fidelidade à referência).'}),
                'fusion_strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.05,
                                              'tooltip': 'Escala do LoRA GLOBAL do txtfusion (semântica do grounding). 0 = txtfusion do adapter desligado (mede quanto vem do built-in).'}),
                'reference_strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.5, 'step': 0.05,
                                                 'tooltip': 'Escala do LATENT da referência antes do empacotamento — o dial real de influência (o base copia os tokens mesmo com block_strength 0). 0.4-0.7 = influência leve.'}),
                'start_percent': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                                            'tooltip': 'Início da janela de atividade do adapter (fração do denoise).'}),
                'end_percent': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01,
                                          'tooltip': 'Fim da janela de atividade.'}),
                'strength_curve': (['constant', 'fade_out', 'fade_in', 'fade_in_out'], {'default': 'constant',
                                   'tooltip': 'Curva da força ao longo dos steps. fade_out = 100% no início → 0 no fim.'}),
                'curve_power': ('FLOAT', {'default': 1.0, 'min': 0.1, 'max': 4.0, 'step': 0.1,
                                          'tooltip': '1 = linear; >1 = curva mais agressiva.'}),
                'layers': ('STRING', {'default': 'all',
                                      'tooltip': "Faixa de blocks afetados: 'all' ou 'inicio-fim' (0-27). Ex.: 4-24."}),
                'layer_taper': (['flat', 'fade_deep', 'fade_shallow'], {'default': 'flat',
                                'tooltip': 'Escala por camada dentro da faixa: flat=igual; fade_deep=cai nas profundas; fade_shallow=cai nas rasas.'}),
                'schedule_reference': ('BOOLEAN', {'default': False,
                                       'tooltip': 'Aplica a mesma curva também ao latent da referência (fade da própria ref).'}),
                'model_variant': (['raw', 'turbo'], {'default': 'raw'}),
                'width': ('INT', {'default': 672, 'min': 64, 'max': 4096, 'step': 16}),
                'height': ('INT', {'default': 384, 'min': 64, 'max': 4096, 'step': 16}),
                'batch_size': ('INT', {'default': 1, 'min': 1, 'max': 16}),
            },
            'optional': {
                'vl_longest_side': ('INT', {'default': 768, 'min': 0, 'max': 2048, 'step': 32,
                                            'tooltip': 'Maior lado da imagem que o Qwen3-VL enxerga no grounding. '
                                                       'Treino usou 768 (jitter 384-768 nos adapters novos). 0 = cap por area (~1MP).'}),
                'vl_prompt_style': (['plain', 'picture_n'], {'default': 'plain',
                                    'tooltip': 'Layout do vision block no prompt. plain = contrato do grounded (sem prefixo).'}),
                'reference_fit': (['training_crop', 'preserve_aspect_1mp'], {'default': 'training_crop',
                                  'tooltip': 'Geometria da referencia no VAE. training_crop = crop-fit do treino (recomendado).'}),
                'reference_timestep': (['zero', 'target'], {'default': 'zero',
                                       'tooltip': 'Modulacao dos tokens da referencia. zero = contrato dos adapters grounded atuais.'}),
                'negative_grounding': (['grounded', 'plain'], {'default': 'grounded',
                                       'tooltip': 'grounded = negativo tambem ve a referencia (uncond treinado com caption_dropout). '
                                                  'plain = negativo so texto, sem vision block.'}),
            }
        }

    RETURN_TYPES = ('MODEL', 'CONDITIONING', 'CONDITIONING', 'LATENT', 'INT', 'FLOAT')
    RETURN_NAMES = ('model', 'positive', 'negative', 'latent', 'steps', 'cfg')
    FUNCTION = 'apply'
    CATEGORY = 'CtxRush/Krea 2 Edit'
    DESCRIPTION = (
        'Omini-Grounded: conditioning grounded (vision block plain + grounding '
        '768) + LoRA condition-only nos blocks e global no txtfusion, tudo em '
        'runtime bf16. Contrato: width-shift, refs a t=0.'
    )

    def apply(self, model, clip, vae, image, positive_prompt, negative_prompt,
              lora_name, block_strength=1.0, fusion_strength=1.0,
              model_variant='raw', width=672, height=384, batch_size=1,
              vl_longest_side=768, vl_prompt_style='plain',
              reference_fit='training_crop', reference_timestep='zero',
              negative_grounding='grounded', reference_strength=1.0,
              start_percent=0.0, end_percent=1.0, strength_curve='constant',
              curve_power=1.0, layers='all', layer_taper='flat',
              schedule_reference=False):
        reference = _build_reference(
            vae, image, width, height, reference_fit,
            vl_longest_side=vl_longest_side,
        )

        def encode(prompt, grounded=True):
            if not grounded:
                tokens = clip.tokenize(prompt, llama_template=KREA2_TEMPLATE)
                return clip.encode_from_tokens_scheduled(tokens)
            if vl_prompt_style == 'picture_n':
                text = f'Picture 1: {VISION_BLOCK}{prompt}'
            else:
                text = f'{VISION_BLOCK}{prompt}'
            tokens = clip.tokenize(text, images=[reference.vl_image], llama_template=KREA2_TEMPLATE)
            return clip.encode_from_tokens_scheduled(tokens)

        positive = encode(positive_prompt)
        negative = encode(negative_prompt, grounded=(negative_grounding == 'grounded'))

        lora_path = folder_paths.get_full_path('loras', lora_name)
        pairs = _load_omini_lora(lora_path)
        patched = model.clone()
        dit = patched.get_model_object('diffusion_model')
        named = dict(dit.named_modules())
        block_entries, fusion_entries, missing = [], [], []
        for path, (a, b) in pairs.items():
            module = named.get(path)
            if module is None:
                missing.append(path)
            elif path.startswith('txtfusion'):
                fusion_entries.append((module, a.cuda(), b.cuda()))
            else:
                block_entries.append((module, a.cuda(), b.cuda(), path))
        if not block_entries:
            raise ValueError('No block LoRA modules matched the diffusion model')
        block_entries = _ctxrush_layer_scales(block_entries, layers, layer_taper)
        if missing:
            print(f'[OminiGrounded] WARNING: {len(missing)} LoRA keys not matched (e.g. {missing[:3]})')
        print(f'[OminiGrounded] runtime LoRA: {len(block_entries)} block linears (masked, '
              f'strength {block_strength}) + {len(fusion_entries)} txtfusion linears '
              f'(global, strength {fusion_strength})')

        src = patched.model.process_latent_in(reference.latent)
        if reference_strength != 1.0:
            src = src * float(reference_strength)
        blocks_state = {'entries': block_entries, 'device': None}

        def wrapper(executor, x, timesteps, context, *args, **kwargs):
            # Keep this variadic because Krea 2 forward positional arguments can
            # change between host versions. Only transformer_options is needed.
            transformer_options = kwargs.get('transformer_options')
            if transformer_options is None:
                transformer_options = next((a for a in args if isinstance(a, dict)), {})
            sigma = (
                float(timesteps.flatten()[0])
                if hasattr(timesteps, 'flatten')
                else float(timesteps)
            )
            multiplier = _ctxrush_schedule_mult(
                sigma,
                start_percent,
                end_percent,
                strength_curve,
                curve_power,
            )
            src_step = (
                src * multiplier
                if schedule_reference and multiplier != 1.0
                else src
            )
            return _krea2_omini_grounded_forward(
                executor.class_obj, x, timesteps, context, src_step, blocks_state,
                fusion_entries, block_strength * multiplier, transformer_options,
                fusion_strength=fusion_strength * multiplier,
                reference_timestep=reference_timestep,
            )

        to = patched.model_options.setdefault('transformer_options', {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, 'ctxrush_omini_grounded', wrapper, to
        )
        mu = _krea2_raw_mu(width, height) if model_variant == 'raw' else KREA2_TURBO_MU
        patched = _apply_krea2_sampling(patched, mu)
        steps, cfg = (28, 5.5) if model_variant == 'raw' else (8, 1.0)
        return (patched, positive, negative, _empty_krea_latent(width, height, batch_size), steps, cfg)


NODE_CLASS_MAPPINGS = {
    'CtxRushKrea2OminiGroundedApply': CtxRushKrea2OminiGroundedApply,
    'CtxRushKrea2OminiApply': CtxRushKrea2OminiApply,
    "CtxRushKrea2EditSetup": CtxRushKrea2EditSetup,
    "CtxRushKrea2ReferenceEncode": CtxRushKrea2ReferenceEncode,
    "CtxRushKrea2EditCFGEncode": CtxRushKrea2EditCFGEncode,
    "CtxRushKrea2EditModelPatch": CtxRushKrea2EditModelPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CtxRushKrea2OminiGroundedApply": "CtxRush - Krea 2 Omini-Grounded (setup completo)",
    "CtxRushKrea2OminiApply": "CtxRush - Krea 2 Omini Apply (condition-only LoRA)",
    "CtxRushKrea2EditSetup": "CtxRush - Krea 2 Edit Setup",
    "CtxRushKrea2ReferenceEncode": "CtxRush - Krea 2 Reference Encode",
    "CtxRushKrea2EditCFGEncode": "CtxRush - Krea 2 Edit CFG Encode",
    "CtxRushKrea2EditModelPatch": "CtxRush - Krea 2 Edit Model Patch",
}
