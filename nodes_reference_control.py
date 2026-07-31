"""Advanced Krea 2 reference-strength and spatial-mask nodes.

These nodes intentionally live beside, rather than replace, the validated
``CtxRushKrea2MultiRefApply`` v2 node.  At reference strength one the
controlled setup takes the same full-reference forward path.  Away from one,
the custom guider evaluates explicit with/without-reference branches and
mixes their velocity predictions.

Two reference baselines are available:

``full_off``
    The no-reference branch has zero VAE reference tokens *and* plain Qwen
    conditioning.  This is the useful "master image strength" axis.

``runner_vae_only``
    The no-reference branch zeroes only the VAE reference while retaining
    grounded Qwen conditioning.  This matches diffusion-pipe's current
    ``--reference-guidance`` semantics.
"""

from __future__ import annotations

import math

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.patcher_extension
import comfy.samplers
import folder_paths
from comfy.text_encoders.krea2 import KREA2_TEMPLATE

from .nodes import (
    KREA2_TURBO_MU,
    _apply_krea2_sampling,
    _crop_fit,
    _ctxrush_layer_scales,
    _empty_krea_latent,
    _fit_vl,
    _krea2_raw_mu,
    _load_omini_lora,
    _require_single_image,
)
from .nodes_multiref import (
    _DEBUG,
    _build_vl_prompt,
    _dbg,
    _krea2_multiref_forward,
    _read_safetensors_metadata,
    _resolve_contract,
    _training_vl_contract,
)
from .reference_guidance import mix_reference, reference_gain_map


CONDITIONING_TYPE = "K2_REF_CONDITIONING"
MASK_TYPE = "K2_REF_MASK"
_GUIDER_BRANCH_OPTION = "ctxrush_k2_reference_branches"


def _resolve_control_contract(
    metadata,
    vl_max_pixels_opt,
    vl_label_opt,
    reference_timestep_opt,
):
    """Resolve metadata strictly for the controlled path.

    The old v2 node keeps its historical behavior untouched.  This new path
    additionally consumes ``vl_image_label`` and refuses metadata contracts
    that its forward cannot faithfully reproduce.
    """

    manual_label = "" if vl_label_opt.strip().lower() == "auto" else vl_label_opt
    contract = _resolve_contract(
        metadata,
        vl_max_pixels_opt,
        manual_label,
        reference_timestep_opt,
    )
    if not manual_label:
        contract["vl_label"] = metadata.get("vl_image_label", "image") or "image"

    family = contract["family"]
    if family and family != "krea2_multiref_grounded":
        raise ValueError(
            f"Controlled v3 expects krea2_multiref_grounded, adapter says {family!r}."
        )

    independent = str(metadata.get("independent_condition", "false")).lower() == "true"
    if independent:
        raise ValueError(
            "This adapter uses independent_condition=true, which requires a dense "
            "role-attention mask not implemented by the controlled v3 node."
        )

    position_mode = metadata.get("position_mode", "width_shift")
    if contract["slot_axis"] not in ("width", "frame"):
        raise ValueError(
            f"Unsupported slot_axis={contract['slot_axis']!r}; expected width or frame."
        )
    if contract["slot_axis"] == "width" and position_mode != "width_shift":
        raise ValueError(
            f"Unsupported position contract: slot_axis=width with "
            f"position_mode={position_mode!r}."
        )
    position_scale = float(metadata.get("reference_position_scale", 1.0) or 1.0)
    if not math.isclose(position_scale, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Controlled v3 does not support reference_position_scale="
            f"{position_scale}; current forward requires 1.0."
        )
    return contract


def _crop_fit_runner_pil(image, width, height):
    """Use the runner's literal PIL ``ImageOps.fit(..., LANCZOS)`` path.

    With ``K2 Load Reference (PIL)`` the input tensor is a lossless float view
    of the decoded uint8 pixels, so the uint8 round-trip reproduces the image
    on which the runner performs its crop/resize.
    """

    import numpy as np
    from PIL import Image, ImageOps

    image = _require_single_image(image)
    if image.shape[-1] < 3:
        raise ValueError(f"Expected an RGB IMAGE, got shape {tuple(image.shape)}")
    array = (
        image[0, :, :, :3]
        .detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    pil = Image.fromarray(array, mode="RGB")
    fitted = ImageOps.fit(
        pil,
        (int(width), int(height)),
        method=Image.Resampling.LANCZOS,
    )
    output = np.asarray(fitted, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(output).unsqueeze(0)


def _reference_batch_for_branches(references, x, transformer_options):
    """Materialize full/zero VAE references for guider-selected batch chunks."""

    flags = transformer_options.get(_GUIDER_BRANCH_OPTION)
    branch_indices = transformer_options.get("cond_or_uncond")
    if flags is None or not branch_indices:
        return references
    if x.shape[0] % len(branch_indices):
        raise RuntimeError(
            f"K2 reference guider cannot split batch {x.shape[0]} across "
            f"{len(branch_indices)} conditioning branches."
        )

    chunk = x.shape[0] // len(branch_indices)
    selected = []
    for source in references:
        source = source.to(device=x.device, dtype=x.dtype)
        if source.shape[0] == 1 and x.shape[0] != 1:
            source = source.expand(x.shape[0], *source.shape[1:])
        elif source.shape[0] != x.shape[0]:
            repeats = math.ceil(x.shape[0] / source.shape[0])
            source = source.repeat(repeats, *([1] * (source.ndim - 1)))[: x.shape[0]]
        source = source.clone()
        for batch_index, branch_index in enumerate(branch_indices):
            if not 0 <= int(branch_index) < len(flags):
                raise RuntimeError(
                    f"Unexpected K2 guider branch index {branch_index}; "
                    f"mapping has {len(flags)} entries."
                )
            if not flags[int(branch_index)]:
                start = batch_index * chunk
                source[start : start + chunk].zero_()
        selected.append(source)
    return selected


class CtxRushKrea2MultiRefControlledApply:
    """Separate v3 setup for independent text/reference guidance."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Output of K2 Training Base. Extra Krea2 LoRAs "
                        "may be inserted before this node; never load Turbo twice."
                    },
                ),
                "clip": ("CLIP", {"tooltip": "Krea 2 Qwen3-VL 4B BF16."}),
                "vae": ("VAE",),
                "image_1": ("IMAGE",),
                "positive_prompt": (
                    "STRING",
                    {"multiline": True, "dynamicPrompts": True},
                ),
                "negative_prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "block_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
                ),
                "fusion_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
                ),
                "model_variant": (["turbo", "raw"], {"default": "turbo"}),
                "width": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 4096, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 4096, "step": 16},
                ),
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 16},
                ),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "vl_max_pixels": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096 * 4096,
                        "step": 28 * 28,
                        "tooltip": "0 reads the adapter metadata (147456 for the "
                        "current adapter).",
                    },
                ),
                "vl_image_label": (
                    "STRING",
                    {
                        "default": "auto",
                        "tooltip": "auto reads vl_image_label from adapter metadata.",
                    },
                ),
                "reference_timestep": (
                    ["auto", "zero", "target"],
                    {"default": "auto"},
                ),
                "layers": (
                    "STRING",
                    {"default": "all", "tooltip": "'all' or e.g. '4-24'."},
                ),
                "layer_taper": (
                    ["flat", "fade_deep", "fade_shallow"],
                    {"default": "flat"},
                ),
                "reference_resize": (
                    ["runner_pil", "comfy_tensor"],
                    {
                        "default": "runner_pil",
                        "tooltip": "runner_pil uses literal PIL ImageOps.fit/LANCZOS. "
                        "Use comfy_tensor for generated/float references that must "
                        "not be quantized to uint8.",
                    },
                ),
                "training_vl_contract": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Restores the Qwen3-VL attention/GQA contract "
                        "used during training.",
                    },
                ),
                "training_base_quant": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Verifies the K2 Training Base numeric contract "
                        "on the first forward.",
                    },
                ),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (
        "MODEL",
        CONDITIONING_TYPE,
        "CONDITIONING",
        "CONDITIONING",
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "INT",
        "FLOAT",
    )
    RETURN_NAMES = (
        "model",
        "reference_conditioning",
        "positive_with_reference",
        "negative_with_reference",
        "positive_no_reference",
        "negative_no_reference",
        "latent",
        "steps",
        "cfg",
    )
    FUNCTION = "apply"
    CATEGORY = "CtxRush/Krea 2 Edit"
    DESCRIPTION = (
        "Separate controlled path. Builds grounded and truly reference-free "
        "conditioning branches for K2 Reference Guider; the validated v2 node "
        "is left untouched."
    )

    def apply(
        self,
        model,
        clip,
        vae,
        image_1,
        positive_prompt,
        negative_prompt,
        lora_name,
        block_strength=1.0,
        fusion_strength=1.0,
        model_variant="turbo",
        width=1024,
        height=1024,
        batch_size=1,
        image_2=None,
        image_3=None,
        image_4=None,
        vl_max_pixels=0,
        vl_image_label="auto",
        reference_timestep="auto",
        layers="all",
        layer_taper="flat",
        reference_resize="runner_pil",
        training_vl_contract=True,
        training_base_quant=True,
        debug=False,
    ):
        _DEBUG.update(
            on=bool(debug),
            calls=0,
            lora_hits=0,
            lora_misses=0,
            logged=False,
        )
        if width % 16 or height % 16:
            raise ValueError("width/height must be multiples of 16.")

        lora_path = folder_paths.get_full_path("loras", lora_name)
        metadata = _read_safetensors_metadata(lora_path)
        contract = _resolve_control_contract(
            metadata,
            vl_max_pixels,
            vl_image_label,
            reference_timestep,
        )

        images = [
            image
            for image in (image_1, image_2, image_3, image_4)
            if image is not None
        ]
        if contract["max_refs"] and len(images) > contract["max_refs"]:
            raise ValueError(
                f"Adapter was trained with max_refs={contract['max_refs']}, "
                f"but {len(images)} references were connected."
            )

        vl_images = []
        reference_latents = []
        for image in images:
            image = _require_single_image(image)
            vl_images.append(_fit_vl(image, contract["vl_max_pixels"]))
            pixels = (
                _crop_fit_runner_pil(image, width, height)
                if reference_resize == "runner_pil"
                else _crop_fit(image, width, height)
            )
            reference_latents.append(vae.encode(pixels))

        def encode(prompt, grounded):
            if grounded:
                text = _build_vl_prompt(
                    len(vl_images),
                    prompt,
                    contract["prompt_style"],
                    contract["vl_label"],
                )
                tokens = clip.tokenize(
                    text,
                    images=vl_images,
                    llama_template=KREA2_TEMPLATE,
                )
            else:
                tokens = clip.tokenize(prompt, llama_template=KREA2_TEMPLATE)
            with _training_vl_contract(training_vl_contract):
                return clip.encode_from_tokens_scheduled(tokens)

        positive_with_reference = encode(positive_prompt, True)
        negative_with_reference = encode(negative_prompt, True)
        positive_no_reference = encode(positive_prompt, False)
        negative_no_reference = encode(negative_prompt, False)

        pairs = _load_omini_lora(lora_path)
        patched = model.clone()
        dit = patched.get_model_object("diffusion_model")
        named = dict(dit.named_modules())
        device = comfy.model_management.get_torch_device()
        block_entries = []
        fusion_entries = []
        missing = []
        for path, (lora_a, lora_b) in pairs.items():
            module = named.get(path)
            if module is None:
                missing.append(path)
            elif path.startswith("txtfusion"):
                fusion_entries.append(
                    (module, lora_a.to(device), lora_b.to(device))
                )
            else:
                block_entries.append(
                    (module, lora_a.to(device), lora_b.to(device), path)
                )
        if not block_entries:
            raise ValueError("No block LoRA module matched the Krea2 model.")
        block_entries = _ctxrush_layer_scales(
            block_entries,
            layers,
            layer_taper,
        )
        if missing:
            print(
                f"[K2Controlled] WARNING: {len(missing)} LoRA modules did not "
                f"match (examples: {missing[:3]})."
            )

        processed_references = [
            patched.model.process_latent_in(latent)
            for latent in reference_latents
        ]
        blocks_state = {"entries": block_entries, "device": None}

        if debug:
            _dbg(
                f"controlled contract={contract} refs={len(images)} "
                f"resize={reference_resize} block_lora={len(block_entries)} "
                f"fusion_lora={len(fusion_entries)}"
            )

        def wrapper(executor, x, timesteps, context, *args, **kwargs):
            transformer_options = kwargs.get("transformer_options")
            if transformer_options is None:
                transformer_options = next(
                    (value for value in args if isinstance(value, dict)),
                    {},
                )
            text_mask = kwargs.get("attention_mask")
            if text_mask is None:
                text_mask = next(
                    (
                        value
                        for value in args
                        if torch.is_tensor(value) and value.ndim <= 2
                    ),
                    None,
                )
            references = _reference_batch_for_branches(
                processed_references,
                x,
                transformer_options,
            )
            return _krea2_multiref_forward(
                executor.class_obj,
                x,
                timesteps,
                context,
                references,
                blocks_state,
                fusion_entries,
                block_strength,
                transformer_options,
                fusion_strength=fusion_strength,
                reference_timestep=contract["reference_timestep"],
                slot_axis=contract["slot_axis"],
                position_offset=contract["position_offset"],
                text_attention_mask=text_mask,
                training_base_quant=training_base_quant,
                control=kwargs.get("control"),
            )

        transformer_options = patched.model_options.setdefault(
            "transformer_options",
            {},
        )
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "ctxrush_multiref_controlled",
            wrapper,
            transformer_options,
        )
        patched._ctxrush_k2_reference_controlled = True

        mu = (
            _krea2_raw_mu(width, height)
            if model_variant == "raw"
            else KREA2_TURBO_MU
        )
        patched = _apply_krea2_sampling(patched, mu)
        steps, cfg = (28, 5.5) if model_variant == "raw" else (8, 1.0)
        conditioning = {
            "positive_with_reference": positive_with_reference,
            "negative_with_reference": negative_with_reference,
            "positive_no_reference": positive_no_reference,
            "negative_no_reference": negative_no_reference,
            "contract": contract,
        }
        return (
            patched,
            conditioning,
            positive_with_reference,
            negative_with_reference,
            positive_no_reference,
            negative_no_reference,
            _empty_krea_latent(width, height, batch_size),
            steps,
            cfg,
        )


class K2ReferenceConditioningPack:
    """Repack four conditioning corners after external ComfyUI modifiers.

    This keeps standard ControlNet/conditioning nodes composable: apply the
    same target control to all four CONDITIONING outputs, then repack them for
    the custom guider.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_with_reference": ("CONDITIONING",),
                "negative_with_reference": ("CONDITIONING",),
                "positive_no_reference": ("CONDITIONING",),
                "negative_no_reference": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = (CONDITIONING_TYPE,)
    RETURN_NAMES = ("reference_conditioning",)
    FUNCTION = "pack"
    CATEGORY = "CtxRush/Krea 2 Edit"
    DESCRIPTION = (
        "Rebuilds the four-corner K2 conditioning bundle after applying the "
        "same ControlNet or conditioning edits to all four branches."
    )

    def pack(
        self,
        positive_with_reference,
        negative_with_reference,
        positive_no_reference,
        negative_no_reference,
    ):
        return (
            {
                "positive_with_reference": positive_with_reference,
                "negative_with_reference": negative_with_reference,
                "positive_no_reference": positive_no_reference,
                "negative_no_reference": negative_no_reference,
            },
        )


class K2ReferenceMask:
    """Describe target-space strength outside a user-provided soft mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": "White/1 uses the guider's reference strength; "
                        "black/0 uses outside_strength. Gray values feather."
                    },
                ),
                "outside_strength": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 8.0,
                        "step": 0.05,
                    },
                ),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (MASK_TYPE, "MASK")
    RETURN_NAMES = ("reference_mask", "preview")
    FUNCTION = "build"
    CATEGORY = "CtxRush/Krea 2 Edit"
    DESCRIPTION = (
        "Target-space reference influence map. White uses the guider strength; "
        "black uses outside_strength; soft edges interpolate linearly."
    )

    def build(self, mask, outside_strength=0.0, invert=False):
        value = mask.detach().float().clamp(0.0, 1.0)
        if invert:
            value = 1.0 - value
        return (
            {
                "mask": value,
                "outside_guidance": float(outside_strength),
            },
            value,
        )


class _K2ReferenceGuider(comfy.samplers.CFGGuider):
    def set_reference_conds(self, conditioning):
        required = (
            "positive_with_reference",
            "negative_with_reference",
            "positive_no_reference",
            "negative_no_reference",
        )
        missing = [name for name in required if name not in conditioning]
        if missing:
            raise ValueError(
                f"Invalid K2 reference conditioning bundle; missing {missing}."
            )
        self.inner_set_conds(
            {
                name: conditioning[name]
                for name in required
            }
        )

    def set_guidance(
        self,
        text_guidance,
        reference_guidance,
        reference_mode,
        reference_mask,
        debug,
    ):
        if text_guidance < 0 or reference_guidance < 0:
            raise ValueError("K2 text/reference guidance must be non-negative.")
        self.text_guidance = float(text_guidance)
        self.reference_guidance = float(reference_guidance)
        self.reference_mode = reference_mode
        self.reference_mask = reference_mask
        self.debug = bool(debug)
        self._logged = False

    def _calculate(
        self,
        names,
        reference_flags,
        x,
        timestep,
        model_options,
    ):
        local_options = comfy.model_patcher.create_model_options_clone(
            model_options
        )
        local_options.setdefault("transformer_options", {})[
            _GUIDER_BRANCH_OPTION
        ] = tuple(bool(value) for value in reference_flags)
        conditions = [self.conds[name] for name in names]
        custom_batch = local_options.get("sampler_calc_cond_batch_function")
        if custom_batch is None:
            outputs = comfy.samplers.calc_cond_batch(
                self.inner_model,
                conditions,
                x,
                timestep,
                local_options,
            )
        else:
            outputs = custom_batch(
                {
                    "conds": conditions,
                    "input": x,
                    "sigma": timestep,
                    "model": self.inner_model,
                    "model_options": local_options,
                }
            )
        return outputs, local_options

    def _text_cfg(
        self,
        positive_prediction,
        negative_prediction,
        positive_name,
        negative_name,
        x,
        timestep,
        model_options,
    ):
        conditions = [
            self.conds[positive_name],
            self.conds[negative_name],
        ]
        outputs = [positive_prediction, negative_prediction]
        for pre_cfg in model_options.get("sampler_pre_cfg_function", []):
            outputs = pre_cfg(
                {
                    "conds": conditions,
                    "conds_out": outputs,
                    "cond_scale": self.text_guidance,
                    "timestep": timestep,
                    "input": x,
                    "sigma": timestep,
                    "model": self.inner_model,
                    "model_options": model_options,
                }
            )
        return comfy.samplers.cfg_function(
            self.inner_model,
            outputs[0],
            outputs[1],
            self.text_guidance,
            x,
            timestep,
            model_options=model_options,
            cond=self.conds[positive_name],
            uncond=self.conds[negative_name],
        )

    def _gain(self, prediction):
        if self.reference_mask is None:
            return reference_gain_map(
                prediction,
                self.reference_guidance,
            )
        return reference_gain_map(
            prediction,
            self.reference_guidance,
            mask=self.reference_mask["mask"],
            outside_guidance=self.reference_mask["outside_guidance"],
        )

    def _full_off(self, x, timestep, model_options):
        text_is_one = math.isclose(
            self.text_guidance,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        uniform = self.reference_mask is None
        reference_is_one = uniform and math.isclose(
            self.reference_guidance,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        reference_is_zero = uniform and math.isclose(
            self.reference_guidance,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )

        if reference_is_one:
            if text_is_one:
                outputs, _ = self._calculate(
                    ["positive_with_reference"],
                    [True],
                    x,
                    timestep,
                    model_options,
                )
                return outputs[0]
            outputs, options = self._calculate(
                ["negative_with_reference", "positive_with_reference"],
                [True, True],
                x,
                timestep,
                model_options,
            )
            return self._text_cfg(
                outputs[1],
                outputs[0],
                "positive_with_reference",
                "negative_with_reference",
                x,
                timestep,
                options,
            )

        if reference_is_zero:
            if text_is_one:
                outputs, _ = self._calculate(
                    ["positive_no_reference"],
                    [False],
                    x,
                    timestep,
                    model_options,
                )
                return outputs[0]
            outputs, options = self._calculate(
                ["negative_no_reference", "positive_no_reference"],
                [False, False],
                x,
                timestep,
                model_options,
            )
            return self._text_cfg(
                outputs[1],
                outputs[0],
                "positive_no_reference",
                "negative_no_reference",
                x,
                timestep,
                options,
            )

        if text_is_one:
            outputs, _ = self._calculate(
                ["positive_no_reference", "positive_with_reference"],
                [False, True],
                x,
                timestep,
                model_options,
            )
            return mix_reference(outputs[0], outputs[1], self._gain(outputs[0]))

        outputs, options = self._calculate(
            [
                "negative_no_reference",
                "positive_no_reference",
                "negative_with_reference",
                "positive_with_reference",
            ],
            [False, False, True, True],
            x,
            timestep,
            model_options,
        )
        no_reference_text = self._text_cfg(
            outputs[1],
            outputs[0],
            "positive_no_reference",
            "negative_no_reference",
            x,
            timestep,
            options,
        )
        with_reference_text = self._text_cfg(
            outputs[3],
            outputs[2],
            "positive_with_reference",
            "negative_with_reference",
            x,
            timestep,
            options,
        )
        return mix_reference(
            no_reference_text,
            with_reference_text,
            self._gain(no_reference_text),
        )

    def _runner_vae_only(self, x, timestep, model_options):
        text_is_one = math.isclose(
            self.text_guidance,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        uniform = self.reference_mask is None
        reference_is_one = uniform and math.isclose(
            self.reference_guidance,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )

        if reference_is_one:
            return self._full_off(x, timestep, model_options)

        if text_is_one:
            outputs, _ = self._calculate(
                ["positive_with_reference", "positive_with_reference"],
                [False, True],
                x,
                timestep,
                model_options,
            )
            return mix_reference(outputs[0], outputs[1], self._gain(outputs[0]))

        outputs, options = self._calculate(
            [
                "negative_with_reference",
                "negative_with_reference",
                "positive_with_reference",
            ],
            [False, True, True],
            x,
            timestep,
            model_options,
        )
        text_with_reference = self._text_cfg(
            outputs[2],
            outputs[1],
            "positive_with_reference",
            "negative_with_reference",
            x,
            timestep,
            options,
        )
        return (
            mix_reference(outputs[0], outputs[1], self._gain(outputs[0]))
            + (text_with_reference - outputs[1])
        )

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        model_options = {} if model_options is None else model_options
        if self.reference_mode == "runner_vae_only":
            prediction = self._runner_vae_only(x, timestep, model_options)
        else:
            prediction = self._full_off(x, timestep, model_options)

        if self.debug and not self._logged:
            self._logged = True
            mask_label = (
                "none"
                if self.reference_mask is None
                else (
                    f"inside={self.reference_guidance} "
                    f"outside={self.reference_mask['outside_guidance']}"
                )
            )
            print(
                f"[K2ReferenceGuider] mode={self.reference_mode} "
                f"text={self.text_guidance} reference={self.reference_guidance} "
                f"mask={mask_label}",
                flush=True,
            )
        return prediction


class K2ReferenceGuider:
    """Custom SamplerAdvanced guider with independent text/image axes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "reference_conditioning": (CONDITIONING_TYPE,),
                "text_guidance": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Turbo canonical=1.0; Raw canonical=5.5.",
                    },
                ),
                "reference_guidance": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 8.0,
                        "step": 0.05,
                        "tooltip": "1=canonical full reference; 0=no-reference "
                        "baseline; >1 amplifies the measured reference direction.",
                    },
                ),
                "reference_mode": (
                    ["full_off", "runner_vae_only"],
                    {
                        "default": "full_off",
                        "tooltip": "full_off controls both VAE reference and Qwen "
                        "grounding. runner_vae_only exactly follows the runner flag, "
                        "which keeps Qwen grounded.",
                    },
                ),
            },
            "optional": {
                "reference_mask": (MASK_TYPE,),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "CtxRush/Krea 2 Edit"
    DESCRIPTION = (
        "Independent text/reference guidance. Connect to "
        "SamplerCustomAdvanced. full_off uses four explicit corners "
        "(negative/positive x reference off/on); runner_vae_only matches the "
        "runner's VAE-only reference guidance."
    )

    def get_guider(
        self,
        model,
        reference_conditioning,
        text_guidance=1.0,
        reference_guidance=1.0,
        reference_mode="full_off",
        reference_mask=None,
        debug=False,
    ):
        if not getattr(model, "_ctxrush_k2_reference_controlled", False):
            raise ValueError(
                "K2 Reference Guider requires the MODEL output of "
                "CtxRush Krea 2 Multi-Ref Controlled (v3)."
            )
        guider = _K2ReferenceGuider(model)
        guider.set_reference_conds(reference_conditioning)
        guider.set_guidance(
            text_guidance,
            reference_guidance,
            reference_mode,
            reference_mask,
            debug,
        )
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "CtxRushKrea2MultiRefControlledApply": CtxRushKrea2MultiRefControlledApply,
    "K2ReferenceConditioningPack": K2ReferenceConditioningPack,
    "K2ReferenceMask": K2ReferenceMask,
    "K2ReferenceGuider": K2ReferenceGuider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CtxRushKrea2MultiRefControlledApply": (
        "CtxRush - Krea 2 Multi-Ref Controlled (v3)"
    ),
    "K2ReferenceConditioningPack": "K2 Reference Conditioning Pack",
    "K2ReferenceMask": "K2 Reference Mask Strength",
    "K2ReferenceGuider": "K2 Reference Guider (text + image)",
}
