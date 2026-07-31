"""Pure tensor math for Krea 2 reference guidance.

This module deliberately has no ComfyUI imports.  The node layer calls these
helpers, while CPU tests prove the interpolation endpoints and spatial-mask
semantics without loading a diffusion model.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def mix_reference(
    no_reference: torch.Tensor,
    with_reference: torch.Tensor,
    gain: float | torch.Tensor,
) -> torch.Tensor:
    """Interpolate/extrapolate a prediction along its reference axis.

    ``gain=0`` returns the no-reference prediction, ``gain=1`` returns the
    full-reference prediction, values in between are convex interpolation,
    and values above one amplify the measured reference direction.
    """

    return no_reference + gain * (with_reference - no_reference)


def full_reference_guidance(
    negative_no_reference: torch.Tensor,
    positive_no_reference: torch.Tensor,
    negative_with_reference: torch.Tensor,
    positive_with_reference: torch.Tensor,
    text_guidance: float,
    reference_gain: float | torch.Tensor,
) -> torch.Tensor:
    """Four-corner text/reference guidance.

    First apply ordinary text CFG independently at ``reference=off`` and
    ``reference=on``.  Then interpolate between those two predictions.  For
    linear CFG this is the bilinear surface over the four measured corners.
    """

    text_no_reference = mix_reference(
        negative_no_reference,
        positive_no_reference,
        float(text_guidance),
    )
    text_with_reference = mix_reference(
        negative_with_reference,
        positive_with_reference,
        float(text_guidance),
    )
    return mix_reference(text_no_reference, text_with_reference, reference_gain)


def runner_reference_guidance(
    negative_zero_vae: torch.Tensor,
    positive_zero_vae: torch.Tensor,
    negative_with_reference: torch.Tensor,
    positive_with_reference: torch.Tensor,
    text_guidance: float,
    reference_gain: float | torch.Tensor,
) -> torch.Tensor:
    """Match diffusion-pipe's current ``--reference-guidance`` decomposition.

    The runner keeps Qwen3-VL grounded and zeroes only the VAE reference.  Its
    branch decomposition changes at text guidance 1, so the same explicit
    cases are retained here.
    """

    if math.isclose(float(text_guidance), 1.0, rel_tol=0.0, abs_tol=1e-9):
        return mix_reference(
            positive_zero_vae,
            positive_with_reference,
            reference_gain,
        )

    text_with_reference = mix_reference(
        negative_with_reference,
        positive_with_reference,
        float(text_guidance),
    )
    return (
        mix_reference(
            negative_zero_vae,
            negative_with_reference,
            reference_gain,
        )
        + (text_with_reference - negative_with_reference)
    )


def reference_gain_map(
    prediction: torch.Tensor,
    inside_guidance: float,
    *,
    mask: torch.Tensor | None = None,
    outside_guidance: float = 0.0,
    invert: bool = False,
) -> float | torch.Tensor:
    """Build a broadcastable target-space reference-guidance map.

    ComfyUI masks use one for the selected area.  A soft/feathered mask yields
    a linear transition between ``outside_guidance`` and
    ``inside_guidance``.  The map is resized only in spatial dimensions and
    broadcasts over channels (and over time for a 5-D prediction).
    """

    if mask is None:
        return float(inside_guidance)
    if prediction.ndim not in (4, 5):
        raise ValueError(
            f"K2 reference guidance expects a 4-D or 5-D prediction, got "
            f"shape {tuple(prediction.shape)}"
        )

    value = mask.detach().to(device=prediction.device, dtype=torch.float32)
    if value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.ndim == 3:
        value = value.unsqueeze(1)
    elif value.ndim == 4:
        if value.shape[1] != 1:
            value = value.mean(dim=1, keepdim=True)
    else:
        raise ValueError(
            f"K2 reference mask expects [H,W], [B,H,W] or [B,1,H,W], got "
            f"shape {tuple(value.shape)}"
        )

    value = value.clamp_(0.0, 1.0)
    if invert:
        value = 1.0 - value
    if value.shape[-2:] != prediction.shape[-2:]:
        value = F.interpolate(
            value,
            size=prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    batch = prediction.shape[0]
    if value.shape[0] != batch:
        if value.shape[0] == 1:
            value = value.expand(batch, -1, -1, -1)
        else:
            repeats = math.ceil(batch / value.shape[0])
            value = value.repeat(repeats, 1, 1, 1)[:batch]

    if prediction.ndim == 5:
        value = value.unsqueeze(2)
    value = value.to(dtype=prediction.dtype)
    outside = float(outside_guidance)
    inside = float(inside_guidance)
    return outside + value * (inside - outside)
