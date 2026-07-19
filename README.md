# CtxRush Krea 2 Edit for ComfyUI

Inference nodes for LoRAs trained by this fork with `model.type = "krea2_edit"`.
They implement the same one-reference dual-conditioning contract used during
training:

```text
reference -> VAE -> clean DiT tokens at t=0 / RoPE frame 1
reference + prompt -> Qwen3-VL -> image-grounded conditioning
```

No third-party Python packages are required beyond ComfyUI's own environment.

## Installation

Copy the `ctxrush_edit` directory into `ComfyUI/custom_nodes` and restart
ComfyUI:

```text
ComfyUI/
  custom_nodes/
    ctxrush_edit/
      __init__.py
      nodes.py
      README.md
```

The Qwen3-VL text encoder checkpoint must include its `visual.*` weights.

## Recommended workflow

Use **CtxRush - Krea 2 Edit Setup** for normal work:

```text
Load Diffusion Model (Krea 2)
  -> Load LoRA (model only)
  -> CtxRush - Krea 2 Edit Setup:model
  -> KSampler:model

Load CLIP (Krea 2) ---------------------> Setup:clip
Load VAE (Qwen Image VAE) --------------> Setup:vae
Load Image (one reference) -------------> Setup:reference

Setup:positive --------------------------> KSampler:positive
Setup:negative --------------------------> KSampler:negative
Setup:latent ----------------------------> KSampler:latent_image
Setup:steps -----------------------------> KSampler:steps (convert widget to input)
Setup:cfg -------------------------------> KSampler:cfg (convert widget to input)
```

The positive prompt describes the target/next scene. It does not need to
describe the source image. The setup node intentionally feeds the same visual
reference to the negative branch with an empty or negative prompt; this keeps
the reference common to both CFG branches.

### Validated starting profile

For the CtxRush run described in `docs/reference_adapters.md`:

| Model | Steps | CFG | Initial resolution |
|---|---:|---:|---|
| Krea 2 Raw | 28 | 5.5 | the evaluated training bucket, e.g. 672x384 |
| Krea 2 Turbo | 8 | 1.0 | the evaluated training bucket |

The setup node returns these `steps` and `cfg` values. They are starting
profiles, not limits.

## Reference sizing

`reference_fit` has two explicit contracts:

- `training_crop` (default): center-crops the reference to the target width and
  height before VAE encoding. This matches the LoRA trained by this
  diffusion-pipe fork.
- `preserve_aspect_1mp`: preserves reference aspect ratio and downsizes it to a
  1-megapixel budget. This matches public Ostris/ai-toolkit Krea Edit LoRAs but
  was not the geometry used by the CtxRush training run.

The Qwen3-VL path always preserves aspect ratio and uses the training budget of
`147456` pixels (384x384 area). Do not raise it expecting more fine detail: the
VAE path carries that detail.

## Modular workflow

The modular nodes expose the same implementation when the all-in-one node is
too restrictive:

```text
Reference Encode -> Edit CFG Encode -> positive / negative
Load LoRA -> Edit Model Patch -> KSampler model
```

`Reference Encode` performs the expensive VAE encoding only once. `Edit CFG
Encode` uses that object for both prompts, preventing accidental mismatch
between the Qwen visual image and the VAE reference. The model patch reads the
reference latent from each conditioning branch and falls back to the stock
Krea 2 forward when no reference is present.

## Important baselines

Setting LoRA strength to zero is not a vanilla text-to-image baseline while a
reference is still attached: the unadapted model still receives an unfamiliar
clean-reference token sequence. To test vanilla Krea 2, bypass the CtxRush
setup/model patch and use ordinary text conditioning without a reference.

This node pack deliberately does not expose RoPE position or reference
timestep knobs. The adapter was trained at fixed frame `1` and clean timestep
`0`; changing them would silently leave the training distribution.

## Omini-Grounded — dials do node (setup completo)

`CtxRush - Krea 2 Omini-Grounded (setup completo)` expõe, além de
`block_strength` (fidelidade à referência, deltas routados) e
`fusion_strength` (semântica do grounding; 0 mede o built-in), os dials
opcionais de contrato:

| Dial | Default | O que faz |
|---|---|---|
| `vl_longest_side` | 768 | Maior lado visto pelo Qwen3-VL no grounding (adapters com jitter 384-768 aceitam a faixa toda; 0 = cap por área ~1MP) |
| `vl_prompt_style` | plain | Layout do vision block (`plain` = contrato do grounded) |
| `reference_fit` | training_crop | Geometria da referência no VAE (crop-fit do treino) |
| `reference_timestep` | zero | Modulação dos tokens da referência (`zero` = contrato dos adapters atuais) |
| `negative_grounding` | grounded | `grounded` = negativo também vê a referência — é o uncond treinado quando o adapter usou `caption_dropout`; `plain` = negativo só texto |

Os defaults reproduzem exatamente o contrato validado; mude-os apenas para
A/B ou para adapters treinados com outro contrato.
