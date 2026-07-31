# Notes

## 2026-07-30 — runner parity

- Root cause: training/runner converts the scaled-FP8 checkpoint to raw
  `float8_e4m3fn` for the 224 `blocks.*` Linears; stock ComfyUI preserved
  `weight_scale`, so the adapter ran on a different base.
- Runtime fix: dequantize, cast to the raw FP8 grid, materialize BF16 weights,
  and set both `_full_precision_mm` and `_full_precision_mm_config` to prevent
  native FP8 activation/GEMM.
- Turbo fix: merge 264 LoRA pairs and ignore the 7 `.diff_b` keys, matching the
  canonical runner. Never combine this with `LoraLoaderModelOnly`.
- Reference fix: decode JPEG with PIL/libjpeg (`K2LoadReferencePIL`), not the
  PyAV path in `LoadImage`.
- Text encoder contract: temporarily restore `BaseLlama.forward` plus explicit
  KV `repeat_interleave`, then restore current methods in `finally`.
- Per-token timestep: call `tmlp+tproj` on the full token timestep tensor, as
  training does; expanding a one-token result selects a different BF16 GEMM.
- Seed parity: ComfyUI RandomNoise is CPU, the runner RNG is CUDA. Same integer
  seed produced independent noise (`relL2≈1.414`, cosine≈0). Added
  `K2RunnerNoise` for `SamplerCustomAdvanced`; output is bit-identical to the
  runner CUDA RNG.
- Corrected first-forward result, natural PIL/context:
  `v relL2=0.027330`, cosine `0.999627`. With runner context/reference forced:
  `v relL2=0.015068`, cosine `0.999887`.
- Smoke test: 1024 Turbo, 8 Euler/simple steps, 55.22 s, 224 target Linears,
  264 turbo merges, 7 diff_b ignored, 224 routed adapter hits, 0 misses.
- Final workflow:
  `/workspace/outputs/testes_novos/WORKFLOW_FINAL_PARITY.json`.
- Final report: `/workspace/RELATORIO_CODEX.md`.
- No models/checkpoints/dumps deleted.

## 2026-07-30 — LoRA and ControlNet compatibility

- Stock model LoRA path validated at 1024 px in this order:
  `UNET -> K2TrainingBase -> LoraLoaderModelOnly -> CtxRush -> sampler`.
- The standard loader reported 270 attached patches; the CtxRush adapter still
  reported 224 routed calls and 0 misses.
- A/B latent with the stock LoRA on vs off:
  `relL2=3.465451`, cosine `0.390968`, MAE `1.964989`; therefore the additional
  model LoRA was applied, not dropped by either clone/wrapper.
- The Turbo LoRA was used only as the available parser/patch test. In real
  workflows Turbo must be fused once inside `K2TrainingBase`; additional LoRAs
  are chained after it.
- Added forwarding for ComfyUI `post_input`, single-block `patches_replace`,
  per-block metadata, and `control["output"]`. Control residuals affect target
  tokens only; text and clean-reference tokens remain untouched.
- No-patch regression is bit-exact against the approved pre-change forward:
  `array_equal=True`, `relL2=0`, cosine `1`.
- Synthetic ControlNet protocol smoke test at 1024 px:
  residual `(1,4096,6144)` reached block 0; CtxRush remained 224/224 with
  0 misses; output vs no-control baseline changed by `relL2=0.076653`,
  cosine `0.997522`, MAE `0.089143`.
- There is no Krea2 ControlNet checkpoint on this machine and current stock
  ComfyUI has no Krea2-specific ControlNet loader. Real ControlNet quality and
  weight compatibility therefore remain unvalidated. Flux/SDXL ControlNets are
  architecturally incompatible.
- Evidence:
  `/workspace/logs/comfy_18198_control_compat.log`,
  `/workspace/comfy/ComfyUI/output/COMPAT_FAKE_CONTROL_00001_.latent`,
  `/workspace/comfy/ComfyUI/output/COMPAT_NOPATCH_00001_.latent`.
- At the close of this test, the main ComfyUI had restarted as PID 652500
  after the functional patch and had 25.6 GiB staged on GPU. No training
  process was running. Unload/stop ComfyUI before the next 1024 training run.

## 2026-07-31 — reference guidance and target-space masks

- Added a separate controlled v3 path; the validated v2 implementation remains
  unchanged.
- `K2ReferenceGuider` uses explicit with/without-reference predictions instead
  of scaling the clean latent. The full mode evaluates the bilinear
  `negative/positive × reference off/on` surface, so text CFG and image
  guidance are independent.
- `runner_vae_only` retains the exact runner convention: the off branch zeros
  the VAE reference but leaves Qwen3-VL grounded.
- `K2ReferenceMask` mixes the measured reference delta in target latent space:
  `v = v0 + G(x,y) * (v1-v0)`. Soft mask values feather linearly.
- Endpoint invariants and spatial broadcasting are covered by six CPU-only
  unit tests. No GPU inference was run while the 1024 training process was
  active.
- Controlled v3 reads `vl_image_label` from adapter metadata, rejects
  unsupported `independent_condition=true` contracts, enforces `max_refs`, and
  offers the runner's literal PIL crop-fit path.
- The setup exposes all four conditioning corners. Apply the same ControlNet
  to each and use `K2ReferenceConditioningPack` to preserve composability.
