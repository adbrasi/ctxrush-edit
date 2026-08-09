# Notes

## 2026-08-08 — storage fp8 no K2 Training Base (crash em máquinas fracas)

- Sintoma: ComfyUI morria DURANTE a execução do `K2 Training Base` em
  placas/máquinas mais fracas. Causa: o node materializava as 224 Linears em
  bf16 (2 bytes/peso) — +11,3 GiB de RAM anônima crescendo ao longo do loop,
  com o checkpoint de 13 GiB ainda residente; o kernel matava o processo.
- Correção: `weight_storage=fp8_1byte` (novo default) guarda o próprio tensor
  `float8_e4m3fn`; o `cast_bias_weight` sobe fp8→bf16 por camada no forward.
  O cast é EXATO (todo ponto do grid e4m3fn é representável em bf16), então a
  saída é bit-idêntica ao storage bf16 — provado em
  `tests/test_trainbase_storage.py` (4 testes CPU, incl. fusão da turbo) e no
  checkpoint real (paridade bit a bit de pesos + forward CPU + forward CUDA +
  merge da turbo com LoRA sintética; 224/224 convertidas, `.diff_b` ignoradas).
- Junto: `layout_type`/`quant_format` zerados nos módulos convertidos (senão o
  `set_weight` chamaria `requantize_from_float` num tensor plano ao assar LoRA
  adicional e crasharia; com None ele faz `weight.to(fp8)` = grid cru, como o
  runner). `bf16_2bytes` mantido como legado; widget novo no FIM do optional
  (regra da auditoria). Contrato de idempotência agora inclui o storage.
- Números: pesos das 224 Linears 11,32 GiB (fp8) vs 22,64 GiB (bf16 antigo).
  Na máquina de 30 GiB da validação, o caminho antigo somaria ~34 GiB → OOM.

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

## Auditoria dos dials v3 (2026-08-06, 5 agentes opus + testes CPU com pesos reais)
- BUG1 (corrigido): widgets novos inseridos no MEIO do optional remapeavam workflows
  salvos por posição (training_vl_contract->ref_bias etc.) => janela do ref_bias
  sempre vazia. Regra permanente: widget novo SEMPRE no fim do bloco.
- BUG2 (corrigido): janela geral usava progress=1-sigma cru (end>=0.69 no-op,
  start>=0.69 kill-switch na grade turbo). Agora _schedule_mult_sigma na mesma
  régua percent_to_sigma da janela do bias (percent = fração dos steps).
- BUG3 (corrigido): máscara do ref_bias alocava (bs,1,L,L) fp32 + cópia bf16
  (centenas de MB/step). Agora direto no dtype de compute + finfo.min.
- Não-bug provado: ref_bias>1 satura por renormalização (razão efetiva
  b/(1+(b-1)*massa_ref); b=4 mede só 1.85-2.69x). Janela quantiza em 9 estados
  com 8 steps (testar 0.00/0.13/0.26, não 0.05). block_strength NÃO move massa
  de atenção (invariante em s) — ele gira o CONTEÚDO de K/V da ref; massa = ref_bias.
- Aberto: "block/fusion acima de 1.0 sem efeito" não reproduzido em CPU (linear,
  sem clamp; saturação refutada em teste com 28 blocos reais). Decidir com as
  linhas de debug em runtime (strength: blocks=..., ref_bias_efetivo=...).
