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

## K2 Prompt Rewriter (`K2 Prompt Rewriter (system + pedido)`)

Monta as duas strings que um node de LLM precisa para reescrever o seu pedido
no dialeto certo. Não chama LLM nenhum — só entrega texto.

- **`dialeto`**: menu de 11 tipos (EDIT, NEXT, PANEL, CTX, POSE, EXPRESSION,
  OBJECT, STYLE, BACKGROUND, CROP, INSCENE), com o número de exemplos de
  treino de cada um ao lado.
- **`pedido`**: escreva em português mesmo.
- **saídas**: `system_prompt` e `user_prompt` (`TYPE: <n>` + `REQUEST: <pedido>`).

Ligue as duas nas entradas equivalentes do seu node de LLM. A resposta vem como
um único objeto JSON — extraia a chave `prompt_final` antes de mandar para o
`CLIPTextEncode`.

**Por que existe:** o adapter foi treinado em 30.000 captions que não são um
estilo só, e sim 11 dialetos com gramáticas de superfície distintas. Um pedido
em português corrido não se parece com nenhum deles, e o modelo responde pior.
O system prompt fica em `prompts/rewriter_manual_dialect.md`; a versão com
roteamento automático (sem seletor) está em `prompts/rewriter_auto_router.md`.

## v2 — Multi-Ref Grounded (`CtxRush - Krea 2 Multi-Ref Grounded (v2)`)

Para adapters `krea2_multiref_grounded` (ex.: k2-context-rush-ofc-beta1).
O node lê o contrato da **metadata do próprio adapter** (grounding 384² por
área, vision blocks `image 1:`, refs a t=0, width-shift cumulativo) — os
dials manuais são só override. Aceita até 4 referências; com N>1, enderece
no prompt como `image 1` / `image 2` (sem `<>`). Negative vazio reproduz o
uncond exato do treino. Saídas steps/cfg/mu já vêm com os defaults oficiais
(raw 28/5.5/mu-por-resolução; turbo 8/1.0/1.15).

Use o v1 (`CtxRushKrea2OminiGroundedApply`) apenas para os adapters
`krea2_omini_grounded` antigos (grounding 768 longest-side, layout plain).

## Workflow v2 com paridade e LoRAs adicionais

```text
UNETLoader (Krea2 raw fp8_scaled, weight_dtype=default)
  -> K2 Training Base (turbo escolhida aqui)
  -> [LoraLoaderModelOnly: LoRA Krea2 adicional, opcional]
  -> [outros LoraLoaderModelOnly compatíveis, opcionais]
  -> CtxRush - Krea 2 Multi-Ref Grounded (v2)
  -> guider / sampler
```

Regras:

- A LoRA **Turbo** pertence exclusivamente ao `K2 Training Base`, que a funde
  como o runner. Não carregue a mesma Turbo novamente no
  `LoraLoaderModelOnly`.
- LoRAs adicionais do diffusion model entram depois do `K2 Training Base` e
  antes do CtxRush. O mecanismo padrão foi validado em 1024 px: 270 patches
  permaneceram anexados, enquanto o CtxRush manteve 224/224 chamadas roteadas.
- A LoRA adicional precisa ter chaves e dimensões compatíveis com Krea2.
  LoRAs de Flux, SDXL ou outra arquitetura não são intercambiáveis.
- LoRA de text encoder não faz parte desse teste. Ela requer `LoraLoader`
  (model + clip) e pesos realmente compatíveis com o Qwen3-VL usado pelo Krea2.

O forward v2 também preserva os hooks `post_input`, `patches_replace` e o
protocolo `control["output"]` dos DiTs single-stream. Um ControlNet sintético
passou de ponta a ponta em 1024 px e alterou a saída, sem interferir nas
224 chamadas do adapter. Isso prova o encanamento do node; não transforma um
ControlNet de Flux/SDXL em Krea2. É necessário um checkpoint ControlNet
treinado ou portado para os 6.144 canais e a geometria de tokens do Krea2.

## v3 controlado — força real da imagem e máscara espacial

O v2 acima continua sendo o caminho canônico e não foi substituído. Para
controlar a influência total da referência, use os nodes separados:

```text
K2 Training Base
  -> [LoRAs Krea2 adicionais, opcionais]
  -> CtxRush - Krea 2 Multi-Ref Controlled (v3):model
  -> K2 Reference Guider
  -> SamplerCustomAdvanced

Controlled (v3):reference_conditioning
  -> K2 Reference Guider:reference_conditioning

MASK -> K2 Reference Mask Strength
     -> K2 Reference Guider:reference_mask
```

O guider mede duas predições sobre o mesmo `x` e timestep:

```text
v = v_sem_referencia + G(x,y) * (v_com_referencia - v_sem_referencia)
```

Consequências verificáveis:

- força `1`, sem máscara: usa um fast path e devolve a predição canônica com
  referência, sem fazer a subtração;
- força `0`: usa a branch sem referência;
- valores entre `0` e `1`: interpolação convexa;
- valores acima de `1`: extrapolação na direção medida da referência;
- máscara branca: força principal do guider;
- máscara preta: `outside_strength`;
- cinza/feather: interpolação linear entre as duas forças.

O modo `full_off` controla os dois canais da imagem: zera os tokens VAE e usa
conditioning Qwen sem vision blocks na branch desligada. Ele avalia os quatro
cantos `negative/positive × reference off/on`, portanto texto e imagem viram
eixos independentes. O modo `runner_vae_only` reproduz literalmente
`--reference-guidance` do runner: zera apenas o VAE e mantém o Qwen grounded.

Essa construção prova os endpoints e a interpolação, não a qualidade
perceptual de cada valor. A branch totalmente desligada é uma intervenção de
inferência (o treino atual não usou `condition_dropout`), e a autoatenção
global pode propagar indiretamente informação entre áreas ao longo dos steps.
A máscara controla diretamente a atualização espacial em cada step, mas não é
uma garantia de segmentação rígida. Faça A/B pareado por seed.

Para paridade use `reference_guidance=1`, sem máscara. Turbo usa
`text_guidance=1`; Raw usa `5.5`. Com Turbo, mudar a força da referência custa
normalmente dois forwards por step. Em Raw, `full_off` pode custar quatro
forwards porque mede os quatro cantos explicitamente.

`reference_resize=runner_pil` aplica o mesmo
`PIL.ImageOps.fit(..., LANCZOS)` do runner. Use `comfy_tensor` apenas quando a
referência veio de outro node como imagem float e não deve ser quantizada para
8 bits.

### ControlNet no caminho controlado

O setup v3 também expõe as quatro saídas `CONDITIONING`. Para usar um
ControlNet, aplique o mesmo controle às quatro branches e reconecte-as com
`K2 Reference Conditioning Pack`. Isso mantém o ControlNet constante enquanto
o guider mede somente o eixo da referência. A compatibilidade dos pesos ainda
depende de existir um ControlNet realmente treinado para a arquitetura Krea2.

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
