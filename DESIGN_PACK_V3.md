# CtxRush K2 — pack v3: controle total, zero superfície do ComfyUI

Proposta de arquitetura. Os nodes v1/v2 ficam intactos.

## O diagnóstico que motiva o pack

Passamos horas caçando divergências entre o node e o runner e achamos duas reais (contrato do
Qwen3-VL, máscara de atenção) — mas o gap não fechou. O padrão é claro: **cada peça do ComfyUI
que fica no caminho é uma chance de divergir**, e elas são muitas:

| peça do ComfyUI | o que ela faz por baixo | risco medido |
|---|---|---|
| `CLIPLoader` + `encode_from_tokens_scheduled` | MRoPE 3D + DeepStack no Qwen3-VL | **relL2 1,36** — confirmado |
| `UNETLoader` + dynamic VRAM loading | materializa pesos sob demanda, em bf16 | quebra o patch in-place; base ~2,8% ≠ treino |
| `LoraLoaderModelOnly` (turbo) | 270 patches lazy do ModelPatcher | ordem de aplicação vs fusão do runner |
| `KSampler` | `prepare_noise`, `process_latent_in/out`, `ModelSamplingFlux`, `CFGGuider`, escolha de sampler/scheduler | ruído diferente; grid errado se trocar scheduler |
| `comfy.ops` | cast/streaming de pesos | pode contornar o monkey-patch |

O v2 tenta corrigir cada uma por dentro. O v3 **não usa nenhuma delas**.

## Princípio

> O ComfyUI entra como **casca** (grafo visual, IMAGE/LATENT, Load/Save, upscalers, filas).
> O miolo — carregamento, encode, forward, sampling — é **o código do runner**, não o do ComfyUI.

## Duas camadas

### Camada A — `K2 Runner Bridge` (fidelidade garantida por construção)

Um node que executa **o pipeline do fork diffusion-pipe**, não uma reimplementação.

```
Load Image ──► K2 Runner Bridge ──► IMAGE ──► Save Image
                 ├ adapter (path)
                 ├ prompt / negative
                 ├ width / height / seed / steps
                 ├ text_guidance / reference_guidance   ← o CFG de 3 vias que o v2 não tem
                 └ variant (raw/turbo) / turbo_lora
```

Implementação: subprocesso isolado rodando `tools/infer_reference_adapter.py` com o venv e o
`sys.path` do fork (incluindo `submodules/ComfyUI`). Troca de dados por arquivos temporários
(PNG/NPY) num diretório de trabalho.

- **Por que subprocesso:** os dois ComfyUI (o do usuário e o do fork) não podem coexistir no mesmo
  interpretador — `comfy.*` colidiria e é justamente essa colisão que criou o bug do Qwen3-VL.
  Isolar por processo é a única garantia forte.
- **Custo:** carrega o modelo a cada chamada (~40 s). Mitigável com um **daemon persistente**:
  o primeiro uso sobe um servidor local que mantém o pipeline em memória e responde por socket;
  os nodes seguintes reaproveitam. É o mesmo truque do ComfyUI com o próprio modelo.
- **Ganho:** por construção, o resultado **é** o do runner. Não há o que divergir.

### Camada B — `K2 Native` (nativo, com controle total)

Para quem quer os latentes dentro do grafo (img2img, upscale, inpaint, encadear com outros nodes).
Reimplementa, mas **sem passar por nenhuma peça de risco**:

```
K2 Load Models ──┐
                 ├─► K2 Reference Encode ──┐
Load Image ──────┘                         ├─► K2 Sampler ──► LATENT ──► K2 Decode ──► IMAGE
                    K2 Conditioning ───────┤
                    K2 Adapter ────────────┘
```

**1. `K2 Load Models`** → `K2_MODELS`
   - carrega DiT/TE/VAE com `disable_dynamic=True`, como o trainer
   - aplica `dequantize()` com o **mesmo dtype do treino** (`float8`) — resolve o eixo dos 13% que
     o toggle in-place não conseguiu alcançar
   - funde a turbo LoRA nos pesos (como `apply_turbo_lora`), em vez de patches lazy
   - **encapsula os modelos num objeto opaco**: o ComfyUI não os gerencia, não faz streaming,
     não os move. Nós controlamos a memória (com offload explícito entre estágios).

**2. `K2 Reference Encode`** (IMAGE, N refs) → `K2_REFS`
   - crop-fit em pixel + VAE encode nativo (geometria do treino)
   - cópia VL por área 384², bicubic+antialias, piso 28px
   - emite os dois tensores e a ordem dos slots explicitamente

**3. `K2 Conditioning`** (prompt + K2_REFS) → `K2_COND`
   - encode pelo TE **do fork**, sem MRoPE/DeepStack
   - emite o contexto cru (30720) + máscara de atenção, como o trainer entrega
   - uncond grounded com caption vazia (o do `caption_dropout`)

**4. `K2 Adapter`** (path, strengths) → `K2_ADAPTER`
   - lê o contrato da metadata; expõe `block_strength`, `fusion_strength`, faixa de layers, curva
   - valida `control_family` e avisa se o adapter não casa com o pipeline

**5. `K2 Sampler`** → `LATENT`
   - **ruído próprio**: `torch.randn(shape, generator=cuda(seed))`, idêntico ao runner
   - **schedule oficial**: `build_krea2_timesteps` — não `ModelSamplingFlux` + scheduler do Comfy
   - **integração explícita**: `x = x + (next-current)·v`
   - **CFG de 3 vias**: `text_guidance` e `reference_guidance` separados, com o branch de
     referência zerada — o dial que hoje **não existe** no node e que o runner tem
   - saída também como `LATENT` do ComfyUI, para encadear

**6. `K2 Decode`** → IMAGE

### Extras que só existem tendo controle

- **`K2 Compare`** — roda A e B (duas configs) com o **mesmo ruído** e devolve as duas imagens
  lado a lado. O experimento pareado que fizemos à mão vira um node.
- **`K2 Trace`** — dumpa os tensores do primeiro forward (`x`, `ref`, `context`, `tvec`, `freqs`,
  `combined`, `out`) em `.npy`. Bisseção com o runner deixa de ser uma sessão de instrumentação
  manual e vira um clique.
- **`K2 Contract Check`** — pega um adapter e um pipeline e responde "batem ou não", listando cada
  campo. Guarda-chuva contra o próximo ComfyUI que mudar algo por baixo.

## Ordem de implementação sugerida

1. **Camada A com daemon** — fecha o problema do usuário *hoje*, com fidelidade garantida.
   É o menor caminho entre "não funciona" e "funciona no ComfyUI".
2. **`K2 Trace`** — porque ele acelera qualquer investigação futura, inclusive a atual.
3. **Camada B**, na ordem Load → Reference → Conditioning → Sampler → Decode, validando cada
   estágio contra o dump do runner antes de passar ao próximo.

## O que isso resolve, item a item

| problema atual | como o v3 resolve |
|---|---|
| Qwen3-VL com MRoPE/DeepStack | usa o TE do fork; nunca toca o do ComfyUI |
| dynamic loading quebrando patch/dtype | `disable_dynamic=True`, modelos sob nosso controle |
| base bf16 ≠ fp8 do treino | `dequantize()` com o dtype do treino, no carregamento |
| turbo como patches lazy | fusão nos pesos, como o runner |
| ruído do `prepare_noise` | `torch.randn` com generator CUDA, igual ao runner |
| scheduler errado lavando a identidade | schedule oficial embutido; não há o que escolher errado |
| `reference_guidance` inexistente | dial de primeira classe no sampler |
| divergência silenciosa futura | `K2 Contract Check` + `K2 Trace` |

## Custo honesto

- **Camada A:** ~200 linhas + daemon. Perde encadeamento de latentes; ganha garantia.
- **Camada B:** ~800–1200 linhas, e passa a ser código nosso para manter quando o Krea 2 ou o
  ComfyUI mudarem. Em troca, nenhuma peça de terceiro no caminho crítico.
- Os dois convivem: A para produção confiável, B para composição no grafo.
