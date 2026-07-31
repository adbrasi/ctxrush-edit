# Manutenção dos nodes — quando algo quebrar

Guia prático para diagnosticar e consertar os nodes deste pack, com foco nos
dois mais delicados: o **K2 Training Base** (patcher do modelo) e o **K2 Runner
Bridge** (wrapper do runner de inferência).

Leia a seção 1 antes de mexer em qualquer coisa. Ela é o que separa "consertar"
de "achar que consertou".

---

## 1. O princípio: o node existe para reproduzir o trainer

O adapter não aprendeu a rodar sobre o KREA 2 que o ComfyUI carrega. Ele
aprendeu a rodar sobre o KREA 2 **como a árvore de treino o constrói**, que é
numericamente diferente. Todo este pack existe para reconstruir aquela base
dentro do ComfyUI.

Consequência direta, e é a regra que mais importa:

> **A referência de correção é o runner, não a sua impressão da imagem.**
> Se a saída parece boa, isso não prova nada — o modo de falha caro aqui é
> silencioso. O adapter carrega, o forward roda, a imagem sai, e está errada.

### A régua numérica

Medições feitas na bisecção de 2026-07-30/31, com `x_T`, contexto e referência
pareados no primeiro forward:

| situação | `v relL2` contra o runner |
|---|---|
| node com `fp8_scaled` + `LoraLoaderModelOnly` (errado) | **0,581** |
| node com a base reconstruída (correto) | **0,031** |
| **efeito total do adapter no `v`, para escala** | **~0,38** |

Olhe a terceira linha antes de tirar qualquer conclusão das outras duas. **O
erro do node quebrado (0,58) era maior que o efeito inteiro do adapter (0,38).**
Ou seja: o node estava "mais errado do que o LoRA era forte". É por isso que a
saída parecia plausível mas nunca mantinha a personagem.

Alvo de saúde: **`relL2 ≤ 0,05`**. Acima de 0,1, alguma das cinco divergências
voltou.

---

## 2. Diagnóstico rápido — sintoma → causa provável

| sintoma | causa mais provável | seção |
|---|---|---|
| personagem "parecida" mas não a mesma; identidade escorrega | base fp8 errada (a nº1) | §4.1 |
| resultado piora depois de atualizar o ComfyUI | monkeypatch do Qwen3-VL parou de pegar | §4.3 |
| saída muda se você reordena os loaders | turbo entrando pelo `LoraLoaderModelOnly` | §4.2 |
| mesma seed dá imagem diferente do runner | RNG do ruído (CPU vs CUDA) | §4.5 |
| erro de contrato ao carregar adapter | metadata × config divergentes | §6 |
| Runner Bridge morre com OOM | VRAM não liberada antes do subprocesso | §5 |
| Runner Bridge: "runner nao encontrado" | env var de caminho | §5 |
| tudo parece certo e o resultado é ruim | **meça, não confie** | §1, §7 |

---

## 3. Antes de debugar: reproduza o problema com o Runner Bridge

O **K2 Runner Bridge** executa o runner real do fork em subprocesso isolado. Ele
é o padrão-ouro **na mesma máquina**, com o mesmo adapter, prompt, referência e
seed.

1. Rode o seu workflow normal e guarde a imagem.
2. Rode o `K2 Runner Bridge` com os mesmos parâmetros.
3. Compare.

- **As duas ruins** → o problema não é o node. É o adapter, o prompt ou o
  pedido. Pare de mexer no pack.
- **Bridge boa, node ruim** → é divergência de node. Siga para a §4.

Esse teste de 5 minutos evita a maior parte das caçadas inúteis.

---

## 4. As cinco divergências, e como cada uma volta a aparecer

Estas foram encontradas e corrigidas em ~6 h de bisecção. Cada uma pode
ressurgir sozinha, principalmente após atualização do ComfyUI.

### 4.1 Base fp8-scaled vs fp8 cru — a causa dominante

O checkpoint é `fp8_scaled`: pesos fp8 + um `weight_scale` por tensor. O
ComfyUI usa a escala. **O trainer descarta o `weight_scale`** e re-quantiza as
224 Linears de `blocks.*` no grid fp8 cru (`models/base.py:536,547`) — 2,5% a
6,7% de erro por matriz, até 26% dos pesos zerados. Depois soma a turbo e
re-quantiza de novo. O adapter aprendeu a compensar **essa** base.

Quem resolve: `K2 Training Base (fp8 do treino + turbo)`, em runtime, sem
checkpoint extra. Alternativa: `scripts/bake_training_base.py`, que assa um
checkpoint permanente.

**Duas armadilhas que já custaram tempo — não caia de novo:**

1. **`.to(fp8)` sobre `QuantizedTensor` é no-op algébrico.** Os pesos são
   `comfy_kitchen.tensor.base.QuantizedTensor`, e `_handle_to`
   (`comfy_kitchen/tensor/base.py:417-436`, linha 430) só reescreve o rótulo
   `orig_dtype`. `torch.equal(depois, antes) == True`, relL2 0,0. **Precisa de
   `dequantize()` antes** — é o que `_to_training_grid()` faz
   (`nodes_trainbase.py:53`).
2. **Sem forçar precisão no matmul, o erro volta.** É preciso setar
   `_full_precision_mm = True` (`ops.py:1277-1282`), senão cai no GEMM fp8
   nativo: 0,178 em vez de 0,031. É o `_force_bf16_matmul()`
   (`nodes_trainbase.py:67`).

**Como verificar que ainda funciona:** o node loga quantas Linears converteu.
Se logar 0, ou se `_is_quantized()` (`nodes_trainbase.py:49`) parar de
reconhecer a classe, ele virou um no-op silencioso.

> `_is_quantized` compara **nome de classe por string**. Se o ComfyUI renomear
> `QuantizedTensor`, o node passa a não fazer nada e ninguém é avisado. É o
> primeiro lugar para olhar depois de atualizar.

### 4.2 As 7 chaves `.diff_b` da turbo

A LoRA turbo tem, além dos pares `lora_down`/`lora_up`, **7 chaves `.diff_b`**
(deltas de bias). O runner **ignora**; o `LoraLoaderModelOnly` do ComfyUI
**aplica**. Resultado: bases diferentes.

O `K2 Training Base` funde a turbo ignorando essas 7 chaves e loga o número.

> **Regra permanente: a turbo entra SEMPRE dentro do `K2 Training Base`.**
> LoRAs de estilo vêm **depois** dele. Nunca use `LoraLoaderModelOnly` para a
> turbo.

### 4.3 Qwen3-VL: MRoPE 3D + DeepStack

O ComfyUI atual tem um `Qwen3VL.forward` próprio
(`comfy/text_encoders/qwen3vl.py`) com **MRoPE 3D** e injeção de **DeepStack**
nos tokens de visão. A árvore de treino não tem esse override e cai em
`BaseLlama.forward`: position_ids simples, sem DeepStack. Divergência de
contexto: `relL2 1,36`.

Quem resolve: `_training_vl_contract()` (`nodes_multiref.py:135`), que durante
o encode troca `Qwen3VL.forward` por `BaseLlama.forward` e restaura no
`finally`.

**Este é o ponto mais frágil do pack.** Ele depende de três nomes internos do
ComfyUI: o módulo `qwen3vl`, a classe `Qwen3VL` e o método `forward`. Qualquer
refatoração quebra — e quebra em silêncio, porque o `if` que detecta o override
(`nodes_multiref.py:167-170`) simplesmente não encontra nada e segue.

**Como verificar:** rode o encode com e sem o contrato e compare o tensor de
contexto. Se forem idênticos, o patch não está pegando.

### 4.4 Atenção GQA

`enable_gqa=True` e `repeat_interleave` explícito produzem resultados
diferentes. O treino usa o segundo. Quem resolve:
`_legacy_llama_attention_forward` (`nodes_multiref.py:74`, o
`repeat_interleave` nas linhas 125-126), instalado pelo mesmo
`_training_vl_contract`.

### 4.5 RNG do ruído

O KSampler do ComfyUI gera ruído em CPU; o runner usa `torch.randn` em CUDA.
Mesma seed, ruído diferente. Só importa quando você quer **comparação pareada**
com o runner — para uso normal, é indiferente.

Quem resolve: `K2 Runner Noise (CUDA, seed pareada)`.

---

## 5. Manutenção do K2 Runner Bridge

Ele monta um TOML a partir da **metadata do adapter**, salva as imagens em
arquivos temporários, executa o runner em subprocesso e lê o PNG de volta.

### Caminhos — o primeiro lugar que quebra em outra máquina

Três variáveis de ambiente, com defaults desta instância de treino:

| variável | default | o que é |
|---|---|---|
| `CTXRUSH_DIFFUSION_PIPE` | `/workspace/projects/diffusion-pipe` | raiz do fork |
| `CTXRUSH_PYTHON` | `/venv/main/bin/python` | python com as deps do fork |
| `CTXRUSH_LOG_DIR` | `/workspace/logs/k2_runner_bridge` | logs de execução |

Na sua máquina, **as três precisam ser ajustadas**. O node checa a existência
do runner e dá erro nomeando a variável (`nodes_runner_bridge.py:414`), então a
mensagem já diz o que fazer.

### OOM

`_free_vram()` chama `unload_all_models()` **antes** de lançar o subprocesso.
Isso é obrigatório: o runner carrega ~13 GB e, com um workflow anterior ainda
residente, o subprocesso morre com OOM — e o erro aparece no log dele, não no
ComfyUI, o que confunde.

Se der OOM mesmo assim: feche outros workflows, ou aumente `blocks_to_swap` no
TOML gerado.

### Logs

Ficam **fora** do diretório temporário de propósito: o `finally` apaga o temp, e
apagar o log justamente quando deu erro destrói a única evidência. Rotação
automática mantém os 20 mais recentes (`_rotate_logs`).

**Quando o Bridge falhar, o log é a primeira coisa a ler** — o traceback real
está lá, não no console do ComfyUI.

### Timeout

`TimeoutExpired` grava a saída parcial no log e sugere aumentar `timeout_s`.
Referência alta e muitos steps custam tempo; não é bug.

### Quando o trainer muda

O TOML é gerado a partir da metadata, então adapters novos se autoconfiguram.
**Mas se o fork ganhar um campo novo de contrato**, o Bridge precisa aprender a
lê-lo — senão aquele campo cai no default do pipeline e diverge em silêncio.
Confira `_write_config()` (`nodes_runner_bridge.py:93`) contra
`expected_contract()` do runner sempre que o trainer mudar.

---

## 6. Erro de contrato ao carregar um adapter

`expected_contract()` (`tools/infer_reference_adapter.py:274`) compara a
metadata do adapter com a config de inferência e **falha** se divergir. Confere:
`model_type`, `sequence_layout`, `reference_model_timestep`, `position_mode`,
`condition_token_stride`, `control_family`.

Duas coisas que valem saber:

- **Nome do arquivo do modelo base NÃO faz parte do contrato.** `base_model_file`
  só gera `WARNING` (`infer_reference_adapter.py:353`). Renomear o base é
  seguro.
- **Nunca edite a metadata do `.safetensors` para "limpar" nomes.** O
  `model_type` de lá é comparado com o `model.type` da config; alterá-lo faz o
  adapter só funcionar com uma config igualmente alterada. Para publicar sem
  expor o nome do base, redija a **cópia publicada da config**
  (`tools/sanitize_config_for_hf.py`), nunca a metadata.

---

## 7. Bisecção: quando você precisa de número, não de opinião

O `nodes_multiref.py` tem uma facilidade de dump opt-in (`_elo_save`,
`_elo_sequence_snapshot`) que grava `.npy` de pontos-chave do forward: índices
selecionados da sequência, `row_mean`, `row_std`, `row_norm`, `tvec`, `freqs` e
a máscara de atenção.

Procedimento que funcionou:

1. Fixe **tudo**: mesmo adapter, prompt, referência, seed, tamanho.
2. Gere os dumps pelo node **e** pelo runner.
3. Compare **na ordem da sequência**: contexto → `x_T` → `tvec`/`freqs` →
   máscara → `v` de saída.
4. A **primeira** etapa que divergir é a causa. As posteriores são consequência.

**Duas armadilhas que eu mesmo caí, documentadas para você não repetir:**

- **`KSamplerAdvanced` com `add_noise=disable` zera o `x_T`.** O
  `CONST.noise_scaling` faz `sigma*noise + (1-sigma)*latent`; com sigma=1 e
  noise=0, `x=0`. Um dump com `std=0,0` invalida todo o teste — e não é óbvio.
  **Confira sempre o `std` do `x_T` antes de comparar qualquer coisa.**
- **Correlação de pixel não serve como métrica.** Não discrimina o que importa
  aqui. Use `relL2` e cosseno nos tensores intermediários.

---

## 8. Checklist depois de atualizar o ComfyUI

Faça na ordem. Leva ~15 minutos e evita semanas de confusão.

1. **O pack carrega?** Os imports em `__init__.py` são tolerantes: cada pack
   falha isolado e imprime `[CtxRush] <pack> indisponivel: ...`. **Leia o
   console na subida** — um pack faltando é silencioso na interface.
2. **`QuantizedTensor` ainda se chama assim?** Se não, `_is_quantized`
   (`nodes_trainbase.py:49`) virou no-op. Sintoma: o log do `K2 Training Base`
   diz 0 Linears convertidas.
3. **`_full_precision_mm` ainda existe?** `_force_bf16_matmul` usa `hasattr`, ou
   seja, some em silêncio se o atributo for renomeado.
4. **`Qwen3VL.forward` ainda é um override próprio?** Se o caminho do módulo ou
   o nome da classe mudou, `_training_vl_contract` não patcheia nada.
5. **Teste pareado com o Runner Bridge** (§3). É a checagem que pega tudo que as
   quatro anteriores não pegaram.

Se algo divergir, o `git log` deste repo tem o histórico das correções — os
commits `01835ff` (causa raiz fp8), `df36df8` (training_base_quant) e `2c7ae4c`
(Runner Bridge) são os que descrevem as decisões.

---

## 9. O que nunca fazer

- **Não use `LoraLoaderModelOnly` para a turbo.** Ela entra dentro do
  `K2 Training Base`.
- **Não edite a metadata do adapter** para limpar nomes (§6).
- **Não confie em inspeção visual** para validar uma correção numérica. Mede.
- **Não conclua de um único teste pareado** sem antes checar o `std` do `x_T`.
- **Não "simplifique" o `_training_vl_contract` restaurando fora do `finally`.**
  Se o encode levantar exceção, o `Qwen3VL.forward` fica patcheado para o resto
  da sessão do ComfyUI e contamina todos os outros workflows.

---

## 10. Mapa dos nodes

| node | papel |
|---|---|
| `K2 Training Base (fp8 do treino + turbo)` | reconstrói a base do treino em runtime — **a peça central** |
| `K2 Load Reference (PIL, como o runner)` | decodifica a referência como o runner (PIL, não PyAV) |
| `K2 Runner Noise (CUDA, seed pareada)` | ruído em CUDA, para comparação pareada |
| `CtxRush - Krea 2 Multi-Ref Grounded (v2)` | o caminho principal de inferência |
| `K2 Runner Bridge (runner exato)` | executa o runner real em subprocesso — **padrão-ouro** |
| `K2 Runner Bridge + 3 LoRAs` | idem, com até 3 LoRAs extras |
| `K2 Load Models` / `Adapter` / `Reference Encode` / `Conditioning` / `Sampler` / `Decode` | pack K2 Native, controle total do pipeline |
| `K2 Prompt Rewriter (system + pedido)` | monta system prompt + pedido para um LLM externo |
