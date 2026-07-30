"""CtxRush — Krea 2 Multi-Ref Grounded Apply (v2).

Node para adapters treinados com ``type = krea2_multiref_grounded`` no fork
diffusion-pipe-easycontrol (branch ic-lora). Diferenças para o node v1
(CtxRushKrea2OminiGroundedApply), todas medidas contra o código que o trainer
executa (models/krea2_multiref.py + models/krea2_edit.py):

1. **Autoconfiguração pela metadata do adapter.** O trainer grava o contrato
   no header do .safetensors (vl_image_max_pixels, vl_prompt_layout,
   slot_axis, max_refs, reference_model_timestep, position_mode…). O node lê
   e aplica; os dials manuais existem só como override explícito ('auto').
   O contrato REAL é o que o código executou — e a metadata do multiref é
   gravada pós-fix a66d7ba, então é confiável.

2. **Vision prompt no vocabulário do treino.** O multiref treina com
   ``image 1: <vision>caption`` (picture_n + label 'image'). O v1 emitia
   'Picture 1:' hardcoded (vocabulário errado) e default 'plain' (sem
   rótulo). Aqui o layout e o rótulo vêm da metadata/opção.

3. **Grounding com cap de ÁREA (384² default), não longest-side.** O treino
   redimensiona a cópia do Qwen3-VL por área total (prepare_vl_image:
   bicubic+antialias, downscale-only, piso 28px). O v1 usava longest-side
   768 por default no caminho grounded — fora do contrato deste adapter.

4. **N referências com offset RoPE CUMULATIVO por slot.** Slot i vive em
   ``w += (i+1)·W`` (com todas as refs crop-fit ao tamanho do target). Offset
   constante tornaria os spans permutation-invariant — indistinguíveis por
   construção (docs/KREA2_MULTIREF_MACRO.md §1). Suporta slot_axis='frame'
   (índice discreto no eixo de frame) para adapters futuros.

5. **Uncond fiel ao caption_dropout do treino.** O uncond cacheado no treino
   é caption VAZIO com os MESMOS vision blocks e referência. O negative
   default aqui reproduz isso (grounded, mesmo layout).

O que é igual ao v1 (e permanece correto): LoRA em runtime bf16 (nunca
fundida em fp8), delta routado mascarado ao span das referências, txtfusion
global com strength separada, refs a t=0 per-token, mu oficial por
resolução (raw) ou 1.15 (turbo), janela/curva de strength por step.
"""

import contextlib
import json
import struct

import torch

import comfy.ldm.common_dit
import comfy.model_management
import comfy.patcher_extension
import folder_paths
from comfy.ldm.flux.layers import timestep_embedding
from comfy.text_encoders.krea2 import KREA2_TEMPLATE
from einops import rearrange

from .nodes import (
    DEFAULT_VL_MAX_PIXELS,
    KREA2_TURBO_MU,
    VISION_BLOCK,
    _FullLoraScope,
    _MaskedLoraScope,
    _NullScope,
    _apply_krea2_sampling,
    _crop_fit,
    _ctxrush_layer_scales,
    _ctxrush_schedule_mult,
    _empty_krea_latent,
    _fit_vl,
    _krea2_raw_mu,
    _load_omini_lora,
    _require_single_image,
)


def _legacy_llama_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    freqs_cis=None,
    optimized_attention=None,
    past_key_value=None,
    sliding_window=None,
):
    """Qwen/Llama attention path used by the ComfyUI revision that trained the adapter."""
    from comfy.text_encoders.llama import apply_rope

    batch_size, seq_length, _ = hidden_states.shape
    xq = self.q_proj(hidden_states)
    xk = self.k_proj(hidden_states)
    xv = self.v_proj(hidden_states)
    xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
    xk = xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
    xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
    if self.q_norm is not None:
        xq = self.q_norm(xq)
    if self.k_norm is not None:
        xk = self.k_norm(xk)
    xq, xk = apply_rope(xq, xk, freqs_cis=freqs_cis)

    present_key_value = None
    if past_key_value is not None:
        index = 0
        num_tokens = xk.shape[2]
        if len(past_key_value) > 0:
            past_key, past_value, index = past_key_value
            if past_key.shape[2] >= index + num_tokens:
                past_key[:, :, index:index + xk.shape[2]] = xk
                past_value[:, :, index:index + xv.shape[2]] = xv
                xk = past_key[:, :, :index + xk.shape[2]]
                xv = past_value[:, :, :index + xv.shape[2]]
                present_key_value = (past_key, past_value, index + num_tokens)
            else:
                xk = torch.cat((past_key[:, :, :index], xk), dim=2)
                xv = torch.cat((past_value[:, :, :index], xv), dim=2)
                present_key_value = (xk, xv, index + num_tokens)
        else:
            present_key_value = (xk, xv, index + num_tokens)
        if sliding_window is not None and xk.shape[2] > sliding_window and seq_length == 1:
            xk = xk[:, :, -sliding_window:]
            xv = xv[:, :, -sliding_window:]
            attention_mask = (
                attention_mask[..., -sliding_window:]
                if attention_mask is not None else None
            )

    xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
    xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
    output = optimized_attention(
        xq, xk, xv, self.num_heads,
        mask=attention_mask, skip_reshape=True,
    )
    return self.o_proj(output), present_key_value


@contextlib.contextmanager
def _training_vl_contract(enabled=True):
    """Restaura o contrato de encode de texto/visão sob o qual o adapter treinou.

    O ComfyUI atual tem um `Qwen3VL.forward` próprio (comfy/text_encoders/
    qwen3vl.py) que usa **MRoPE 3D** e injeta **DeepStack** nos tokens de
    visão. O ComfyUI que treinou este adapter (submodules/ComfyUI do fork
    diffusion-pipe) não tem esse override e cai em `BaseLlama.forward`:
    position_ids simples, sem DeepStack.

    Medido, mesmo prompt/imagem/pesos: o contexto diverge com relL2 = 1.36
    (maior que o próprio sinal), cosseno médio 0.83, 152/169 tokens abaixo de
    0.99. SEM imagem os dois batem em 8 casas — a divergência é exclusiva do
    caminho de grounding visual.

    Consequência: o `txtfusion` (o canal que LÊ a referência) recebe uma
    distribuição fora daquela em que foi treinado. Os blocks continuam
    aplicando o delta, então estilo e composição saem — mas a identidade não.
    É o sintoma "personagem parecido, não o mesmo".

    Este contexto desfaz o override APENAS durante o encode, e restaura em
    seguida. Em ComfyUI antigo (sem override) é no-op.
    """
    if not enabled:
        yield
        return
    try:
        import comfy.text_encoders.qwen3vl as _q3
        import comfy.text_encoders.llama as _ll
        from comfy.text_encoders.llama import BaseLlama
    except Exception:
        yield
        return
    own = _q3.Qwen3VL.__dict__.get('forward')
    attention_forward = _ll.Attention.forward
    if own is not None:
        _q3.Qwen3VL.forward = BaseLlama.forward
    _ll.Attention.forward = _legacy_llama_attention_forward
    try:
        yield
    finally:
        _ll.Attention.forward = attention_forward
        if own is not None:
            _q3.Qwen3VL.forward = own


def _read_safetensors_metadata(path):
    try:
        with open(path, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_len))
        return header.get('__metadata__', {}) or {}
    except Exception:
        return {}


def _resolve_contract(metadata, vl_max_pixels_opt, vl_label_opt, reference_timestep_opt):
    """Metadata do adapter primeiro; opção manual só quando != 'auto'."""
    contract = {}

    md_pixels = int(metadata.get('vl_image_max_pixels', 0) or 0)
    contract['vl_max_pixels'] = (
        vl_max_pixels_opt if vl_max_pixels_opt > 0
        else (md_pixels if md_pixels >= 28 * 28 else DEFAULT_VL_MAX_PIXELS)
    )

    layout = metadata.get('vl_prompt_layout', 'picture_n_vision_blocks')
    contract['prompt_style'] = 'plain' if layout.startswith('plain') else 'picture_n'

    # O trainer ainda não grava o rótulo na metadata; o multiref usa 'image'.
    contract['vl_label'] = vl_label_opt.strip() or 'image'

    if reference_timestep_opt != 'auto':
        contract['reference_timestep'] = reference_timestep_opt
    else:
        contract['reference_timestep'] = (
            'target' if float(metadata.get('reference_model_timestep', 0) or 0) > 0
            else 'zero'
        )

    contract['slot_axis'] = metadata.get('slot_axis', 'width')
    contract['position_offset'] = float(metadata.get('reference_position_offset', 1.0) or 1.0)
    contract['max_refs'] = int(metadata.get('max_refs', 0) or 0)
    contract['family'] = metadata.get('control_family', '')
    return contract


def _build_vl_prompt(num_images, prompt, style, label):
    if style == 'plain':
        return VISION_BLOCK * num_images + prompt
    return ''.join(f'{label} {i + 1}: {VISION_BLOCK}' for i in range(num_images)) + prompt


# --- DEBUG -----------------------------------------------------------------
# O modo de falha caro deste node é SILENCIOSO: o adapter carrega, o forward
# roda, a imagem sai — e o delta simplesmente não chegou aos tokens certos.
# Nesse caso a saída fica "parecida com a referência" em vez de idêntica, que
# é indistinguível de adapter fraco ou de prompt ruim. O debug abaixo torna
# cada elo verificável no console.
_DEBUG = {'on': False, 'calls': 0, 'lora_hits': 0, 'lora_misses': 0, 'logged': False}


def _dbg(msg):
    if _DEBUG['on']:
        print(f'[MultiRef][debug] {msg}', flush=True)


def _tensor_stats(name, t):
    try:
        f = t.detach().float()
        return (f'{name}: shape={tuple(t.shape)} dtype={t.dtype} dev={t.device} '
                f'min={f.min():.4f} max={f.max():.4f} mean={f.mean():.4f} std={f.std():.4f} '
                f'nan={int(torch.isnan(f).sum())}')
    except Exception as e:  # pragma: no cover
        return f'{name}: <stats falhou: {e!r}>'


def _elo_save(prefix, name, value):
    """Opt-in tensor dump for runner-vs-node forward bisection."""
    if not prefix:
        return
    import os
    import numpy as np

    path = f'{prefix}.{name}.npy'
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(path):
        np.save(path, value.detach().float().cpu().numpy())


def _elo_sequence_snapshot(prefix, name, combined, text_length, target_length,
                           timestep_features=None, tvec=None, freqs=None,
                           attention_mask=None):
    if not prefix:
        return
    sequence_length = combined.shape[1]
    indices = sorted({
        0,
        max(text_length - 1, 0),
        text_length,
        text_length + target_length // 2,
        text_length + target_length - 1,
        text_length + target_length,
        sequence_length - 1,
    })
    indices = [index for index in indices if 0 <= index < sequence_length]
    index_tensor = torch.tensor(indices, device=combined.device)
    _elo_save(prefix, f'{name}.indices', index_tensor)
    _elo_save(prefix, f'{name}.selected', combined[:, indices])
    work = combined.detach().float()
    _elo_save(prefix, f'{name}.row_mean', work.mean(dim=-1))
    _elo_save(prefix, f'{name}.row_std', work.std(dim=-1))
    _elo_save(prefix, f'{name}.row_norm', torch.linalg.vector_norm(work, dim=-1))
    if timestep_features is not None:
        _elo_save(prefix, f'{name}.timestep_features', timestep_features)
    if tvec is not None:
        _elo_save(prefix, f'{name}.tvec_selected', tvec[:, indices])
    if freqs is not None:
        _elo_save(prefix, f'{name}.freqs_selected', freqs.index_select(-4, index_tensor))
    if attention_mask is not None:
        _elo_save(prefix, f'{name}.attention_mask', attention_mask)


class _TrainingBaseQuant:
    """VERIFICA (não conserta) se a base carregada é a base de treino.

    CAUSA RAIZ nº1 do gap node-vs-runner, medida com x_T/contexto/referência
    pareados no 1º forward (dumps em /workspace/outputs/node_test/):

        node com fp8_scaled + LoraLoaderModelOnly   v relL2 = 0.58090
        node com a base assada (scripts/bake_...)   v relL2 = 0.03116
        (efeito TOTAL do adapter no v, p/ escala:   relL2 ~ 0.38)

    O trainer roda com ``diffusion_model_dtype = 'float8'``
    (train.toml:34) e em ``models/base.py:536,547`` DESCARTA o
    ``weight_scale`` do checkpoint fp8_scaled, re-quantizando as 224 Linears
    de ``blocks.*`` no grid fp8 CRU (2.5%–6.7% de erro por matriz, até 26%
    dos pesos zerados). Depois ``apply_turbo_lora`` soma a turbo e
    re-quantiza de novo. O adapter aprendeu a compensar ESSA base.

    A versão anterior desta classe tentava consertar in-place com
    ``w.data.to(fp8).to(w.dtype)``. Isso é um **no-op algébrico**: os pesos
    são ``comfy_kitchen.tensor.base.QuantizedTensor`` e ``_handle_to``
    (comfy_kitchen/tensor/base.py:417-436, linha 430) só reescreve o rótulo
    ``orig_dtype`` — ``torch.equal(depois, antes) == True``, relL2 0.0.

    O conserto correto é assar o checkpoint uma vez (ver
    ``scripts/bake_training_base.py``), o que reproduz os pesos do runner
    BIT A BIT. Aqui só avisamos quando a base carregada não é aquela.
    """

    _checked = set()

    def __init__(self, entries):
        self.modules = [e[0] for e in entries]

    def __enter__(self):
        for module in self.modules[:4]:
            key = id(module)
            if key in _TrainingBaseQuant._checked:
                continue
            _TrainingBaseQuant._checked.add(key)
            w = getattr(module, 'weight', None)
            if w is None or getattr(w, 'ndim', 0) != 2:
                continue
            try:
                full = w.dequantize() if hasattr(w, 'dequantize') else w.data
                sl = full[:128, :128].float()
                erro = (sl.to(torch.float8_e4m3fn).float() - sl).norm() / sl.norm().clamp_min(1e-12)
            except Exception as exc:                      # pragma: no cover
                _dbg(f'base: nao consegui inspecionar os pesos ({exc})')
                continue
            if float(erro) > 1e-6:
                _dbg('AVISO: a base carregada NAO esta no grid fp8 do treino '
                     f'(erro de quantizacao residual {float(erro) * 100:.2f}% > 0). '
                     'O adapter foi treinado sobre pesos fp8_e4m3fn SEM escala '
                     '(train.toml diffusion_model_dtype=float8). Rode '
                     'scripts/bake_training_base.py e carregue o .safetensors '
                     'assado no UNETLoader, SEM LoraLoaderModelOnly (a turbo ja '
                     'vai fundida). Medido: v relL2 0.581 -> 0.031.')
            else:
                _dbg('base OK: pesos no grid fp8 do treino.')
        return self

    def __exit__(self, *exc):
        return False


class _CountingMaskedLoraScope(_MaskedLoraScope):
    """Igual ao original, mas conta quantas chamadas realmente aplicaram o
    delta e quantas passaram batido pela guarda de seq_len. Se `aplicado` for
    0, a LoRA routada NÃO está agindo — a causa nº1 de 'parece mas não é'."""

    def __enter__(self):
        s, e = self.span
        seq_len, scale = self.seq_len, self.scale
        for entry in self.entries:
            module, lora_a, lora_b = entry[:3]
            entry_scale = entry[3] if len(entry) > 3 else 1.0
            orig = module.forward

            def wrapped(x, *args, _orig=orig, _a=lora_a, _b=lora_b,
                        _entry_scale=entry_scale, **kwargs):
                out = _orig(x, *args, **kwargs)
                if x.ndim >= 3 and x.shape[-2] == seq_len:
                    piece = x[..., s:e, :].to(_a.dtype)
                    delta = torch.nn.functional.linear(
                        torch.nn.functional.linear(piece, _a), _b)
                    out[..., s:e, :] += (delta * scale * _entry_scale).to(out.dtype)
                    _DEBUG['lora_hits'] += 1
                else:
                    _DEBUG['lora_misses'] += 1
                return out

            self._originals.append((module, orig))
            module.forward = wrapped
        return self
# ---------------------------------------------------------------------------


def _krea2_multiref_forward(m, x, timesteps, context, ref_latents, blocks_state,
                            fusion_entries, strength, transformer_options,
                            fusion_strength, reference_timestep, slot_axis,
                            position_offset, text_attention_mask=None,
                            training_base_quant=False, control=None):
    """Espelho fiel de Krea2MultiRefInitialLayer.forward (o que treinou):
    N spans concatenados após o target, offset RoPE cumulativo por slot,
    t=0 (ou target) per-token nos spans, txtfusion antes do txtmlp."""
    import os as _os_probe
    transformer_options = (transformer_options or {}).copy()
    dump_prefix = _os_probe.environ.get('NODE_DUMP')
    source_prefix = _os_probe.environ.get('NODE_BISECT_SOURCE')
    if source_prefix:
        import numpy as _np_probe
        override_context = _os_probe.environ.get('NODE_BISECT_CONTEXT', '1') != '0'
        override_reference = _os_probe.environ.get('NODE_BISECT_REFERENCE', '1') != '0'
        if override_context:
            raw_context = torch.from_numpy(_np_probe.load(source_prefix + '.cond0.npy'))
            context = raw_context.to(device=context.device, dtype=context.dtype)
        if override_reference:
            raw_reference = torch.from_numpy(_np_probe.load(source_prefix + '.ref.npy'))
            ref_latents = [raw_reference]
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)
    bs, _, H_orig, W_orig = x.shape
    patch = m.patch
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2:]
    h_, w_ = H // patch, W // patch
    device = x.device

    context = m._unpack_context(context)
    _elo_save(dump_prefix, 'context_raw', context)
    tgt = m.first(rearrange(x, 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=patch, pw=patch))
    _elo_save(dump_prefix, 'target_tokens', tgt)

    # --- N spans de referência, offset cumulativo -------------------------
    ref_tokens_list, ref_pos_list, ref_token_lengths = [], [], []
    width_cursor = float(w_)
    for slot, src in enumerate(ref_latents):
        if src.ndim == 5:
            src = src.reshape(src.shape[0] * src.shape[2], src.shape[1], *src.shape[-2:])
        src = src.to(device, x.dtype)
        if src.shape[0] != bs:
            src = src[:1].expand(bs, *src.shape[1:])
        src = comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch))
        grid_h, grid_w = src.shape[-2] // patch, src.shape[-1] // patch
        ref_tokens_list.append(m.first(
            rearrange(src, 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=patch, pw=patch)
        ))
        ref_token_lengths.append(grid_h * grid_w)
        pos = torch.zeros(grid_h, grid_w, 3, device=device, dtype=torch.float32)
        pos[..., 1] = torch.arange(grid_h, device=device, dtype=torch.float32)[:, None]
        pos[..., 2] = torch.arange(grid_w, device=device, dtype=torch.float32)[None, :]
        pos = pos.reshape(1, grid_h * grid_w, 3).repeat(bs, 1, 1)
        if slot_axis == 'frame':
            pos[..., 0] = position_offset + slot
        else:
            pos[..., 2] = pos[..., 2] + width_cursor
        width_cursor += float(grid_w)
        ref_pos_list.append(pos)
    ref = torch.cat(ref_tokens_list, dim=1)
    refpos = torch.cat(ref_pos_list, dim=1)
    _elo_save(dump_prefix, 'reference_input', ref_latents[0])
    _elo_save(dump_prefix, 'reference_tokens', ref)

    # O runner calcula duas coisas distintas:
    #   1. o embedding escalar do timestep do target, usado pela camada final;
    #   2. tmlp+tproj sobre TODOS os timesteps por token numa unica chamada.
    # Expandir um tvec calculado com shape (B, 1, D) e matematicamente
    # equivalente, mas escolhe outro kernel GEMM BF16 e diverge numericamente
    # do treino ao longo dos 28 blocos.
    t = m.tmlp(timestep_embedding(timesteps, m.tdim).unsqueeze(1).to(tgt.dtype))

    fusion_scale = strength if fusion_strength is None else fusion_strength
    use_fusion = bool(fusion_entries) and fusion_scale > 0
    with _FullLoraScope(fusion_entries, fusion_scale) if use_fusion else _NullScope():
        context = m.txtfusion(context, mask=None, transformer_options=transformer_options)
    _elo_save(dump_prefix, 'context_txtfusion', context)
    context = m.txtmlp(context)
    _elo_save(dump_prefix, 'context_txtmlp', context)

    txtlen, tgtlen, reflen = context.shape[1], tgt.shape[1], ref.shape[1]

    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    grid = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    grid[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    grid[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    tgtpos = grid.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
    img = torch.cat([tgt, ref], dim=1)
    imgpos = torch.cat([tgtpos, refpos], dim=1)

    # Preserva o protocolo de patches do ComfyUI. IP-Adapters e outros model
    # patches usam `post_input` para alterar os tokens já projetados. Como o
    # roteamento da nossa LoRA depende da fronteira target/ref, recusamos
    # silenciosamente perigosa mudança de comprimento.
    transformer_options["reference_image_num_tokens"] = ref_token_lengths
    patches = transformer_options.get("patches", {})
    if "post_input" in patches:
        for patch_fn in patches["post_input"]:
            patched_input = patch_fn({
                "img": img,
                "txt": context,
                "img_ids": imgpos,
                "txt_ids": txtpos,
                "transformer_options": transformer_options,
            })
            img = patched_input["img"]
            context = patched_input["txt"]
            imgpos = patched_input["img_ids"]
            txtpos = patched_input["txt_ids"]
        if context.shape[1] != txtlen or img.shape[1] != tgtlen + reflen:
            raise RuntimeError(
                "Um model patch alterou a quantidade de tokens do Krea2. "
                "CtxRush precisa preservar as fronteiras texto/target/referencia "
                "para rotear a LoRA condition-only com seguranca."
            )

    combined = torch.cat([context, img], dim=1)

    target_timesteps = timesteps[:, None].expand(bs, txtlen + tgtlen)
    if reference_timestep == 'target':
        reference_timesteps = timesteps[:, None].expand(bs, reflen)
    else:
        reference_timesteps = timesteps.new_zeros(bs, reflen)
    per_token_timesteps = torch.cat([target_timesteps, reference_timesteps], dim=1)
    embedded_timesteps = timestep_embedding(
        per_token_timesteps.reshape(-1), m.tdim
    ).reshape(bs, combined.shape[1], m.tdim)
    tvec = m.tproj(m.tmlp(embedded_timesteps.to(combined.dtype)))

    freqs = m.pe_embedder(torch.cat([txtpos, imgpos], dim=1))

    entries = blocks_state['entries']
    if blocks_state.get('device') != device:
        entries = [
            (entry[0], entry[1].to(device), entry[2].to(device), *entry[3:])
            for entry in entries
        ]
        blocks_state['entries'] = entries
        blocks_state['device'] = device

    seq_len = txtlen + tgtlen + reflen
    _DEBUG['calls'] += 1
    if _DEBUG['on'] and not _DEBUG['logged']:
        _DEBUG['logged'] = True
        _dbg(f'--- forward #1 (os demais steps sao iguais) ---')
        _dbg(f'sequencia: texto={txtlen} target={tgtlen} refs={reflen} total={seq_len} '
             f'| grid target={h_}x{w_} | refs={len(ref_latents)} | batch={bs}')
        _dbg(f'span da LoRA routada: [{txtlen + tgtlen}, {seq_len}) '
             f'= exatamente os {reflen} tokens de referencia')
        _dbg(f'RoPE: target w=[0,{w_}) | ref w=[{w_},{int(refpos[..., 2].max()) + 1}) '
             f'| eixo frame ref={float(refpos[..., 0].max()):.0f} | slot_axis={slot_axis}')
        _dbg(f'timestep ref={reference_timestep} (sigma do step={float(timesteps.flatten()[0]):.4f})')
        _dbg(_tensor_stats('latente da referencia', ref_latents[0]))
        _dbg(_tensor_stats('contexto de texto (pos txtfusion)', context))
        _dbg(f'strength: blocks={strength} txtfusion={fusion_scale} '
             f'| entries: {len(entries)} blocks, {len(fusion_entries)} txtfusion')
    # MASCARA DE ATENCAO — o trainer SEMPRE passa uma (krea2_multiref.py:150-166):
    #   valid_keys = cat([text_attention_mask, ones(target+refs)]) -> (B,1,1,L)
    # Sem ela, o softmax de cada bloco tambem distribui peso sobre as chaves de
    # PADDING do texto, diluindo os tokens reais — inclusive os da referencia.
    # Como o adapter aprendeu a ler a referencia SOB a mascara, omiti-la faz o
    # sinal da referencia chegar enfraquecido ("como se a LoRA estivesse
    # desligada"). Reconstruimos exatamente o mesmo tensor.
    if text_attention_mask is None:
        text_valid = torch.ones(bs, txtlen, dtype=torch.bool, device=device)
    else:
        text_valid = text_attention_mask.to(device=device, dtype=torch.bool)
        if text_valid.ndim > 2:
            text_valid = text_valid.reshape(text_valid.shape[0], -1)
        if text_valid.shape[0] != bs:
            text_valid = text_valid[:1].expand(bs, -1)
        if text_valid.shape[1] != txtlen:  # contrato quebrado -> nao arrisca
            text_valid = torch.ones(bs, txtlen, dtype=torch.bool, device=device)
    image_valid = torch.ones(bs, tgtlen + reflen, dtype=torch.bool, device=device)
    attention_mask = torch.cat([text_valid, image_valid], dim=1)[:, None, None, :]
    if _DEBUG['calls'] == 1:
        _elo_sequence_snapshot(
            dump_prefix, 'layer00_initial', combined, txtlen, tgtlen,
            timestep_features=t, tvec=tvec, freqs=freqs,
            attention_mask=attention_mask,
        )
    if _DEBUG['on'] and _DEBUG['calls'] == 1:
        origem = 'conditioning' if text_attention_mask is not None else 'ASSUMIDA (tudo valido)'
        _dbg(f'mascara de atencao: shape={tuple(attention_mask.shape)} '
             f'chaves validas={int(attention_mask.sum())}/{seq_len} '
             f'(texto {int(text_valid.sum())}/{txtlen}) | origem={origem}')

    quant_ctx = (_TrainingBaseQuant(entries)
                 if (training_base_quant and _DEBUG['calls'] == 1) else _NullScope())
    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    transformer_options["total_blocks"] = len(m.blocks)
    transformer_options["block_type"] = "single"
    transformer_options["img_slice"] = [txtlen, combined.shape[1]]
    control_hits = 0
    with quant_ctx, _CountingMaskedLoraScope(entries, txtlen + tgtlen, seq_len, seq_len, strength):
        for block_index, block in enumerate(m.blocks):
            transformer_options["block_index"] = block_index
            replace = blocks_replace.get(("single_block", block_index))
            if replace is not None:
                def block_wrap(args, _block=block):
                    return {"img": _block(
                        args["img"],
                        args["vec"],
                        args["pe"],
                        args.get("attn_mask"),
                        transformer_options=args.get("transformer_options", {}),
                    )}

                replaced = replace({
                    "img": combined,
                    "vec": tvec,
                    "pe": freqs,
                    "attn_mask": attention_mask,
                    "transformer_options": transformer_options,
                }, {"original_block": block_wrap})
                combined = replaced["img"]
            else:
                combined = block(
                    combined, tvec, freqs, attention_mask,
                    transformer_options=transformer_options,
                )

            # Protocolo padrão dos ControlNets de DiT single-stream. Só o
            # target recebe o residual; texto e referência limpa permanecem
            # intactos. Isto torna o forward compatível com um ControlNet que
            # realmente tenha sido treinado para hidden-size/tokens do Krea2.
            if control is not None:
                control_out = control.get("output") if isinstance(control, dict) else None
                if control_out is not None and block_index < len(control_out):
                    add = control_out[block_index]
                    if add is not None:
                        add = add.to(device=combined.device, dtype=combined.dtype)
                        if add.shape[0] != combined.shape[0]:
                            add = comfy.utils.repeat_to_batch_size(add, combined.shape[0])
                        if add.shape[-1] != combined.shape[-1]:
                            raise RuntimeError(
                                "ControlNet incompativel com Krea2: hidden size "
                                f"{add.shape[-1]} != {combined.shape[-1]}."
                            )
                        token_count = min(tgtlen, add.shape[1])
                        combined[:, txtlen:txtlen + token_count] += add[:, :token_count]
                        control_hits += 1
            if _DEBUG['calls'] == 1:
                _elo_sequence_snapshot(
                    dump_prefix, f'block{block_index:02d}',
                    combined, txtlen, tgtlen,
                )
    if _DEBUG['on'] and _DEBUG['calls'] == 1:
        hits, misses = _DEBUG['lora_hits'], _DEBUG['lora_misses']
        veredito = ('OK' if hits and not misses else
                    'PARCIAL — alguma chamada nao casou seq_len' if hits else
                    'FALHA: o delta da LoRA NAO foi aplicado em nenhuma chamada')
        _dbg(f'LoRA routada aplicada em {hits} chamadas, ignorada em {misses} -> {veredito}')
        if control is not None:
            _dbg(f'ControlNet residual aplicado em {control_hits} blocos')

    final = m.last(combined, t)
    out = final[:, txtlen:txtlen + tgtlen, :]
    if _DEBUG['on'] and _DEBUG['calls'] == 1:
        import os as _os
        d = _os.environ.get('NODE_DUMP')
        if d and not _os.path.exists(d + '.x.npy'):
            import numpy as _np
            _np.save(d + '.x.npy', x.detach().float().cpu().numpy())
            _np.save(d + '.ref.npy', ref_latents[0].detach().float().cpu().numpy())
            _np.save(d + '.ctx.npy', context.detach().float().cpu().numpy())
            _dbg(f'DUMP node: x/ref/ctx salvos (t={float(timesteps.flatten()[0]):.4f})')
    out = rearrange(out, 'b (h w) (c ph pw) -> b c (h ph) (w pw)',
                    h=h_, w=w_, ph=patch, pw=patch, c=m.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, m.channels, H_orig, W_orig).movedim(1, 2)
    if _DEBUG['on'] and _DEBUG['calls'] == 1:
        import os as _os
        d = _os.environ.get('NODE_DUMP')
        if d and not _os.path.exists(d + '.v.npy'):
            import numpy as _np
            _np.save(d + '.v.npy', out.detach().float().cpu().numpy())
    return out


class CtxRushKrea2MultiRefApply:
    """All-in-one v2 para adapters krea2_multiref_grounded: contrato lido da
    metadata do adapter, N referências (offset cumulativo), grounding com cap
    de área e vision blocks no vocabulário do treino ('image 1:')."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL', {'tooltip': 'Saida do K2 Training Base. LoRAs adicionais '
                                               'compativeis com Krea2 podem passar antes por '
                                               'LoraLoaderModelOnly; nao carregue a turbo de novo. '
                                               'O adapter CtxRush e aplicado aqui em runtime bf16.'}),
                'clip': ('CLIP', {'tooltip': 'Krea 2 Qwen3-VL com torre visual.'}),
                'vae': ('VAE',),
                'image_1': ('IMAGE', {'tooltip': 'Referência 1 (slot que "image 1" endereça).'}),
                'positive_prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True,
                                               'tooltip': 'Caption no estilo do treino. Com N>1, endereça como "image 1"/"image 2" (sem <>).'}),
                'negative_prompt': ('STRING', {'default': '', 'multiline': True, 'dynamicPrompts': True,
                                               'tooltip': 'Vazio = o uncond EXATO do treino (caption_dropout grounded).'}),
                'lora_name': (folder_paths.get_filename_list('loras'),),
                'block_strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.05,
                                             'tooltip': 'Deltas ROUTADOS nos blocks (fidelidade à referência).'}),
                'fusion_strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.05,
                                              'tooltip': 'LoRA global do txtfusion (leitura semântica; rank 128 nos adapters novos).'}),
                'model_variant': (['raw', 'turbo'], {'default': 'raw',
                                  'tooltip': 'raw: 28 steps / CFG 5.5 / mu por resolução. turbo (base turbo ou turbo-lora fundida): 8 / 1.0 / 1.15.'}),
                'width': ('INT', {'default': 512, 'min': 64, 'max': 4096, 'step': 16}),
                'height': ('INT', {'default': 512, 'min': 64, 'max': 4096, 'step': 16}),
                'batch_size': ('INT', {'default': 1, 'min': 1, 'max': 16}),
            },
            'optional': {
                'image_2': ('IMAGE',),
                'image_3': ('IMAGE',),
                'image_4': ('IMAGE',),
                'vl_max_pixels': ('INT', {'default': 0, 'min': 0, 'max': 4096 * 4096, 'step': 28 * 28,
                                          'tooltip': 'Área máx. da cópia que o Qwen3-VL vê, POR referência. 0 = auto (metadata do adapter; 147456 = 384² no treino atual).'}),
                'vl_image_label': ('STRING', {'default': 'image',
                                              'tooltip': "Rótulo dos vision blocks. Treino multiref usa 'image' -> 'image 1: <vis>'. Adapters ostris antigos: 'Picture'."}),
                'reference_timestep': (['auto', 'zero', 'target'], {'default': 'auto',
                                       'tooltip': 'auto = metadata do adapter (multiref grava t=0).'}),
                'negative_grounding': (['grounded', 'plain'], {'default': 'grounded',
                                       'tooltip': 'grounded = uncond do treino (referências visíveis, mesmo layout). plain = só texto.'}),
                'reference_noise': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.05,
                                              'tooltip': 'Mistura ruído na(s) referência(s) pela fórmula do flow. 0 = fidelidade total.'}),
                'start_percent': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'end_percent': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'strength_curve': (['constant', 'fade_out', 'fade_in', 'fade_in_out'], {'default': 'constant'}),
                'curve_power': ('FLOAT', {'default': 1.0, 'min': 0.1, 'max': 4.0, 'step': 0.1}),
                'layers': ('STRING', {'default': 'all', 'tooltip': "'all' ou faixa 'inicio-fim' (0-27)."}),
                'layer_taper': (['flat', 'fade_deep', 'fade_shallow'], {'default': 'flat'}),
                'training_vl_contract': ('BOOLEAN', {'default': True,
                          'tooltip': 'Encoda texto/visao sob o contrato do TREINO. O ComfyUI atual '
                                     'usa MRoPE 3D + DeepStack no Qwen3-VL; o ComfyUI que treinou '
                                     'este adapter nao usava. Sem isto o txtfusion (o canal que LE a '
                                     'referencia) recebe uma distribuicao fora da de treino e a '
                                     'identidade do personagem nao e preservada — "parecido, nao o '
                                     'mesmo". Desligue so para adapters treinados com ComfyUI novo.'}),
                'training_base_quant': ('BOOLEAN', {'default': True,
                          'tooltip': 'VERIFICA (nao conserta) se a base carregada e a base de '
                                     'treino. O trainer roda com diffusion_model_dtype=float8 e '
                                     're-quantiza as 224 Linears de blocks.* no grid fp8 SEM '
                                     'escala; o adapter aprendeu sobre esses pesos. Carregar o '
                                     'fp8_scaled normal + LoraLoaderModelOnly da v relL2 0.581 '
                                     'contra o runner; a base assada por '
                                     'scripts/bake_training_base.py da 0.031. Deixe ligado: so '
                                     'imprime um aviso no 1o forward se a base estiver errada.'}),
                'debug': ('BOOLEAN', {'default': False,
                          'tooltip': 'Loga no console TUDO que define o resultado: hashes e dtypes dos '
                                     'modelos, contrato lido da metadata, prompt VL literal, geometria da '
                                     'sequencia, span da LoRA e — o mais importante — se o delta routado '
                                     'foi de fato aplicado. Ligue quando a saida "parecer" mas nao for.'}),
            },
        }

    RETURN_TYPES = ('MODEL', 'CONDITIONING', 'CONDITIONING', 'LATENT', 'INT', 'FLOAT')
    RETURN_NAMES = ('model', 'positive', 'negative', 'latent', 'steps', 'cfg')
    FUNCTION = 'apply'
    CATEGORY = 'CtxRush/Krea 2 Edit'
    DESCRIPTION = (
        'v2 multiref: contrato autoconfigurado pela metadata do adapter, até 4 '
        'referências com offset RoPE cumulativo, grounding por área (384² no '
        'treino atual) e vision blocks "image N:".'
    )

    def apply(self, model, clip, vae, image_1, positive_prompt, negative_prompt,
              lora_name, block_strength=1.0, fusion_strength=1.0,
              model_variant='raw', width=512, height=512, batch_size=1,
              image_2=None, image_3=None, image_4=None,
              vl_max_pixels=0, vl_image_label='image',
              reference_timestep='auto', negative_grounding='grounded',
              reference_noise=0.0, start_percent=0.0, end_percent=1.0,
              strength_curve='constant', curve_power=1.0,
              layers='all', layer_taper='flat', training_vl_contract=True,
              training_base_quant=True, debug=False):
        _DEBUG.update(on=bool(debug), calls=0, lora_hits=0, lora_misses=0, logged=False)
        lora_path = folder_paths.get_full_path('loras', lora_name)
        metadata = _read_safetensors_metadata(lora_path)
        contract = _resolve_contract(metadata, vl_max_pixels, vl_image_label, reference_timestep)

        if debug:
            import hashlib
            import os
            _dbg('=' * 62)
            try:
                with open(lora_path, 'rb') as f:
                    head = hashlib.sha256(f.read(8 << 20)).hexdigest()[:16]
                size_mb = os.path.getsize(lora_path) / 2**20
            except Exception as e:
                head, size_mb = f'<{e!r}>', -1
            _dbg(f'adapter: {os.path.basename(lora_path)} ({size_mb:.0f} MB, sha256[8MB]={head})')
            _dbg(f'metadata do adapter: ' + ', '.join(
                f'{k}={v}' for k, v in sorted(metadata.items())
                if k in ('control_family', 'model_type', 'position_mode', 'slot_axis',
                         'max_refs', 'reference_model_timestep', 'vl_image_max_pixels',
                         'vl_prompt_layout', 'caption_dropout', 'lora_targets',
                         'diffusion_pipe_commit', 'base_model_file')) or '<VAZIA>')
            if not metadata:
                _dbg('AVISO: adapter sem metadata — o contrato caiu nos defaults. '
                     'Confira se este safetensors saiu deste trainer.')
            try:
                dm = model.model.diffusion_model
                dts = {}
                for p in dm.parameters():
                    dts[str(p.dtype)] = dts.get(str(p.dtype), 0) + 1
                dominante = max(dts, key=dts.get)
                _dbg(f'base DiT: {type(dm).__name__} | dtypes={dts}')
                # O delta da LoRA e aplicado em runtime bf16 sobre a saida das
                # Linears. Se os pesos do DiT estiverem materializados em fp8, a
                # saida ja vem quantizada e o delta jovem do adapter e engolido
                # pelo passo de quantizacao — a referencia "quase" aparece.
                # bf16 e o contrato: o proprio trainer roda o base fp8-scaled
                # DEQUANTIZADO para bf16 em tempo de execucao.
                if 'bfloat16' in dominante:
                    _dbg(f'dtype do DiT: OK (bfloat16 dominante, {dts[dominante]} tensores) — '
                         f'o delta da LoRA e aplicado com fidelidade')
                elif 'float8' in dominante:
                    _dbg(f'dtype do DiT: ATENCAO — pesos dominantes em {dominante}. '
                         f'O adapter foi treinado com o base DEQUANTIZADO em bf16; com o DiT '
                         f'materializado em fp8 o delta routado perde resolucao numerica e a '
                         f'fidelidade da referencia cai. Carregue o UNETLoader com '
                         f'weight_dtype=default (nao fp8_e4m3fn/fp8_e5m2).')
                else:
                    _dbg(f'dtype do DiT: {dominante} dominante ({dts[dominante]} tensores) — '
                         f'fora do contrato de treino (bfloat16); resultado pode divergir.')
                try:
                    te_dts = {}
                    for p in clip.cond_stage_model.parameters():
                        te_dts[str(p.dtype)] = te_dts.get(str(p.dtype), 0) + 1
                    te_dom = max(te_dts, key=te_dts.get)
                    te_fp8 = any('float8' in k for k in te_dts)
                    _dbg(f'text encoder (Qwen3-VL): dtypes={te_dts}')
                    if te_fp8:
                        n8 = sum(v for k, v in te_dts.items() if 'float8' in k)
                        _dbg(f'dtype do text encoder: ATENCAO — {n8} tensores em fp8. '
                             f'O treino usou qwen3vl_4b_bf16. O grounding depende deste '
                             f'encoder para LER a referencia; em fp8 a percepcao fina '
                             f'(rosto/identidade) degrada antes de qualquer outra coisa. '
                             f'Se a saida sai "parecida mas nao igual", teste o bf16.')
                    else:
                        _dbg(f'dtype do text encoder: OK ({te_dom} dominante) — '
                             f'precisao igual ou melhor que a do treino (bf16)')
                except Exception as e:
                    _dbg(f'text encoder: <inspecao falhou: {e!r}>')
            except Exception as e:
                _dbg(f'base DiT: <inspecao falhou: {e!r}>')
            _dbg(f'contrato efetivo: {contract}')

        if contract['family'] and contract['family'] != 'krea2_multiref_grounded':
            print(f"[MultiRef] WARNING: adapter é '{contract['family']}', não "
                  f"'krea2_multiref_grounded' — confira se este é o node certo "
                  f"(o v1 CtxRushKrea2OminiGroundedApply cobre o omini_grounded).")

        images = [img for img in (image_1, image_2, image_3, image_4) if img is not None]
        if contract['max_refs'] and len(images) > contract['max_refs']:
            print(f"[MultiRef] WARNING: {len(images)} referências, mas o adapter "
                  f"treinou com max_refs={contract['max_refs']} — geometria OOD.")

        if width % 16 or height % 16:
            raise ValueError('width/height devem ser múltiplos de 16.')

        # Geometria do treino: TODA referência crop-fit ao tamanho do target
        # em PIXEL + encode nativo no VAE (nunca resize em latente).
        vl_images, ref_latents = [], []
        for img in images:
            img = _require_single_image(img)
            vl_images.append(_fit_vl(img, contract['vl_max_pixels']))
            latent = vae.encode(_crop_fit(img, width, height))
            ref_latents.append(latent)

        def encode(prompt, grounded=True):
            if not grounded:
                tokens = clip.tokenize(prompt, llama_template=KREA2_TEMPLATE)
                with _training_vl_contract(training_vl_contract):
                    return clip.encode_from_tokens_scheduled(tokens)
            text = _build_vl_prompt(len(vl_images), prompt,
                                    contract['prompt_style'], contract['vl_label'])
            if debug:
                shown = text.replace(VISION_BLOCK, '<VIS>')
                _dbg(f'prompt VL enviado ao Qwen3-VL ({len(text)} chars): '
                     f'{shown[:300]}{"…" if len(shown) > 300 else ""}')
            tokens = clip.tokenize(text, images=vl_images, llama_template=KREA2_TEMPLATE)
            with _training_vl_contract(training_vl_contract):
                cond = clip.encode_from_tokens_scheduled(tokens)
            if debug:
                try:
                    t0 = cond[0][0]
                    _dbg(f'contexto do encode: shape={tuple(t0.shape)} '
                         f'norm={t0.float().norm():.2f} std={t0.float().std():.5f} '
                         f'| contrato_treino={"ON" if training_vl_contract else "OFF"} '
                         f'(ALVO medido no runner: norm~16489 std~6.358; '
                         f'ComfyUI novo sem o fix: norm~17178 std~7.54)')
                except Exception:
                    pass
            return cond

        if debug:
            for i, (im, lat) in enumerate(zip(vl_images, ref_latents)):
                _dbg(_tensor_stats(f'ref{i+1} vista pelo Qwen3-VL (px)', im))
                _dbg(_tensor_stats(f'ref{i+1} latente VAE', lat))
            _dbg(f'geracao: {width}x{height} | variante={model_variant} | '
                 f'mu={_krea2_raw_mu(width, height) if model_variant == "raw" else KREA2_TURBO_MU:.6f}')
            _dbg('LEMBRETE: no KSampler use sampler=euler e scheduler=simple (ou '
                 'sgm_uniform). karras/exponential/normal usam outro grid de sigmas e '
                 'lavam a identidade da referencia.')

        positive = encode(positive_prompt)
        negative = encode(negative_prompt, grounded=(negative_grounding == 'grounded'))

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
            raise ValueError('Nenhum módulo de block LoRA casou com o diffusion model')
        block_entries = _ctxrush_layer_scales(block_entries, layers, layer_taper)
        if missing:
            print(f'[MultiRef] WARNING: {len(missing)} chaves LoRA sem match (ex. {missing[:3]})')
        print(f"[MultiRef] contrato: refs={len(images)} slot_axis={contract['slot_axis']} "
              f"t_ref={contract['reference_timestep']} vl={contract['vl_max_pixels']}px² "
              f"layout={contract['prompt_style']}/'{contract['vl_label']}' | "
              f"LoRA: {len(block_entries)} block linears (routed, {block_strength}) + "
              f"{len(fusion_entries)} txtfusion (global, {fusion_strength})")

        processed = []
        for latent in ref_latents:
            src = patched.model.process_latent_in(latent)
            if reference_noise > 0:
                epsilon = torch.randn_like(src)
                src = (1.0 - float(reference_noise)) * src + float(reference_noise) * epsilon
            processed.append(src)
        blocks_state = {'entries': block_entries, 'device': None}

        def wrapper(executor, x, timesteps, context, *args, **kwargs):
            transformer_options = kwargs.get('transformer_options')
            if transformer_options is None:
                transformer_options = next((a for a in args if isinstance(a, dict)), {})
            # O ComfyUI entrega a mascara de padding do texto como
            # 'attention_mask' (kwarg) ou no primeiro posicional apos context
            # (ver comfy/ldm/krea2/model.py:276). O trainer a usa em todos os
            # blocos; sem ela o padding do texto rouba peso do softmax.
            text_mask = kwargs.get('attention_mask')
            if text_mask is None:
                text_mask = next((a for a in args if torch.is_tensor(a) and a.ndim <= 2), None)
            sigma = (float(timesteps.flatten()[0]) if hasattr(timesteps, 'flatten')
                     else float(timesteps))
            multiplier = _ctxrush_schedule_mult(
                sigma, start_percent, end_percent, strength_curve, curve_power)
            return _krea2_multiref_forward(
                executor.class_obj, x, timesteps, context, processed, blocks_state,
                fusion_entries, block_strength * multiplier, transformer_options,
                fusion_strength=fusion_strength * multiplier,
                reference_timestep=contract['reference_timestep'],
                slot_axis=contract['slot_axis'],
                position_offset=contract['position_offset'],
                text_attention_mask=text_mask,
                training_base_quant=training_base_quant,
                control=kwargs.get('control'),
            )

        to = patched.model_options.setdefault('transformer_options', {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            'ctxrush_multiref', wrapper, to,
        )
        mu = _krea2_raw_mu(width, height) if model_variant == 'raw' else KREA2_TURBO_MU
        patched = _apply_krea2_sampling(patched, mu)
        steps, cfg = (28, 5.5) if model_variant == 'raw' else (8, 1.0)
        return (patched, positive, negative,
                _empty_krea_latent(width, height, batch_size), steps, cfg)


NODE_CLASS_MAPPINGS = {
    'CtxRushKrea2MultiRefApply': CtxRushKrea2MultiRefApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'CtxRushKrea2MultiRefApply': 'CtxRush - Krea 2 Multi-Ref Grounded (v2)',
}
