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


def _krea2_multiref_forward(m, x, timesteps, context, ref_latents, blocks_state,
                            fusion_entries, strength, transformer_options,
                            fusion_strength, reference_timestep, slot_axis,
                            position_offset):
    """Espelho fiel de Krea2MultiRefInitialLayer.forward (o que treinou):
    N spans concatenados após o target, offset RoPE cumulativo por slot,
    t=0 (ou target) per-token nos spans, txtfusion antes do txtmlp."""
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
    tgt = m.first(rearrange(x, 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=patch, pw=patch))

    # --- N spans de referência, offset cumulativo -------------------------
    ref_tokens_list, ref_pos_list = [], []
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

    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    grid = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    grid[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    grid[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    tgtpos = grid.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
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
    out = rearrange(out, 'b (h w) (c ph pw) -> b c (h ph) (w pw)',
                    h=h_, w=w_, ph=patch, pw=patch, c=m.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, m.channels, H_orig, W_orig).movedim(1, 2)
    return out


class CtxRushKrea2MultiRefApply:
    """All-in-one v2 para adapters krea2_multiref_grounded: contrato lido da
    metadata do adapter, N referências (offset cumulativo), grounding com cap
    de área e vision blocks no vocabulário do treino ('image 1:')."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL', {'tooltip': 'Krea 2 SEM Load LoRA (adapter aplicado em runtime bf16).'}),
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
              layers='all', layer_taper='flat'):
        lora_path = folder_paths.get_full_path('loras', lora_name)
        metadata = _read_safetensors_metadata(lora_path)
        contract = _resolve_contract(metadata, vl_max_pixels, vl_image_label, reference_timestep)

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
                return clip.encode_from_tokens_scheduled(tokens)
            text = _build_vl_prompt(len(vl_images), prompt,
                                    contract['prompt_style'], contract['vl_label'])
            tokens = clip.tokenize(text, images=vl_images, llama_template=KREA2_TEMPLATE)
            return clip.encode_from_tokens_scheduled(tokens)

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
