#!/usr/bin/env python
"""Assa um checkpoint Krea2 com a MESMA base numerica que o trainer usou.

Motivo (medido, nao hipotese)
-----------------------------
O adapter krea2_multiref_grounded foi treinado com
``diffusion_model_dtype = 'float8'`` (train.toml:34). O trainer, em
``models/base.py:536`` (``p.dequantize()``) e ``:547``
(``p.data = p.data.to(float8_e4m3fn)``), DESCARTA o ``weight_scale`` do
checkpoint fp8_scaled e re-quantiza os pesos no grid fp8 CRU. Depois
``tools/infer_reference_adapter.py:apply_turbo_lora`` soma o delta da turbo e
RE-QUANTIZA de novo (``novo.to(base.dtype)``, base.dtype == float8_e4m3fn).

Peso efetivo do trainer/runner nas 224 Linears de blocks.*:

    W_run = fp8( fp8( bf16(qdata * scale) ).float() + up @ down )

O ComfyUI carrega o mesmo arquivo em fp8_scaled e computa com
``qdata * scale`` (exato). Diferenca medida contra o dump real dos pesos do
runner (/workspace/outputs/node_test/full/weights_runner.npz):

    blocks.0.attn.wq    relL2 4.07%   (6.6% dos pesos viram zero no runner)
    blocks.13.mlp.down  relL2 2.55%
    blocks.27.attn.wo   relL2 3.57%

Este script reproduz W_run BIT A BIT (verificado: np.array_equal == True) e
grava num .safetensors carregavel pelo UNETLoader, mantendo 1 byte/peso
(qdata fp8 + weight_scale = 1.0, que e exato porque os valores ja estao no
grid fp8).

Tambem funde a turbo LoRA do jeito do runner: SO ``lora_down``/``lora_up``
(escala 1.0, o arquivo nao tem ``.alpha``), IGNORANDO as 7 chaves ``.diff_b``
que o ``LoraLoaderModelOnly`` do ComfyUI aplica e o runner nao. Depois de
assar, o LoraLoaderModelOnly deve sair do grafo.

Uso
---
    python scripts/bake_training_base.py \
        --base   /workspace/models/krea2/diffusion_models/krea2_raw_fp8_scaled.safetensors \
        --turbo  /workspace/models/krea2/loras/krea2_turbo_lora_rank_64_bf16.safetensors \
        --out    /workspace/models/krea2/diffusion_models/krea2_trainbase_turbo_fp8.safetensors
"""

import argparse
import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# models/krea2.py:22 — modulos que o trainer NAO quantiza.
KEEP_IN_HIGH_PRECISION = ('first', 'last', 'tmlp', 'tproj', 'txtfusion', 'txtmlp')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--turbo', default=None, help='LoRA turbo a fundir (opcional)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--plain-fp8', action='store_true',
                    help='grava as 224 Linears como float8_e4m3fn CRU (sem weight_scale, sem '
                         'quant metadata), reproduzindo tambem o CAMINHO DE COMPUTE do trainer '
                         '(comfy.ops Linear + comfy_cast_weights: fp8 -> bf16 -> GEMM bf16) em '
                         'vez do GEMM quantizado do comfy_kitchen')
    args = ap.parse_args()

    fb = safe_open(args.base, 'pt')
    meta = dict(fb.metadata() or {})
    qmeta = json.loads(meta.get('_quantization_metadata', '{}')) or {}
    qlayers = qmeta.get('layers', {})

    deltas = {}
    if args.turbo:
        ft = safe_open(args.turbo, 'pt')
        pares = {}
        for k in ft.keys():
            if k.endswith('.lora_down.weight'):
                pares.setdefault(k[:-len('.lora_down.weight')], {})['down'] = k
            elif k.endswith('.lora_up.weight'):
                pares.setdefault(k[:-len('.lora_up.weight')], {})['up'] = k
            elif k.endswith('.alpha'):
                pares.setdefault(k[:-len('.alpha')], {})['alpha'] = k
            # .diff / .diff_b: ignorados de proposito (o runner ignora)
        for pref, parts in pares.items():
            if 'down' not in parts or 'up' not in parts:
                continue
            down = ft.get_tensor(parts['down']).float()
            up = ft.get_tensor(parts['up']).float()
            escala = 1.0
            if 'alpha' in parts:
                escala = float(ft.get_tensor(parts['alpha'])) / down.shape[0]
            name = pref.removeprefix('diffusion_model.')
            deltas[name] = (up @ down) * escala

    keys = list(fb.keys())
    scaled = {k[:-len('.weight_scale')] for k in keys if k.endswith('.weight_scale')}

    out = {}
    n_fp8 = n_bf16 = n_delta = n_overflow = 0
    for k in keys:
        if k.endswith('.weight_scale'):
            continue                                    # reemitido abaixo quando preciso
        t = fb.get_tensor(k)

        if not k.endswith('.weight'):
            out[k] = t                                  # bias, norms: intocados
            continue

        name = k[:-len('.weight')]
        keep = any(kw in name for kw in KEEP_IN_HIGH_PRECISION)
        delta = deltas.get(name)
        if delta is not None:
            n_delta += 1

        if name in scaled:
            s = fb.get_tensor(name + '.weight_scale')
            w = (t.float() * s.float()).to(torch.bfloat16)   # base.py:536
        else:
            w = t                                            # ja bf16

        if keep or w.ndim != 2:
            # keep_in_high_precision: fica em bf16, turbo somada em fp32
            w = w.float()
            if delta is not None:
                w = w + delta
            out[k] = w.to(torch.bfloat16)
            qlayers.pop(name, None)
            n_bf16 += 1
        else:
            w8 = w.to(torch.float8_e4m3fn)                   # base.py:547
            if delta is not None:
                w8 = (w8.float() + delta).to(torch.float8_e4m3fn)   # apply_turbo_lora
            bad = int(torch.isnan(w8.float()).sum())
            if bad:
                n_overflow += bad
            out[k] = w8
            if args.plain_fp8:
                qlayers.pop(name, None)
            else:
                out[name + '.weight_scale'] = torch.tensor(1.0, dtype=torch.float32)
                # full_precision_matrix_mult: o trainer troca o modulo por
                # comfy.ops Linear com comfy_cast_weights -> fp8 vira bf16 e o
                # GEMM e bf16. Sem esta flag o comfy_kitchen usa o GEMM fp8
                # nativo nas 140 Linears que nao a tinham, e o v final fica
                # relL2 0.178 em vez de 0.066 (medido).
                qlayers[name] = {'format': 'float8_e4m3fn',
                                 'full_precision_matrix_mult': True}
            n_fp8 += 1

    qmeta['layers'] = qlayers
    meta['_quantization_metadata'] = json.dumps(qmeta)
    meta['ctxrush_trainbase'] = 'fp8_e4m3fn_unscaled(blocks) + turbo fundida sem diff_b'

    print(f'fp8 (grid de treino): {n_fp8} | bf16 (keep_in_high_precision): {n_bf16} '
          f'| deltas turbo fundidos: {n_delta} | NaN por overflow fp8: {n_overflow}')
    print(f'gravando {args.out} ...')
    save_file(out, args.out, metadata=meta)
    print('ok')


if __name__ == '__main__':
    main()
