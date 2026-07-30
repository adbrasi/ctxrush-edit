"""CtxRush — K2 Training Base (runtime, sem checkpoint extra).

PROBLEMA (medido pelo enxame de 15 agentes + Codex, 2026-07-30):
o adapter foi treinado sobre uma base numericamente diferente da que o
ComfyUI executa, e essa é a causa dominante do "adapter fraco no ComfyUI".

    train.toml:34  diffusion_model_dtype = 'float8'
    models/base.py:536   p.dequantize()                    <- desfaz o fp8-SCALED
    models/base.py:547   p.data.to(float8_e4m3fn)          <- re-quantiza SEM escala

O checkpoint é fp8-**scaled** (qdata fp8 + weight_scale por tensor). O trainer
DESCARTA o weight_scale e joga os pesos no grid fp8 cru. Isso degrada as 224
Linears de `blocks.*` em 2,5%-6,7%, e entre 6,6% e 26,8% dos pesos viram ZERO
por denormal underflow. O ComfyUI roda o mesmo arquivo com a escala preservada
(exato). Efeito medido no primeiro forward: v relL2 0,58 / cos 0,913.

Prova causal: virando SÓ essa flag dentro do runner, o runner vira o node.

POR QUE A PRIMEIRA TENTATIVA (`_TrainingBaseQuant`) FOI NO-OP:
`w.data.to(fp8).to(dtype)` sobre um `QuantizedTensor` cai num fast-path
metadata-only (`comfy_kitchen/tensor/base.py:430`): o `_qdata` nunca é
recomputado, só o rótulo `orig_dtype` muda. `torch.equal(depois, antes)=True`.
**Faltava o `dequantize()` antes**, exatamente como o trainer faz.

SEGUNDA CAUSA, também tratada aqui: a turbo LoRA tem 535 chaves = 264 pares
lora_down/up + **7 chaves `.diff_b`** (bias deltas). O runner
(`infer_reference_adapter.py:481-489`) **ignora as 7**; o `LoraLoaderModelOnly`
do ComfyUI **aplica**. Isso muda o `timestep_features` (e portanto o `tvec`
adaLN dos 28 blocos) em 1,8%. Aqui a turbo é fundida do jeito do runner.

Ordem exata reproduzida:  W = fp8( fp8(W_dequantizado) + up@down )

Este node NÃO cria arquivo nenhum: opera nos pesos já carregados, tensor a
tensor (sem pico de memória), e é idempotente.
"""

import torch
import torch.nn as nn

import comfy.model_management
import comfy.utils
import folder_paths

# Igual a models/krea2.py:22 — o trainer degrada SÓ os blocks; estes ficam
# em alta precisão e não podem ser tocados.
KEEP_IN_HIGH_PRECISION = ('first', 'last', 'tmlp', 'tproj', 'txtfusion', 'txtmlp')


def _is_quantized(t):
    return t.__class__.__name__ == 'QuantizedTensor'


def _to_training_grid(t):
    """W -> fp8_e4m3fn puro (sem escala) -> de volta ao dtype de compute.

    Voltar para bf16 é deliberado: o trainer marca `comfy_cast_weights=True`
    (base.py:553), ou seja o matmul acontece em bf16 sobre valores que estão
    no grid fp8. Deixar o tensor em fp8 faria o comfy_kitchen usar GEMM fp8
    nativo — que o enxame mediu como PIOR (relL2 0.178 contra 0.031).
    """
    if _is_quantized(t):
        t = t.dequantize()          # <- o passo que faltava na 1a tentativa
    out_dtype = t.dtype if t.dtype in (torch.bfloat16, torch.float16, torch.float32) else torch.bfloat16
    return t.to(torch.float8_e4m3fn).to(out_dtype)


def _force_bf16_matmul(module):
    """Desliga o GEMM FP8 nativo que continua associado ao layout do arquivo.

    ``MixedPrecisionOps.Linear.forward`` não consulta ``comfy_cast_weights``
    para decidir se quantiza a ativação. A chave efetiva é
    ``_full_precision_mm`` (ops.py:1277-1282). O runner dequantiza o peso FP8
    cru para o dtype da entrada e chama ``F.linear``; esta flag reproduz esse
    caminho.
    """
    if hasattr(module, '_full_precision_mm'):
        module._full_precision_mm = True
    if hasattr(module, '_full_precision_mm_config'):
        module._full_precision_mm_config = True


def _iter_target_linears(root, prefix=''):
    """Só as Linears 2-D dentro de `blocks.*`, fora da lista de alta precisão."""
    for name, module in root.named_children():
        full = f'{prefix}{name}' if prefix else name
        w = getattr(module, 'weight', None)
        if w is not None and getattr(w, 'ndim', 0) == 2:
            if full.startswith('blocks.') and not any(k in full for k in KEEP_IN_HIGH_PRECISION):
                yield full, module
        yield from _iter_target_linears(module, f'{full}.')


class K2TrainingBase:
    """Coloca a base no contrato numérico do treino, em runtime."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL', {'tooltip': 'Krea 2 recem carregado pelo UNETLoader '
                                               '(weight_dtype=default), SEM LoraLoader.'}),
                'fp8_training_grid': ('BOOLEAN', {'default': True,
                    'tooltip': 'Re-quantiza as 224 Linears dos blocks para o grid fp8 CRU do '
                               'treino (descartando o weight_scale). Esta e a causa dominante '
                               'do adapter parecer fraco: sem isso, v relL2 0.58 contra o runner.'}),
            },
            'optional': {
                'turbo_lora': (['nenhuma'] + folder_paths.get_filename_list('loras'), {'default': 'nenhuma',
                    'tooltip': 'Funde a turbo COMO O RUNNER: so lora_down/up, IGNORANDO as 7 chaves '
                               '.diff_b que o LoraLoaderModelOnly aplicaria. Use aqui em vez do '
                               'LoraLoaderModelOnly — os dois juntos aplicariam a turbo duas vezes.'}),
                'turbo_strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 2.0, 'step': 0.05}),
                'verbose': ('BOOLEAN', {'default': True}),
            },
        }

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply'
    CATEGORY = 'CtxRush/K2'
    DESCRIPTION = ('Reproduz em memoria a base numerica do treino (fp8 sem escala + turbo fundida '
                   'como o runner). Sem checkpoint extra, sem arquivo novo.')

    def apply(self, model, fp8_training_grid, turbo_lora='nenhuma',
              turbo_strength=1.0, verbose=True):
        out = model.clone()
        dit = out.get_model_object('diffusion_model')

        def log(msg):
            if verbose:
                print(f'[K2TrainingBase] {msg}', flush=True)

        contract = (bool(fp8_training_grid), str(turbo_lora), float(turbo_strength))
        previous_contract = getattr(dit, '_ctxrush_training_base_contract', None)
        if previous_contract is not None:
            if previous_contract == contract:
                log('contrato ja aplicado nesta instancia; reutilizando sem quantizar/fundir novamente')
                return (out,)
            raise RuntimeError(
                'K2 Training Base ja foi aplicado nesta instancia com outros parametros. '
                'Recarregue o UNET antes de mudar fp8_training_grid, turbo_lora ou turbo_strength.'
            )

        targets = dict(_iter_target_linears(dit))
        log(f'{len(targets)} Linears alvo em blocks.* (alta precisao preservada: '
            f'{", ".join(KEEP_IN_HIGH_PRECISION)})')

        # --- 1) base no grid fp8 do treino --------------------------------
        n_quant = 0
        if fp8_training_grid:
            for full, module in targets.items():
                w = module.weight
                new = _to_training_grid(w.data if isinstance(w, nn.Parameter) else w)
                module.register_parameter('weight', nn.Parameter(new, requires_grad=False))
                # matmul em alta precisao sobre valores no grid fp8 (base.py:553)
                module.comfy_cast_weights = True
                _force_bf16_matmul(module)
                n_quant += 1
            log(f'base re-quantizada no grid fp8 do treino: {n_quant} Linears; '
                'GEMM BF16 forcado')

        # --- 2) turbo fundida do jeito do runner --------------------------
        n_fused, n_skipped = 0, 0
        if turbo_lora != 'nenhuma':
            path = folder_paths.get_full_path('loras', turbo_lora)
            sd = comfy.utils.load_torch_file(path, safe_load=True)
            pairs, diff_b = {}, 0
            for k, v in sd.items():
                if k.endswith('.diff_b'):
                    diff_b += 1
                    continue                       # o runner IGNORA estas
                for suffix, slot in (('.lora_down.weight', 'down'), ('.lora_up.weight', 'up')):
                    if k.endswith(suffix):
                        base = k[: -len(suffix)]
                        base = base.replace('diffusion_model.', '', 1)
                        pairs.setdefault(base, {})[slot] = v
            named = dict(dit.named_modules())
            for base, du in pairs.items():
                if 'down' not in du or 'up' not in du:
                    continue
                module = named.get(base)
                if module is None or not hasattr(module, 'weight'):
                    n_skipped += 1
                    continue
                w = module.weight
                cur = w.data if isinstance(w, nn.Parameter) else w
                dtype = cur.dtype if not _is_quantized(cur) else torch.bfloat16
                base_w = cur.dequantize() if _is_quantized(cur) else cur
                up = du['up'].to(base_w.device, torch.float32)
                down = du['down'].to(base_w.device, torch.float32)
                delta = (up @ down).reshape(base_w.shape).to(torch.float32)
                merged = base_w.to(torch.float32) + float(turbo_strength) * delta
                if fp8_training_grid and base in targets:
                    # re-quantiza DEPOIS da soma: W = fp8(fp8(W_deq) + delta)
                    merged = merged.to(torch.float8_e4m3fn).to(dtype)
                else:
                    merged = merged.to(dtype)
                module.register_parameter('weight', nn.Parameter(merged, requires_grad=False))
                module.comfy_cast_weights = True
                if base in targets:
                    _force_bf16_matmul(module)
                n_fused += 1
            del sd
            log(f'turbo fundida em {n_fused} Linears (como o runner) | '
                f'{diff_b} chaves .diff_b IGNORADAS (o runner tambem ignora) | '
                f'{n_skipped} sem match')

        dit._ctxrush_training_base_contract = contract
        log('pronto — NAO use LoraLoaderModelOnly para a turbo junto com este node')
        return (out,)


class K2LoadReferencePIL:
    """Carrega a imagem de referência com PIL, como o runner.

    O `LoadImage` do ComfyUI decodifica JPEG com PyAV/ffmpeg; o runner usa
    PIL/libjpeg. Medido no mesmo arquivo: 65,7% dos pixels diferem em >=1/255
    (max 12/255), o latente da referência diverge relL2 6,6% e o v do primeiro
    forward 4,1%. Não é o principal, mas é real e de graça corrigir.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import os
        d = folder_paths.get_input_directory()
        files = sorted(f for f in os.listdir(d)
                       if os.path.isfile(os.path.join(d, f))) if os.path.isdir(d) else []
        return {
            'required': {
                'image': (files, {'image_upload': True}),
            },
            'optional': {
                'path_override': ('STRING', {'default': '',
                    'tooltip': 'Caminho absoluto, se a imagem nao estiver em input/.'}),
            },
        }

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'load'
    CATEGORY = 'CtxRush/K2'
    DESCRIPTION = 'Decodifica com PIL/libjpeg, igual ao runner (o LoadImage usa PyAV/ffmpeg).'

    def load(self, image, path_override=''):
        import numpy as np
        from PIL import Image, ImageOps
        path = path_override.strip() or folder_paths.get_annotated_filepath(image)
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        if img.mode == 'RGBA' or ('transparency' in img.info and img.mode != 'RGB'):
            rgba = img.convert('RGBA')
            canvas = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            canvas.alpha_composite(rgba)
            img = canvas.convert('RGB')
        else:
            img = img.convert('RGB')
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return (torch.from_numpy(arr).unsqueeze(0),)


class _RunnerCudaNoise:
    """Objeto NOISE que reproduz o RNG CUDA usado pelo runner canônico."""

    def __init__(self, seed):
        self.seed = int(seed)

    def generate_noise(self, input_latent):
        latent = input_latent['samples']
        if getattr(latent, 'is_nested', False):
            raise RuntimeError(
                'K2 Runner Noise nao suporta latente nested; use batch Krea2 regular.'
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                'K2 Runner Noise exige CUDA: o runner canonico gera x_T com '
                "torch.Generator(device='cuda')."
            )
        device = comfy.model_management.get_torch_device()
        if device.type != 'cuda':
            device = torch.device('cuda')
        generator = torch.Generator(device=device).manual_seed(self.seed)
        # O runner usa torch.randn(..., device='cuda') sem dtype explicito:
        # portanto o x_T nasce em float32, independentemente do dtype do vazio.
        return torch.randn(
            latent.shape,
            dtype=torch.float32,
            layout=latent.layout,
            generator=generator,
            device=device,
        )


class K2RunnerNoise:
    """Gera exatamente o mesmo x_T que o runner para uma dada seed.

    O KSampler/RandomNoise padrão chama ``torch.manual_seed`` e gera em CPU.
    O runner chama ``torch.Generator(device='cuda').manual_seed`` e gera na
    GPU. CPU e CUDA usam streams diferentes: com a mesma seed, medimos
    relL2=1.414 e cosseno≈0 entre os dois ruídos. Para comparação realmente
    pareada, conecte este NOISE ao ``SamplerCustomAdvanced``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'noise_seed': ('INT', {
                    'default': 0,
                    'min': 0,
                    'max': 0xffffffffffffffff,
                    'control_after_generate': True,
                    'tooltip': 'Seed CUDA do runner. Use com SamplerCustomAdvanced.',
                }),
            },
        }

    RETURN_TYPES = ('NOISE',)
    FUNCTION = 'get_noise'
    CATEGORY = 'CtxRush/K2'
    DESCRIPTION = ('Ruido CUDA bit a bit compativel com o runner. '
                   'Use com SamplerCustomAdvanced, nao com KSampler.')

    def get_noise(self, noise_seed):
        return (_RunnerCudaNoise(noise_seed),)


NODE_CLASS_MAPPINGS = {
    'K2TrainingBase': K2TrainingBase,
    'K2LoadReferencePIL': K2LoadReferencePIL,
    'K2RunnerNoise': K2RunnerNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'K2TrainingBase': 'K2 Training Base (fp8 do treino + turbo)',
    'K2LoadReferencePIL': 'K2 Load Reference (PIL, como o runner)',
    'K2RunnerNoise': 'K2 Runner Noise (CUDA, seed pareada)',
}
