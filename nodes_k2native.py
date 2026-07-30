"""CtxRush K2 Native — pack v3.

Nodes que reproduzem o runner (tools/infer_reference_adapter.py) DENTRO do
ComfyUI, sem passar por nenhuma peça do ComfyUI que possa divergir do
contrato de treino. Sem subprocesso: tudo no mesmo interpretador.

O que NÃO é usado, e por quê (cada item é uma divergência medida ou provável):

| peça evitada            | o que ela fazia de errado                              |
|-------------------------|--------------------------------------------------------|
| dynamic VRAM loading    | materializa pesos sob demanda -> quebra patch e dtype   |
| UNETLoader padrão       | mantém bf16; o treino roda a base re-quantizada em fp8  |
| LoraLoaderModelOnly     | turbo como 270 patches lazy; o runner FUNDE nos pesos   |
| Qwen3VL.forward atual   | MRoPE 3D + DeepStack -> contexto relL2 1.36 fora do treino |
| KSampler                | prepare_noise, process_latent_in/out, ModelSamplingFlux,|
|                         | CFGGuider e escolha de scheduler que lava a identidade  |

O que É usado do ComfyUI: os loaders de arquivo, o VAE, os tipos IMAGE/LATENT
e o grafo. Ou seja, a casca.

Fluxo:
    K2 Load Models ─┬─► K2 Reference Encode ─┐
                    ├─► K2 Conditioning ─────┼─► K2 Sampler ─► LATENT ─► K2 Decode
                    │   K2 Adapter ──────────┘
"""

import math

import torch

import comfy.model_management
import comfy.ops
import comfy.sd
import comfy.utils
import folder_paths

from .nodes import (
    KREA2_TURBO_MU,
    _crop_fit,
    _ctxrush_layer_scales,
    _fit_vl,
    _load_omini_lora,
    _require_single_image,
)
from .nodes_multiref import (
    _DEBUG,
    _build_vl_prompt,
    _dbg,
    _krea2_multiref_forward,
    _read_safetensors_metadata,
    _resolve_contract,
    _training_vl_contract,
)


# --------------------------------------------------------------------------
# schedule oficial — cópia fiel de tools/krea2_sampling.py (função pura).
# Não usamos ModelSamplingFlux + scheduler do ComfyUI: o grid de sigmas é
# parte do contrato validado e a escolha errada de scheduler (karras,
# exponential, normal) tira ~15 dos 28 steps do regime de alto ruído, que é
# onde identidade e layout são decididos.
# --------------------------------------------------------------------------
def build_krea2_timesteps(sequence_length, steps, *, min_resolution=256,
                          max_resolution=1280, spatial_compression=8,
                          patch_size=2, y1=0.5, y2=1.15, mu=None):
    token_stride = spatial_compression * patch_size
    x1 = (min_resolution // token_stride) ** 2
    x2 = (max_resolution // token_stride) ** 2
    resolved_mu = float(mu) if mu is not None else (
        (y2 - y1) / (x2 - x1) * sequence_length + (y1 - ((y2 - y1) / (x2 - x1)) * x1)
    )
    shift = math.exp(resolved_mu)
    ts = []
    for i in range(steps + 1):
        base = 1.0 - i / steps
        ts.append(shift * base / (1.0 + (shift - 1.0) * base))
    ts[0], ts[-1] = 1.0, 0.0
    return ts, resolved_mu


def _training_dequantize(module, dtype, keep_high=('txtfusion', 'txtmlp', 'tmlp',
                                                   'tproj', 'first', 'last', 'pe_embedder'),
                         prefix=''):
    """Espelha models/base.py:529-548 do trainer.

    O treino roda com diffusion_model_dtype='float8': dequantiza o checkpoint
    fp8-SCALED e RE-QUANTIZA as matrizes 2-D para float8_e4m3fn puro, SEM
    escala (degrada ~2.8%). O adapter aprendeu sobre ESSA base. Carregar em
    bf16 deixa o efeito do adapter ~13% mais fraco (medido).

    Só toca em parâmetros 2-D e fora da lista de alta precisão, igual ao
    trainer (que preserva ndim==1 e keep_in_high_precision).
    """
    n = 0
    for mod_name, child in module.named_children():
        full = f'{prefix}{mod_name}'
        for p_name, p in child.named_parameters(recurse=False):
            name = f'{full}.{p_name}'
            if p.ndim == 1 or any(k in name for k in keep_high):
                continue
            p.data = p.data.to(dtype)
            n += 1
        n += _training_dequantize(child, dtype, keep_high, prefix=f'{full}.')
    return n


class K2LoadModels:
    """Carrega DiT, text encoder e VAE sob o contrato de treino."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'unet_name': (folder_paths.get_filename_list('diffusion_models'),),
                'clip_name': (folder_paths.get_filename_list('text_encoders'),),
                'vae_name': (folder_paths.get_filename_list('vae'),),
                'base_dtype': (['float8_como_no_treino', 'bfloat16'], {'default': 'float8_como_no_treino',
                               'tooltip': 'O treino roda a base re-quantizada em fp8 puro. bf16 deixa '
                                          'o adapter ~13% mais fraco.'}),
            },
            'optional': {
                'turbo_lora': (['nenhuma'] + folder_paths.get_filename_list('loras'), {'default': 'nenhuma',
                               'tooltip': 'FUNDIDA nos pesos antes do adapter, como o runner faz — '
                                          'nao aplicada como patch lazy.'}),
                'debug': ('BOOLEAN', {'default': False}),
            },
        }

    RETURN_TYPES = ('K2_MODELS',)
    FUNCTION = 'load'
    CATEGORY = 'CtxRush/K2 Native'
    DESCRIPTION = 'Carrega os modelos sem dynamic loading, com a base no dtype do treino e a turbo fundida.'

    def load(self, unet_name, clip_name, vae_name, base_dtype,
             turbo_lora='nenhuma', debug=False):
        _DEBUG['on'] = bool(debug)
        unet_path = folder_paths.get_full_path('diffusion_models', unet_name)
        # disable_dynamic=True: o trainer carrega assim (models/base.py:566).
        # Sem isso o ComfyUI materializa pesos sob demanda e qualquer mudança
        # que a gente faça nos parâmetros não alcança o tensor que executa.
        dit = comfy.sd.load_diffusion_model(
            unet_path, model_options={'dtype': torch.bfloat16}, disable_dynamic=True)

        if turbo_lora != 'nenhuma':
            lora_path = folder_paths.get_full_path('loras', turbo_lora)
            sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
            dit, _ = comfy.sd.load_lora_for_models(dit, None, sd, 1.0, 0.0)
            dit.patch_model()          # materializa: vira peso, não patch pendente
            _dbg(f'turbo LoRA FUNDIDA nos pesos ({len(sd)} chaves)')
            del sd

        quantized = 0
        if base_dtype == 'float8_como_no_treino':
            target = dit.model.diffusion_model
            quantized = _training_dequantize(target, torch.float8_e4m3fn)
            _dbg(f'base re-quantizada para fp8 (contrato de treino) em {quantized} tensores 2-D')

        clip = comfy.sd.load_clip(
            ckpt_paths=[folder_paths.get_full_path('text_encoders', clip_name)],
            embedding_directory=folder_paths.get_folder_paths('embeddings'),
            clip_type=comfy.sd.CLIPType.KREA2)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
            folder_paths.get_full_path('vae', vae_name)))

        if debug:
            dts = {}
            for p in dit.model.diffusion_model.parameters():
                dts[str(p.dtype)] = dts.get(str(p.dtype), 0) + 1
            _dbg(f'K2 Load: dtypes do DiT={dts} | turbo={"fundida" if turbo_lora != "nenhuma" else "nao"} '
                 f'| dynamic loading=DESLIGADO')

        return ({'dit': dit, 'clip': clip, 'vae': vae,
                 'turbo': turbo_lora != 'nenhuma',
                 'base_dtype': base_dtype, 'quantized': quantized},)


class K2Adapter:
    """Carrega o adapter e resolve o contrato a partir da metadata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'lora_name': (folder_paths.get_filename_list('loras'),),
                'block_strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.05}),
                'fusion_strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.05}),
            },
            'optional': {
                'layers': ('STRING', {'default': 'all'}),
                'layer_taper': (['flat', 'fade_deep', 'fade_shallow'], {'default': 'flat'}),
                'vl_max_pixels': ('INT', {'default': 0, 'min': 0, 'max': 4096 * 4096, 'step': 784}),
                'vl_image_label': ('STRING', {'default': 'image'}),
                'reference_timestep': (['auto', 'zero', 'target'], {'default': 'auto'}),
            },
        }

    RETURN_TYPES = ('K2_ADAPTER',)
    FUNCTION = 'load'
    CATEGORY = 'CtxRush/K2 Native'

    def load(self, lora_name, block_strength, fusion_strength, layers='all',
             layer_taper='flat', vl_max_pixels=0, vl_image_label='image',
             reference_timestep='auto'):
        path = folder_paths.get_full_path('loras', lora_name)
        metadata = _read_safetensors_metadata(path)
        contract = _resolve_contract(metadata, vl_max_pixels, vl_image_label, reference_timestep)
        pairs = _load_omini_lora(path)
        _dbg(f'K2 Adapter: {len(pairs)} pares LoRA | contrato={contract}')
        return ({'pairs': pairs, 'contract': contract, 'metadata': metadata,
                 'block_strength': block_strength, 'fusion_strength': fusion_strength,
                 'layers': layers, 'layer_taper': layer_taper},)


class K2ReferenceEncode:
    """Referências -> latentes VAE + cópias para o Qwen3-VL, na geometria do treino."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'models': ('K2_MODELS',),
                'adapter': ('K2_ADAPTER',),
                'image_1': ('IMAGE',),
                'width': ('INT', {'default': 512, 'min': 64, 'max': 4096, 'step': 16}),
                'height': ('INT', {'default': 512, 'min': 64, 'max': 4096, 'step': 16}),
            },
            'optional': {
                'image_2': ('IMAGE',), 'image_3': ('IMAGE',), 'image_4': ('IMAGE',),
            },
        }

    RETURN_TYPES = ('K2_REFS',)
    FUNCTION = 'encode'
    CATEGORY = 'CtxRush/K2 Native'

    def encode(self, models, adapter, image_1, width, height,
               image_2=None, image_3=None, image_4=None):
        if width % 16 or height % 16:
            raise ValueError('width/height devem ser múltiplos de 16')
        images = [i for i in (image_1, image_2, image_3, image_4) if i is not None]
        vae = models['vae']
        vl_images, latents = [], []
        for img in images:
            img = _require_single_image(img)
            vl_images.append(_fit_vl(img, adapter['contract']['vl_max_pixels']))
            # crop-fit em PIXEL + encode nativo: a geometria do treino.
            latents.append(vae.encode(_crop_fit(img, width, height)))
        _dbg(f'K2 Refs: {len(images)} referência(s) em {width}x{height}')
        return ({'vl_images': vl_images, 'latents': latents,
                 'width': width, 'height': height},)


class K2Conditioning:
    """Encode de texto+visão sob o contrato de treino do Qwen3-VL."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'models': ('K2_MODELS',),
                'adapter': ('K2_ADAPTER',),
                'refs': ('K2_REFS',),
                'positive_prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
                'negative_prompt': ('STRING', {'default': '', 'multiline': True}),
            },
            'optional': {
                'training_vl_contract': ('BOOLEAN', {'default': True,
                    'tooltip': 'Desfaz MRoPE 3D + DeepStack do Qwen3-VL do ComfyUI atual durante o '
                               'encode. O adapter treinou sem eles (contexto divergia relL2 1.36).'}),
                'negative_grounding': (['grounded', 'plain'], {'default': 'grounded'}),
            },
        }

    RETURN_TYPES = ('K2_COND',)
    FUNCTION = 'encode'
    CATEGORY = 'CtxRush/K2 Native'

    def encode(self, models, adapter, refs, positive_prompt, negative_prompt,
               training_vl_contract=True, negative_grounding='grounded'):
        clip = models['clip']
        contract = adapter['contract']

        def run(prompt, grounded):
            if not grounded:
                tokens = clip.tokenize(prompt)
            else:
                text = _build_vl_prompt(len(refs['vl_images']), prompt,
                                        contract['prompt_style'], contract['vl_label'])
                tokens = clip.tokenize(text, images=refs['vl_images'])
            with _training_vl_contract(training_vl_contract):
                out = clip.encode_from_tokens(tokens, return_dict=True)
            return out

        pos = run(positive_prompt, True)
        neg = run(negative_prompt, negative_grounding == 'grounded')
        _dbg(f'K2 Cond: contrato_treino={"ON" if training_vl_contract else "OFF"}')
        return ({'pos': pos, 'neg': neg},)


class K2Sampler:
    """O laço do runner: ruído próprio, schedule oficial, Euler explícito, CFG de 3 vias."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'models': ('K2_MODELS',),
                'adapter': ('K2_ADAPTER',),
                'refs': ('K2_REFS',),
                'cond': ('K2_COND',),
                'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
                'steps': ('INT', {'default': 8, 'min': 1, 'max': 200}),
                'text_guidance': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 20.0, 'step': 0.1,
                                  'tooltip': 'CFG de texto. Turbo=1.0, Raw=5.5.'}),
                'reference_guidance': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 8.0, 'step': 0.1,
                                       'tooltip': 'CFG da REFERENCIA (3 vias, como o runner). '
                                                  '>1 amplifica a influencia da imagem. Custa um '
                                                  'forward extra por step.'}),
            },
            'optional': {
                'mu': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 4.0, 'step': 0.01,
                       'tooltip': '0 = derivar da resolucao (Raw) ou 1.15 (Turbo).'}),
                'debug': ('BOOLEAN', {'default': False}),
            },
        }

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'sample'
    CATEGORY = 'CtxRush/K2 Native'

    def sample(self, models, adapter, refs, cond, seed, steps,
               text_guidance, reference_guidance, mu=0.0, debug=False):
        _DEBUG.update(on=bool(debug), calls=0, lora_hits=0, lora_misses=0, logged=False)
        device = comfy.model_management.get_torch_device()
        dit = models['dit']
        comfy.model_management.load_model_gpu(dit)
        m = dit.model.diffusion_model
        contract = adapter['contract']

        # --- entradas LoRA -------------------------------------------------
        named = dict(m.named_modules())
        block_entries, fusion_entries = [], []
        for path, (a, b) in adapter['pairs'].items():
            mod = named.get(path)
            if mod is None:
                continue
            if path.startswith('txtfusion'):
                fusion_entries.append((mod, a.to(device), b.to(device)))
            else:
                block_entries.append((mod, a.to(device), b.to(device), path))
        block_entries = _ctxrush_layer_scales(block_entries, adapter['layers'], adapter['layer_taper'])
        blocks_state = {'entries': block_entries, 'device': None}

        # --- latentes ------------------------------------------------------
        w, h = refs['width'], refs['height']
        latent_shape = (1, 16, h // 8, w // 8)
        # RUÍDO IGUAL AO DO RUNNER: torch.randn com generator CUDA. O
        # prepare_noise do ComfyUI usa outro RNG e produz outra amostra.
        generator = torch.Generator(device=device).manual_seed(int(seed))
        x = torch.randn(latent_shape, generator=generator, device=device, dtype=torch.float32)

        process_in = dit.model.process_latent_in
        references = [process_in(l).to(device) for l in refs['latents']]
        zero_refs = [torch.zeros_like(r) for r in references]

        pos_ctx = cond['pos']['cond'].to(device)
        neg_ctx = cond['neg']['cond'].to(device)
        pos_mask = cond['pos'].get('attention_mask')
        neg_mask = cond['neg'].get('attention_mask')

        def call(ctx, mask, ref_list):
            return _krea2_multiref_forward(
                m, x, t, ctx, ref_list, blocks_state, fusion_entries,
                adapter['block_strength'], {},
                fusion_strength=adapter['fusion_strength'],
                reference_timestep=contract['reference_timestep'],
                slot_axis=contract['slot_axis'],
                position_offset=contract['position_offset'],
                text_attention_mask=mask,
            ).float()

        # --- CFG de 3 vias, idêntico ao denoise() do runner -----------------
        def predict_velocity():
            if text_guidance == 1.0 and reference_guidance == 1.0:
                return call(pos_ctx, pos_mask, references)
            if text_guidance != 1.0 and reference_guidance == 1.0:
                neg = call(neg_ctx, neg_mask, references)
                full = call(pos_ctx, pos_mask, references)
                return neg + text_guidance * (full - neg)
            if text_guidance == 1.0:
                no_ref = call(pos_ctx, pos_mask, zero_refs)
                full = call(pos_ctx, pos_mask, references)
                return no_ref + reference_guidance * (full - no_ref)
            uncond = call(neg_ctx, neg_mask, zero_refs)
            with_ref = call(neg_ctx, neg_mask, references)
            full = call(pos_ctx, pos_mask, references)
            return (uncond + reference_guidance * (with_ref - uncond)
                    + text_guidance * (full - with_ref))

        patch = int(getattr(m, 'patch', 2))
        seqlen = (h // 8 // patch) * (w // 8 // patch)
        resolved_mu = (mu if mu > 0 else (KREA2_TURBO_MU if models['turbo'] else None))
        schedule, used_mu = build_krea2_timesteps(seqlen, steps, mu=resolved_mu)
        _dbg(f'K2 Sampler: {w}x{h} tokens={seqlen} steps={steps} mu={used_mu:.6f} '
             f'| tg={text_guidance} rg={reference_guidance} | schedule OFICIAL, Euler explicito')

        pbar = comfy.utils.ProgressBar(steps)
        with torch.no_grad():
            for cur, nxt in zip(schedule[:-1], schedule[1:]):
                t = x.new_full((x.shape[0],), cur)
                v = predict_velocity()
                x = x + (nxt - cur) * v      # integração explícita, como o runner
                pbar.update(1)

        return ({'samples': dit.model.process_latent_out(x.to(torch.float32)),
                 'downscale_ratio_spacial': 8},)


class K2Decode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'models': ('K2_MODELS',), 'samples': ('LATENT',)}}

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'decode'
    CATEGORY = 'CtxRush/K2 Native'

    def decode(self, models, samples):
        lat = samples['samples']
        lat = models['dit'].model.process_latent_in(lat)
        return (models['vae'].decode(lat),)


NODE_CLASS_MAPPINGS = {
    'K2LoadModels': K2LoadModels,
    'K2Adapter': K2Adapter,
    'K2ReferenceEncode': K2ReferenceEncode,
    'K2Conditioning': K2Conditioning,
    'K2Sampler': K2Sampler,
    'K2Decode': K2Decode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'K2LoadModels': 'K2 Load Models (contrato de treino)',
    'K2Adapter': 'K2 Adapter',
    'K2ReferenceEncode': 'K2 Reference Encode',
    'K2Conditioning': 'K2 Conditioning (VL do treino)',
    'K2Sampler': 'K2 Sampler (schedule oficial + CFG 3 vias)',
    'K2Decode': 'K2 Decode',
}
