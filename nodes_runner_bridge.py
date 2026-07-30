"""CtxRush — K2 Runner Bridge: o runner do fork, dentro do ComfyUI.

POR QUE ESTE PACK EXISTE
------------------------
Os nodes nativos (v2 e K2 Native) reimplementam o contrato de treino dentro do
ComfyUI. Isso funciona, mas custou ~6 h de bisseção para achar CINCO
divergências silenciosas entre o ComfyUI e a árvore que treinou o adapter:

  1. base fp8-scaled vs fp8 CRU sem escala (v relL2 0.58 — a causa dominante)
  2. Qwen3-VL com MRoPE 3D + DeepStack no ComfyUI novo (contexto relL2 1.36)
  3. atenção GQA por `enable_gqa=True` vs `repeat_interleave`
  4. as 7 chaves `.diff_b` da turbo, que o runner ignora e o ComfyUI aplica
  5. ruído do KSampler em CPU vs `torch.randn` CUDA do runner

Cada uma foi corrigida — mas cada atualização do ComfyUI pode criar a sexta.
Este pack elimina essa classe inteira de problema: em vez de imitar o runner,
ele **executa o runner**, com o código e a árvore de dependências do fork.

COMO
----
Subprocesso isolado. Os dois ComfyUI (o do usuário e o `submodules/ComfyUI` do
fork) não podem coexistir no mesmo interpretador — `comfy.*` colidiria, e foi
exatamente essa colisão que criou a divergência nº 2. Isolar por processo é a
única garantia forte.

Troca de dados por arquivos temporários (PNG de entrada, PNG de saída). Custo:
o modelo recarrega a cada execução (~40 s em 1024). Em troca, o resultado é o
do runner por construção — não há o que divergir.

QUANDO USAR CADA PACK
---------------------
  - **Runner Bridge (este)**: produção confiável, validação, quando o
    resultado tem que ser exatamente o do runner.
  - **Multi-Ref v2 / K2 Native**: quando você precisa dos latentes dentro do
    grafo (img2img, upscale, encadear com outros nodes) e aceita manter as
    correções em dia.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import folder_paths

# Raiz do fork diffusion-pipe. Pode ser sobrescrita pela env var
# CTXRUSH_DIFFUSION_PIPE se o repo estiver em outro lugar.
FORK_ROOT = Path(os.environ.get('CTXRUSH_DIFFUSION_PIPE', '/workspace/projects/diffusion-pipe'))
PYTHON = os.environ.get('CTXRUSH_PYTHON', '/venv/main/bin/python')
RUNNER = 'tools/infer_reference_adapter.py'


def _read_metadata(path):
    """Metadata do header do safetensors, para montar a config do runner."""
    import struct
    try:
        with open(path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            return json.loads(f.read(n)).get('__metadata__', {}) or {}
    except Exception:
        return {}


def _lora_ranks(path):
    """Descobre rank dos blocks e do txtfusion lendo os próprios tensores.

    O runner precisa disto no `[adapter]` da config; ler do arquivo evita
    depender de o usuário lembrar com que rank treinou.
    """
    try:
        from safetensors import safe_open
        rank_blocks, rank_txtfusion = None, None
        with safe_open(path, framework='pt') as f:
            for k in f.keys():
                if '.lora_A.' not in k:
                    continue
                r = f.get_slice(k).get_shape()[0]
                if 'txtfusion' in k:
                    rank_txtfusion = r if rank_txtfusion is None else rank_txtfusion
                else:
                    rank_blocks = r if rank_blocks is None else rank_blocks
        return rank_blocks or 64, rank_txtfusion or 0
    except Exception:
        return 64, 0


def _write_config(work: Path, adapter_path: str, diffusion_model: str,
                  text_encoder: str, vae: str, max_refs: int,
                  vl_image_label_fallback: str = 'image') -> Path:
    """Gera o TOML que o runner espera, a partir da metadata do adapter.

    Autoconfigurável de propósito: o contrato REAL é o que o trainer gravou no
    header, não o que alguém digitou num campo.
    """
    md = _read_metadata(adapter_path)
    if not md:
        print('[K2RunnerBridge] AVISO: adapter SEM metadata — o contrato caiu inteiro nos '
              'defaults. Confira se este safetensors saiu deste trainer.', flush=True)
    rank, txtfusion_rank = _lora_ranks(adapter_path)
    family = md.get('control_family') or md.get('model_type') or 'krea2_multiref_grounded'
    section = family if family.startswith('krea2_') else 'krea2_multiref_grounded'
    ref_ts = 'target' if float(md.get('reference_model_timestep', 0) or 0) > 0 else 'zero'
    vl_px = int(md.get('vl_image_max_pixels', 147456) or 147456)
    layout = md.get('vl_prompt_layout', 'picture_n_vision_blocks')
    style = 'plain' if str(layout).startswith('plain') else 'picture_n'

    def _bool(chave, padrao):
        v = md.get(chave)
        return padrao if v is None else str(v).strip().lower() in ('1', 'true', 'yes')

    # Campos que ANTES eram omitidos (caindo no default do pipeline) ou
    # chutados. Cada um muda o forward de forma silenciosa se divergir:
    #   independent_condition   -> mascara de atencao densa LxL (krea2_multiref.py:155)
    #   reference_position_scale-> escala as coords RoPE da referencia (:95,107)
    #   vl_longest_side         -> troca prepare_vl_image por a variante longest-side
    #   condition_only_lora     -> routing do delta; estava hardcoded true
    independent = _bool('independent_condition', False)
    cond_only = _bool('condition_only_lora', True)
    pos_scale = float(md.get('reference_position_scale', 1.0) or 1.0)
    vl_longest = int(md.get('vl_longest_side', 0) or 0)

    # O rotulo do bloco de visao entra LITERALMENTE no prompt do Qwen3-VL. O
    # default do trainer e 'Picture', mas todo este projeto treina com 'image'.
    # Adapters novos gravam a chave (models/krea2_edit.py); os antigos nao —
    # nesse caso avisamos que foi inferido em vez de fingir certeza.
    label = md.get('vl_image_label')
    if label is None:
        label = vl_image_label_fallback
        print(f'[K2RunnerBridge] AVISO: o adapter nao grava vl_image_label; usando '
              f'{label!r}. Se ele foi treinado com outro rotulo, o prompt de visao '
              f'sai fora da distribuicao. (Adapters novos gravam esta chave.)', flush=True)

    toml = f"""# Gerado pelo K2 Runner Bridge a partir da metadata do adapter.
output_dir = '{work}'
dataset = '{work}/dummy_dataset.toml'
epochs = 1
micro_batch_size_per_gpu = 1
gradient_accumulation_steps = 1
save_every_n_steps = 1000000

[model]
type = '{section}'
diffusion_model = '{diffusion_model}'
vae = '{vae}'
text_encoders = [
    {{path = '{text_encoder}', type = 'krea2'}},
]
dtype = 'bfloat16'
diffusion_model_dtype = 'float8'
timestep_sample_method = 'logit_normal'
flux_shift = true

[{section}]
max_refs = {max(1, int(md.get('max_refs', max_refs) or max_refs))}
slot_axis = '{md.get('slot_axis', 'width')}'
position_mode = '{md.get('position_mode', 'width_shift')}'
reference_position_offset = {float(md.get('reference_position_offset', 1.0) or 1.0)}
reference_position_scale = {pos_scale}
reference_timestep = '{ref_ts}'
condition_dropout = 0.0
condition_only_lora = {str(cond_only).lower()}
independent_condition = {str(independent).lower()}
vl_prompt_style = '{style}'
vl_image_label = '{label}'
vl_longest_side = {vl_longest}
vl_image_max_pixels = {vl_px}
caption_dropout = {float(md.get('caption_dropout', 0.1) or 0.1)}
txtfusion_rank = {txtfusion_rank}

[adapter]
type = 'lora'
rank = {rank}
dtype = 'bfloat16'

[optimizer]
type = 'AdamW8bitKahan'
lr = 1e-4
"""
    cfg = work / 'runner_config.toml'
    cfg.write_text(toml)
    (work / 'dummy_dataset.toml').write_text(
        "resolutions = [512]\nenable_ar_bucket = false\nframe_buckets = [1]\n")
    return cfg


def _save_images(tensor, work: Path):
    """Salva TODAS as imagens do batch, uma por slot de referencia.

    O runner aceita --reference repetido e a ORDEM das flags E a ordem dos
    slots — e o que "image 1"/"image 2" enderecam no caption. Pegar so
    tensor[0] descartaria as demais em SILENCIO: o pipeline preencheria os
    slots restantes com zeros, que e a representacao in-distribution de
    "referencia ausente" (krea2_multiref.py:275-283). Sai imagem plausivel
    que simplesmente ignorou a segunda referencia — o pior tipo de bug.
    """
    from PIL import Image
    if tensor.ndim != 4:
        raise ValueError(f'IMAGE precisa ser (B,H,W,C); veio {tuple(tensor.shape)}')
    canais = tensor.shape[-1]
    if canais not in (3, 4):
        raise ValueError(f'IMAGE com {canais} canais nao suportada (esperado 3 ou 4)')
    if canais == 4:
        print('[K2RunnerBridge] AVISO: IMAGE com alpha — o canal foi DESCARTADO '
              '(o runner trabalha em RGB). Componha o fundo antes deste node.', flush=True)
    caminhos = []
    for i in range(tensor.shape[0]):
        arr = np.clip(tensor[i, ..., :3].detach().float().cpu().numpy() * 255.0,
                      0, 255).astype(np.uint8)
        destino = work / f'reference_{i:02d}.png'
        Image.fromarray(arr, 'RGB').save(destino, 'PNG')
        caminhos.append(destino)
    if len(caminhos) > 1:
        print(f'[K2RunnerBridge] {len(caminhos)} referencias -> slots image 1..{len(caminhos)} '
              f'(a ordem do batch e a ordem dos slots)', flush=True)
    return caminhos


def _load_image(path: Path):
    from PIL import Image
    with Image.open(path) as im:
        arr = np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _resolve(kind, name):
    if not name or name == 'nenhuma':
        return None
    p = folder_paths.get_full_path(kind, name)
    return p or name


# Logs de execucao ficam FORA do diretorio temporario: o `finally` apaga o
# temp, e apagar o log justamente quando deu erro destroi a unica evidencia.
LOG_DIR = Path(os.environ.get('CTXRUSH_LOG_DIR', '/workspace/logs/k2_runner_bridge'))


def _free_vram(motivo=''):
    """Libera a VRAM que o ComfyUI esta segurando.

    OBRIGATORIO antes de lançar o runner: o subprocesso carrega ~13 GB (o DiT
    tem 12.8B) e a GPU tem 32 GB. Se o ComfyUI ainda estiver com modelos
    residentes de um workflow anterior, o subprocesso morre com OOM — e o erro
    aparece como "runner falhou", que nao diz nada ao usuario.
    """
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception as e:  # pragma: no cover
        print(f'[K2RunnerBridge] aviso: nao consegui liberar VRAM ({e!r})', flush=True)
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            livre, total = torch.cuda.mem_get_info()
            print(f'[K2RunnerBridge] VRAM livre antes do runner{motivo}: '
                  f'{livre / 2**30:.1f} de {total / 2**30:.1f} GB', flush=True)
            return livre / 2**30
    except Exception:
        pass
    return None


def _rotate_logs(keep=20):
    """Mantem so os N logs mais recentes — sem isto o diretorio cresce sem fim."""
    try:
        logs = sorted(LOG_DIR.glob('runner_*.log'), key=lambda p: p.stat().st_mtime)
        for old in logs[:-keep]:
            old.unlink(missing_ok=True)
    except Exception:
        pass


def _run(cmd, work, timeout):
    """Executa o runner no venv e na raiz do fork, com log preservado."""
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _rotate_logs()
    log_path = LOG_DIR / f'runner_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}.log'
    livre = _free_vram()
    if livre is not None and livre < 14.0:
        print(f'[K2RunnerBridge] ATENCAO: so {livre:.1f} GB livres. O runner precisa de '
              f'~14 GB (ou menos com blocks_to_swap). Se falhar, aumente blocks_to_swap '
              f'ou feche outros processos na GPU.', flush=True)
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(FORK_ROOT), env=env,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        parcial = ((exc.stdout or b'').decode('utf-8', 'replace') if isinstance(exc.stdout, bytes)
                   else (exc.stdout or ''))
        log_path.write_text(f'TIMEOUT apos {timeout}s\n\n{parcial}')
        raise RuntimeError(
            f'O runner estourou o timeout de {timeout}s. Se a geracao e grande (1024+ '
            f'ou muitos steps), aumente timeout_s. Log: {log_path}') from exc
    log = (proc.stdout or '') + '\n' + (proc.stderr or '')
    log_path.write_text(' '.join(cmd) + '\n\n' + log)
    if proc.returncode != 0:
        cauda = '\n'.join(log.strip().splitlines()[-25:])
        dica = ''
        if any(t in log for t in ('out of memory', 'OutOfMemoryError', 'CUDA error: out of memory')):
            dica = ('\n\nISTO FOI OOM DE VRAM. Aumente blocks_to_swap (tente 8, depois 16 ou 24), '
                    'reduza width/height, ou feche outros processos na GPU. Com blocks_to_swap=24 '
                    'o modelo roda em ~10 GB.')
        raise RuntimeError(
            f'O runner falhou (rc={proc.returncode}). Log completo: {log_path}\n'
            f'Ultimas linhas:\n{cauda}{dica}')
    print(f'[K2RunnerBridge] runner concluiu em {time.time() - started:.1f}s '
          f'(log: {log_path})', flush=True)
    # o subprocesso ja morreu e devolveu a VRAM ao driver; isto so limpa o
    # cache do alocador do PROPRIO ComfyUI, para o proximo node comecar limpo
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return log


class K2RunnerBridge:
    """Executa o runner do fork e devolve a imagem. Fidelidade por construção."""

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list('loras')
        return {
            'required': {
                'image': ('IMAGE', {'tooltip': 'Imagem de referencia.'}),
                'adapter': (loras, {'tooltip': 'O adapter de edicao treinado (LoRA).'}),
                'diffusion_model': (folder_paths.get_filename_list('diffusion_models'),),
                'text_encoder': (folder_paths.get_filename_list('text_encoders'),),
                'vae': (folder_paths.get_filename_list('vae'),),
                'prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
                'width': ('INT', {'default': 1024, 'min': 64, 'max': 4096, 'step': 16}),
                'height': ('INT', {'default': 1024, 'min': 64, 'max': 4096, 'step': 16}),
                'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
                'variant': (['turbo', 'raw'], {'default': 'turbo',
                            'tooltip': 'turbo: 8 steps / CFG 1.0 / mu 1.15. raw: 28 / 5.5 / mu por resolucao.'}),
            },
            'optional': {
                'turbo_lora': (['nenhuma'] + loras, {'default': 'nenhuma',
                               'tooltip': 'Fundida no base antes do adapter, como o runner faz. '
                                          'Necessaria se o diffusion_model for o RAW e voce quer turbo.'}),
                'negative_prompt': ('STRING', {'default': '', 'multiline': True}),
                'steps': ('INT', {'default': 0, 'min': 0, 'max': 200,
                          'tooltip': '0 = padrao da variante (turbo 8, raw 28).'}),
                'text_guidance': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 20.0, 'step': 0.1,
                                  'tooltip': '0 = padrao da variante (turbo 1.0, raw 5.5).'}),
                'reference_guidance': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 8.0, 'step': 0.1,
                                       'tooltip': 'CFG da REFERENCIA (3 vias). >1 amplifica a imagem. '
                                                  'Custa um forward extra por step.'}),
                'adapter_scale': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.05}),
                'reference_fit': (['crop', 'stretch', 'native_latent'], {'default': 'crop'}),
                'blocks_to_swap': ('INT', {'default': 0, 'min': 0, 'max': 28,
                                   'tooltip': 'Aumente se faltar VRAM (o modelo tem 12.8B).'}),
                'timeout_s': ('INT', {'default': 1800, 'min': 60, 'max': 7200}),
            },
        }

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'run'
    CATEGORY = 'CtxRush/K2 Runner'
    DESCRIPTION = ('Executa tools/infer_reference_adapter.py do fork em subprocesso isolado. '
                   'O resultado E o do runner, nao uma reimplementacao.')

    def _build(self, work, image, adapter, diffusion_model, text_encoder, vae,
               prompt, width, height, seed, variant, turbo_lora, negative_prompt,
               steps, text_guidance, reference_guidance, adapter_scale,
               reference_fit, blocks_to_swap, extra_loras=()):
        ref_pngs = _save_images(image, work)
        adapter_path = _resolve('loras', adapter)
        cfg = _write_config(work, adapter_path,
                            _resolve('diffusion_models', diffusion_model),
                            _resolve('text_encoders', text_encoder),
                            _resolve('vae', vae), max_refs=len(ref_pngs))
        out_png = work / 'out.png'
        cmd = [PYTHON, RUNNER,
               '--config', str(cfg),
               '--adapter', adapter_path,
               '--prompt', prompt,
               '--negative-prompt', negative_prompt or '',
               '--width', str(width), '--height', str(height),
               '--seed', str(seed),
               '--krea-variant', variant,
               '--reference-guidance', str(reference_guidance),
               '--adapter-scale', str(adapter_scale),
               '--reference-fit', reference_fit,
               '--output', str(out_png)]
        for ref in ref_pngs:           # uma flag por slot, na ordem do batch
            cmd += ['--reference', str(ref)]
        if steps:
            cmd += ['--steps', str(steps)]
        if text_guidance:
            cmd += ['--text-guidance', str(text_guidance)]
        if blocks_to_swap:
            cmd += ['--blocks-to-swap', str(blocks_to_swap)]
        tl = _resolve('loras', turbo_lora)
        if tl:
            cmd += ['--turbo-lora', tl]
        for path, strength in extra_loras:
            cmd += ['--extra-lora', f'{path}:{strength}']
        return cmd, out_png

    def run(self, image, adapter, diffusion_model, text_encoder, vae, prompt,
            width, height, seed, variant, turbo_lora='nenhuma', negative_prompt='',
            steps=0, text_guidance=0.0, reference_guidance=1.0, adapter_scale=1.0,
            reference_fit='crop', blocks_to_swap=0, timeout_s=1800):
        if not (FORK_ROOT / RUNNER).exists():
            raise RuntimeError(
                f'runner nao encontrado em {FORK_ROOT / RUNNER}. Ajuste a env var '
                f'CTXRUSH_DIFFUSION_PIPE para a raiz do diffusion-pipe.')
        work = Path(tempfile.mkdtemp(prefix='k2runner_'))
        try:
            cmd, out_png = self._build(
                work, image, adapter, diffusion_model, text_encoder, vae, prompt,
                width, height, seed, variant, turbo_lora, negative_prompt, steps,
                text_guidance, reference_guidance, adapter_scale, reference_fit,
                blocks_to_swap)
            _run(cmd, work, timeout_s)
            if not out_png.exists():
                raise RuntimeError(f'runner terminou sem gerar {out_png}')
            return (_load_image(out_png),)
        finally:
            shutil.rmtree(work, ignore_errors=True)


class K2RunnerBridgeMultiLoRA(K2RunnerBridge):
    """Igual, mais até 3 LoRAs de estilo fundidas no base antes do adapter.

    A ordem reproduz o runner: base -> turbo -> LoRAs extras -> adapter
    (routado, nunca fundido). O adapter fica sempre por cima, porque o delta
    condition-only só vale nas rows da referência.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = K2RunnerBridge.INPUT_TYPES()
        loras = ['nenhuma'] + folder_paths.get_filename_list('loras')
        for i in (1, 2, 3):
            base['optional'][f'lora_{i}'] = (loras, {'default': 'nenhuma'})
            base['optional'][f'lora_{i}_strength'] = (
                'FLOAT', {'default': 1.0, 'min': -2.0, 'max': 2.0, 'step': 0.05})
        return base

    FUNCTION = 'run_multi'
    DESCRIPTION = ('Runner Bridge com ate 3 LoRAs de estilo. Ordem: base -> turbo -> '
                   'LoRAs -> adapter de edicao (routado).')

    def run_multi(self, image, adapter, diffusion_model, text_encoder, vae, prompt,
                  width, height, seed, variant, turbo_lora='nenhuma', negative_prompt='',
                  steps=0, text_guidance=0.0, reference_guidance=1.0, adapter_scale=1.0,
                  reference_fit='crop', blocks_to_swap=0, timeout_s=1800,
                  lora_1='nenhuma', lora_1_strength=1.0,
                  lora_2='nenhuma', lora_2_strength=1.0,
                  lora_3='nenhuma', lora_3_strength=1.0):
        if not (FORK_ROOT / RUNNER).exists():
            raise RuntimeError(
                f'runner nao encontrado em {FORK_ROOT / RUNNER}. Ajuste CTXRUSH_DIFFUSION_PIPE.')
        extras = []
        for name, strength in ((lora_1, lora_1_strength),
                               (lora_2, lora_2_strength),
                               (lora_3, lora_3_strength)):
            path = _resolve('loras', name)
            if path and float(strength) != 0.0:
                extras.append((path, float(strength)))
        work = Path(tempfile.mkdtemp(prefix='k2runner_'))
        try:
            cmd, out_png = self._build(
                work, image, adapter, diffusion_model, text_encoder, vae, prompt,
                width, height, seed, variant, turbo_lora, negative_prompt, steps,
                text_guidance, reference_guidance, adapter_scale, reference_fit,
                blocks_to_swap, extra_loras=extras)
            if extras:
                print(f'[K2RunnerBridge] {len(extras)} LoRA(s) extra: '
                      + ', '.join(f'{Path(p).name}@{s}' for p, s in extras), flush=True)
            _run(cmd, work, timeout_s)
            if not out_png.exists():
                raise RuntimeError(f'runner terminou sem gerar {out_png}')
            return (_load_image(out_png),)
        finally:
            shutil.rmtree(work, ignore_errors=True)


NODE_CLASS_MAPPINGS = {
    'K2RunnerBridge': K2RunnerBridge,
    'K2RunnerBridgeMultiLoRA': K2RunnerBridgeMultiLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'K2RunnerBridge': 'K2 Runner Bridge (runner exato)',
    'K2RunnerBridgeMultiLoRA': 'K2 Runner Bridge + 3 LoRAs',
}
