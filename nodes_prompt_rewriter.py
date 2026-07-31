"""K2 Prompt Rewriter — entrega o system prompt e o pedido formatado para um LLM.

O modelo de edicao foi treinado em 30.000 captions que NAO sao um estilo so:
sao 11 dialetos com gramaticas de superficie diferentes (EDIT, NEXT, PANEL,
CTX, POSE, EXPRESSION, OBJECT, STYLE, BACKGROUND, CROP, INSCENE). Pedir a
edicao em portugues corrido produz um caption fora de qualquer um deles, e o
adapter responde pior.

Este node nao chama LLM nenhum. Ele so monta as duas strings que o SEU node de
LLM precisa:

  system_prompt -> prompts/rewriter_manual_dialect.md (embutido no repo)
  user_prompt   -> "TYPE: <n>\nREQUEST: <seu pedido>"

Esse formato de duas linhas e exatamente o que o system prompt declara aceitar
na sua secao 1; qualquer outro formato faz ele responder o JSON de erro.

A saida do LLM e um objeto JSON unico `{"prompt_final": "..."}`. Extraia o
valor dessa chave antes de mandar para o CLIPTextEncode.
"""
from pathlib import Path

_AQUI = Path(__file__).parent
_ARQUIVO_SISTEMA = _AQUI / 'prompts' / 'rewriter_manual_dialect.md'

# Rotulo -> numero do TYPE. A ordem e a do menu do system prompt (secao 3);
# os totais sao a distribuicao real das 30.000 captions de treino e servem
# para o usuario escolher com base no que o modelo realmente viu.
DIALETOS = [
    ('1 — EDIT · edicao geral ilustrada (11.752)', 1),
    ('2 — NEXT · proxima cena / continuidade (4.129)', 2),
    ('3 — PANEL · pagina ou painel de manga (4.950)', 3),
    ('4 — CTX · descricao completa do alvo + tags (865)', 4),
    ('5 — POSE · pose, cabeca, olhar — foto (1.279)', 5),
    ('6 — EXPRESSION · expressao facial — foto (1.075)', 6),
    ('7 — OBJECT · add/remove/troca de objeto ou roupa (1.375)', 7),
    ('8 — STYLE · transferencia de estilo ou meio (1.380)', 8),
    ('9 — BACKGROUND · fundo, clima, luz — foto (1.341)', 9),
    ('10 — CROP · crop, zoom, reenquadre 2D (1.381)', 10),
    ('11 — INSCENE · camera se move na mesma cena (473)', 11),
]
_ROTULOS = [rotulo for rotulo, _ in DIALETOS]
_NUMERO = dict(DIALETOS)

_cache_sistema = None


def _system_prompt() -> str:
    """Le o system prompt embutido, uma vez por processo."""
    global _cache_sistema
    if _cache_sistema is None:
        if not _ARQUIVO_SISTEMA.exists():
            raise FileNotFoundError(
                f'[K2PromptRewriter] system prompt nao encontrado em {_ARQUIVO_SISTEMA}. '
                'O arquivo vem junto com o repositorio dos nodes — se sumiu, '
                'refaca o git pull ou copie prompts/rewriter_manual_dialect.md.'
            )
        _cache_sistema = _ARQUIVO_SISTEMA.read_text(encoding='utf-8')
    return _cache_sistema


class K2PromptRewriterPrompt:
    """Monta system prompt + pedido no formato TYPE/REQUEST.

    Ligue `system_prompt` e `user_prompt` nas entradas equivalentes do seu node
    de LLM. Nao troque a ordem: o system prompt assume que o pedido chega com o
    seletor na primeira linha.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'dialeto': (_ROTULOS, {'default': _ROTULOS[0]}),
                'pedido': ('STRING', {
                    'multiline': True,
                    'default': '',
                    'placeholder': 'Seu pedido, em portugues mesmo. '
                                   'Ex.: deixa ela com um sorriso discreto, mantem o resto',
                }),
            },
        }

    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('system_prompt', 'user_prompt')
    FUNCTION = 'montar'
    CATEGORY = 'CtxRush/K2'
    DESCRIPTION = ('Entrega o system prompt do rewriter e o pedido no formato '
                   'TYPE/REQUEST. A saida do LLM e {"prompt_final": "..."}.')

    def montar(self, dialeto, pedido):
        texto = (pedido or '').strip()
        if not texto:
            # Falhar aqui e melhor que gastar uma chamada de LLM para receber
            # um caption inventado a partir de um pedido vazio.
            raise ValueError(
                '[K2PromptRewriter] o campo `pedido` esta vazio — '
                'escreva o que voce quer editar.'
            )

        numero = _NUMERO.get(dialeto)
        if numero is None:
            raise ValueError(
                f'[K2PromptRewriter] dialeto desconhecido: {dialeto!r}. '
                f'Escolha um dos {len(_ROTULOS)} do menu.'
            )

        # Quebra de linha interna viraria uma segunda "linha de comando" e
        # confundiria o parser do system prompt, que le o pedido como o que vem
        # depois de REQUEST:. Colapsamos para uma linha so.
        texto = ' '.join(texto.split())

        user_prompt = f'TYPE: {numero}\nREQUEST: {texto}'
        return (_system_prompt(), user_prompt)


NODE_CLASS_MAPPINGS = {
    'K2PromptRewriterPrompt': K2PromptRewriterPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'K2PromptRewriterPrompt': 'K2 Prompt Rewriter (system + pedido)',
}
