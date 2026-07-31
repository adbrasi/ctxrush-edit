"""CtxRush Krea 2 Edit nodes for ComfyUI."""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .nodes_multiref import (
    NODE_CLASS_MAPPINGS as _MULTIREF_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _MULTIREF_NAMES,
)

NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_MULTIREF_CLASSES}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_MULTIREF_NAMES}

# Pack v3 (K2 Native). Import tolerante: se algo faltar na versão do ComfyUI
# do usuário, os nodes v1/v2 continuam carregando normalmente.
try:
    from .nodes_k2native import (
        NODE_CLASS_MAPPINGS as _K2_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _K2_NAMES,
    )
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_K2_CLASSES}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_K2_NAMES}
except Exception as _err:  # pragma: no cover
    print(f'[CtxRush] pack K2 Native indisponivel: {_err!r}')


try:
    from .nodes_trainbase import (
        NODE_CLASS_MAPPINGS as _TB_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _TB_NAMES,
    )
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_TB_CLASSES}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_TB_NAMES}
except Exception as _err:  # pragma: no cover
    print(f'[CtxRush] K2 Training Base indisponivel: {_err!r}')


try:
    from .nodes_runner_bridge import (
        NODE_CLASS_MAPPINGS as _RB_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _RB_NAMES,
    )
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_RB_CLASSES}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_RB_NAMES}
except Exception as _err:  # pragma: no cover
    print(f'[CtxRush] K2 Runner Bridge indisponivel: {_err!r}')


try:
    from .nodes_reference_control import (
        NODE_CLASS_MAPPINGS as _RC_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _RC_NAMES,
    )
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_RC_CLASSES}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_RC_NAMES}
except Exception as _err:  # pragma: no cover
    print(f'[CtxRush] K2 Reference Control indisponivel: {_err!r}')


try:
    from .nodes_prompt_rewriter import (
        NODE_CLASS_MAPPINGS as _PR_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _PR_NAMES,
    )
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_PR_CLASSES}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_PR_NAMES}
except Exception as _err:  # pragma: no cover
    print(f'[CtxRush] K2 Prompt Rewriter indisponivel: {_err!r}')


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
