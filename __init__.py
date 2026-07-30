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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
