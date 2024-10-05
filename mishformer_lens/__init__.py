import contextlib
from . import ast_patching

PATCHING_CTX_MANAGERS: list[contextlib.AbstractContextManager] = []

for patcher in ast_patching.PATCHERS.values():
    ctx_manager = patcher()
    PATCHING_CTX_MANAGERS.append(ctx_manager)
    ctx_manager.__enter__()

from .hooked_transformer import HookedTransformer
