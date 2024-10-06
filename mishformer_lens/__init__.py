import contextlib
from . import ast_patching

PATCHING_CTX_MANAGERS: list[contextlib.AbstractContextManager] = []

def exit_all_ast_patchers():
    for ctx_manager in PATCHING_CTX_MANAGERS:
        ctx_manager.__exit__(None, None, None)
    PATCHING_CTX_MANAGERS.clear()

def enter_all_ast_patchers():
    for patcher in ast_patching.PATCHERS.values():
        ctx_manager = patcher()
        ctx_manager.__enter__()
        PATCHING_CTX_MANAGERS.append(ctx_manager)

@contextlib.contextmanager
def ast_patching_disabled():
    try:
        exit_all_ast_patchers()
        yield
    finally:
        enter_all_ast_patchers()

enter_all_ast_patchers()

from .hooked_transformer import HookedTransformer
