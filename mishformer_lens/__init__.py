import contextlib
from . import ast_patching

AST_PATCHING_CTX_MANAGERS: list[contextlib.AbstractContextManager] = []

def exit_all_ast_patchers():
    for ctx_manager in AST_PATCHING_CTX_MANAGERS:
        ctx_manager.__exit__(None, None, None)
    AST_PATCHING_CTX_MANAGERS.clear()

def enter_all_ast_patchers():
    for ast_patcher in ast_patching.PATCHERS.values():
        ctx_manager = ast_patcher()
        ctx_manager.__enter__()
        AST_PATCHING_CTX_MANAGERS.append(ctx_manager)

@contextlib.contextmanager
def ast_patching_disabled():
    try:
        exit_all_ast_patchers()
        yield
    finally:
        enter_all_ast_patchers()

enter_all_ast_patchers()

from .hooked_transformer import HookedTransformer
