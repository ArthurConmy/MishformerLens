#%%

# Auto-reload modules:
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic(u"%load_ext autoreload")
    ipython.magic(u"%autoreload 2")

# Deal with HuggingFace cache:
import os
cache_dir = '/workspace/hf/'
assert os.path.exists(cache_dir), f"Cache directory {cache_dir} does not exist"
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir

# Import torch and disable gradients:
import torch
torch.set_grad_enabled(False)

# HF Transformers patching:
import transformers
import logging
from typing import Optional, Union, Literal, NamedTuple
from packaging import version
import torch
import jaxtyping
from jaxtyping import Float, Int
from mishax import ast_patcher

if "patcher" in globals():
    print("Unpatching...")
    patcher.__exit__(None, None, None)
    del patcher

# Apply patches now:
()
patcher.__enter__()
# N.B. patcher.__exit__(None,None,None) exits.

#%%

# TL imports:
from transformer_lens import hook_points
from transformer_lens import HookedTransformerConfig
from transformer_lens import HookedTransformer as TransformerLensHookedTransformer
from transformer_lens import loading_from_pretrained
from transformer_lens import utils as transformer_lens_utils
from transformer_lens import ActivationCache
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

#%%

# TODO(v1): remove copy paste from TransformerLens
# BEGIN COPY PASTE
SingleLoss = Float[torch.Tensor, ""]  # Type alias for a single element tensor
LossPerToken = Float[torch.Tensor, "batch pos-1"]
Loss = Union[SingleLoss, LossPerToken]

class Output(NamedTuple):
    """Output Named Tuple.

    Named tuple object for if we want to output both logits and loss.
    """

    logits: Float[torch.Tensor, "batch pos d_vocab"]
    loss: Loss

DTYPE_FROM_STRING = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}
# END COPY PASTE

#%%

DEVICE = "cuda:0"  # "mps"

#%%

# Load model:
model = HookedTransformer.from_pretrained(
    "gpt2",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
    fold_value_biases=False,
    device=DEVICE,
    attn_implementation="eager",
)

#%%

# Run model:
output = model("Hey")

#%%

logits, cache = model.run_with_cache("Hey")
assert any(["hook_q" in k for k in cache.keys()])

#%%

# %%

#%%
