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

# Other necessary imports
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import mishformer_lens
from mishformer_lens import HookedTransformer
from mishformer_lens import hooked_transformer as hooked_transformer_lib
# Ensure CUDA is available
assert torch.cuda.is_available(), "CUDA is not available. Please use a GPU runtime."

from transformer_lens import HookedTransformer as TLHookedTransformer

DEVICE = "cuda"
MY_STRING = "Hello, world!"
DTYPE = torch.float32

#%%

# Use the context manager to disable patching temporarily
with mishformer_lens.ast_patching_disabled():
    tl_model = TLHookedTransformer.from_pretrained("EleutherAI/pythia-70m", device=DEVICE)
    tl_model.set_use_hook_mlp_in(True)
    tokens = tl_model.to_tokens(MY_STRING, prepend_bos=True).to(DEVICE)
    tl_raw_logits = tl_model(tokens)[0]
    tl_logits, tl_cache = tl_model.run_with_cache(tokens)

# Patching is automatically re-enabled after the with block

# Now you can continue with the rest of your notebook code
# The patchers will be enabled again here

#%%

with mishformer_lens.ast_patching_disabled():
    hf_model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m",
        torch_dtype=DTYPE,
    ).to(DEVICE)
    hf_raw_logits = hf_model(tokens)[0]

#%%

def check_performance(tl_model, hf_model, margin):
    """
    Check that the TransformerLens model and the HuggingFace have
    approximately the same confidence in the expected answer.
    """
    prompt = " Unable"
    tokens = tl_model.tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)

    expected_token = tl_model.tokenizer.encode(" to")[0]  # Assume this is the expected token to predict

    tl_logits = tl_model(tokens, prepend_bos=False)[0, -1].float()
    hf_logits = hf_model(tokens).logits[0, -1].float()
    tl_prob = torch.softmax(tl_logits, dim=-1)[expected_token].item()
    hf_prob = torch.softmax(hf_logits, dim=-1)[expected_token].item()
    
    print(f"TransformerLens probability: {tl_prob:.4f}")
    print(f"HuggingFace probability: {hf_prob:.4f}")
    print(f"Difference: {abs(tl_prob - hf_prob):.4f}")
    assert tl_prob + margin > hf_prob, f"TL prob {tl_prob} not within {margin} of HF prob {hf_prob}"

#%%

model = HookedTransformer.from_pretrained_no_processing(
    "EleutherAI/pythia-70m",
    torch_dtype=DTYPE,
    # attn_implementation='eager',
)

#%%

logits, cache = model.run_with_cache(tokens)

#%%

torch.testing.assert_close(logits, hf_raw_logits)

#%%
