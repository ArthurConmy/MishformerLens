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
from mishformer_lens import HookedTransformer
from mishformer_lens import hooked_transformer as hooked_transformer_lib
# Ensure CUDA is available
assert torch.cuda.is_available(), "CUDA is not available. Please use a GPU runtime."

from transformer_lens import HookedTransformer as TLHookedTransformer

DEVICE = "cuda"

#%%

tl_model = TLHookedTransformer.from_pretrained("EleutherAI/pythia-70m", device=DEVICE)
print(tl_model.hook_dict.keys())

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

# Test the function
pythia_named_modules = ['_ast_patched_hf_model.' + s for s in [
    '', 'gpt_neox', 'gpt_neox.embed_in', 'gpt_neox.emb_dropout', 'gpt_neox.layers', 'gpt_neox.layers.0', 
    'gpt_neox.layers.0.input_layernorm', 'gpt_neox.layers.0.input_layernorm.hook_normalized', 
    'gpt_neox.layers.0.post_attention_layernorm', 'gpt_neox.layers.0.post_attention_layernorm.hook_normalized', 
    'gpt_neox.layers.0.post_attention_dropout', 'gpt_neox.layers.0.post_mlp_dropout', 
    'gpt_neox.layers.0.attention', 'gpt_neox.layers.0.attention.rotary_emb', 
    'gpt_neox.layers.0.attention.query_key_value', 'gpt_neox.layers.0.attention.dense', 
    'gpt_neox.layers.0.attention.attention_dropout', 'gpt_neox.layers.0.mlp', 
    'gpt_neox.layers.0.mlp.dense_h_to_4h', 'gpt_neox.layers.0.mlp.dense_4h_to_h', 
    'gpt_neox.layers.0.mlp.hook_pre', 'gpt_neox.layers.0.mlp.hook_post', 'gpt_neox.layers.0.mlp.act', 
    'gpt_neox.layers.0.hook_resid_pre', 'gpt_neox.layers.0.hook_resid_mid', 
    'gpt_neox.layers.0.hook_resid_post', 'gpt_neox.layers.0.hook_mlp_in', 
    'gpt_neox.layers.0.hook_mlp_out', 'gpt_neox.layers.0.hook_attn_out', 
    'gpt_neox.final_layer_norm', 'gpt_neox.final_layer_norm.hook_normalized', 
    'gpt_neox.final_layer_norm.hook_scale', 'gpt_neox.hook_embed', 'gpt_neox.rotary_emb', 'embed_out'
]]

for module_name in pythia_named_modules:
    try:
        mapped_name = hooked_transformer_lib.map_pythia_module_names(module_name)
    except ValueError as e:
        assert 'hook' not in module_name, f"Error: {e}"
    else:
        assert 'hook' in module_name, module_name

#%%

dtype = torch.float32
model = HookedTransformer.from_pretrained_no_processing("EleutherAI/pythia-70m", torch_dtype=dtype)

#%%
# For low precision, the processing is not advised.
hf_model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-70m",
    torch_dtype=dtype,
).to(DEVICE)

def check_logits(margin):
    """Check the loading and inferences for different dtypes."""
    print(f"\nChecking EleutherAI/pythia-70m with dtype {dtype}")

    for layer_name, layer in model.state_dict().items():
        assert layer.dtype in [dtype, torch.bool] or "IGNORE" in layer_name, f"Layer {layer_name} has incorrect dtype {layer.dtype}"

    check_performance(model, hf_model, margin)

#%%

check_logits(margin=1e-9)

#%%

# Add no-op hooks and test the same thing
hook_names = model.hook_dict.keys()
no_op_hook = lambda x, hook: x * 1.0

try:
    for hook_name in hook_names:
        model.add_hook(hook_name, no_op_hook)

    check_logits(margin=1e-9)
finally:
    model.reset_hooks()
# It's still fine

#%%

