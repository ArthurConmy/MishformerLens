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
MODEL_NAME = "EleutherAI/pythia-70m"
# MODEL_NAME = "gpt2"

#%%

# Use the context manager to disable patching temporarily
with mishformer_lens.ast_patching_disabled():
    tl_model = TLHookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=DEVICE)
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
        MODEL_NAME,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    hf_raw_logits = hf_model(tokens)[0]

    hf_eager_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        attn_implementation='eager',
    ).to(DEVICE)
    hf_eager_raw_logits = hf_eager_model(tokens)[0]

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
    MODEL_NAME,
    torch_dtype=DTYPE,
    attn_implementation='eager',
)

#%%

logits, cache = model.run_with_cache(tokens)

#%%

# Only our implementation with eager attention is correct to default 1e-5 and 1e-6 ish torch assert_close defaults.
#
# Both TL and eager MFLens are correct to 1e-4 to the ground truth HF logits.
torch.testing.assert_close(logits, hf_raw_logits, atol=1e-4, rtol=1e-4)

#%%

# Check if the keys in cache are the same as in tl_cache
cache_keys = set(cache.keys())
tl_cache_keys = set(tl_cache.keys())
if cache_keys != tl_cache_keys:
    print("The cache keys are different.")
    print("Keys in cache but not in tl_cache:")
    print(cache_keys - tl_cache_keys)
    print("Keys in tl_cache but not in cache:")
    print(tl_cache_keys - cache_keys)
else:
    print("The cache keys are the same in both caches.")
# Print the number of keys in each cache
print(f"Number of keys in cache: {len(cache_keys)}")
print(f"Number of keys in tl_cache: {len(tl_cache_keys)}")

#%%

# Check if everything in each of the two caches is allclose.
# TODO(v0.1): investigate why the TL and our Pythia-70M cache appear to differ quite a bit?!
def check_caches_allclose(cache1, cache2, atol=0.005, rtol=0.006):
    all_close = True
    bads = []
    for key in set(cache1.keys()) & set(cache2.keys()):
        if "rot_" in key:
            continue
        if isinstance(cache1[key], torch.Tensor) and isinstance(cache2[key], torch.Tensor):
            try:
                if 'hook_attn_scores' in key:
                    mask = (cache1[key] < -1e7) & (cache2[key] < -1e7)
                    diff = torch.abs(cache1[key] - cache2[key])
                    diff[mask] = 0  # Set difference to 0 where both values are < -1e7
                    if torch.all(diff <= atol + rtol * torch.abs(cache2[key])):
                        print(f"Key '{key}' is close.")
                    else:
                        raise AssertionError("Tensors are not close")
                else:
                    torch.testing.assert_close(cache1[key], cache2[key], atol=atol, rtol=rtol)
                    print(f"Key '{key}' is close.")
            except AssertionError as e:
                all_close = False
                e.add_note(f"Key '{key}' is not close:")
                e.add_note(f"Max absolute difference: {torch.max(torch.abs(cache1[key] - cache2[key]))}")
                e.add_note(f"Mean absolute difference: {torch.mean(torch.abs(cache1[key] - cache2[key]))}")
                bads.append((key, str(e)))
        else:
            print(f"Key '{key}' is not a tensor, skipping comparison.")
    
    if all_close:
        print("All tensors in the caches are close.")
    else:
        print("Some tensors in the caches are not close.")

    return all_close, bads

# Run the check
print("Checking if caches are allclose:")
all_close, bads = check_caches_allclose(cache, tl_cache)
if all_close:
    print("All cache entries are close!!! 🎉🥳🚀")
else:
    print("The following cache entries are not close:")
    for key, error in bads:
        print(f"- {key}: {error}")

#%%

# If you want to check with different tolerance levels, you can do:
# print("\nChecking with stricter tolerance:")
# check_caches_allclose(cache, tl_cache, atol=1e-6, rtol=1e-6)

# print("\nChecking with looser tolerance:")
# check_caches_allclose(cache, tl_cache, atol=1e-4, rtol=1e-4)

#%%

torch.testing.assert_close(logits, hf_eager_raw_logits, atol=1e-10, rtol=1e-10)

# %%

try:
    torch.testing.assert_close(hf_raw_logits, hf_eager_raw_logits, atol=0.001, rtol=1e-6)
except AssertionError as e:
    print(f"Assertion failed, but this WAS expected: {e}")
else:
    raise AssertionError("HuggingFace and eager HuggingFace logits are close, this was actually unexpected (unless you chose GPT-2 Small!)")

# %%
