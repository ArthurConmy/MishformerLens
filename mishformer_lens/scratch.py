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

# Apply patches now:
patcher = ModuleASTPatcher(
    transformers.models.gpt2.modeling_gpt2,
    PatchSettings(
        prefix="""from transformer_lens.hook_points import HookPoint
""",
        allow_num_matches_upto=dict(
            # # o1 preview is wrong here:
            # GPT2MLP=1,
            # GPT2Model=1,
            # GPT2Block=12,  # Number of transformer blocks in GPT-2 Small
            # GPT2Attention=1,
            # GPT2LayerNorm=1,
        ),
    ),
    # Patching GPT2Model to add hooks for embeddings and final layer norm
    GPT2Model=[
        # Add hook points in __init__
        (
            """self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)""",
            """self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
self.hook_embed = HookPoint()
self.hook_pos_embed = HookPoint()
self.ln_f.hook_scale = HookPoint()
self.ln_f.hook_normalized = HookPoint()
""",
        ),
        # Wrap embeddings with hooks in forward
        (
            """inputs_embeds = self.wte(input_ids)""",
            """inputs_embeds = self.hook_embed(self.wte(input_ids))""",
        ),
        (
            """position_embeds = self.wpe(position_ids)""",
            """position_embeds = self.hook_pos_embed(self.wpe(position_ids))""",
        ),
        # Add hooks after final layer norm in forward
        (
            """hidden_states = self.ln_f(hidden_states)""",
            """hidden_states = self.ln_f.hook_normalized(self.ln_f(hidden_states))""",
        ),
    ],
    # Patching GPT2Block to add hooks for layer norms and residual connections
    GPT2Block=[
        # Add hook points in __init__
        (
            """self.mlp = GPT2MLP(inner_dim, config)""",
            """self.mlp = GPT2MLP(inner_dim, config)
self.hook_resid_pre = HookPoint()
self.hook_resid_mid = HookPoint()
self.hook_resid_post = HookPoint()
self.hook_mlp_in = HookPoint()
self.hook_mlp_out = HookPoint()
self.hook_attn_in = HookPoint()
self.hook_attn_out = HookPoint()
self.ln_1.hook_scale = HookPoint()
self.ln_1.hook_normalized = HookPoint()
self.ln_2.hook_scale = HookPoint()
self.ln_2.hook_normalized = HookPoint()
""",
        ),
        # Wrap hidden_states with hooks in forward
        (
            """residual = hidden_states""",
            """residual = hidden_states
hidden_states = self.hook_resid_pre(hidden_states)
""",
        ),
        (
            """hidden_states = self.ln_1(hidden_states)""",
            """hidden_states = self.ln_1.hook_normalized(self.ln_1(hidden_states))""",
        ),
        # Before self.attn
        (
            """attn_outputs = self.attn(""",
            """hidden_states = self.hook_attn_in(hidden_states)
attn_outputs = self.attn(""",
        ),
        # After self.attn
        (
            """attn_output = attn_outputs[0]""",
            """attn_output = attn_outputs[0]
attn_output = self.hook_attn_out(attn_output)
""",
        ),
        (
            """hidden_states = attn_output + residual""",
            """hidden_states = attn_output + residual
hidden_states = self.hook_resid_mid(hidden_states)
""",
        ),
        # Wrap MLP inputs and outputs
        (
            """residual = hidden_states""",
            """residual = hidden_states
hidden_states = self.hook_mlp_in(hidden_states)
""",
        ),
        (
            """hidden_states = self.ln_2(hidden_states)""",
            """hidden_states = self.ln_2.hook_normalized(self.ln_2(hidden_states))""",
        ),
        (
            """feed_forward_hidden_states = self.mlp(hidden_states)""",
            """feed_forward_hidden_states = self.mlp(hidden_states)
feed_forward_hidden_states = self.hook_mlp_out(feed_forward_hidden_states)
""",
        ),
        (
            """hidden_states = residual + feed_forward_hidden_states""",
            """hidden_states = residual + feed_forward_hidden_states
hidden_states = self.hook_resid_post(hidden_states)
""",
        ),
    ],
    # Patching GPT2Attention to add hooks for attention components
    GPT2Attention=[
        # Add hook points in __init__
        (
            """self.c_proj = Conv1D(self.embed_dim, self.embed_dim)""",
            """self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
self.hook_q = HookPoint()
self.hook_k = HookPoint()
self.hook_v = HookPoint()
self.hook_z = HookPoint()
self.hook_attn_scores = HookPoint()
self.hook_pattern = HookPoint()
self.hook_result = HookPoint()
""",
        ),
        # In forward, wrap q, k, v with hooks
        (
            """query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)""",
            """query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
query = self.hook_q(query)
key = self.hook_k(key)
value = self.hook_v(value)
""",
        ),
        # Wrap attention weights and outputs
        (
            """attn_weights = torch.matmul(query, key.transpose(-1, -2))""",
            """attn_weights = torch.matmul(query, key.transpose(-1, -2))
attn_weights = self.hook_attn_scores(attn_weights)
""",
        ),
        (
            """attn_weights = nn.functional.softmax(attn_weights, dim=-1)""",
            """attn_weights = nn.functional.softmax(attn_weights, dim=-1)
attn_weights = self.hook_pattern(attn_weights)
""",
        ),
        (
            """attn_output = torch.matmul(attn_weights, value)""",
            """attn_output = torch.matmul(attn_weights, value)
attn_output = self.hook_z(attn_output)
""",
        ),
    ],
    # Patching GPT2MLP to add hook_post
    GPT2MLP=[
        # Add hook_post in __init__
        (
            """self.dropout = nn.Dropout(config.resid_pdrop)""",
            """self.dropout = nn.Dropout(config.resid_pdrop)
self.hook_post = HookPoint()
""",
        ),
        # Wrap MLP output with hook_post in forward
        (
            """hidden_states = self.dropout(hidden_states)""",
            """hidden_states = self.dropout(hidden_states)
hidden_states = self.hook_post(hidden_states)
""",
        ),
    ],
)()

patcher.__enter__()

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

class HookedTransformer(TransformerLensHookedTransformer):
    def __init__(
        self,
        # TODO(v1): ensure these are all supported, or error:
        ast_patched_hf_model: transformers.AutoModelForCausalLM,
        cfg: Union[HookedTransformerConfig, dict],
        tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None,
        default_padding_side: Literal["left", "right"] = "right",
    ):
        super().__init__(
            cfg=cfg,
            tokenizer=tokenizer,
            move_to_device=False,
            initialize_modules=False,
            default_padding_side=default_padding_side
        )
        self._ast_patched_hf_model = ast_patched_hf_model
        self.setup()  # N.B. there's an internal TransformerLens .setup() already called, that does nothing

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        refactor_factored_attn_matrices: bool = False,
        checkpoint_index: Optional[int] = None,
        checkpoint_value: Optional[int] = None,
        hf_model: Optional[transformers.AutoModelForCausalLM] = None,
        device: Optional[Union[str, torch.device]] = None,
        n_devices: int = 1,
        tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None,
        move_to_device: bool = True,
        fold_value_biases: bool = True,
        default_prepend_bos: bool = True,
        default_padding_side: Literal["left", "right"] = "right",
        dtype="float32",
        **from_pretrained_kwargs,
    ) -> "HookedTransformer":
        assert hf_model is None, "You cannot pass hf_model to MishformerLens, we literally wrap HuggingFace"

        assert not fold_ln, "TODO(v1): layer norm folding with MishformerLens"
        assert not center_unembed, "TODO(v1): center_unembed"
        assert not center_writing_weights, "TODO(v1): center_writing_weights"
        assert not fold_value_biases, "TODO(v1): fold_value_biases"

        assert not refactor_factored_attn_matrices, "TODO: refactor_factored_attn_matrices"

        official_model_name = loading_from_pretrained.get_official_model_name(model_name)
        # TODO(v0): document/extend to all models, not just AutoModelForCausalLM
        ast_patched_hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            official_model_name,
            # TODO(v1): support Pythia checkpoint numbers
        )

        # TODO(v1): maybe stop or reduce this copy+paste from transformer_lens
        # BEGIN COPY PASTE
        hf_cfg = ast_patched_hf_model.config.to_dict()
        qc = hf_cfg.get("quantization_config", {})
        load_in_4bit = qc.get("load_in_4bit", False)
        load_in_8bit = qc.get("load_in_8bit", False)
        quant_method = qc.get("quant_method", "")
        assert not load_in_8bit, "8-bit quantization is not supported"
        assert not (
            load_in_4bit and (version.parse(torch.__version__) < version.parse("2.1.1"))
        ), "Quantization is only supported for torch versions >= 2.1.1"
        assert not (
            load_in_4bit and ("llama" not in model_name.lower())
        ), "Quantization is only supported for Llama models"
        if load_in_4bit:
            assert (
                qc.get("quant_method", "") == "bitsandbytes"
            ), "Only bitsandbytes quantization is supported"
        if isinstance(dtype, str):
            # Convert from string to a torch dtype:
            dtype = DTYPE_FROM_STRING[dtype]
        # END COPY PASTE

        cfg = loading_from_pretrained.get_pretrained_model_config(
            official_model_name,
            hf_cfg=hf_cfg,
            checkpoint_index=checkpoint_index,
            checkpoint_value=checkpoint_value,
            fold_ln=fold_ln,
            device=device,
            n_devices=n_devices,
            default_prepend_bos=default_prepend_bos,
            dtype=dtype,
            **from_pretrained_kwargs,
        )
        if move_to_device:
            # TODO(v0) or TODO(v1): does this need be more mem efficient?
            ast_patched_hf_model = ast_patched_hf_model.to(cfg.device)

        # TODO(v0): add all these checks!
        assert not cfg.use_attn_result, "TODO(v1): add attn_result support"

        return cls(
            ast_patched_hf_model=ast_patched_hf_model,
            cfg=cfg,
            tokenizer=tokenizer,
            default_padding_side=default_padding_side,
        )

    def check_hooks_to_add(
        self,
        hook_point,
        hook_point_name,
        hook,
        dir="fwd",
        is_permanent=False,
        prepend=False,
    ) -> None:
        pass  # TODO(v1): implement

    def input_to_embed(self, *args, **kwargs):
        raise RuntimeError("input_to_embed should not be called in MishformerLens")

    # TODO(v1): review all those @typing.overloads...
    def forward(
        self,
        input: Union[
            str,
            list[str],
            Int[torch.Tensor, "batch pos"],
            Float[torch.Tensor, "batch pos d_model"],
        ],
        return_type: Optional[str] = "logits",
        loss_per_token: bool = False,  # TODO(v1): upstream refactor of this
        prepend_bos: Optional[Union[bool, None]] = transformer_lens_utils.USE_DEFAULT_VALUE,
        padding_side: Optional[Literal["left", "right"]] = transformer_lens_utils.USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,  # TODO(v0): WTF is this?
        shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [batch pos]
        stop_at_layer: Optional[int] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Union[
        None,
        Float[torch.Tensor, "batch pos d_vocab"],
        Loss,
        tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
    ]:
        assert start_at_layer is None and stop_at_layer is None, "MishformerLens does not support start and stop at layer"
        assert attention_mask is None, "MishformerLens does not support attention_mask passed to forward(...)"
        assert past_kv_cache is None, "MishformerLens does not support past_kv_cache passed to forward(...)"
        assert shortformer_pos_embed is None, "MishformerLens does not support shortformer_pos_embed passed to forward(...)"

        with transformer_lens_utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            if isinstance(input, str) or isinstance(input, list):
                # If text, convert to tokens (batch_size=1)
                assert (
                    self.tokenizer is not None
                ), "Must provide a tokenizer if passing a string to the model"
                # This is only intended to support passing in a single string
                tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
            else:
                tokens = input
            if len(tokens.shape) == 1:
                # If tokens are a rank 1 tensor, add a dummy batch dimension to avoid things breaking.
                tokens = tokens[None]

            if tokens.device.type != self.cfg.device:
                # TODO(v1): TransformerLens does wacky
                # .to(devices.get_device_for_block_index(0, self.cfg))
                # stuff here. Why? Can HF Transformers just avoid this?
                tokens = tokens.to(self.cfg.device)

            assert type(tokens) == torch.Tensor, (type(tokens), tokens)
            logits = self._ast_patched_hf_model.forward(tokens).logits

            if return_type == "logits":
                return logits
            else:
                assert (
                    tokens is not None
                ), "tokens must be passed in if return_type is 'loss' or 'both'"
                loss = self.loss_fn(logits, tokens, attention_mask, per_token=loss_per_token)
                if return_type == "loss":
                    return loss
                elif return_type == "both":
                    return Output(logits, loss)
                else:
                    logging.warning(f"Invalid return_type passed in: {return_type}")
                    return None

    # def run_with_cache(
    #     self, *model_args, return_cache_object=True, remove_batch_dim=False, **kwargs
    # ) -> tuple[
    #     Union[
    #         None,
    #         Float[torch.Tensor, "batch pos d_vocab"],
    #         Loss,
    #         tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
    #     ],
    #     Union[ActivationCache, dict[str, torch.Tensor]],
    # ]:
    #     pass

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
    device=DEVICE
)

#%%

# Run model:
output = model("Hey")

#%%

def yapper_and_editor(z, hook):
    print('yap')
    z*=2
    return z

logits1 = model("Hey")

logits2 = model.run_with_hooks(
    "Hey",
    fwd_hooks=[
        ('_ast_patched_hf_model.transformer.h.0.mlp.hook_pre', yapper_and_editor)
    ]
)

logits1, logits2 # these sure look different...

#%%

