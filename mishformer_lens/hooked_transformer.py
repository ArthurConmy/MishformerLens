import einops
import transformers
import logging
from typing import Optional, Union, Literal, NamedTuple
from packaging import version
import torch
import jaxtyping
from jaxtyping import Float, Int

from transformer_lens.utilities.addmm import batch_addmm
from transformer_lens import hook_points
from transformer_lens import HookedTransformerConfig
from transformer_lens import HookedTransformer as TransformerLensHookedTransformer
from transformer_lens import loading_from_pretrained
from transformer_lens import utils as transformer_lens_utils
from transformer_lens import ActivationCache
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

# TODO(v1): remove copy paste from TransformerLens
# we have to because HookedTransformer is the name of both the class and the module........ in TransformerLens
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

def map_gpt2_module_names(name: str) -> str:
    if name == '_ast_patched_hf_model.transformer.hook_embed':
        mapped_name = 'hook_embed'
    elif name == '_ast_patched_hf_model.transformer.hook_pos_embed':
        mapped_name = 'hook_pos_embed'
    elif name.startswith('_ast_patched_hf_model.transformer.ln_f'):
        mapped_name = 'ln_final' + name[len('_ast_patched_hf_model.transformer.ln_f'):]
    elif name.startswith('_ast_patched_hf_model.transformer.h.'):
        rest = name[len('_ast_patched_hf_model.transformer.h.'):]
        mapped_name = 'blocks.' + rest
    else:
        raise ValueError(f"Cannot map name '{name}'")
    
    # Apply substitutions and return
    return mapped_name.replace('ln_1', 'ln1').replace('ln_2', 'ln2')    

# TODO(v0.1): frankenstein of model commits; refactor this
def map_pythia_module_names(name: str) -> str:
    if not name.startswith('_ast_patched_hf_model.'):
        raise ValueError(f"Cannot map name '{name}'")
    name = name[len('_ast_patched_hf_model.'):]

    if name == '' or name == 'gpt_neox' or name == 'embed_out':
        raise ValueError(f"Cannot map name '{name}'")

    if name == 'gpt_neox.hook_embed':
        return 'hook_embed'
    
    if name.startswith('gpt_neox.layers.'):
        parts = name.split('.')
        layer_num = parts[2]
        rest = '.'.join(parts[3:])

        if 'hook' in rest:
            if rest == 'input_layernorm.hook_normalized':
                return f'blocks.{layer_num}.ln1.hook_normalized'
            elif rest == 'post_attention_layernorm.hook_normalized':
                return f'blocks.{layer_num}.ln2.hook_normalized'
            elif rest.startswith('attention.hook_'):
                attn_hook = rest.split('hook_')[1]
                return f'blocks.{layer_num}.attn.hook_{attn_hook}'
            elif rest == 'mlp.hook_pre':
                return f'blocks.{layer_num}.mlp.hook_pre'
            elif rest == 'mlp.hook_post':
                return f'blocks.{layer_num}.mlp.hook_post'
            elif rest.startswith('hook_'):
                return f'blocks.{layer_num}.{rest}'
            elif rest == 'input_layernorm.hook_scale':
                return f'blocks.{layer_num}.ln1.hook_scale'
            elif rest == 'post_attention_layernorm.hook_scale':
                return f'blocks.{layer_num}.ln2.hook_scale'
        else:
            raise ValueError(f"Cannot map name '{name}' (no 'hook' substring)")

    if 'hook' in name and name.startswith('gpt_neox.final_layer_norm'):
        if name == 'gpt_neox.final_layer_norm.hook_normalized':
            return 'ln_final.hook_normalized'
        elif name == 'gpt_neox.final_layer_norm.hook_scale':
            return 'ln_final.hook_scale'

    raise ValueError(f"Cannot map name '{name}'")

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

        logging.warning("TODO(v1): MishformerLens sets set_use_hook_mlp_in==True always: it is set on for this model.")

        # N.B. there's an internal TransformerLens .setup() already called, that does nothing
        if self.cfg.model_name.startswith('pythia'):
            naming_transformation = map_pythia_module_names
        elif self.cfg.model_name.startswith('gpt2'):
            naming_transformation = map_gpt2_module_names
        else:
            raise ValueError(f"Model name {self.cfg.model_name} not supported")
        self.setup(naming_transformation=naming_transformation)

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
            **from_pretrained_kwargs,  # TODO(v1): we pass this to config too, sort out
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
        tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,  # TODO(v0.1): WTF is this?
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
        assert past_kv_cache is None, "TODO(v0.1): MishformerLens does not support past_kv_cache passed to forward(...)"
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

    @property
    def W_U(self) -> Float[torch.Tensor, "d_model d_vocab"]:
        # TODO(v0.1): this will be different for non-GPT-2 models
        return self._ast_patched_hf_model.transformer.wte.weight.T

    @property
    def b_U(self) -> Float[torch.Tensor, "d_vocab"]:
        # TODO(v0.1): this will be different for non-GPT-2 models
        return torch.zeros_like(self.W_U[0])  # lol

    @property
    def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
        """Stack the attn output weights across all layers."""
        return torch.stack([einops.rearrange(
            self._ast_patched_hf_model.transformer.h[i].attn.c_proj.weight,
            "(heads d_head) d_model -> heads d_head d_model", heads=self.cfg.n_heads
        ) for i in range(self.cfg.n_layers)], dim=0)

    def unembed(self, residual: Float[torch.Tensor, "batch pos d_model"]) -> Float[torch.Tensor, "batch pos d_vocab"]:
        return batch_addmm(self.b_U, self.W_U, residual)
