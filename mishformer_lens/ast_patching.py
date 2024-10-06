import transformers
import inspect
from mishax import ast_patcher

PATCHERS: dict[str, ast_patcher.ModuleASTPatcher] = {}

def register_patcher(patcher: ast_patcher.ModuleASTPatcher):
    patcher_name = patcher.module.__name__
    if patcher_name in PATCHERS:
        raise ValueError(f"Patcher {patcher_name} already registered")
    PATCHERS[patcher_name] = patcher
    return patcher

# TODO(v1): split off these into different files
register_patcher(ast_patcher.ModuleASTPatcher(
    transformers.models.gpt2.modeling_gpt2,
    ast_patcher.PatchSettings(
        prefix="""from transformer_lens.hook_points import HookPoint
from mishformer_lens.ast_patching_utils import layer_norm_scale, einops_rearrange_factory
import einops
""",
        allow_num_matches_upto=dict(
            # # o1 preview is wrong here:
            # GPT2MLP=1,
            # GPT2Model=1,
            # GPT2LayerNorm=1,
            GPT2Attention=2,
        ),
    ),
    # Patching GPT2Model to add hooks for embeddings and final layer norm
    GPT2Model=[
        # Add hook points in __init__
        #
        # TODO(v1): add support for editing LayerNorm scale; we can't currently as nn.LayerNorm does not expose this
        (
            """self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)""",
            """self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
self.hook_embed = HookPoint()
self.hook_pos_embed = HookPoint()
self.ln_f.hook_normalized = HookPoint()
self.ln_f.hook_scale = HookPoint()
""",
        ),
        # Wrap embeddings with hooks in forward
        (
            """inputs_embeds = self.wte(input_ids)""",
            """inputs_embeds = self.hook_embed(self.wte(input_ids))""",
        ),
        (
            """position_embeds = self.wpe(position_ids)""",
            """position_embeds = self.hook_pos_embed(einops.repeat(self.wpe(position_ids), "1 pos d_model -> batch pos d_model", batch=inputs_embeds.shape[0]))""",
        ),
        # Add hooks after final layer norm in forward
        # TODO(v0.1): also add scale hooks for the LNs inside the model
        (
            """hidden_states = self.ln_f(hidden_states)""",
            """manually_computed_layer_norm_scale = layer_norm_scale(hidden_states, self.ln_f.eps)
should_be_no_op_layer_norm_scale = self.ln_f.hook_scale(manually_computed_layer_norm_scale)
torch.testing.assert_close(should_be_no_op_layer_norm_scale, manually_computed_layer_norm_scale, atol=1e-3, rtol=1e-3)  # N.B. writing to this hook point is a no-op, so we assert that their's <1e-3 error  # TODO(v1): maybe add a disable for this? Or ideally better solution in general; we could allow the below to be multiplied by (true_scale / our_scale)
hidden_states = self.ln_f.hook_normalized(self.ln_f(hidden_states))""",
        ),
    ],
    GPT2SdpaAttention=[
        # Add hook points in __init__
        (
            """super().__init__(*args, **kwargs)""",
            """super().__init__(*args, **kwargs)
self.hook_q = HookPoint(input_callable = einops_rearrange_factory("batch head_index pos d_head -> batch pos head_index d_head"), output_callable=einops_rearrange_factory("batch pos head_index d_head -> batch head_index pos d_head"))
self.hook_k = HookPoint(input_callable = einops_rearrange_factory("batch head_index pos d_head -> batch pos head_index d_head"), output_callable=einops_rearrange_factory("batch pos head_index d_head -> batch head_index pos d_head"))
self.hook_v = HookPoint(input_callable = einops_rearrange_factory("batch head_index pos d_head -> batch pos head_index d_head"), output_callable=einops_rearrange_factory("batch pos head_index d_head -> batch head_index pos d_head"))
self.hook_z = HookPoint(input_callable = einops_rearrange_factory("batch head_index pos d_head -> batch pos head_index d_head"), output_callable=einops_rearrange_factory("batch pos head_index d_head -> batch head_index pos d_head"))
# # Not supported here :'(
# self.hook_attn_scores = HookPoint()
# self.hook_pattern = HookPoint()
self.hook_result = HookPoint()
""",
        ),
        # In forward, wrap q, k, v with hooks
        (
            """query = self._split_heads(query, self.num_heads, self.head_dim)
key = self._split_heads(key, self.num_heads, self.head_dim)
value = self._split_heads(value, self.num_heads, self.head_dim)""",
            """query = self.hook_q(self._split_heads(query, self.num_heads, self.head_dim))
key = self.hook_k(self._split_heads(key, self.num_heads, self.head_dim))
value = self.hook_v(self._split_heads(value, self.num_heads, self.head_dim))
""",
        ),
        # Hook after attention output
        (
            """attn_output = attn_output.transpose(1, 2).contiguous()""",
            """attn_output = attn_output.transpose(1, 2).contiguous()
attn_output = self.hook_z(attn_output)""",
        ),
    ],
    # Patching GPT2Attention to add hooks for attention components
    GPT2Attention=[
        # Add hook points in __init__
        (
            """self.c_proj = Conv1D(self.embed_dim, self.embed_dim)""",
            """self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
self.hook_q = HookPoint(input_callable = einops_rearrange_factory("batch head_index pos d_head -> batch pos head_index d_head"), output_callable=einops_rearrange_factory("batch pos head_index d_head -> batch head_index pos d_head"))
self.hook_k = HookPoint(input_callable = einops_rearrange_factory("batch head_index pos d_head -> batch pos head_index d_head"), output_callable=einops_rearrange_factory("batch pos head_index d_head -> batch head_index pos d_head"))
self.hook_v = HookPoint(input_callable = einops_rearrange_factory("batch head_index pos d_head -> batch pos head_index d_head"), output_callable=einops_rearrange_factory("batch pos head_index d_head -> batch head_index pos d_head"))
self.hook_z = HookPoint(input_callable = einops_rearrange_factory("batch head_index pos d_head -> batch pos head_index d_head"), output_callable=einops_rearrange_factory("batch pos head_index d_head -> batch head_index pos d_head"))
self.hook_attn_scores = HookPoint()
self.hook_pattern = HookPoint()
self.hook_result = HookPoint()
""",
        ),
        # In forward, wrap q, k, v with hooks
        (
            """query = self._split_heads(query, self.num_heads, self.head_dim)
key = self._split_heads(key, self.num_heads, self.head_dim)
value = self._split_heads(value, self.num_heads, self.head_dim)""",
            """query = self.hook_q(self._split_heads(query, self.num_heads, self.head_dim))
key = self.hook_k(self._split_heads(key, self.num_heads, self.head_dim))
value = self.hook_v(self._split_heads(value, self.num_heads, self.head_dim))
""",
        ),
        # Wrap attention scores and patterns
        (
            """attn_weights = nn.functional.softmax(attn_weights, dim=-1)""",
            """attn_weights = self.hook_pattern(nn.functional.softmax(self.hook_attn_scores(attn_weights), dim=-1))""",
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
        # Add hook_post and hook_pre in __init__
        (
            """self.dropout = nn.Dropout(config.resid_pdrop)""",
            """self.dropout = nn.Dropout(config.resid_pdrop)
self.hook_pre = HookPoint()
self.hook_post = HookPoint()
""",
        ),
        # Wrap act fn with hook_post and hook_pre
        (
            """hidden_states = self.act(hidden_states)""",
            """hidden_states = self.hook_post(self.act(self.hook_pre(hidden_states)))""",
        ),
    ],
    # Patching GPT2FlashAttention2 to add hooks
    # Patching GPT2FlashAttention2 to add hooks
    GPT2FlashAttention2=[
        # Add hook points in __init__
        (
            """super().__init__(*args, **kwargs)""",
            """super().__init__(*args, **kwargs)
self.hook_q = HookPoint(input_callable = einops_rearrange_factory("batch head_index pos d_head -> batch pos head_index d_head"), output_callable=einops_rearrange_factory("batch pos head_index d_head -> batch head_index pos d_head"))
self.hook_k = HookPoint(input_callable = einops_rearrange_factory("batch head_index pos d_head -> batch pos head_index d_head"), output_callable=einops_rearrange_factory("batch pos head_index d_head -> batch head_index pos d_head"))
self.hook_v = HookPoint(input_callable = einops_rearrange_factory("batch head_index pos d_head -> batch pos head_index d_head"), output_callable=einops_rearrange_factory("batch pos head_index d_head -> batch head_index pos d_head"))
self.hook_z = HookPoint(input_callable = einops_rearrange_factory("batch head_index pos d_head -> batch pos head_index d_head"), output_callable=einops_rearrange_factory("batch pos head_index d_head -> batch head_index pos d_head"))
# N.B. these are likely not supported:
self.hook_attn_scores = HookPoint()
self.hook_pattern = HookPoint()
# TODO(v1): add support for attn_result:
self.hook_result = HookPoint()
""",
        ),
        # In forward, wrap q, k, v with hooks
        (
            """query = self._split_heads(query, self.num_heads, self.head_dim)
key = self._split_heads(key, self.num_heads, self.head_dim)
value = self._split_heads(value, self.num_heads, self.head_dim)""",
            """query = self.hook_q(self._split_heads(query, self.num_heads, self.head_dim))
key = self.hook_k(self._split_heads(key, self.num_heads, self.head_dim))
value = self.hook_v(self._split_heads(value, self.num_heads, self.head_dim))
""",
        ),
        # Hook after attention output
        (
            """attn_weights_reshaped = attn_output.reshape(bsz, query_length, self.num_heads * self.head_dim)""",
            """attn_output = self.hook_z(attn_output)
attn_weights_reshaped = attn_output.reshape(bsz, query_length, self.num_heads * self.head_dim)
attn_output = self.hook_z(attn_output)
""",
        ),
    ],
    # Patching GPT2Block to add hooks for layer norms and residual connections
    GPT2Block=[
        # Add hook points in __init__
        # TODO(v1): add support for LayerNorm scale; we can't currently as nn.LayerNorm does not expose this
        (
            """self.mlp = GPT2MLP(inner_dim, config)""",
            """self.mlp = GPT2MLP(inner_dim, config)
self.hook_resid_pre = HookPoint()
self.hook_resid_mid = HookPoint()
self.hook_resid_post = HookPoint()
self.hook_mlp_in = HookPoint()
self.hook_mlp_out = HookPoint()
self.hook_attn_out = HookPoint()
self.ln_1.hook_normalized = HookPoint()
self.ln_2.hook_normalized = HookPoint()
""",
        ),
        # N.B. Mishax does not patch class level dicts, so we need to rip this one out otherwise attention is not instrumented.
        ("""attention_class = GPT2_ATTENTION_CLASSES[config._attn_implementation]""",
         """if config._attn_implementation == "eager":
    attention_class = GPT2Attention
elif config._attn_implementation == "flash_attention_2":
    warnings.warn("MishformerLens does not have tested support for FlashAttention2, use with caution! Pass `attn_implementation='eager'` to MishformerLens' HookedTransformer.from_pretrained to avoid flash attention.")
    attention_class = GPT2FlashAttention2
elif config._attn_implementation == "sdpa":
    warnings.warn("MishformerLens may not expose patterns for SDPA attention, use with caution! Pass `attn_implementation='eager'` to MishformerLens' HookedTransformer.from_pretrained to avoid SDPA attention.")
    attention_class = GPT2SdpaAttention
else:
    raise ValueError(f"Unknown attention implementation: {{config._attn_implementation}}")"""),
        # Wrap hidden_states with hooks in forward
        (
            """residual = hidden_states
hidden_states = self.ln_1(hidden_states)""",
            """residual = self.hook_resid_pre(hidden_states)
hidden_states = self.ln_1(hidden_states)""",
        ),
        (
            """hidden_states = self.ln_1(hidden_states)""",
            """hidden_states = self.ln_1.hook_normalized(self.ln_1(hidden_states))""",
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
            """residual = hidden_states
hidden_states = self.ln_2(hidden_states)""",
            """residual = self.hook_mlp_in(hidden_states)
hidden_states = self.ln_2(hidden_states)""",
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
))


register_patcher(ast_patcher.ModuleASTPatcher(
    transformers.models.gpt_neox.modeling_gpt_neox,
    ast_patcher.PatchSettings(
        prefix="""from transformer_lens.hook_points import HookPoint
from mishformer_lens.ast_patching_utils import layer_norm_scale, einops_rearrange_factory
import einops
import warnings
""",
        allow_num_matches_upto=dict(),
    ),
    GPTNeoXModel=[
        # Add hook points in __init__
        (
            """self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)""",
            """self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
self.hook_embed = HookPoint()
self.final_layer_norm.hook_normalized = HookPoint()
self.final_layer_norm.hook_scale = HookPoint()
""",
        ),
        # Wrap embeddings with hooks in forward
        (
            """inputs_embeds = self.embed_in(input_ids)""",
            """inputs_embeds = self.hook_embed(self.embed_in(input_ids))""",
        ),
        # Wrap final layer norm with hooks
        (
            """hidden_states = self.final_layer_norm(hidden_states)""",
            """manually_computed_layer_norm_scale = layer_norm_scale(hidden_states, self.final_layer_norm.eps)
should_be_no_op_layer_norm_scale = self.final_layer_norm.hook_scale(manually_computed_layer_norm_scale)
torch.testing.assert_close(should_be_no_op_layer_norm_scale, manually_computed_layer_norm_scale, atol=1e-3, rtol=1e-3)
hidden_states = self.final_layer_norm.hook_normalized(self.final_layer_norm(hidden_states))""",
        ),
    ],
    GPTNeoXAttention=[
        # Add hook points in __init__
        (
            """self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)""",
            """self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
self.hook_q = HookPoint(input_callable=einops_rearrange_factory("batch head pos d_head -> batch pos head d_head"), output_callable=einops_rearrange_factory("batch pos head d_head -> batch head pos d_head"))
self.hook_k = HookPoint(input_callable=einops_rearrange_factory("batch head pos d_head -> batch pos head d_head"), output_callable=einops_rearrange_factory("batch pos head d_head -> batch head pos d_head"))
self.hook_v = HookPoint(input_callable=einops_rearrange_factory("batch head pos d_head -> batch pos head d_head"), output_callable=einops_rearrange_factory("batch pos head d_head -> batch head pos d_head"))
self.hook_z = HookPoint(input_callable=einops_rearrange_factory("batch head pos d_head -> batch pos head d_head"), output_callable=einops_rearrange_factory("batch pos head d_head -> batch head pos d_head"))
self.hook_attn_scores = HookPoint()
self.hook_pattern = HookPoint()
self.hook_result = HookPoint()
self.hook_rot_q = HookPoint()
self.hook_rot_k = HookPoint()
""",
        ),
        # In forward, wrap q, k, v with hooks
        (
            """query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)""",
            """query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)
query = self.hook_q(query)
key = self.hook_k(key)
value = self.hook_v(value)
""",
        ),
        # Hook rotary embeddings
        (
            """query = torch.cat((query, query_pass), dim=-1)""",
            """query = self.hook_rot_q(torch.cat((query, query_pass), dim=-1))""",
        ),
        (
            """key = torch.cat((key, key_pass), dim=-1)""",
            """key = self.hook_rot_k(torch.cat((key, key_pass), dim=-1))""",
        ),
        # Hook attention score and pattern hooking
        (
            """attn_weights = nn.functional.softmax(attn_scores, dim=-1)""",
            """attn_scores = self.hook_attn_scores(attn_scores)
attn_weights = self.hook_pattern(nn.functional.softmax(attn_scores, dim=-1))""",
        ),
        # Hook after attention output
        (
            """attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)""",
            """attn_output = self._merge_heads(self.hook_z(attn_output), self.num_attention_heads, self.head_size)""",
        ),
    ],
    GPTNeoXMLP=[
        # Add hook_post and hook_pre in __init__
        (
            """self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)""",
            """self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
self.hook_pre = HookPoint()
self.hook_post = HookPoint()
""",
        ),
        # Wrap activation function with hooks
        (
            """hidden_states = self.act(hidden_states)""",
            """hidden_states = self.hook_post(self.act(self.hook_pre(hidden_states)))""",
        ),
    ],
    GPTNeoXLayer=[
        # Add the new patch for attention class selection
        (
            """self.attention = GPT_NEOX_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)""",
            """if config._attn_implementation == "eager":
    attention_class = GPTNeoXAttention
elif config._attn_implementation == "flash_attention_2":
    warnings.warn("MishformerLens does not have tested support for FlashAttention2, use with caution! Pass `attn_implementation='eager'` to MishformerLens' HookedTransformer.from_pretrained to avoid flash attention.")
    attention_class = GPTNeoXFlashAttention2
elif config._attn_implementation == "sdpa":
    warnings.warn("MishformerLens may not expose patterns for SDPA attention, use with caution! Pass `attn_implementation='eager'` to MishformerLens' HookedTransformer.from_pretrained to avoid SDPA attention.")
    attention_class = GPTNeoXSdpaAttention
else:
    raise ValueError(f"Unknown attention implementation: {config._attn_implementation}")
self.attention = attention_class(config, layer_idx)""",
        ),
        # Add hook points in __init__
        (
            """self.mlp = GPTNeoXMLP(config)""",
            """self.mlp = GPTNeoXMLP(config)
self.hook_resid_pre = HookPoint()
# self.hook_resid_mid = HookPoint()
self.hook_resid_post = HookPoint()
self.hook_mlp_in = HookPoint()
self.hook_mlp_out = HookPoint()
self.hook_attn_out = HookPoint()
self.input_layernorm.hook_normalized = HookPoint()
self.post_attention_layernorm.hook_normalized = HookPoint()
""",
        ),
        # Wrap hidden_states with hooks in forward
        (
            """attention_layer_outputs = self.attention(
                self.input_layernorm(hidden_states),
                attention_mask=attention_mask,
                position_ids=position_ids,
                layer_past=layer_past,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )""",
            """hidden_states = self.hook_resid_pre(hidden_states)
attention_layer_outputs = self.attention(
    self.input_layernorm.hook_normalized(self.input_layernorm(hidden_states)),
    attention_mask=attention_mask,
    position_ids=position_ids,
    layer_past=layer_past,
    head_mask=head_mask,
    use_cache=use_cache,
    output_attentions=output_attentions,
    cache_position=cache_position,
    position_embeddings=position_embeddings,
)""",
        ),
        # After self.attention
        (
            """attn_output = attention_layer_outputs[0]""",
            """attn_output = attention_layer_outputs[0]
attn_output = self.hook_attn_out(attn_output)""",
        ),
        # Wrap MLP inputs and outputs
        (
            """mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))""",
            """mlp_output = self.mlp(self.post_attention_layernorm.hook_normalized(self.post_attention_layernorm(self.hook_mlp_in(hidden_states))))
mlp_output = self.hook_mlp_out(mlp_output)""",
        ),
        (
            """mlp_output = self.post_mlp_dropout(mlp_output)
hidden_states = mlp_output + attn_output + hidden_states""",
            """mlp_output = self.post_mlp_dropout(mlp_output)
hidden_states = mlp_output + attn_output + hidden_states
hidden_states = self.hook_resid_post(hidden_states)""",
        ),
        (
            """mlp_output = self.post_mlp_dropout(mlp_output)
hidden_states = mlp_output + attn_output""",
            """mlp_output = self.post_mlp_dropout(mlp_output)
hidden_states = mlp_output + attn_output
hidden_states = self.hook_resid_post(hidden_states)""",
        ),
    ],
))
