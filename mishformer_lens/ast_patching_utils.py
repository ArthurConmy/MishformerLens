import einops
from typing import Callable, Dict, Optional, Union
import torch
from beartype import beartype as typechecker
import jaxtyping as jt
from transformer_lens import hook_points

@jt.jaxtyped(typechecker=typechecker)
def layer_norm_scale(x: jt.Float[torch.Tensor, "batch pos length"], eps: float) -> jt.Float[torch.Tensor, "batch pos 1"]:
    centered_x = x - x.mean(-1, keepdim=True)  # [batch, pos, length]
    return (
        (centered_x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    )

def einops_rearrange_factory(
    einops_string: str,
    einops_kwargs: Optional[Dict[str, int]] = None
) -> hook_points.InputOrOutputCallable:

    if einops_kwargs is None:
        einops_kwargs = {}

    def einops_rearrange(x: Union[torch.Tensor, tuple[torch.Tensor, ...]]) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        if isinstance(x, tuple):
            return tuple(einops.rearrange(x_element, einops_string, **einops_kwargs) for x_element in x)
        else:
            return einops.rearrange(x, einops_string, **einops_kwargs)

    return einops_rearrange
