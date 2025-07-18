import torch
import random
from typing import List, Tuple, Callable, Dict
from jaxtyping import Float, Int
from torch import Tensor
import contextlib
import functools
from tqdm import tqdm
import einops

def get_sae_fwd_pre_hook(sae, reconstruct_bos_token: bool = False):
    def hook_fn(module, input, input_ids: Int[Tensor, "batch_size seq_len"]=None, attention_mask: Int[Tensor, "batch_size seq_len"]=None):
        nonlocal sae, reconstruct_bos_token

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        dtype = activation.dtype
        batch_size, seq_pos, d_model = activation.shape

        # This is needed when running Eleuther's base model SAEs
        # reshaped_activation = einops.rearrange(activation, "b s d -> (b s) d")
        # reconstructed_activation = sae(reshaped_activation).sae_out.to(dtype)
        # reconstructed_activation = einops.rearrange(reconstructed_activation, "(b s) d -> b s d", b=batch_size, s=seq_pos)

        reconstructed_activation = sae(activation.to(sae.dtype)).to(dtype)

        if not reconstruct_bos_token:
            if attention_mask is not None:
                # We don't want to reconstruct at the first sequence token (<|begin_of_text|>)
                bos_token_positions: Int[Tensor, 'batch_size'] = (attention_mask == 0).sum(dim=1)
                reconstructed_activation[:, bos_token_positions, :] = activation[:, bos_token_positions, :]
            elif seq_pos > 1:
                # we assume that the first token is always the <|begin_of_text|> token in case
                # the prompt contains multiple sequence positions (if seq_pos == 1 we're probably generating)
                reconstructed_activation[:, 0, :] = activation[:, 0, :]

        if isinstance(input, tuple):
            return (reconstructed_activation, *input[1:])
        else:
            return reconstructed_activation
    return hook_fn


def get_sae_hooks(model_block_modules: List[torch.nn.Module], sae_dict, reconstruct_bos_token: bool = False):
    """
    Get the hooks for the SAE layers.

    args:
        model_block_modules: List[nn.Module]: the model block modules to hook
        sae_dict: Dict[str, Sae]: the SAE layers. The keys in the dictionary have the format 'layer_<layer_number>'.
        reconstruct_bos_token: bool: whether to reconstruct the <|begin_of_text|> token
    """

    fwd_hooks = []

    fwd_pre_hooks = [
        (
            model_block_modules[layer], 
            get_sae_fwd_pre_hook(sae=sae_dict[f"layer_{layer}"], reconstruct_bos_token=reconstruct_bos_token)
        )
        for layer in range(len(model_block_modules))
        if f"layer_{layer}" in sae_dict
    ]

    return fwd_hooks, fwd_pre_hooks