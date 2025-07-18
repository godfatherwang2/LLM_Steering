import torch
import random
from typing import List, Tuple, Callable, Dict
from jaxtyping import Float, Int
from torch import Tensor
import contextlib
import functools
from tqdm import tqdm
import einops

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def get_direction_ablation_input_pre_hook(direction: Tensor):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_direction_ablation_output_hook(directions: List[Tensor]):

    if isinstance(directions, Tensor):
        directions = [directions]

    def hook_fn(module, input, output):
        nonlocal directions

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        for direction in directions:
            direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            direction = direction.to(activation)
            activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn

def get_positional_direction_ablation_output_hook(directions: List[Tensor]):

    if isinstance(directions, Tensor):
        directions = [directions]

    def hook_fn(module, input, output, steering_positions: List[List[int]], **kwargs):
        nonlocal directions

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        cur_seq_len = activation.size(1)

        # don't steer at generation tokens
        if cur_seq_len > 1:
            for direction in directions:
                direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
                direction = direction.to(activation)
                for batch_idx, positions in enumerate(steering_positions):
                    activation[batch_idx, positions] -= (activation[batch_idx, positions] @ direction).unsqueeze(-1) * direction 

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn

def get_activation_addition_output_hook(vectors: List[Float[Tensor, "d_model"]], coeffs: List[Float[Tensor, ""]]):

    if isinstance(vectors, Tensor):
        vectors = [vectors]
        coeffs = [coeffs]

    def hook_fn(module, input, output):
        nonlocal vectors

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        for vector, coeff in zip(vectors, coeffs):
            vector = vector.to(activation)
            activation += coeff * vector

        if isinstance(input, tuple):
            return (activation, *output[1:])
        else:
            return activation
    return hook_fn

def get_positional_activation_addition_output_hook(vectors: List[Float[Tensor, "d_model"]], coeffs: List[Float[Tensor, ""]]):

    if isinstance(vectors, Tensor):
        vectors = [vectors]
        coeffs = [coeffs]

    def hook_fn(module, input, output, steering_positions: List[List[int]], **kwargs):
        nonlocal vectors

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        cur_seq_len = activation.size(1)

        # don't steer at generation tokens
        if cur_seq_len > 1:
            for vector, coeff in zip(vectors, coeffs):
                vector = vector.to(activation)
                for batch_idx, positions in enumerate(steering_positions):
                    activation[batch_idx, positions] += coeff * vector

        if isinstance(input, tuple):
            return (activation, *output[1:])
        else:
            return activation
    return hook_fn

def get_all_direction_ablation_hooks(
    model_base,
    direction: Float[Tensor, 'd_model'],
):
    fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]

    return fwd_pre_hooks, fwd_hooks

def get_directional_patching_input_pre_hook(direction: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 
        activation += coeff * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_activation_addition_input_pre_hook(vector: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal vector

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        vector = vector.to(activation)
        activation += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_mean_activations_pre_hook(layer, cache: Float[Tensor, "pos layer d_model"], n_samples, positions: List[int]):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)
    return hook_fn

def get_mean_activations(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1]):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the mean activations in high-precision to avoid numerical issues
    mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64, device=model.device)

    fwd_pre_hooks = [(block_modules[layer], get_mean_activations_pre_hook(layer=layer, cache=mean_activations, n_samples=n_samples, positions=positions)) for layer in range(n_layers)]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return mean_activations

def get_mean_diff(model, tokenizer, harmful_instructions, harmless_instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1]):
    mean_activations_1 = get_mean_activations(model, tokenizer, harmful_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)
    mean_activations_2 = get_mean_activations(model, tokenizer, harmless_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)

    mean_diff: Float[Tensor, "n_positions n_layers d_model"] = mean_activations_1 - mean_activations_2

    return mean_diff

def get_activations_pre_hook(cache: Float[Tensor, "pos layer d_model"], n_samples, positions: List[int]):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[:, :] += activation[:, positions, :]
    return hook_fn

@torch.no_grad()
def get_activations(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=32, layers=None,positions=[-1], verbose=True) -> Float[Tensor, 'n layer pos d_model']:
    torch.cuda.empty_cache()

    if layers is None:
        layers = range(model.config.num_hidden_layers)

    n_positions = len(positions)
    n_layers = len(layers)
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the activations in high-precision to avoid numerical issues
    activations = torch.zeros((len(instructions), n_layers, n_positions, d_model), device=model.device)

    for i in tqdm(range(0, len(instructions), batch_size), disable=not verbose):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        inputs_len = len(inputs.input_ids)

        fwd_pre_hooks = [(block_modules[layer], get_activations_pre_hook(cache=activations[i:i+inputs_len, layer_idx, :, :], n_samples=n_samples, positions=positions)) for layer_idx, layer in enumerate(layers)]

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return activations



def get_sae_fwd_pre_hook(sae, reconstruct_bos_token: bool = False):
    def hook_fn(module, input, input_ids: Int[Tensor, "batch_size seq_len"]=None, attention_mask: Int[Tensor, "batch_size seq_len"]=None):
        nonlocal sae, reconstruct_bos_token

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        dtype = activation.dtype
        batch_size, seq_pos, d_model = activation.shape

        reshaped_activation = einops.rearrange(activation, "b s d -> (b s) d")
        reconstructed_activation = sae(reshaped_activation).sae_out.to(dtype)
        reconstructed_activation = einops.rearrange(reconstructed_activation, "(b s) d -> b s d", b=batch_size, s=seq_pos)

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


def get_sae_hooks(model_block_modules: List[torch.nn.Module], sae_dict: Dict, reconstruct_bos_token: bool = False):
    """
    Get the hooks for the SAE layers.

    args:
        model_block_modules: List[nn.Module]: the model block modules to hook
        sae_dict: Dict[str, Sae]: the SAE layers. The keys in the dictionary have the format 'layer_<layer_number>'.
        reconstruct_bos_token: bool: whether to reconstruct the <|begin_of_text|> token
    """
    from sae.sparse_autoencoder import Sae

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