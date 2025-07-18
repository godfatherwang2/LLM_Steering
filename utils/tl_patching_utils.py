from numpy import mean
import torch
import functools
import einops
from tqdm import tqdm
from typing import List, Optional, Union, Tuple
from jaxtyping import Int, Float
from torch import Tensor
from transformer_lens.hook_points import HookPoint
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens import utils

def metric(
    logits: Float[Tensor, 'batch seq d_vocab_out'],
    clean_toks: Int[Tensor, 'batch seq'],
    corrupt_toks: Int[Tensor, 'batch seq'],
    epsilon: Float = 1e-6
):
    if logits.device == torch.device('cpu') and logits.dtype == torch.float16:
        # fix casting issues
        logits = logits.to(torch.float64)

    # we only care about the last tok position
    logits = logits[:, -1, :]

    probs = torch.nn.functional.softmax(logits, dim=-1)
    clean_probs = probs[:, clean_toks].sum(dim=-1)
    corrupt_probs = probs[:, corrupt_toks].sum(dim=-1)

    return torch.log(clean_probs + epsilon) - torch.log(corrupt_probs + epsilon)

def activation_patching_hook(
    activation: Float[Tensor, "batch seq d_activation"],
    hook: HookPoint,
    pos: Union[int, List[int]],
    cache_to_patch_from: ActivationCache,
    patch_mean_activation: bool=False,
) -> Float[Tensor, "batch seq d_activation"]:

    if patch_mean_activation:
        activation[:, pos, :] = cache_to_patch_from[hook.name][:, pos, :].mean(dim=0)
    else:
        activation[:, pos, :] = cache_to_patch_from[hook.name][:, pos, :]

    return activation

def head_patching_hook(
    activation: Float[Tensor, "batch seq n_heads d_activation"],
    hook: HookPoint,
    pos: Union[int, List[int]],
    head: int,
    cache_to_patch_from: ActivationCache,
    scale_factor: float=1.0,
    patch_mean_activation: bool=False,
) -> Float[Tensor, "batch seq n_heads d_activation"]:

    if pos is None:
        pos = list(range(activation.shape[1]))
    elif type(pos) is not list:
        pos = [pos] 

    if patch_mean_activation:
        activation[:, pos, head, :] = cache_to_patch_from[hook.name][:, pos, head, :].mean(dim=0) * scale_factor
    else:
        activation[:, pos, head, :] = cache_to_patch_from[hook.name][:, pos, head, :] * scale_factor

    return activation

def directional_head_patching_hook(
    activation: Float[Tensor, "batch seq n_heads d_activation"],
    hook: HookPoint,
    pos: Union[int, List[int]],
    head: int,
    cache_to_patch_from: ActivationCache,
    direction: Float[Tensor, "d_activation"],
) -> Float[Tensor, "batch seq n_heads d_activation"]:
    if pos is None:
        pos = list(range(activation.shape[1]))
    elif type(pos) is not list:
        pos = [pos] 

    direction = direction.to(activation)
    direction = direction / direction.norm()

    for p in pos:
        activation[:, p, head, :] = activation[:, p, head, :] - ((activation[:, p, head, :] @ direction).unsqueeze(-1) * direction) # ablate out the direction
        activation[:, p, head, :] = activation[:, p, head, :] + (cache_to_patch_from[hook.name][:, p, head, :] @ direction).unsqueeze(-1) * direction # add the direction back in

    return activation

def directional_ablation_head_patching_hook(
    activation: Float[Tensor, "batch seq n_heads d_activation"],
    hook: HookPoint,
    pos: Union[int, List[int]],
    head: int,
    cache_to_patch_from: ActivationCache,
    direction: Float[Tensor, "d_activation"],
) -> Float[Tensor, "batch seq n_heads d_activation"]:
    if pos is None:
        pos = list(range(activation.shape[1]))
    elif type(pos) is not list:
        pos = [pos] 

    direction = direction.to(activation)
    direction = direction / direction.norm()

    for p in pos:
        activation[:, p, head, :] = cache_to_patch_from[hook.name][:, p, head, :]
        activation[:, p, head, :] = activation[:, p, head, :] - ((activation[:, p, head, :] @ direction).unsqueeze(-1) * direction) # ablate out the direction

    return activation

def direction_ablation_hook(
    activation: Float[Tensor, "batch seq d_activation"],
    hook: HookPoint,
    pos: Optional[Union[int, List[int]]],
    direction: Float[Tensor, "d_activation"],
):
    if pos is None:
        pos = list(range(activation.shape[1]))
    elif type(pos) is not list:
        pos = [pos]

    direction = direction.to(activation)
    direction = direction / direction.norm()

    for p in pos:
        activation[:, p, :] = activation[:, p, :] - ((activation[:, p, :] @ direction).unsqueeze(-1) * direction)


    return activation

def mean_direction_patching_hook(
    activation: Float[Tensor, "batch seq d_activation"],
    hook: HookPoint,
    positions: List[int],
    directions: List[Float[Tensor, "d_activation"]],
    mean_magnitudes: List[float],
):

    for i in range(len(directions)):
        directions[i] = directions[i].to(activation)
        directions[i] = directions[i] / directions[i].norm()

    for i, p in enumerate(positions):
        activation[:, p, :] = activation[:, p, :] - ((activation[:, p, :] @ directions[i]).unsqueeze(-1) * directions[i]) # ablate out the direction
        activation[:, p, :] = activation[:, p, :] + (directions[i] * mean_magnitudes[i]) # add the direction back in with the mean magnitude

    return activation

def pattern_patching_hook(
    activation: Float[Tensor, "batch head seq seq"],
    hook: HookPoint,
    head: Union[int, List[int]],
    query_pos: Union[int, List[int]],
    cache_to_patch_from: ActivationCache,
):
    cached_seq_length = min(cache_to_patch_from[hook.name].shape[-1], activation.shape[-1])

    if isinstance(head, int):
        head = [head]

    if len(head) == activation.shape[1]:
        # path all heads
        activation[:, :, query_pos, :cached_seq_length] = cache_to_patch_from[hook.name][:, :, query_pos, :cached_seq_length]
    else:
        for head_idx in head:
            activation[:, head_idx, query_pos, :cached_seq_length] = cache_to_patch_from[hook.name][:, head_idx, query_pos, :cached_seq_length]

    return activation

def direction_patching_hook(
    activation: Float[Tensor, "batch seq d_activation"],
    hook: HookPoint,
    pos: Union[int, List[int]],
    cache_to_patch_from: ActivationCache,
    direction: Float[Tensor, "d_activation"],
    patch_mean_activation: bool=False,
):
    if pos is None:
        pos = list(range(activation.shape[1]))
    elif type(pos) is not list:
        pos = [pos]

    direction = direction.to(activation.device)
    direction = direction / direction.norm()

    old_coefs = einops.einsum(activation[:, pos, :], direction, 'batch pos d_activation, d_activation -> batch pos')
    new_coefs = einops.einsum(cache_to_patch_from[hook.name][:, pos, :], direction, 'batch pos d_activation, d_activation -> batch pos')

    if patch_mean_activation:
        new_coefs = new_coefs.mean(dim=0)
        new_coefs = einops.repeat(new_coefs, 'pos -> batch pos', batch=activation.shape[0])

    activation[:, pos, :] = activation[:, pos, :] - einops.einsum(old_coefs - new_coefs,  direction, 'batch pos, d_activation -> batch pos d_activation')

    return activation

def directional_patching_hook(act, hook, direction, original_pos, patch_pos, patch_cache):
    '''
        Patches activation with the direction vector at original_pos, subtracting the component of the 
        direction in the original activation at original_pos and adding the component of the direction in the new activation at patch_pos.
    '''
    new_acts = patch_cache[hook.name]
    
    # act is of shape batch, seq_len, d_model
    if original_pos is None:
        old_act = act
    elif isinstance(original_pos, int):
        old_act = act[:, original_pos, :]
    else:
        old_act = act[list(range(act.shape[0])), original_pos, :]

    if old_act.ndim == 2:
        old_act = old_act.unsqueeze(1)

    if patch_pos is None:
        new_act = new_acts
    elif isinstance(patch_pos, int):
        new_act = new_acts[:, patch_pos, :].unsqueeze(1)
    else:
        new_act = new_acts[list(range(act.shape[0])), patch_pos, :]

    if new_act.ndim == 2:
        new_act = new_act.unsqueeze(1)

    old_act = old_act
    new_act = new_act.mean(dim=0) 

    direction = (direction / (direction.norm(dim=-1, keepdim=True) + 1e-6))
    direction = direction.to(torch.float16)

    mean_proj_old = einops.einsum(
        old_act, 
        direction, 
        "b p d, d -> b p"
    ).unsqueeze(-1) * direction.unsqueeze(0) # batch, seq_len, d_model
    mean_proj_new = einops.einsum(
        new_act, 
        direction,
        "p d, d -> p"
    ).unsqueeze(-1) * direction.unsqueeze(0) # seq_len, d_model

    if original_pos is None:
        act = act - mean_proj_old + mean_proj_new
    elif isinstance(original_pos, int):
        act[:, original_pos, :] = act[:, original_pos, :] - mean_proj_old + mean_proj_new
    else:
        act[list(range(act.shape[0])), original_pos, :] = act[list(range(act.shape[0])), original_pos, :] - mean_proj_old + mean_proj_new

    return act 

def activation_steering_hook(
    activation: Float[Tensor, "batch seq d_activation"],
    hook: HookPoint,
    pos: Union[int, List[int]],
    vector: Float[Tensor, "d_activation"],
    scale_factor: Optional[float]=None,
    relative_scale_factor: Optional[float]=None,
    rescale: bool=False,
) -> Float[Tensor, "batch seq n_heads d_activation"]:

    if scale_factor is None and relative_scale_factor is None:
        raise ValueError("Either scale_factor or relative_scale_factor must be provided")

    if type(pos) is not list:
        pos = [pos]

    vector = vector.to(activation.device).unsqueeze(0).unsqueeze(0) # 1, 1, d_activation

    original_norms = activation[:, pos, :].norm(dim=-1, keepdim=True) # batch, pos, 1

    if relative_scale_factor is not None:
        scale_factor = relative_scale_factor * original_norms

    activation[:, pos, :] = activation[:, pos, :] + vector * scale_factor

    if rescale:
        new_norms = activation[:, pos, :].norm(dim=-1, keepdim=True)
        activation[:, pos, :] = activation[:, pos, :] * (original_norms / new_norms)

    return activation


def generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, "batch_size seq_len"],
    max_tokens_generated=64,
    fwd_hooks=[],
    include_prompt=False,
    verbose=True,
    skip_special_tokens=True,
) -> List[str]:

    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks

    model.reset_hooks()
    
    for i in tqdm(range(max_tokens_generated), disable=not verbose):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :-max_tokens_generated + i])

            # greedy sampling (temperature=0)
            next_tokens = logits[:, -1, :].argmax(dim=-1)

            all_toks[:,-max_tokens_generated+i] = next_tokens

    if include_prompt:
        return model.tokenizer.batch_decode(all_toks, skip_special_tokens=skip_special_tokens)
    else:
        return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=skip_special_tokens)

def generate_with_patch(
    model: HookedTransformer,
    prompt_toks: Int[Tensor, "batch_size seq_len"],
    cache_to_patch_from: ActivationCache,
    activation_type: str,
    pos: Union[int, List[int]],
    layers: List[int],
    max_tokens_generated: int=64,
    include_prompt: bool=False,
) -> List[List[str]]:

    hook_fn = functools.partial(
        activation_patching_hook,
        pos=pos,
        cache_to_patch_from=cache_to_patch_from,
        patch_mean_activation=True,
    )

    fwd_hooks = [(utils.get_act_name(activation_type, layer), hook_fn) for layer in layers]

    if prompt_toks.ndim == 1:
        prompt_toks = prompt_toks.unsqueeze(0)

    generation = generate_with_hooks(
        model=model,
        toks=prompt_toks,
        fwd_hooks=fwd_hooks,
        max_tokens_generated=max_tokens_generated,
        include_prompt=include_prompt,
    )

    return generation

def generate_with_head_patch(
    model: HookedTransformer,
    prompt_toks: Int[Tensor, "batch_size seq_len"],
    cache_to_patch_from: ActivationCache,
    activation_type: str,
    pos: Union[int, List[int]],
    heads: List[Tuple[int, int]],
    scale_factor: float=1.0,
    max_tokens_generated: int=64,
    include_prompt: bool=False,
) -> List[List[str]]:
    
    fwd_hooks = [
        (
            utils.get_act_name(activation_type, head[0]),
            functools.partial(
                head_patching_hook,
                pos=pos,
                head=head[1],
                cache_to_patch_from=cache_to_patch_from,
                scale_factor=scale_factor,
                patch_mean_activation=True,
            )
        )
        for head in heads
    ]

    if prompt_toks.ndim == 1:
        prompt_toks = prompt_toks.unsqueeze(0)

    generation = generate_with_hooks(
        model=model,
        toks=prompt_toks,
        fwd_hooks=fwd_hooks,
        max_tokens_generated=max_tokens_generated,
        include_prompt=include_prompt,
    )

    return generation

def generate_with_pattern_patch(
    model: HookedTransformer,
    prompt_toks: Int[Tensor, "batch_size seq_len"],
    cache_to_patch_from: ActivationCache,
    query_pos: Union[int, List[int]],
    heads: List[Tuple[int, int]],
    max_tokens_generated: int=64,
    include_prompt: bool=False,
) -> List[List[str]]:
    
    fwd_hooks = [
        (
            utils.get_act_name('pattern', head[0]),
            functools.partial(
                pattern_patching_hook,
                query_pos=query_pos,
                head=head[1],
                cache_to_patch_from=cache_to_patch_from,
            )
        )
        for head in heads
    ]

    if prompt_toks.ndim == 1:
        prompt_toks = prompt_toks.unsqueeze(0)

    generation = generate_with_hooks(
        model=model,
        toks=prompt_toks,
        fwd_hooks=fwd_hooks,
        max_tokens_generated=max_tokens_generated,
        include_prompt=include_prompt,
    )

    return generation

def generate_with_steering(
    model: HookedTransformer,
    prompt_toks: Int[Tensor, "batch_size seq_len"],
    activation_type: str,
    pos: Union[int, List[int]],
    layers: List[int],
    vector: Float[Tensor, "d_activation"],
    scale_factor: float=None,
    relative_scale_factor: float=None,
    max_tokens_generated: int=64,
    include_prompt: bool=False,
) -> List[List[str]]:
 
    hook_fn = functools.partial(
        activation_steering_hook,
        pos=pos,
        vector=vector,
        scale_factor=scale_factor,
        relative_scale_factor=relative_scale_factor,
    )

    fwd_hooks = [(utils.get_act_name(activation_type, layer), hook_fn) for layer in layers]

    if prompt_toks.ndim == 1:
        prompt_toks = prompt_toks.unsqueeze(0)

    generation = generate_with_hooks(
        model=model,
        toks=prompt_toks,
        fwd_hooks=fwd_hooks,
        max_tokens_generated=max_tokens_generated,
        include_prompt=include_prompt,
    )

    return generation