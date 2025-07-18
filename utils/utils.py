import torch
import random
from typing import List, Tuple, Callable, Union
from jaxtyping import Float, Int
from torch import Tensor
import contextlib
import functools
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import einops
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
from IPython.display import display, HTML
#from utils.hf_patching_utils import get_activations
import matplotlib.colors as mcolors
from colorama import Fore
import textwrap
import gc
import itertools
from datasets import load_dataset

def clear_memory():
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()

model_alias_to_model_name = {"gemma-2b-it" : "google/gemma-2b-it",
                             "gemma-2-2b-it": "google/gemma-2-2b-it",
                             "Meta-Llama-3-8B-Instruct" : "meta-llama/Meta-Llama-3-8B",
                             "meta-llama/Llama-3.1-8B" : "meta-llama/Llama-3.1-8B",
                             "meta-llama_Llama-3.1-8B" : "meta-llama/Llama-3.1-8B",
                             "gemma-2-9b": "/home/wangxin/models/google/gemma-2-9b",
                             "gemma-2-9b-it": "/home/wangxin/models/gemma-2-9b-it",
                             "gemma-2-2b":  "google/gemma-2-2b"}

model_is_chat_model = {"gemma-2b-it" : True,
                        "Meta-Llama-3-8B-Instruct" : True,
                        "meta-llama/Llama-3.1-8B" : False,
                        "meta-llama_Llama-3.1-8B" : False,
                        "meta-llama/Llama-3.1-8B-Instruct" : True,
                        "meta-llama_Llama-3.1-8B-Instruct" : True,
                        "gemma-2-9b-it": True,
                        "gemma-2-2b":  False,
                        "gemma-2-2b-it":  True,
                        "gemma-2-9b": False}

slice_str_dict = {
    "gemma-2b-it": "model\n",
    "gemma-2-2b-it": "model\n",
    "gemma-2-9b-it": "model\n",
    "Meta-Llama-3-8B-Instruct": "assistant<|end_header_id|>"
}
token_labels_model = {"gemma-2b-it": ["{penultimate_entity_tok}", "{last_entity_tok}", "?", "<end_of_turn>", "\n1", "<start_of_turn>", "model", "\n2"],
                      "Meta-Llama-3-8B-Instruct": ["{penultimate_entity_tok}", "{last_entity_tok}", "'?", '<|eot_id|>', '<|start_header_id|>', 'assistant', '<|end_header_id|>']}

def paper_plot(fig, tickangle=0):
    """
    Applies styling to the given plotly figure object targeting paper plot quality.
    """
    fig.update_layout({
        'template': 'plotly_white',
    })
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', tickangle=tickangle,
                    gridcolor='rgb(200,200,200)', griddash='dash', zeroline=False)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black',
                    gridcolor='rgb(200,200,200)', griddash='dash', zeroline=False)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    return fig

def plot_across_pos_and_layer(
    all_scores: List[Union[Float[Tensor, 'n_pos n_layer'], Float[Tensor, 'batch_size n_pos n_layer']]],
    labels: List[str],
    position_labels: List[str],
    title: str,
    positions=None,
    x_ticks=None,
):
    if isinstance(all_scores[0], Tensor):
        all_scores = [scores.cpu().numpy() for scores in all_scores]

    assert len(all_scores[0].shape) in [2, 3]

    if len(all_scores[0].shape) == 2:
        n_pos, n_layer = all_scores[0].shape
    else:
        n_elements, n_pos, n_layer = all_scores[0].shape

    x_ticks = list(range(0, n_layer)) if x_ticks is None else x_ticks

    if positions is None:
        positions = list(range(-n_pos, 0))

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(9, 5))  # width and height in inches

    # Add a trace for each position to extract
    for scores, label in zip(all_scores, labels):
        for position in positions:
            if scores.shape == (n_pos, n_layer):
                ax.plot(
                    x_ticks,
                    scores[position],
                    label=f'{label} {position} {repr(position_labels[position])}'
                )
            else:
                ax.plot(
                    x_ticks,
                    scores[:, position].mean(axis=0),
                    label=f'{label} {position} {repr(position_labels[position])}'
                )
                # compute the numpy mean and std
                ax.fill_between(x_ticks, scores[:, position].mean(axis=0) + scores[:, position].std(axis=0), scores[:, position].mean(axis=0) - scores[:, position].std(axis=0), alpha=0.25)

    ax.set_title(title)
    ax.set_xlabel('Layer source of direction (resid_pre)')
    ax.set_ylabel('Score')
    ax.legend(title='Position source of direction', loc='upper right')

def compute_cosim(a: Float[Tensor, "... d_model"], b: Float[Tensor, "... d_model"], eps=1e-5):
    b = b.to(a)

    return einops.einsum(
        a / (a.norm(dim=-1, keepdim=True) + eps), b / (b.norm(dim=-1, keepdim=True)),
        "... d_model, ... d_model -> ..."
    )

def print_tokens(prompt, tokenizer):
    print([tokenizer.decode(tok) for tok in tokenizer.encode(prompt)])

def run_pca(X, n_components):
    # Standardize data before applying PCA
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_embedded = pca.fit_transform(X)
    return X_embedded, pca, scaler

def value_to_color_hex(value, vmin=-1.0, vmax=1.0):
    # Create a custom colormap
    colors = ['red', 'white', 'green']
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    # Normalize the value
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Get the color
    rgba = cmap(norm(value))
    
    return mcolors.to_hex(rgba)

def value_to_color_rgb(value: float, vmin: float=-0.15, vmax: float = 0.15) -> str:
    """Convert a value to a color on a white-to-orange scale."""
    normalized = min(1, max(0, value / (vmax - vmin)))
    r = 255
    g = int(255 * (1 - normalized) + 165 * normalized)  # Transition from 255 to 165
    b = int(255 * (1 - normalized))  # Transition from 255 to 0
    return f"rgb({r}, {g}, {b})"

def display_token_values(subspace: Float[Tensor, "seq_len d_subspace"], tokens: List[str], vmin=-0.15, vmax=0.15):
    if isinstance(subspace, Tensor):
        subspace = subspace.cpu().numpy()

    html_content = "<div style='font-family: monospace; white-space: pre-wrap;'>"
    for idx in range(len(subspace)):
        value = subspace[idx]
        color = value_to_color_hex(value, vmin=vmin, vmax=vmax)
        html_content += f"<span style='background-color: {color}; padding: 5px; margin: 5px 5px; display: inline-block; border-radius: 3px;'>{tokens[idx].strip()} ({value:.2f})</span>"
    html_content += "</div>"

    display(HTML(html_content))

# def display_direction_activation(model_wrapper, tokenizer, instruction, direction, layer, device="cuda", normalize=True):
#     toks = model_wrapper.tokenize_instructions_fn(instructions=[instruction]).input_ids

#     act = get_activations(model_wrapper.model, model_wrapper.tokenizer, instructions=[instruction], tokenize_instructions_fn=model_wrapper.tokenize_instructions_fn, block_modules=model_wrapper.model_block_modules, positions=list(range(0, len(toks[0]))))

#     act = act[0, :, layer, :]

#     tokens_with_steering = [tokenizer.decode(tok) for tok in toks[0]]
#     direction = direction.to(act)

#     if normalize:
#         subspace = (act / act.norm(dim=-1, keepdim=True)) @ (direction / direction.norm(dim=-1, keepdim=True))
#     else:
#         subspace = act @ direction

#     display_token_values(subspace, tokens_with_steering)


# def display_sae_feature_activation(model_wrapper, tokenizer, instruction, sae, feature_idx, layer, device="cuda", normalize=True):
#     toks = model_wrapper.tokenize_instructions_fn(instructions=[instruction]).input_ids

#     act = get_activations(model_wrapper.model, model_wrapper.tokenizer, instructions=[instruction], tokenize_instructions_fn=model_wrapper.tokenize_instructions_fn, block_modules=model_wrapper.model_block_modules, positions=list(range(0, len(toks[0]))))

#     act = act[0, :, layer, :]

#     sae_feat_acts = sae.encode(act)[:, feature_idx]

#     tokens_with_steering = [tokenizer.decode(tok) for tok in toks[0]]
#     print(tokens_with_steering)

#     display_token_values(sae_feat_acts, tokens_with_steering, tokenizer)

def print_contrastive_generations(prompts, orig_generations, steered_generations):
    for i in range(len(steered_generations)):
        print("Prompt: ", repr(prompts[i]))
        print(Fore.GREEN + f"Original Completion:", textwrap.fill(repr(orig_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
        print(Fore.RED + f"Steered Completion:", textwrap.fill(repr(steered_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
        print(Fore.RESET)

def batch_iterator_chat(repo_id, tokenize_instructions_fn, batch_size, eoi_toks, instruction_key='instruction', output_key='output', split='train'):
    """Yields (padded) batches from a chat dataset."""

    dataset = load_dataset(repo_id, split=split)
    dataset = dataset.shuffle(seed=42)

    # filter out instructions that have inputs
    if 'input' in dataset.features:
        dataset = dataset.filter(lambda x: x['input'].strip() != '')

    dataset_instructions = dataset[instruction_key]
    dataset_outputs = dataset[output_key]

    it_instructions = iter(dataset_instructions)
    it_outputs = iter(dataset_outputs)
    while True:
        instructions_batch = list(itertools.islice(it_instructions, batch_size))
        outputs_batch = list(itertools.islice(it_outputs, batch_size))
        if not instructions_batch or not outputs_batch:
            break
        inputs = tokenize_instructions_fn(instructions=instructions_batch, outputs=outputs_batch)

        loss_mask = inputs["attention_mask"].clone()
        loss_mask[:, -1] = 0 # loss should not be computed for last token position

        # also mask out all tokens before the eoi token region
        for b in range(inputs["input_ids"].shape[0]):
            for i in range(inputs["input_ids"].shape[1]):
                # print(inputs["input_ids"][b, i:i+eoi_toks.shape[0]])
                # print(eoi_toks)

                if torch.all(inputs["input_ids"][b, i:i+eoi_toks.shape[0]] == eoi_toks):
                    loss_mask[b, :i + eoi_toks.shape[0] - 1] = 0
                    break

                # normally the above condition works. but the tokenization instruction tokens in Llama2 is not clean, and so we need this hack
                if eoi_toks.shape[0] == 6 and (inputs["input_ids"][b, i:i+eoi_toks.shape[0]] == eoi_toks).sum().item() >= eoi_toks.shape[0] - 2:
                    loss_mask[b, :i + eoi_toks.shape[0] - 1] = 0
                    break

        yield inputs, loss_mask

def batch_iterator_text(repo_id, tokenizer, batch_size, max_length, split='train'):
    """Yields (padded) batches from a text dataset."""
    dataset = load_dataset(repo_id, split=split, streaming=True, trust_remote_code=True)

    it_dataset = iter(dataset)
    while True:
        batch = list(itertools.islice(it_dataset, batch_size))
        if not batch:
            break
        inputs = tokenizer([b['text'] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=max_length)

        loss_mask = inputs["attention_mask"].clone()
        loss_mask[:, -1] = 0 # loss should not be computed for last token position

        yield inputs, loss_mask

def k_largest_indices(
    x: Float[Tensor, "rows cols"],
    k: int,
    largest: bool = True,
    buffer: tuple[int, int] | None = (5, -5),
) -> Int[Tensor, "k 2"]:
    """
    Args:
        x:
            2D array of floats (these will be the values of feature activations or losses for each
            token in our batch)
        k:
            Number of indices to return
        largest:
            Whether to return the indices for the largest or smallest values
        buffer:
            Positions to avoid at the start / end of the sequence, i.e. we can include the slice buffer[0]: buffer[1].
            If None, then we use all sequences

    Returns:
        The indices of the top or bottom `k` elements in `x`. In other words, output[i, :] is the (row, column) index of
        the i-th largest/smallest element in `x`.
    """
    if buffer is None:
        buffer = (0, x.size(1))
    x = x[:, buffer[0] : buffer[1]]
    indices = x.flatten().topk(k=k, largest=largest).indices
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer[0]
    return torch.stack((rows, cols), dim=1)

def get_orthogonalized_matrix(matrix: Float[Tensor, '... d_model'], vec: Float[Tensor, 'd_model']) -> Float[Tensor, '... d_model']:
    vec = vec / torch.norm(vec)
    vec = vec.to(matrix)

    proj = einops.einsum(matrix, vec.unsqueeze(-1), '... d_model, d_model single -> ... single') * vec
    return matrix - proj

def tl_orthogonalize_gemma_weights(tl_model: HookedTransformer, direction: Float[Tensor, "d_model"]):
    tl_model.W_E.data = get_orthogonalized_matrix(tl_model.W_E.data, direction)

    for block in tl_model.blocks:
        block.attn.W_O.data = get_orthogonalized_matrix(block.attn.W_O.data, direction)
        block.mlp.W_out.data = get_orthogonalized_matrix(block.mlp.W_out.data, direction)

def find_string_in_tokens(target, tokens, tokenizer) -> slice:
    assert target in tokenizer.decode(tokens), f"The target {target} isn't contained in the whole array of tokens {[tokenizer.decode([tok]) for tok in tokens]}"
    # we first perform the binary search over the end index of the slice
    end_idx_left, end_idx_right = 0, len(tokens) 
    while end_idx_left != end_idx_right:
        mid = (end_idx_right + end_idx_left) // 2
        if target in tokenizer.decode(tokens[:mid]):
            end_idx_right = mid
        else:
            end_idx_left = mid + 1
    end_idx = end_idx_left
    # now we perform the binary search over the start index of the slice
    start_idx_left, start_idx_right = 0, end_idx-1 
    while start_idx_left != start_idx_right:
        mid = (start_idx_right + start_idx_left + 1) // 2
        if target in tokenizer.decode(tokens[mid:end_idx]):
            start_idx_left = mid
        else:
            start_idx_right = mid-1
    start_idx = start_idx_left
    target_slice = slice(start_idx, end_idx)
    assert target in tokenizer.decode(tokens[target_slice])
    return target_slice

def slice_to_list(slice_obj, length=None):
    start, stop, step = slice_obj.start, slice_obj.stop, slice_obj.step
    
    # If length is not provided, use stop if it's not None, else raise an error
    if length is None:
        if stop is not None:
            length = stop
        else:
            raise ValueError("Length must be provided if stop is None")
    
    # Adjust start, stop, and step
    start = 0 if start is None else start
    stop = length if stop is None else stop
    step = 1 if step is None else step
    
    return list(range(start, stop, step))