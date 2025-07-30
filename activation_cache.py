import os
import sys
sys.path.append('./')
sys.path.append('../')
# sys.path.append('../..')
# sys.path.append('../../..')

import argparse
import os
import torch
from typing import List, Callable, Dict, Callable, Union, Tuple
import os
import gc
import time
import json
from tqdm import tqdm
import glob
from jaxtyping import Int, Float
from torch import Tensor
import torch
import numpy as np
import random
from torch.utils.data import Dataset, Sampler, DataLoader
from functools import partial

from utils.hf_patching_utils import add_hooks
from utils.hf_models.model_base import ModelBase
from utils.hf_models.model_factory import construct_model_base
from utils.utils import clear_memory, model_alias_to_model_name, find_string_in_tokens


random.seed(10)

DEFAULT_CACHE_DIR = os.path.join("dataset/cached_activations")

def _get_activations_forward_hook(cache: Float[Tensor, "pos d_model"]):
    def hook_fn(module, input_tuple, output_tuple):
        nonlocal cache
        if isinstance(output_tuple, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output_tuple[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output_tuple
        cache[:, :] = activation[:, :].to(cache)
    return hook_fn


@torch.no_grad()
def _get_activations_fixed_seq_len(model, tokenizer, prompts: List[str], block_modules: List[torch.nn.Module], seq_len: int = 512, layers: List[int]=None, batch_size=32, save_device: Union[torch.device, str] = "cuda", verbose=True) -> Tuple[Int[Tensor, 'n seq_len'], Float[Tensor, 'n layer seq_len d_model']]:
    torch.cuda.empty_cache()

    if layers is None:
        layers = range(model.config.num_hidden_layers)

    n_layers = len(layers)
    d_model = model.config.hidden_size

    # we store the activations in high-precision to avoid numerical issues
    activations = torch.zeros((len(prompts), n_layers, seq_len, d_model), device=save_device)
    all_input_ids = torch.zeros((len(prompts), seq_len), dtype=torch.long, device=save_device)

    # Fill all_input_ids with tokenizer.pad_token_id
    all_input_ids.fill_(tokenizer.pad_token_id)

    for i in tqdm(range(0, len(prompts), batch_size), disable=not verbose):
        inputs = tokenizer(prompts[i:i+batch_size], return_tensors="pt", padding=True, truncation=True, max_length=seq_len)

        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        inputs_len = len(input_ids)
        num_input_toks = input_ids.shape[-1]
        
        fwd_hooks = [
            (block_modules[layer], _get_activations_forward_hook(cache=activations[i:i+inputs_len, layer_idx, :num_input_toks, :])) 
            for layer_idx, layer in enumerate(layers)
        ]

        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        all_input_ids[i:i+inputs_len, :num_input_toks] = input_ids

    return all_input_ids, activations

# %%

def get_compute_activation_fn(model_base: ModelBase, layers: List[int]=None, seq_len: int = 128, batch_size: int = 32):

    # print(f"WARNING: Saving only activations at the last 6 token positions")

    def compute_activation_fn(prompts: List[str]):
        nonlocal model_base, layers, seq_len, batch_size
        n_layers = model_base.model.config.num_hidden_layers
        if layers is None:
            layers = range(n_layers)

        input_ids, activations  = _get_activations_fixed_seq_len(
            model_base.model,
            model_base.tokenizer,
            prompts=prompts,
            block_modules=model_base.model_block_modules,
            seq_len=seq_len,
            layers=layers,
            batch_size=batch_size,
            save_device='cuda',
            verbose=False
        )
        activations: Float[Tensor, 'n n_layers seq_len d_model'] = activations.cpu()
        input_ids: Int[Tensor, 'n seq_len'] = input_ids.cpu()

        n_layers = activations.shape[1]
        assert n_layers == len(layers), f"Expected {len(layers)} layers, but got {n_layers}"

        # save only the activations at the last 6 token positions
        # activations = activations[:, :, -6:, :]

        return input_ids, activations

    return compute_activation_fn


def cache_activations(model_base: ModelBase, prompts: List[str], compute_activation_fn: Callable,
                      tokens_to_cache: str, layers: List[int], foldername: str,
                      batch_size: int = 32, seq_len: int = 128, shard_size: int = 1000,
                      substrings: List[str] = None, cache_type="after_substring") -> None:

    d_model = model_base.model.config.hidden_size
    if layers is None:
        n_layers = model_base.model.config.num_hidden_layers
        layers = list(range(n_layers)) # 确保 layers 是一个列表
    else:
        n_layers = len(layers)
    # === 第一阶段：预计算 total_tokens (此部分代码完全不变) ===
    total_tokens = 0
    valid_prompts = []
    valid_substrings = []
    valid_start_pos = []
    valid_lens = []
    valid_ids = []
    for prompt, substring in tqdm(zip(prompts, substrings), desc="Preprocessing and calculating size",total=len(prompts)):
        inputs = model_base.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
        if substring in model_base.tokenizer.decode(inputs.input_ids[0]):
            s = find_string_in_tokens(substring, inputs.input_ids[0], model_base.tokenizer) 
            if s is None: continue
            non_padding_mask = (inputs.input_ids[0] != model_base.tokenizer.pad_token_id)
            non_padding_indices = torch.nonzero(non_padding_mask, as_tuple=True)[0]
            if len(non_padding_indices) == 0: continue
            last_non_padding_index = non_padding_indices[-1].item()
            if cache_type == "after_substring": start_pos = s.stop
            elif cache_type == "from_substring": start_pos = s.start
            else: raise ValueError
            if start_pos >= seq_len: continue
            valid_length = min(last_non_padding_index - start_pos + 1, seq_len - start_pos)
            if valid_length <= 0: continue
            total_tokens += valid_length
            valid_prompts.append(prompt)
            valid_substrings.append(substring)
            valid_start_pos.append(start_pos)
            valid_lens.append(valid_length)
            valid_ids.append(inputs.input_ids[0][start_pos:start_pos+valid_length])
    
    print(f"Total tokens to be cached: {total_tokens}")

    # === 新功能: 创建元数据文件 ===
    # 在计算出 total_tokens 后，我们就可以确定所有形状信息了
    metadata = {
        "description": "Metadata for cached activations.",
        "activations_original_shape": (n_layers, total_tokens, d_model),
        "token_ids_shape": (total_tokens,),
        "layers_cached": layers
    }
    metadata_filepath = os.path.join(foldername, "metadata.json")
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_filepath}")
    # === 第二阶段：存储激活值 ===
    # 为每一层创建独立的二维 memmap 文件
    memmap_files_acts = {}
    for layer_idx, layer_num in enumerate(layers):
        # #[修改点]: 按照要求更改文件名格式为 "acts_{layerid}.dat"
        filepath = os.path.join(foldername, f"acts_{layer_num}.dat")
        print(f"Creating memmap file for layer {layer_num} at: {filepath}")
        memmap_files_acts[layer_idx] = np.memmap(filepath, dtype='float32', mode='w+',
                                                 shape=(total_tokens, d_model))
    # `ids.dat` 的创建保持不变
    memmap_file_ids = np.memmap(os.path.join(foldername, "ids.dat"), dtype='int32', mode='w+',
                             shape=(total_tokens,))
    
    current_idx = 0
    for i in tqdm(range(0, len(valid_prompts), batch_size), desc="Computing and writing activations"):
        batch_prompts = valid_prompts[i:i+batch_size]
        batch_start_pos = valid_start_pos[i:i+batch_size]
        batch_valid_len = valid_lens[i:i+batch_size]
        batch_valid_ids = valid_ids[i:i+batch_size]

        input_ids, activations = compute_activation_fn(prompts=batch_prompts)
        for j in range(len(activations)):
            act = activations[j]
            start_pos = batch_start_pos[j]
            valid_len = batch_valid_len[j]
            valid_act_all_layers = act[:, start_pos:start_pos+valid_len, :]

            # 循环遍历每一层，并写入对应的文件
            for layer_idx in range(n_layers):
                layer_act_slice = valid_act_all_layers[layer_idx, :, :]
                memmap_files_acts[layer_idx][current_idx:current_idx+valid_len, :] = layer_act_slice.float().cpu().numpy()
            # 写入 token ID
            memmap_file_ids[current_idx:current_idx+valid_len] = batch_valid_ids[j].cpu().numpy()
            current_idx += valid_len
        gc.collect()

    # Flush all memmap files to disk
    print("Flushing all memmap files to disk...")
    for memmap_file in memmap_files_acts.values():
        memmap_file.flush()
    memmap_file_ids.flush()
    print("Caching complete.")


def load_pile_data(tokens_to_cache: str):
    from datasets import load_dataset

    assert tokens_to_cache == 'random'
    prompts = load_dataset("NeelNanda/pile-10k")['train']['text']
    substrings = []
    for prompt in prompts:
        substrings.append(random.choices(prompt.split(), k=1)[0])

    return prompts, substrings


def load_pkusaferlhf_instructions_to_cache(model_alias: str, tokens_to_cache: str, entity_type: str = None, balance_data: bool = True):
    from dataset.load_data import load_pkusaferlhf_queries, balance_data
    
    queries = load_pkusaferlhf_queries(model_alias=model_alias)

    print(f"Initial number of loaded queries is {len(queries)}")

    for query in queries:
        assert query['label'] == True or query['label'] == False

    # if balance_data == True:
    #     queries = balance_data(queries, labels=[True, False], shuffle=True)
    
    
    if tokens_to_cache == "response":
        substrings = ["\nResponse:\n" for query in queries]

    prompts = ([query["prompt"] for query in queries if query['label']==True], [query["prompt"] for query in queries if query['label']==False])
    substrings = (substrings[:len(prompts[0])], substrings[len(prompts[0]):])

    print(f"Number of deduplicated and balanced queries is {len(prompts[0])}, {len(prompts[1])}")

    return prompts, substrings

def load_safeedit_instructions_to_cache(model_alias: str, tokens_to_cache: str, entity_type: str = None, balance_data: bool = True):
    from dataset.load_data import load_safeedit_queries
    queries = load_safeedit_queries(model_alias=model_alias,apply_chat_format=True)
    substrings = [tokens_to_cache for query in queries]
    prompts = ([query["prompt"] for query in queries if query['label']==True], [query["prompt"] for query in queries if query['label']==False])
    substrings = (substrings[:len(prompts[0])], substrings[len(prompts[0]):])
    return prompts, substrings

def main(args):
    
    torch.set_grad_enabled(False)

    model_alias = args.model_alias
    batch_size = args.batch_size
    tokens_to_cache = args.tokens_to_cache # "response" / "entity" / "last_eoi" / "?"
    dataset = args.dataset
    model_path = model_alias_to_model_name[model_alias]
    
    layers = list(map(int, (args.layers).split(','))) # None means all layers
    n_positions = 1 # starting from the tokens_to_cache, how many positions to take to the left
    #layers_str = "_".join([str(l) for l in layers])

    model_base = construct_model_base(model_path)
    shard_size = None

    model_base.tokenizer.pad_token = model_base.tokenizer.eos_token
    model_base.tokenizer.truncation_side="right"
    model_base.tokenizer.padding_side = 'right'
    

    def get_activations(model_base, prompts, substrings, seq_len, batch_size, tokens_to_cache, dataset_name, shard_size, label, cache_type):
        print('len prompts', len(prompts))
        print('prompt:', prompts[0])
        print('substring:', substrings[0])

        compute_activation_fn = get_compute_activation_fn(model_base, layers=layers, seq_len=seq_len, batch_size=batch_size)

        # Update shard size to be the minimum of the number of prompts and the shard size
        if shard_size is None:
            shard_size = len(prompts)
        else:
            shard_size = min(shard_size, len(prompts))

        print('shard_size', shard_size)

        foldername = f"{DEFAULT_CACHE_DIR}/{model_alias}_{dataset_name}/label_{label}_shard_size_{shard_size}"
        os.makedirs(foldername, exist_ok=True)

        cache_activations(model_base, prompts, compute_activation_fn, tokens_to_cache, layers, foldername, batch_size, seq_len, shard_size, substrings, cache_type=cache_type)

    if dataset == "PKU-SafeRLHF":
        seq_len = 1024
        dataset_name = dataset
        prompts, substrings = load_pkusaferlhf_instructions_to_cache(model_alias, tokens_to_cache)
        # (1, 6467401, 2304) (1, 6467392, 2304)
        get_activations(model_base, prompts[0], substrings[0], seq_len, batch_size, tokens_to_cache, dataset_name, shard_size, label="safe",cache_type="after_substring")
        # (1, 8075346, 2304)
        get_activations(model_base, prompts[1], substrings[1], seq_len, batch_size, tokens_to_cache, dataset_name, shard_size, label="unsafe", cache_type="after_substring")
    elif dataset == 'pile':
        seq_len = 128
        dataset_name = dataset
        prompts, substrings = load_pile_data(tokens_to_cache)
        # (1, 462142, 2304)
        get_activations(model_base, prompts, substrings, seq_len, batch_size, tokens_to_cache, dataset_name, shard_size, label="random_token", cache_type="from_substring")
    elif dataset == 'safeedit':
        seq_len = 1024
        dataset_name = dataset
        prompts, substrings = load_safeedit_instructions_to_cache(model_alias, tokens_to_cache)
        get_activations(model_base, prompts[0], substrings[0], seq_len, batch_size, tokens_to_cache, dataset_name, shard_size, label="safe", cache_type="after_substring")
        get_activations(model_base, prompts[1], substrings[1], seq_len, batch_size, tokens_to_cache, dataset_name, shard_size, label="unsafe", cache_type="after_substring")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cache activations for a given model')
    parser.add_argument('--model_alias', type=str, default="gemma-2-9b-it", help='Alias of the model to use')
    parser.add_argument('--tokens_to_cache', type=str, default="<start_of_turn>model\n", help='How to find the position to cache. Options: "response", "entity", "last_eoi", "?", "random"')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--layers', type=str, default="20", help='layers to collect activations')
    parser.add_argument('--dataset', type=str, default="safeedit", help='Dataset to use. Options: "PKU-SafeRLHF", "wikidata", "triviaqa", "pile"')
    args = parser.parse_args()
    main(args)
    