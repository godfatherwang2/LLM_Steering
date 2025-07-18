import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
import glob
import faiss
from torch import Tensor
import torch
import numpy as np
import random
from torch.utils.data import Dataset, Sampler, DataLoader

from utils.hf_patching_utils import add_hooks
from utils.hf_models.model_base import ModelBase
from utils.hf_models.model_factory import construct_model_base
from utils.utils import clear_memory, model_alias_to_model_name, find_string_in_tokens

from tqdm import tqdm
from jaxtyping import Float, Int
random.seed(10)

DEFAULT_CACHE_DIR = os.path.join("dataset/cached_activations")

def _get_activations_pre_hook(cache: Float[Tensor, "pos d_model"]):
    def hook_fn(module, input):
        nonlocal cache
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[:, :] += activation[:, :].to(cache)
    return hook_fn

@torch.no_grad()
def _get_activations_fixed_seq_len(model, tokenizer, prompts: List[str], block_modules: List[torch.nn.Module], seq_len: int = 512, layers: List[int]=None, batch_size=32, save_device: Union[torch.device, str] = "cuda", verbose=True) -> Tuple[Float[Tensor, 'n seq_len'], Float[Tensor, 'n layer seq_len d_model']]:
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

        fwd_pre_hooks = [
            (block_modules[layer], _get_activations_pre_hook(cache=activations[i:i+inputs_len, layer_idx, :num_input_toks, :])) 
            for layer_idx, layer in enumerate(layers)
        ]

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
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
                      tokens_to_cache: str, layers: List[int],foldername: str,
                      batch_size: int = 32, seq_len: int = 128, shard_size: int = 1000,
                      substrings: List[str] = None, cache_type="after_substring") -> None:

    d_model = model_base.model.config.hidden_size
    if layers is None:
        n_layers = model_base.model.config.num_hidden_layers
    else:
        n_layers = len(layers)

    # 第一阶段：预计算总token数
    total_tokens = 0
    valid_prompts = []
    valid_substrings = []
    valid_start_pos = []
    valid_lens = []
    valid_ids = []
    for prompt, substring in tqdm(zip(prompts, substrings)):
        inputs = model_base.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
        # print("#"*50)
        # print(substring)
        # print("#"*50)
        # print(inputs.input_ids)
        
        # print(model_base.tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))
        # print(model_base.tokenizer.decode(inputs.input_ids[0]))
        # print("#"*50)
        if substring in model_base.tokenizer.decode(inputs.input_ids[0]):
            # 计算该样本有效token数并累加
            s = find_string_in_tokens(substring, inputs.input_ids[0], model_base.tokenizer) 
            if s is None:
                continue
            non_padding_mask = (inputs.input_ids[0] != model_base.tokenizer.pad_token_id)
            non_padding_indices = torch.nonzero(non_padding_mask, as_tuple=True)[0]
            if len(non_padding_indices) == 0:
                continue
            last_non_padding_index = non_padding_indices[-1].item()
            
            # 计算子字符串后的起始位置
            if cache_type == "after_substring":
                start_pos = s.stop
            elif cache_type == "from_substring":
                start_pos = s.start
            else:
                raise ValueError
            
            # 确保起始位置不超过序列长度
            if start_pos >= seq_len:
                continue
            
            # 计算有效长度（子字符串后到最后一个非padding token）
            valid_length = min(last_non_padding_index - start_pos + 1, seq_len - start_pos)
            assert valid_length > 0, f"substring: {substring}\nprompt: {prompt}\n last_non_padding_index: {last_non_padding_index} start_pos: {start_pos} seq_len: {seq_len}\n {model_base.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])}"
            total_tokens += valid_length
            valid_prompts.append(prompt)
            valid_substrings.append(substring)
            valid_start_pos.append(start_pos)
            valid_lens.append(valid_length)
            valid_ids.append(inputs.input_ids[0][start_pos:start_pos+valid_length])
        # print(valid_length)
        # exit()
    print((n_layers, total_tokens, d_model))

    # 第二阶段：存储拼接后的激活值
    memmap_file_acts = np.memmap(f"{foldername}/acts.dat", dtype='float32', mode='w+', 
                            shape=(n_layers, total_tokens, d_model))
    memmap_file_ids = np.memmap(f"{foldername}/ids.dat", dtype='int32', mode='w+', 
                             shape=(total_tokens))
    current_idx = 0
    for i in tqdm(range(0, len(valid_prompts), batch_size), desc="Computing activations"):
        batch_prompts = valid_prompts[i:i+batch_size]
        batch_start_pos = valid_start_pos[i:i+batch_size]
        batch_valid_len = valid_lens[i:i+batch_size]
        batch_valid_ids = valid_ids[i:i+batch_size]

        input_ids, activations = compute_activation_fn(prompts=batch_prompts)
        for j in range(len(activations)):
            try:
                act = activations[j]
                start_pos = batch_start_pos[j]
                valid_len = batch_valid_len[j]
                valid_act = act[:, start_pos:start_pos+valid_len, :]  # 截取有效部分
                memmap_file_acts[:, current_idx:current_idx+valid_len, :] = valid_act.float().cpu().numpy()
                memmap_file_ids[current_idx:current_idx+valid_len] = batch_valid_ids[j].cpu().numpy()
                current_idx += valid_len
            except:
                print('tokens', model_base.tokenizer.convert_ids_to_tokens(input_ids[j]))
                print('token(s) cached:', model_base.tokenizer.decode(input_ids[j, start_pos:start_pos+valid_len]))
                print('start_pos:', start_pos, 'valid_length:', valid_len)
                print(' batch_valid_ids:',  batch_valid_ids[j])
            if i == 0 and j == 0:
                print('tokens', model_base.tokenizer.convert_ids_to_tokens(input_ids[j]))
                print('token(s) cached:', model_base.tokenizer.decode(input_ids[j, start_pos:start_pos+valid_len]))
                print('start_pos:', start_pos, 'valid_length:', valid_len)
        gc.collect()
    # Flush changes to disk
    memmap_file_acts.flush()
    memmap_file_ids.flush()

    # full_input_ids = model_base.tokenizer(prompts, return_tensors="pt", padding=True).input_ids[:,:seq_len]    
    # slices_to_cache = [find_string_in_tokens(substring, full_input_ids[j], model_base.tokenizer) for j, substring in enumerate(substrings)]
    # discarded_indexes = [i for i, s in enumerate(slices_to_cache) if s is None]

    # # Create memmap files

    # memmap_file_acts = np.memmap(f"{foldername}/acts.dat", dtype='float32', mode='w+', 
    #                         shape=(shard_size, n_layers, seq_len, d_model))
    # memmap_file_ids = np.memmap(f"{foldername}/ids.dat", dtype='float32', mode='w+', 
    #                         shape=(shard_size, seq_len))
    # memmap_file_lengths = np.memmap(f"{foldername}/lengths.dat", dtype='int32', mode='w+',
    #                                 shape=(shard_size,))

    # total_n = 0
    # for i in tqdm(range(0, len(prompts), batch_size), desc="Computing activations"):

    #     batch_prompts = prompts[i:i+batch_size]
    #     batch_substrings_tmp = substrings[i:i+batch_size] if substrings is not None else None
    #     # remove bos
    #     batch_prompts_tmp = [prompt.replace('<bos>', '') for prompt in batch_prompts]

    #     ### Check if the substrings are in the input ids
    #     inputs_ = model_base.tokenizer(batch_prompts_tmp, return_tensors="pt", padding=True, truncation=True, max_length=seq_len, truncation_side="right")
    #     batch_prompts = []
    #     batch_substrings = []
    #     for p in range(len(batch_prompts_tmp)):
    #         ids = inputs_.input_ids[p]
    #         prompt_til_seq_len = model_base.tokenizer.decode(ids)
    #         if batch_substrings_tmp[p] in prompt_til_seq_len:
    #             batch_prompts.append(batch_prompts_tmp[p])
    #             batch_substrings.append(batch_substrings_tmp[p])
        
    #     # Get input ids and activations
    #     input_ids, activations = compute_activation_fn(prompts=batch_prompts)
        

    #     # Filter activations based on substrings
    #     if tokens_to_cache is not None:
    #         # Get slices of activations to cache
    #         slices_to_cache = [find_string_in_tokens(substring, input_ids[j], model_base.tokenizer) for j, substring in enumerate(batch_substrings)]
    #         activations_sliced_list = []
    #         input_ids_list = []
    #         valid_lengths = []
    #         for j, s in enumerate(slices_to_cache):
    #             # TODO: this may fail if seq_len is short
    #             if s is None:
    #                 continue
    #             non_padding_mask = (input_ids[j] != model_base.tokenizer.pad_token_id)
    #             non_padding_indices = torch.nonzero(non_padding_mask, as_tuple=True)[0]
    #             if len(non_padding_indices) == 0:
    #                 continue
    #             last_non_padding_index = non_padding_indices[-1].item()
                
    #             # 计算子字符串后的起始位置
    #             start_pos = s.stop
                
    #             # 确保起始位置不超过序列长度
    #             if start_pos >= seq_len:
    #                 continue
                
    #             # 计算有效长度（子字符串后到最后一个非padding token）
    #             valid_length = min(last_non_padding_index - start_pos + 1, seq_len - start_pos)
    #             valid_length = max(valid_length, 1)  # 至少保留一个位置
                
    #             # 存储实际有效长度
    #             valid_lengths.append(valid_length)
                
    #             # 提取从起始位置到末尾的激活值
    #             activations_sliced = activations[j, :, start_pos:start_pos+valid_length, :]

    #             if i == 0 and j == 0:
    #                 print('tokens', model_base.tokenizer.convert_ids_to_tokens(input_ids[j]))
    #                 print('token(s) cached:', model_base.tokenizer.decode(input_ids[j, start_pos:start_pos+valid_length]))
    #                 print('start_pos:', start_pos, 'valid_length:', valid_length)
    #             activations_sliced_list.append(activations_sliced)
    #             input_ids_list.append(input_ids[j])

    #          # 将激活值列表拼接为张量
    #         if activations_sliced_list:
    #             # 找出最大有效长度
    #             max_length = max(valid_lengths)
                
    #             # 创建填充后的激活值张量
    #             padded_activations = torch.zeros(len(activations_sliced_list), n_layers, max_length, d_model)
    #             for idx, act in enumerate(activations_sliced_list):
    #                 padded_activations[idx, :, :act.shape[1], :] = act
                
    #             activations = padded_activations
    #             input_ids = torch.stack(input_ids_list, dim=0)
    #         else:
    #             activations = torch.tensor([])
    #             input_ids = torch.tensor([])

    #     added_batch_size = activations.shape[0]
    #     if added_batch_size == 0:
    #         continue
    #     end_idx = min(shard_size,total_n+added_batch_size)
    #     if end_idx == shard_size:
    #         # if reaching the end of the shard, take the remaining examples
    #         n = end_idx - total_n
    #     else:
    #         # add the full batch
    #         n = added_batch_size
    #     # TODO: avoid upcast to float?
    #     memmap_file_acts[total_n:end_idx] = activations[:n].float().cpu().numpy()
    #     memmap_file_ids[total_n:end_idx] = input_ids[:n].cpu().numpy()
    #     memmap_file_lengths[total_n:end_idx] = np.array(valid_lengths, dtype='int32')
    #     total_n += n
    
    #     if end_idx == shard_size:
    #         print('Files loaded')
    #         print('total_n', total_n)
    #         break
    # # Flush changes to disk
    # memmap_file_acts.flush()
    # memmap_file_ids.flush()
# %%
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
    model_base.tokenizer.truncation_side="left"
    model_base.tokenizer.padding_side = 'left'

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

        foldername = f"{DEFAULT_CACHE_DIR}/{tokens_to_cache}/{model_alias}_{dataset_name}/{tokens_to_cache}_label_{label}_shard_size_{shard_size}"
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
        prompts, substrings = load_safeedit_queries(model_alias, tokens_to_cache)
        get_activations(model_base, prompts, substrings, seq_len, batch_size, tokens_to_cache, dataset_name, shard_size, label="safe", cache_type="after_substring")
    else:
        raise ValueError(f"Unknown dataset name: {dataset}")


if __name__ == '__main__':
    from dataset.load_data import load_pkusaferlhf_queries, balance_data, load_safeedit_queries
    '''
    Pile activations: python -m utils.activation_cache --model_alias gemma-2-2b --tokens_to_cache random --batch_size 128 --dataset pile
    Wikidata activations: python -m utils.activation_cache --model_alias gemma-2-2b --tokens_to_cache entity --batch_size 128  --dataset wikidata
    '''
    datas = load_safeedit_queries(model_alias="gemma-2-9b-it",apply_chat_format=True)
    parser = argparse.ArgumentParser(description='Cache activations for a given model')
    parser.add_argument('--model_alias', type=str, default="gemma-2-9b-it", help='Alias of the model to use')
    parser.add_argument('--tokens_to_cache', type=str, default="response", help='How to find the position to cache. Options: "response", "entity", "last_eoi", "?", "random"')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--layers', type=str, help='layers to collect activations')
    parser.add_argument('--dataset', type=str, default="safeedit", help='Dataset to use. Options: "PKU-SafeRLHF", "wikidata", "triviaqa", "pile", "safeedit"')
    args = parser.parse_args()

    main(args)
