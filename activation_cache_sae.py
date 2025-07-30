import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
sys.path.append('./')
sys.path.append('../')

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
from dataclasses import dataclass
from transformer_lens import HookedTransformer

from utils.utils import clear_memory, model_alias_to_model_name, find_string_in_tokens
from sae_utils import load_gemma_2_sae, load_sae_from_dir

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

DEFAULT_CACHE_DIR = os.path.join("dataset/cached_activations_sae")

@dataclass
class SAECacheConfig:
    """SAE缓存配置类"""
    model_alias: str = "gemma-2-9b-it"
    dataset: str = "safeedit"
    tokens_to_cache: str = "<start_of_turn>model\n"
    batch_size: int = 16
    layers: str = "20"
    seq_len: int = 1024
    sae_path: str = "/home/wangxin/models/gemma-scope-9b-it-res/layer_20/width_16k/average_l0_91"
    cache_type: str = "after_substring"  # "after_substring" or "from_substring"


@torch.no_grad()
def get_activations_and_sae_features_fixed_seq_len_transformer_lens(
    model: HookedTransformer, sae_model, prompts: List[str], 
    layers: List[int] = None, seq_len: int = 512, 
    save_device: Union[torch.device, str] = "cuda", verbose=True
) -> Tuple[Int[Tensor, 'n seq_len'], Float[Tensor, 'n layer seq_len d_model'], Float[Tensor, 'n layer seq_len sae_features']]:
    """使用transformer_lens获取固定序列长度的激活值和SAE特征"""
    torch.cuda.empty_cache()

    if layers is None:
        layers = range(model.cfg.n_layers)

    n_layers = len(layers)
    d_model = model.cfg.d_model
    sae_features_dim = sae_model.W_enc.shape[-1]  # 使用最后一维作为特征维度

    # 存储激活值和SAE特征
    activations = torch.zeros((len(prompts), n_layers, seq_len, d_model), device=save_device)
    sae_features = torch.zeros((len(prompts), n_layers, seq_len, sae_features_dim), device=save_device)
    all_input_ids = torch.zeros((len(prompts), seq_len), dtype=torch.long, device=save_device)

    # 填充pad token
    all_input_ids.fill_(model.tokenizer.pad_token_id)

    # 直接处理整个batch
    batch_tokens = model.tokenizer(prompts, padding=True, truncation=True, max_length=seq_len, return_tensors="pt")
    input_ids = batch_tokens["input_ids"].to(model.cfg.device)
    attention_mask = batch_tokens["attention_mask"].to(model.cfg.device)

    num_input_toks = input_ids.shape[-1]
    
    # 为每一层获取激活值
    for layer_idx, layer_num in enumerate(layers):
        layer_name = f"blocks.{layer_num}.hook_resid_post"
        
        # 设置SAE的hook来获取激活值
        filter_not_input = lambda name: "_input" not in name
        sae_cache = {}

        def sae_fwd_hook(act, hook):
            sae_cache[hook.name] = act.detach()
        
        sae_model.reset_hooks()
        sae_model.add_hook(filter_not_input, sae_fwd_hook, "fwd")

        # 设置模型的hook来替换激活值
        def reconstr_direct(activations, hook):
            raw_device = activations.device
            sae_cache["activations"] = activations.detach()            
            output = sae_model(activations).to(raw_device)
            return output

        model.reset_hooks()

        with torch.no_grad():
            direct_output = model.run_with_hooks(
                input_ids,
                attention_mask=attention_mask,
                fwd_hooks=[
                    (
                        layer_name,
                        reconstr_direct,
                    ),
                ],
                return_type="loss",
            )
            
            # 获取原始激活值和SAE特征
            layer_activations = sae_cache["activations"]
            layer_sae_features = sae_cache["hook_sae_acts_post"]
            
            # 保存到对应的缓存位置
            activations[:, layer_idx, :num_input_toks, :] = layer_activations
            sae_features[:, layer_idx, :num_input_toks, :] = layer_sae_features

        # 清理hooks
        sae_model.reset_hooks()
        model.reset_hooks()

    all_input_ids[:, :num_input_toks] = input_ids

    return all_input_ids, activations, sae_features


def get_compute_activation_and_sae_fn_transformer_lens(model: HookedTransformer, sae_model, layers: List[int] = None, seq_len: int = 128):
    """获取计算激活值和SAE特征的函数（使用transformer_lens）"""
    def compute_activation_and_sae_fn(prompts: List[str]):
        nonlocal model, sae_model, layers, seq_len
        n_layers = model.cfg.n_layers
        if layers is None:
            layers = range(n_layers)

        input_ids, activations, sae_features = get_activations_and_sae_features_fixed_seq_len_transformer_lens(
            model,
            sae_model,
            prompts=prompts,
            layers=layers,
            seq_len=seq_len,
            save_device='cuda',
            verbose=False
        )
        
        activations: Float[Tensor, 'n n_layers seq_len d_model'] = activations.cpu()
        sae_features: Float[Tensor, 'n n_layers seq_len sae_features'] = sae_features.cpu()
        input_ids: Int[Tensor, 'n seq_len'] = input_ids.cpu()

        n_layers = activations.shape[1]
        assert n_layers == len(layers), f"Expected {len(layers)} layers, but got {n_layers}"

        return input_ids, activations, sae_features

    return compute_activation_and_sae_fn


def cache_activations_and_sae_features_transformer_lens(
    model: HookedTransformer, sae_model, prompts: List[str], 
    compute_activation_and_sae_fn: Callable, tokens_to_cache: str, 
    layers: List[int], foldername: str, batch_size: int = 32, 
    seq_len: int = 128, shard_size: int = 1000, substrings: List[str] = None, 
    cache_type="after_substring"
) -> None:
    """缓存激活值和SAE特征（使用transformer_lens）"""
    d_model = model.cfg.d_model
    sae_features_dim = sae_model.W_enc.shape[-1]
    
    if layers is None:
        n_layers = model.cfg.n_layers
        layers = list(range(n_layers))
    else:
        n_layers = len(layers)
    
    # === 第一阶段：预计算 total_tokens ===
    total_tokens = 0
    valid_prompts = []
    valid_substrings = []
    valid_start_pos = []
    valid_lens = []
    valid_ids = []
    
    for prompt, substring in tqdm(zip(prompts, substrings), desc="Preprocessing and calculating size", total=len(prompts)):
        inputs = model.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
        if substring in model.tokenizer.decode(inputs["input_ids"][0]):
            s = find_string_in_tokens(substring, inputs["input_ids"][0], model.tokenizer) 
            if s is None: 
                continue
            non_padding_mask = (inputs["input_ids"][0] != model.tokenizer.pad_token_id)
            non_padding_indices = torch.nonzero(non_padding_mask, as_tuple=True)[0]
            if len(non_padding_indices) == 0: 
                continue
            last_non_padding_index = non_padding_indices[-1].item()
            if cache_type == "after_substring": 
                start_pos = s.stop
            elif cache_type == "from_substring": 
                start_pos = s.start
            else: 
                raise ValueError
            if start_pos >= seq_len: 
                continue
            valid_length = min(last_non_padding_index - start_pos + 1, seq_len - start_pos)
            if valid_length <= 0: 
                continue
            total_tokens += valid_length
            valid_prompts.append(prompt)
            valid_substrings.append(substring)
            valid_start_pos.append(start_pos)
            valid_lens.append(valid_length)
            valid_ids.append(inputs["input_ids"][0][start_pos:start_pos+valid_length])
    
    print(f"Total tokens to be cached: {total_tokens}")

    # === 创建元数据文件 ===
    metadata = {
        "description": "Metadata for cached activations and SAE features.",
        "activations_original_shape": (n_layers, total_tokens, d_model),
        "sae_features_shape": (n_layers, total_tokens, sae_features_dim),
        "token_ids_shape": (total_tokens,),
        "layers_cached": layers,
        "sae_features_dim": sae_features_dim,
        "d_model": d_model
    }
    metadata_filepath = os.path.join(foldername, "metadata.json")
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_filepath}")
    
    # === 创建样本索引范围文件 ===
    sample_indices = []
    current_idx = 0
    for i, valid_len in enumerate(valid_lens):
        start_idx = current_idx
        end_idx = current_idx + valid_len
        sample_indices.append({
            "sample_id": i,
            "start_index": start_idx,
            "end_index": end_idx,
            "length": valid_len
        })
        current_idx = end_idx
    
    sample_indices_filepath = os.path.join(foldername, "sample_indices.json")
    with open(sample_indices_filepath, 'w') as f:
        json.dump(sample_indices, f, indent=4)
    print(f"Sample indices saved to {sample_indices_filepath}")
    
    # === 第二阶段：存储激活值和SAE特征 ===
    # 为每一层创建独立的memmap文件
    memmap_files_acts = {}
    memmap_files_sae = {}
    
    for layer_idx, layer_num in enumerate(layers):
        # 激活值文件
        acts_filepath = os.path.join(foldername, f"acts_{layer_num}.dat")
        print(f"Creating memmap file for activations layer {layer_num} at: {acts_filepath}")
        memmap_files_acts[layer_idx] = np.memmap(acts_filepath, dtype='float32', mode='w+',
                                                 shape=(total_tokens, d_model))
        
        # SAE特征文件
        sae_filepath = os.path.join(foldername, f"sae_{layer_num}.dat")
        print(f"Creating memmap file for SAE features layer {layer_num} at: {sae_filepath}")
        memmap_files_sae[layer_idx] = np.memmap(sae_filepath, dtype='float32', mode='w+',
                                                shape=(total_tokens, sae_features_dim))
    
    # Token IDs文件
    memmap_file_ids = np.memmap(os.path.join(foldername, "ids.dat"), dtype='int32', mode='w+',
                             shape=(total_tokens,))
    
    # 重置current_idx用于实际写入数据
    current_idx = 0
    for i in tqdm(range(0, len(valid_prompts), batch_size), desc="Computing and writing activations and SAE features"):
        batch_prompts = valid_prompts[i:i+batch_size]
        batch_start_pos = valid_start_pos[i:i+batch_size]
        batch_valid_len = valid_lens[i:i+batch_size]
        batch_valid_ids = valid_ids[i:i+batch_size]

        input_ids, activations, sae_features = compute_activation_and_sae_fn(prompts=batch_prompts)
        
        for j in range(len(activations)):
            act = activations[j]
            sae_feat = sae_features[j]
            start_pos = batch_start_pos[j]
            valid_len = batch_valid_len[j]
            valid_act_all_layers = act[:, start_pos:start_pos+valid_len, :]
            valid_sae_all_layers = sae_feat[:, start_pos:start_pos+valid_len, :]

            # 循环遍历每一层，并写入对应的文件
            for layer_idx in range(n_layers):
                layer_act_slice = valid_act_all_layers[layer_idx, :, :]
                layer_sae_slice = valid_sae_all_layers[layer_idx, :, :]
                
                memmap_files_acts[layer_idx][current_idx:current_idx+valid_len, :] = layer_act_slice.float().cpu().numpy()
                memmap_files_sae[layer_idx][current_idx:current_idx+valid_len, :] = layer_sae_slice.float().cpu().numpy()
            
            # 写入 token ID
            memmap_file_ids[current_idx:current_idx+valid_len] = batch_valid_ids[j].cpu().numpy()
            current_idx += valid_len
        
        gc.collect()

    # Flush all memmap files to disk
    print("Flushing all memmap files to disk...")
    for memmap_file in memmap_files_acts.values():
        memmap_file.flush()
    for memmap_file in memmap_files_sae.values():
        memmap_file.flush()
    memmap_file_ids.flush()
    print("Caching complete.")


def cache_activations_and_sae_features_with_query_cache_transformer_lens(
    model: HookedTransformer, sae_model, prompts: List[str], 
    compute_activation_and_sae_fn: Callable, tokens_to_cache: str, 
    layers: List[int], foldername: str, batch_size: int = 32, 
    seq_len: int = 128, shard_size: int = 1000, substrings: List[str] = None, 
    cache_type="after_substring"
) -> None:
    """缓存激活值和SAE特征，包括查询后的隐藏状态和生成部分的平均值（使用transformer_lens）"""
    d_model = model.cfg.d_model
    sae_features_dim = sae_model.W_enc.shape[-1]
    
    if layers is None:
        n_layers = model.cfg.n_layers
        layers = list(range(n_layers))
    else:
        n_layers = len(layers)
    
    # === 第一阶段：预计算 total_tokens ===
    total_tokens = 0
    valid_prompts = []
    valid_substrings = []
    valid_start_pos = []
    valid_lens = []
    valid_ids = []
    
    for prompt, substring in tqdm(zip(prompts, substrings), desc="Preprocessing and calculating size", total=len(prompts)):
        inputs = model.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
        if substring in model.tokenizer.decode(inputs["input_ids"][0]):
            s = find_string_in_tokens(substring, inputs["input_ids"][0], model.tokenizer) 
            if s is None: 
                continue
            non_padding_mask = (inputs["input_ids"][0] != model.tokenizer.pad_token_id)
            non_padding_indices = torch.nonzero(non_padding_mask, as_tuple=True)[0]
            if len(non_padding_indices) == 0: 
                continue
            last_non_padding_index = non_padding_indices[-1].item()
            if cache_type == "after_substring": 
                start_pos = s.stop - 1
            elif cache_type == "from_substring": 
                start_pos = s.start
            else: 
                raise ValueError
            if start_pos >= seq_len: 
                continue
            valid_length = min(last_non_padding_index - start_pos + 1, seq_len - start_pos)
            if valid_length <= 0: 
                continue
            # 对于新函数，我们只缓存从start_pos+1开始的部分，所以实际长度要减1
            actual_valid_length = max(0, valid_length - 1)
            total_tokens += actual_valid_length
            valid_prompts.append(prompt)
            valid_substrings.append(substring)
            valid_start_pos.append(start_pos)
            valid_lens.append(actual_valid_length)  # 使用实际长度
            valid_ids.append(inputs["input_ids"][0][start_pos+1:start_pos+valid_length])  # 从start_pos+1开始
    
    print(f"Total tokens to be cached: {total_tokens}")

    # === 创建元数据文件 ===
    metadata = {
        "description": "Metadata for cached activations and SAE features with query cache.",
        "activations_original_shape": (n_layers, total_tokens, d_model),
        "sae_features_shape": (n_layers, total_tokens, sae_features_dim),
        "token_ids_shape": (total_tokens,),
        "layers_cached": layers,
        "sae_features_dim": sae_features_dim,
        "d_model": d_model
    }
    metadata_filepath = os.path.join(foldername, "metadata.json")
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {metadata_filepath}")
    
    # === 创建样本索引范围文件 ===
    sample_indices = []
    current_idx = 0
    for i, valid_len in enumerate(valid_lens):
        start_idx = current_idx
        end_idx = current_idx + valid_len
        sample_indices.append({
            "sample_id": i,
            "start_index": start_idx,
            "end_index": end_idx,
            "length": valid_len
        })
        current_idx = end_idx
    
    sample_indices_filepath = os.path.join(foldername, "sample_indices.json")
    with open(sample_indices_filepath, 'w') as f:
        json.dump(sample_indices, f, indent=4)
    print(f"Sample indices saved to {sample_indices_filepath}")
    
    # === 第二阶段：存储激活值和SAE特征 ===
    # 为每一层创建独立的memmap文件
    memmap_files_acts = {}
    memmap_files_sae = {}
    
    for layer_idx, layer_num in enumerate(layers):
        # 激活值文件
        acts_filepath = os.path.join(foldername, f"acts_{layer_num}.dat")
        print(f"Creating memmap file for activations layer {layer_num} at: {acts_filepath}")
        memmap_files_acts[layer_idx] = np.memmap(acts_filepath, dtype='float32', mode='w+',
                                                 shape=(total_tokens, d_model))
        
        # SAE特征文件
        sae_filepath = os.path.join(foldername, f"sae_{layer_num}.dat")
        print(f"Creating memmap file for SAE features layer {layer_num} at: {sae_filepath}")
        memmap_files_sae[layer_idx] = np.memmap(sae_filepath, dtype='float32', mode='w+',
                                                shape=(total_tokens, sae_features_dim))
    
    # Token IDs文件
    memmap_file_ids = np.memmap(os.path.join(foldername, "ids.dat"), dtype='int32', mode='w+',
                             shape=(total_tokens,))
    
    # 新增：查询后隐藏状态文件（每个样本一个向量）
    query_acts_filepath = os.path.join(foldername, "query_acts.dat")
    print(f"Creating memmap file for query activations at: {query_acts_filepath}")
    query_acts_memmap = np.memmap(query_acts_filepath, dtype='float32', mode='w+',
                                  shape=(len(valid_prompts), n_layers, d_model))
    
    # 新增：平均激活值文件（每个样本一个向量）
    avg_acts_filepath = os.path.join(foldername, "avg_acts.dat")
    print(f"Creating memmap file for average activations at: {avg_acts_filepath}")
    avg_acts_memmap = np.memmap(avg_acts_filepath, dtype='float32', mode='w+',
                                shape=(len(valid_prompts), n_layers, d_model))
    
    # 新增：平均SAE特征文件（每个样本一个向量）
    avg_sae_filepath = os.path.join(foldername, "avg_sae.dat")
    print(f"Creating memmap file for average SAE features at: {avg_sae_filepath}")
    avg_sae_memmap = np.memmap(avg_sae_filepath, dtype='float32', mode='w+',
                               shape=(len(valid_prompts), n_layers, sae_features_dim))
    
    # 重置current_idx用于实际写入数据
    current_idx = 0
    for i in tqdm(range(0, len(valid_prompts), batch_size), desc="Computing and writing activations and SAE features"):
        batch_prompts = valid_prompts[i:i+batch_size]
        batch_start_pos = valid_start_pos[i:i+batch_size]
        batch_valid_len = valid_lens[i:i+batch_size]
        batch_valid_ids = valid_ids[i:i+batch_size]

        input_ids, activations, sae_features = compute_activation_and_sae_fn(prompts=batch_prompts)
        
        for j in range(len(activations)):
            act = activations[j]
            sae_feat = sae_features[j]
            start_pos = batch_start_pos[j]
            valid_len = batch_valid_len[j]
            sample_idx = i + j
            
            # 1. 保存查询后的隐藏状态（start_pos位置）
            query_act_all_layers = act[:, start_pos:start_pos+1, :]  # 只取start_pos位置
            query_acts_memmap[sample_idx, :, :] = query_act_all_layers.squeeze(1).float().cpu().numpy()
            
            # 2. 保存从start_pos+1开始的隐藏状态和SAE特征
            if valid_len > 0:  # 确保有生成部分
                valid_act_all_layers = act[:, start_pos+1:start_pos+1+valid_len, :]  # 从start_pos+1开始，长度为valid_len
                valid_sae_all_layers = sae_feat[:, start_pos+1:start_pos+1+valid_len, :]  # 从start_pos+1开始，长度为valid_len
                
                # 循环遍历每一层，并写入对应的文件
                for layer_idx in range(n_layers):
                    layer_act_slice = valid_act_all_layers[layer_idx, :, :]
                    layer_sae_slice = valid_sae_all_layers[layer_idx, :, :]
                    memmap_files_acts[layer_idx][current_idx:current_idx+valid_len, :] = layer_act_slice.float().cpu().numpy()
                    memmap_files_sae[layer_idx][current_idx:current_idx+valid_len, :] = layer_sae_slice.float().cpu().numpy()

                # 3. 计算并保存平均值
                avg_acts_memmap[sample_idx, :, :] = valid_act_all_layers.mean(dim=1).float().cpu().numpy()
                avg_sae_memmap[sample_idx, :, :] = valid_sae_all_layers.mean(dim=1).float().cpu().numpy()
                
                # 写入 token ID（从start_pos+1开始）
                memmap_file_ids[current_idx:current_idx+valid_len] = batch_valid_ids[j].cpu().numpy()
                current_idx += valid_len
            else:
                # 如果只有start_pos位置，用零填充平均值
                avg_acts_memmap[sample_idx, :, :] = 0.0
                avg_sae_memmap[sample_idx, :, :] = 0.0
    
        gc.collect()

    # Flush all memmap files to disk
    print("Flushing all memmap files to disk...")
    for memmap_file in memmap_files_acts.values():
        memmap_file.flush()
    for memmap_file in memmap_files_sae.values():
        memmap_file.flush()
    memmap_file_ids.flush()
    query_acts_memmap.flush()
    avg_acts_memmap.flush()
    avg_sae_memmap.flush()
    print("Caching complete.")

def load_safeedit_instructions_to_cache(model_alias: str, tokens_to_cache: str, entity_type: str = None, balance_data: bool = True):
    """加载safeedit数据集用于缓存"""
    from dataset.load_data import load_safeedit_queries
    queries = load_safeedit_queries(model_alias=model_alias, apply_chat_format=True)[:8]
    substrings = [tokens_to_cache for query in queries]
    prompts = ([query["prompt"] for query in queries if query['label']==True], [query["prompt"] for query in queries if query['label']==False])
    substrings = (substrings[:len(prompts[0])], substrings[len(prompts[0]):])
    return prompts, substrings

def main(args):
    torch.set_grad_enabled(False)

    config = SAECacheConfig(
        model_alias=args.model_alias,
        dataset=args.dataset,
        tokens_to_cache=args.tokens_to_cache,
        batch_size=args.batch_size,
        layers=args.layers,
        seq_len=args.seq_len,
        sae_path=args.sae_path,
        cache_type=args.cache_type
    )
    
    layers = list(map(int, config.layers.split(',')))
    model_path = model_alias_to_model_name[config.model_alias]
    
    print("Loading model and SAE...")
    # 使用transformer_lens加载模型
    model = HookedTransformer.from_pretrained(model_path, n_devices=1, torch_dtype=torch.bfloat16)
    model.eval()
    model.reset_hooks()
    
    # 加载SAE模型
    if "gemma" in config.model_alias.lower():
        sae_model, sparsity = load_gemma_2_sae(config.sae_path, device="cuda")
    else:
        sae_model, sparsity = load_sae_from_dir(config.sae_path, device="cuda")
    
    sae_model.eval()
    
    # 设置tokenizer参数
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.truncation_side = "right"
    model.tokenizer.padding_side = 'right'
    
    def get_activations_and_sae_transformer_lens(model, prompts, substrings, seq_len, batch_size, tokens_to_cache, dataset_name, shard_size, label, cache_type):
        print('len prompts', len(prompts))
        print('prompt:', prompts[0])
        print('substring:', substrings[0])

        compute_activation_and_sae_fn = get_compute_activation_and_sae_fn_transformer_lens(
            model, sae_model, layers=layers, seq_len=seq_len
        )

        if shard_size is None:
            shard_size = len(prompts)
        else:
            shard_size = min(shard_size, len(prompts))

        print('shard_size', shard_size)

        foldername = f"{DEFAULT_CACHE_DIR}/{config.model_alias}_{dataset_name}/label_{label}_shard_size_{shard_size}"
        os.makedirs(foldername, exist_ok=True)

        cache_activations_and_sae_features_transformer_lens(
            model, sae_model, prompts, compute_activation_and_sae_fn, 
            tokens_to_cache, layers, foldername, batch_size, seq_len, 
            shard_size, substrings, cache_type=cache_type
        )

    if config.dataset == 'safeedit':
        seq_len = config.seq_len
        dataset_name = config.dataset
        prompts, substrings = load_safeedit_instructions_to_cache(config.model_alias, config.tokens_to_cache)
        
        get_activations_and_sae_transformer_lens(
            model, prompts[0], substrings[0], seq_len, config.batch_size, 
            config.tokens_to_cache, dataset_name, None, label="safe", 
            cache_type=config.cache_type
        )
        get_activations_and_sae_transformer_lens(
            model, prompts[1], substrings[1], seq_len, config.batch_size, 
            config.tokens_to_cache, dataset_name, None, label="unsafe", 
            cache_type=config.cache_type
        )
    else:
        raise ValueError(f"Dataset {config.dataset} not supported")


def main_with_query_cache(args):
    """新的main函数，包含查询缓存功能"""
    torch.set_grad_enabled(False)

    config = SAECacheConfig(
        model_alias=args.model_alias,
        dataset=args.dataset,
        tokens_to_cache=args.tokens_to_cache,
        batch_size=args.batch_size,
        layers=args.layers,
        seq_len=args.seq_len,
        sae_path=args.sae_path,
        cache_type=args.cache_type
    )
    
    layers = list(map(int, config.layers.split(',')))
    model_path = model_alias_to_model_name[config.model_alias]
    
    print("Loading model and SAE...")
    # 使用transformer_lens加载模型
    model = HookedTransformer.from_pretrained(model_path, n_devices=1, torch_dtype=torch.bfloat16)
    model.eval()
    model.reset_hooks()
    
    # 加载SAE模型
    if "gemma" in config.model_alias.lower():
        sae_model, sparsity = load_gemma_2_sae(config.sae_path, device="cuda")
    else:
        sae_model, sparsity = load_sae_from_dir(config.sae_path, device="cuda")
    
    sae_model.eval()
    
    # 设置tokenizer参数
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.truncation_side = "right"
    model.tokenizer.padding_side = 'right'
    
    def get_activations_and_sae_with_query_cache_transformer_lens(model, prompts, substrings, seq_len, batch_size, tokens_to_cache, dataset_name, shard_size, label, cache_type):
        print('len prompts', len(prompts))
        print('prompt:', prompts[0])
        print('substring:', substrings[0])

        compute_activation_and_sae_fn = get_compute_activation_and_sae_fn_transformer_lens(
            model, sae_model, layers=layers, seq_len=seq_len
        )

        if shard_size is None:
            shard_size = len(prompts)
        else:
            shard_size = min(shard_size, len(prompts))

        print('shard_size', shard_size)

        foldername = f"{DEFAULT_CACHE_DIR}/{config.model_alias}_{dataset_name}/label_{label}_shard_size_{shard_size}"
        os.makedirs(foldername, exist_ok=True)

        cache_activations_and_sae_features_with_query_cache_transformer_lens(
            model, sae_model, prompts, compute_activation_and_sae_fn, 
            tokens_to_cache, layers, foldername, batch_size, seq_len, 
            shard_size, substrings, cache_type=cache_type
        )

    if config.dataset == 'safeedit':
        seq_len = config.seq_len
        dataset_name = config.dataset
        prompts, substrings = load_safeedit_instructions_to_cache(config.model_alias, config.tokens_to_cache)
        
        get_activations_and_sae_with_query_cache_transformer_lens(
            model, prompts[0], substrings[0], seq_len, config.batch_size, 
            config.tokens_to_cache, dataset_name, None, label="safe", 
            cache_type=config.cache_type
        )
        get_activations_and_sae_with_query_cache_transformer_lens(
            model, prompts[1], substrings[1], seq_len, config.batch_size, 
            config.tokens_to_cache, dataset_name, None, label="unsafe", 
            cache_type=config.cache_type
        )
    else:
        raise ValueError(f"Dataset {config.dataset} not supported")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cache activations and SAE features for a given model')
    parser.add_argument('--model_alias', type=str, default="gemma-2-9b-it", help='Alias of the model to use')
    parser.add_argument('--tokens_to_cache', type=str, default="<start_of_turn>model\n", help='How to find the position to cache')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--layers', type=str, default="20", help='layers to collect activations')
    parser.add_argument('--dataset', type=str, default="safeedit", help='Dataset to use')
    parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length')
    parser.add_argument('--sae_path', type=str, default="/home/wangxin/models/gemma-scope-9b-it-res/layer_20/width_16k/average_l0_91", help='Path to SAE model')
    parser.add_argument('--cache_type', type=str, default="after_substring", help='Cache type: after_substring or from_substring')
    parser.add_argument('--with_query_cache', default=True, help='Use new caching function with query cache')
    
    args = parser.parse_args()
    
    if args.with_query_cache:
        main_with_query_cache(args)
    else:
        main(args)
