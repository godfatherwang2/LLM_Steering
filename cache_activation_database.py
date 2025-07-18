# activation_cache.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from typing import List, Dict, Callable, Union, Tuple
from functools import partial
import gc
import json
import numpy as np
import random
from torch import Tensor
from tqdm import tqdm
from jaxtyping import Float, Int
import argparse

# 导入必要的辅助工具和模型模块
from utils.hf_patching_utils import add_hooks
from utils.hf_models.model_base import ModelBase
from utils.hf_models.model_factory import construct_model_base
from utils.utils import model_alias_to_model_name

# --- 全局设置 ---
random.seed(42)
torch.set_grad_enabled(False)
DEFAULT_CACHE_DIR = os.path.join("dataset/cached_activations_safeedit_response_top5")

def inspect_model_hooks(model: torch.nn.Module):
    found_hooks = False
    print("--- 开始检查模型钩子 ---")
    # model.named_modules() 会返回一个包含模块名称和模块对象的迭代器
    for name, module in model.named_modules():
        # 检查三种主要的钩子类型
        forward_pre_hooks = list(module._forward_pre_hooks.values())
        forward_hooks = list(module._forward_hooks.values())
        backward_hooks = list(module._backward_hooks.values())
        # 如果任何一种钩子存在，就打印出来
        if forward_pre_hooks or forward_hooks or backward_hooks:
            found_hooks = True
            print(f"\n[模块]: {name}")
            if forward_pre_hooks:
                print(f"  - Forward Pre-Hooks ({len(forward_pre_hooks)}个): {forward_pre_hooks}")
            if forward_hooks:
                print(f"  - Forward Hooks ({len(forward_hooks)}个): {forward_hooks}")
            if backward_hooks:
                print(f"  - Backward Hooks ({len(backward_hooks)}个): {backward_hooks}")
    if not found_hooks:
        print("\n未在模型中发现任何钩子。")
    print("\n--- 检查完毕 ---")

def get_activations(
    model_base: ModelBase, 
    prompts: List[str],
    layers: List[int],
    hook_points: List[str],
    seq_len: int,
) -> Tuple[Int[Tensor, 'batch seq_len'], Dict[str, Float[Tensor, 'batch layer seq_len d_model']]]:
    """
    为一个批次的prompts，在指定的层和钩子点上，提取激活值。
    此版本现在可以智能地处理pre-hooks和post-hooks。
    """
    n_prompts = len(prompts)
    n_layers = len(layers)
    d_model = model_base.model.config.hidden_size
    tokenizer = model_base.tokenizer

    activations_dict = {hp: torch.zeros((n_prompts, n_layers, seq_len, d_model), device='cuda', dtype=torch.float32) for hp in hook_points}
    
    inputs = tokenizer(prompts, return_tensors="pt", padding='longest', truncation=True, max_length=seq_len)
    input_ids = inputs.input_ids.to(model_base.model.device)
    attention_mask = inputs.attention_mask.to(model_base.model.device)

    # --- 智能设置 pre-hooks 和 post-hooks ---
    fwd_pre_hooks = []
    fwd_post_hooks = []
    
    for hp in hook_points:
        # get_modules 现在返回 (模块, 钩子类型)
        modules_and_types = model_base.get_modules(hp, layers)
        
        for i, (module, hook_type) in enumerate(modules_and_types):
            cache_slice = activations_dict[hp][:, i, :, :]

            if hook_type == 'pre':
                # 为pre-hook创建钩子函数 (捕获输入)
                def create_pre_hook(cache):
                    def hook_fn(_, input_tensor):
                        act = input_tensor[0].clone().to(cache.device, cache.dtype)
                        cache[:act.shape[0], :act.shape[1]] += act
                    return hook_fn
                fwd_pre_hooks.append((module, create_pre_hook(cache_slice)))
            
            elif hook_type == 'post':
                # 为post-hook创建钩子函数 (捕获输出)
                def create_post_hook(cache):
                    def hook_fn(_, input_tensor, output_tensor):
                        act = output_tensor[0].clone().to(cache.device, cache.dtype)
                        cache[:act.shape[0], :act.shape[1]] += act
                    return hook_fn
                fwd_post_hooks.append((module, create_post_hook(cache_slice)))

    # --- 执行前向传播并捕获激活值 ---
    # add_hooks 需要能同时接受两种钩子
    with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_post_hooks):
        #inspect_model_hooks(model_base.model)
        model_base.model(input_ids=input_ids, attention_mask=attention_mask)
    
    cpu_activations = {hp: act.cpu() for hp, act in activations_dict.items()}
    return input_ids.cpu(), cpu_activations

def cache_response_activations(
    model_base: "ModelBase",
    queries: List[Dict],
    layers: List[int],
    hook_points: List[str],
    top_k: Union[int, str],
    foldername: str,
    batch_size: int = 32,
    seq_len: int = 1024
):
    """
    针对给定的查询，提取回答部分指定数量token的、指定模块的激活值并保存。
    此版本整合了所有修正，能够精确处理带有左填充的批处理数据。

    Args:
        model_base: 封装好的模型和tokenizer。
        queries: 包含 'prompt' 和 'response' 的字典列表。
        layers: 需要提取的层列表。
        hook_points: 需要提取激活值的模块名称列表 (例如: 'pre_mlp', 'block_output')。
        top_k: 缓存回答的前k个token，或使用 "all" 缓存全部。
        foldername: 保存缓存文件的目录路径。
        batch_size: 处理时使用的批次大小。
        seq_len: 模型支持的最大序列长度。
    """
    tokenizer = model_base.tokenizer
    
    print(f"--- 开始缓存任务 ---")
    print(f"目标目录: {foldername}")
    print(f"缓存模式: top_k = {top_k}, 提取模块: {hook_points}")

    # 1. 预计算和过滤有效数据
    valid_queries, response_lengths = [], []
    for query in tqdm(queries, desc="[1/4] 预计算和过滤数据"):
        # 确保拼接后的长度不超过最大序列长度
        if len(tokenizer.encode(query['prompt'] + query['response'], add_special_tokens=False)) > seq_len:
            continue
        valid_queries.append(query)
        response_lengths.append(len(tokenizer.encode(query['response'], add_special_tokens=False)))

    num_valid = 32#len(valid_queries)
    if num_valid == 0:
        print("错误: 没有找到有效的查询，程序退出。")
        return

    # 2. 计算总共需要缓存的token数量
    if top_k == "all":
        total_tokens_to_cache = sum(response_lengths)
    else:
        # 只缓存实际存在的token
        total_tokens_to_cache = sum(min(top_k, length) for length in response_lengths)
    
    print(f"找到 {num_valid} 条有效数据，总计将缓存 {total_tokens_to_cache} 个token的激活值。")

    # 3. 创建存储文件
    os.makedirs(foldername, exist_ok=True)
    keys_memmaps = {
        hp: np.memmap(f"{foldername}/keys_{hp}.dat", dtype='float32', mode='w+', 
                      shape=(total_tokens_to_cache, len(layers), model_base.model.config.hidden_size))
        for hp in hook_points
    }
    values_filepath = f"{foldername}/values.jsonl"
    
    # 4. 核心处理循环
    current_token_idx = 0
    with open(values_filepath, 'w') as f_values:
        for i in tqdm(range(0, num_valid, batch_size), desc="[2/4] 计算激活值"):
            # 准备当前批次的数据
            batch_queries = valid_queries[i:i+batch_size]
            batch_response_lengths = response_lengths[i:i+batch_size]
            batch_full_text = [q['prompt'] + q['response'] for q in batch_queries]
            
            # 获取一个批次的激活值和经过padding的token_ids
            input_ids_padded, activations_dict = get_activations(
                model_base, batch_full_text, layers, hook_points, seq_len
            )

            # 预计算无填充的prompt长度
            prompts_only = [q['prompt'] for q in batch_queries]
            prompt_token_lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts_only]

            # [3/4] 精确提取并写入数据
            for j in range(len(batch_queries)):
                # --- 精确计算起始位置的核心逻辑 ---
                current_padded_ids = input_ids_padded[j]
                non_padding_indices = torch.nonzero(current_padded_ids != tokenizer.pad_token_id, as_tuple=True)[0]
                
                if len(non_padding_indices) == 0: continue # 跳过全padding的序列
                
                padding_offset = non_padding_indices[0].item()
                response_start_corrected = padding_offset + prompt_token_lengths[j]
                
                # --- 确定缓存数量 ---
                num_to_cache = batch_response_lengths[j] if top_k == "all" else min(top_k, batch_response_lengths[j])
                if num_to_cache == 0: continue
                response_end_corrected = response_start_corrected + num_to_cache

                # --- 存储Keys ---
                for hp in hook_points:
                    target_activations = activations_dict[hp][j, :, response_start_corrected:response_end_corrected, :]
                    keys_memmaps[hp][current_token_idx : current_token_idx + num_to_cache] = target_activations.permute(1, 0, 2).numpy()

                # --- 存储Values ---
                target_token_ids = input_ids_padded[j, response_start_corrected:response_end_corrected]
                for k in range(num_to_cache):
                    value_record = {
                        "token_text": tokenizer.decode([target_token_ids[k]]),
                        "token_id": target_token_ids[k].item(),
                        "token_index_in_response": k
                    }
                    f_values.write(json.dumps(value_record) + '\n')

                current_token_idx += num_to_cache

    # 5. 确保所有数据都写入磁盘
    for memmap in tqdm(keys_memmaps.values(), desc="[4/4] 刷新文件到磁盘"):
        memmap.flush()
        
    print(f"--- 缓存任务完成！---")

def cache_query_as_key_v2(
    model_base: ModelBase,
    queries: List[Dict],
    layers: List[int],
    hook_points: List[str],
    top_k: Union[int, str],
    foldername: str,
    batch_size: int = 32,
    seq_len: int = 1024
):
    """
    第二版缓存函数。
    提取问题(prompt)最后一个token的激活值作为Key，
    并提取回答(response)的前k个token作为Value。
    """
    tokenizer = model_base.tokenizer
    
    print(f"--- 开始缓存任务 (V2: Query as Key) ---")
    print(f"目标目录: {foldername}")
    print(f"Key: Prompt最后一个token的激活值, Value: Response前{top_k}个token")

    # 1. 预计算和过滤有效数据
    valid_queries = []
    for query in tqdm(queries, desc="[1/4] 预计算和过滤数据"):
        # 确保问题和回答本身都不会超长
        if len(tokenizer.encode(query['prompt'])) > seq_len or len(tokenizer.encode(query['response'])) == 0:
            continue
        valid_queries.append(query)

    num_valid = len(valid_queries)
    if num_valid == 0:
        print("错误: 没有找到有效的查询，程序退出。"); return
    
    print(f"找到 {num_valid} 条有效数据用于缓存。")

    # 2. 创建存储文件
    # Key的数量等于有效查询的数量
    os.makedirs(foldername, exist_ok=True)
    keys_memmaps = {
        hp: np.memmap(f"{foldername}/keys_{hp}.dat", dtype='float32', mode='w+', 
                      shape=(num_valid, len(layers), model_base.model.config.hidden_size))
        for hp in hook_points
    }
    values_filepath = f"{foldername}/values.jsonl"
    
    # 3. 核心处理循环
    current_query_idx = 0
    with open(values_filepath, 'w') as f_values:
        for i in tqdm(range(0, num_valid, batch_size), desc="[2/4] 计算激活值"):
            batch_queries = valid_queries[i:i+batch_size]
            
            # --- Key 的提取 ---
            # 只将问题(prompt)送入模型
            prompts_only = [q['prompt'] for q in batch_queries]
            input_ids_padded, activations_dict = get_activations(
                model_base, prompts_only, layers, hook_points, seq_len
            )

            # [3/4] 提取Key和Value并写入文件
            for j in range(len(batch_queries)):
                current_padded_ids = input_ids_padded[j]
                non_padding_indices = torch.nonzero(current_padded_ids != tokenizer.pad_token_id, as_tuple=True)[0]
                if len(non_padding_indices) == 0: continue
                
                # 找到prompt最后一个非填充token的索引
                last_token_idx = non_padding_indices[-1].item()

                # 存储Key (在最后一个token位置的激活值)
                for hp in hook_points:
                    last_token_activation = activations_dict[hp][j, :, last_token_idx, :]
                    keys_memmaps[hp][current_query_idx] = last_token_activation.numpy()

                # --- Value 的提取 ---
                response_text = batch_queries[j]['response']
                response_token_ids = tokenizer.encode(response_text, add_special_tokens=False)
                
                num_to_store = len(response_token_ids) if top_k == "all" else min(top_k, len(response_token_ids))
                
                value_tokens_info = []
                for k in range(num_to_store):
                    token_id = response_token_ids[k]
                    value_tokens_info.append({
                        "token_text": tokenizer.decode([token_id]),
                        "token_id": token_id,
                        "token_index_in_response": k
                    })
                
                # 将整个Value列表作为一个JSON对象写入
                f_values.write(json.dumps({"response_tokens": value_tokens_info}) + '\n')
                
                current_query_idx += 1

    # 4. 确保所有数据都写入磁盘
    for memmap in tqdm(keys_memmaps.values(), desc="[4/4] 刷新文件到磁盘"):
        memmap.flush()
        
    print(f"--- 缓存任务完成 (V2)！---")

from typing import Literal
def load_safeedit_queries(
    model_alias: str,
    apply_chat_format: bool=False,
):
    if '/' in model_alias:
        model_alias = model_alias.split('/')[-1]

    from datasets import load_dataset
    dataset = load_dataset("zjunlp/SafeEdit")['train']
    
    if apply_chat_format:
        if "gemma" in model_alias:
            from utils.hf_models.gemma_model import format_instruction_gemma_chat 
            format_instructions_chat_fn = partial(format_instruction_gemma_chat, system=None, include_trailing_whitespace=True)
        elif "Llama-3" in model_alias:
            from utils.hf_models.llama3_model import format_instruction_llama3_chat
            format_instructions_chat_fn = partial(format_instruction_llama3_chat, system=None, include_trailing_whitespace=True)
        else:
            raise ValueError(f"Invalid model alias: {model_alias}")
    
    queries = []
    for _d in dataset:
        queries.append({"prompt": _d["adversarial prompt"], "response": _d["safe generation"], "label": True})
        queries.append({"prompt": _d["adversarial prompt"], "response": _d["unsafe generation"], "label": False})

    if apply_chat_format:
        for q in tqdm(queries, desc="应用聊天模板"):
            q['prompt'] = format_instructions_chat_fn(instruction=q['prompt'])
    return queries


# --- 主执行函数 ---
def main(args):
    model_alias = args.model_alias
    model_path = model_alias_to_model_name.get(model_alias, model_alias)
    
    print("构建模型...")
    model_base = construct_model_base(model_path)
    model_base.tokenizer.pad_token = model_base.tokenizer.eos_token
    model_base.tokenizer.padding_side = 'left' # 使用左填充以正确处理回答部分的激活值
    
    print("加载并格式化数据集...")
    all_queries = load_safeedit_queries(model_alias, apply_chat_format=True)
    
    safe_queries = [q for q in all_queries if q['label'] is True]
    unsafe_queries = [q for q in all_queries if q['label'] is False]
    
    safe_queries = safe_queries
    unsafe_queries = unsafe_queries

    layers_to_cache = list(map(int, args.layers.split(',')))
    hook_points_to_cache = args.hook_points.split(',')
    
    # 处理 top_k 参数
    top_k_val = "all" if args.top_k == "all" else int(args.top_k)

    safe_foldername = os.path.join(DEFAULT_CACHE_DIR, model_alias, "safe")
    unsafe_foldername = os.path.join(DEFAULT_CACHE_DIR, model_alias, "unsafe")
    # --- 调用重构后的缓存函数 ---
    
    cache_response_activations(
        model_base, safe_queries, layers_to_cache, hook_points_to_cache, top_k_val, safe_foldername, args.batch_size, args.seq_len
    )
    
    cache_response_activations(
        model_base, unsafe_queries, layers_to_cache, hook_points_to_cache, top_k_val, unsafe_foldername, args.batch_size, args.seq_len
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='为数据集缓存指定模块和数量的激活值')
    parser.add_argument('--model_alias', type=str, default="gemma-2-9b-it", help='模型别名')
    parser.add_argument('--layers', type=str, default="24", help='层列表，逗号分隔')
    parser.add_argument('--hook_points', type=str, default='block_output', help="提取激活值的模块点，逗号分隔 (例如: 'pre_attn,post_attn,pre_mlp')")
    parser.add_argument('--top_k', type=str, default='5', help='缓存回答的前k个token，或使用 "all" 缓存全部')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--seq_len', type=int, default=1024, help='最大序列长度')
    args = parser.parse_args()
    main(args)