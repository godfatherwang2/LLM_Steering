import sys
sys.path.append('./')
sys.path.append('../')
import torch
import faiss
import numpy as np
import argparse
from typing import List
from tqdm import tqdm
# 假设您的项目结构如下，以便正确导入模块
from utils.hf_models.model_factory import construct_model_base
from utils.hf_patching_utils import add_hooks
from utils.hf_models.model_base import ModelBase
from utils.utils import model_alias_to_model_name
from utils.hf_models.gemma_model import format_instruction_gemma_chat
from dataset.load_data import load_safeedit_test_data
import json
import os

class ActivationSteeringHook:
    """
    一个有状态的钩子管理器，用于在模型生成时进行实时激活值操控。
    """
    def __init__(self, safe_index_path: str, unsafe_index_path: str, top_k: int = 10, multiplier: float = 1.0, steer_token_limit: int = 5, data_path_info: tuple = None, norm_magnitude: bool = False, caa_vectors_dir: str = "results/caa_vectors", model_alias: str = None, dataset_name: str = None, layer: int = None):
        """
        初始化激活值操控钩子。

        Args:
            safe_index_path: 安全激活值FAISS索引文件的路径。
            unsafe_index_path: 不安全激活值FAISS索引文件的路径。
            top_k: 从FAISS中检索的最近邻数量。
            multiplier: 引导向量的强度系数。
            steer_token_limit: 只在前k个生成的token上进行引导。
            norm_magnitude: 是否将steer_vector缩放到与CAA向量相同的范数。
            caa_vectors_dir: CAA向量目录。
            model_alias: 模型别名。
            dataset_name: 数据集名称。
            layer: 层号。
        """
        print("--- 初始化激活值操控钩子 ---")
        self.top_k = top_k
        self.multiplier = multiplier
        self.steer_token_limit = steer_token_limit
        self.norm_magnitude = norm_magnitude
        
        # 如果启用norm_magnitude，加载CAA向量
        if self.norm_magnitude:
            if caa_vectors_dir and model_alias and dataset_name and layer is not None:
                self.caa_vector = self._load_caa_vector(caa_vectors_dir, model_alias, dataset_name, layer)
                print(f"加载CAA向量用于范数归一化，范数: {torch.norm(self.caa_vector, p=2):.6f}")
            else:
                raise ValueError("启用norm_magnitude时需要提供caa_vectors_dir, model_alias, dataset_name, layer参数")
        else:
            self.caa_vector = None
        
        # 加载FAISS索引
        print(f"加载 'safe' FAISS 索引: {safe_index_path}")
        self.safe_index = faiss.read_index(safe_index_path)
        print(f"加载 'unsafe' FAISS 索引: {unsafe_index_path}")
        self.unsafe_index = faiss.read_index(unsafe_index_path)

        cached_safe,cached_unsafe,metadata_safe,metadata_unsafe = data_path_info
        n_layers, total_tokens_safe, d_model = metadata_safe["activations_original_shape"]
        n_layers, total_tokens_unsafe, d_model = metadata_unsafe["activations_original_shape"]

        self.safe_array = np.ascontiguousarray(np.memmap(cached_safe, dtype='float32', mode='r',shape=(total_tokens_safe,d_model)))
        self.unsafe_array = np.ascontiguousarray(np.memmap(cached_unsafe, dtype='float32', mode='r',shape=(total_tokens_unsafe,d_model)))
        
        self.generation_step = 0 # 用于跟踪生成token的计数器

    def _load_caa_vector(self, caa_vectors_dir: str, model_alias: str, dataset_name: str, layer: int) -> torch.Tensor:
        """
        加载CAA向量
        
        Args:
            caa_vectors_dir: CAA向量目录
            model_alias: 模型别名
            dataset_name: 数据集名称
            layer: 层号
        
        Returns:
            caa_vector: CAA向量
        """
        filename = f"{model_alias}_{dataset_name}_layer_{layer}_caa_vector.pt"
        filepath = os.path.join(caa_vectors_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CAA向量文件不存在: {filepath}")
        
        caa_vector = torch.load(filepath)
        print(f"加载CAA向量: {filepath}, 形状: {caa_vector.shape}")
        return caa_vector

    def reset_state(self):
        """在每次新的生成任务开始前重置状态。"""
        self.generation_step = 0

    def _calculate_steer_vector(self, activation: torch.Tensor) -> torch.Tensor:
        """
        计算引导向量 (已修正所有已知bug)。
        """
        # a. 准备查询向量 (处理bfloat16)
        query_vector = activation.detach().cpu().float().numpy().reshape(1, -1)
        # b. 搜索最近邻
        safe_dists, safe_indices = self.safe_index.search(query_vector, self.top_k)
        unsafe_dists, unsafe_indices = self.unsafe_index.search(query_vector, self.top_k)
        # c. 使用正确的索引重构向量
        #safe_vectors = torch.from_numpy(self.safe_index.reconstruct_batch(safe_indices[0])).to(activation.device, dtype=activation.dtype)
        #unsafe_vectors = torch.from_numpy(self.unsafe_index.reconstruct_batch(unsafe_indices[0])).to(activation.device, dtype=activation.dtype)
        safe_vectors = torch.from_numpy(self.safe_array[safe_indices[0]]).to(activation.device, dtype=activation.dtype)
        unsafe_vectors = torch.from_numpy(self.unsafe_array[unsafe_indices[0]]).to(activation.device, dtype=activation.dtype)

        safe_dists_accurate = torch.linalg.norm(safe_vectors - activation, dim=-1)
        unsafe_dists_accurate = torch.linalg.norm(unsafe_vectors - activation, dim=-1)

        # d. 加权平均
        safe_weights = torch.softmax(-safe_dists_accurate, dim=-1)
        unsafe_weights = torch.softmax(-unsafe_dists_accurate, dim=-1)
        avg_safe_vector = (safe_weights.unsqueeze(0) @ safe_vectors).squeeze(0)
        avg_unsafe_vector = (unsafe_weights.unsqueeze(0) @ unsafe_vectors).squeeze(0)
        
        # e. 计算引导向量
        steer_vector =  (avg_safe_vector - avg_unsafe_vector)
        
        # 如果启用norm_magnitude，将steer_vector缩放到与CAA向量相同的范数
        if self.norm_magnitude and self.caa_vector is not None:
            caa_norm = torch.norm(self.caa_vector, p=2)
            steer_norm = torch.norm(steer_vector, p=2)
            
            if steer_norm > 0:  # 避免除零
                norm_multiplier = caa_norm / steer_norm
                steer_vector = steer_vector * norm_multiplier
        steer_vector = self.multiplier * steer_vector
        return steer_vector.to(activation.dtype)

    def create_hook_fn(self):
        """创建并返回实际的钩子函数 (已修正)。"""
        def hook_fn(_,intput_tuple):
            # 解包Gemma的输出 (hidden_states, past_key_value_cache)
            if(len(intput_tuple) != 1):
                hidden_states, past_key_values = intput_tuple[0],intput_tuple[1]
            else:
                hidden_states, past_key_values = intput_tuple[0],None
            # 只在逐字生成阶段且未超过引导限制时操作
            if hidden_states.shape[1] == 1 and self.generation_step < self.steer_token_limit:
                current_activation = hidden_states[0, -1, :]
                steer_vector = self._calculate_steer_vector(current_activation)
                hidden_states[0, -1, :] = current_activation + steer_vector
                self.generation_step += 1
            
            if(past_key_values is not None):
                return (hidden_states, past_key_values)
            else:
                return (hidden_states,)
        return hook_fn

@torch.no_grad()
def steered_generate(
    model_base: ModelBase,
    steering_hook: ActivationSteeringHook,
    prompt: str,
    layer_to_steer: int,
    hook_point: str,
    max_new_tokens: int = 50,
):
    """
    使用激活值操控钩子来生成文本。
    """
    tokenizer = model_base.tokenizer
    model = model_base.model

    # 1. 使用您的 get_modules 方法获取要挂钩的模块
    target_module = model_base.model_block_modules[layer_to_steer]
    # 2. 重置钩子状态并准备输入
    inputs = tokenizer(prompt, return_tensors="pt")
    steering_hook.reset_state()
    # 3. 添加钩子并执行生成
    hook_fn = steering_hook.create_hook_fn()

    with add_hooks(module_forward_pre_hooks=[(target_module,hook_fn)], module_forward_hooks=[]):
        outputs = model.generate(
            inputs.input_ids.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)


def main(args):
    print("加载模型...")
    model_base = construct_model_base(model_alias_to_model_name[args.model_alias])
    tokenizer = model_base.tokenizer
    # --- 2. 准备FAISS索引路径 ---
    base_cache_dir = "dataset/cached_activations"
    safe_index_path = os.path.join(base_cache_dir,args.model_alias+"_"+args.dataset,"label_safe_shard_size_"+str(args.shared_size),f"faiss_index_layer_{args.layer}.index")
    unsafe_index_path = os.path.join(base_cache_dir,args.model_alias+"_"+args.dataset,"label_unsafe_shard_size_"+str(args.shared_size),f"faiss_index_layer_{args.layer}.index")


    cached_safe = os.path.join(base_cache_dir,args.model_alias+"_"+args.dataset,"label_safe_shard_size_"+str(args.shared_size),f"acts_{args.layer}.dat")
    cached_unsafe = os.path.join(base_cache_dir,args.model_alias+"_"+args.dataset,"label_unsafe_shard_size_"+str(args.shared_size),f"acts_{args.layer}.dat")
    
    metadata_safe = json.load(open(os.path.join(base_cache_dir,args.model_alias+"_"+args.dataset,"label_safe_shard_size_"+str(args.shared_size),f"metadata.json")))
    metadata_unsafe = json.load(open(os.path.join(base_cache_dir,args.model_alias+"_"+args.dataset,"label_unsafe_shard_size_"+str(args.shared_size),f"metadata.json")))
    data_path_info = (cached_safe,cached_unsafe,metadata_safe,metadata_unsafe)

    # --- 3. 初始化引导钩子 ---
    steering_hook = ActivationSteeringHook(
        safe_index_path=safe_index_path,
        unsafe_index_path=unsafe_index_path,
        top_k=args.top_k,
        multiplier=args.multiplier,
        steer_token_limit=args.max_new_tokens,
        data_path_info=data_path_info,
        norm_magnitude=args.norm_magnitude,
        caa_vectors_dir="results/caa_vectors",
        model_alias=args.model_alias,
        dataset_name=args.dataset,
        layer=args.layer
    )
    
    if args.test_dataset == "safeedit":
        queries = load_safeedit_test_data(model_alias=args.model_alias,apply_chat_format=True)
    else:
        raise ValueError(f"Invalid test dataset: {args.test_dataset}")

    questions = []
    generations = []
    for query in tqdm(queries, desc="Generating outputs"):
        steered_output = steered_generate(
            model_base,
            steering_hook,
            query['prompt'],
            layer_to_steer=args.layer,
            hook_point=args.hook_point,
            max_new_tokens=args.max_new_tokens
        )
        pure_output = steered_output.split("<end_of_turn>\n<start_of_turn>model\n")[1]
        generations.append(pure_output)
        questions.append(query['prompt'].replace("<end_of_turn>\n<start_of_turn>model\n", "").replace("<start_of_turn>user\n", ""))

    final_data = [{'question': q, 'generation': g} for q, g in zip(questions, generations)]
    # 在路径中添加top_k和norm_magnitude信息
    norm_info = "norm" if args.norm_magnitude else "no_norm"
    rst_dir = f"results/generations/{args.model_alias}/{args.test_dataset}/flexible_caa/topk_{args.top_k}_{norm_info}"   
    os.makedirs(rst_dir, exist_ok=True)
    with open(os.path.join(rst_dir, f"{args.dataset}_{args.layer}.json"), "w") as f:
        json.dump(final_data, f,indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用激活值操控进行引导式生成')
    parser.add_argument('--model_alias', type=str, default="gemma-2-9b-it", help='模型别名')
    parser.add_argument('--dataset', type=str, default="safeedit", help='Dataset name used for caching')
    parser.add_argument('--shared_size', type=int, default=4050)
    parser.add_argument('--layer', type=int, default=24, help='要进行干预的层')
    parser.add_argument('--hook_point', type=str, default='block_output', help='用于引导的激活值来源')
    parser.add_argument('--top_k', type=int, default=10, help='FAISS检索的近邻数量')
    parser.add_argument('--multiplier', type=float, default=1.0, help='引导向量的强度系数')
    parser.add_argument('--test_dataset', type=str, default="safeedit")
    parser.add_argument('--max_new_tokens', type=str, default=50)
    parser.add_argument('--norm_magnitude', action='store_true', default=False, help='是否将steer_vector缩放到与CAA向量相同的范数')
    args = parser.parse_args()
    main(args)