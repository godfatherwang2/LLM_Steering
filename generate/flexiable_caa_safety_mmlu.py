import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from dataset.load_data import MMLUDataset
import json
import os
from transformers import StoppingCriteria

def get_few_shot_example():
    """
    返回一个few-shot示例，示意模型直接生成选项答案
    """
    return """Question: What is the capital of France?

A) London
B) Paris
C) Berlin
D) Madrid

Answer: B

Question: Which planet is closest to the Sun?

A) Venus
B) Earth
C) Mercury
D) Mars

Answer: C

"""

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)

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
        def hook_fn(module, input_tuple, output_tuple):
            # 解包输出（因为这是forward hook，在模块执行后调用）
            if isinstance(output_tuple, tuple):
                hidden_states = output_tuple[0]
            else:
                hidden_states = output_tuple
            
            # 检查是否还在引导限制内
            if self.generation_step < self.steer_token_limit:
                # 获取最后一个token的激活值
                current_activation = hidden_states[0, -1, :]
                steer_vector = self._calculate_steer_vector(current_activation)
                # 修改最后一个token的隐藏状态
                hidden_states[0, -1, :] = current_activation + steer_vector
                self.generation_step += 1
            
            # 返回修改后的输出
            if isinstance(output_tuple, tuple):
                return (hidden_states,) + output_tuple[1:]
            else:
                return hidden_states
        return hook_fn

def get_pred(logits, answer_tokens_idx):
    """
    从logits中获取预测结果
    Args:
        logits: 模型输出的logits
        answer_tokens_idx: 答案token的索引字典
    Returns:
        list: 每个选项的预测分数
    """
    predictions = []
    for answer, token_ids in answer_tokens_idx.items():
        # 计算每个选项的平均logit分数
        scores = [logits[token_id] for token_id in token_ids]
        avg_score = np.mean(scores)
        predictions.append((answer, avg_score))
    return predictions

def batch_infer(
    model_base: ModelBase,
    prompts: List[str],
    batch_size: int = 4,
    max_new_tokens: int = 1,
    return_score: bool = False,
    fwd_pre_hooks: List = [],
    fwd_hooks: List = [],
):
    """
    使用ModelBase结构进行批量推理
    
    Args:
        model_base: ModelBase实例
        prompts: 输入prompt列表
        batch_size: 批处理大小
        max_new_tokens: 最大生成token数
        return_score: 是否返回logits分数
        fwd_pre_hooks: 前向预钩子列表
        fwd_hooks: 前向钩子列表
    
    Returns:
        list: 推理结果列表
    """
    answers = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Batch inference"):
        batch_prompts = prompts[i:i+batch_size]
        
        # 使用ModelBase的tokenize_instructions_fn进行tokenization
        tokenized_inputs = model_base.tokenize_instructions_fn(instructions=batch_prompts)
        
        # 将输入移动到模型设备
        input_ids = tokenized_inputs.input_ids.to(model_base.model.device)
        attention_mask = tokenized_inputs.attention_mask.to(model_base.model.device)
        
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            with torch.no_grad():
                if return_score:
                    # 获取logits用于分数计算
                    model_base.model.config.return_dict_in_generate = True
                    outputs = model_base.model.generate(
                        do_sample=False,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                        output_scores=True,
                        max_new_tokens=1,
                    )
                    
                    # 获取最后一个位置的logits
                    last_logits = outputs.scores[0]  # [batch_size, vocab_size]
                    
                    for j in range(len(batch_prompts)):
                        answers.append({
                            "score": last_logits[j].cpu().detach().numpy()
                        })
                else:
                    # 生成文本
                    generation_config = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": False,
                        "pad_token_id": model_base.tokenizer.eos_token_id,
                    }
                    
                    outputs = model_base.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    
                    # 提取生成的token
                    input_length = input_ids.shape[1]
                    generated_tokens = outputs.sequences[:, input_length:]
                    
                    # 解码生成的文本
                    generated_texts = model_base.tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    
                    answers.extend(generated_texts)
        torch.cuda.empty_cache()
    return answers

@torch.no_grad()
def mmlu_evaluate(
    model_base: ModelBase,
    mmlu_dataset: MMLUDataset,
    steering_hook: ActivationSteeringHook,
    layer_to_steer: int,
    batch_size: int = 16,
    max_new_tokens: int = 1,
):
    """
    在MMLU数据集上评估模型性能
    """
    print("加载MMLU测试数据...")
    mmlu_test = mmlu_dataset.get_test_data()
    
    print("构建prompt列表...")
    prompt_tokens_list = []
    
    few_shot_example = get_few_shot_example()
    system_prompt = "The following are some multiple choice questions. Please answer with **only the letter of the correct option**.\n\n"
    
    for i in tqdm(range(len(mmlu_test)), desc="Processing prompts"):
        ques = mmlu_test[i]["prompt"]
        if args.model_alias == "gemma-2-9b-it":
            # 构建完整的提示词：系统提示 + few-shot示例 + 实际问题
            full_prompt = f"{few_shot_example}{ques}"
            ques = f"<start_of_turn>user\n{system_prompt}<end_of_turn>\n<start_of_turn>model\n{full_prompt}"
                prompt_tokens_list.append(ques)
        else:
            raise NotImplementedError("Only gemma-2-9b-it is supported for now")

    print("开始批量推理...")
    
    # 获取要挂钩的模块
    target_module = model_base.model_block_modules[layer_to_steer]
    hook_fn = steering_hook.create_hook_fn()
    
    mmlu_preds = batch_infer(
        model_base=model_base,
        prompts=prompt_tokens_list,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        return_score=True,
        fwd_hooks=[(target_module, hook_fn)],
    )

    print("计算准确率...")
    mmlu_acc = mmlu_dataset.get_accuracy(mmlu_preds, tokenizer=model_base.tokenizer)
    
    return mmlu_acc, mmlu_preds

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
    
    print("初始化MMLU数据集...")
    mmlu_dataset = MMLUDataset(data_dir=args.data_dir)
    
    print("开始MMLU评测...")
    mmlu_acc, mmlu_preds = mmlu_evaluate(
        model_base=model_base,
        mmlu_dataset=mmlu_dataset,
        steering_hook=steering_hook,
        layer_to_steer=args.layer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    
    print(f"MMLU准确率: {mmlu_acc:.4f}")
    
    # 保存结果
    results = {
        "mmlu_accuracy": mmlu_acc,
        "model_alias": args.model_alias,
        "dataset": args.dataset,
        "shared_size": args.shared_size,
        "layer": args.layer,
        "hook_point": args.hook_point,
        "top_k": args.top_k,
        "multiplier": args.multiplier,
        "norm_magnitude": args.norm_magnitude,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    
    # 在路径中添加top_k和norm_magnitude信息
    norm_info = "norm" if args.norm_magnitude else "no_norm"
    rst_dir = f"results/generations/{args.model_alias}/mmlu/safeedit/flexible_caa/topk_{args.top_k}_{norm_info}/"
    os.makedirs(rst_dir, exist_ok=True)
    
    output_filename = f"{args.dataset}_{args.layer}.json"
    with open(os.path.join(rst_dir, output_filename), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"结果已保存到: {os.path.join(rst_dir, output_filename)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用激活值操控进行MMLU评测')
    parser.add_argument('--model_alias', type=str, default="gemma-2-9b-it", help='模型别名')
    parser.add_argument('--dataset', type=str, default="safeedit", help='Dataset name used for caching')
    parser.add_argument('--shared_size', type=int, default=4050)
    parser.add_argument('--layer', type=int, default=20, help='要进行干预的层')
    parser.add_argument('--hook_point', type=str, default='block_output', help='用于引导的激活值来源')
    parser.add_argument('--top_k', type=int, default=10, help='FAISS检索的近邻数量')
    parser.add_argument('--multiplier', type=float, default=1.0, help='引导向量的强度系数')
    parser.add_argument('--data_dir', type=str, default="/home/wangxin/projects/steer-target-atoms-main/data/mmlu", help='MMLU数据集路径')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    parser.add_argument('--max_new_tokens', type=int, default=1, help='最大生成token数')
    parser.add_argument('--norm_magnitude', action='store_true', default=False, help='是否将steer_vector缩放到与CAA向量相同的范数')
    parser.add_argument('--qa',type=bool, default=True, help='是否使用QA格式')
    args = parser.parse_args()
    main(args) 