import sys
import os
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

import json
import os
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
import torch
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

from utils.hf_patching_utils import add_hooks
from utils.hf_models.model_base import ModelBase
from utils.hf_models.model_factory import construct_model_base
from utils.utils import model_alias_to_model_name
from utils.hf_models.gemma_model import format_instruction_gemma_chat
from dataset.load_data import MMLUDataset
import numpy as np

class STAVectorHook:
    """
    使用预计算STA向量的简单钩子管理器
    """
    def __init__(self, sta_vector: torch.Tensor, multiplier: float = 1.0, from_position: int = 0):
        """
        初始化STA向量钩子

        Args:
            sta_vector: 预计算的STA向量
            multiplier: 引导向量的强度系数
            from_position: 从哪个位置开始添加STA向量
        """
        print("--- 初始化STA向量钩子 ---")
        self.sta_vector = sta_vector
        self.multiplier = multiplier
        self.from_position = from_position

    def reset_state(self):
        """在每次新的生成任务开始前重置状态"""
        pass  # 对于STA向量，不需要重置状态

    def create_hook_fn(self):
        """创建并返回实际的钩子函数"""
        def hook_fn(module, input_tuple, output_tuple):
            # 解包输出（因为这是forward hook，在模块执行后调用）
            if isinstance(output_tuple, tuple):
                hidden_states = output_tuple[0]
            else:
                hidden_states = output_tuple
            
            # 获取position_ids（如果存在）
            position_ids = None
            if len(input_tuple) > 0 and hasattr(input_tuple[0], 'position_ids'):
                position_ids = input_tuple[0].position_ids
            elif len(input_tuple) > 2:
                position_ids = input_tuple[2]  # 假设position_ids在第三个位置
            
            # 如果找不到position_ids，使用简单的token计数
            if position_ids is None:
                seq_len = hidden_states.shape[1]
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            
            # 从指定位置开始添加STA向量
            from_pos = self.from_position
            mask = position_ids >= from_pos
            mask = mask.unsqueeze(-1).float()
            
            # 使用float32进行计算，避免精度问题
            steer_vector = self.multiplier * self.sta_vector.to(hidden_states.device)
            modified_hidden_states = hidden_states.to(torch.float32) + mask * steer_vector.to(torch.float32)
            modified_hidden_states = modified_hidden_states.to(hidden_states.dtype)
            
            # 返回修改后的输出
            if isinstance(output_tuple, tuple):
                return (modified_hidden_states,) + output_tuple[1:]
            else:
                return modified_hidden_states
        
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

def load_caa_vector(model_alias: str, dataset_name: str, layer: int, caa_vectors_dir: str = "results/caa_vectors") -> torch.Tensor:
    """
    加载预计算的CAA向量
    
    Args:
        model_alias: 模型别名
        dataset_name: 数据集名称
        layer: 层号
        caa_vectors_dir: CAA向量目录
    
    Returns:
        caa_vector: CAA向量
    """
    # 根据文件夹中的文件命名格式，直接使用层号作为文件名
    filename = f"{layer}.pt"
    filepath = os.path.join(caa_vectors_dir,model_alias+"_"+dataset_name, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CAA向量文件不存在: {filepath}")
    
    caa_vector = torch.load(filepath)
    print(f"加载CAA向量: {filepath}, 形状: {caa_vector.shape}")
    # 使用float32避免精度问题
    caa_vector = caa_vector.to(torch.float32)
    return caa_vector
    
def load_sta_vector(model_alias: str, dataset_name: str, layer: int, sta_vectors_dir: str = "results/sta_vectors") -> torch.Tensor:
    """
    加载预计算的STA向量
    """
    filepath = "/home/wangxin/projects/safety_steer/sae_analysis_results/gemma-2-9b-it_safeedit_layer_20_act_and_fre_trim_0.35.pt"#os.path.join(sta_vectors_dir,model_alias+"_"+dataset_name,filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"STA向量文件不存在: {filepath}")
    
    sta_vector = torch.load(filepath)
    print(f"加载STA向量: {filepath}, 形状: {sta_vector.shape}")
    return sta_vector

@torch.no_grad()
def mmlu_evaluate(
    model_base: ModelBase,
    mmlu_dataset: MMLUDataset,
    steering_hook: STAVectorHook,
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
    
    # 加载CAA向量和STA向量
    print("加载CAA向量...")
    caa_vector = load_caa_vector(
        model_alias=args.model_alias,
        dataset_name=args.dataset,
        layer=args.layer,
        caa_vectors_dir=args.caa_vectors_dir
    )
    
    print("加载STA向量...")
    sta_vector = load_sta_vector(
        model_alias=args.model_alias,
        dataset_name=args.dataset,
        layer=args.layer,
        sta_vectors_dir=args.sta_vectors_dir
    )
    
    # 归一化STA向量到与CAA向量相同的范数
    multiplier = torch.norm(caa_vector, p=2) / torch.norm(sta_vector, p=2)
    sta_vector = sta_vector * multiplier
    
    # 初始化STA向量钩子
    steering_hook = STAVectorHook(
        sta_vector=sta_vector,
        multiplier=args.multiplier,
        from_position=args.from_position
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
        "layer": args.layer,
        "multiplier": args.multiplier,
        "from_position": args.from_position,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    
    rst_dir = f"results/generations/{args.model_alias}/mmlu/safeedit/sta/"
    os.makedirs(rst_dir, exist_ok=True)
    
    output_filename = f"{args.dataset}_{args.layer}_multiplier_{args.multiplier}.json"
    with open(os.path.join(rst_dir, output_filename), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"结果已保存到: {os.path.join(rst_dir, output_filename)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用STA向量进行MMLU评测')
    parser.add_argument('--model_alias', type=str, default="gemma-2-9b-it", help='模型别名')
    parser.add_argument('--dataset', type=str, default="safeedit", help='数据集名称')
    parser.add_argument('--layer', type=int, default=20, help='要进行干预的层')
    parser.add_argument('--multiplier', type=float, default=1.0, help='STA向量的强度系数')
    parser.add_argument('--from_position', type=int, default=0, help='从哪个位置开始添加STA向量')
    parser.add_argument('--data_dir', type=str, default="/home/wangxin/projects/steer-target-atoms-main/data/mmlu", help='MMLU数据集路径')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    parser.add_argument('--max_new_tokens', type=int, default=1, help='最大生成token数')
    parser.add_argument('--caa_vectors_dir', type=str, default="results/caa_vectors", help='CAA向量目录')
    parser.add_argument('--sta_vectors_dir', type=str, default="results/sta_vectors", help='STA向量目录')
    args = parser.parse_args()
    main(args) 