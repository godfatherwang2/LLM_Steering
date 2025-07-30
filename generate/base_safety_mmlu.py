import sys
sys.path.append('./')
sys.path.append('../')
import os
import torch
import numpy as np
import argparse
from typing import List, Dict
from tqdm import tqdm
from datasets import load_dataset
import json

# 假设您的项目结构如下，以便正确导入模块
from utils.hf_models.model_factory import construct_model_base
from utils.hf_models.model_base import ModelBase
from utils.utils import model_alias_to_model_name
from dataset.load_data import MMLUDataset
from utils.hf_patching_utils import add_hooks

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

# MMLU数据集相关的提示词模板
MMLU_SYSTEM_PROMPT = "The following are some multiple choice questions. Please answer with **only the letter of the correct option**.\n\n"

MMLU_CASE_PROMPT = """Question: {question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Answer: """

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
                    input_length = input_ids.shape[1]
                    generated_tokens = outputs.sequences[:, input_length:]
                    generated_texts = model_base.tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    # 获取最后一个位置的logits
                    scores = outputs.scores[0].cpu().detach().numpy()  # [batch_size, vocab_size]
                    
                    for j in range(len(generated_texts)):
                        answers.append({
                            "score": scores[j]
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
    system_prompt = MMLU_SYSTEM_PROMPT
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
    mmlu_preds = batch_infer(
        model_base=model_base,
        prompts=prompt_tokens_list,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        return_score=True,
    )

    print("计算准确率...")
    mmlu_acc = mmlu_dataset.get_accuracy(mmlu_preds, tokenizer=model_base.tokenizer)
    
    return mmlu_acc, mmlu_preds

def main(args):
    print("加载模型...")
    model_base = construct_model_base(model_alias_to_model_name[args.model_alias])
    
    print("初始化MMLU数据集...")
    mmlu_dataset = MMLUDataset(data_dir=args.data_dir)
    
    print("开始MMLU评测...")
    mmlu_acc, mmlu_preds = mmlu_evaluate(
        model_base=model_base,
        mmlu_dataset=mmlu_dataset,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    
    print(f"MMLU准确率: {mmlu_acc:.4f}")
    
    # 保存结果
    results = {
        "mmlu_accuracy": mmlu_acc,
        "model_alias": args.model_alias,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    
    rst_dir = f"results/generations/{args.model_alias}/mmlu/safeedit/vanilla/"
    os.makedirs(rst_dir, exist_ok=True)
    
    with open(os.path.join(rst_dir, "mmlu_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"结果已保存到: {os.path.join(rst_dir, 'mmlu_results.json')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMLU数据集评测')
    parser.add_argument('--model_alias', type=str, default="gemma-2-9b-it", help='模型别名')
    parser.add_argument('--data_dir', type=str, default="/home/wangxin/projects/steer-target-atoms-main/data/mmlu", help='MMLU数据集路径')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    parser.add_argument('--max_new_tokens', type=int, default=1, help='最大生成token数')
    parser.add_argument('--qa',type=bool, default=True, help='是否使用QA格式')
    args = parser.parse_args()
    main(args)
