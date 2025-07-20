import sys
sys.path.append('./')
sys.path.append('../')
import torch
import numpy as np
import argparse
from typing import List
from tqdm import tqdm
from utils.hf_models.model_factory import construct_model_base
from utils.hf_patching_utils import add_hooks
from utils.hf_models.model_base import ModelBase
from utils.utils import model_alias_to_model_name
from utils.hf_models.gemma_model import format_instruction_gemma_chat
from dataset.load_data import load_safeedit_test_data
import json
import os

class CAAVectorHook:
    """
    使用预计算CAA向量的简单钩子管理器
    """
    def __init__(self, caa_vector: torch.Tensor, multiplier: float = 1.0, from_position: int = 0):
        """
        初始化CAA向量钩子

        Args:
            caa_vector: 预计算的CAA向量
            multiplier: 引导向量的强度系数
            from_position: 从哪个位置开始添加CAA向量（类似参考代码中的from_pos）
        """
        print("--- 初始化CAA向量钩子 ---")
        self.caa_vector = caa_vector
        self.multiplier = multiplier
        self.from_position = from_position

    def reset_state(self):
        """在每次新的生成任务开始前重置状态"""
        pass  # 对于CAA向量，不需要重置状态

    def create_hook_fn(self):
        """创建并返回实际的钩子函数"""
        def hook_fn(_, input_tuple):
            # 解包输出
            if len(input_tuple) != 1:
                hidden_states, past_key_values = input_tuple[0], input_tuple[1]
            else:
                hidden_states, past_key_values = input_tuple[0], None
            
            # 获取position_ids（如果存在）
            position_ids = None
            if 'position_ids' in input_tuple[0].__dict__:
                position_ids = input_tuple[0].position_ids
            elif len(input_tuple) > 2:
                position_ids = input_tuple[2]  # 假设position_ids在第三个位置
            
            # 如果找不到position_ids，使用简单的token计数
            if position_ids is None:
                seq_len = hidden_states.shape[1]
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            
            # 从指定位置开始添加CAA向量（类似add_vector_from_position）
            from_pos = self.from_position
            mask = position_ids >= from_pos
            mask = mask.unsqueeze(-1).float()
            
            # 使用float32进行计算，避免精度问题
            steer_vector = self.multiplier * self.caa_vector.to(hidden_states.device)
            hidden_states = hidden_states.to(torch.float32) + mask * steer_vector.to(torch.float32)
            hidden_states = hidden_states.to(torch.bfloat16)
            # 返回修改后的输出
            if past_key_values is not None:
                return (hidden_states, past_key_values)
            else:
                return (hidden_states,)
        
        return hook_fn


@torch.no_grad()
def steered_generate(
    model_base: ModelBase,
    steering_hook: CAAVectorHook,
    prompt: str,
    layer_to_steer: int,
    max_new_tokens: int = 50,
):
    """
    使用CAA向量钩子来生成文本
    """
    tokenizer = model_base.tokenizer
    model = model_base.model

    # 获取要挂钩的模块
    target_module = model_base.model_block_modules[layer_to_steer]
    
    # 重置钩子状态并准备输入
    inputs = tokenizer(prompt, return_tensors="pt")
    steering_hook.reset_state()
    
    # 添加钩子并执行生成
    hook_fn = steering_hook.create_hook_fn()
    
    with add_hooks(module_forward_pre_hooks=[(target_module, hook_fn)], module_forward_hooks=[]):
        outputs = model.generate(
            inputs.input_ids.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=False)


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
    filename = f"{model_alias}_{dataset_name}_layer_{layer}_caa_vector.pt"
    filepath = os.path.join(caa_vectors_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CAA向量文件不存在: {filepath}")
    
    caa_vector = torch.load(filepath)
    print(f"加载CAA向量: {filepath}, 形状: {caa_vector.shape}")
    # 使用float32避免精度问题
    caa_vector = caa_vector.to(torch.float32)
    return caa_vector
    

def main(args):
    print("加载模型...")
    model_base = construct_model_base(model_alias_to_model_name[args.model_alias])
    
    # 加载CAA向量
    print("加载CAA向量...")
    caa_vector = load_caa_vector(
        model_alias=args.model_alias,
        dataset_name=args.dataset,
        layer=args.layer,
        caa_vectors_dir=args.caa_vectors_dir
    )
    
    # 初始化CAA向量钩子
    steering_hook = CAAVectorHook(
        caa_vector=caa_vector,
        multiplier=args.multiplier,
        from_position=args.from_position
    )
    
    # 加载测试数据
    if args.test_dataset == "safeedit":
        queries = load_safeedit_test_data(model_alias=args.model_alias, apply_chat_format=True)
    else:
        raise ValueError(f"Invalid test dataset: {args.test_dataset}")

    questions = []
    generations = []
    
    print("开始生成...")
    for query in tqdm(queries, desc="Generating outputs"):
        steered_output = steered_generate(
            model_base,
            steering_hook,
            query['prompt'],
            layer_to_steer=args.layer,
            max_new_tokens=args.max_new_tokens
        )
        
        # 提取纯输出
        pure_output = steered_output.split("<end_of_turn>\n<start_of_turn>model\n")[1]
        generations.append(pure_output)
        questions.append(query['prompt'].replace("<end_of_turn>\n<start_of_turn>model\n", "").replace("<start_of_turn>user\n", ""))

    # 保存结果
    final_data = [{'question': q, 'generation': g} for q, g in zip(questions, generations)]
    rst_dir = f"results/generations/{args.model_alias}/{args.test_dataset}/caa"   
    os.makedirs(rst_dir, exist_ok=True)
    
    output_filename = f"{args.dataset}_{args.layer}.json"
    with open(os.path.join(rst_dir, output_filename), "w") as f:
        json.dump(final_data, f, indent=2)
    
    print(f"结果已保存到: {os.path.join(rst_dir, output_filename)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用CAA向量进行安全引导生成')
    parser.add_argument('--model_alias', type=str, default="gemma-2-9b-it", help='模型别名')
    parser.add_argument('--dataset', type=str, default="safeedit", help='数据集名称')
    parser.add_argument('--layer', type=int, default=24, help='要进行干预的层')
    parser.add_argument('--multiplier', type=float, default=1.0, help='CAA向量的强度系数')
    parser.add_argument('--from_position', type=int, default=0, help='从哪个位置开始添加CAA向量')
    parser.add_argument('--test_dataset', type=str, default="safeedit", help='测试数据集')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='最大生成token数')
    parser.add_argument('--caa_vectors_dir', type=str, default="results/caa_vectors", help='CAA向量目录')

    args = parser.parse_args()
    main(args)
