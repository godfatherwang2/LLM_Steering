import os
import sys
sys.path.append('./')
sys.path.append('../')

import torch
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from jaxtyping import Int, Float
from torch import Tensor
import gc
import numpy as np

# 设置随机种子
random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

from utils.hf_models.model_base import ModelBase
from utils.hf_models.model_factory import construct_model_base
from utils.hf_patching_utils import add_hooks
from utils.utils import clear_memory, model_alias_to_model_name
from dataset.load_data import load_safeedit_queries

@dataclass
class Config:
    """配置类，用于存储所有参数"""
    batch_size: int = 1
    layers: List[int] = None
    model_name: str = "gemma-2-9b-it"
    model_name_or_path: str = "/home/wangxin/models/gemma-2-9b-it"
    output_dir: str = "results/caa_vectors"
    dataset_name: str = "safeedit"
    shard_size: int = 4050


class ModelManager:
    """模型管理器，负责加载和管理模型"""
    
    def __init__(self, config: Config, device: str = "cuda"):
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load_models(self):
        """加载模型和tokenizer"""
        print("正在加载模型...")
        
        # 加载模型
        model_path = model_alias_to_model_name.get(self.config.model_name, self.config.model_name_or_path)
        self.model = construct_model_base(model_path)
        self.tokenizer = self.model.tokenizer
        
        print("模型加载完成")


class PromptProcessor:
    """提示词处理器，负责处理safeedit数据集的提示词"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def format_prompt_for_gemma_2_9b_it(self, question: str) -> str:
        """为Gemma-2-9b-it格式化提示词"""
        return f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    
    def process_safeedit_prompts(self) -> List[Dict]:
        """处理safeedit数据集的提示词"""
        print("加载safeedit数据集...")
        queries = load_safeedit_queries(model_alias=self.config.model_name, apply_chat_format=True)
        
        prompts = []
        for query in queries:
            question = query["prompt"]
            response = query["response"]
            label = query["label"]
            
            # 根据模型类型格式化提示词
            if self.config.model_name == "gemma-2-9b-it":
                ques = self.format_prompt_for_gemma_2_9b_it(question)
            else:
                raise NotImplementedError(f"Model {self.config.model_name} not supported")
            
            if ques and response:
                prompts.append({
                    "ques": ques,
                    "response": response,
                    "label": label,
                })
        
        print(f"处理了 {len(prompts)} 个提示词")
        print(f"Safe样本: {sum(1 for p in prompts if p['label'])}")
        print(f"Unsafe样本: {sum(1 for p in prompts if not p['label'])}")
        return prompts


def _get_activations_forward_hook(cache: Float[Tensor, "pos d_model"]):
    """获取激活值的forward hook函数"""
    def hook_fn(module, input_tuple, output_tuple):
        nonlocal cache
        # 解包输出（因为这是forward hook，在模块执行后调用）
        if isinstance(output_tuple, tuple):
            hidden_states = output_tuple[0]
        else:
            hidden_states = output_tuple
        
        # 保存激活值到缓存
        cache[:, :] = hidden_states[:, :].to(cache)
        
        # 返回原始输出，不修改
        if isinstance(output_tuple, tuple):
            return output_tuple
        else:
            return hidden_states
    return hook_fn


@torch.no_grad()
def get_activations_for_tokens(model: ModelBase, tokens: torch.Tensor, layers: List[int], 
                              positions: List[int] = [-1], batch_size: int = 1) -> Dict[int, torch.Tensor]:
    """获取指定层和位置的激活值"""
    torch.cuda.empty_cache()
    
    n_layers = len(layers)
    d_model = model.model.config.hidden_size
    n_positions = len(positions)
    
    # 初始化激活值缓存
    activations = torch.zeros((len(tokens), n_layers, n_positions, d_model), device=model.model.device)
    
    # 创建forward hooks
    fwd_hooks = [
        (model.model_block_modules[layer], _get_activations_forward_hook(cache=activations[:, layer_idx, :, :])) 
        for layer_idx, layer in enumerate(layers)
    ]
    
    # 分批处理
    for i in range(0, len(tokens), batch_size):
        batch_tokens = tokens[i:i+batch_size]
        
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            model.model(
                input_ids=batch_tokens,
                attention_mask=torch.ones_like(batch_tokens),
            )
    
    # 整理结果
    result = {}
    for layer_idx, layer in enumerate(layers):
        result[layer] = activations[:, layer_idx, :, :].detach().cpu()
    
    return result


def process_batch_activations_hf(model: ModelBase, prompts: List[Dict], layers: List[int], 
                                batch_size: int = 1):
    """使用Hugging Face hooks分批处理激活值"""
    pos_activations = dict([(layer, []) for layer in layers])
    neg_activations = dict([(layer, []) for layer in layers])
    
    # 分批处理
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i + batch_size]
        
        for prompt in batch_prompts:
            ques = prompt["ques"]
            response = prompt["response"]
            label = prompt["label"]
            
            # 编码token
            ques_tokens = model.tokenizer.encode(ques, return_tensors="pt")
            full_tokens = model.tokenizer.encode(ques + response, return_tensors="pt")
            
            ques_tokens = ques_tokens.to(model.model.device)
            full_tokens = full_tokens.to(model.model.device)
            
            # 获取激活值（取最后一个token）
            acts = get_activations_for_tokens(model, full_tokens, layers, positions=[-1], batch_size=1)
            
            for layer in layers:
                activation = acts[layer][0, 0, :]  # 取最后一个token的激活值
                
                if label:  # safe
                    pos_activations[layer].append(activation)
                else:  # unsafe
                    neg_activations[layer].append(activation)
            
            # 清理GPU缓存
            del ques_tokens, full_tokens
            torch.cuda.empty_cache()
    
    return pos_activations, neg_activations


class ResultSaver:
    """结果保存器，负责保存CAA向量"""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_caa_vector(self, caa_vector: torch.Tensor, model_name: str, dataset_name: str, layer: int):
        """保存CAA向量"""
        filename = f"{model_name}_{dataset_name}_layer_{layer}_caa_vector.pt"
        save_path = os.path.join(self.output_dir, filename)
        
        torch.save(caa_vector, save_path)
        print(f"CAA向量已保存到: {save_path}")
        
        return save_path


class CAAVectorGenerator:
    """CAA向量生成器主类"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model_manager = ModelManager(config)
        self.prompt_processor = PromptProcessor(config)
        self.result_saver = ResultSaver(config.output_dir)
        
    def run(self):
        """运行CAA向量生成流程"""
        print(f"开始生成CAA向量，模型: {self.config.model_name}")
        
        # 1. 加载模型
        self.model_manager.load_models()
        
        # 2. 处理数据
        prompts = self.prompt_processor.process_safeedit_prompts()
        
        # 3. 生成激活值
        pos_activations, neg_activations = process_batch_activations_hf(
            model=self.model_manager.model,
            prompts=prompts,
            layers=self.config.layers,
            batch_size=self.config.batch_size
        )
        
        # 4. 计算并保存CAA向量
        self._compute_and_save_caa_vectors(pos_activations, neg_activations)
        
        print("CAA向量生成完成")
    
    def _compute_and_save_caa_vectors(self, pos_activations: Dict[int, List[torch.Tensor]], 
                                     neg_activations: Dict[int, List[torch.Tensor]]):
        """计算并保存CAA向量"""
        print("计算并保存CAA向量...")
        
        for layer in self.config.layers:
            print(f"处理第 {layer} 层...")
            
            # 在CPU上计算向量
            all_pos_layer = torch.stack(pos_activations[layer])
            all_neg_layer = torch.stack(neg_activations[layer])
            caa_vector = (all_pos_layer - all_neg_layer).mean(dim=0)

            # 保存CAA向量
            self.result_saver.save_caa_vector(
                caa_vector=caa_vector,
                model_name=self.config.model_name,
                dataset_name=self.config.dataset_name,
                layer=layer
            )
            
            # 清理内存
            del all_pos_layer, all_neg_layer, caa_vector
            torch.cuda.empty_cache()
            
            print(f"第 {layer} 层处理完成")


def parse_arguments() -> Config:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CAA向量生成工具")
    parser.add_argument("--batch_size", default=1, type=int, help="批处理大小")
    parser.add_argument("--layers", type=str, default="20", help="要处理的层，用逗号分隔")
    parser.add_argument("--model_name", default="gemma-2-9b-it", type=str, help="模型名称")
    parser.add_argument("--model_name_or_path", default="/home/wangxin/models/gemma-2-9b-it", type=str, help="模型路径")
    parser.add_argument("--output_dir", default="results/caa_vectors", type=str, help="输出目录")
    parser.add_argument("--dataset_name", default="safeedit", type=str, help="数据集名称")
    
    args = parser.parse_args()
    
    # 解析层列表
    layers = [int(l) for l in args.layers.split(',')]
    
    return Config(
        batch_size=args.batch_size,
        layers=layers,
        model_name=args.model_name,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
    )


def main():
    """主函数"""
    config = parse_arguments()
    print(f"配置参数: {config}")
    
    generator = CAAVectorGenerator(config)
    generator.run()

if __name__ == "__main__":
    main()