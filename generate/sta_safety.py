import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

import json
import os
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
import torch
from transformer_lens import HookedTransformer
from utils.utils import model_alias_to_model_name
from dataset.load_data import load_safeedit_test_data, load_gsm_test_data, load_samsum_test_data
from transformers import StoppingCriteria

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

class STAVectorHookTransformerLens:
    """
    使用transformer_lens的STA向量钩子管理器
    """
    def __init__(self, sta_vector: torch.Tensor, multiplier: float = 1.0, from_position: int = 0):
        """
        初始化STA向量钩子

        Args:
            sta_vector: 预计算的STA向量
            multiplier: 引导向量的强度系数
            from_position: 从哪个位置开始添加STA向量
        """
        print("--- 初始化STA向量钩子 (Transformer Lens) ---")
        self.sta_vector = sta_vector
        self.multiplier = multiplier
        self.from_position = from_position

    def reset_state(self):
        """在每次新的生成任务开始前重置状态"""
        pass  # 对于STA向量，不需要重置状态

    def create_hook_fn(self):
        """创建并返回实际的钩子函数"""
        def hook_fn(activations, hook):
            # 获取当前序列长度
            seq_len = activations.shape[1]
            # 创建位置掩码
            position_ids = torch.arange(seq_len, device=activations.device).unsqueeze(0)
            
            # 从指定位置开始添加STA向量
            from_pos = self.from_position
            mask = position_ids >= from_pos
            mask = mask.unsqueeze(-1).float()
            # 使用float32进行计算，避免精度问题
            steer_vector = self.multiplier * self.sta_vector.to(activations.device)
            modified_activations = activations.to(torch.float32) + mask * steer_vector.to(torch.float32)
            return modified_activations
        
        return hook_fn


@torch.no_grad()
def steered_generate_transformer_lens(
    model: HookedTransformer,
    steering_hook: STAVectorHookTransformerLens,
    prompt: str,
    layer_to_steer: int,
    max_new_tokens: int = 50,
    test_dataset: str = "safeedit",
):
    """
    使用transformer_lens和STA向量钩子来生成文本
    """
    # 重置钩子状态
    steering_hook.reset_state()
    
    # 创建钩子函数
    hook_fn = steering_hook.create_hook_fn()
    
    # 设置钩子点名称
    hook_point = f"blocks.{layer_to_steer}.hook_resid_post"
    
    # 使用transformer_lens的hooks上下文管理器
    with model.hooks(fwd_hooks=[(hook_point, hook_fn)]):
        if test_dataset == "GSM":
            stop_id_sequences = [model.tokenizer.encode("Question:", add_special_tokens=False)]
            stopping_criteria = [KeyWordsCriteria(stop_id_sequences)]
            outputs = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_type="tensor",
            verbose=False,
            stopping_criteria=stopping_criteria,
            )
            return model.tokenizer.decode(outputs[0], skip_special_tokens=False)
        else:
            outputs = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_type="tensor",
                verbose=False,
            )
            return model.tokenizer.decode(outputs[0], skip_special_tokens=False)


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
    filepath = os.path.join(caa_vectors_dir, model_alias+"_"+dataset_name, filename)
    
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
    filename = f"steering_vector.pt"
    filepath = "/home/wangxin/projects/safety_steer/sae_analysis_results/gemma-2-9b-it_safeedit_layer_20_act_and_fre_trim_0.35.pt"
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"STA向量文件不存在: {filepath}")
    
    sta_vector = torch.load(filepath)
    print(f"加载STA向量: {filepath}, 形状: {sta_vector.shape}")
    return sta_vector


def main(args):
    print("加载模型...")
    # 使用transformer_lens加载模型
    model_path = model_alias_to_model_name[args.model_alias]
    model = HookedTransformer.from_pretrained(model_path, n_devices=1, torch_dtype=torch.bfloat16)
    model.eval()
    model.reset_hooks()
    
    # 加载CAA向量
    print("加载CAA向量...")
    caa_vector = load_caa_vector(
        model_alias=args.model_alias,
        dataset_name=args.dataset,
        layer=args.layer,
        caa_vectors_dir=args.caa_vectors_dir
    )
    
    # 加载STA向量
    print("加载STA向量...")
    sta_vector = load_sta_vector(
        model_alias=args.model_alias,
        dataset_name=args.dataset,
        layer=args.layer,
        sta_vectors_dir=args.sta_vectors_dir
    )
    
    # 计算multiplier，使STA向量的范数与CAA向量相同
    multiplier = torch.norm(caa_vector, p=2) / torch.norm(sta_vector, p=2)
    sta_vector = sta_vector * multiplier
    
    # 初始化STA向量钩子
    steering_hook = STAVectorHookTransformerLens(
        sta_vector=sta_vector,
        multiplier=args.multiplier,
        from_position=args.from_position
    )
    
    # 加载测试数据
    if args.test_dataset == "safeedit":
        queries = load_safeedit_test_data(model_alias=args.model_alias, apply_chat_format=True)
    elif args.test_dataset == "StrongREJECT":
        queries = load_safeedit_test_data(model_alias=args.model_alias, apply_chat_format=True)
    elif args.test_dataset == "BeaverTails":
        queries = load_safeedit_test_data(model_alias=args.model_alias, apply_chat_format=True)
    elif args.test_dataset == "GSM":
        queries = load_gsm_test_data(model_alias=args.model_alias, apply_chat_format=True)
        if args.model_alias == "gemma-2-9b-it":
            args.max_new_tokens = 1024
        else:
            args.max_new_tokens = 512
    elif args.test_dataset == "samsum":
        queries = load_samsum_test_data(model_alias=args.model_alias, apply_chat_format=True)
    else:
        raise ValueError(f"Invalid test dataset: {args.test_dataset}")

    questions = []
    generations = []
    gt_answers = []
    
    print("开始生成...")
    for query in tqdm(queries, desc="Generating outputs"):
        steered_output = steered_generate_transformer_lens(
            model,
            steering_hook,
            query['prompt'],
            layer_to_steer=args.layer,
            max_new_tokens=args.max_new_tokens,
            test_dataset=args.test_dataset
        )
        # 提取纯输出
        pure_output = steered_output.split("<end_of_turn>\n<start_of_turn>model\n")[1] if "<end_of_turn>\n<start_of_turn>model\n" in steered_output else steered_output
        generations.append(pure_output)
        questions.append(query['prompt'].replace("<end_of_turn>\n<start_of_turn>model\n", "").replace("<start_of_turn>user\n", ""))
        if args.test_dataset == "GSM" or args.test_dataset == "samsum":
            gt_answers.append(query['answer'])

    # 保存结果
    if args.test_dataset == "GSM" or args.test_dataset == "samsum":
        final_data = [{'question': q, 'generation': g, 'answer': a} for q, g, a in zip(questions, generations, gt_answers)]
    else:
        final_data = [{'question': q, 'generation': g} for q, g in zip(questions, generations)]
    
    rst_dir = f"results/generations/{args.model_alias}/{args.test_dataset}/sae_transformer_lens"   
    os.makedirs(rst_dir, exist_ok=True)
    
    output_filename = f"{args.dataset}_{args.layer}.json"
    with open(os.path.join(rst_dir, output_filename), "w") as f:
        json.dump(final_data, f, indent=2)
    
    print(f"结果已保存到: {os.path.join(rst_dir, output_filename)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用transformer_lens和STA向量进行安全引导生成')
    parser.add_argument('--model_alias', type=str, default="gemma-2-9b-it", help='模型别名')
    parser.add_argument('--dataset', type=str, default="safeedit", help='数据集名称')
    parser.add_argument('--layer', type=int, default=20, help='要进行干预的层')
    parser.add_argument('--multiplier', type=float, default=1.0, help='CAA向量的强度系数')
    parser.add_argument('--from_position', type=int, default=0, help='从哪个位置开始添加STA向量')
    parser.add_argument('--test_dataset', type=str, default="safeedit", help='测试数据集')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='最大生成token数')
    parser.add_argument('--caa_vectors_dir', type=str, default="results/caa_vectors", help='CAA向量目录')
    parser.add_argument('--sta_vectors_dir', type=str, default="results/sta_vectors", help='STA向量目录')
    
    args = parser.parse_args()
    
    main(args)
