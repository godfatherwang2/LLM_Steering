import sys
sys.path.append('./')
sys.path.append('../')

import os
import torch
import numpy as np
import argparse
import json
from tqdm import tqdm
from typing import List, Tuple


def compute_caa_vector_from_cache(
    model_alias: str,
    dataset_name: str,
    layer: int,
    shard_size: int,
    cache_dir: str = "dataset/cached_activations"
) -> torch.Tensor:
    """
    从cached_activations计算CAA向量
    
    Args:
        model_alias: 模型别名
        dataset_name: 数据集名称
        layer: 层号
        shard_size: 分片大小
        cache_dir: 缓存目录
    
    Returns:
        caa_vector: CAA向量，形状为(d_model,)
    """
    print(f"--- 计算CAA向量: {model_alias}/{dataset_name}/layer_{layer} ---")
    
    # 构建路径
    base_path = os.path.join(cache_dir, f"{model_alias}_{dataset_name}")
    safe_path = os.path.join(base_path, f"label_safe_shard_size_{shard_size}")
    unsafe_path = os.path.join(base_path, f"label_unsafe_shard_size_{shard_size}")
    
    # 加载元数据
    safe_metadata_path = os.path.join(safe_path, "metadata.json")
    unsafe_metadata_path = os.path.join(unsafe_path, "metadata.json")
    
    if not os.path.exists(safe_metadata_path) or not os.path.exists(unsafe_metadata_path):
        raise FileNotFoundError(f"元数据文件不存在: {safe_metadata_path} 或 {unsafe_metadata_path}")
    
    with open(safe_metadata_path, 'r') as f:
        safe_metadata = json.load(f)
    with open(unsafe_metadata_path, 'r') as f:
        unsafe_metadata = json.load(f)
    
    # 获取维度信息
    n_layers, total_tokens_safe, d_model = safe_metadata["activations_original_shape"]
    n_layers, total_tokens_unsafe, d_model = unsafe_metadata["activations_original_shape"]
    
    print(f"Safe tokens: {total_tokens_safe}, Unsafe tokens: {total_tokens_unsafe}")
    print(f"Model dimension: {d_model}")
    
    # 加载激活值数据
    safe_acts_path = os.path.join(safe_path, f"acts_{layer}.dat")
    unsafe_acts_path = os.path.join(unsafe_path, f"acts_{layer}.dat")
    
    if not os.path.exists(safe_acts_path) or not os.path.exists(unsafe_acts_path):
        raise FileNotFoundError(f"激活值文件不存在: {safe_acts_path} 或 {unsafe_acts_path}")
    
    print("加载安全激活值...")
    safe_activations = np.memmap(safe_acts_path, dtype='float32', mode='r', 
                                shape=(total_tokens_safe, d_model))
    
    print("加载不安全激活值...")
    unsafe_activations = np.memmap(unsafe_acts_path, dtype='float32', mode='r', 
                                  shape=(total_tokens_unsafe, d_model))
    
    # 计算平均值
    print("计算安全激活值平均值...")
    safe_mean = np.mean(safe_activations, axis=0)
    
    print("计算不安全激活值平均值...")
    unsafe_mean = np.mean(unsafe_activations, axis=0)
    
    # 计算CAA向量
    print("计算CAA向量...")
    caa_vector = safe_mean - unsafe_mean
    
    # 转换为torch tensor
    caa_vector_tensor = torch.from_numpy(caa_vector).float()
    
    print(f"CAA向量计算完成，形状: {caa_vector_tensor.shape}")
    print(f"CAA向量统计信息:")
    print(f"  均值: {caa_vector_tensor.mean().item():.6f}")
    print(f"  标准差: {caa_vector_tensor.std().item():.6f}")
    print(f"  最小值: {caa_vector_tensor.min().item():.6f}")
    print(f"  最大值: {caa_vector_tensor.max().item():.6f}")
    
    return caa_vector_tensor


def save_caa_vector(
    caa_vector: torch.Tensor,
    model_alias: str,
    dataset_name: str,
    layer: int,
    output_dir: str = "results/caa_vectors"
) -> str:
    """
    保存CAA向量
    
    Args:
        caa_vector: CAA向量
        model_alias: 模型别名
        dataset_name: 数据集名称
        layer: 层号
        output_dir: 输出目录
    
    Returns:
        saved_path: 保存路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建保存路径
    filename = f"{model_alias}_{dataset_name}_layer_{layer}_caa_vector.pt"
    save_path = os.path.join(output_dir, filename)
    
    # 保存向量
    torch.save(caa_vector, save_path)
    print(f"CAA向量已保存到: {save_path}")
    
    return save_path


def compute_all_layers_caa_vectors(
    model_alias: str,
    dataset_name: str,
    layers: List[int],
    shard_size: int,
    cache_dir: str = "dataset/cached_activations",
    output_dir: str = "results/caa_vectors"
) -> List[str]:
    """
    计算所有指定层的CAA向量
    
    Args:
        model_alias: 模型别名
        dataset_name: 数据集名称
        layers: 层号列表
        shard_size: 分片大小
        cache_dir: 缓存目录
        output_dir: 输出目录
    
    Returns:
        saved_paths: 所有保存路径的列表
    """
    saved_paths = []
    
    for layer in tqdm(layers, desc="计算各层CAA向量"):
        try:
            # 计算CAA向量
            caa_vector = compute_caa_vector_from_cache(
                model_alias=model_alias,
                dataset_name=dataset_name,
                layer=layer,
                shard_size=shard_size,
                cache_dir=cache_dir
            )
            
            # 保存CAA向量
            saved_path = save_caa_vector(
                caa_vector=caa_vector,
                model_alias=model_alias,
                dataset_name=dataset_name,
                layer=layer,
                output_dir=output_dir
            )
            
            saved_paths.append(saved_path)
            
        except Exception as e:
            print(f"计算第{layer}层CAA向量时出错: {e}")
            continue
    
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description='从cached_activations计算CAA向量')
    parser.add_argument('--model_alias', type=str, default="gemma-2-9b-it", help='模型别名')
    parser.add_argument('--dataset', type=str, default="safeedit", help='数据集名称')
    parser.add_argument('--layer', type=int, default=24, help='要计算的层号')
    parser.add_argument('--layers', type=str, default=None, help='要计算的层号列表，用逗号分隔')
    parser.add_argument('--shard_size', type=int, default=4050, help='分片大小')
    parser.add_argument('--cache_dir', type=str, default="dataset/cached_activations", help='缓存目录')
    parser.add_argument('--output_dir', type=str, default="results/caa_vectors", help='输出目录')
    
    args = parser.parse_args()
    
    if args.layers:
        # 计算多个层
        layers = [int(l) for l in args.layers.split(',')]
        print(f"计算层: {layers}")
        saved_paths = compute_all_layers_caa_vectors(
            model_alias=args.model_alias,
            dataset_name=args.dataset,
            layers=layers,
            shard_size=args.shard_size,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir
        )
        print(f"成功计算并保存了 {len(saved_paths)} 个CAA向量")
    else:
        # 计算单个层
        print(f"计算第 {args.layer} 层的CAA向量")
        caa_vector = compute_caa_vector_from_cache(
            model_alias=args.model_alias,
            dataset_name=args.dataset,
            layer=args.layer,
            shard_size=args.shard_size,
            cache_dir=args.cache_dir
        )
        
        saved_path = save_caa_vector(
            caa_vector=caa_vector,
            model_alias=args.model_alias,
            dataset_name=args.dataset,
            layer=args.layer,
            output_dir=args.output_dir
        )
        print(f"CAA向量计算完成: {saved_path}")


if __name__ == "__main__":
    main() 