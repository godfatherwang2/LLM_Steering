import os
import sys
import json
import numpy as np
import torch
import argparse
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from refrencecode.generate_sae_caa_vector import load_gemma_sae
# 添加路径
sys.path.append('./')
sys.path.append('../')

from activation_cache_sae import DEFAULT_CACHE_DIR


def load_sample_indices(cache_dir: str, label: str, shard_size: int = None) -> Tuple[List[Dict], str]:
    """加载样本索引范围信息"""
    if shard_size is None:
        # 尝试自动检测文件夹
        possible_folders = [f for f in os.listdir(cache_dir) if f.startswith(f"label_{label}_shard_size_")]
        if len(possible_folders) == 1:
            foldername = os.path.join(cache_dir, possible_folders[0])
        else:
            raise ValueError(f"Multiple or no folders found for label {label} in {cache_dir}")
    else:
        foldername = os.path.join(cache_dir, f"label_{label}_shard_size_{shard_size}")
    
    sample_indices_filepath = os.path.join(foldername, "sample_indices.json")
    if not os.path.exists(sample_indices_filepath):
        raise FileNotFoundError(f"Sample indices file not found: {sample_indices_filepath}")
    
    with open(sample_indices_filepath, 'r') as f:
        sample_indices = json.load(f)
    
    return sample_indices, foldername


def load_sae_features_for_sample(sample_info: Dict, cache_dir: str, layer_num: int) -> np.ndarray:
    """加载指定样本的SAE特征"""
    start_idx = sample_info["start_index"]
    end_idx = sample_info["end_index"]
    
    # 加载元数据获取特征维度
    metadata_filepath = os.path.join(cache_dir, "metadata.json")
    with open(metadata_filepath, 'r') as f:
        metadata = json.load(f)
    sae_features_dim = metadata["sae_features_dim"]
    
    # 加载SAE特征文件
    sae_filepath = os.path.join(cache_dir, f"sae_{layer_num}.dat")
    if not os.path.exists(sae_filepath):
        raise FileNotFoundError(f"SAE features file not found: {sae_filepath}")
    
    # 使用numpy memmap读取指定范围的数据
    sae_features = np.memmap(sae_filepath, dtype='float32', mode='r', 
                            shape=(metadata["sae_features_shape"][1], sae_features_dim))
    
    # 提取指定范围的特征
    sample_features = sae_features[start_idx:end_idx, :].copy()
    
    return sample_features


def calculate_activation_frequency(features: np.ndarray, activation_threshold: float = 0.0) -> np.ndarray:
    """计算每个特征的激活频率
    
    Args:
        features: shape (n_tokens, n_features)
        activation_threshold: 激活阈值，大于此值认为是激活
    
    Returns:
        激活频率: shape (n_features,)
    """
    # 计算每个token中每个特征是否激活
    activations = features > activation_threshold
    
    # 计算每个特征的激活频率（激活的token数 / 总token数）
    activation_freq = np.mean(activations, axis=0)
    
    return activation_freq


def analyze_sae_feature_importance(safe_cache_dir: str, unsafe_cache_dir: str, 
                                  layer_num: int, activation_threshold: float = 0.0) -> Dict:
    """分析SAE特征的重要性
    
    Args:
        safe_cache_dir: 安全样本缓存目录
        unsafe_cache_dir: 非安全样本缓存目录
        layer_num: 层号
        activation_threshold: 激活阈值
    
    Returns:
        分析结果字典
    """
    print(f"Loading sample indices...")
    safe_indices, safe_folder = load_sample_indices(safe_cache_dir, "safe")
    unsafe_indices, unsafe_folder = load_sample_indices(unsafe_cache_dir, "unsafe")
    
    print(f"Safe samples: {len(safe_indices)}, Unsafe samples: {len(unsafe_indices)}")
    
    # 确保样本数量一致
    min_samples = min(len(safe_indices), len(unsafe_indices))
    safe_indices = safe_indices[:min_samples]
    unsafe_indices = unsafe_indices[:min_samples]
    
    print(f"Valid sample pairs: {len(safe_indices)}")
    
    # 加载元数据获取特征维度
    metadata_filepath = os.path.join(safe_folder, "metadata.json")
    with open(metadata_filepath, 'r') as f:
        metadata = json.load(f)
    sae_features_dim = metadata["sae_features_dim"]
    
    # 存储每对样本的激活频率差异和激活值大小差异
    activation_freq_diffs = []
    activation_value_diffs = []
    safe_activation_freqs = []
    unsafe_activation_freqs = []
    safe_activation_values = []
    unsafe_activation_values = []
    
    print(f"Analyzing {len(safe_indices)} sample pairs...")
    for safe_info, unsafe_info in tqdm(zip(safe_indices, unsafe_indices), total=len(safe_indices)):
        
        # 加载SAE特征
        safe_features = load_sae_features_for_sample(safe_info, safe_folder, layer_num)
        unsafe_features = load_sae_features_for_sample(unsafe_info, unsafe_folder, layer_num, )
        
        # 计算激活频率
        safe_freq = calculate_activation_frequency(safe_features, activation_threshold)
        unsafe_freq = calculate_activation_frequency(unsafe_features, activation_threshold)
        
        # 计算激活值大小（平均值）
        safe_value = np.mean(safe_features, axis=0)
        unsafe_value = np.mean(unsafe_features, axis=0)
        
        # 计算差异
        freq_diff = safe_freq - unsafe_freq
        value_diff = safe_value - unsafe_value
        
        activation_freq_diffs.append(freq_diff)
        activation_value_diffs.append(value_diff)
        safe_activation_freqs.append(safe_freq)
        unsafe_activation_freqs.append(unsafe_freq)
        safe_activation_values.append(safe_value)
        unsafe_activation_values.append(unsafe_value)
            
    if not activation_freq_diffs:
        raise ValueError("No valid sample pairs processed")
    
    # 转换为numpy数组
    activation_freq_diffs = np.array(activation_freq_diffs)  # shape: (n_pairs, n_features)
    activation_value_diffs = np.array(activation_value_diffs)  # shape: (n_pairs, n_features)
    safe_activation_freqs = np.array(safe_activation_freqs)
    unsafe_activation_freqs = np.array(unsafe_activation_freqs)
    safe_activation_values = np.array(safe_activation_values)
    unsafe_activation_values = np.array(unsafe_activation_values)
    
    # 计算统计量
    mean_freq_diff = np.mean(activation_freq_diffs, axis=0)  # 平均频率差异
    std_freq_diff = np.std(activation_freq_diffs, axis=0)    # 频率差异标准差
    abs_mean_freq_diff = np.abs(mean_freq_diff)             # 平均频率差异的绝对值
    
    mean_value_diff = np.mean(activation_value_diffs, axis=0)  # 平均激活值差异
    std_value_diff = np.std(activation_value_diffs, axis=0)    # 激活值差异标准差
    abs_mean_value_diff = np.abs(mean_value_diff)             # 平均激活值差异的绝对值
    
    # 创建特征重要性排序（基于频率差异）
    feature_importance = []
    for feature_idx in range(sae_features_dim):
        feature_importance.append({
            "feature_idx": feature_idx,
            "mean_freq_diff": mean_freq_diff[feature_idx],
            "abs_mean_freq_diff": abs_mean_freq_diff[feature_idx],
            "std_freq_diff": std_freq_diff[feature_idx],
            "mean_value_diff": mean_value_diff[feature_idx],
            "abs_mean_value_diff": abs_mean_value_diff[feature_idx],
            "std_value_diff": std_value_diff[feature_idx]
        })
    
    # 按绝对平均频率差异排序
    feature_importance.sort(key=lambda x: x["abs_mean_freq_diff"], reverse=True)
    
    results = {
        "layer_num": layer_num,
        "activation_threshold": activation_threshold,
        "n_sample_pairs": len(activation_freq_diffs),
        "n_features": sae_features_dim,
        "feature_importance": feature_importance,
        "overall_stats": {
            "mean_abs_freq_diff": np.mean(abs_mean_freq_diff),
            "std_abs_freq_diff": np.std(abs_mean_freq_diff),
            "max_abs_freq_diff": np.max(abs_mean_freq_diff),
            "min_abs_freq_diff": np.min(abs_mean_freq_diff),
            "mean_abs_value_diff": np.mean(abs_mean_value_diff),
            "std_abs_value_diff": np.std(abs_mean_value_diff),
            "max_abs_value_diff": np.max(abs_mean_value_diff),
            "min_abs_value_diff": np.min(abs_mean_value_diff)
        }
    }
    
    return results


def save_analysis_results(results: Dict, output_dir: str, model_alias: str):
    """保存分析结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存总体统计信息为JSON
    overall_stats = {
        "layer_num": results["layer_num"],
        "activation_threshold": results["activation_threshold"],
        "n_sample_pairs": results["n_sample_pairs"],
        "n_features": results["n_features"],
        "overall_stats": results["overall_stats"]
    }
    
    # 确保所有数值都是Python原生类型，避免JSON序列化问题
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    overall_stats = convert_numpy_types(overall_stats)
    
    overall_stats_filepath = os.path.join(output_dir, f"overall_stats_{model_alias}_layer_{results['layer_num']}.json")
    with open(overall_stats_filepath, 'w') as f:
        json.dump(overall_stats, f, indent=4)
    
    # 保存特征信息为CSV
    feature_importance_df = pd.DataFrame(results["feature_importance"])
    csv_filepath = os.path.join(output_dir, f"feature_importance_{model_alias}_layer_{results['layer_num']}.csv")
    feature_importance_df.to_csv(csv_filepath, index=False)
    
    # 创建可视化
    create_visualizations(results, output_dir, model_alias)
    
    print(f"Results saved to {output_dir}")
    print(f"Overall stats: {overall_stats_filepath}")
    print(f"Feature importance: {csv_filepath}")


def create_visualizations(results: Dict, output_dir: str, model_alias: str):
    """创建可视化图表"""
    feature_importance = results["feature_importance"]
    
    # 提取数据
    feature_indices = [f["feature_idx"] for f in feature_importance]
    abs_mean_freq_diffs = [f["abs_mean_freq_diff"] for f in feature_importance]
    abs_mean_value_diffs = [f["abs_mean_value_diff"] for f in feature_importance]
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle(f'SAE Feature Analysis - {model_alias} Layer {results["layer_num"]}', fontsize=16)
    
    # 绘制频率差绝对值的柱状图（前50个特征）
    top_n = min(50, len(feature_importance))
    ax1.bar(range(top_n), abs_mean_freq_diffs[:top_n], alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_title(f'Top {top_n} Features by Absolute Frequency Difference')
    ax1.set_xlabel('Feature Index (Sorted by |Mean Frequency Difference|)')
    ax1.set_ylabel('|Mean Frequency Difference|')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 绘制激活值差绝对值的柱状图（前50个特征）
    # 按激活值差异绝对值排序
    sorted_by_value = sorted(feature_importance, key=lambda x: x["abs_mean_value_diff"], reverse=True)
    top_value_features = sorted_by_value[:top_n]
    top_value_indices = [f["feature_idx"] for f in top_value_features]
    top_value_diffs = [f["abs_mean_value_diff"] for f in top_value_features]
    
    ax2.bar(range(top_n), top_value_diffs, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax2.set_title(f'Top {top_n} Features by Absolute Value Difference')
    ax2.set_xlabel('Feature Index (Sorted by |Mean Value Difference|)')
    ax2.set_ylabel('|Mean Value Difference|')
    ax2.set_xticks(range(top_n))
    ax2.set_xticklabels(top_value_indices, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_filepath = os.path.join(output_dir, f"sae_analysis_plot_{model_alias}_layer_{results['layer_num']}.png")
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {plot_filepath}")


def create_steering_vector_visualizations(results: Dict, output_dir: str, model_alias: str, dataset: str, select_type: str, trim: float):
    """创建steering vector的可视化图表"""
    feature_importance = results["feature_importance"]
    
    # 按频率差异绝对值排序
    sorted_by_freq = sorted(feature_importance, key=lambda x: x["abs_mean_freq_diff"], reverse=True)
    
    # 提取数据
    abs_freq_diffs = [f["abs_mean_freq_diff"] for f in sorted_by_freq]
    abs_value_diffs = [f["abs_mean_value_diff"] for f in sorted_by_freq]
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle(f'Steering Vector Analysis - {model_alias} {dataset} Layer {results["layer_num"]}\n{select_type} (trim={trim})', fontsize=16)
    
    # 绘制频率差异绝对值的柱状图（前50个特征）
    top_n = min(50, len(sorted_by_freq))
    ax1.bar(range(top_n), abs_freq_diffs[:top_n], alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    ax1.set_title(f'Top {top_n} Features by Absolute Frequency Difference')
    ax1.set_xlabel('Feature Index (Sorted by |Mean Frequency Difference|)')
    ax1.set_ylabel('|Mean Frequency Difference|')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 绘制激活值差异绝对值的柱状图（前50个特征）
    # 按激活值差异绝对值排序
    sorted_by_value = sorted(feature_importance, key=lambda x: x["abs_mean_value_diff"], reverse=True)
    top_value_features = sorted_by_value[:top_n]
    top_value_diffs = [f["abs_mean_value_diff"] for f in top_value_features]
    
    ax2.bar(range(top_n), top_value_diffs, alpha=0.7, color='orange', edgecolor='darkorange')
    ax2.set_title(f'Top {top_n} Features by Absolute Value Difference')
    ax2.set_xlabel('Feature Index (Sorted by |Mean Value Difference|)')
    ax2.set_ylabel('|Mean Value Difference|')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_filepath = os.path.join(output_dir, f"steering_vector_plot_{model_alias}_{dataset}_layer_{results['layer_num']}_{select_type}_{trim}.png")
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Steering vector plot saved: {plot_filepath}")





def signed_min_max_normalize(tensor):
    """Min-Max归一化，保留正负符号"""
    abs_tensor = tensor.abs()
    min_val = abs_tensor.min()
    max_val = abs_tensor.max()
    normalized = (abs_tensor - min_val) / (max_val - min_val)
    return tensor.sign() * normalized  # 恢复正负符号


def calculate_steering_vector(safe_cache_dir: str, unsafe_cache_dir: str, 
                            layer_num: int, sae_model, 
                            trim: float = 0.35, select_type: str = "act_and_fre_trim",
                            activation_threshold: float = 0.0) -> Dict:
    """计算steering vector
    
    Args:
        safe_cache_dir: 安全样本缓存目录
        unsafe_cache_dir: 非安全样本缓存目录
        layer_num: 层号
        sae_model: SAE模型
        trim: 剪枝比例 (0.0-1.0)
        select_type: 选择类型 ("act_and_fre_trim", "act_trim", "fre_trim")
        activation_threshold: 激活阈值
    
    Returns:
        包含steering vector和中间结果的字典
    """
    print(f"Calculating steering vector for layer {layer_num} with select_type={select_type}, trim={trim}")
    
    # 加载样本索引
    safe_indices, safe_folder = load_sample_indices(safe_cache_dir, "safe")
    unsafe_indices, unsafe_folder = load_sample_indices(unsafe_cache_dir, "unsafe")
    
    # 确保样本数量一致
    min_samples = min(len(safe_indices), len(unsafe_indices))
    safe_indices = safe_indices[:min_samples]
    unsafe_indices = unsafe_indices[:min_samples]
    
    print(f"Processing {len(safe_indices)} sample pairs...")
    
    # 存储所有样本的特征激活值
    safe_feature_acts_list = []
    unsafe_feature_acts_list = []
    
    # 处理每对样本
    for safe_info, unsafe_info in tqdm(zip(safe_indices, unsafe_indices), total=len(safe_indices)):
        # 加载SAE特征
        safe_features = load_sae_features_for_sample(safe_info, safe_folder, layer_num)
        unsafe_features = load_sae_features_for_sample(unsafe_info, unsafe_folder, layer_num)
        
        # 在token维度上求平均（模拟参考代码的逻辑）
        safe_feature_acts = np.mean(safe_features, axis=0)  # shape: (n_features,)
        unsafe_feature_acts = np.mean(unsafe_features, axis=0)  # shape: (n_features,)
        
        safe_feature_acts_list.append(safe_feature_acts)
        unsafe_feature_acts_list.append(unsafe_feature_acts)
    
    # 转换为numpy数组
    safe_feature_acts = np.array(safe_feature_acts_list)  # shape: (n_samples, n_features)
    unsafe_feature_acts = np.array(unsafe_feature_acts_list)  # shape: (n_samples, n_features)
    
    # 计算统计量（与参考代码一致）
    pos_feature_freq = (safe_feature_acts > activation_threshold).astype(float).sum(0)  # 激活频率
    neg_feature_freq = (unsafe_feature_acts > activation_threshold).astype(float).sum(0)  # 激活频率
    
    pos_act_mean = safe_feature_acts.mean(0)  # 平均激活值
    neg_act_mean = unsafe_feature_acts.mean(0)  # 平均激活值
    
    feature_score = pos_act_mean - neg_act_mean  # 特征得分
    
    # 转换为torch tensor
    pos_feature_freq = torch.tensor(pos_feature_freq, dtype=torch.float32)
    neg_feature_freq = torch.tensor(neg_feature_freq, dtype=torch.float32)
    pos_act_mean = torch.tensor(pos_act_mean, dtype=torch.float32)
    neg_act_mean = torch.tensor(neg_act_mean, dtype=torch.float32)
    feature_score = torch.tensor(feature_score, dtype=torch.float32)
    
    print(f"Feature score norm: {torch.norm(feature_score)}")
    print(f"Positive feature freq norm: {torch.norm(pos_feature_freq)}")
    print(f"Negative feature freq norm: {torch.norm(neg_feature_freq)}")
    
    # 初始化特征重要性列表
    feature_importance = []
    sae_features_dim = len(feature_score)
    
    # 根据select_type生成steering vector
    if select_type == "act_and_fre_trim":
        # 激活值和频率结合方法
        diff_data = pos_feature_freq - neg_feature_freq
        
        # 1. Min-Max归一化，保留正负符号
        norm_act = signed_min_max_normalize(feature_score)  # 激活值差值归一化
        norm_diff = signed_min_max_normalize(diff_data)   # 激活频率差值归一化
        
        # 2. 符号一致性筛选
        mask = ((norm_act > 0) & (norm_diff > 0)) | ((norm_act < 0) & (norm_diff < 0))
        print(f"Symbol consistency mask: {mask.sum()}")
        
        # 4. 阈值筛选
        scores = torch.zeros_like(norm_diff)  # 初始化综合得分
        scores[mask] = norm_diff[mask]  # 仅计算符号一致的维度得分
        
        
        threshold_fre = torch.sort(torch.abs(scores), descending=True, stable=True).values[int(trim * len(scores))]
        print(f'Frequency threshold: {threshold_fre}')
        prune_mask = torch.abs(scores) >= threshold_fre
        print(f"Frequency prune mask: {prune_mask.sum()}")
        
        # 5. 激活值阈值筛选
        act_threshold = torch.sort(torch.abs(feature_score), descending=True, stable=True).values[int(trim * len(feature_score))]
        print(f'Activation threshold: {act_threshold}')
        act_top_mask = torch.abs(feature_score) >= act_threshold
        print(f"Activation top mask: {act_top_mask.sum()}")
        
        # 6. 结合掩码
        combined_mask = prune_mask & act_top_mask
        print(f"Combined mask: {combined_mask.sum()}")
        
        # 7. 生成steering vector
        act_data_combined = feature_score.clone()
        act_data_combined[~combined_mask] = 0
        
        torch.save(act_data_combined.cpu(), "/home/wangxin/projects/safety_steer/sae_analysis_results/gemma-2-9b-it_safeedit_layer_20_static_sae_features.pt")
        print(f"Static SAE features saved to: /home/wangxin/projects/safety_steer/sae_analysis_results/gemma-2-9b-it_safeedit_layer_20_static_sae_features.pt")
        steering_vector = act_data_combined @ sae_model.W_dec.cpu()
        
        # 创建特征重要性信息
        for feature_idx in range(sae_features_dim):
            feature_importance.append({
                "abs_mean_freq_diff": abs(diff_data[feature_idx]).item(),
                "abs_mean_value_diff": abs(feature_score[feature_idx]).item()
            })
        
    elif select_type == "act_trim":
        # 仅基于激活值
        act_threshold = torch.sort(torch.abs(feature_score), descending=True, stable=True).values[int(trim * len(feature_score))]
        print(f'Activation threshold: {act_threshold}')
        act_mask = torch.abs(feature_score) >= act_threshold
        print(f"Activation mask: {act_mask.sum()}")
        
        act_data_act = feature_score.clone()
        act_data_act[~act_mask] = 0
        
        steering_vector = act_data_act @ sae_model.W_dec.cpu()
        
        # 创建特征重要性信息
        for feature_idx in range(sae_features_dim):
            feature_importance.append({
                "abs_mean_freq_diff": 0.0,  # act_trim方法不涉及频率差异
                "abs_mean_value_diff": abs(feature_score[feature_idx]).item()
            })
        
    elif select_type == "fre_trim":
        # 仅基于频率
        diff_data = pos_feature_freq - neg_feature_freq
        
        # Min-Max归一化
        norm_diff = signed_min_max_normalize(diff_data)
        
        # 阈值筛选
        threshold_fre = torch.sort(torch.abs(norm_diff), descending=True, stable=True).values[int(trim * len(norm_diff))]
        print(f'Frequency threshold: {threshold_fre}')
        prune_mask = torch.abs(norm_diff) >= threshold_fre
        print(f"Frequency prune mask: {prune_mask.sum()}")
        
        act_data_fre = feature_score.clone()
        act_data_fre[~prune_mask] = 0
        
        steering_vector = act_data_fre @ sae_model.W_dec.cpu()
        
        # 创建特征重要性信息
        for feature_idx in range(sae_features_dim):
            feature_importance.append({
                "abs_mean_freq_diff": abs(diff_data[feature_idx]).item(),
                "abs_mean_value_diff": abs(feature_score[feature_idx]).item()
            })
        
    else:
        raise ValueError(f"Unknown select_type: {select_type}")
    
    print(f"Steering vector shape: {steering_vector.shape}")
    print(f"Steering vector norm: {torch.norm(steering_vector)}")
    
    results = {
        "steering_vector": steering_vector,
        "feature_score": feature_score,
        "pos_feature_freq": pos_feature_freq,
        "neg_feature_freq": neg_feature_freq,
        "pos_act_mean": pos_act_mean,
        "neg_act_mean": neg_act_mean,
        "feature_importance": feature_importance,
        "select_type": select_type,
        "trim": trim,
        "layer_num": layer_num,
        "n_sample_pairs": len(safe_indices),
        "n_features": sae_features_dim
    }
    
    return results


def analyse_sae_feature_importance():
    parser = argparse.ArgumentParser(description='Analyze SAE feature importance')
    parser.add_argument('--model_alias', type=str, default='gemma-2-9b-it', help='Model alias')
    parser.add_argument('--dataset', type=str, default='safeedit', help='Dataset name')
    parser.add_argument('--layer', type=int, default=20, help='Layer number to analyze')
    parser.add_argument('--activation_threshold', type=float, default=0.0, help='Activation threshold')
    parser.add_argument('--output_dir', type=str, default='./sae_analysis_results', help='Output directory')

    args = parser.parse_args()
    
    # 构建缓存目录路径
    cache_base_dir = f"{DEFAULT_CACHE_DIR}/{args.model_alias}_{args.dataset}"
    safe_cache_dir = cache_base_dir #os.path.join(cache_base_dir, "label_safe_shard_size_4050")
    unsafe_cache_dir = cache_base_dir #os.path.join(cache_base_dir, "label_unsafe_shard_size_4050")
    
    if not os.path.exists(safe_cache_dir):
        raise FileNotFoundError(f"Safe cache directory not found: {safe_cache_dir}")
    if not os.path.exists(unsafe_cache_dir):
        raise FileNotFoundError(f"Unsafe cache directory not found: {unsafe_cache_dir}")
    
    print(f"Analyzing SAE features for {args.model_alias} layer {args.layer}")
    print(f"Safe cache: {safe_cache_dir}")
    print(f"Unsafe cache: {unsafe_cache_dir}")
    
    # 执行分析
    results = analyze_sae_feature_importance(
        safe_cache_dir=safe_cache_dir,
        unsafe_cache_dir=unsafe_cache_dir,
        layer_num=args.layer,
        activation_threshold=args.activation_threshold,
    )
    
    # 保存结果
    output_dir = os.path.join(args.output_dir, f"{args.model_alias}_{args.dataset}")
    save_analysis_results(results, output_dir, args.model_alias)
    
    # 打印摘要
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Layer: {results['layer_num']}")
    print(f"Sample pairs analyzed: {results['n_sample_pairs']}")
    print(f"Total features: {results['n_features']}")
    print(f"Mean absolute frequency difference: {results['overall_stats']['mean_abs_freq_diff']:.6f}")
    print(f"Max absolute frequency difference: {results['overall_stats']['max_abs_freq_diff']:.6f}")
    print(f"Mean absolute value difference: {results['overall_stats']['mean_abs_value_diff']:.6f}")
    print(f"Max absolute value difference: {results['overall_stats']['max_abs_value_diff']:.6f}")
    
    # 显示前10个最重要的特征
    print("\nTop 10 most important features (by frequency difference):")
    print("Feature\t|Freq Diff|\t|Value Diff|")
    print("-" * 60)
    for i, feature in enumerate(results['feature_importance'][:10]):
        print(f"{i:6d}\t{feature['abs_mean_freq_diff']:.6f}\t\t{feature['abs_mean_value_diff']:.6f}")

def calculate_sta_vector_for_layer() -> Dict:
    parser = argparse.ArgumentParser(description='Analyze SAE feature importance')
    parser.add_argument('--model_alias', type=str, default='gemma-2-9b-it', help='Model alias')
    parser.add_argument('--dataset', type=str, default='safeedit', help='Dataset name')
    parser.add_argument('--layer', type=int, default=20, help='Layer number to analyze')
    parser.add_argument('--activation_threshold', type=float, default=0.0, help='Activation threshold')
    parser.add_argument('--output_dir', type=str, default='./sae_analysis_results', help='Output directory')
    parser.add_argument('--trim', type=float, default=0.35, help='Trim ratio')
    parser.add_argument('--select_type', type=str, default='act_and_fre_trim', help='Select type')
    parser.add_argument('--sae_model', type=str, default='/home/wangxin/models/gemma-scope-9b-it-res/layer_20/width_16k/average_l0_91', help='SAE model')
    args = parser.parse_args()
    
    sae_model,_ = load_gemma_sae(args.sae_model)  
    sta_vector = calculate_steering_vector(os.path.join(DEFAULT_CACHE_DIR, args.model_alias+"_"+args.dataset), os.path.join(DEFAULT_CACHE_DIR, args.model_alias+"_"+args.dataset),args.layer,sae_model,args.trim, args.select_type,args.activation_threshold)
    vector = sta_vector["steering_vector"]
    feature_score = sta_vector["feature_score"]  # 获取没有mask后的SAE中间特征
    
    # 保存steering vector
    vector_filepath = os.path.join(args.output_dir, f"{args.model_alias}_{args.dataset}_layer_{args.layer}_{args.select_type}_{args.trim}.pt")
    torch.save(vector, vector_filepath)
    
    # 保存没有mask后的SAE中间特征（feature_score）
    feature_score_filepath = os.path.join(args.output_dir, f"{args.model_alias}_{args.dataset}_layer_{args.layer}_feature_score.pt")
    torch.save(feature_score, feature_score_filepath)
    
    # 保存特征重要性信息
    feature_importance_df = pd.DataFrame(sta_vector["feature_importance"])
    feature_filepath = os.path.join(args.output_dir, f"feature_importance_{args.model_alias}_{args.dataset}_layer_{args.layer}.csv")
    feature_importance_df.to_csv(feature_filepath, index=False)
    
    print(f"Steering vector saved: {vector_filepath}")
    print(f"Feature score (unmasked SAE features) saved: {feature_score_filepath}")
    print(f"Feature importance saved: {feature_filepath}")
    # 创建可视化
    #create_steering_vector_visualizations(sta_vector, args.output_dir, args.model_alias, args.dataset, args.select_type, args.trim)
    
    return sta_vector

if __name__ == "__main__":
    calculate_sta_vector_for_layer()

