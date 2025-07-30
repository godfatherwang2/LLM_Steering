import os
import json
import numpy as np
import pandas as pd
import argparse
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# 添加路径
import sys
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
    """计算每个特征的激活频率"""
    activations = features > activation_threshold
    activation_freq = np.mean(activations, axis=0)
    return activation_freq


def extract_top_features(intermediate_results_file: str, top_percentile: float = 0.1) -> List[int]:
    """从中间结果文件中提取前N%的特征维度"""
    with open(intermediate_results_file, 'r') as f:
        results = json.load(f)
    
    # 获取绝对平均频率差
    abs_mean_diffs = np.array(results["abs_mean_frequency_diffs"])
    
    # 计算前N%的阈值
    threshold = np.percentile(abs_mean_diffs, (1 - top_percentile) * 100)
    
    # 获取前N%的特征索引
    top_feature_indices = np.where(abs_mean_diffs >= threshold)[0].tolist()
    
    print(f"Total features: {len(abs_mean_diffs)}")
    print(f"Top {top_percentile*100}% features: {len(top_feature_indices)}")
    print(f"Threshold: {threshold:.6f}")
    
    return top_feature_indices


def load_features_for_classification(safe_cache_dir: str, unsafe_cache_dir: str, 
                                   layer_num: int, top_feature_indices: List[int],
                                   activation_threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """加载用于分类的特征数据"""
    print("Loading sample indices...")
    safe_indices, safe_folder = load_sample_indices(safe_cache_dir, "safe")
    unsafe_indices, unsafe_folder = load_sample_indices(unsafe_cache_dir, "unsafe")
    
    print(f"Safe samples: {len(safe_indices)}, Unsafe samples: {len(unsafe_indices)}")
    
    # 确保样本数量一致
    min_samples = min(len(safe_indices), len(unsafe_indices))
    safe_indices = safe_indices[:min_samples]
    unsafe_indices = unsafe_indices[:min_samples]
    
    # 存储特征和标签
    features_list = []
    labels_list = []
    
    print("Loading features for classification...")
    for i, (safe_info, unsafe_info) in enumerate(tqdm(zip(safe_indices, unsafe_indices))):
        try:
            # 加载SAE特征
            safe_features = load_sae_features_for_sample(safe_info, safe_folder, layer_num)
            unsafe_features = load_sae_features_for_sample(unsafe_info, unsafe_folder, layer_num)
            
            # 计算激活频率
            safe_freq = calculate_activation_frequency(safe_features, activation_threshold)
            unsafe_freq = calculate_activation_frequency(unsafe_features, activation_threshold)
            
            # 提取前N%的特征
            safe_top_features = safe_freq[top_feature_indices]
            unsafe_top_features = unsafe_freq[top_feature_indices]
            
            # 添加到数据集中
            features_list.extend([safe_top_features, unsafe_top_features])
            labels_list.extend([1, 0])  # 1表示安全，0表示非安全
            
        except Exception as e:
            print(f"Error processing sample pair {i}: {e}")
            continue
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    print(f"Final dataset shape: {features.shape}")
    print(f"Labels distribution: {np.bincount(labels)}")
    
    return features, labels


def train_classifiers(X_train: np.ndarray, y_train: np.ndarray, 
                     X_test: np.ndarray, y_test: np.ndarray,
                     feature_indices: List[int]) -> Dict:
    """训练多个分类器并评估性能"""
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        
        # 训练分类器
        clf.fit(X_train, y_train)
        
        # 预测
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
        
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'classifier': clf,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"{name} Classification Report:")
        print(classification_report(y_test, y_pred))
    
    return results


def save_classification_results(results: Dict, output_dir: str, model_alias: str, 
                              layer_num: int, feature_indices: List[int]):
    """保存分类结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存分类器模型
    for name, result in results.items():
        model_filepath = os.path.join(output_dir, f"{name.lower()}_{model_alias}_layer_{layer_num}.joblib")
        joblib.dump(result['classifier'], model_filepath)
        print(f"Saved {name} model: {model_filepath}")
    
    # 保存特征索引
    feature_indices_filepath = os.path.join(output_dir, f"top_features_{model_alias}_layer_{layer_num}.json")
    with open(feature_indices_filepath, 'w') as f:
        json.dump({
            'feature_indices': feature_indices,
            'n_features': len(feature_indices)
        }, f, indent=4)
    
    # 保存分类结果
    classification_results = {}
    for name, result in results.items():
        classification_results[name] = {
            'accuracy': result['accuracy'],
            'classification_report': result['classification_report'],
            'confusion_matrix': result['confusion_matrix'].tolist()
        }
    
    results_filepath = os.path.join(output_dir, f"classification_results_{model_alias}_layer_{layer_num}.json")
    with open(results_filepath, 'w') as f:
        json.dump(classification_results, f, indent=4)
    
    print(f"Classification results saved: {results_filepath}")


def create_classification_plots(results: Dict, output_dir: str, model_alias: str, layer_num: int):
    """创建分类结果的可视化图表"""
    n_classifiers = len(results)
    fig, axes = plt.subplots(2, n_classifiers, figsize=(5*n_classifiers, 10))
    
    if n_classifiers == 1:
        axes = axes.reshape(2, 1)
    
    for i, (name, result) in enumerate(results.items()):
        # 混淆矩阵
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, i])
        axes[0, i].set_title(f'{name} Confusion Matrix')
        axes[0, i].set_xlabel('Predicted')
        axes[0, i].set_ylabel('Actual')
        
        # 准确率条形图
        metrics = ['precision', 'recall', 'f1-score']
        scores = [result['classification_report']['weighted avg'][metric] for metric in metrics]
        axes[1, i].bar(metrics, scores, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[1, i].set_title(f'{name} Metrics')
        axes[1, i].set_ylabel('Score')
        axes[1, i].set_ylim(0, 1)
        for j, score in enumerate(scores):
            axes[1, i].text(j, score + 0.01, f'{score:.3f}', ha='center')
    
    plt.tight_layout()
    plot_filepath = os.path.join(output_dir, f"classification_plots_{model_alias}_layer_{layer_num}.png")
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Classification plots saved: {plot_filepath}")


def main():
    parser = argparse.ArgumentParser(description="Train SAE feature classifier")
    parser.add_argument("--safe_cache_dir", type=str,default="/home/wangxin/projects/safety_steer/dataset/cached_activations_sae/gemma-2-9b-it_safeedit", help="Safe samples cache directory")
    parser.add_argument("--unsafe_cache_dir", type=str,default="/home/wangxin/projects/safety_steer/dataset/cached_activations_sae/gemma-2-9b-it_safeedit", help="Unsafe samples cache directory")
    parser.add_argument("--intermediate_results", type=str,default="./sae_analysis_results/gemma-2-9b-it_safeedit/intermediate_results_gemma-2-9b-it_layer_20.json", help="Intermediate results file path")
    parser.add_argument("--output_dir", type=str,default="./output/gemma-2-9b-it_safeedit/classifier", help="Output directory")
    parser.add_argument("--model_alias", type=str,default="gemma-2-9b-it", help="Model alias")
    parser.add_argument("--layer", type=int, default=20, help="Layer number")
    parser.add_argument("--activation_threshold", type=float, default=0.0, help="Activation threshold")
    parser.add_argument("--top_percentile", type=float, default=0.1, help="Top percentile of features to use")
    parser.add_argument("--test_size", type=float, default=0.5, help="Test set size")
    parser.add_argument("--random_state", type=int, default=42, help="Random state")
    
    args = parser.parse_args()
    
    print("=== SAE Feature Classification ===")
    print(f"Model: {args.model_alias}")
    print(f"Layer: {args.layer}")
    print(f"Top percentile: {args.top_percentile*100}%")
    print(f"Activation threshold: {args.activation_threshold}")
    
    # 1. 提取前N%的特征维度
    print("\n1. Extracting top features...")
    top_feature_indices = extract_top_features(args.intermediate_results, args.top_percentile)
    
    # 2. 加载特征数据
    print("\n2. Loading features for classification...")
    features, labels = load_features_for_classification(
        args.safe_cache_dir, args.unsafe_cache_dir, args.layer, 
        top_feature_indices, args.activation_threshold
    )
    
    # 3. 数据预处理
    print("\n3. Preprocessing data...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 4. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=args.test_size, 
        random_state=args.random_state, stratify=labels
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 5. 训练分类器
    print("\n4. Training classifiers...")
    results = train_classifiers(X_train, y_train, X_test, y_test, top_feature_indices)
    
    # 6. 保存结果
    print("\n5. Saving results...")
    save_classification_results(results, args.output_dir, args.model_alias, args.layer, top_feature_indices)
    
    # 7. 创建可视化
    print("\n6. Creating visualizations...")
    create_classification_plots(results, args.output_dir, args.model_alias, args.layer)
    
    # 8. 保存scaler
    scaler_filepath = os.path.join(args.output_dir, f"scaler_{args.model_alias}_layer_{args.layer}.joblib")
    joblib.dump(scaler, scaler_filepath)
    print(f"Saved scaler: {scaler_filepath}")
    
    print("\n=== Classification Complete ===")
    
    # 显示最佳结果
    best_classifier = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"Best classifier: {best_classifier[0]} (Accuracy: {best_classifier[1]['accuracy']:.4f})")


if __name__ == "__main__":
    main() 