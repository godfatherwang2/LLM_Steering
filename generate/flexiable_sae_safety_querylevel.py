import os
import sys
sys.path.append('./')
sys.path.append('../')
import torch
import faiss
import numpy as np
import argparse
import json
from typing import List, Dict, Optional
from tqdm import tqdm
from transformer_lens import HookedTransformer
from utils.utils import model_alias_to_model_name
from dataset.load_data import load_safeedit_test_data, load_gsm_test_data, load_samsum_test_data, load_strongreject_test_data, load_beavertails_test_data
from transformers import StoppingCriteria
from sae_lens import SAE
from sae_utils import load_gemma_2_sae
from build_faiss_index import build_faiss_index_from_cache_sae


# 全局路径管理字典
def get_paths_config(model_alias: str, dataset_name: str, layer: int, shared_size: int, search_mode: str = "activation", 
                    top_freq: float = 0.1, top_value: float = 0.1, filter_type: str = "mix"):
    """获取路径配置"""
    base_cache_dir = "dataset/cached_activations_sae"
    
    # 构建选择策略和阈值的路径标识
    selection_info = f"freq{int(top_freq*100)}p_value{int(top_value*100)}p_{filter_type}"
    
    config = {
        "base_cache_dir": base_cache_dir,
        "model_alias": model_alias,
        "dataset_name": dataset_name,
        "layer": layer,
        "shared_size": shared_size,
        "search_mode": search_mode,
        "selection_info": selection_info,
        # 缓存目录
        "safe_cache_dir": os.path.join(base_cache_dir, f"{model_alias}_{dataset_name}", f"label_safe_shard_size_{shared_size}"),
        "unsafe_cache_dir": os.path.join(base_cache_dir, f"{model_alias}_{dataset_name}", f"label_unsafe_shard_size_{shared_size}"),
        # 激活值索引路径
        "safe_acts_index_path": os.path.join(base_cache_dir, f"{model_alias}_{dataset_name}", f"label_safe_shard_size_{shared_size}", f"faiss_index_acts_layer_{layer}.index"),
        "unsafe_acts_index_path": os.path.join(base_cache_dir, f"{model_alias}_{dataset_name}", f"label_unsafe_shard_size_{shared_size}", f"faiss_index_acts_layer_{layer}.index"),
        # 查询對應隐藏状态索引路径
        "safe_query_acts_index_path": os.path.join(base_cache_dir, f"{model_alias}_{dataset_name}", f"label_safe_shard_size_{shared_size}", f"faiss_index_query_acts_layer_{layer}.index"),
        "unsafe_query_acts_index_path": os.path.join(base_cache_dir, f"{model_alias}_{dataset_name}", f"label_unsafe_shard_size_{shared_size}", f"faiss_index_query_acts_layer_{layer}.index"),
    }
    
    return config


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

class SAESteeringHookQueryLevel:
    """
    基于查询后隐藏状态的SAE特征引导钩子管理器
    """
    def __init__(self, 
                 paths_config: Dict,
                 top_k: int = 10, 
                 multiplier: float = 1.0, 
                 norm_magnitude: bool = False, 
                 caa_vectors_dir: str = "results/caa_vectors", 
                 top_freq: float = 0.1,
                 top_value: float = 0.1,
                 filter_type: str = "mix",
                 sae_layer: int = None,
                 sae_model_path: str = None,
                 search_mode: str = "activation",
                 static_weight: float = 0.5,
                 static_steer_vector_path: str = None,
                 steer_pos: str = "all",
                 distance_metric: str = "euclidean",
                 use_softmax_weight: bool = True):
        """
        初始化SAE特征引导钩子

        Args:
            paths_config: 路径配置字典
            top_k: FAISS检索的最近邻数量
            multiplier: 引导向量的强度系数
            norm_magnitude: 是否将steer_vector缩放到与CAA向量相同的范数
            caa_vectors_dir: CAA向量目录
            top_freq: 使用前N%的频率差异特征
            top_value: 使用前N%的激活值差异特征
            filter_type: 特征筛选类型 ("freq", "value", "mix")
            sae_layer: SAE特征对应的层号
            sae_model_path: SAE模型路径
            search_mode: 搜索模式 ("activation", "keySAE", "hybrid", "fixed_features")
            static_weight: 混合模式中静态向量的权重 (0.0-1.0)
            static_steer_vector_path: 静态预计算的steer_vector路径
            steer_pos: 引导位置 ("all", "current") - all: 修改所有token激活值, current: 仅修改当前token激活值
            distance_metric: 距离度量 ("euclidean", "cosine") - euclidean: 欧几里得距离, cosine: 余弦相似度
            use_softmax_weight: 是否使用softmax加权 (仅适用于activation模式) - True: 使用softmax加权, False: 使用简单平均
        """
        print("--- 初始化SAE特征引导钩子 (Query Level) ---")
        self.paths_config = paths_config
        self.top_k = top_k
        self.multiplier = multiplier
        self.sae_layer = sae_layer or paths_config["layer"]
        self.sae_model_path = sae_model_path
        self.search_mode = search_mode
        self.norm_magnitude = norm_magnitude
        self.static_weight = static_weight
        self.filter_type = filter_type
        self.top_freq = top_freq
        self.top_value = top_value
        self.steer_pos = steer_pos
        self.distance_metric = distance_metric
        self.use_softmax_weight = use_softmax_weight

        # 混合模式和固定特征模式相关初始化
        if self.search_mode in ["hybrid", "fixed_features"]:
            if static_steer_vector_path is None:
                # 使用默认路径构建静态向量路径
                static_steer_vector_path = f"/home/wangxin/projects/safety_steer/sae_analysis_results/gemma-2-9b-it_safeedit_layer_20_act_and_fre_trim_0.35.pt"
            
            self.static_steer_vector = self._load_static_steer_vector(static_steer_vector_path)
            
            if self.search_mode == "hybrid":
                print(f"混合模式：静态向量权重 = {self.static_weight}, 动态向量权重 = {1.0 - self.static_weight}")
            elif self.search_mode == "fixed_features":
                print(f"固定特征模式：使用全局预计算的SAE特征差值")
        else:
            self.static_steer_vector = None
        
        # 如果启用norm_magnitude，加载CAA向量
        if self.norm_magnitude:
            if caa_vectors_dir and paths_config["model_alias"] and paths_config["dataset_name"] and paths_config["layer"] is not None:
                self.caa_vector = self._load_caa_vector(caa_vectors_dir, paths_config["model_alias"], paths_config["dataset_name"], paths_config["layer"])
                print(f"加载CAA向量用于范数归一化，范数: {torch.norm(self.caa_vector, p=2):.6f}")
            else:
                raise ValueError("启用norm_magnitude时需要提供caa_vectors_dir, model_alias, dataset_name, layer参数")
        else:
            self.caa_vector = None
        

        # 注释掉FAISS索引加载，改为直接使用numpy计算距离
        # 加载查询后隐藏状态FAISS索引
        # print(f"加载安全查询后隐藏状态FAISS索引: {paths_config['safe_query_acts_index_path']}")
        # self.safe_query_acts_index = faiss.read_index(paths_config['safe_query_acts_index_path'])
        # print(f"加载不安全查询后隐藏状态FAISS索引: {paths_config['unsafe_query_acts_index_path']}")
        # self.unsafe_query_acts_index = faiss.read_index(paths_config['unsafe_query_acts_index_path'])

        # 加载查询后隐藏状态数据（query_acts.dat）
        print("加载查询后隐藏状态缓存数据...")
        safe_query_acts_file = os.path.join(paths_config["safe_cache_dir"], "query_acts.dat")
        unsafe_query_acts_file = os.path.join(paths_config["unsafe_cache_dir"], "query_acts.dat")
        
        # 获取元数据以确定形状
        safe_acts_metadata_file = os.path.join(paths_config["safe_cache_dir"], "metadata.json")
        unsafe_acts_metadata_file = os.path.join(paths_config["unsafe_cache_dir"], "metadata.json")
        
        with open(safe_acts_metadata_file, 'r') as f:
            safe_acts_metadata = json.load(f)
        with open(unsafe_acts_metadata_file, 'r') as f:
            unsafe_acts_metadata = json.load(f)
            
        n_layers, d_model = safe_acts_metadata["activations_original_shape"][0], safe_acts_metadata["activations_original_shape"][2]
        
        # 加载样本索引以确定样本数量
        safe_sample_indices_file = os.path.join(paths_config["safe_cache_dir"], "sample_indices.json")
        unsafe_sample_indices_file = os.path.join(paths_config["unsafe_cache_dir"], "sample_indices.json")
        
        with open(safe_sample_indices_file, 'r') as f:
            safe_sample_indices = json.load(f)
        with open(unsafe_sample_indices_file, 'r') as f:
            unsafe_sample_indices = json.load(f)
            
        n_safe_samples = len(safe_sample_indices)
        n_unsafe_samples = len(unsafe_sample_indices)
        
        self.safe_query_acts_array = np.ascontiguousarray(np.memmap(safe_query_acts_file, dtype='float32', mode='r', shape=(n_safe_samples, n_layers, d_model)))
        self.unsafe_query_acts_array = np.ascontiguousarray(np.memmap(unsafe_query_acts_file, dtype='float32', mode='r', shape=(n_unsafe_samples, n_layers, d_model)))
        
        print(f"安全查询后隐藏状态形状: {self.safe_query_acts_array.shape}")
        print(f"不安全查询后隐藏状态形状: {self.unsafe_query_acts_array.shape}")
        
        # 加载SAE模型
        self.sae_model = self._load_sae_model()
        
        # 加载平均SAE特征数据（avg_sae.dat）
        print("加载平均SAE特征缓存数据...")
        safe_avg_sae_file = os.path.join(paths_config["safe_cache_dir"], "avg_sae.dat")
        unsafe_avg_sae_file = os.path.join(paths_config["unsafe_cache_dir"], "avg_sae.dat")
        
        sae_features_dim = safe_acts_metadata["sae_features_dim"]
        
        self.safe_avg_sae_features = np.ascontiguousarray(np.memmap(safe_avg_sae_file, dtype='float32', mode='r', 
                                                                   shape=(n_safe_samples, n_layers, sae_features_dim)))
        self.unsafe_avg_sae_features = np.ascontiguousarray(np.memmap(unsafe_avg_sae_file, dtype='float32', mode='r', 
                                                                     shape=(n_unsafe_samples, n_layers, sae_features_dim)))
        
        print(f"安全平均SAE特征形状: {self.safe_avg_sae_features.shape}")
        print(f"不安全平均SAE特征形状: {self.unsafe_avg_sae_features.shape}")
        
        # 加载重要SAE特征维度
        self.important_sae_features = self._load_important_sae_features(
            paths_config["model_alias"], paths_config["dataset_name"], top_freq, top_value, filter_type
        )
            
        # 加载SAE Decoder权重
        self.sae_decoder_weights = self._load_sae_decoder_weights()
        
        # 初始化steer_vector为None，将在第一次生成时计算
        self.steer_vector = None
        self.is_steer_vector_computed = False
        
        # 加载全局SAE特征差值（用于fixed_features模式）
        if self.search_mode == "fixed_features":
            self.global_sae_features = self._load_global_sae_features()

    def _load_caa_vector(self, caa_vectors_dir: str, model_alias: str, dataset_name: str, layer: int) -> torch.Tensor:
        """加载CAA向量"""
        filename = f"{layer}.pt"
        filepath = os.path.join(caa_vectors_dir,f"{model_alias}_{dataset_name}",filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CAA向量文件不存在: {filepath}")
        
        caa_vector = torch.load(filepath)
        print(f"加载CAA向量: {filepath}, 形状: {caa_vector.shape}")
        return caa_vector

    def _load_static_steer_vector(self, static_steer_vector_path: str) -> torch.Tensor:
        """加载静态预计算的steer_vector"""
        if not os.path.exists(static_steer_vector_path):
            raise FileNotFoundError(f"静态steer_vector文件不存在: {static_steer_vector_path}")
        
        static_steer_vector = torch.load(static_steer_vector_path)
        print(f"加载静态steer_vector: {static_steer_vector_path}, 形状: {static_steer_vector.shape}")
        return static_steer_vector

    def _load_global_sae_features(self) -> torch.Tensor:
        """加载全局SAE特征差值（feature_score）"""
        # 构建全局SAE特征文件路径
        global_sae_features_path = f"./sae_analysis_results/{self.paths_config['model_alias']}_{self.paths_config['dataset_name']}_layer_{self.sae_layer}_feature_score.pt"
        
        if not os.path.exists(global_sae_features_path):
            raise FileNotFoundError(f"全局SAE特征文件不存在: {global_sae_features_path}")
        
        global_sae_features = torch.load(global_sae_features_path)
        print(f"加载全局SAE特征差值: {global_sae_features_path}, 形状: {global_sae_features.shape}")
        return global_sae_features

    def _calculate_distance(self, query_vector: np.ndarray, target_vectors: np.ndarray) -> np.ndarray:
        """计算查询向量与目标向量之间的距离"""
        if self.distance_metric == "euclidean":
            # 欧几里得距离
            distances = np.linalg.norm(target_vectors - query_vector, axis=1)
        elif self.distance_metric == "cosine":
            # 余弦相似度（转换为距离：1 - 相似度）
            # 归一化向量
            query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
            target_norms = target_vectors / (np.linalg.norm(target_vectors, axis=1, keepdims=True) + 1e-8)
            # 计算余弦相似度
            similarities = np.dot(target_norms, query_norm.T).flatten()
            # 转换为距离（1 - 相似度）
            distances = 1 - similarities
        else:
            raise ValueError(f"不支持的距离度量: {self.distance_metric}")
        
        return distances

    def _load_sae_model(self):
        """加载SAE模型"""
        print("加载SAE模型...")
        if self.sae_model_path is None:
            # 使用默认路径
            sae_path = "/home/wangxin/models/gemma-scope-9b-it-res/layer_20/width_16k/average_l0_91"
        else:
            sae_path = self.sae_model_path
            
        sae_model, sparsity = load_gemma_2_sae(sae_path, device="cuda")
        sae_model.eval()
        print(f"SAE模型加载完成，稀疏度: {sparsity}")
        return sae_model

    def _load_important_sae_features(self, model_alias: str, dataset_name: str, top_freq: float, top_value: float, filter_type: str) -> List[int]:
        """加载重要的SAE特征维度"""
        print(f"加载重要SAE特征维度 (freq_top{top_freq*100}%, value_top{top_value*100}%, filter_type={filter_type})...")
        
        # 构建特征重要性文件路径 - 使用新的命名格式
        feature_file = f"./sae_analysis_results/feature_importance_{model_alias}_{dataset_name}_layer_{self.sae_layer}.csv"
        
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"特征重要性文件不存在: {feature_file}")
        
        import pandas as pd
        feature_df = pd.read_csv(feature_file)
        total_features = len(feature_df)
        
        # 计算前N%的数量
        top_freq_count = int(total_features * top_freq)
        top_value_count = int(total_features * top_value)
        
        # 按频率差异绝对值排序
        freq_sorted = feature_df.sort_values("abs_mean_freq_diff", ascending=False)
        # 使用行索引作为特征索引（因为简化后的数据结构不包含feature_idx列）
        top_freq_features = freq_sorted.head(top_freq_count).index.tolist()
        
        # 按激活值差异绝对值排序
        value_sorted = feature_df.sort_values("abs_mean_value_diff", ascending=False)
        # 使用行索引作为特征索引
        top_value_features = value_sorted.head(top_value_count).index.tolist()
        
        # 根据filter_type选择特征
        if filter_type == "freq":
            important_features = top_freq_features
        elif filter_type == "value":
            important_features = top_value_features
        elif filter_type == "mix":
            # 取交集，但保持重要性排序
            intersection = set(top_freq_features) & set(top_value_features)
            # 按照频率差异的重要性重新排序交集
            intersection_features = [idx for idx in top_freq_features if idx in intersection]
            important_features = intersection_features
        else:
            raise ValueError(f"不支持的filter_type: {filter_type}")
        
        # 调整特征维度以确保FAISS兼容性
        n_important_features = len(important_features)
        m_candidates = [8, 16, 32, 64, 128, 256]
        valid_m = []
        for m in m_candidates:
            if n_important_features % m == 0:
                valid_m.append(m)
        
        if not valid_m:
            # 如果没有找到合适的m，调整特征维度到最近的能被8整除的值
            adjusted_n_features = (n_important_features // 8) * 8
            if adjusted_n_features == 0:
                adjusted_n_features = 8
            print(f"警告: 特征维度 {n_important_features} 不能被常用的m值整除")
            print(f"调整特征维度到 {adjusted_n_features} (减少 {n_important_features - adjusted_n_features} 个重要性较低的特征)")
            
            # 调整特征选择（保留最重要的特征）
            if adjusted_n_features < n_important_features:
                important_features = important_features[:adjusted_n_features]
                print(f"调整后的重要SAE特征维度数量: {len(important_features)}")
        
        print(f"总特征数量: {total_features}")
        print(f"频率差异前{top_freq*100}%特征数量: {len(top_freq_features)}")
        print(f"激活值差异前{top_value*100}%特征数量: {len(top_value_features)}")
        print(f"最终重要SAE特征数量: {len(important_features)}")
        
        return important_features

    def _load_sae_decoder_weights(self) -> torch.Tensor:
        """加载SAE Decoder权重"""
        decoder_weights = self.sae_model.W_dec
        print(f"加载SAE Decoder权重, 形状: {decoder_weights.shape}")
        return decoder_weights

    def reset_state(self):
        """在每次新的生成任务开始前重置状态"""
        self.steer_vector = None
        self.is_steer_vector_computed = False

    def _calculate_sae_steer_vector(self, query_activation: torch.Tensor) -> torch.Tensor:
        """计算基于查询后隐藏状态的SAE特征引导向量"""
        if self.search_mode == "hybrid":
            return self._calculate_sae_steer_vector_hybrid_mode(query_activation)
        elif self.search_mode == "fixed_features":
            return self._calculate_sae_steer_vector_fixed_features_mode(query_activation)
        else:
            return self._calculate_sae_steer_vector_activation_mode(query_activation)

    def _calculate_sae_steer_vector_activation_mode(self, query_activation: torch.Tensor) -> torch.Tensor:
        """计算基于查询后隐藏状态的SAE特征引导向量 - activation模式"""
        # 1. 准备查询向量（使用查询后隐藏状态进行FAISS检索）
        query_vector = query_activation.detach().cpu().float().numpy().reshape(1, -1)
        
        # 2. 直接使用numpy计算距离，找到最近邻
        # 注释掉FAISS索引搜索，改为直接计算
        # safe_dists, safe_indices = self.safe_query_acts_index.search(query_vector, self.top_k)
        # unsafe_dists, unsafe_indices = self.unsafe_query_acts_index.search(query_vector, self.top_k)
        
        # 直接计算查询向量与所有安全查询激活值的距离
        safe_query_acts_all = self.safe_query_acts_array[:, 0, :]  # [n_safe_samples, d_model]
        safe_dists_all = self._calculate_distance(query_vector, safe_query_acts_all)  # [n_safe_samples]
        safe_indices = np.argsort(safe_dists_all)[:self.top_k]  # [top_k]
        safe_dists = safe_dists_all[safe_indices]  # [top_k]
        
        # 直接计算查询向量与所有不安全查询激活值的距离
        unsafe_query_acts_all = self.unsafe_query_acts_array[:, 0, :]  # [n_unsafe_samples, d_model]
        unsafe_dists_all = self._calculate_distance(query_vector, unsafe_query_acts_all)  # [n_unsafe_samples]
        unsafe_indices = np.argsort(unsafe_dists_all)[:self.top_k]  # [top_k]
        unsafe_dists = unsafe_dists_all[unsafe_indices]  # [top_k]
        
        # 直接使用已计算的距离，无需重新计算
        safe_dists_recomputed = safe_dists  # [top_k]
        unsafe_dists_recomputed = unsafe_dists  # [top_k]
        
        # 4. 根据use_softmax_weight选项计算权重
        if self.use_softmax_weight:
            # 使用softmax加权
            safe_dists_normalized = (safe_dists_recomputed - safe_dists_recomputed.min()) / (safe_dists_recomputed.max() - safe_dists_recomputed.min() + 1e-8)
            unsafe_dists_normalized = (unsafe_dists_recomputed - unsafe_dists_recomputed.min()) / (unsafe_dists_recomputed.max() - unsafe_dists_recomputed.min() + 1e-8)
            
            safe_weights = torch.softmax(torch.tensor(-safe_dists_normalized, dtype=torch.float32), dim=0)  # [top_k]
            unsafe_weights = torch.softmax(torch.tensor(-unsafe_dists_normalized, dtype=torch.float32), dim=0)  # [top_k]
        else:
            # 使用简单平均（等权重）
            safe_weights = torch.ones(self.top_k, dtype=torch.float32) / self.top_k  # [top_k]
            unsafe_weights = torch.ones(self.top_k, dtype=torch.float32) / self.top_k  # [top_k]
        
        # 5. 加载这些索引对应的平均SAE特征
        safe_avg_sae_features = self.safe_avg_sae_features[safe_indices, 0, :]  # [top_k, sae_features_dim]
        unsafe_avg_sae_features = self.unsafe_avg_sae_features[unsafe_indices, 0, :]  # [top_k, sae_features_dim]
        
        # 6. 对安全/不安全SAE向量进行加权平均
        safe_sae_weighted = torch.sum(safe_weights.unsqueeze(1) * torch.tensor(safe_avg_sae_features, dtype=torch.float32), dim=0)  # [sae_features_dim]
        unsafe_sae_weighted = torch.sum(unsafe_weights.unsqueeze(1) * torch.tensor(unsafe_avg_sae_features, dtype=torch.float32), dim=0)  # [sae_features_dim]
        
        # 7. 相减得到SAE特征差异
        sae_diff = safe_sae_weighted - unsafe_sae_weighted  # [sae_features_dim]
        
        # 8. Mask掉非重要维度
        if self.important_sae_features is not None:
            mask = torch.zeros_like(sae_diff)
            mask[self.important_sae_features] = 1.0
            sae_diff = sae_diff * mask
        
        # 9. 与W_dec相乘计算得到最终steer_vector
        steer_vector = sae_diff @ self.sae_decoder_weights.cpu()  # [d_model]
        
        # 10. 应用强度系数
        steer_vector = steer_vector * self.multiplier
        
        # 11. 如果启用范数归一化，缩放到与CAA向量相同的范数
        if self.norm_magnitude and self.caa_vector is not None:
            caa_norm = torch.norm(self.caa_vector, p=2)
            steer_norm = torch.norm(steer_vector, p=2)
            if steer_norm > 0:
                steer_vector = steer_vector * (caa_norm / steer_norm)
        
        return steer_vector

    def _calculate_sae_steer_vector_hybrid_mode(self, query_activation: torch.Tensor) -> torch.Tensor:
        """计算基于查询后隐藏状态的SAE特征引导向量 - 混合模式"""
        # 1. 计算动态steer_vector（与activation模式类似，但使用最近邻样本的重要特征维度）
        dynamic_steer_vector = self._calculate_dynamic_steer_vector_from_neighbors(query_activation)
        
        # 2. 获取静态steer_vector
        static_steer_vector = self.static_steer_vector.clone()
        
        # 3. 加权融合
        hybrid_steer_vector = (self.static_weight * static_steer_vector + 
                              (1.0 - self.static_weight) * dynamic_steer_vector)
        
        # 5. 应用强度系数
        hybrid_steer_vector = hybrid_steer_vector * self.multiplier
        
        # 6. 如果启用范数归一化，缩放到与CAA向量相同的范数
        if self.norm_magnitude and self.caa_vector is not None:
            caa_norm = torch.norm(self.caa_vector, p=2)
            steer_norm = torch.norm(hybrid_steer_vector, p=2)
            if steer_norm > 0:
                hybrid_steer_vector = hybrid_steer_vector * (caa_norm / steer_norm)
        
        return hybrid_steer_vector

    def _calculate_dynamic_steer_vector_from_neighbors(self, query_activation: torch.Tensor) -> torch.Tensor:
        """基于最近邻样本计算动态steer_vector，参考analyse文件的calculate_steering_vector函数"""
        # 1. 准备查询向量（使用查询后隐藏状态进行检索）
        query_vector = query_activation.detach().cpu().float().numpy().reshape(1, -1)
        
        # 2. 找到最近邻
        safe_query_acts_all = self.safe_query_acts_array[:, 0, :]  # [n_safe_samples, d_model]
        safe_dists_all = self._calculate_distance(query_vector, safe_query_acts_all)  # [n_safe_samples]
        safe_indices = np.argsort(safe_dists_all)[:self.top_k]  # [top_k]
        
        unsafe_query_acts_all = self.unsafe_query_acts_array[:, 0, :]  # [n_unsafe_samples, d_model]
        unsafe_dists_all = self._calculate_distance(query_vector, unsafe_query_acts_all)  # [n_unsafe_samples]
        unsafe_indices = np.argsort(unsafe_dists_all)[:self.top_k]  # [top_k]
        
        # 3. 获取最近邻样本的SAE特征
        safe_feature_acts = self.safe_avg_sae_features[safe_indices, 0, :]  # [top_k, sae_features_dim]
        unsafe_feature_acts = self.unsafe_avg_sae_features[unsafe_indices, 0, :]  # [top_k, sae_features_dim]
        
        # 4. 参考analyse文件的calculate_steering_vector函数计算统计量
        activation_threshold = 0.0
        
        # 计算激活频率（与analyse文件一致）
        pos_feature_freq = (safe_feature_acts > activation_threshold).astype(float).sum(0)  # [sae_features_dim]
        neg_feature_freq = (unsafe_feature_acts > activation_threshold).astype(float).sum(0)  # [sae_features_dim]
        
        # 计算平均激活值（与analyse文件一致）
        pos_act_mean = safe_feature_acts.mean(0)  # [sae_features_dim]
        neg_act_mean = unsafe_feature_acts.mean(0)  # [sae_features_dim]
        
        # 计算特征得分（与analyse文件一致）
        feature_score = pos_act_mean - neg_act_mean  # [sae_features_dim]
        
        # 转换为torch tensor
        pos_feature_freq = torch.tensor(pos_feature_freq, dtype=torch.float32)
        neg_feature_freq = torch.tensor(neg_feature_freq, dtype=torch.float32)
        pos_act_mean = torch.tensor(pos_act_mean, dtype=torch.float32)
        neg_act_mean = torch.tensor(neg_act_mean, dtype=torch.float32)
        feature_score = torch.tensor(feature_score, dtype=torch.float32)
        
        # 5. 根据filter_type选择重要特征维度（参考analyse文件的逻辑）
        sae_features_dim = len(feature_score)
        
        if self.filter_type == "mix":
            # 使用act_and_fre_trim逻辑（参考analyse文件）
            diff_data = pos_feature_freq - neg_feature_freq
            
            # Min-Max归一化，保留正负符号
            norm_act = self._signed_min_max_normalize(feature_score)
            norm_diff = self._signed_min_max_normalize(diff_data)
            
            # 符号一致性筛选
            mask = ((norm_act > 0) & (norm_diff > 0)) | ((norm_act < 0) & (norm_diff < 0))
            
            # 综合得分计算
            scores = torch.zeros_like(norm_diff)
            scores[mask] = norm_diff[mask]
            
            # 硬阈值筛选
            trim = self.top_freq  # 使用top_freq作为trim比例
            threshold_fre = torch.sort(torch.abs(scores), descending=True, stable=True).values[int(trim * len(scores))]
            prune_mask = torch.abs(scores) >= threshold_fre
            
            # 激活值硬阈值筛选
            act_threshold = torch.sort(torch.abs(feature_score), descending=True, stable=True).values[int(trim * len(feature_score))]
            act_top_mask = torch.abs(feature_score) >= act_threshold
            
            # 结合掩码（硬结合）
            combined_mask = prune_mask & act_top_mask
            
            # 生成steering vector
            act_data_combined = feature_score.clone()
            act_data_combined[~combined_mask] = 0
            
        elif self.filter_type == "freq":
            # 使用fre_trim逻辑（参考analyse文件）
            diff_data = pos_feature_freq - neg_feature_freq
            norm_diff = self._signed_min_max_normalize(diff_data)
            
            trim = self.top_freq
            threshold_fre = torch.sort(torch.abs(norm_diff), descending=True, stable=True).values[int(trim * len(norm_diff))]
            prune_mask = torch.abs(norm_diff) >= threshold_fre
            
            act_data_combined = feature_score.clone()
            act_data_combined[~prune_mask] = 0
            
        elif self.filter_type == "value":
            # 使用act_trim逻辑（参考analyse文件）
            trim = self.top_value
            act_threshold = torch.sort(torch.abs(feature_score), descending=True, stable=True).values[int(trim * len(feature_score))]
            act_mask = torch.abs(feature_score) >= act_threshold
            
            act_data_combined = feature_score.clone()
            act_data_combined[~act_mask] = 0
            
        else:
            raise ValueError(f"不支持的filter_type: {self.filter_type}")
        
        # 6. 与W_dec相乘计算得到动态steer_vector
        dynamic_steer_vector = act_data_combined @ self.sae_decoder_weights.cpu()  # [d_model]
        
        return dynamic_steer_vector

    def _calculate_sae_steer_vector_fixed_features_mode(self, query_activation: torch.Tensor) -> torch.Tensor:
        """计算基于固定特征的SAE特征引导向量 - fixed_features模式"""
        # 1. 准备查询向量（使用查询后隐藏状态进行检索）
        query_vector = query_activation.detach().cpu().float().numpy().reshape(1, -1)
        
        # 2. 找到最近邻
        safe_query_acts_all = self.safe_query_acts_array[:, 0, :]  # [n_safe_samples, d_model]
        safe_dists_all = np.linalg.norm(safe_query_acts_all - query_vector, axis=1)  # [n_safe_samples]
        safe_indices = np.argsort(safe_dists_all)[:self.top_k]  # [top_k]
        
        unsafe_query_acts_all = self.unsafe_query_acts_array[:, 0, :]  # [n_unsafe_samples, d_model]
        unsafe_dists_all = np.linalg.norm(unsafe_query_acts_all - query_vector, axis=1)  # [n_unsafe_samples]
        unsafe_indices = np.argsort(unsafe_dists_all)[:self.top_k]  # [top_k]
        
        # 3. 获取最近邻样本的SAE特征
        safe_feature_acts = self.safe_avg_sae_features[safe_indices, 0, :]  # [top_k, sae_features_dim]
        unsafe_feature_acts = self.unsafe_avg_sae_features[unsafe_indices, 0, :]  # [top_k, sae_features_dim]
        
        # 4. 计算动态重要特征维度（基于最近邻样本）
        activation_threshold = 0.0
        
        # 计算激活频率
        pos_feature_freq = (safe_feature_acts > activation_threshold).astype(float).sum(0)  # [sae_features_dim]
        neg_feature_freq = (unsafe_feature_acts > activation_threshold).astype(float).sum(0)  # [sae_features_dim]
        
        # 计算平均激活值
        pos_act_mean = safe_feature_acts.mean(0)  # [sae_features_dim]
        neg_act_mean = unsafe_feature_acts.mean(0)  # [sae_features_dim]
        
        # 计算特征得分
        feature_score = pos_act_mean - neg_act_mean  # [sae_features_dim]
        
        # 转换为torch tensor
        pos_feature_freq = torch.tensor(pos_feature_freq, dtype=torch.float32)
        neg_feature_freq = torch.tensor(neg_feature_freq, dtype=torch.float32)
        feature_score = torch.tensor(feature_score, dtype=torch.float32)
        
        # 5. 根据filter_type选择动态重要特征维度
        sae_features_dim = len(feature_score)
        
        if self.filter_type == "mix":
            # 使用act_and_fre_trim逻辑
            diff_data = pos_feature_freq - neg_feature_freq
            
            # Min-Max归一化，保留正负符号
            norm_act = self._signed_min_max_normalize(feature_score)
            norm_diff = self._signed_min_max_normalize(diff_data)
            
            # 符号一致性筛选
            mask = ((norm_act > 0) & (norm_diff > 0)) | ((norm_act < 0) & (norm_diff < 0))
            
            # 综合得分计算
            scores = torch.zeros_like(norm_diff)
            scores[mask] = norm_diff[mask]
            
            # 硬阈值筛选
            trim = self.top_freq  # 使用top_freq作为trim比例
            threshold_fre = torch.sort(torch.abs(scores), descending=True, stable=True).values[int(trim * len(scores))]
            prune_mask = torch.abs(scores) > threshold_fre
            
            # 激活值硬阈值筛选
            act_threshold = torch.sort(torch.abs(feature_score), descending=True, stable=True).values[int(trim * len(feature_score))]
            act_top_mask = torch.abs(feature_score) > act_threshold
            
            # 结合掩码（硬结合）
            combined_mask = prune_mask & act_top_mask
            
        elif self.filter_type == "freq":
            # 使用fre_trim逻辑
            diff_data = pos_feature_freq - neg_feature_freq
            norm_diff = self._signed_min_max_normalize(diff_data)
            
            trim = self.top_freq
            threshold_fre = torch.sort(torch.abs(norm_diff), descending=True, stable=True).values[int(trim * len(norm_diff))]
            combined_mask = torch.abs(norm_diff) >= threshold_fre
            
        elif self.filter_type == "value":
            # 使用act_trim逻辑
            trim = self.top_value
            act_threshold = torch.sort(torch.abs(feature_score), descending=True, stable=True).values[int(trim * len(feature_score))]
            combined_mask = torch.abs(feature_score) >= act_threshold
            
        else:
            raise ValueError(f"不支持的filter_type: {self.filter_type}")
        
        # 6. 使用动态重要特征维度筛选全局SAE特征差值
        global_sae_features_masked = self.global_sae_features.clone()
        global_sae_features_masked[~combined_mask] = 0
        
        # 7. 与W_dec相乘计算得到steer_vector
        steer_vector = global_sae_features_masked @ self.sae_decoder_weights.cpu()  # [d_model]
        
        return steer_vector


    def _signed_min_max_normalize(self, tensor):
        """Min-Max归一化，保留正负符号（参考analyse文件）"""
        abs_tensor = tensor.abs()
        min_val = abs_tensor.min()
        max_val = abs_tensor.max()
        normalized = (abs_tensor - min_val) / (max_val - min_val + 1e-8)
        return tensor.sign() * normalized  # 恢复正负符号

    def create_hook_fn(self):
        """创建并返回实际的钩子函数"""
        def hook_fn(activations, hook):
            # 检查是否已经计算过steer_vector
            if not self.is_steer_vector_computed:
                # 获取查询后的隐藏状态（最后一个token的激活值）
                query_activation = activations[0, -1, :]
                # 计算steer_vector
                self.steer_vector = self._calculate_sae_steer_vector(query_activation).to(activations.device)
                self.is_steer_vector_computed = True
            
            # 根据steer_pos选项应用steer_vector
            if self.steer_pos == "all":
                # 对所有位置的隐藏状态应用steer_vector
                steer_vector_expanded = self.steer_vector.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
                activations = activations + steer_vector_expanded
            elif self.steer_pos == "current":
                # 仅对当前token（最后一个token）的隐藏状态应用steer_vector
                steer_vector_expanded = self.steer_vector.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
                activations[:, -1:, :] = activations[:, -1:, :] + steer_vector_expanded
            else:
                raise ValueError(f"不支持的steer_pos: {self.steer_pos}")
            
            return activations
        
        return hook_fn

@torch.no_grad()
def steered_generate(
    model: HookedTransformer,
    steering_hook: SAESteeringHookQueryLevel,
    prompt: str,
    layer_to_steer: int,
    max_new_tokens: int = 50,
    test_dataset: str = "safeedit",
):
    """使用SAE特征引导钩子来生成文本"""
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
        
def main(args):
    print("加载模型...")
    # 使用transformer_lens加载模型
    model_path = model_alias_to_model_name[args.model_alias]
    model = HookedTransformer.from_pretrained(model_path, n_devices=1, torch_dtype=torch.bfloat16)
    model.eval()
    model.reset_hooks()
    
    # 获取路径配置
    paths_config = get_paths_config(
        args.model_alias, args.dataset, args.layer, args.shared_size, args.search_mode,
        args.top_freq, args.top_value, args.filter_type
    )
    
    # 初始化SAE特征引导钩子
    steering_hook = SAESteeringHookQueryLevel(
        paths_config=paths_config,
        top_k=args.top_k,
        multiplier=args.multiplier,
        norm_magnitude=args.norm_magnitude,
        caa_vectors_dir="results/caa_vectors",
        top_freq=args.top_freq,
        top_value=args.top_value,
        filter_type=args.filter_type,
        sae_layer=args.sae_layer,
        sae_model_path=args.sae_model_path,
        search_mode=args.search_mode,
        static_weight=args.static_weight,
        static_steer_vector_path=args.static_steer_vector_path,
        steer_pos=args.steer_pos,
        distance_metric=args.distance_metric,
        use_softmax_weight=args.use_softmax_weight
    )
    
    # 加载测试数据
    if args.test_dataset == "safeedit":
        queries = load_safeedit_test_data(model_alias=args.model_alias, apply_chat_format=True)
    elif args.test_dataset == "StrongREJECT":
        queries = load_strongreject_test_data(model_alias=args.model_alias, apply_chat_format=True)
    elif args.test_dataset == "BeaverTails":
        queries = load_beavertails_test_data(model_alias=args.model_alias, apply_chat_format=True)
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
        steered_output = steered_generate(
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
    
    # 在路径中添加top_k、norm_magnitude、search_mode和选择策略信息
    norm_info = "norm" if args.norm_magnitude else "no_norm"
    sae_filter_info = f"freq{int(args.top_freq*100)}p_value{int(args.top_value*100)}p_{args.filter_type}"
    search_mode_info = f"search_{args.search_mode}"
    
    # 如果是混合模式，添加权重信息
    if args.search_mode == "hybrid":
        static_weight_info = f"static{int(args.static_weight*100)}p"
        search_mode_info = f"{search_mode_info}_{static_weight_info}"
    
    # 添加steer_pos、distance_metric和softmax_weight信息
    steer_pos_info = f"steer_{args.steer_pos}"
    distance_info = f"dist_{args.distance_metric}"
    softmax_info = f"softmax_{args.use_softmax_weight}"
    
    rst_dir = f"results/generations/{args.model_alias}/{args.test_dataset}/flexible_sae_querylevel/{args.dataset}_{args.layer}"   
    os.makedirs(rst_dir, exist_ok=True)
    
    output_filename = f"topk_{args.top_k}_{norm_info}_{sae_filter_info}_{search_mode_info}_{steer_pos_info}_{distance_info}_{softmax_info}_{str(args.multiplier)}.json"
    with open(os.path.join(rst_dir, output_filename), "w") as f:
        json.dump(final_data, f, indent=2)
    
    print(f"结果已保存到: {os.path.join(rst_dir, output_filename)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用SAE特征进行安全引导生成 (Query Level)')
    parser.add_argument('--model_alias', type=str, default="gemma-2-9b-it", help='模型别名')
    parser.add_argument('--dataset', type=str, default="safeedit", help='Dataset name used for caching')
    parser.add_argument('--shared_size', type=int, default=4050)
    parser.add_argument('--layer', type=int, default=20, help='要进行干预的层')
    parser.add_argument('--sae_layer', type=int, default=20, help='SAE特征对应的层号')
    parser.add_argument('--top_k', type=int, default=10, help='FAISS检索的近邻数量')
    parser.add_argument('--multiplier', type=float, default=1.0, help='引导向量的强度系数')
    parser.add_argument('--test_dataset', type=str, default="safeedit", help='测试数据集')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='最大生成token数')
    parser.add_argument('--norm_magnitude', action='store_true', default=True, help='是否将steer_vector缩放到与CAA向量相同的范数')
    parser.add_argument('--top_freq', type=float, default=0.35, help='按频率差异绝对值排序的前N%特征')
    parser.add_argument('--top_value', type=float, default=0.35, help='按激活值差异绝对值排序的前N%特征')
    parser.add_argument('--filter_type', type=str, default="mix", choices=["freq", "value", "mix"], help='特征筛选方式: freq(仅频率), value(仅激活值), mix(交集)')
    parser.add_argument('--sae_model_path', type=str, default="/home/wangxin/models/gemma-scope-9b-it-res/layer_20/width_16k/average_l0_91", help='SAE模型路径')
    parser.add_argument('--search_mode', type=str, default="activation", choices=["activation", "hybrid", "fixed_features"], help='搜索模式: activation(基于激活值检索), hybrid(混合模式), fixed_features(固定特征模式)')
    parser.add_argument('--static_weight', type=float, default=0.0, help='混合模式中静态向量的权重 (0.0-1.0)')
    parser.add_argument('--static_steer_vector_path', type=str, default=None, help='静态预计算的steer_vector路径')
    parser.add_argument('--steer_pos', type=str, default="all", choices=["all", "current"], help='引导位置: all(修改所有token激活值), current(仅修改当前token激活值)')
    parser.add_argument('--distance_metric', type=str, default="cosine", choices=["euclidean", "cosine"], help='距离度量: euclidean(欧几里得距离), cosine(余弦相似度)')
    parser.add_argument('--use_softmax_weight', action='store_true', default=False, help='是否使用softmax加权 (仅适用于activation模式): True(使用softmax加权), False(使用简单平均)')
    args = parser.parse_args()
    main(args)
