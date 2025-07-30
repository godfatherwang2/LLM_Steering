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
        # 重要SAE特征索引路径（keySAE模式）- 包含选择策略和阈值
        "safe_key_sae_index_path": os.path.join(base_cache_dir, f"{model_alias}_{dataset_name}", f"label_safe_shard_size_{shared_size}", f"faiss_index_key_sae_layer_{layer}_{selection_info}.index"),
        "unsafe_key_sae_index_path": os.path.join(base_cache_dir, f"{model_alias}_{dataset_name}", f"label_unsafe_shard_size_{shared_size}", f"faiss_index_key_sae_layer_{layer}_{selection_info}.index"),
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


class SAESteeringHook:
    """
    基于SAE特征的安全引导钩子管理器
    """
    def __init__(self, 
                 paths_config: Dict,
                 top_k: int = 10, 
                 multiplier: float = 1.0, 
                 steer_token_limit: int = 5, 
                 norm_magnitude: bool = False, 
                 caa_vectors_dir: str = "results/caa_vectors", 
                 top_freq: float = 0.1,
                 top_value: float = 0.1,
                 filter_type: str = "mix",
                 sae_layer: int = None,
                 selective_steering: bool = False,
                 sae_model_path: str = None,
                 search_mode: str = "activation"):
        """
        初始化SAE特征引导钩子

        Args:
            paths_config: 路径配置字典
            top_k: FAISS检索的最近邻数量
            multiplier: 引导向量的强度系数
            steer_token_limit: 只在前k个生成的token上进行引导
            norm_magnitude: 是否将steer_vector缩放到与CAA向量相同的范数
            caa_vectors_dir: CAA向量目录
            top_freq: 使用前N%的频率差异特征
            top_value: 使用前N%的激活值差异特征
            filter_type: 特征筛选类型 ("freq", "value", "mix")
            sae_layer: SAE特征对应的层号
            selective_steering: 是否只对关键维度SAE特征相较于负样本更近的激活值进行steer
            sae_model_path: SAE模型路径
            search_mode: 搜索模式 ("activation", "keySAE")
        """
        print("--- 初始化SAE特征引导钩子 ---")
        self.paths_config = paths_config
        self.top_k = top_k
        self.multiplier = multiplier
        self.steer_token_limit = steer_token_limit
        self.norm_magnitude = norm_magnitude
        self.sae_layer = sae_layer or paths_config["layer"]
        self.selective_steering = selective_steering
        self.sae_model_path = sae_model_path
        self.search_mode = search_mode
        
        # 如果启用norm_magnitude，加载CAA向量
        if self.norm_magnitude:
            if caa_vectors_dir and paths_config["model_alias"] and paths_config["dataset_name"] and paths_config["layer"] is not None:
                self.caa_vector = self._load_caa_vector(caa_vectors_dir, paths_config["model_alias"], paths_config["dataset_name"], paths_config["layer"])
                print(f"加载CAA向量用于范数归一化，范数: {torch.norm(self.caa_vector, p=2):.6f}")
            else:
                raise ValueError("启用norm_magnitude时需要提供caa_vectors_dir, model_alias, dataset_name, layer参数")
        else:
            self.caa_vector = None
        
        # 根据search_mode加载不同的索引
        if self.search_mode == "activation":
            # 加载激活值FAISS索引
            print(f"加载安全激活值FAISS索引: {paths_config['safe_acts_index_path']}")
            self.safe_acts_index = faiss.read_index(paths_config['safe_acts_index_path'])
            print(f"加载不安全激活值FAISS索引: {paths_config['unsafe_acts_index_path']}")
            self.unsafe_acts_index = faiss.read_index(paths_config['unsafe_acts_index_path'])
            
            
        # 加载激活值数据
        print("加载激活值缓存数据...")
        safe_acts_metadata_file = os.path.join(paths_config["safe_cache_dir"], "metadata.json")
        unsafe_acts_metadata_file = os.path.join(paths_config["unsafe_cache_dir"], "metadata.json")
        
        with open(safe_acts_metadata_file, 'r') as f:
            safe_acts_metadata = json.load(f)
        with open(unsafe_acts_metadata_file, 'r') as f:
            unsafe_acts_metadata = json.load(f)
            
        n_layers, total_tokens_safe, d_model = safe_acts_metadata["activations_original_shape"]
        n_layers, total_tokens_unsafe, d_model = unsafe_acts_metadata["activations_original_shape"]
        
        safe_acts_file = os.path.join(paths_config["safe_cache_dir"], f"acts_{self.sae_layer}.dat")
        unsafe_acts_file = os.path.join(paths_config["unsafe_cache_dir"], f"acts_{self.sae_layer}.dat")
        
        self.safe_acts_array = np.ascontiguousarray(np.memmap(safe_acts_file, dtype='float32', mode='r', shape=(total_tokens_safe, d_model)))
        self.unsafe_acts_array = np.ascontiguousarray(np.memmap(unsafe_acts_file, dtype='float32', mode='r', shape=(total_tokens_unsafe, d_model)))
        
        print(f"安全激活值形状: {self.safe_acts_array.shape}")
        print(f"不安全激活值形状: {self.unsafe_acts_array.shape}")
        
        # 加载SAE模型
        self.sae_model = self._load_sae_model()
        
        # 加载SAE特征数据
        print("加载SAE特征缓存数据...")
        safe_sae_metadata_file = os.path.join(paths_config["safe_cache_dir"], "metadata.json")
        unsafe_sae_metadata_file = os.path.join(paths_config["unsafe_cache_dir"], "metadata.json")
        
        with open(safe_sae_metadata_file, 'r') as f:
            safe_sae_metadata = json.load(f)
        with open(unsafe_sae_metadata_file, 'r') as f:
            unsafe_sae_metadata = json.load(f)
            
        safe_sae_file = os.path.join(paths_config["safe_cache_dir"], f"sae_{self.sae_layer}.dat")
        unsafe_sae_file = os.path.join(paths_config["unsafe_cache_dir"], f"sae_{self.sae_layer}.dat")
        
        self.safe_sae_features = np.ascontiguousarray(np.memmap(safe_sae_file, dtype='float32', mode='r', 
                                                               shape=(safe_sae_metadata["sae_features_shape"][1], safe_sae_metadata["sae_features_dim"])))
        self.unsafe_sae_features = np.ascontiguousarray(np.memmap(unsafe_sae_file, dtype='float32', mode='r', 
                                                                 shape=(unsafe_sae_metadata["sae_features_shape"][1], unsafe_sae_metadata["sae_features_dim"])))
        
        print(f"安全SAE特征形状: {self.safe_sae_features.shape}")
        print(f"不安全SAE特征形状: {self.unsafe_sae_features.shape}")
        
        # 加载重要SAE特征维度
        self.important_sae_features = self._load_important_sae_features(
            paths_config["model_alias"], paths_config["dataset_name"], top_freq, top_value, filter_type
        )
        # 如果是keySAE模式，需要创建重要SAE特征的索引
        if self.search_mode == "keySAE":
            print("keySAE模式：检查并创建重要SAE特征索引...")
            safe_key_sae_index_path = paths_config["safe_key_sae_index_path"]
            unsafe_key_sae_index_path = paths_config["unsafe_key_sae_index_path"]
            # 检查索引文件是否存在
            if not os.path.exists(safe_key_sae_index_path) or not os.path.exists(unsafe_key_sae_index_path):
                print("重要SAE特征索引不存在，正在创建...")
                self._create_key_sae_indexes()
                print("重要SAE特征索引创建完成")
            else:
                print("重要SAE特征索引已存在，跳过创建")
            # 加载重要SAE特征FAISS索引
            print(f"加载安全重要SAE特征FAISS索引: {paths_config['safe_key_sae_index_path']}")
            self.safe_key_sae_index = faiss.read_index(paths_config['safe_key_sae_index_path'])
            print(f"加载不安全重要SAE特征FAISS索引: {paths_config['unsafe_key_sae_index_path']}")
            self.unsafe_key_sae_index = faiss.read_index(paths_config['unsafe_key_sae_index_path'])
        # 加载SAE Decoder权重
        self.sae_decoder_weights = self._load_sae_decoder_weights()
        
        self.generation_step = 0  # 用于跟踪生成token的计数器

    def _load_caa_vector(self, caa_vectors_dir: str, model_alias: str, dataset_name: str, layer: int) -> torch.Tensor:
        """加载CAA向量"""
        filename = f"{layer}.pt"
        filepath = os.path.join(caa_vectors_dir,f"{model_alias}_{dataset_name}",filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CAA向量文件不存在: {filepath}")
        
        caa_vector = torch.load(filepath)
        print(f"加载CAA向量: {filepath}, 形状: {caa_vector.shape}")
        return caa_vector

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

    def _create_key_sae_indexes(self):
        """为keySAE模式创建重要SAE特征的索引"""
        print("为keySAE模式创建重要SAE特征索引...")
        
        print(f"重要SAE特征维度数量: {len(self.important_sae_features)}")
        
        # 2. 加载所有SAE激活值，只取重要维度
        safe_key_sae_features = self.safe_sae_features[:, self.important_sae_features]  # [total_tokens, n_important_features]
        unsafe_key_sae_features = self.unsafe_sae_features[:, self.important_sae_features]  # [total_tokens, n_important_features]
        
        print(f"安全重要SAE特征形状: {safe_key_sae_features.shape}")
        print(f"不安全重要SAE特征形状: {unsafe_key_sae_features.shape}")
        
        # 3. 构建FAISS索引
        n_important_features = len(self.important_sae_features)
        
        # 根据FAISS官方推荐：nlist = sqrt(n_vectors) 或 4 * sqrt(n_vectors)
        n_vectors = safe_key_sae_features.shape[0]
        nlist = min(4096, max(1024, int(4 * np.sqrt(n_vectors))))
        
        # 计算合适的m值，确保维度能被整除
        m_candidates = [8, 16, 32, 64, 128, 256]
        valid_m = []
        for m in m_candidates:
            if n_important_features % m == 0:
                valid_m.append(m)
        
        # 选择最合适的m值（优先选择较小的值以保持精度）
        if valid_m:
            m = min(valid_m, key=lambda x: abs(x - n_important_features // 64))
        else:
            # 如果仍然没有合适的m，使用最小的可能值
            m = 8
        
        bits = 8
        
        # 创建安全样本的索引
        safe_quantizer = faiss.IndexFlatL2(n_important_features)
        safe_index = faiss.IndexIVFPQ(safe_quantizer, n_important_features, nlist, m, bits)
        
        # 训练索引
        num_train_samples = min(safe_key_sae_features.shape[0], nlist * 100)
        train_indices = np.random.choice(safe_key_sae_features.shape[0], num_train_samples, replace=False)
        safe_index.train(safe_key_sae_features[train_indices])
        
        # 添加所有向量
        batch_size_add = 50000
        for i in range(0, safe_key_sae_features.shape[0], batch_size_add):
            safe_index.add(safe_key_sae_features[i:i+batch_size_add])
        safe_index.make_direct_map()
        
        # 创建不安全样本的索引
        unsafe_quantizer = faiss.IndexFlatL2(n_important_features)
        unsafe_index = faiss.IndexIVFPQ(unsafe_quantizer, n_important_features, nlist, m, bits)
        
        # 训练索引
        num_train_samples = min(unsafe_key_sae_features.shape[0], nlist * 100)
        train_indices = np.random.choice(unsafe_key_sae_features.shape[0], num_train_samples, replace=False)
        unsafe_index.train(unsafe_key_sae_features[train_indices])
        
        # 添加所有向量
        for i in range(0, unsafe_key_sae_features.shape[0], batch_size_add):
            unsafe_index.add(unsafe_key_sae_features[i:i+batch_size_add])
        unsafe_index.make_direct_map()
        
        # 4. 保存索引到原目录
        safe_index_path = self.paths_config["safe_key_sae_index_path"]
        unsafe_index_path = self.paths_config["unsafe_key_sae_index_path"]
        
        faiss.write_index(safe_index, safe_index_path)
        faiss.write_index(unsafe_index, unsafe_index_path)
        
        print(f"重要SAE特征索引已保存:")
        print(f"  安全索引: {safe_index_path}")
        print(f"  不安全索引: {unsafe_index_path}")

    def reset_state(self):
        """在每次新的生成任务开始前重置状态"""
        self.generation_step = 0

    def _calculate_sae_steer_vector(self, activation: torch.Tensor) -> torch.Tensor:
        """计算基于SAE特征的引导向量"""
        if self.search_mode == "activation":
            return self._calculate_sae_steer_vector_activation_mode(activation)
        elif self.search_mode == "keySAE":
            return self._calculate_sae_steer_vector_key_sae_mode(activation)
        else:
            raise ValueError(f"不支持的search_mode: {self.search_mode}")

    def _calculate_sae_steer_vector_activation_mode(self, activation: torch.Tensor) -> torch.Tensor:
        """计算基于SAE特征的引导向量 - activation模式"""
        # 1. 准备查询向量（使用激活值进行FAISS检索）
        query_vector = activation.detach().cpu().float().numpy().reshape(1, -1)
        
        # 2. 使用激活值FAISS索引搜索最近邻
        safe_dists, safe_indices = self.safe_acts_index.search(query_vector, self.top_k)
        unsafe_dists, unsafe_indices = self.unsafe_acts_index.search(query_vector, self.top_k)
        
        # 3. 加载对应的激活值，重新计算距离
        safe_acts = self.safe_acts_array[safe_indices[0]]  # [top_k, d_model]
        unsafe_acts = self.unsafe_acts_array[unsafe_indices[0]]  # [top_k, d_model]
        
        # 重新计算距离
        safe_dists_recomputed = np.linalg.norm(safe_acts - query_vector, axis=1)  # [top_k]
        unsafe_dists_recomputed = np.linalg.norm(unsafe_acts - query_vector, axis=1)  # [top_k]
        
        # 4. 将距离归一化到0-1之间，然后计算softmax权重
        safe_dists_normalized = (safe_dists_recomputed - safe_dists_recomputed.min()) / (safe_dists_recomputed.max() - safe_dists_recomputed.min() + 1e-8)
        unsafe_dists_normalized = (unsafe_dists_recomputed - unsafe_dists_recomputed.min()) / (unsafe_dists_recomputed.max() - unsafe_dists_recomputed.min() + 1e-8)
        
        safe_weights = torch.softmax(torch.tensor(-safe_dists_normalized, dtype=torch.float32), dim=0)  # [top_k]
        unsafe_weights = torch.softmax(torch.tensor(-unsafe_dists_normalized, dtype=torch.float32), dim=0)  # [top_k]
        
        # 5. 加载这些索引对应的SAE向量
        safe_sae_features = self.safe_sae_features[safe_indices[0]]  # [top_k, sae_features_dim]
        unsafe_sae_features = self.unsafe_sae_features[unsafe_indices[0]]  # [top_k, sae_features_dim]
        
        # 6. 对安全/不安全SAE向量进行加权平均
        safe_sae_weighted = torch.sum(safe_weights.unsqueeze(1) * torch.tensor(safe_sae_features, dtype=torch.float32), dim=0)  # [sae_features_dim]
        unsafe_sae_weighted = torch.sum(unsafe_weights.unsqueeze(1) * torch.tensor(unsafe_sae_features, dtype=torch.float32), dim=0)  # [sae_features_dim]
        
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

    def _calculate_sae_steer_vector_key_sae_mode(self, activation: torch.Tensor) -> torch.Tensor:
        """计算基于SAE特征的引导向量 - keySAE模式"""
        # 1. 通过SAE模型获取当前激活值的SAE中间层特征
        with torch.no_grad():
            # 使用SAE的hook系统获取中间层激活值
            sae_cache = {}
            
            def sae_fwd_hook(act, hook):
                sae_cache[hook.name] = act.detach()
            
            # 重置并添加hook
            self.sae_model.reset_hooks()
            filter_not_input = lambda name: "_input" not in name
            self.sae_model.add_hook(filter_not_input, sae_fwd_hook, "fwd")
            
            # 前向传播获取SAE特征
            _ = self.sae_model(activation.unsqueeze(0))  # [1, sae_features_dim]
            
            # 获取SAE隐藏层中间特征
            if "hook_sae_acts_post" not in sae_cache:
                print("警告: 未找到SAE隐藏层特征，使用输出特征作为备选")
                current_sae_features = sae_cache["hook_sae_output"].squeeze(0)
            else:
                current_sae_features = sae_cache["hook_sae_acts_post"].squeeze(0)  # [sae_features_dim]
            
            # 清理hook
            self.sae_model.reset_hooks()
        
        # 2. 取中间特征的重要维度
        current_key_sae_features = current_sae_features[self.important_sae_features]  # [n_important_features]
        
        # 3. 准备查询向量（使用重要SAE特征进行FAISS检索）
        query_vector = current_key_sae_features.detach().cpu().float().numpy().reshape(1, -1)
        
        # 4. 使用重要SAE特征FAISS索引搜索最近邻
        safe_dists, safe_indices = self.safe_key_sae_index.search(query_vector, self.top_k)
        unsafe_dists, unsafe_indices = self.unsafe_key_sae_index.search(query_vector, self.top_k)
        
        # 5. 加载对应的完整SAE向量
        safe_sae_features = self.safe_sae_features[safe_indices[0]]  # [top_k, sae_features_dim]
        unsafe_sae_features = self.unsafe_sae_features[unsafe_indices[0]]  # [top_k, sae_features_dim]
        
        # 6. 重新计算距离（基于重要维度）
        safe_key_sae_features = safe_sae_features[:, self.important_sae_features]  # [top_k, n_important_features]
        unsafe_key_sae_features = unsafe_sae_features[:, self.important_sae_features]  # [top_k, n_important_features]
        
        safe_dists_recomputed = np.linalg.norm(safe_key_sae_features - query_vector, axis=1)  # [top_k]
        unsafe_dists_recomputed = np.linalg.norm(unsafe_key_sae_features - query_vector, axis=1)  # [top_k]
        
        # 7. 将距离归一化到0-1之间，然后计算softmax权重
        safe_dists_normalized = (safe_dists_recomputed - safe_dists_recomputed.min()) / (safe_dists_recomputed.max() - safe_dists_recomputed.min() + 1e-8)
        unsafe_dists_normalized = (unsafe_dists_recomputed - unsafe_dists_recomputed.min()) / (unsafe_dists_recomputed.max() - unsafe_dists_recomputed.min() + 1e-8)
        
        safe_weights = torch.softmax(torch.tensor(-safe_dists_normalized, dtype=torch.float32), dim=0)  # [top_k]
        unsafe_weights = torch.softmax(torch.tensor(-unsafe_dists_normalized, dtype=torch.float32), dim=0)  # [top_k]
        
        # 8. 对安全/不安全SAE向量进行加权平均
        safe_sae_weighted = torch.sum(safe_weights.unsqueeze(1) * torch.tensor(safe_sae_features, dtype=torch.float32), dim=0)  # [sae_features_dim]
        unsafe_sae_weighted = torch.sum(unsafe_weights.unsqueeze(1) * torch.tensor(unsafe_sae_features, dtype=torch.float32), dim=0)  # [sae_features_dim]
        
        # 9. 相减得到SAE特征差异
        sae_diff = safe_sae_weighted - unsafe_sae_weighted  # [sae_features_dim]
        
        # 10. Mask掉非重要维度
        if self.important_sae_features is not None:
            mask = torch.zeros_like(sae_diff)
            mask[self.important_sae_features] = 1.0
            sae_diff = sae_diff * mask
        
        # 11. 与W_dec相乘计算得到最终steer_vector
        steer_vector = sae_diff @ self.sae_decoder_weights.cpu()  # [d_model]
        
        # 12. 应用强度系数
        steer_vector = steer_vector * self.multiplier
        
        # 13. 如果启用范数归一化，缩放到与CAA向量相同的范数
        if self.norm_magnitude and self.caa_vector is not None:
            caa_norm = torch.norm(self.caa_vector, p=2)
            steer_norm = torch.norm(steer_vector, p=2)
            if steer_norm > 0:
                steer_vector = steer_vector * (caa_norm / steer_norm)
        
        return steer_vector

    def create_hook_fn(self):
        """创建并返回实际的钩子函数"""
        def hook_fn(activations, hook):
            # 检查是否还在引导限制内
            if self.generation_step < self.steer_token_limit:
                # 获取最后一个token的激活值
                current_activation = activations[0, -1, :]
                steer_vector = self._calculate_sae_steer_vector(current_activation).to(activations.device)
                # 修改最后一个token的隐藏状态
                activations[0, -1, :] = current_activation + steer_vector
                self.generation_step += 1
            
            return activations
        
        return hook_fn


@torch.no_grad()
def steered_generate(
    model: HookedTransformer,
    steering_hook: SAESteeringHook,
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
        # 使用transformer_lens的generate方法
        outputs = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_type="tensor",
            verbose=False
        )
        
        # 手动处理停止条件（如果需要）
        if test_dataset == "GSM":
            # 解码并检查是否包含停止词
            decoded_output = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
            if "Question:" in decoded_output:
                decoded_output = decoded_output.split("Question:")[0]
            return decoded_output
        else:
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
    steering_hook = SAESteeringHook(
        paths_config=paths_config,
        top_k=args.top_k,
        multiplier=args.multiplier,
        steer_token_limit=args.max_new_tokens,
        norm_magnitude=args.norm_magnitude,
        caa_vectors_dir="results/caa_vectors",
        top_freq=args.top_freq,
        top_value=args.top_value,
        filter_type=args.filter_type,
        sae_layer=args.sae_layer,
        selective_steering=args.selective_steering,
        sae_model_path=args.sae_model_path,
        search_mode=args.search_mode
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
    selective_info = "selective" if args.selective_steering else "all"
    search_mode_info = f"search_{args.search_mode}"
    selection_strategy_info = f"{args.filter_type}"
    rst_dir = f"results/generations/{args.model_alias}/{args.test_dataset}/flexible_sae/{args.dataset}_{args.layer}"   
    os.makedirs(rst_dir, exist_ok=True)
    
    output_filename = f"topk_{args.top_k}_{norm_info}_{sae_filter_info}_{selective_info}_{search_mode_info}_{selection_strategy_info}.json"
    with open(os.path.join(rst_dir, output_filename), "w") as f:
        json.dump(final_data, f, indent=2)
    
    print(f"结果已保存到: {os.path.join(rst_dir, output_filename)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用SAE特征进行安全引导生成')
    parser.add_argument('--model_alias', type=str, default="gemma-2-9b-it", help='模型别名')
    parser.add_argument('--dataset', type=str, default="safeedit", help='Dataset name used for caching')
    parser.add_argument('--shared_size', type=int, default=4050)
    parser.add_argument('--layer', type=int, default=20, help='要进行干预的层')
    parser.add_argument('--sae_layer', type=int, default=20, help='SAE特征对应的层号')
    parser.add_argument('--top_k', type=int, default=10, help='FAISS检索的近邻数量')
    parser.add_argument('--multiplier', type=float, default=1.0, help='引导向量的强度系数')
    parser.add_argument('--test_dataset', type=str, default="safeedit", help='测试数据集')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='最大生成token数')
    parser.add_argument('--norm_magnitude', action='store_true', default=False, help='是否将steer_vector缩放到与CAA向量相同的范数')
    parser.add_argument('--top_freq', type=float, default=0.35, help='按频率差异绝对值排序的前N%特征')
    parser.add_argument('--top_value', type=float, default=0.35, help='按激活值差异绝对值排序的前N%特征')
    parser.add_argument('--filter_type', type=str, default="mix", choices=["freq", "value", "mix"], help='特征筛选方式: freq(仅频率), value(仅激活值), mix(交集)')
    parser.add_argument('--selective_steering', action='store_true', default=False, help='只对关键维度SAE特征相较于负样本更近的激活值进行steer')
    parser.add_argument('--sae_model_path', type=str, default="/home/wangxin/models/gemma-scope-9b-it-res/layer_20/width_16k/average_l0_91", help='SAE模型路径')
    parser.add_argument('--search_mode', type=str, default="keySAE", choices=["activation", "keySAE"], help='搜索模式: activation(基于激活值检索), keySAE(基于重要SAE特征检索)')
    
    args = parser.parse_args()
    
    main(args)
    