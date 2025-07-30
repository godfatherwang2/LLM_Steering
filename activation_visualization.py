import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append('./')
import torch
import numpy as np
import pandas as pd
import json
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# 导入必要的模块
from transformer_lens import HookedTransformer
from utils.utils import model_alias_to_model_name
from dataset.load_data import load_safeedit_test_data
from sae_utils import load_gemma_2_sae
from evaluate.eval_SafeEdit import predict
from transformers import RobertaForSequenceClassification, RobertaTokenizer


class SteerVectorUtils:
    """Steer Vector分析工具类"""
    
    @staticmethod
    def signed_min_max_normalize(tensor):
        """Min-Max归一化，保留正负符号"""
        abs_tensor = tensor.abs()
        min_val = abs_tensor.min()
        max_val = abs_tensor.max()
        normalized = (abs_tensor - min_val) / (max_val - min_val + 1e-8)
        return tensor.sign() * normalized



    @staticmethod
    def calculate_sparsity(tensor: torch.Tensor) -> Dict:
        """计算张量的稀疏性"""
        total_elements = tensor.numel()
        zero_elements = (tensor == 0).sum().item()
        near_zero_elements = (torch.abs(tensor) < 1e-6).sum().item()
        
        return {
            "zero_ratio": zero_elements / total_elements,
            "near_zero_ratio": near_zero_elements / total_elements,
            "total_elements": total_elements,
            "zero_elements": zero_elements,
            "near_zero_elements": near_zero_elements
        }

    @staticmethod
    def predict_safety_confidence(model, tokenizer, sequences: List[str], batch_size: int = 32) -> List[float]:
        """预测安全置信度"""
        confidences = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                batch_confidences = probs[:, 0].cpu().numpy().tolist()
            
            confidences.extend(batch_confidences)
        
        return confidences


class SteerVectorAnalyzer:
    """Steer Vector性能分析器 - 核心分析类"""
    
    def __init__(self, 
                 static_result_path: str,
                 dynamic_result_path: str,
                 safety_classifier_dir: str,
                 model_alias: str = "gemma-2-9b-it",
                 output_dir: str = "steer_analysis_results"):
        """初始化分析器"""
        self.static_result_path = static_result_path
        self.dynamic_result_path = dynamic_result_path
        self.safety_classifier_dir = safety_classifier_dir
        self.model_alias = model_alias
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模型和数据
        self._load_models()
        self._load_query_activation_data()
    
    def _load_models(self):
        """加载所有必要的模型"""
        print("加载安全分类器...")
        self.safety_classifier_model = RobertaForSequenceClassification.from_pretrained(
            self.safety_classifier_dir).to('cuda')
        self.safety_classifier_tokenizer = RobertaTokenizer.from_pretrained(self.safety_classifier_dir)
        
        print("加载主模型...")
        model_path = model_alias_to_model_name[self.model_alias]
        self.model = HookedTransformer.from_pretrained(model_path, n_devices=1, torch_dtype=torch.bfloat16)
        self.model.eval()
        self.model.reset_hooks()
        
        print("加载SAE模型...")
        sae_path = "/home/wangxin/models/gemma-scope-9b-it-res/layer_20/width_16k/average_l0_91"
        self.sae_model, sparsity = load_gemma_2_sae(sae_path, device="cuda")
        self.sae_model.eval()
        
        # 加载CAA向量用于范数归一化
        self.caa_vector = self._load_caa_vector()
        
    def _load_caa_vector(self) -> torch.Tensor:
        """加载CAA向量"""
        caa_path = f"results/caa_vectors/{self.model_alias}_safeedit/20.pt"
        if not os.path.exists(caa_path):
            raise FileNotFoundError(f"CAA向量文件不存在: {caa_path}")
        
        caa_vector = torch.load(caa_path)
        print(f"加载CAA向量: {caa_path}, 形状: {caa_vector.shape}")
        return caa_vector
    
    def _load_query_activation_data(self):
        """加载查询激活数据"""
        print("加载查询激活数据...")
        
        # 加载安全样本的查询激活
        safe_query_acts_path = f"dataset/cached_activations_sae/{self.model_alias}_safeedit/label_safe_shard_size_4050/query_acts.dat"
        safe_avg_sae_path = f"dataset/cached_activations_sae/{self.model_alias}_safeedit/label_safe_shard_size_4050/avg_sae.dat"
        
        # 加载非安全样本的查询激活
        unsafe_query_acts_path = f"dataset/cached_activations_sae/{self.model_alias}_safeedit/label_unsafe_shard_size_4050/query_acts.dat"
        unsafe_avg_sae_path = f"dataset/cached_activations_sae/{self.model_alias}_safeedit/label_unsafe_shard_size_4050/avg_sae.dat"
        
        # 加载样本索引
        safe_indices_path = f"dataset/cached_activations_sae/{self.model_alias}_safeedit/label_safe_shard_size_4050/sample_indices.json"
        unsafe_indices_path = f"dataset/cached_activations_sae/{self.model_alias}_safeedit/label_unsafe_shard_size_4050/sample_indices.json"
        
        # 检查文件是否存在
        for path in [safe_query_acts_path, safe_avg_sae_path, unsafe_query_acts_path, unsafe_avg_sae_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"查询激活文件不存在: {path}")
        
        # 使用numpy memmap加载数据
        self.safe_query_acts_array = np.memmap(safe_query_acts_path, dtype='float32', mode='r')
        self.safe_avg_sae_features = np.memmap(safe_avg_sae_path, dtype='float32', mode='r')
        self.unsafe_query_acts_array = np.memmap(unsafe_query_acts_path, dtype='float32', mode='r')
        self.unsafe_avg_sae_features = np.memmap(unsafe_avg_sae_path, dtype='float32', mode='r')
        
        # 加载样本索引
        with open(safe_indices_path, 'r') as f:
            self.safe_sample_indices = json.load(f)
        with open(unsafe_indices_path, 'r') as f:
            self.unsafe_sample_indices = json.load(f)
        
        # 重塑数组
        n_safe_samples = len(self.safe_sample_indices)
        n_unsafe_samples = len(self.unsafe_sample_indices)
        
        # 加载全局固定的重要SAE特征维度（用于fixed_features模式）
        self.important_sae_features = self._load_important_sae_features()
        
        self.safe_query_acts_array = self.safe_query_acts_array.reshape(n_safe_samples, 1, -1)
        self.safe_avg_sae_features = self.safe_avg_sae_features.reshape(n_safe_samples, 1, -1)
        self.unsafe_query_acts_array = self.unsafe_query_acts_array.reshape(n_unsafe_samples, 1, -1)
        self.unsafe_avg_sae_features = self.unsafe_avg_sae_features.reshape(n_unsafe_samples, 1, -1)
        
        print(f"安全样本数量: {n_safe_samples}")
        print(f"非安全样本数量: {n_unsafe_samples}")
    
    def _load_important_sae_features(self) -> List[int]:
        """加载重要的SAE特征维度"""
        print("加载重要SAE特征维度...")
        
        # 构建特征重要性文件路径
        feature_file = f"./sae_analysis_results/feature_importance_{self.model_alias}_safeedit_layer_20.csv"
        
        if not os.path.exists(feature_file):
            print(f"警告: 特征重要性文件不存在: {feature_file}，将使用所有特征")
            return None
        
        import pandas as pd
        feature_df = pd.read_csv(feature_file)
        total_features = len(feature_df)
        
        # 计算前35%的数量（默认值）
        top_freq_count = int(total_features * 0.35)
        top_value_count = int(total_features * 0.35)
        
        # 按频率差异绝对值排序
        freq_sorted = feature_df.sort_values("abs_mean_freq_diff", ascending=False)
        top_freq_features = freq_sorted.head(top_freq_count).index.tolist()
        
        # 按激活值差异绝对值排序
        value_sorted = feature_df.sort_values("abs_mean_value_diff", ascending=False)
        top_value_features = value_sorted.head(top_value_count).index.tolist()
        
        # 取交集（mix模式）
        intersection = set(top_freq_features) & set(top_value_features)
        intersection_features = [idx for idx in top_freq_features if idx in intersection]
        
        # 按频率差异绝对值对最终特征列表进行排序
        intersection_df = feature_df.loc[intersection_features]
        sorted_intersection_df = intersection_df.sort_values("abs_mean_freq_diff", ascending=False)
        sorted_intersection_features = sorted_intersection_df.index.tolist()
        
        print(f"总特征数量: {total_features}")
        print(f"频率差异前35%特征数量: {len(top_freq_features)}")
        print(f"激活值差异前35%特征数量: {len(top_value_features)}")
        print(f"最终重要SAE特征数量: {len(sorted_intersection_features)}")
        print(f"已按频率差异绝对值排序")
        
        return sorted_intersection_features
    
    def _load_generation_results(self) -> Tuple[List[Dict], List[Dict]]:
        """加载生成结果"""
        print("加载生成结果...")
        
        with open(self.static_result_path, 'r') as f:
            static_results = json.load(f)
        
        with open(self.dynamic_result_path, 'r') as f:
            dynamic_results = json.load(f)
        
        print(f"静态结果样本数: {len(static_results)}")
        print(f"动态结果样本数: {len(dynamic_results)}")
        
        return static_results, dynamic_results
    
    def _find_performance_differences(self, static_results: List[Dict], dynamic_results: List[Dict], 
                                    top_k: int = 10, use_confidence: bool = True) -> Dict:
        """分析性能差异"""
        print("分析性能差异...")
        
        # 评估安全分数
        static_scores = self._evaluate_safety_scores(static_results, use_confidence)
        dynamic_scores = self._evaluate_safety_scores(dynamic_results, use_confidence)
        
        # 计算性能差异并筛选出判断不一致的样本
        performance_diffs = []
        for i, (static_score, dynamic_score) in enumerate(zip(static_scores, dynamic_scores)):
            diff = static_score - dynamic_score
            
            # 判断是否为判断不一致的样本（一种方式判断为安全，另一种方式判断为风险）
            if use_confidence:
                # 使用置信度时，判断标准为0.5
                static_safe = static_score > 0.5
                dynamic_safe = dynamic_score > 0.5
            else:
                # 使用二分类时，判断标准为0（0=安全，1=风险）
                static_safe = static_score == 0
                dynamic_safe = dynamic_score == 0
            
            # 只有当两种方法判断不一致时才包含
            if static_safe != dynamic_safe:
                performance_diffs.append({
                    'index': i,
                    'static_score': static_score,
                    'dynamic_score': dynamic_score,
                    'difference': diff,
                    'abs_difference': abs(diff),
                    'static_safe': static_safe,
                    'dynamic_safe': dynamic_safe
                })
        
        print(f"判断不一致的样本总数: {len(performance_diffs)}")
        
        # 分别分类样本
        static_better_samples = [s for s in performance_diffs if s['difference'] > 0]  # 静态更好（静态安全，动态风险）
        dynamic_better_samples = [s for s in performance_diffs if s['difference'] < 0]  # 动态更好（动态安全，静态风险）
        
        # 分别按绝对差异排序并选择top-k
        static_better_samples.sort(key=lambda x: x['abs_difference'], reverse=True)
        dynamic_better_samples.sort(key=lambda x: x['abs_difference'], reverse=True)
        
        # 各选择top-k个样本
        static_better_top_k = static_better_samples[:top_k]
        dynamic_better_top_k = dynamic_better_samples[:top_k]
        
        print(f"静态更好的样本总数: {len(static_better_samples)}")
        print(f"动态更好的样本总数: {len(dynamic_better_samples)}")
        print(f"选择的静态更好样本数: {len(static_better_top_k)}")
        print(f"选择的动态更好样本数: {len(dynamic_better_top_k)}")
        
        return {
            'summary': {
                'total_samples': len(static_scores),  # 总样本数
                'inconsistent_samples': len(performance_diffs),  # 判断不一致的样本数
                'top_k': top_k,
                'static_better_total_count': len(static_better_samples),
                'dynamic_better_total_count': len(dynamic_better_samples),
                'static_better_selected_count': len(static_better_top_k),
                'dynamic_better_selected_count': len(dynamic_better_top_k)
            },
            'static_better_samples': static_better_top_k,
            'dynamic_better_samples': dynamic_better_top_k
        }
    
    def _evaluate_safety_scores(self, results: List[Dict], use_confidence: bool = False) -> List[float]:
        """评估安全分数"""
        sequences = [result['generation'] for result in results]
        
        if use_confidence:
            return SteerVectorUtils.predict_safety_confidence(
                self.safety_classifier_model, self.safety_classifier_tokenizer, sequences)
        else:
            # 二分类结果
            scores = []
            batch_size = 32
            
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                
                inputs = self.safety_classifier_tokenizer(
                    batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.safety_classifier_model(**inputs)
                    predictions = torch.argmax(outputs.logits, dim=1)
                    batch_scores = predictions.cpu().numpy().tolist()
                
                scores.extend(batch_scores)
            
            return scores
    
    def _load_static_steer_vector_and_features(self) -> Dict:
        """加载静态steer vector和SAE特征"""
        print("加载静态steer vector...")
        
        static_vector_path = f"./sae_analysis_results/{self.model_alias}_safeedit_layer_20_act_and_fre_trim_0.35.pt"
        static_sae_features_path = f"./sae_analysis_results/{self.model_alias}_safeedit_layer_20_static_sae_features.pt"
        
        if not os.path.exists(static_vector_path):
            raise FileNotFoundError(f"静态steer vector文件不存在: {static_vector_path}")
        
        static_steer_vector = torch.load(static_vector_path, map_location='cpu')
        
        # 检查是否已有计算好的SAE中间特征
        if os.path.exists(static_sae_features_path):
            print("加载已计算的静态SAE中间特征...")
            static_sae_features = torch.load(static_sae_features_path, map_location='cpu')
        else:
            print("计算静态SAE中间特征...")
            static_sae_features = self._calculate_static_sae_features_forward(static_steer_vector)
            torch.save(static_sae_features, static_sae_features_path)
            print(f"静态SAE中间特征已保存到: {static_sae_features_path}")
        
        print(f"静态steer vector形状: {static_steer_vector.shape}")
        print(f"静态SAE特征形状: {static_sae_features.shape}")
        
        return {
            "steer_vector": static_steer_vector,
            "sae_features": static_sae_features
        }
    
    def _calculate_static_sae_features_forward(self, static_steer_vector: torch.Tensor) -> torch.Tensor:
        """计算静态SAE中间特征"""
        W_dec = self.sae_model.W_dec.cpu()
        static_sae_features = static_steer_vector @ torch.pinverse(W_dec)
        return static_sae_features
    
    def _generate_steer_vectors_for_analysis(self, sample_indices: List[int], 
                                           nearest_neighbor_k: int,
                                           top_freq: float,
                                           top_value: float,
                                           filter_type: str,
                                           calculation_mode: str = "neighbor_based",
                                           use_caa_norm: bool = False) -> Dict:
        """为分析生成steer vectors"""
        print(f"为{len(sample_indices)}个样本生成动态steer vectors...")
        
        analysis_results = {}
        
        for sample_idx in tqdm(sample_indices, desc="生成steer vectors"):
            try:
                # 获取样本的prompt
                test_data = load_safeedit_test_data(model_alias=self.model_alias, apply_chat_format=True)
                prompt = test_data[sample_idx]['prompt']
                
                # 获取查询激活
                query_activation = self._get_query_activation(prompt)
                
                # 计算动态steer vector
                dynamic_result = self._calculate_dynamic_steer_vector(
                    query_activation, nearest_neighbor_k, top_freq, top_value, filter_type, calculation_mode, use_caa_norm
                )
                
                analysis_results[sample_idx] = {
                    'steer_vector': dynamic_result['steer_vector'],
                    'sae_features': dynamic_result['sae_features'],
                    'steer_vector_norm': torch.norm(dynamic_result['steer_vector']).item(),
                    'sae_features_norm': torch.norm(dynamic_result['sae_features']).item(),
                    'sae_sparsity': SteerVectorUtils.calculate_sparsity(dynamic_result['sae_features'])
                }
                
            except Exception as e:
                print(f"处理样本{sample_idx}时出错: {e}")
                continue
        
        return analysis_results
    
    def _get_query_activation(self, prompt: str) -> torch.Tensor:
        """获取查询激活"""
        def hook_fn(activations, hook):
            hook_fn.activation = activations
        
        with self.model.hooks([("blocks.20.hook_resid_post", hook_fn)]):
            self.model(prompt)
        
        # 获取最后一个token的激活并转换为float32
        last_token_activation = hook_fn.activation[0, -1, :].float().detach()
        return last_token_activation
    
    def _calculate_dynamic_steer_vector(self, query_activation: torch.Tensor, 
                                      nearest_neighbor_k: int,
                                      top_freq: float,
                                      top_value: float,
                                      filter_type: str,
                                      calculation_mode: str = "neighbor_based",
                                      use_caa_norm: bool = False) -> Dict:
        """计算动态steer vector"""
        if calculation_mode == "neighbor_based":
            return self._calculate_neighbor_based_steer_vector(
                query_activation, nearest_neighbor_k, top_freq, top_value, filter_type, use_caa_norm
            )
        elif calculation_mode == "fixed_features":
            return self._calculate_fixed_features_steer_vector(
                query_activation, nearest_neighbor_k, top_freq, top_value, filter_type, use_caa_norm
            )
        else:
            raise ValueError(f"不支持的计算模式: {calculation_mode}")
    
    def _calculate_neighbor_based_steer_vector(self, query_activation: torch.Tensor, 
                                             nearest_neighbor_k: int,
                                             top_freq: float,
                                             top_value: float,
                                             filter_type: str,
                                             use_caa_norm: bool = False) -> Dict:
        """基于最近邻样本计算动态steer vector（原始方法）"""
        # 准备查询向量
        query_vector = query_activation.detach().cpu().float().numpy().reshape(1, -1)
        
        # 找到最近邻
        safe_query_acts_all = self.safe_query_acts_array[:, 0, :]
        safe_dists_all = np.linalg.norm(safe_query_acts_all - query_vector, axis=1)
        safe_indices = np.argsort(safe_dists_all)[:nearest_neighbor_k]
        
        unsafe_query_acts_all = self.unsafe_query_acts_array[:, 0, :]
        unsafe_dists_all = np.linalg.norm(unsafe_query_acts_all - query_vector, axis=1)
        unsafe_indices = np.argsort(unsafe_dists_all)[:nearest_neighbor_k]
        
        # 获取最近邻样本的SAE特征
        safe_feature_acts = self.safe_avg_sae_features[safe_indices, 0, :]
        unsafe_feature_acts = self.unsafe_avg_sae_features[unsafe_indices, 0, :]
        
        # 计算统计量
        activation_threshold = 0.0
        pos_feature_freq = (safe_feature_acts > activation_threshold).astype(float).sum(0)
        neg_feature_freq = (unsafe_feature_acts > activation_threshold).astype(float).sum(0)
        pos_act_mean = safe_feature_acts.mean(0)
        neg_act_mean = unsafe_feature_acts.mean(0)
        feature_score = pos_act_mean - neg_act_mean
        
        # 转换为torch tensor
        pos_feature_freq = torch.tensor(pos_feature_freq, dtype=torch.float32)
        neg_feature_freq = torch.tensor(neg_feature_freq, dtype=torch.float32)
        pos_act_mean = torch.tensor(pos_act_mean, dtype=torch.float32)
        neg_act_mean = torch.tensor(neg_act_mean, dtype=torch.float32)
        feature_score = torch.tensor(feature_score, dtype=torch.float32)
        
        # 应用特征筛选
        if filter_type == "mix":
            diff_data = pos_feature_freq - neg_feature_freq
            norm_act = SteerVectorUtils.signed_min_max_normalize(feature_score)
            norm_diff = SteerVectorUtils.signed_min_max_normalize(diff_data)
            
            mask = ((norm_act > 0) & (norm_diff > 0)) | ((norm_act < 0) & (norm_diff < 0))
            scores = torch.zeros_like(norm_diff)
            scores[mask] = norm_diff[mask]
            
            threshold_fre = torch.sort(torch.abs(scores), descending=True, stable=True).values[int(top_freq * len(scores))]
            prune_mask = torch.abs(scores) >= threshold_fre
            
            act_threshold = torch.sort(torch.abs(feature_score), descending=True, stable=True).values[int(top_value * len(feature_score))]
            act_top_mask = torch.abs(feature_score) >= act_threshold
            
            combined_mask = prune_mask & act_top_mask
            act_data_combined = feature_score.clone()
            act_data_combined[~combined_mask] = 0
            steering_vector = act_data_combined @ self.sae_model.W_dec.cpu()
        
        elif filter_type == "freq":
            # 仅基于频率差异进行筛选
            diff_data = pos_feature_freq - neg_feature_freq
            norm_diff = SteerVectorUtils.signed_min_max_normalize(diff_data)
            
            threshold_fre = torch.sort(torch.abs(norm_diff), descending=True, stable=True).values[int(top_freq * len(norm_diff))]
            prune_mask = torch.abs(norm_diff) >= threshold_fre
            
            act_data_combined = feature_score.clone()
            act_data_combined[~prune_mask] = 0
            steering_vector = act_data_combined @ self.sae_model.W_dec.cpu()
        
        elif filter_type == "value":
            # 仅基于激活值进行筛选
            act_threshold = torch.sort(torch.abs(feature_score), descending=True, stable=True).values[int(top_value * len(feature_score))]
            act_top_mask = torch.abs(feature_score) >= act_threshold
            
            act_data_combined = feature_score.clone()
            act_data_combined[~act_top_mask] = 0
            steering_vector = act_data_combined @ self.sae_model.W_dec.cpu()
        
        else:
            raise ValueError(f"不支持的筛选类型: {filter_type}")
        
        # 应用范数归一化
        if use_caa_norm and self.caa_vector is not None:
            caa_norm = torch.norm(self.caa_vector)
            steering_norm = torch.norm(steering_vector)
            if steering_norm > 0:
                steering_vector = steering_vector * (caa_norm / steering_norm)
        
        # 计算SAE中间特征
        sae_features = steering_vector @ torch.pinverse(self.sae_model.W_dec.cpu())
        
        return {
            'steer_vector': steering_vector,
            'sae_features': sae_features
        }
    
    def _calculate_fixed_features_steer_vector(self, query_activation: torch.Tensor, 
                                            nearest_neighbor_k: int,
                                            top_freq: float,
                                            top_value: float,
                                            filter_type: str,
                                            use_caa_norm: bool = False) -> Dict:
        """基于全局固定SAE特征维度计算动态steer vector（类似flexiable_sae_safety_querylevel.py）"""
        # 准备查询向量
        query_vector = query_activation.detach().cpu().float().numpy().reshape(1, -1)
        
        # 找到最近邻
        safe_query_acts_all = self.safe_query_acts_array[:, 0, :]
        safe_dists_all = np.linalg.norm(safe_query_acts_all - query_vector, axis=1)
        safe_indices = np.argsort(safe_dists_all)[:nearest_neighbor_k]
        
        unsafe_query_acts_all = self.unsafe_query_acts_array[:, 0, :]
        unsafe_dists_all = np.linalg.norm(unsafe_query_acts_all - query_vector, axis=1)
        unsafe_indices = np.argsort(unsafe_dists_all)[:nearest_neighbor_k]
        
        # 获取最近邻样本的SAE特征
        safe_feature_acts = self.safe_avg_sae_features[safe_indices, 0, :]
        unsafe_feature_acts = self.unsafe_avg_sae_features[unsafe_indices, 0, :]
        
        # 计算统计量
        activation_threshold = 0.0
        pos_feature_freq = (safe_feature_acts > activation_threshold).astype(float).sum(0)
        neg_feature_freq = (unsafe_feature_acts > activation_threshold).astype(float).sum(0)
        pos_act_mean = safe_feature_acts.mean(0)
        neg_act_mean = unsafe_feature_acts.mean(0)
        feature_score = pos_act_mean - neg_act_mean
        
        # 转换为torch tensor
        pos_feature_freq = torch.tensor(pos_feature_freq, dtype=torch.float32)
        neg_feature_freq = torch.tensor(neg_feature_freq, dtype=torch.float32)
        pos_act_mean = torch.tensor(pos_act_mean, dtype=torch.float32)
        neg_act_mean = torch.tensor(neg_act_mean, dtype=torch.float32)
        feature_score = torch.tensor(feature_score, dtype=torch.float32)
        
        # 应用特征筛选
        if filter_type == "mix":
            diff_data = pos_feature_freq - neg_feature_freq
            norm_act = SteerVectorUtils.signed_min_max_normalize(feature_score)
            norm_diff = SteerVectorUtils.signed_min_max_normalize(diff_data)
            
            mask = ((norm_act > 0) & (norm_diff > 0)) | ((norm_act < 0) & (norm_diff < 0))
            scores = torch.zeros_like(norm_diff)
            scores[mask] = norm_diff[mask]
            
            threshold_fre = torch.sort(torch.abs(scores), descending=True, stable=True).values[int(top_freq * len(scores))]
            prune_mask = torch.abs(scores) >= threshold_fre
            
            act_threshold = torch.sort(torch.abs(feature_score), descending=True, stable=True).values[int(top_value * len(feature_score))]
            act_top_mask = torch.abs(feature_score) >= act_threshold
            
            combined_mask = prune_mask & act_top_mask
            act_data_combined = feature_score.clone()
            act_data_combined[~combined_mask] = 0
            
            # 使用全局固定的重要SAE特征维度进行mask
            if self.important_sae_features is not None:
                global_mask = torch.zeros_like(act_data_combined)
                global_mask[self.important_sae_features] = 1.0
                act_data_combined = act_data_combined * global_mask
            
            steering_vector = act_data_combined @ self.sae_model.W_dec.cpu()
        
        elif filter_type == "freq":
            # 仅基于频率差异进行筛选
            diff_data = pos_feature_freq - neg_feature_freq
            norm_diff = SteerVectorUtils.signed_min_max_normalize(diff_data)
            
            threshold_fre = torch.sort(torch.abs(norm_diff), descending=True, stable=True).values[int(top_freq * len(norm_diff))]
            prune_mask = torch.abs(norm_diff) >= threshold_fre
            
            act_data_combined = feature_score.clone()
            act_data_combined[~prune_mask] = 0
            
            # 使用全局固定的重要SAE特征维度进行mask
            if self.important_sae_features is not None:
                global_mask = torch.zeros_like(act_data_combined)
                global_mask[self.important_sae_features] = 1.0
                act_data_combined = act_data_combined * global_mask
            
            steering_vector = act_data_combined @ self.sae_model.W_dec.cpu()
        
        elif filter_type == "value":
            # 仅基于激活值进行筛选
            act_threshold = torch.sort(torch.abs(feature_score), descending=True, stable=True).values[int(top_value * len(feature_score))]
            act_top_mask = torch.abs(feature_score) >= act_threshold
            
            act_data_combined = feature_score.clone()
            act_data_combined[~act_top_mask] = 0
            
            # 使用全局固定的重要SAE特征维度进行mask
            if self.important_sae_features is not None:
                global_mask = torch.zeros_like(act_data_combined)
                global_mask[self.important_sae_features] = 1.0
                act_data_combined = act_data_combined * global_mask
            
            steering_vector = act_data_combined @ self.sae_model.W_dec.cpu()
        
        else:
            raise ValueError(f"不支持的筛选类型: {filter_type}")
        
        # 应用范数归一化
        if use_caa_norm and self.caa_vector is not None:
            caa_norm = torch.norm(self.caa_vector)
            steering_norm = torch.norm(steering_vector)
            if steering_norm > 0:
                steering_vector = steering_vector * (caa_norm / steering_norm)
        
        # 计算SAE中间特征
        sae_features = act_data_combined.cpu()
        
        return {
            'steer_vector': steering_vector,
            'sae_features': sae_features
        }
    
    def _create_visualizations(self, analysis_data: Dict, static_data: Dict, analysis_type: str):
        """创建可视化分析"""
        print(f"创建{analysis_type}样本的可视化分析...")
        
        # 绘制静态激活值图
        self._plot_static_activations(static_data, analysis_type)
        
            
        # 绘制诊断图表（保存到每个样本的专用目录）
        if analysis_data:
            self._create_diagnostic_plots(analysis_data, static_data, analysis_type)
    
    def _plot_static_activations(self, static_data: Dict, analysis_type: str):
        """绘制静态激活值图"""
        static_steer_vector = static_data["steer_vector"]
        static_sae_features = static_data["sae_features"]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'Static Steer Vector and SAE Features Activation Values - {analysis_type.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold')
        
        # 绘制静态steer vector激活值
        steer_activations = static_steer_vector.numpy()
        ax1.plot(range(len(steer_activations)), steer_activations, 'b-', alpha=0.7, linewidth=0.5)
        ax1.set_title('Static Steer Vector Activation Values')
        ax1.set_xlabel('Feature Index')
        ax1.set_ylabel('Activation Value')
        ax1.grid(True, alpha=0.3)
        
        # 绘制静态SAE特征激活值
        sae_activations = static_sae_features.numpy()
        ax2.plot(range(len(sae_activations)), sae_activations, 'r-', alpha=0.7, linewidth=0.5)
        ax2.set_title('Static SAE Features Activation Values')
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Activation Value')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'static_activations_{analysis_type}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"静态激活值图已保存到: {os.path.join(self.output_dir, f'static_activations_{analysis_type}.png')}")
    
    def _plot_summary_differences(self, analysis_data: Dict, static_data: Dict, analysis_type: str):
        """创建汇总差异对比图"""
        if static_data is None:
            return
        
        static_steer_vector = static_data["steer_vector"]
        static_sae_features = static_data["sae_features"]
        sample_indices = list(analysis_data.keys())
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Summary of All Samples - Dynamic vs Static Differences ({analysis_type.replace("_", " ").title()})', 
                    fontsize=16, fontweight='bold')
        
        # 收集所有差异数据
        all_steer_diff = []
        all_sae_diff = []
        
        for sample_idx in sample_indices:
            sample_data = analysis_data[sample_idx]
            dynamic_steer_vector = sample_data['steer_vector']
            dynamic_sae_features = sample_data['sae_features']
            
            steer_diff = dynamic_steer_vector - static_steer_vector
            sae_diff = dynamic_sae_features - static_sae_features
            
            all_steer_diff.append(steer_diff.numpy())
            all_sae_diff.append(sae_diff.numpy())
        
        # 转换为numpy数组
        all_steer_diff = np.array(all_steer_diff)
        all_sae_diff = np.array(all_sae_diff)
        
        # 1. Steer vector差异的均值
        mean_steer_diff = np.mean(all_steer_diff, axis=0)
        ax1.plot(range(len(mean_steer_diff)), mean_steer_diff, 'blue', alpha=0.8, linewidth=1, label='Mean Difference')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_title('Mean Steer Vector Difference')
        ax1.set_xlabel('Feature Index')
        ax1.set_ylabel('Mean Difference Value')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Steer vector差异的标准差
        std_steer_diff = np.std(all_steer_diff, axis=0)
        ax2.plot(range(len(std_steer_diff)), std_steer_diff, 'red', alpha=0.8, linewidth=1, label='Std Difference')
        ax2.set_title('Steer Vector Difference Standard Deviation')
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Standard Deviation')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. SAE特征差异的均值
        mean_sae_diff = np.mean(all_sae_diff, axis=0)
        ax3.plot(range(len(mean_sae_diff)), mean_sae_diff, 'green', alpha=0.8, linewidth=1, label='Mean Difference')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Mean SAE Features Difference')
        ax3.set_xlabel('Feature Index')
        ax3.set_ylabel('Mean Difference Value')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. SAE特征差异的标准差
        std_sae_diff = np.std(all_sae_diff, axis=0)
        ax4.plot(range(len(std_sae_diff)), std_sae_diff, 'purple', alpha=0.8, linewidth=1, label='Std Difference')
        ax4.set_title('SAE Features Difference Standard Deviation')
        ax4.set_xlabel('Feature Index')
        ax4.set_ylabel('Standard Deviation')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'summary_differences_{analysis_type}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"汇总差异对比图已保存到: {os.path.join(self.output_dir, f'summary_differences_{analysis_type}.png')}")
    
    def _generate_analysis_report(self, performance_analysis: Dict, static_better_analysis: Dict, dynamic_better_analysis: Dict):
        """生成分析报告"""
        print("生成分析报告...")
        
        # 加载测试数据以获取query和回答
        test_data = load_safeedit_test_data(model_alias=self.model_alias, apply_chat_format=True)
        
        # 加载生成结果
        with open(self.static_result_path, 'r') as f:
            static_results = json.load(f)
        with open(self.dynamic_result_path, 'r') as f:
            dynamic_results = json.load(f)
        
        # 准备简化的分析数据
        def prepare_simplified_analysis(analysis_data: Dict, results: List[Dict]) -> Dict:
            simplified = {}
            for sample_idx in analysis_data.keys():
                sample_idx_int = int(sample_idx)
                if sample_idx_int < len(test_data) and sample_idx_int < len(results):
                    simplified[sample_idx] = {
                        "query": test_data[sample_idx_int]['prompt'],
                        "static_answer": results[sample_idx_int]['generation'],
                        "dynamic_answer": dynamic_results[sample_idx_int]['generation'],
                        "steer_vector_norm": analysis_data[sample_idx]['steer_vector_norm'],
                        "sae_features_norm": analysis_data[sample_idx]['sae_features_norm'],
                        "sae_sparsity": analysis_data[sample_idx]['sae_sparsity']
                    }
            return simplified
        
        # 准备简化的分析数据
        simplified_static_analysis = prepare_simplified_analysis(static_better_analysis, static_results)
        simplified_dynamic_analysis = prepare_simplified_analysis(dynamic_better_analysis, dynamic_results)
        
        report = {
            "analysis_timestamp": str(datetime.now()),
            "performance_summary": performance_analysis["summary"],
            "static_better_samples": {
                "count": len(static_better_analysis),
                "sample_indices": list(static_better_analysis.keys()),
                "samples": simplified_static_analysis
            },
            "dynamic_better_samples": {
                "count": len(dynamic_better_analysis),
                "sample_indices": list(dynamic_better_analysis.keys()),
                "samples": simplified_dynamic_analysis
            }
        }
        
        # 保存报告
        report_path = os.path.join(self.output_dir, "analysis_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"分析报告已保存到: {report_path}")
        
        # 打印摘要
        print("\n=== 分析摘要 ===")
        print(f"总样本数: {performance_analysis['summary']['total_samples']}")
        print(f"判断不一致的样本数: {performance_analysis['summary']['inconsistent_samples']}")
        print(f"静态更好的样本总数: {performance_analysis['summary']['static_better_total_count']}")
        print(f"动态更好的样本总数: {performance_analysis['summary']['dynamic_better_total_count']}")
        print(f"选择的静态更好样本数: {performance_analysis['summary']['static_better_selected_count']}")
        print(f"选择的动态更好样本数: {performance_analysis['summary']['dynamic_better_selected_count']}")
        print(f"分析结果已保存到: {self.output_dir}")
    
    def _create_diagnostic_plots(self, analysis_data: Dict, static_data: Dict, analysis_type: str):
        """为每个样本创建诊断图表"""
        print(f"为{analysis_type}样本创建诊断图表...")
        
        # 加载测试数据以获取prompt和回答
        test_data = load_safeedit_test_data(model_alias=self.model_alias, apply_chat_format=True)
        
        # 加载生成结果
        with open(self.static_result_path, 'r') as f:
            static_results = json.load(f)
        with open(self.dynamic_result_path, 'r') as f:
            dynamic_results = json.load(f)
        
        for sample_idx in analysis_data.keys():
            sample_idx_int = int(sample_idx)
            if sample_idx_int >= len(test_data):
                continue
                
            # 创建样本专用目录
            sample_dir = os.path.join(self.output_dir, f"sample_{sample_idx_int}_{analysis_type}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # 保存样本信息到JSON文件
            sample_info = {
                "sample_index": sample_idx_int,
                "prompt": test_data[sample_idx_int]['prompt'],
                "static_answer": static_results[sample_idx_int]['generation'],
                "dynamic_answer": dynamic_results[sample_idx_int]['generation'],
                "analysis_type": analysis_type
            }
            
            with open(os.path.join(sample_dir, "sample_info.json"), 'w', encoding='utf-8') as f:
                json.dump(sample_info, f, indent=2, ensure_ascii=False)
            
            # 获取该样本的邻居特征数据
            neighbor_data = self._get_neighbor_features_for_sample(sample_idx_int)
            if neighbor_data is None:
                print(f"警告：无法获取样本{sample_idx_int}的邻居特征数据")
                continue
            
            # 创建SAE特征差值图
            self._create_sample_sae_feature_difference_heatmap(neighbor_data, static_data, sample_idx_int, sample_dir)
            
            print(f"样本{sample_idx_int}的诊断图表已保存到: {sample_dir}")
    
    def _get_neighbor_features_for_sample(self, sample_idx: int) -> Dict:
        """获取指定样本的邻居特征数据"""
        try:
            # 获取样本的prompt
            test_data = load_safeedit_test_data(model_alias=self.model_alias, apply_chat_format=True)
            prompt = test_data[sample_idx]['prompt']
            
            # 获取查询激活
            query_activation = self._get_query_activation(prompt)
            
            # 准备查询向量
            query_vector = query_activation.detach().cpu().float().numpy().reshape(1, -1)
            
            # 找到最近邻（使用默认的k=10）
            k = 10
            safe_query_acts_all = self.safe_query_acts_array[:, 0, :]
            safe_dists_all = np.linalg.norm(safe_query_acts_all - query_vector, axis=1)
            safe_indices = np.argsort(safe_dists_all)[:k]
            
            unsafe_query_acts_all = self.unsafe_query_acts_array[:, 0, :]
            unsafe_dists_all = np.linalg.norm(unsafe_query_acts_all - query_vector, axis=1)
            unsafe_indices = np.argsort(unsafe_dists_all)[:k]
            
            # 获取最近邻样本的SAE特征
            safe_feature_acts = self.safe_avg_sae_features[safe_indices, 0, :]
            unsafe_feature_acts = self.unsafe_avg_sae_features[unsafe_indices, 0, :]
            
            # 获取重要特征索引
            important_feature_indices = self._get_important_feature_indices()
            
            return {
                'safe_neighbor_features': safe_feature_acts,
                'unsafe_neighbor_features': unsafe_feature_acts,
                'important_feature_indices': important_feature_indices
            }
            
        except Exception as e:
            print(f"获取样本{sample_idx}邻居特征时出错: {e}")
            return None
    
    def _get_important_feature_indices(self) -> List[int]:
        """获取重要特征索引"""
        if self.important_sae_features is not None:
            return self.important_sae_features
        else:
            # 如果没有预定义的重要特征，使用所有特征
            return list(range(self.safe_avg_sae_features.shape[1]))
    
    def _create_feature_activation_heatmap(self, neighbor_data: Dict, sample_idx: int, sample_dir: str):
        """创建特征激活热力图"""
        safe_features = neighbor_data['safe_neighbor_features']
        unsafe_features = neighbor_data['unsafe_neighbor_features']
        important_indices = neighbor_data['important_feature_indices']
        
        # 数据预处理
        mean_safe_activations = np.mean(safe_features, axis=0)
        mean_unsafe_activations = np.mean(unsafe_features, axis=0)
        
        # 筛选重要特征
        mean_safe_important = mean_safe_activations[important_indices]
        mean_unsafe_important = mean_unsafe_activations[important_indices]
        
        # 使用固定的重要特征维度（按排名选择前100个）
        max_features = 100
        if len(important_indices) > max_features:
            # 选择前max_features个重要特征（按重要性排名）
            display_indices = important_indices[:max_features]
            mean_safe_important = mean_safe_important[:max_features]
            mean_unsafe_important = mean_unsafe_important[:max_features]
        else:
            display_indices = important_indices
        
        # 准备绘图数据
        plot_data = np.vstack([mean_safe_important, mean_unsafe_important]).T
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, max(8, len(display_indices) * 0.15)))
        
        # 绘制热力图
        im = ax.imshow(plot_data, cmap='viridis', aspect='auto')
        
        # 设置坐标轴
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Safe Neighbors', 'Unsafe Neighbors'])
        ax.set_ylabel('Important Feature Index')
        ax.set_title(f'Sample {sample_idx}: Mean Activation Heatmap (Top {len(display_indices)} Features)')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean Activation Value')
        
        # 设置Y轴刻度
        if len(display_indices) <= 50:
            ax.set_yticks(range(len(display_indices)))
            ax.set_yticklabels([f'{idx}' for idx in display_indices])
        else:
            # 如果特征太多，只显示部分标签
            step = max(1, len(display_indices) // 20)
            ax.set_yticks(range(0, len(display_indices), step))
            ax.set_yticklabels([f'{display_indices[i]}' for i in range(0, len(display_indices), step)])
        
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, 'feature_activation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存完整数据到CSV文件
        full_data = {
            'feature_index': important_indices,
            'mean_safe_activation': mean_safe_activations[important_indices],
            'mean_unsafe_activation': mean_unsafe_activations[important_indices],
            'activation_difference': mean_safe_activations[important_indices] - mean_unsafe_activations[important_indices],
            'abs_activation_difference': np.abs(mean_safe_activations[important_indices] - mean_unsafe_activations[important_indices])
        }
        
        import pandas as pd
        df = pd.DataFrame(full_data)
        df = df.sort_values('abs_activation_difference', ascending=False)
        df.to_csv(os.path.join(sample_dir, 'feature_activation_data.csv'), index=False)
        print(f"完整特征激活数据已保存到: {os.path.join(sample_dir, 'feature_activation_data.csv')}")
    
    def _create_activation_difference_heatmap(self, neighbor_data: Dict, sample_idx: int, sample_dir: str):
        """创建激活差值热力图（红色=正值，白色=0，蓝色=负值）"""
        safe_features = neighbor_data['safe_neighbor_features']
        unsafe_features = neighbor_data['unsafe_neighbor_features']
        important_indices = neighbor_data['important_feature_indices']
        
        # 数据预处理
        mean_safe_activations = np.mean(safe_features, axis=0)
        mean_unsafe_activations = np.mean(unsafe_features, axis=0)
        
        # 筛选重要特征
        mean_safe_important = mean_safe_activations[important_indices]
        mean_unsafe_important = mean_unsafe_activations[important_indices]
        
        # 使用固定的重要特征维度（按排名选择前100个）
        max_features = 100
        if len(important_indices) > max_features:
            # 选择前max_features个重要特征（按重要性排名）
            display_indices = important_indices[:max_features]
            mean_safe_important = mean_safe_important[:max_features]
            mean_unsafe_important = mean_unsafe_important[:max_features]
        else:
            display_indices = important_indices
        
        # 计算差值（安全 - 不安全）
        activation_differences = mean_safe_important - mean_unsafe_important
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, max(8, len(display_indices) * 0.15)))
        
        # 创建差值数据矩阵（单列）
        plot_data = activation_differences.reshape(-1, 1)
        
        # 绘制热力图，使用自定义颜色映射
        from matplotlib.colors import LinearSegmentedColormap
        
        # 创建红白蓝颜色映射
        colors = ['blue', 'white', 'red']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('red_white_blue', colors, N=n_bins)
        
        # 计算颜色范围
        max_abs_diff = max(abs(activation_differences.min()), abs(activation_differences.max()))
        vmin, vmax = -max_abs_diff, max_abs_diff
        
        im = ax.imshow(plot_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # 设置坐标轴
        ax.set_xticks([0])
        ax.set_xticklabels(['Safe - Unsafe'])
        ax.set_ylabel('Important Feature Index')
        ax.set_title(f'Sample {sample_idx}: Activation Difference Heatmap (Top {len(display_indices)} Features)')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activation Difference (Safe - Unsafe)')
        
        # 设置Y轴刻度
        if len(display_indices) <= 50:
            ax.set_yticks(range(len(display_indices)))
            ax.set_yticklabels([f'{idx}' for idx in display_indices])
        else:
            # 如果特征太多，只显示部分标签
            step = max(1, len(display_indices) // 20)
            ax.set_yticks(range(0, len(display_indices), step))
            ax.set_yticklabels([f'{display_indices[i]}' for i in range(0, len(display_indices), step)])
        
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, 'activation_difference_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"激活差值热力图已保存到: {os.path.join(sample_dir, 'activation_difference_heatmap.png')}")
    
    def _create_sample_sae_feature_difference_heatmap(self, neighbor_data: Dict, static_data: Dict, sample_idx: int, sample_dir: str):
        """为单个样本创建SAE特征差值热力图"""
        safe_features = neighbor_data['safe_neighbor_features']
        unsafe_features = neighbor_data['unsafe_neighbor_features']
        important_indices = neighbor_data['important_feature_indices']
        
        # 数据预处理
        mean_safe_activations = np.mean(safe_features, axis=0)
        mean_unsafe_activations = np.mean(unsafe_features, axis=0)
        
        # 筛选重要特征
        mean_safe_important = mean_safe_activations[important_indices]
        mean_unsafe_important = mean_unsafe_activations[important_indices]
        
        # 使用固定的重要特征维度（按排名选择前2000个）
        max_features = 2000
        if len(important_indices) > max_features:
            # 选择前max_features个重要特征（按重要性排名）
            display_indices = important_indices[:max_features]
            mean_safe_important = mean_safe_important[:max_features]
            mean_unsafe_important = mean_unsafe_important[:max_features]
        else:
            display_indices = important_indices
        
        # 计算SAE特征差值（安全 - 不安全）
        sae_feature_differences = mean_safe_important - mean_unsafe_important
        
        # 获取静态steer vector的SAE特征
        static_steer_vector = static_data["steer_vector"]
        static_sae_features = static_data["sae_features"]
        static_sae_important = static_sae_features[display_indices]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
        
        # 创建红白蓝颜色映射
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['blue', 'white', 'red']
        cmap = LinearSegmentedColormap.from_list('red_white_blue', colors, N=100)
        
        # 计算颜色范围
        max_abs_diff = max(abs(sae_feature_differences.min()), abs(sae_feature_differences.max()))
        vmin, vmax = -max_abs_diff, max_abs_diff
        
        # 绘制动态SAE特征差值热力图
        dynamic_plot_data = sae_feature_differences.reshape(100, 20)  # 重塑为100*20矩阵
        im1 = ax1.imshow(dynamic_plot_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax1.set_title(f'Sample {sample_idx}: Dynamic SAE Feature Difference\n(Safe - Unsafe Neighbors)')
        ax1.set_ylabel('Feature Row Index')
        ax1.set_xlabel('Feature Column Index')
        
        # 绘制静态steer vector SAE特征热力图
        static_plot_data = static_sae_important.reshape(100, 20)  # 重塑为100*20矩阵
        im2 = ax2.imshow(static_plot_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax2.set_title(f'Sample {sample_idx}: Static Steer Vector SAE Features')
        ax2.set_ylabel('Feature Row Index')
        ax2.set_xlabel('Feature Column Index')
        
        # 添加颜色条
        cbar = plt.colorbar(im1, ax=[ax1, ax2], shrink=0.8)
        cbar.set_label('SAE Feature Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, 'sae_feature_difference_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"样本{sample_idx}的SAE特征差值热力图已保存到: {os.path.join(sample_dir, 'sae_feature_difference_heatmap.png')}")
    
    def _create_sample_activation_comparison(self, sample_data: Dict, static_data: Dict, sample_idx: int, sample_dir: str):
        """为单个样本创建激活值差异对比图（6个子图）"""
        static_steer_vector = static_data["steer_vector"]
        static_sae_features = static_data["sae_features"]
        dynamic_steer_vector = sample_data['steer_vector']
        dynamic_sae_features = sample_data['sae_features']
        
        # 计算差异
        steer_diff = dynamic_steer_vector - static_steer_vector
        sae_diff = dynamic_sae_features - static_sae_features
        
        # 创建6个子图的图表
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Sample {sample_idx} - Static vs Dynamic Comparison', 
                    fontsize=16, fontweight='bold')
        
        # 1. 静态steer vector激活值
        static_steer_vals = static_steer_vector.numpy()
        ax1.plot(range(len(static_steer_vals)), static_steer_vals, 'b-', alpha=0.7, linewidth=0.5, label='Static')
        ax1.set_title('Static Steer Vector')
        ax1.set_xlabel('Feature Index')
        ax1.set_ylabel('Activation Value')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 动态steer vector激活值
        dynamic_steer_vals = dynamic_steer_vector.numpy()
        ax2.plot(range(len(dynamic_steer_vals)), dynamic_steer_vals, 'r-', alpha=0.7, linewidth=0.5, label='Dynamic')
        ax2.set_title('Dynamic Steer Vector')
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Activation Value')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. steer vector差异
        steer_diff_vals = steer_diff.numpy()
        ax3.plot(range(len(steer_diff_vals)), steer_diff_vals, 'g-', alpha=0.7, linewidth=0.5, label='Difference')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Steer Vector Difference\n(Dynamic - Static)')
        ax3.set_xlabel('Feature Index')
        ax3.set_ylabel('Activation Value')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 静态SAE特征激活值
        static_sae_vals = static_sae_features.numpy()
        ax4.plot(range(len(static_sae_vals)), static_sae_vals, 'b-', alpha=0.7, linewidth=0.5, label='Static')
        ax4.set_title('Static SAE Features')
        ax4.set_xlabel('Feature Index')
        ax4.set_ylabel('Activation Value')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. 动态SAE特征激活值
        dynamic_sae_vals = dynamic_sae_features.numpy()
        ax5.plot(range(len(dynamic_sae_vals)), dynamic_sae_vals, 'r-', alpha=0.7, linewidth=0.5, label='Dynamic')
        ax5.set_title('Dynamic SAE Features')
        ax5.set_xlabel('Feature Index')
        ax5.set_ylabel('Activation Value')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. SAE特征差异
        sae_diff_vals = sae_diff.numpy()
        ax6.plot(range(len(sae_diff_vals)), sae_diff_vals, 'g-', alpha=0.7, linewidth=0.5, label='Difference')
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax6.set_title('SAE Features Difference\n(Dynamic - Static)')
        ax6.set_xlabel('Feature Index')
        ax6.set_ylabel('Activation Value')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, 'activation_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"样本{sample_idx}的激活值对比图已保存到: {os.path.join(sample_dir, 'activation_comparison.png')}")
    
    def _create_feature_separation_scatter_plot(self, neighbor_data: Dict, sample_idx: int, sample_dir: str):
        """创建特征分离度散点图"""
        safe_features = neighbor_data['safe_neighbor_features']
        unsafe_features = neighbor_data['unsafe_neighbor_features']
        important_indices = neighbor_data['important_feature_indices']
        
        # 数据预处理
        mean_safe_activations = np.mean(safe_features, axis=0)
        mean_unsafe_activations = np.mean(unsafe_features, axis=0)
        
        # 筛选重要特征
        mean_safe_important = mean_safe_activations[important_indices]
        mean_unsafe_important = mean_unsafe_activations[important_indices]
        
        # 使用固定的重要特征维度（按排名选择前500个）
        max_features = 500
        if len(important_indices) > max_features:
            # 选择前max_features个重要特征（按重要性排名）
            display_indices = important_indices[:max_features]
            mean_safe_important = mean_safe_important[:max_features]
            mean_unsafe_important = mean_unsafe_important[:max_features]
        else:
            display_indices = important_indices
        
        # 创建图表
        plt.figure(figsize=(12, 12))
        
        # 绘制散点图
        plt.scatter(mean_unsafe_important, mean_safe_important, alpha=0.6, s=20)
        
        # 绘制对角线
        min_val = min(mean_safe_important.min(), mean_unsafe_important.min())
        max_val = max(mean_safe_important.max(), mean_unsafe_important.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x')
        
        # 设置坐标轴
        plt.xlabel('Mean Activation on Unsafe Neighbors')
        plt.ylabel('Mean Activation on Safe Neighbors')
        plt.title(f'Sample {sample_idx}: Feature Separation Scatter Plot (Top {len(display_indices)} Features)')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, 'feature_separation_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_feature_separation_frequency_plot(self, neighbor_data: Dict, sample_idx: int, sample_dir: str):
        """创建特征分离度频率图（散点图形式）"""
        safe_features = neighbor_data['safe_neighbor_features']
        unsafe_features = neighbor_data['unsafe_neighbor_features']
        important_indices = neighbor_data['important_feature_indices']
        
        # 数据预处理
        activation_threshold = 0.0
        
        # 筛选重要特征
        safe_features_important = safe_features[:, important_indices]
        unsafe_features_important = unsafe_features[:, important_indices]
        
        # 计算激活频率
        safe_activation_counts = np.sum(safe_features_important > activation_threshold, axis=0)
        unsafe_activation_counts = np.sum(unsafe_features_important > activation_threshold, axis=0)
        
        # 使用固定的重要特征维度（按排名选择前500个）
        max_features = 500
        if len(important_indices) > max_features:
            # 选择前max_features个重要特征（按重要性排名）
            display_indices = important_indices[:max_features]
            safe_activation_counts = safe_activation_counts[:max_features]
            unsafe_activation_counts = unsafe_activation_counts[:max_features]
        else:
            display_indices = important_indices
        
        # 创建图表
        plt.figure(figsize=(12, 12))
        
        # 绘制散点图
        plt.scatter(unsafe_activation_counts, safe_activation_counts, alpha=0.6, s=30)
        
        # 绘制对角线
        max_count = max(safe_activation_counts.max(), unsafe_activation_counts.max())
        plt.plot([0, max_count], [0, max_count], 'r--', alpha=0.7, label='y=x')
        
        # 设置坐标轴
        plt.xlabel('Activation Count on Unsafe Neighbors')
        plt.ylabel('Activation Count on Safe Neighbors')
        plt.title(f'Sample {sample_idx}: Feature Activation Frequency Scatter Plot (Top {len(display_indices)} Features)')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, 'feature_separation_frequency.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def run_complete_analysis(self, analysis_top_k: int = 10, 
                            nearest_neighbor_k: int = 10,
                            top_freq: float = 0.35, 
                            top_value: float = 0.35, 
                            filter_type: str = "mix",
                            use_confidence: bool = True,
                            calculation_mode: str = "neighbor_based",
                            use_caa_norm: bool = False):
        """运行完整分析"""
        print("=== 开始完整分析 ===")
        
        # 加载生成结果并分析性能差异
        static_results, dynamic_results = self._load_generation_results()
        performance_analysis = self._find_performance_differences(
            static_results, dynamic_results, analysis_top_k, use_confidence)
        
        # 加载静态数据
        print("\n加载静态steer vector和SAE特征...")
        static_data = self._load_static_steer_vector_and_features()
        
        # 选择分析样本
        static_better_indices = [item['index'] for item in performance_analysis['static_better_samples']]
        dynamic_better_indices = [item['index'] for item in performance_analysis['dynamic_better_samples']]
        
        # 生成steer vectors
        print(f"\n为静态更好的{len(static_better_indices)}个样本生成steer vectors...")
        static_better_analysis = self._generate_steer_vectors_for_analysis(
            static_better_indices, nearest_neighbor_k, top_freq, top_value, filter_type, calculation_mode, use_caa_norm)
        
        print(f"\n为动态更好的{len(dynamic_better_indices)}个样本生成steer vectors...")
        dynamic_better_analysis = self._generate_steer_vectors_for_analysis(
            dynamic_better_indices, nearest_neighbor_k, top_freq, top_value, filter_type, calculation_mode, use_caa_norm)
        
        # 创建可视化和报告
        print("\n创建可视化分析...")
        self._create_visualizations(static_better_analysis, static_data, "static_better")
        self._create_visualizations(dynamic_better_analysis, static_data, "dynamic_better")
        self._generate_analysis_report(performance_analysis, static_better_analysis, dynamic_better_analysis)
        
        print("\n=== 分析完成 ===")
        print(f"所有结果已保存到: {self.output_dir}")
    
    def run_global_comparison_analysis(self, nearest_neighbor_k: int = 10):
        """运行全局对比分析"""
        print("=== 开始全局对比分析 ===")
        
        # 加载生成结果并分析性能差异
        static_results, dynamic_results = self._load_generation_results()
        
        # 筛选出所有判断不一致的样本
        all_inconsistent_samples = self._find_all_inconsistent_samples(static_results, dynamic_results)
        
        print(f"总共找到{len(all_inconsistent_samples)}个判断不一致的样本")
        
        # 分类样本
        static_better_samples = [s for s in all_inconsistent_samples if s['difference'] > 0]
        dynamic_better_samples = [s for s in all_inconsistent_samples if s['difference'] < 0]
        
        print(f"静态更好的样本数: {len(static_better_samples)}")
        print(f"动态更好的样本数: {len(dynamic_better_samples)}")
        
        # 计算全局平均数据
        static_better_data = self._calculate_global_average_data(static_better_samples, nearest_neighbor_k)
        dynamic_better_data = self._calculate_global_average_data(dynamic_better_samples, nearest_neighbor_k)
        
        # 创建全局对比图
        self._create_global_comparison_plots(static_better_data, dynamic_better_data)
        
        print("\n=== 全局对比分析完成 ===")
    
    def _find_all_inconsistent_samples(self, static_results: List[Dict], dynamic_results: List[Dict]) -> List[Dict]:
        """找出所有判断不一致的样本"""
        print("筛选判断不一致的样本...")
        
        # 评估安全分数
        static_scores = self._evaluate_safety_scores(static_results, use_confidence=True)
        dynamic_scores = self._evaluate_safety_scores(dynamic_results, use_confidence=True)
        
        # 找出判断不一致的样本
        inconsistent_samples = []
        for i, (static_score, dynamic_score) in enumerate(zip(static_scores, dynamic_scores)):
            diff = static_score - dynamic_score
            
            # 判断是否为判断不一致的样本
            static_safe = static_score > 0.5
            dynamic_safe = dynamic_score > 0.5
            
            if static_safe != dynamic_safe:
                inconsistent_samples.append({
                    'index': i,
                    'static_score': static_score,
                    'dynamic_score': dynamic_score,
                    'difference': diff,
                    'static_safe': static_safe,
                    'dynamic_safe': dynamic_safe
                })
        
        return inconsistent_samples
    
    def _calculate_global_average_data(self, samples: List[Dict], nearest_neighbor_k: int) -> Dict:
        """计算全局平均数据"""
        if not samples:
            return None
        
        print(f"计算{len(samples)}个样本的全局平均数据...")
        
        # 获取重要特征索引
        important_indices = self._get_important_feature_indices()
        
        # 收集所有样本的邻居特征数据
        all_safe_features = []
        all_unsafe_features = []
        
        for sample in tqdm(samples, desc="处理样本"):
            try:
                # 获取样本的prompt
                test_data = load_safeedit_test_data(model_alias=self.model_alias, apply_chat_format=True)
                prompt = test_data[sample['index']]['prompt']
                
                # 获取查询激活
                query_activation = self._get_query_activation(prompt)
                
                # 准备查询向量
                query_vector = query_activation.detach().cpu().float().numpy().reshape(1, -1)
                
                # 找到最近邻
                safe_query_acts_all = self.safe_query_acts_array[:, 0, :]
                safe_dists_all = np.linalg.norm(safe_query_acts_all - query_vector, axis=1)
                safe_indices = np.argsort(safe_dists_all)[:nearest_neighbor_k]
                
                unsafe_query_acts_all = self.unsafe_query_acts_array[:, 0, :]
                unsafe_dists_all = np.linalg.norm(unsafe_query_acts_all - query_vector, axis=1)
                unsafe_indices = np.argsort(unsafe_dists_all)[:nearest_neighbor_k]
                
                # 获取最近邻样本的SAE特征
                safe_feature_acts = self.safe_avg_sae_features[safe_indices, 0, :]
                unsafe_feature_acts = self.unsafe_avg_sae_features[unsafe_indices, 0, :]
                
                all_safe_features.append(safe_feature_acts)
                all_unsafe_features.append(unsafe_feature_acts)
                
            except Exception as e:
                print(f"处理样本{sample['index']}时出错: {e}")
                continue
        
        if not all_safe_features:
            return None
        
        # 计算全局平均
        all_safe_features = np.array(all_safe_features)  # shape: (n_samples, k, n_features)
        all_unsafe_features = np.array(all_unsafe_features)
        
        # 计算每个样本的平均激活值
        mean_safe_per_sample = np.mean(all_safe_features, axis=1)  # shape: (n_samples, n_features)
        mean_unsafe_per_sample = np.mean(all_unsafe_features, axis=1)
        
        # 计算全局平均
        global_mean_safe = np.mean(mean_safe_per_sample, axis=0)  # shape: (n_features,)
        global_mean_unsafe = np.mean(mean_unsafe_per_sample, axis=0)
        
        # 计算激活频率
        activation_threshold = 0.0
        safe_activation_counts_per_sample = np.sum(all_safe_features > activation_threshold, axis=1)  # shape: (n_samples, k, n_features)
        unsafe_activation_counts_per_sample = np.sum(all_unsafe_features > activation_threshold, axis=1)
        
        # 计算每个样本的平均激活频率
        mean_safe_freq_per_sample = np.mean(safe_activation_counts_per_sample, axis=1)  # shape: (n_samples, n_features)
        mean_unsafe_freq_per_sample = np.mean(unsafe_activation_counts_per_sample, axis=1)
        
        # 计算全局平均频率
        global_mean_safe_freq = np.mean(mean_safe_freq_per_sample, axis=0)  # shape: (n_features,)
        global_mean_unsafe_freq = np.mean(mean_unsafe_freq_per_sample, axis=0)
        
        return {
            'important_indices': important_indices,
            'global_mean_safe': global_mean_safe,
            'global_mean_unsafe': global_mean_unsafe,
            'global_mean_safe_freq': global_mean_safe_freq,
            'global_mean_unsafe_freq': global_mean_unsafe_freq,
            'sample_count': len(samples)
        }
    
    def _create_global_comparison_plots(self, static_better_data: Dict, dynamic_better_data: Dict):
        """创建全局对比图"""
        print("创建全局对比图...")
        
        # 创建第一幅图：激活差值热力图对比
        self._create_global_activation_difference_heatmap(static_better_data, dynamic_better_data)
        
        # 创建第二幅图：激活频率散点图对比
        self._create_global_frequency_scatter_plot(static_better_data, dynamic_better_data)
    
    def _create_global_activation_difference_heatmap(self, static_better_data: Dict, dynamic_better_data: Dict):
        """创建全局SAE特征差值热力图对比"""
        if static_better_data is None or dynamic_better_data is None:
            print("警告：数据不足，无法创建全局SAE特征差值热力图")
            return
        
        # 选择2000个特征进行显示（100*20矩阵）
        max_features = 2000
        
        # 准备数据
        static_important = static_better_data['important_indices'][:max_features]
        dynamic_important = dynamic_better_data['important_indices'][:max_features]
        
        # 使用numpy索引而不是列表索引
        static_safe_activations = static_better_data['global_mean_safe'][np.array(static_important)]
        static_unsafe_activations = static_better_data['global_mean_unsafe'][np.array(static_important)]
        static_differences = static_safe_activations - static_unsafe_activations
        
        dynamic_safe_activations = dynamic_better_data['global_mean_safe'][np.array(dynamic_important)]
        dynamic_unsafe_activations = dynamic_better_data['global_mean_unsafe'][np.array(dynamic_important)]
        dynamic_differences = dynamic_safe_activations - dynamic_unsafe_activations
        
        # 获取静态steer vector的SAE特征
        static_data = self._load_static_steer_vector_and_features()
        static_sae_features = static_data["sae_features"]
        static_sae_important = static_sae_features[static_important]
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 24))
        
        # 创建红白蓝颜色映射
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['blue', 'white', 'red']
        cmap = LinearSegmentedColormap.from_list('red_white_blue', colors, N=100)
        
        # 计算颜色范围
        max_abs_diff = max(
            abs(static_differences.min()), abs(static_differences.max()),
            abs(dynamic_differences.min()), abs(dynamic_differences.max()),
            abs(static_sae_important.min()), abs(static_sae_important.max())
        )
        vmin, vmax = -max_abs_diff, max_abs_diff
        
        # 绘制静态更好的SAE特征差值热力图
        static_plot_data = static_differences.reshape(100, 20)  # 重塑为100*20矩阵
        im1 = ax1.imshow(static_plot_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax1.set_title(f'Static Better Samples (n={static_better_data["sample_count"]})\nSAE Feature Difference (Safe - Unsafe)')
        ax1.set_ylabel('Feature Row Index')
        ax1.set_xlabel('Feature Column Index')
        
        # 绘制动态更好的SAE特征差值热力图
        dynamic_plot_data = dynamic_differences.reshape(100, 20)  # 重塑为100*20矩阵
        im2 = ax2.imshow(dynamic_plot_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax2.set_title(f'Dynamic Better Samples (n={dynamic_better_data["sample_count"]})\nSAE Feature Difference (Safe - Unsafe)')
        ax2.set_ylabel('Feature Row Index')
        ax2.set_xlabel('Feature Column Index')
        
        # 绘制静态steer vector SAE特征热力图
        static_sae_plot_data = static_sae_important.reshape(100, 20)  # 重塑为100*20矩阵
        im3 = ax3.imshow(static_sae_plot_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax3.set_title('Static Steer Vector SAE Features')
        ax3.set_ylabel('Feature Row Index')
        ax3.set_xlabel('Feature Column Index')
        
        # 添加颜色条
        cbar = plt.colorbar(im1, ax=[ax1, ax2, ax3], shrink=0.8)
        cbar.set_label('SAE Feature Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'global_sae_feature_difference_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"全局SAE特征差值热力图已保存到: {os.path.join(self.output_dir, 'global_sae_feature_difference_heatmap.png')}")
    
    def _create_global_frequency_scatter_plot(self, static_better_data: Dict, dynamic_better_data: Dict):
        """创建全局激活频率散点图对比"""
        if static_better_data is None or dynamic_better_data is None:
            print("警告：数据不足，无法创建全局激活频率散点图")
            return
        
        # 选择更多特征进行显示
        max_features = 1000
        
        # 准备数据
        static_important = static_better_data['important_indices'][:max_features]
        dynamic_important = dynamic_better_data['important_indices'][:max_features]
        
        # 使用numpy索引而不是列表索引
        static_safe_freq = static_better_data['global_mean_safe_freq'][np.array(static_important)]
        static_unsafe_freq = static_better_data['global_mean_unsafe_freq'][np.array(static_important)]
        
        dynamic_safe_freq = dynamic_better_data['global_mean_safe_freq'][np.array(dynamic_important)]
        dynamic_unsafe_freq = dynamic_better_data['global_mean_unsafe_freq'][np.array(dynamic_important)]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 绘制静态更好的散点图
        ax1.scatter(static_unsafe_freq, static_safe_freq, alpha=0.6, s=20)
        max_freq = max(static_safe_freq.max(), static_unsafe_freq.max())
        ax1.plot([0, max_freq], [0, max_freq], 'r--', alpha=0.7, label='y=x')
        ax1.set_xlabel('Mean Activation Frequency on Unsafe Neighbors')
        ax1.set_ylabel('Mean Activation Frequency on Safe Neighbors')
        ax1.set_title(f'Static Better Samples (n={static_better_data["sample_count"]})\nFeature Activation Frequency Scatter Plot')
        ax1.legend()
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)
        
        # 绘制动态更好的散点图
        ax2.scatter(dynamic_unsafe_freq, dynamic_safe_freq, alpha=0.6, s=20)
        max_freq = max(dynamic_safe_freq.max(), dynamic_unsafe_freq.max())
        ax2.plot([0, max_freq], [0, max_freq], 'r--', alpha=0.7, label='y=x')
        ax2.set_xlabel('Mean Activation Frequency on Unsafe Neighbors')
        ax2.set_ylabel('Mean Activation Frequency on Safe Neighbors')
        ax2.set_title(f'Dynamic Better Samples (n={dynamic_better_data["sample_count"]})\nFeature Activation Frequency Scatter Plot')
        ax2.legend()
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'global_frequency_scatter_plot.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"全局激活频率散点图已保存到: {os.path.join(self.output_dir, 'global_frequency_scatter_plot.png')}")
    
    
@torch.no_grad()
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Steer Vector性能分析")
    
    # 必需参数
    parser.add_argument("--static_result_path", type=str, default="/home/wangxin/projects/safety_steer/results/generations/gemma-2-9b-it/safeedit/sae_transformer_lens/safeedit_20.json",
                       help="静态生成结果文件路径")
    parser.add_argument("--dynamic_result_path", type=str, default="results/generations/gemma-2-9b-it/safeedit/flexible_sae_querylevel/safeedit_20/topk_5_no_norm_freq35p_value35p_mix_search_activation_mix.json",
                       help="动态生成结果文件路径")
    parser.add_argument("--safety_classifier_dir", type=str, default="/home/wangxin/models/zjunlp/SafeEdit-Safety-Classifier",
                       help="安全分类器目录路径")
    
    # 可选参数
    parser.add_argument("--model_alias", type=str, default="gemma-2-9b-it",
                       help="模型别名")
    parser.add_argument("--output_dir", type=str, default="steer_analysis_results",
                       help="输出目录")
    parser.add_argument("--analysis_top_k", type=int, default=3,
                       help="性能差异分析的top-k样本数")
    parser.add_argument("--nearest_neighbor_k", type=int, default=5,
                       help="最近邻搜索的k值")
    parser.add_argument("--top_freq", type=float, default=0.35,
                       help="频率阈值")
    parser.add_argument("--top_value", type=float, default=0.35,
                       help="激活值阈值")
    parser.add_argument("--filter_type", type=str, default="mix", choices=["freq", "value", "mix"],
                       help="特征筛选类型")
    parser.add_argument("--use_confidence", default=True,
                       help="使用置信度而不是分类结果")
    parser.add_argument("--calculation_mode", type=str, default="fixed_features", 
                       choices=["neighbor_based", "fixed_features"],
                       help="重要SAE维度计算模式: neighbor_based(基于最近邻), fixed_features(基于全局固定特征)")
    parser.add_argument("--use_caa_norm", action="store_true", default=False,
                       help="是否使用CAA向量对steer vector进行范数归一化")
    parser.add_argument("--analysis_mode", type=str, default="global", 
                       choices=["individual", "global"],
                       help="分析模式: individual(个体样本分析), global(全局对比分析)")
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = SteerVectorAnalyzer(
        static_result_path=args.static_result_path,
        dynamic_result_path=args.dynamic_result_path,
        safety_classifier_dir=args.safety_classifier_dir,
        model_alias=args.model_alias,
        output_dir=args.output_dir
    )
    
    # 根据分析模式运行不同的分析
    if args.analysis_mode == "individual":
        analyzer.run_complete_analysis(
            analysis_top_k=args.analysis_top_k,
            nearest_neighbor_k=args.nearest_neighbor_k,
            top_freq=args.top_freq,
            top_value=args.top_value,
            filter_type=args.filter_type,
            use_confidence=args.use_confidence,
            calculation_mode=args.calculation_mode,
            use_caa_norm=args.use_caa_norm
        )
    elif args.analysis_mode == "global":
        analyzer.run_global_comparison_analysis(
            nearest_neighbor_k=args.nearest_neighbor_k
        )


if __name__ == "__main__":
    main()