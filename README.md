# Safety Steer - 安全引导生成框架

本项目实现了基于激活值操控和SAE特征引导的安全文本生成框架，支持多种引导策略和距离度量方式。

## 主要脚本

### 1. 激活值缓存 (activation_cache.py)

缓存模型在特定数据集上的激活值，为后续的FAISS索引构建做准备。

```bash
python activation_cache.py \
    --model_alias gemma-2-9b-it \
    --dataset safeedit \
    --tokens_to_cache "<start_of_turn>model\n" \
    --batch_size 16 \
    --layers "24"
```

**参数说明：**
- `--model_alias`: 模型别名
- `--dataset`: 数据集名称 (safeedit/PKU-SafeRLHF/pile)
- `--tokens_to_cache`: 回答部分分隔符
- `--batch_size`: 批处理大小
- `--layers`: 要缓存的层号，用逗号分隔

### 2. SAE特征缓存 (sae_feature_cache.py)

缓存模型的SAE特征，为SAE特征引导做准备。

```bash
python sae_feature_cache.py \
    --model_alias gemma-2-9b-it \
    --dataset safeedit \
    --layer 20 \
    --batch_size 16
```

**参数说明：**
- `--model_alias`: 模型别名
- `--dataset`: 数据集名称
- `--layer`: 要缓存的层号
- `--batch_size`: 批处理大小

### 3. FAISS索引构建 (build_faiss_index.py)

基于缓存的激活值构建FAISS索引，用于快速相似性搜索。

```bash
python build_faiss_index.py \
    --model_alias gemma-2-9b-it \
    --dataset safeedit \
    --shard_size 5
```

**参数说明：**
- `--model_alias`: 模型别名
- `--dataset`: 数据集名称

### 4. SAE特征索引构建 (build_faiss_index_sae.py)

基于缓存的SAE特征构建FAISS索引，用于SAE特征相似性搜索。

```bash
python build_faiss_index_sae.py \
    --model_alias gemma-2-9b-it \
    --dataset safeedit \
    --layer 20 \
    --shard_size 4050
```

**参数说明：**
- `--model_alias`: 模型别名
- `--dataset`: 数据集名称
- `--layer`: SAE特征对应的层号
- `--shard_size`: 数据集分片大小

### 5. SAE特征分析 (sae_feature_analyse.py)

分析SAE特征重要性并计算静态引导向量。

```bash
python sae_feature_analyse.py \
    --model_alias gemma-2-9b-it \
    --dataset safeedit \
    --layer 20 \
    --select_type act_and_fre_trim \
    --trim 0.35 \
    --activation_threshold 0.0
```

**参数说明：**
- `--model_alias`: 模型别名
- `--dataset`: 数据集名称
- `--layer`: 要分析的层号
- `--select_type`: 特征选择类型 (act_trim/fre_trim/act_and_fre_trim)
- `--trim`: 特征筛选比例
- `--activation_threshold`: 激活阈值

### 6. 激活值引导生成 (flexiable_caa_safety.py)

使用激活值操控进行安全文本生成。

```bash
python generate/flexiable_caa_safety.py \
    --model_alias gemma-2-9b-it \
    --dataset safeedit \
    --shared_size 4050 \
    --layer 24 \
    --top_k 10 \
    --multiplier 1.0 \
    --max_new_tokens 50 \
    --norm_magnitude
```

**参数说明：**
- `--model_alias`: 模型别名
- `--dataset`: 源数据集名称
- `--shared_size`: 源数据集样本数量(用于定位)
- `--layer`: 要干预的层号
- `--top_k`: FAISS检索的近邻数量
- `--multiplier`: 引导向量强度系数
- `--max_new_tokens`: 最大生成token数
- `--norm_magnitude`: 是否进行范数归一化(相对于CAA)

### 7. SAE特征引导生成 (flexiable_sae_safety_querylevel.py)

使用SAE特征引导进行安全文本生成，支持多种搜索模式和距离度量。

```bash
python generate/flexiable_sae_safety_querylevel.py \
    --model_alias gemma-2-9b-it \
    --dataset safeedit \
    --shared_size 4050 \
    --layer 20 \
    --sae_layer 20 \
    --top_k 10 \
    --multiplier 1.0 \
    --max_new_tokens 50 \
    --norm_magnitude \
    --search_mode activation \
    --filter_type mix \
    --top_freq 0.35 \
    --top_value 0.35 \
    --steer_pos all \
    --distance_metric euclidean \
    --use_softmax_weight \
    --test_dataset safeedit
```

**参数说明：**
- `--model_alias`: 模型别名
- `--dataset`: 源数据集名称
- `--shared_size`: 源数据集样本数量
- `--layer`: 要干预的层号
- `--sae_layer`: SAE特征对应的层号
- `--top_k`: FAISS检索的近邻数量
- `--multiplier`: 引导向量强度系数
- `--max_new_tokens`: 最大生成token数
- `--norm_magnitude`: 是否进行范数归一化(相对于CAA)
- `--search_mode`: 搜索模式 (activation/hybrid/fixed_features)
- `--filter_type`: 特征筛选类型 (freq/value/mix)
- `--top_freq`: 频率差异前N%特征比例
- `--top_value`: 激活值差异前N%特征比例
- `--steer_pos`: 引导位置 (all/current)
- `--distance_metric`: 距离度量 (euclidean/cosine)
- `--use_softmax_weight`: 是否使用softmax加权 (仅适用于activation模式)
- `--test_dataset`: 测试数据集名称

## 使用流程

### 激活值引导流程
1. **缓存激活值**：运行 `activation_cache.py` 缓存模型激活值
2. **构建索引**：运行 `build_faiss_index.py` 构建FAISS索引
3. **生成文本**：运行 `flexiable_caa_safety.py` 进行安全文本生成

### SAE特征引导流程
1. **缓存SAE特征**：运行 `sae_feature_cache.py` 缓存SAE特征
2. **构建SAE索引**：运行 `build_faiss_index_sae.py` 构建SAE特征索引
3. **分析特征重要性**：运行 `sae_feature_analyse.py` 计算静态引导向量
4. **生成文本**：运行 `flexiable_sae_safety_querylevel.py` 进行SAE特征引导生成

