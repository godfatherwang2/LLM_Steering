
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

### 2. FAISS索引构建 (build_faiss_index.py)

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

### 3. 安全生成 (flexiable_caa_safety.py)

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

## 使用流程

1. **缓存激活值**：运行 `activation_cache.py` 缓存模型激活值
2. **构建索引**：运行 `build_faiss_index.py` 构建FAISS索引
3. **生成文本**：运行 `flexiable_caa_safety.py` 进行安全文本生成

## 输出文件

- 激活值缓存：`dataset/cached_activations/`
- FAISS索引：`dataset/cached_activations/{model_alias}_{dataset}/`
- 生成结果：`results/generations/{model_alias}/{test_dataset}/flexible_caa/`
