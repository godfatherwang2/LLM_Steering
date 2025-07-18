# build_faiss_index.py

import numpy as np
import faiss
import os
import argparse
from tqdm import tqdm

# 和 activation_cache.py 中保持一致
DEFAULT_CACHE_DIR = os.path.join("dataset/cached_activations_safeedit_response_top5")

def build_and_save_faiss_index(
    model_alias: str,
    dataset_label: str, # 'safe' or 'unsafe'
    hook_point: str,
    # --- FAISS 参数 ---
    nlist: int = 1024,      # 聚类中心的数量
    m: int = 8,             # 乘积量化的子向量数量
    bits: int = 8           # 每个子向量的量化位数
):
    """
    为指定层的激活值构建FAISS索引并保存到磁盘。
    """
    print(f"--- 开始为 {dataset_label}/{hook_point} 构建FAISS索引 ---")

    # 1. 确定激活值文件的路径
    foldername = os.path.join(DEFAULT_CACHE_DIR, model_alias, dataset_label)
    keys_filename = f"keys_{hook_point}.dat"
    keys_filepath = os.path.join(foldername, keys_filename)

    if not os.path.exists(keys_filepath):
        print(f"错误：激活值文件未找到于 {keys_filepath}")
        return

    # 2. 使用 memmap 加载激活值文件，避免消耗大量内存
    print("加载激活值文件 (memmap)...")
    # memmap 不需要知道完整形状，它会从文件头读取
    keys_to_index = np.memmap(keys_filepath, dtype='float32', mode='r')
    
    # 从文件大小推断形状
    # 假设形状是 (total_tokens, n_layers, d_model)
    # FAISS 需要 (N, D) 的二维数组，所以我们需要 reshape
    d_model = keys_to_index.shape[-1]
    # 我们将 (total_tokens, n_layers, d_model) -> (total_tokens * n_layers, d_model)
    # 注意：这意味着我们混合了所有层的激活值。如果想为每层单独建索引，逻辑需要调整。
    # 根据您之前的设计，您似乎是想在一个索引中包含所有指定层的激活值。
    if keys_to_index.ndim == 3:
        print(f"原始数据形状: {keys_to_index.shape}")
        num_vectors = keys_to_index.shape[0] * keys_to_index.shape[1]
        keys_to_index = keys_to_index.reshape(num_vectors, d_model)
        print(f"Reshape后的数据形状: {keys_to_index.shape}")

    # 确保数据是C-contiguous的，这对FAISS是推荐的
    keys_to_index = np.ascontiguousarray(keys_to_index)
    total_tokens, d_model = keys_to_index.shape
    
    # 3. 构建FAISS索引 (使用 IndexIVFPQ，适合大规模数据)
    print("构建FAISS索引...")
    quantizer = faiss.IndexFlatL2(d_model)
    index = faiss.IndexIVFPQ(quantizer, d_model, nlist, m, bits)

    # 4. 训练索引
    print("训练索引 (这可能需要一些时间)...")
    # 从所有keys中随机抽样一部分进行训练
    np.random.seed(42)
    num_train_samples = min(total_tokens, nlist * 100)
    train_indices = np.random.choice(total_tokens, num_train_samples, replace=False)
    training_vectors = keys_to_index[train_indices]

    index.train(training_vectors)
    print("训练完成。")

    # 5. 将所有keys分批添加到索引中
    print("将所有向量添加到索引中...")
    batch_size_add = 100000
    for i in tqdm(range(0, total_tokens, batch_size_add), desc="添加向量"):
        index.add(keys_to_index[i:i+batch_size_add])
    
    print("所有向量添加完毕。")

    # 6. 保存索引到磁盘 (遵循 steer_generate.py 的命名约定)
    # steer_generate.py 期待 faiss_index_{layer_str}.index
    # 这里的 layer_str 实际上是 keys_{hook_point}.dat
    index_filename = f"faiss_index_{keys_filename}.index"
    save_path = os.path.join(foldername, index_filename)
    print(f"正在保存索引到: {save_path}")
    faiss.write_index(index, save_path)
    print(f"--- 索引构建完成并已保存！ ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='为缓存的激活值构建FAISS索引')
    parser.add_argument('--model_alias', type=str, default="gemma-2-9b-it", help='模型别名')
    parser.add_argument('--label', type=str, default="safe", help='数据的标签 ("safe" 或 "unsafe")')
    parser.add_argument('--hook_point', type=str, default="block_output", help='您缓存激活值时使用的hook_point (例如 "block_output")')

    args = parser.parse_args()
    build_and_save_faiss_index(
        model_alias=args.model_alias,
        dataset_label=args.label,
        hook_point=args.hook_point
    )
    parser.label = "unsafe"
    build_and_save_faiss_index(
        model_alias=args.model_alias,
        dataset_label=args.label,
        hook_point=args.hook_point
    )