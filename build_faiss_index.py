import numpy as np
import faiss
import os
import argparse
import json
from tqdm import tqdm

# 和 cache_activation_database.py 中保持一致
DEFAULT_CACHE_DIR = os.path.join("dataset/cached_activations_safeedit_response_top5")

def build_and_save_faiss_index(
    model_alias: str,
    dataset_label: str,
    hook_point: str,
    nlist: int = 1024,
    m: int = 8,
    bits: int = 8
):
    """
    为指定层的激活值构建FAISS索引并保存到磁盘。
    此版本会先读取metadata.json文件以获取正确的shape。
    """
    print(f"--- 开始为 {dataset_label}/{hook_point} 构建FAISS索引 ---")

    # 1. 确定文件路径
    foldername = os.path.join(DEFAULT_CACHE_DIR, model_alias, dataset_label)
    keys_filename = f"keys_{hook_point}.dat"
    keys_filepath = os.path.join(foldername, keys_filename)
    metadata_filepath = os.path.join(foldername, "metadata.json")

    # 2. 加载元数据以获取 shape
    if not os.path.exists(metadata_filepath):
        print(f"错误: 元数据文件未找到于 {metadata_filepath}，无法确定数据形状。")
        return
        
    print(f"[1/5] 加载元数据: {metadata_filepath}")
    with open(metadata_filepath, 'r') as f:
        metadata = json.load(f)
    
    shape_from_metadata = tuple(metadata['shape'])
    dtype_from_metadata = metadata['dtype']
    
    # 3. 使用元数据加载激活值文件
    print("[2/5] 加载激活值文件 (memmap)...")
    if not os.path.exists(keys_filepath):
        print(f"错误：激活值文件未找到于 {keys_filepath}")
        return
        
    keys_to_index = np.memmap(keys_filepath, dtype=dtype_from_metadata, mode='r', shape=shape_from_metadata)
    
    # Reshape: 将 (N, M, D) -> (N * M, D) 以便FAISS处理
    if keys_to_index.ndim == 3:
        num_vectors = keys_to_index.shape[0] * keys_to_index.shape[1]
        d_model = keys_to_index.shape[2]
        keys_to_index = keys_to_index.reshape(num_vectors, d_model)
    
    keys_to_index = np.ascontiguousarray(keys_to_index)
    total_tokens, d_model = keys_to_index.shape
    print(f"数据准备完毕，共 {total_tokens} 个维度为 {d_model} 的向量。")

    # 4. 构建、训练、添加并保存索引
    print("[3/5] 构建并训练FAISS索引...")
    quantizer = faiss.IndexFlatL2(d_model)
    index = faiss.IndexIVFPQ(quantizer, d_model, nlist, m, bits)

    num_train_samples = min(total_tokens, nlist * 100)
    train_indices = np.random.choice(total_tokens, num_train_samples, replace=False)
    
    with tqdm(total=1, desc="训练中") as pbar:
        index.train(keys_to_index[train_indices])
        pbar.update(1)

    print("[4/5] 将所有向量添加到索引中...")
    batch_size_add = 100000
    for i in tqdm(range(0, total_tokens, batch_size_add), desc="添加向量"):
        index.add(keys_to_index[i:i+batch_size_add])
    
    index_filename = f"faiss_index_{keys_filename}.index"
    save_path = os.path.join(foldername, index_filename)
    print(f"[5/5] 正在保存索引到: {save_path}")
    faiss.write_index(index, save_path)
    
    print(f"--- 索引构建完成！ ---")

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
    args.label = "unsafe"
    build_and_save_faiss_index(
        model_alias=args.model_alias,
        dataset_label=args.label,
        hook_point=args.hook_point
    )