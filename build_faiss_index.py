import numpy as np
import faiss
import os
import argparse
import json
from tqdm import tqdm

# This should match the directory structure of your caching script
DEFAULT_CACHE_DIR = os.path.join("dataset/cached_activations")

def build_faiss_index_from_cache(
    model_alias: str,
    dataset_name: str,
    label: str,
    shard_size: int, # This is part of the folder name in your script
    # --- FAISS Parameters ---
    nlist: int = 1024,
    m: int = 8,
    bits: int = 8
):
    """
    Builds separate FAISS indexes for each layer's cached activations,
    based on the specific output of the provided activation_cache.py script.
    """
    print(f"--- Starting FAISS index build for {model_alias}/{dataset_name}/{label} ---")

    # 1. Construct the path to the cache directory
    # This must exactly match the folder naming convention from your caching script
    foldername = f"{DEFAULT_CACHE_DIR}/{model_alias}_{dataset_name}/label_{label}_shard_size_{shard_size}"
    metadata_filepath = os.path.join(foldername, "metadata.json")

    # 2. Load the metadata file to get parameters
    if not os.path.exists(metadata_filepath):
        print(f"FATAL ERROR: Metadata file not found at {metadata_filepath}")
        print("Please ensure you have run the activation_cache.py script first.")
        return
        
    print(f"Loading metadata from: {metadata_filepath}")
    with open(metadata_filepath, 'r') as f:
        metadata = json.load(f)
    
    # Extract necessary info from metadata
    layers_to_index = metadata['layers_cached']
    n_layers, total_tokens, d_model = metadata['activations_original_shape']
    
    print(f"Found {len(layers_to_index)} layers to index: {layers_to_index}")

    # 3. Iterate through each layer and build a separate index
    for layer_num in layers_to_index:
        print("\n" + "="*50)
        print(f"Processing Layer {layer_num}")
        print("="*50)

        # a. Define paths for the current layer
        keys_filename = f"acts_{layer_num}.dat"
        keys_filepath = os.path.join(foldername, keys_filename)

        if not os.path.exists(keys_filepath):
            print(f"Warning: Skipping layer {layer_num}, file not found at {keys_filepath}")
            continue

        # b. Load the 2D memmap file for the current layer
        print(f"[1/4] Loading memmap file: {keys_filename}")
        layer_keys_shape = (total_tokens, d_model)
        layer_keys = np.memmap(keys_filepath, dtype='float32', mode='r', shape=layer_keys_shape)
        layer_keys = np.ascontiguousarray(layer_keys)

        # c. Build and Train the FAISS index for this layer
        print(f"[2/4] Building and training index for Layer {layer_num}...")
        quantizer = faiss.IndexFlatL2(d_model)
        index = faiss.IndexIVFPQ(quantizer, d_model, nlist, m, bits)
        
        num_train_samples = total_tokens #min(total_tokens, nlist * 100)
        train_indices = np.random.choice(total_tokens, num_train_samples, replace=False)
        
        index.train(layer_keys[train_indices])

        # d. Add all vectors for this layer to the index
        print(f"[3/4] Adding {total_tokens} vectors to the index for Layer {layer_num}...")
        batch_size_add = 100000
        for i in range(0, total_tokens, batch_size_add):
            index.add(layer_keys[i:i+batch_size_add])
        print(f"为 Layer {layer_num} 创建 Direct Map...")
        index.make_direct_map()
        # e. Save the final index file with a layer-specific name
        index_filename = f"faiss_index_layer_{layer_num}.index"
        save_path = os.path.join(foldername, index_filename)
        print(f"[4/4] Saving index to: {save_path}")
        faiss.write_index(index, save_path)
        print(f"--- Successfully created index for Layer {layer_num} ---")

    print("\n--- All indexing tasks complete! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build layer-specific FAISS indexes from cached activations')
    parser.add_argument('--model_alias', type=str, default="gemma-2-9b-it", help='Alias of the model used for caching')
    parser.add_argument('--dataset', type=str,default="safeedit", help='Dataset name used for caching')
    #parser.add_argument('--label', type=str,default="safe", help='Data label used for caching (e.g., "safe", "unsafe")')
    parser.add_argument('--shard_size', type=int, default=5, help='Shard size used in the folder name during caching')
    
    args = parser.parse_args()
    args.label = "safe"
    build_faiss_index_from_cache(
        model_alias=args.model_alias,
        dataset_name=args.dataset,
        label=args.label,
        shard_size=args.shard_size
    )
    args.label = "unsafe"
    build_faiss_index_from_cache(
        model_alias=args.model_alias,
        dataset_name=args.dataset,
        label=args.label,
        shard_size=args.shard_size
    )

