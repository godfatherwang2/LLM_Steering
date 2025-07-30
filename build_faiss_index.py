import numpy as np
import faiss
import os
import argparse
import json
from tqdm import tqdm

# This should match the directory structure of your caching script
DEFAULT_CACHE_DIR = os.path.join("dataset/cached_activations")
DEFAULT_SAE_CACHE_DIR = os.path.join("dataset/cached_activations_sae")

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

def build_faiss_index_from_cache_sae(
    model_alias: str,
    dataset_name: str,
    label: str,
    shard_size: int, # This is part of the folder name in your script
    nlist: int = 2048,
    m: int = 8,
    bits: int = 8
):
    """
    Builds separate FAISS indexes for each layer's cached activations and SAE features,
    based on the specific output of the provided activation_cache.py script.
    """
    print(f"--- Starting FAISS index build for {model_alias}/{dataset_name}/{label} ---")

    # 1. Construct the path to the cache directory
    # This must exactly match the folder naming convention from your caching script
    foldername = f"{DEFAULT_SAE_CACHE_DIR}/{model_alias}_{dataset_name}/label_{label}_shard_size_{shard_size}"
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
    sae_features_dim = metadata.get('sae_features_dim', None)
    
    print(f"Found {len(layers_to_index)} layers to index: {layers_to_index}")
    print(f"Original activation dimension: {d_model}")
    if sae_features_dim:
        print(f"SAE features dimension: {sae_features_dim}")

    # 3. Iterate through each layer and build separate indexes for acts and sae
    for layer_num in layers_to_index:
        print("\n" + "="*50)
        print(f"Processing Layer {layer_num}")
        print("="*50)

        # ===== Process Original Activations (acts) =====
        print(f"\n--- Processing Original Activations for Layer {layer_num} ---")
        
        # a. Define paths for the current layer's acts
        acts_filename = f"acts_{layer_num}.dat"
        acts_filepath = os.path.join(foldername, acts_filename)

        if os.path.exists(acts_filepath):
            # b. Load the 2D memmap file for the current layer's acts
            print(f"[1/4] Loading memmap file: {acts_filename}")
            acts_shape = (total_tokens, d_model)
            acts_data = np.memmap(acts_filepath, dtype='float32', mode='r', shape=acts_shape)
            acts_data = np.ascontiguousarray(acts_data)

            # c. Build and Train the FAISS index for acts
            print(f"[2/4] Building and training index for Layer {layer_num} acts...")
            quantizer = faiss.IndexFlatL2(d_model)
            index = faiss.IndexIVFPQ(quantizer, d_model, nlist, m, bits)
            
            num_train_samples = min(total_tokens, nlist * 100)
            train_indices = np.random.choice(total_tokens, num_train_samples, replace=False)
            
            index.train(acts_data[train_indices])

            # d. Add all vectors for this layer to the index
            print(f"[3/4] Adding {total_tokens} vectors to the acts index for Layer {layer_num}...")
            batch_size_add = 100000
            for i in range(0, total_tokens, batch_size_add):
                index.add(acts_data[i:i+batch_size_add])
            print(f"为 Layer {layer_num} acts 创建 Direct Map...")
            index.make_direct_map()
            
            # e. Save the final index file with a layer-specific name
            acts_index_filename = f"faiss_index_acts_layer_{layer_num}.index"
            acts_save_path = os.path.join(foldername, acts_index_filename)
            print(f"[4/4] Saving acts index to: {acts_save_path}")
            faiss.write_index(index, acts_save_path)
            print(f"--- Successfully created acts index for Layer {layer_num} ---")
        else:
            print(f"Warning: Skipping acts for layer {layer_num}, file not found at {acts_filepath}")

        # ===== Process SAE Features (sae) =====
        print(f"\n--- Processing SAE Features for Layer {layer_num} ---")
        
        # a. Define paths for the current layer's sae
        sae_filename = f"sae_{layer_num}.dat"
        sae_filepath = os.path.join(foldername, sae_filename)

        if os.path.exists(sae_filepath) and sae_features_dim is not None:
            # b. Load the 2D memmap file for the current layer's sae
            print(f"[1/4] Loading memmap file: {sae_filename}")
            sae_shape = (total_tokens, sae_features_dim)
            sae_data = np.memmap(sae_filepath, dtype='float32', mode='r', shape=sae_shape)
            sae_data = np.ascontiguousarray(sae_data)

            # c. Build and Train the FAISS index for sae with larger budget
            print(f"[2/4] Building and training index for Layer {layer_num} sae...")
            
            # For SAE features, use larger budget due to higher dimensionality
            sae_nlist = min(4096, max(nlist * 2, sae_features_dim // 4))  # Larger nlist for SAE
            sae_m = min(16, max(m * 2, sae_features_dim // 512))  # Larger m for SAE
            sae_bits = min(12, max(bits * 1.5, 8))  # More bits for SAE
            
            print(f"SAE index parameters: nlist={sae_nlist}, m={sae_m}, bits={sae_bits}")
            
            quantizer = faiss.IndexFlatL2(sae_features_dim)
            index = faiss.IndexIVFPQ(quantizer, sae_features_dim, sae_nlist, sae_m, sae_bits)
            
            num_train_samples = min(total_tokens, sae_nlist * 100)
            train_indices = np.random.choice(total_tokens, num_train_samples, replace=False)
            
            index.train(sae_data[train_indices])

            # d. Add all vectors for this layer to the index
            print(f"[3/4] Adding {total_tokens} vectors to the sae index for Layer {layer_num}...")
            batch_size_add = 50000  # Smaller batch size for SAE due to higher dimensionality
            for i in range(0, total_tokens, batch_size_add):
                index.add(sae_data[i:i+batch_size_add])
            print(f"为 Layer {layer_num} sae 创建 Direct Map...")
            index.make_direct_map()
            
            # e. Save the final index file with a layer-specific name
            sae_index_filename = f"faiss_index_sae_layer_{layer_num}.index"
            sae_save_path = os.path.join(foldername, sae_index_filename)
            print(f"[4/4] Saving sae index to: {sae_save_path}")
            faiss.write_index(index, sae_save_path)
            print(f"--- Successfully created sae index for Layer {layer_num} ---")
        else:
            if not os.path.exists(sae_filepath):
                print(f"Warning: Skipping sae for layer {layer_num}, file not found at {sae_filepath}")
            elif sae_features_dim is None:
                print(f"Warning: Skipping sae for layer {layer_num}, sae_features_dim not found in metadata")

    print("\n--- All indexing tasks complete! ---")
    

def build_faiss_index_from_cache_query(
    model_alias: str,
    dataset_name: str,
    label: str,
    shard_size: int,
    nlist: int = 2048,
    m: int = 8,
    bits: int = 8
):
    """
    Builds FAISS indexes for query activations (query_acts.dat),
    based on the specific output of the activation_cache_sae.py script with query cache.
    """
    print(f"--- Starting FAISS index build for query activations {model_alias}/{dataset_name}/{label} ---")

    # 1. Construct the path to the cache directory
    # This must exactly match the folder naming convention from your caching script
    foldername = f"{DEFAULT_SAE_CACHE_DIR}/{model_alias}_{dataset_name}/label_{label}_shard_size_{shard_size}"
    metadata_filepath = os.path.join(foldername, "metadata.json")

    # 2. Load the metadata file to get parameters
    if not os.path.exists(metadata_filepath):
        print(f"FATAL ERROR: Metadata file not found at {metadata_filepath}")
        print("Please ensure you have run the activation_cache_sae.py script with --with_query_cache first.")
        return
        
    print(f"Loading metadata from: {metadata_filepath}")
    with open(metadata_filepath, 'r') as f:
        metadata = json.load(f)
    
    # Extract necessary info from metadata
    layers_to_index = metadata['layers_cached']
    n_layers, d_model = metadata['activations_original_shape'][0], metadata['activations_original_shape'][2]
    
    print(f"Found {len(layers_to_index)} layers to index: {layers_to_index}")
    print(f"Query activation dimension: {d_model}")

    # 3. Load sample indices to get the number of samples
    sample_indices_filepath = os.path.join(foldername, "sample_indices.json")
    if not os.path.exists(sample_indices_filepath):
        print(f"FATAL ERROR: Sample indices file not found at {sample_indices_filepath}")
        print("Please ensure you have run the activation_cache_sae.py script with --with_query_cache first.")
        return
        
    with open(sample_indices_filepath, 'r') as f:
        sample_indices = json.load(f)
    
    n_samples = len(sample_indices)
    print(f"Number of samples: {n_samples}")

    # 4. Define path for query activations
    query_acts_filename = "query_acts.dat"
    query_acts_filepath = os.path.join(foldername, query_acts_filename)

    if not os.path.exists(query_acts_filepath):
        print(f"FATAL ERROR: Query activations file not found at {query_acts_filepath}")
        print("Please ensure you have run the activation_cache_sae.py script with --with_query_cache first.")
        return

    # 5. Load the query activations data
    print(f"[1/4] Loading memmap file: {query_acts_filename}")
    query_acts_shape = (n_samples, n_layers, d_model)
    query_acts_data = np.memmap(query_acts_filepath, dtype='float32', mode='r', shape=query_acts_shape)
    query_acts_data = np.ascontiguousarray(query_acts_data)

    # 6. Build and Train the FAISS index for query activations
    print(f"[2/4] Building and training index for query activations...")
    quantizer = faiss.IndexFlatL2(d_model)
    index = faiss.IndexIVFPQ(quantizer, d_model, nlist, m, bits)
    
    num_train_samples = min(n_samples, nlist * 100)
    train_indices = np.random.choice(n_samples, num_train_samples, replace=False)
    
    # Use the first layer's query activations for training (or you can use all layers)
    train_data = query_acts_data[train_indices, 0, :]  # [num_train_samples, d_model]
    index.train(train_data)

    # 7. Add all query activation vectors to the index
    print(f"[3/4] Adding {n_samples} query activation vectors to the index...")
    batch_size_add = 100000
    for i in range(0, n_samples, batch_size_add):
        # Use the first layer's query activations for indexing
        batch_data = query_acts_data[i:i+batch_size_add, 0, :]  # [batch_size, d_model]
        index.add(batch_data)
    
    print(f"为 query activations 创建 Direct Map...")
    index.make_direct_map()
    
    # 8. Save the final index file
    index_filename = f"faiss_index_query_acts_layer_{layers_to_index[0]}.index"
    save_path = os.path.join(foldername, index_filename)
    print(f"[4/4] Saving query activations index to: {save_path}")
    faiss.write_index(index, save_path)
    print(f"--- Successfully created query activations index ---")

    print("\n--- Query activations indexing task complete! ---")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build layer-specific FAISS indexes from cached activations and SAE features')
    parser.add_argument('--model_alias', type=str, default="gemma-2-9b-it", help='Alias of the model used for caching')
    parser.add_argument('--dataset', type=str,default="safeedit", help='Dataset name used for caching')
    parser.add_argument('--shard_size', type=int, default=4050, help='Shard size used in the folder name during caching')
    parser.add_argument('--cache_type', type=str, default="query", help='cache_type: query, sae, acts')
    
    args = parser.parse_args()
    cache_func = {
        "query": build_faiss_index_from_cache_query,
        "sae": build_faiss_index_from_cache_sae,
        "acts": build_faiss_index_from_cache
    }
    for label in ["safe", "unsafe"]:
        cache_func[args.cache_type](
            model_alias=args.model_alias,
            dataset_name=args.dataset,
            label=label,
            shard_size=args.shard_size
        )

