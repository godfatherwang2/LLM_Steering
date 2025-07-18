# %%
import sys
sys.path.append("..")

from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import huggingface_hub
import os
import subprocess
from typing import Literal, Tuple
from utils.utils import clear_memory

# %%
DEFAULT_MODELS = [
    ("model", "google/gemma-2b-it"),
    ("model", "google/gemma-2-9b-it"),
    ("model", "meta-llama/Meta-Llama-3-8B-Instruct"),
    # ("sae", "EleutherAI/sae-llama-3-8b-32x"),
    ("sae", "obalcells/sae-llama-3-8b-instruct"),
]

def parse_args():
    parser = argparse.ArgumentParser(description="Load specified models or default models.")
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        help='List of model names to load. If not provided, default models will be loaded.')
    return parser.parse_args()

def load_models(models=None):
    if models is None:
        models = DEFAULT_MODELS
    
    for model_type, model_name in models:
        print(f"Loading {model_name}...")
        if model_type == 'model':
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            print(f"Loaded model {model_name}")
            print()
            del model, tokenizer
            clear_memory()
        elif model_type == 'sae':
            path = huggingface_hub.snapshot_download(model_name)
            print(f"Loaded SAEs from {model_name} and saved to {path}")
            print()

def get_gemma2_sae_path_and_id(
    layer: int,
    d_sae: int = 16384,
    hook_name: Literal["post_mlp_residual", "mlp_output", "attn_output_pre_linear"] = "post_mlp_residual",
    sparsity: Literal["0_00015", "0_00025", "0_00045", "0_0006", "0_0007"] = "0_00015",
    it: bool=True,
    gemma_2_sae_weights_root: str = None,
) -> Tuple[str, str]:
    if it == True:
        assert hook_name in ["post_mlp_residual"], "Only post_mlp_residual is supported for IT models"
        assert layer in [9, 20], "Only layers 9 and 20 are supported for IT models"

    model_name = "gemma2-9b" if not it else "gemma2-9b-IT"

    if gemma_2_sae_weights_root is None:
        gemma_2_sae_weights_root = os.environ.get("GEMMA_2_SAE_WEIGHTS_ROOT", "/root/gemma-2-saes")

    sae_path = os.path.join(gemma_2_sae_weights_root, model_name, str(layer), hook_name, str(d_sae), str(sparsity), "params")
    sae_id = f"{layer}/{hook_name}/{d_sae}/{sparsity}"

    return sae_path, sae_id

def download_gemma2_sae(
    layer: int,
    d_sae: int = 16384,
    hook_name: Literal["post_mlp_residual", "mlp_output", "attn_output_pre_linear"] = "post_mlp_residual",
    sparsity: Literal["0_00015", "0_00025", "0_00045", "0_0006", "0_0007"] = "0_00015",
    it: bool=True,
) -> Tuple[str, str]:
    if it == True:
        assert hook_name in ["post_mlp_residual"], "Only post_mlp_residual is supported for IT models"
        assert layer in [9, 20], "Only layers 9 and 20 are supported for IT models"

    model_name = "gemma2-9b" if not it else "gemma2-9b-IT"

    bucket_path = f"gs://gemma-2-saes/{model_name}/{layer}/{hook_name}/{d_sae}/{sparsity}/params"

    local_path, sae_id = get_gemma2_sae_path_and_id(layer, d_sae, hook_name, sparsity, it)

    # Create the local directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Run the gcloud command to download the SAE
    gcloud_command = f"gsutil -m cp -r {bucket_path} {local_path}"
    print(f"Downloading SAE from {bucket_path} to {local_path}")

    try:
        subprocess.run(gcloud_command, shell=True, check=True)
        print(f"Successfully downloaded SAE from {bucket_path} to {local_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading SAE: {e}")
        return None

    return local_path, sae_id


if __name__ == "__main__":

    args = parse_args()
    models_to_load = args.models

    load_models(models_to_load)

    print(f"Downloading Gemma2 IT SAEs...")
    for layer in [9, 20]:
        for d_sae in [16384, 131072]:
            for sparsity in ["0_00015", "0_00045", "0_0007"]:
                download_gemma2_sae(layer=layer, d_sae=d_sae, hook_name="post_mlp_residual", sparsity=sparsity, it=True)


# %%
