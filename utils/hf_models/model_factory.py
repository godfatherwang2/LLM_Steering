import torch

from typing import Union

from utils.hf_models.model_base import ModelBase

def construct_model_base(model_path: str) -> ModelBase:
    if 'qwen' in model_path.lower():
        from utils.hf_models.qwen_model import QwenModel
        return QwenModel(model_path)
    if 'llama-3' in model_path.lower():
        from utils.hf_models.llama3_model import Llama3Model
        return Llama3Model(model_path)
    elif 'gemma' in model_path.lower():
        from utils.hf_models.gemma_model import GemmaModel
        return GemmaModel(model_path) 
    else:
        raise ValueError(f"Unknown model family: {model_path}")