import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/home/wangxin/models/gemma-2-9b-it"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(model)
print(tokenizer)