import sys
sys.path.append('./')
sys.path.append('../')
import os
import torch
import numpy as np
import argparse
from typing import List
from tqdm import tqdm
# 假设您的项目结构如下，以便正确导入模块
from utils.hf_models.model_factory import construct_model_base
from utils.hf_models.model_base import ModelBase
from utils.utils import model_alias_to_model_name
from utils.hf_models.gemma_model import format_instruction_gemma_chat
from dataset.load_data import load_safeedit_test_data, load_gsm_test_data, load_samsum_test_data
import json
from transformers import StoppingCriteria

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)



@torch.no_grad()
def standard_generate(
    model_base: ModelBase,
    prompt: str,
    max_new_tokens: int = 50,
):
    tokenizer = model_base.tokenizer
    model = model_base.model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if args.test_dataset == "GSM":
        stop_id_sequences=[tokenizer.encode("Question:", add_special_tokens=False)]
        stopping_criteria=[KeyWordsCriteria(stop_id_sequences)]
        outputs = model.generate(inputs.input_ids, max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria)
    else:
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

def main(args):
    print("加载模型...")
    model_base = construct_model_base(model_alias_to_model_name[args.model_alias])
    tokenizer = model_base.tokenizer
    if args.test_dataset == "safeedit":
        queries = load_safeedit_test_data(model_alias=args.model_alias, apply_chat_format=True)
    elif args.test_dataset == "StrongREJECT":
        queries = load_safeedit_test_data(model_alias=args.model_alias, apply_chat_format=True)
    elif args.test_dataset == "BeaverTails":
        queries = load_safeedit_test_data(model_alias=args.model_alias, apply_chat_format=True)
    elif args.test_dataset == "GSM":
        queries = load_gsm_test_data(model_alias=args.model_alias, apply_chat_format=True)
        if args.model_alias == "gemma-2-9b-it":
            args.max_new_tokens=1024
        else:
            args.max_new_tokens=512
    elif args.test_dataset == "samsum":
        queries = load_samsum_test_data(model_alias=args.model_alias, apply_chat_format=True)
    else:
        raise ValueError(f"Invalid test dataset: {args.test_dataset}")


    questions = []
    generations = []

    gt_answers = []
    for query in tqdm(queries, desc="Generating outputs"):
        output = standard_generate(
            model_base,
            query['prompt'],
            max_new_tokens=int(args.max_new_tokens)
        )
        pure_output = output.split("<end_of_turn>\n<start_of_turn>model\n")[1] if "<end_of_turn>\n<start_of_turn>model\n" in output else output
        generations.append(pure_output)
        questions.append(query['prompt'].replace("<end_of_turn>\n<start_of_turn>model\n", "").replace("<start_of_turn>user\n", ""))
        if args.test_dataset == "GSM" or args.test_dataset == "samsum":
            gt_answers.append(query['answer'])

    if args.test_dataset == "GSM" or args.test_dataset == "samsum":
        final_data = [{'question': q, 'generation': g, 'answer': a} for q, g, a in zip(questions, generations, gt_answers)]
    else:
        final_data = [{'question': q, 'generation': g} for q, g in zip(questions, generations)]
    rst_dir = f"results/generations/{args.model_alias}/{args.test_dataset}/vanilla"
    os.makedirs(rst_dir, exist_ok=True)
    with open(os.path.join(rst_dir, "outputs.json"), "w") as f:
        json.dump(final_data, f,indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='标准生成')
    parser.add_argument('--model_alias', type=str, default="gemma-2-9b-it", help='模型别名')
    parser.add_argument('--test_dataset', type=str, default="safeedit")
    parser.add_argument('--max_new_tokens', type=str, default=50)
    args = parser.parse_args()
    main(args)