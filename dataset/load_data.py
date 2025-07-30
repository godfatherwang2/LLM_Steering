import sys
sys.path.append("/home/wangxin/projects/safety_steer")
import os
import json
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import random
from jaxtyping import Float, Int
import torch
from torch import Tensor
from typing import Literal, Dict, List
import glob
import csv
from functools import partial
from datasets import Dataset
import re
from sklearn.model_selection import train_test_split
import datasets
from utils.generation_utils import is_generation_refusal
from datasets import load_dataset
import numpy as np

entities_dataset_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "processed", "entity_prompts")
triviaqa_dataset_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "processed", "triviaqa")
keen_dataset_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "processed", "keen")
mmlu_dataset_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "processed", "mmlu")
keen_raw_data_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "raw", "keen")


MMLU_CASE_PROMPT = """Question: {question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Answer: """

PROMPT_TYPES_BY_ENTITY = {
    "book": ["book_is_known", "book_attribute_creator"],
    "movie": ["movie_is_known", "movie_directors", "movie_cast", "movie_screenwriters", "movie_genres", "movie_release_year", "movie_durations", "movie_directors_free", "movie_cast_free", "movie_screenwriters_free", "movie_genres_free", "movie_release_year_free", "movie_durations_free"],
    "city": ["city_is_known", "city_country", "city_location", "city_population", "city_elevation", "city_country_free", "city_location_free", "city_population_free", "city_elevation_free"],
    "player": ["player_is_known", "player_place_birth", "player_date_birth", "player_teams_list", "player_weight", "player_height", "player_place_birth_free", "player_date_birth_free", "player_teams_list_free", "player_weight_free", "player_height_free"],
    "song": ["song_is_known", "song_album", "song_publication_year", "song_genres", "song_performers", "song_album_free", "song_publication_year_free", "song_genres_free", "song_performers_free"],
    "scotus_case": ["case_existence", "citation_retrieval", "majority_author"],
    "fake_scotus_case": ["case_existence", "citation_retrieval", "majority_author"],
}

VALID_ENTITY_TYPES = list(PROMPT_TYPES_BY_ENTITY.keys())
VALID_ENTITY_PROMPT_TYPES = [prompt_type for entity_type in PROMPT_TYPES_BY_ENTITY for prompt_type in PROMPT_TYPES_BY_ENTITY[entity_type]]

def load_entity_queries(
    model_alias: str,
    prompt_type: VALID_ENTITY_PROMPT_TYPES,
    entity_type: VALID_ENTITY_TYPES=None,
    entity_knowledge_type: Literal["known", "unknown", "fake", "other"]=None,
    generation_type: Literal["refusal", "error", "correct"]=None,
    remove_teacher_forced: bool = False,
    check_all_generations: bool=False,
    return_only_prompts: bool=False,
):
    """
    Loads the queries for a given model and prompt type.

    Args:
        model_alias: The model alias.
        prompt_type: The prompt type to load (e.g. 'movie_directors' or 'movie_is_known')
        entity_type: The entity type to load (e.g. 'movie'). The entity type is inferred from the prompt type if not provided.
        entity_knowledge_type: The entity knowledge type to load ('known', 'unknown', 'fake', 'other').
            If None, all entity knowledge types are loaded.
        generation_type: The generation type. If None, all generation types are loaded.
        remove_teacher_forced: Whether to remove teacher forced queries, i.e. "...<instruction><|end_header_id|>assistant\nAnswer=" -> "...<instruction><|end_header_id|>assistant\n"
        check_all_generations: Whether to check all generations.
        return_only_prompts: Whether to return only the prompts or also include the rest of the query data.

    Returns:
        list: The queries or prompts.
    """
    assert entity_type is None or entity_type in VALID_ENTITY_TYPES, f"Invalid entity type: {entity_type}"
    assert generation_type is None or generation_type in ["refusal", "error", "correct"], f"Invalid generation type: {generation_type}"
    assert entity_knowledge_type is None or entity_knowledge_type in ["known", "unknown", "fake", "other"], f"Invalid entity knowledge type: {entity_knowledge}"

    if '_' in prompt_type and prompt_type.split("_")[0] in PROMPT_TYPES_BY_ENTITY:
        assert prompt_type in VALID_ENTITY_PROMPT_TYPES, f"Invalid prompt type: {prompt_type}"
        # prompt type might include the entity type (e.g 'movie_directors' or 'song_album')
        file_name = f"{prompt_type}.json"
        entity_type = prompt_type.split("_")[0]
    else:
        assert entity_type is not None, "Entity type must be provided if prompt type does not include it"
        # assert f"{entity_type}_{prompt_type}" in VALID_ENTITY_PROMPT_TYPES, f"Invalid prompt type: {prompt_type}"
        file_name = f"{entity_type}_{prompt_type}.json"

    file_path = os.path.join(entities_dataset_dir_path, model_alias, file_name)

    if os.path.exists(file_path) == False:
        print(f"WARNING: file path {file_path} doesn't exist")
        return []

    queries = json.load(open(file_path, 'r'))

    if generation_type is not None:
        def include_query(query):
            generations = [query['generation']] + [g for g in query['extra_data']['multiple_generations']] if check_all_generations else [query['generation']]
            correct = (query['score'] == 1) if not check_all_generations else (query['score'] == 1 and query['extra_data']['num_correct_generations'] == len(query['extra_data']['multiple_generations']))

            if generation_type == "refusal":
                return all([is_generation_refusal(g) for g in generations])
            elif generation_type == "error":
                return (not correct) and all([not is_generation_refusal(g) for g in generations])
            elif generation_type == "correct":
                return correct and all([not is_generation_refusal(g) for g in generations])
            else:
                raise ValueError(f"Invalid generation type: {generation_type}")

        queries = [q for q in queries if include_query(q)]

    if entity_knowledge_type is not None:
        entity_knowledge = json.load(open(os.path.join(entities_dataset_dir_path, model_alias, "entity_knowledge.json")))

        assert entity_type in entity_knowledge, f"Entity type {entity_type} not found in entity knowledge"

        entities_to_include = set(entity_knowledge[entity_type][entity_knowledge_type])

        queries = [q for q in queries if q["entity"] in entities_to_include]

    if remove_teacher_forced:
        eoi_str = {
            "gemma-2b-it": "model\n",
            "Meta-Llama-3-8B-Instruct": "assistant<|end_header_id|>",
        }
        slice_prompt_fn = lambda prompt: prompt[:prompt.index(eoi_str[model_alias])+len(eoi_str[model_alias])]
        queries = [{**q, "prompt": slice_prompt_fn(q["prompt"])} for q in queries]

    if return_only_prompts:
        return [q["prompt"] for q in queries]

    return queries

def load_pile_prompts():
    from datasets import load_dataset

    prompts = load_dataset("NeelNanda/pile-10k")['train']['text']

    return prompts

def load_triviaqa_queries(
    model_alias: str,
    return_only_prompts: bool=False,
    apply_chat_format: bool=False,
    label: Literal["refusal", "error", "correct", "other"]=None,
    strict_filtering: bool = False,
    split: Literal["train", "validation", "test"]=None,
):
    """
    Loads the TriviaQA queries for a given model.

    Args:
        model_alias: The model alias.
        return_only_prompts: Whether to return only the prompts or also include the rest of the query data.
        apply_chat_format: Whether to apply the chat format to the queries if they aren't already in the chat format.
        label: The label to filter the queries by, e.g. "refusal", "error", "correct", "other".

    Returns:
        list: The queries or prompts.
    """
    def load_json_batches(directory):
        # Get all batch JSON files in sorted order
        batch_files = sorted(glob.glob(os.path.join(directory, "batch_*.json")))
        
        combined_data = []
        
        # Load and merge each batch file
        for batch_file in batch_files:
            with open(batch_file, 'r') as f:
                batch_data = json.load(f)
                combined_data.extend(batch_data)
                        
        return combined_data

    if '/' in model_alias:
        model_alias = model_alias.split('/')[-1]

    file_path = os.path.join(triviaqa_dataset_dir_path, f"{model_alias}")

    if apply_chat_format:
        if model_alias == "gemma-2b-it" or model_alias == "gemma-7b-it" or model_alias == "gemma-2-9b-it":
            from utils.hf_models.gemma_model import format_instruction_gemma_chat 
            format_instructions_chat_fn = partial(format_instruction_gemma_chat, output=None, system=None, include_trailing_whitespace=True)
        elif model_alias == "Meta-Llama-3-8B-Instruct" or model_alias == "Meta-Llama-3-70B-Instruct":
            from utils.hf_models.llama3_model import format_instruction_llama3_chat
            format_instructions_chat_fn = partial(format_instruction_llama3_chat, output=None, system=None, include_trailing_whitespace=True)
        else:
            raise ValueError(f"Invalid model alias: {model_alias}")
    
    # We're no longer using the arrow dataset
    # dataset = Dataset.load_from_disk(file_path)
    # queries: List[Dict] = [item for item in dataset]
    queries = load_json_batches(file_path)
    #queries = json.load(open(file_path, "r"))

    # Update label to refusal if the greedy completion is a refusal
    for query in queries:
        if is_generation_refusal(query['greedy_completion'])==True:
            query['label'] = 'refusal'
    if split is not None:
        print('Loading split', split)
        split_file_path = os.path.join(triviaqa_dataset_dir_path, model_alias, f"{model_alias}_triviaqa_splits.json")
        if not os.path.exists(split_file_path):

            print(f"Creating splits for {model_alias} and saving to {split_file_path}")
        
            train_indices, val_and_test_indices = train_test_split(range(len(queries)), train_size=0.8, random_state=42)
            val_indices, test_indices = train_test_split(val_and_test_indices, train_size=0.5, random_state=42)

            train_questions = set([queries[i]['question'] for i in train_indices])
            val_questions = set([queries[i]['question'] for i in val_indices]) - train_questions
            test_questions = set([queries[i]['question'] for i in test_indices]) - train_questions - val_questions
        
            split_questions = {
                'train': list(train_questions),
                'validation': list(val_questions),
                'test': list(test_questions)
            }
        
            # Save splits to file
            with open(split_file_path, 'w') as f:
                json.dump(split_questions, f, indent=4)
        
            print(f"Splits created and saved to {split_file_path}")

        with open(split_file_path, 'r') as f:
            split_questions: Dict[str, List[str]] = json.load(f)
            split_questions = {k: set(v) for k, v in split_questions.items()}

        assert split in split_questions, f"Invalid split: {split}"
        queries = [q for q in queries if q['question'] in split_questions[split]]

    if apply_chat_format:
        for q in queries:
            q['prompt'] = format_instructions_chat_fn(instruction=q['question'])

    # Filter based on generation_type if specified
    if label is not None:
        queries = [q for q in queries if q['label'] == label]

    # The labels of the sampled completions must all match with the label of the greedy generation
    if strict_filtering:
        queries = [q for q in queries if all([sampled_label == q['label'] for sampled_label in q['string_matching_sampled_labels']])]

    if return_only_prompts:
        return [q["prompt"] for q in queries]

    return queries

def load_wikidata_queries(
    model_alias: str,
    return_only_prompts: bool=False,
    apply_chat_format: bool=False,
    label: Literal["refusal", "error", "correct", "other"]=None,
    include_prompt_types: List[str]=None,
    strict_filtering: bool = False,
    split: Literal["train", "validation", "test"]=None,
    free_generation: bool = False,
    entity_type_and_entity_name_format: bool = False,
    entity_name_format: bool = False
):
    """
    Loads the TriviaQA queries for a given model.

    Args:
        model_alias: The model alias.
        return_only_prompts: Whether to return only the prompts or also include the rest of the query data.
        apply_chat_format: Whether to apply the chat format to the queries if they aren't already in the chat format.
        label: The label to filter the queries by, e.g. "refusal", "error", "correct", "other".
        entity_type_and_entity_name_format: Whether to format the prompt as The {entity_type} '{entity}'
        entity_name_format: Whether to format prompt as The '{entity}'

    Returns:
        list: The queries or prompts.
    """

    if '/' in model_alias:
        model_alias = model_alias.split('/')[-1]

    if free_generation == True:
        file_path = os.path.join(entities_dataset_dir_path, f"{model_alias}", "free_generation")
    else:
        file_path = os.path.join(entities_dataset_dir_path, f"{model_alias}")

    print('file_path', file_path)

    if apply_chat_format:
        if model_alias == "gemma-2b-it" or model_alias == "gemma-7b-it" or model_alias == "gemma-2-9b-it" or model_alias == "gemma-2-2b-it":
            from utils.hf_models.gemma_model import format_instruction_gemma_chat 
            format_instructions_chat_fn = partial(format_instruction_gemma_chat, output=None, system=None, include_trailing_whitespace=True)
        elif model_alias == "Meta-Llama-3-8B-Instruct" or model_alias == "Meta-Llama-3-70B-Instruct":
            from utils.hf_models.llama3_model import format_instruction_llama3_chat
            format_instructions_chat_fn = partial(format_instruction_llama3_chat, output=None, system=None, include_trailing_whitespace=True)
        else:
            raise ValueError(f"Invalid model alias: {model_alias}")

    file_paths = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.json')]
    queries = []
    for file_path in file_paths:
        if include_prompt_types is not None and all(prompt_type not in file_path for prompt_type in include_prompt_types):
            continue
        with open(file_path, 'r') as f:
            queries.extend(json.load(f))

    if split is not None:
        split_file_path = os.path.join(entities_dataset_dir_path, f"{model_alias}_wikidata_splits.json")

        if not os.path.exists(split_file_path):
            print(f"Creating splits for {model_alias} and saving to {split_file_path}")
        
            train_indices, val_and_test_indices = train_test_split(range(len(queries)), train_size=0.8, random_state=42)
            val_indices, test_indices = train_test_split(val_and_test_indices, train_size=0.5, random_state=42)

            train_questions = set([queries[i]['question'] for i in train_indices])
            val_questions = set([queries[i]['question'] for i in val_indices]) - train_questions
            test_questions = set([queries[i]['question'] for i in test_indices]) - train_questions - val_questions
        
            split_questions = {
                'train': list(train_questions),
                'validation': list(val_questions),
                'test': list(test_questions)
            }
        
            # Save splits to file
            with open(split_file_path, 'w') as f:
                json.dump(split_questions, f, indent=4)
        
            print(f"Splits created and saved to {split_file_path}")

        with open(split_file_path, 'r') as f:
            split_questions: Dict[str, List[str]] = json.load(f)
            split_questions = {k: set(v) for k, v in split_questions.items()}

        assert split in split_questions, f"Invalid split: {split}"
        queries = [q for q in queries if q['question'] in split_questions[split]]

    
    # Update label to refusal if the greedy completion is a refusal
    for query in queries:
        if 'greedy_completion' not in query:
            print(query)
        if is_generation_refusal(query['greedy_completion'])==True:
            query['label'] = 'refusal'

    if apply_chat_format:
        for q in queries:
            q['prompt'] = format_instructions_chat_fn(instruction=q['question'])
    
    if entity_type_and_entity_name_format:
        for q in queries:
            entity_type = q['entity_type']
            entity = q['entity']
            q['prompt'] = f"The {entity_type} '{entity}'"

    if entity_name_format:
        for q in queries:
            entity = q['entity']
            q['prompt'] = f"The '{entity}'"

    # Filter based on generation_type if specified
    if label is not None:
        queries = [q for q in queries if q['label'] == label]

    # The labels of the sampled completions must all match with the label of the greedy generation
    if strict_filtering:
        queries = [q for q in queries if all([sampled_label == q['label'] for sampled_label in q['string_matching_sampled_labels']])]

    if return_only_prompts:
        return [q["prompt"] for q in queries]

    return queries

def load_pkusaferlhf_queries(
    model_alias: str,
    return_only_prompts: bool=False,
    apply_chat_format: bool=False,
    label: Literal["refusal", "error", "correct", "other"]=None,
    include_prompt_types: List[str]=None,
    strict_filtering: bool = False,
    split: Literal["train", "validation", "test"]=None,
    free_generation: bool = False,
    entity_type_and_entity_name_format: bool = False,
    entity_name_format: bool = False
):
    """
    Loads the pkusaferlhf queries for a given model.

    Args:
        model_alias: The model alias.
        return_only_prompts: Whether to return only the prompts or also include the rest of the query data.
        apply_chat_format: Whether to apply the chat format to the queries if they aren't already in the chat format.
        label: The label to filter the queries by, e.g. "refusal", "error", "correct", "other".
        entity_type_and_entity_name_format: Whether to format the prompt as The {entity_type} '{entity}'
        entity_name_format: Whether to format prompt as The '{entity}'

    Returns:
        list: The queries or prompts.
    """

    if '/' in model_alias:
        model_alias = model_alias.split('/')[-1]

    from datasets import load_dataset
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")["train"]

    if apply_chat_format:
        if model_alias == "gemma-2b-it" or model_alias == "gemma-7b-it" or model_alias == "gemma-2-9b-it" or model_alias == "gemma-2-2b-it":
            from utils.hf_models.gemma_model import format_instruction_gemma_chat 
            format_instructions_chat_fn = partial(format_instruction_gemma_chat, system=None, include_trailing_whitespace=True)
        elif model_alias == "Meta-Llama-3-8B-Instruct" or model_alias == "Meta-Llama-3-70B-Instruct":
            from utils.hf_models.llama3_model import format_instruction_llama3_chat
            format_instructions_chat_fn = partial(format_instruction_llama3_chat, system=None, include_trailing_whitespace=True)
        else:
            raise ValueError(f"Invalid model alias: {model_alias}")
    
    queries = []
    for _d in dataset:
        response_0 = {"prompt": _d["prompt"], "response": _d["response_0"], "label": _d["is_response_0_safe"]}
        response_1 = {"prompt": _d["prompt"], "response": _d["response_1"], "label": _d["is_response_1_safe"]}
        if response_0["response"] is not None and len(response_0["response"]) > 0:
            queries.append(response_0)
        if response_1["response"] is not None and len(response_1["response"]) > 0:
            queries.append(response_1)


    if apply_chat_format:
        for q in queries:
            q['prompt'] = format_instructions_chat_fn(instruction=q['question'], output=q["response"])
    else:
        for q in queries:
            q['prompt'] = "Question:\n" + q['prompt'] + "\nResponse:\n" + q["response"]

    # Filter based on generation_type if specified
    if label is not None:
        queries = [q for q in queries if q['label'] == label]


    if return_only_prompts:
        return [q["prompt"] for q in queries]

    return queries

def load_safeedit_queries(
    model_alias: str,
    apply_chat_format: bool=False,
):
    if '/' in model_alias:
        model_alias = model_alias.split('/')[-1]

    from datasets import load_dataset
    dataset = load_dataset("zjunlp/SafeEdit")['train']
    
    if apply_chat_format:
        if "gemma" in model_alias:
            from utils.hf_models.gemma_model import format_instruction_gemma_chat 
            format_instructions_chat_fn = partial(format_instruction_gemma_chat, system=None, include_trailing_whitespace=True)
        elif "Llama-3" in model_alias:
            from utils.hf_models.llama3_model import format_instruction_llama3_chat
            format_instructions_chat_fn = partial(format_instruction_llama3_chat, system=None, include_trailing_whitespace=True)
        else:
            raise ValueError(f"Invalid model alias: {model_alias}")
    
    queries = []
    for _d in dataset:
        queries.append({"prompt": _d["adversarial prompt"], "response": _d["safe generation"], "label": True})
        queries.append({"prompt": _d["adversarial prompt"], "response": _d["unsafe generation"], "label": False})

    if apply_chat_format:
        for q in tqdm(queries, desc="应用聊天模板"):
            q['prompt'] = format_instructions_chat_fn(instruction=q['prompt'],output=q['response'])
    return queries

def load_safeedit_for_sae_feature_selection(model_alias: str, apply_chat_format: bool=False):
    dataset = load_dataset("zjunlp/SafeEdit")['train']
    queries = []
    for _d in dataset:
        queries.append({"question": _d["adversarial prompt"], "not_matching": _d["unsafe generation"], "matching": _d["safe generation"]})
    return queries


def load_safeedit_test_data(model_alias: str, apply_chat_format: bool=False):
    dataset = load_dataset("json",data_files="/home/wangxin/projects/safety_steer/dataset/SafeEdit/SafeEdit_test.json")['train']
    queries = []
    if apply_chat_format:
        if model_alias == "gemma-2b-it" or model_alias == "gemma-7b-it" or model_alias == "gemma-2-9b-it" or model_alias == "gemma-2-2b-it":
            from utils.hf_models.gemma_model import format_instruction_gemma_chat 
            format_instructions_chat_fn = partial(format_instruction_gemma_chat, system=None, include_trailing_whitespace=True)
        elif model_alias == "Meta-Llama-3-8B-Instruct" or model_alias == "Meta-Llama-3-70B-Instruct":
            from utils.hf_models.llama3_model import format_instruction_llama3_chat
            format_instructions_chat_fn = partial(format_instruction_llama3_chat, system=None, include_trailing_whitespace=True)
        else:
            raise ValueError(f"Invalid model alias: {model_alias}")
        
    for _d in dataset:
        queries.append({"prompt": _d["adversarial prompt"]})
    if apply_chat_format:
        for q in queries:
            q['prompt'] = format_instructions_chat_fn(instruction=q['prompt'])
    return queries 

def load_strongreject_test_data(model_alias: str, apply_chat_format: bool=False):
    dataset = load_dataset("/home/wangxin/datas/walledai/StrongREJECT",split="train")
    if apply_chat_format:
        if model_alias == "gemma-2b-it" or model_alias == "gemma-7b-it" or model_alias == "gemma-2-9b-it" or model_alias == "gemma-2-2b-it":
            from utils.hf_models.gemma_model import format_instruction_gemma_chat 
            format_instructions_chat_fn = partial(format_instruction_gemma_chat, system=None, include_trailing_whitespace=True)
        elif model_alias == "Meta-Llama-3-8B-Instruct" or model_alias == "Meta-Llama-3-70B-Instruct":
            from utils.hf_models.llama3_model import format_instruction_llama3_chat
            format_instructions_chat_fn = partial(format_instruction_llama3_chat, system=None, include_trailing_whitespace=True)
        else:
            raise ValueError(f"Invalid model alias: {model_alias}")

    queries = []
    for _d in dataset:
        queries.append({"prompt": _d["prompt"]})
    if apply_chat_format:
        for q in queries:
            q['prompt'] = format_instructions_chat_fn(instruction=q['prompt'])
    return queries


def load_beavertails_test_data(model_alias: str, apply_chat_format: bool=False):
    dataset = load_dataset("/home/wangxin/datas/PKU-Alignment/BeaverTails-Evaluation", split="test")
    if apply_chat_format:
        if model_alias == "gemma-2b-it" or model_alias == "gemma-7b-it" or model_alias == "gemma-2-9b-it" or model_alias == "gemma-2-2b-it":
            from utils.hf_models.gemma_model import format_instruction_gemma_chat 
            format_instructions_chat_fn = partial(format_instruction_gemma_chat, system=None, include_trailing_whitespace=True)
        elif model_alias == "Meta-Llama-3-8B-Instruct" or model_alias == "Meta-Llama-3-70B-Instruct":
            from utils.hf_models.llama3_model import format_instruction_llama3_chat
            format_instructions_chat_fn = partial(format_instruction_llama3_chat, system=None, include_trailing_whitespace=True)
        else:
            raise ValueError(f"Invalid model alias: {model_alias}")

    queries = []
    for _d in dataset:
        queries.append({"prompt": _d["prompt"]})
    if apply_chat_format:
        for q in queries:
            q['prompt'] = format_instructions_chat_fn(instruction=q['prompt'])
    return queries

def load_gsm_test_data(
        model_alias: str,
        data_name="gsm",
        data_path="/home/wangxin/projects/steer-target-atoms-main/data/safety",
        apply_chat_format=False,
    ):
        data_file = os.path.join(
            data_path,
            data_name,
            "test.jsonl",
        )
        dataset = load_dataset("json", data_files=data_file, split="train")
        original_columns = dataset.column_names


        if apply_chat_format:
            if model_alias == "gemma-2b-it" or model_alias == "gemma-7b-it" or model_alias == "gemma-2-9b-it" or model_alias == "gemma-2-2b-it":
                from utils.hf_models.gemma_model import format_instruction_gemma_chat 
                format_instructions_chat_fn = partial(format_instruction_gemma_chat, system=None, include_trailing_whitespace=True)
            elif model_alias == "Meta-Llama-3-8B-Instruct" or model_alias == "Meta-Llama-3-70B-Instruct":
                from utils.hf_models.llama3_model import format_instruction_llama3_chat
                format_instructions_chat_fn = partial(format_instruction_llama3_chat, system=None, include_trailing_whitespace=True)
            else:
                raise ValueError(f"Invalid model alias: {model_alias}")

        def return_prompt_and_responses(samples) -> Dict[str, str]: 
            prompts = []
            answers = []
            questions = []
            for question in samples["question"]:
                question = "Answer the following question.\n\n" + "Question: " + question.strip() + "\n\nPlease wrap the final answer in $\\boxed{{}}$ tag."
                if apply_chat_format:
                    question = format_instructions_chat_fn(instruction=question)
                questions.append(question)
            for answer in samples["answer"]:
                answer = re.sub(r"(\d),(\d)", r"\1\2", answer.split("####")[1].strip())
                assert float(answer), f"answer is not a valid number: {answer}"
                answers.append(answer)
            return {
                "prompt": questions,
                "answer": answers,
            }

        return dataset.map(
            return_prompt_and_responses,
            batched=True,
            remove_columns=original_columns,
        )

def load_samsum_test_data(model_alias: str, apply_chat_format: bool=False):
    system_prompt = "You are a helpful assistant for dialog summarization, your response should be **concise** and to the point."
    dataset = load_dataset("/home/wangxin/datas/samsum",split="test")
    original_columns = dataset.column_names
    if apply_chat_format:
        if model_alias == "gemma-2b-it" or model_alias == "gemma-7b-it" or model_alias == "gemma-2-9b-it" or model_alias == "gemma-2-2b-it":
            from utils.hf_models.gemma_model import format_instruction_gemma_chat 
            format_instructions_chat_fn = partial(format_instruction_gemma_chat, system=None, include_trailing_whitespace=True)
        elif model_alias == "Meta-Llama-3-8B-Instruct" or model_alias == "Meta-Llama-3-70B-Instruct":
            from utils.hf_models.llama3_model import format_instruction_llama3_chat
            format_instructions_chat_fn = partial(format_instruction_llama3_chat, system=None, include_trailing_whitespace=True)
        else:
            raise ValueError(f"Invalid model alias: {model_alias}")
    def return_prompt_and_responses(samples) -> Dict[str, str]: 
        prompts = []
        questions = []
        for dialogue in samples['dialogue']:
            prompt = system_prompt + f"Summarize this dialog:\n{dialogue}"
            if apply_chat_format:
                prompt = format_instructions_chat_fn(instruction=prompt)
            prompts.append(prompt)
        return {
            "prompt": prompts, 
            "answer": samples["summary"],
        }
    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_columns,
    )

def balance_data(queries: List[Dict], labels: List[int], shuffle=True):
    # Count the number of examples for each label
    label_counts = {label: 0 for label in set(labels)}
    for query in queries:
        if query['label'] in labels:
            label_counts[query['label']] += 1
    
    # Determine the target count (minimum of the two label counts)
    target_count = min(label_counts.values())
    
    # Create balanced lists
    balanced_queries = []
    label_counts = {label: 0 for label in set(labels)}
    
    for query in queries:
        if query['label'] in labels and label_counts[query['label']] < target_count:
            balanced_queries.append(query)
            label_counts[query['label']] += 1
        
        if all(label_counts.values()) == target_count:
            break

    # Shuffle the balanced queries if requested
    if shuffle:
        random.shuffle(balanced_queries)
    
    return balanced_queries

def get_pred(logits, answer_tokens_idx):

    soft_max_prob = np.exp(logits) / np.sum(np.exp(logits))
    pred = []

    for answer, token_indices in answer_tokens_idx.items():
        prob = float(soft_max_prob[token_indices].sum())
        pred.append((answer, prob))

    return pred

class MMLUDataset:
    """MMLU Dataset for the evaluation of steered model on

    basically, it is a wrapper around the mmlu dataset,
    the testing steps are as follows:
    1. load the data from the mmlu dataset
    2. generate the prompts and responses using the template
    3. pass the prompts to the model for the prediction
    4. evaluate the predictions

    We use the devset from the cais/mmlu for zero-shot evaluation
    https://huggingface.co/datasets/cais/mmlu

    Args:
        data_dir: str, path to the mmlu dataset
    """
    def __init__(self, data_dir="/home/wangxin/projects/steer-target-atoms-main/data/mmlu"):
        self.data_dir = data_dir
        self.dataset = load_dataset(self.data_dir, "all")["test"]  # cais/mmlu
        # process answer to A,B,C,D
        project_to_answer = {
            0: "A",
            1: "B", 
            2: "C",
            3: "D",
        }
        self.answers = self.dataset["answer"]
        self.answers = [project_to_answer[i] for i in self.answers]

    def get_test_data(self):
        def return_prompt_and_responses(samples) -> Dict[str, str]:
            prompts = []
            for i in range(len(samples["question"])):
                question = samples["question"][i]
                choices = samples["choices"][i]
                # 只返回格式化后的问题和选项，不包含系统提示和few-shot示例
                prompt = MMLU_CASE_PROMPT.format(
                    question=question,
                    choice_a=choices[0],
                    choice_b=choices[1], 
                    choice_c=choices[2],
                    choice_d=choices[3],
                )
                prompts.append(prompt)

            return {
                "prompt": prompts,
                "choices": samples["choices"],
            }

        return self.dataset.map(
            return_prompt_and_responses,
            batched=True,
            remove_columns=["subject", "answer"],
        )

    def get_accuracy(self, predictions, tokenizer):
        """
        Args:
            predictions: list, list of predicted answers
            tokenizer: transformers.PreTrainedTokenizer

        Returns:
            accuracy: float, accuracy of the predictions
        """
        assert len(predictions) == len(self.answers)

        answer_tokens_idx = {
            "A": [
                tokenizer.encode(token, add_special_tokens=False)[0]
                for token in ["a", "A"]
            ],
            "B": [
                tokenizer.encode(token, add_special_tokens=False)[0]
                for token in ["b", "B"]
            ],
            "C": [
                tokenizer.encode(token, add_special_tokens=False)[0]
                for token in ["c", "C"]
            ],
            "D": [
                tokenizer.encode(token, add_special_tokens=False)[0]
                for token in ["d", "D"]
            ],
        }
        
        acc = []

        for i in range(len(predictions)):
            logits = predictions[i]["score"]
            ans = self.answers[i]
            pred = get_pred(logits, answer_tokens_idx)
            pred = max(pred, key=lambda x: x[1])[0]
            acc.append(1) if pred == ans else acc.append(0)

        return np.mean(acc)


if __name__ == "__main__":
    data = load_safeedit_test_data(model_alias="gemma-2-9b-it", apply_chat_format=True)