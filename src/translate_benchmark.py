"""
Benchmark Translation Pipeline
==============================

This module handles the translation of question-answer benchmarks (e.g., MMLU,
ARC, Hellaswag, Winogrande) from English to target languages using LLM-based
translation methods.

The pipeline:
1. Loads benchmark data from HuggingFace datasets
2. Distributes work across multiple processes for parallel translation
3. Applies the selected translation method (SC, USI, BoN, T-RANK)
4. Saves results to JSON and optionally pushes to HuggingFace Hub

Key insight from the paper: Translating questions and answer options within
the same prompt context is essential for sentence completion tasks, as it
preserves semantic relationships and prevents contextual misleading.

Functions:
    translate_question: Route to appropriate translation method
    iterate_and_translate_benchmark: Process benchmark entries sequentially
    process_shard: Worker function for parallel processing
    run_subset: Distribute work across worker pool
    run_benchmark_translation: Main entry point for benchmark translation
"""

import openai
import os
from datasets import load_dataset
from tqdm import tqdm
import json
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback
import langcodes
import huggingface_hub
from .initialization import Config
from .benchmark.methods import translate_using_sc, translate_using_usi, translate_using_trank, translate_using_best_of_n
from .benchmark.utils import get_data_key
from .benchmark.save_to_hf import push_data_to_hf
from credentials import hf_token
warnings.filterwarnings("ignore")


def translate_question(choices, question, target_lang, cfg=None):
    """
    Translate a benchmark question with its answer choices to target language.

    Routes to the appropriate translation method based on configuration:
    - SC: Self-Correction with optional verification
    - USI: Universal Self-Improvement with candidate fusion
    - TRANK: Translation Ranking with multi-round competitive ranking
    - BoN: Best-of-N with independent scoring

    Args:
        choices: Answer options (list of strings or single string)
        question: The question text to translate
        target_lang: Target language name (e.g., "Bulgarian")
        cfg: Configuration object with method and model settings

    Returns:
        For SC without check: [translated_question, translated_choices]
        For SC with check: [[translated_q, translated_c], [corrected_q, corrected_c]]
        For USI/BoN: [corrected_question, corrected_choices]
        For TRANK: [question, choices, ranking_scores, raw_ranks]

    Raises:
        Exception: If invalid method type is specified
    """
    method_type = cfg.task_config.method
    
    if method_type == 'SC':
        return translate_using_sc(choices, question, target_lang, cfg=cfg)
    elif method_type == 'USI':
        return translate_using_usi(choices, question, target_lang, cfg=cfg)
    elif method_type == 'TRANK':
        return translate_using_trank(choices, question, target_lang, cfg=cfg)
    elif method_type == 'BoN':
        return translate_using_best_of_n(choices, question, target_lang, cfg=cfg)
    else:
        raise Exception('Please choose a valid prompting method type (SC/USI/TRANK/BoN).')


def iterate_and_translate_benchmark(mmlu_data, target_lang, cfg=None):
    """
    Iterate through benchmark entries and translate each question-answer pair.

    Processes each entry by:
    1. Extracting question and answer fields (supports nested dot notation)
    2. Translating using the configured method
    3. Storing translated fields with '_translated' suffix
    4. For TRANK: Also storing ranking metadata ('ranks', 'raw_ranks')

    Args:
        mmlu_data: List of benchmark entries (dictionaries)
        target_lang: Target language name
        cfg: Configuration object

    Returns:
        List of dictionaries with original and translated fields
    """
    method_type = cfg.task_config.method
    question_fields = cfg.task_config.question_fields
    answer_fields = cfg.task_config.answer_fields
    agent_check = False

    if method_type == "SC":
        agent_check = cfg.task_config.agent_check
    
    translated_data = []
    for item in tqdm(mmlu_data, desc=f"Translating benchmark", total=len(mmlu_data)):
        translated_dict = item.copy()

        for q_field in question_fields:
            question = get_data_key(item, q_field, default = "") 

            if question == "":
                translated_dict[f"{q_field}_translated"] = ""
                if agent_check:
                    translated_dict[f"{q_field}_corrected"] = ""

            
            for a_field in answer_fields:
                choices = get_data_key(item, a_field)

                if choices == None:
                    translated_dict[f"{a_field}_translated"] = None
                    if agent_check:
                        translated_dict[f"{a_field}_corrected"] = None
                    continue
                
                if agent_check:
                    translated, corrected = translate_question(choices, question, target_lang, cfg=cfg)
                    translated_dict[f"{q_field}_translated"] = translated[0]
                    translated_dict[f"{a_field}_translated"] = translated[1]
                    translated_dict[f"{q_field}_corrected"] = corrected[0]
                    translated_dict[f"{a_field}_corrected"] = corrected[1]
                else:
                    if method_type == "TRANK": 
                        translated_question, translated_choices, reasoning, ranks_raw = translate_question(choices, question, target_lang, cfg=cfg)
                        translated_dict[f"{q_field}_translated"] = translated_question
                        translated_dict[f"{a_field}_translated"] = translated_choices
                        translated_dict[f"ranks"] = reasoning
                        translated_dict[f"raw_ranks"] = ranks_raw
                    else:
                        translated_question, translated_choices = translate_question(choices, question, target_lang, cfg=cfg)
                        translated_dict[f"{q_field}_translated"] = translated_question
                        translated_dict[f"{a_field}_translated"] = translated_choices
        
        translated_data.append(translated_dict)

    return translated_data

def process_shard(shard, target_lang, cfg=None):
    """
    Worker function to process a shard of data.
    """
    try:
        translated = iterate_and_translate_benchmark(
            shard,
            target_lang=target_lang,
            cfg=cfg
        )
        return translated
    except Exception as e:
        print(f"Error processing shard: {e}")
        traceback.print_exc()
        return []

def run_subset(sample_data, cfg):    
    
    target_language = cfg.task_config.target_language

    if cfg.task_config.max_workers:
        num_processes = cpu_count() - 1
    else:
        num_processes = cfg.task_config.num_workers
    print(f"Using {num_processes} processes for multiprocessing.")

    total_size = len(sample_data)
    shard_size = total_size // num_processes + (total_size % num_processes > 0)

    shards = []
    for i in range(num_processes):
        start = i * shard_size
        end = min((i + 1) * shard_size, total_size)
        if start >= end:
            break
        shard = sample_data.select(range(start, end))
        shards.append(list(shard)) 

    print(f"Total examples: {total_size}")
    print(f"Number of shards: {len(shards)}")
    print(f"Size of each shard: {shard_size}")

    worker_func = partial(
        process_shard,
        target_lang=target_language,
        cfg=cfg
    )

    with Pool(processes=num_processes) as pool:
        results = []
        for result in tqdm(pool.imap(worker_func, shards), total=len(shards), desc="Processing shards"):
            results.extend(result)

    return results

def run_benchmark_translation(cfg: Config) -> None:
    """
    Main entry point for benchmark translation.

    Orchestrates the full translation pipeline:
    1. Authenticates with HuggingFace Hub if token provided
    2. Iterates over specified subsets and splits
    3. Loads benchmark data from HuggingFace datasets
    4. Distributes translation work across worker processes
    5. Saves results to JSON file
    6. Optionally pushes to HuggingFace Hub

    Output filename format:
        {output_dir}/{benchmark_name}_{subset}_{split}_{language}_{method}.json

    Args:
        cfg: Configuration object containing:
            - task_config.benchmark: Dataset source configuration
            - task_config.method: Translation method (SC/USI/TRANK/BoN)
            - task_config.target_language: Target language
            - output_dir: Directory for output files
            - save_to_hf: Whether to push to HuggingFace Hub
    """
    if len(hf_token) == 0: 
        if cfg.task_config.save_to_hf:
            raise "Please pass your Hugging Face token to proceed."
    else:
        huggingface_hub.login(token=hf_token)

    method_type = cfg.task_config.method # SC, USI, TRANK, BoN
    target_language = cfg.task_config.target_language
    language = langcodes.Language.find(target_language).language
    
    subset_list = cfg.task_config.benchmark.subset
    split_list = cfg.task_config.benchmark.split

    for subset in subset_list:
        for split in split_list:
            try:
                bench_name = cfg.task_config.benchmark.name
                dataset = load_dataset(bench_name, subset, trust_remote_code=True)
                if split != "":
                    sample_data = dataset[split]
                else:
                    sample_data = dataset
            except Exception:
                raise Exception(traceback.format_exc())
            
            if cfg.task_config.benchmark.n_entries: # sample part of dataset if specified
                sample_data = sample_data.select(range(cfg.task_config.benchmark.n_entries))

            results = run_subset(sample_data, cfg)

            sanitized_bench_name = bench_name.replace("/", "_")
            if method_type == "SC" and cfg.task_config.agent_check == False:
                method_type = "default"
            elif method_type == "SC" and cfg.task_config.agent_check == True:
                method_type = "SC"
            file_path = f'{cfg.output_dir}/{sanitized_bench_name}_{subset}_{split}_{language}_{method_type}.json'

            with open(file_path, 'w', encoding='utf-8') as filename:
                json.dump(results, filename, ensure_ascii=False, indent=2)

            print(f"Translation completed. Raw results saved to {file_path}")
        if cfg.task_config.save_to_hf:
            push_data_to_hf(cfg, sanitized_bench_name, subset, language, cfg.output_dir)

if __name__ == "__main__":
    run_benchmark_translation()