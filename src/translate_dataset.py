"""
Dataset Translation Pipeline
============================

This module handles the translation of text datasets (e.g., FLORES, WMT) from
English to target languages using LLM-based translation methods.

Unlike benchmark translation, dataset translation processes individual text
fields without the question-answer structure. This is suitable for:
- Machine translation evaluation datasets (FLORES, WMT24++)
- Parallel corpora creation
- General text translation tasks

The pipeline:
1. Loads dataset from HuggingFace datasets
2. Distributes work across multiple processes
3. Translates specified fields using the selected method
4. Saves results to JSON and optionally pushes to HuggingFace Hub

Functions:
    translate_text: Route to appropriate translation method
    iterate_and_translate_dataset: Process dataset entries sequentially
    process_shard: Worker function for parallel processing
    run_dataset_translation: Main entry point for dataset translation
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
from .dataset.methods import translate_using_sc, translate_using_usi, translate_using_trank, translate_using_best_of_n
from .dataset.utils import get_data_key
from .dataset.save_to_hf import push_data_to_hf
from credentials import hf_token
warnings.filterwarnings("ignore")


def translate_text(original_text, target_lang, cfg=None):
    """
    Translate text to target language using the configured method.

    Routes to the appropriate translation method based on configuration:
    - SC: Self-Correction with optional verification
    - USI: Universal Self-Improvement with candidate fusion
    - TRANK: Translation Ranking with multi-round competitive ranking
    - BoN: Best-of-N with independent scoring

    Args:
        original_text: Text to translate (string or list of strings)
        target_lang: Target language name (e.g., "Ukrainian")
        cfg: Configuration object with method and model settings

    Returns:
        For SC without check: translated_text
        For SC with check: [translated_text, corrected_text]
        For USI/BoN: corrected_text
        For TRANK: [corrected_text, ranking_scores, raw_ranks]

    Raises:
        Exception: If invalid method type is specified
    """
    method_type = cfg.task_config.method

    if original_text is str and original_text is None:
        return None
    elif original_text is list and len(original_text) == 0:
        return []

    if method_type == 'SC':
        return translate_using_sc(original_text, target_lang, cfg)
    elif method_type == 'USI':
        return translate_using_usi(original_text, target_lang, cfg)
    elif method_type == 'TRANK':
        return translate_using_trank(original_text, target_lang, cfg)
    elif method_type == 'BoN':
        return translate_using_best_of_n(original_text, target_lang, cfg)
    else:
        raise Exception('Please choose a valid prompting method type (SC/USI/TRANK/BoN).')


def iterate_and_translate_dataset(data, target_lang, cfg=None):
    """
    Iterate through dataset entries and translate specified fields.

    Processes each entry by:
    1. Extracting fields specified in config (supports nested dot notation)
    2. Translating each field using the configured method
    3. Storing translated fields with '_translated' suffix
    4. For TRANK: Also storing ranking metadata

    Args:
        data: List of dataset entries (dictionaries)
        target_lang: Target language name
        cfg: Configuration object

    Returns:
        List of dictionaries with original and translated fields
    """
    columns = cfg.task_config.fields
    method_type = cfg.task_config.method
    agent_check = False

    if method_type == "SC":
        agent_check = cfg.task_config.agent_check
    
    translated_data = []
    for item in tqdm(data, desc=f"Translating dataset", total=len(data)):

        translated_dict = item.copy()
    
        for field in columns:
            # original_text = item.get(field, "")
            original_text = get_data_key(item, field, default="")
            if original_text == "":
                translated_dict[f"{field}_translated"] = ""
                if agent_check:
                    translated_dict[f"{field}_corrected"] = ""
                continue
                
            if agent_check:
                translated, corrected = translate_text(original_text, target_lang, cfg)
                translated_dict[f"{field}_translated"] = translated
                translated_dict[f"{field}_corrected"] = corrected
            else:
                if method_type == "TRANK": 
                    translated_text, reasoning, ranks_raw = translate_text(original_text, target_lang, cfg)
                    translated_dict[f"{field}_translated"] = translated_text
                    translated_dict[f"{field}_ranks"] = reasoning
                    translated_dict[f"{field}_raw_ranks"] = ranks_raw
                else:
                    translated_text = translate_text(original_text, target_lang, cfg)
                    translated_dict[f"{field}_translated"] = translated_text

        translated_data.append(translated_dict)

    return translated_data

def process_shard(shard, target_lang, cfg=None):
    """
    Worker function to process a shard of data.
    """

    try:
        translated = iterate_and_translate_dataset(
            shard,
            target_lang=target_lang,
            cfg=cfg
        )
        return translated
    except Exception as e:
        print(f"Error processing shard: {e}")
        traceback.print_exc()
        return []

def run_dataset_translation(cfg: Config) -> None:
    """
    Main entry point for dataset translation.

    Orchestrates the full translation pipeline:
    1. Authenticates with HuggingFace Hub if token provided
    2. Iterates over specified subsets and splits
    3. Loads dataset from HuggingFace datasets
    4. Distributes translation work across worker processes
    5. Saves results to JSON file
    6. Optionally pushes to HuggingFace Hub

    Output filename format:
        {output_dir}/{dataset_name}_{subset}_{split}_{language}_{method}.json

    Args:
        cfg: Configuration object containing:
            - task_config.dataset: Dataset source configuration
            - task_config.method: Translation method (SC/USI/TRANK/BoN)
            - task_config.target_language: Target language
            - task_config.fields: List of fields to translate
            - output_dir: Directory for output files
            - save_to_hf: Whether to push to HuggingFace Hub
    """
    if len(hf_token) == 0: 
        if cfg.task_config.save_to_hf:
            raise "Please pass your Hugging Face token to proceed."
    else:
        huggingface_hub.login(token=hf_token)
    
    subset_list = cfg.task_config.dataset.subset
    split_list = cfg.task_config.dataset.split

    for subset in subset_list:
        for split in split_list:

            try:
                bench_name = cfg.task_config.dataset.name
                dataset = load_dataset(bench_name, subset)
                sample_data = dataset[split]
                sample_data = sample_data
            except Exception:
                raise Exception(traceback.format_exc())

            if cfg.task_config.dataset.n_entries: # sample part of dataset if specified
                sample_data = sample_data.select(range(cfg.task_config.dataset.n_entries))

            method_type = cfg.task_config.method
            target_language = cfg.task_config.target_language
            language = langcodes.Language.find(target_language).language
            
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
                    break  # No more data to assign
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

            sanitized_bench_name = bench_name.replace("/", "_")
            if method_type == "SC" and cfg.task_config.agent_check == False:
                method_type = "default"
            elif method_type == "SC" and cfg.task_config.agent_check == True:
                method_type = "SC"
            file_path = f'{cfg.output_dir}/{sanitized_bench_name}_{subset}_{split}_{language}_{method_type}.json'

            with open(file_path, 'w', encoding='utf-8') as filename:
                json.dump(results, filename, ensure_ascii=False, indent=2)

            print(f"Translation completed. Results saved to {file_path}")
        if cfg.task_config.save_to_hf:
            push_data_to_hf(cfg, sanitized_bench_name, subset, language, cfg.output_dir)

if __name__ == "__main__":
    run_dataset_translation()