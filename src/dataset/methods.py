"""
Dataset Translation Methods
===========================

This module implements four translation methods for dataset text fields,
as described in the paper "Recovered in Translation".

These methods are optimized for translating individual text fields (strings
or lists of strings) rather than question-answer pairs. They are suitable
for machine translation datasets like FLORES and WMT.

Methods:
    1. Self-Correction (SC): Simple translation with optional verification
    2. Universal Self-Improvement (USI): Multi-candidate generation with fusion
    3. Translation Ranking (T-RANK): Multi-round competitive ranking
    4. Best-of-N (BoN): Independent scoring of N candidates

Additional Features:
    - Large text handling: Automatic chunking for texts > 10k words
    - List support: Can translate lists of strings maintaining count

Reference:
    Yukhymenko, H., Alexandrov, A., & Vechev, M. (2025).
    Recovered in Translation: Efficient Pipeline for Automated Translation
    of Benchmarks and Datasets.
"""

import openai
import random
import os
import warnings
from collections import defaultdict
import time
import langcodes
from pydantic import BaseModel, Field
from typing import List
import huggingface_hub
warnings.filterwarnings("ignore")
from .utils import generate_pos_combinations, extract_output, extract_corrected_translation_trank, parse_ranks, get_prompt_template, fill_multi_gen_check_prompt, prompt_openai_model, resample_text_list, split_text_into_chunks, trank_get_final_prompt_template
from .model_factory import prompt_llm_model

def translate_large_text(text, system_prompt, cfg, temp, judge, schema, max_words=10000):
    """
    Translate large texts by splitting into sentence-boundary chunks.

    For texts exceeding max_words, splits at sentence boundaries to maintain
    coherence, translates each chunk separately, then concatenates results.

    Args:
        text: Large text to translate
        system_prompt: System prompt for LLM
        cfg: Configuration object
        temp: Temperature for generation
        judge: Whether this is a judge model call
        schema: Pydantic schema for structured output
        max_words: Maximum words per chunk (default 10000)

    Returns:
        Concatenated translated text
    """
    chunks = split_text_into_chunks(text, max_words=max_words)
    translated_chunks = []
    print('text too big, splitting...')
    for chunk in chunks:
        user_prompt_translate, _ = get_prompt_template(chunk, cfg.task_config.target_language, cfg.task_config.method, cfg)
        prompt = user_prompt_translate
        # print(chunk)
        translated_chunk = prompt_llm_model(system_prompt, prompt, cfg, temp, judge, schema, "structured")
        print('chunk translated')
        translated_chunks.append(translated_chunk)
    print('done')
    return " ".join(translated_chunks)

def translate_using_sc(original_text, target_lang, cfg):
    """
    Translate text using Self-Correction (SC) method.

    Simple 0-shot prompting with optional self-check. Handles large texts
    by automatic chunking at sentence boundaries.

    Args:
        original_text: Text to translate (string or list)
        target_lang: Target language name
        cfg: Configuration with agent_check flag

    Returns:
        Without check: translated_text
        With check: [translated_text, corrected_text]
    """
    check = cfg.task_config.agent_check
    max_words = 5000

    user_prompt_translate, user_prompt_check = get_prompt_template(original_text, target_lang, 'SC', cfg)
    system_prompt = f"You are a professional translator with degree in Linguistics that translates English texts into {target_lang}."
    temp = cfg.task_config.temperature_translator 
    try_counter_translate = 0
    translated_text = None 

    class TranslationOutput(BaseModel):
        translation_final: str = Field(description=f"Translated text string into {target_lang}.")

    answer_list = False
    if type(original_text) == list: # default assumed type == str
        answer_list = True
        class TranslationOutput(BaseModel):
            translation_final: List[str] = Field(description=f"Translated list of text strings into {target_lang}.")

    while translated_text is None:
        try:
            try_counter_translate += 1
            if try_counter_translate > 1: # if model fails to format translated output well then regenerate
                temp = 0.5
            elif try_counter_translate > 5:
                print('Translation failed')
                return [None, None]
            # translated_text = prompt_openai_model_structured(system_prompt, user_prompt_translate, target_lang, cfg, None, temp, False)
            total_words = len(user_prompt_translate.split())
            if total_words > max_words:
                translated_text = translate_large_text(user_prompt_translate, system_prompt, cfg, temp, False, TranslationOutput, max_words=max_words)
            else:
                translated_text = prompt_llm_model(system_prompt, user_prompt_translate, cfg, temp, False, TranslationOutput, "structured")
        except Exception as e:
            # print('Error for question: ', original_text, ';', e)
            print('Error for question: ', ';', e)

    if check:
        if answer_list:
            class CorrectedTranslationOutput(BaseModel):
                translation_final: List[str] = Field(description=f"Corrected translated list of text strings into {target_lang}.")
        else:
            class CorrectedTranslationOutput(BaseModel):
                translation_final: str = Field(description=f"Corrected translated text into {target_lang}.")

        user_prompt_check_prepared = user_prompt_check.replace('<translated_text>', translated_text)
        temp = cfg.task_config.temperature_judge
        corrected_text = None
        try_counter_check = 0
        while corrected_text is None: 
            try:
                try_counter_check += 1
                if try_counter_check > 1: # if model fails to format translated output well then regenerate
                    temp = 0.7
                elif try_counter_check > 5:
                    print('Translation failed')
                    return [None, None]
                
                # corrected_text = prompt_openai_model_structured(system_prompt, user_prompt_check_prepared, target_lang, cfg, CorrectedTranslationOutput, temp, True)
                corrected_text = prompt_llm_model(system_prompt, user_prompt_check_prepared, cfg, temp, True, CorrectedTranslationOutput, "structured")
                if answer_list and len(corrected_text) != len(original_text):
                    if corrected_text == "skip":
                        corrected_text = ["skip"]
                    else:
                        corrected_text = resample_text_list(corrected_text, original_text, target_lang, cfg)

            except Exception as e:
                print('Error for text: ', original_text, ';', e)
        return [translated_text, corrected_text]
        
    else:
        if answer_list and len(translated_text) != len(original_text):
            print('fixing')
            if translated_text == "skip":
                translated_text = ["skip"]
            else:
                print(len(translated_text), len(original_text))
                while len(translated_text) != len(original_text):
                    translated_text = resample_text_list(translated_text, original_text, target_lang, cfg)
                print('fixed')
        return translated_text
    

def translate_using_usi(original_text, target_lang, cfg):
    """
    Translate text using Universal Self-Improvement (USI) method.

    Samples N candidate translations, then uses an evaluator LLM to combine
    the best features into a refined output. Particularly suitable for
    short dataset texts like FLORES and WMT entries.

    Model calls: N + 1

    Args:
        original_text: Text to translate (string or list)
        target_lang: Target language name
        cfg: Configuration with n_samples, multi_prompt settings

    Returns:
        corrected_text (fused best translation)
    """
    no_samples = cfg.task_config.n_samples

    user_prompt_translate, user_prompt_check = get_prompt_template(original_text, target_lang, 'USI', cfg)
    system_prompt = f"You are a professional translator with degree in Linguistics that translates English texts into {target_lang}."
    temp = cfg.task_config.temperature_translator 

    answer_list = False
    if type(original_text) == list: # default assumed type == str
        answer_list = True

    translated_text_samples = []
    attempts = 0

    for prompt in user_prompt_translate:
        translated_text_1_prompt = []
        while len(translated_text_1_prompt) < no_samples:
            
            translated_text = None 
            while translated_text is None:
                try:
                    if attempts > 5:
                        translated_text = "fail"
                        attempts = 0
                        # print('failed to translate text: ', original_text)
                        break
                    translated_text = prompt_llm_model(system_prompt, prompt, cfg, temp, False, None, "structured")
                except Exception as e:
                    print('Error for question: ', original_text, ';', e)
                attempts += 1
            translated_text_1_prompt.append(translated_text)
        translated_text_samples.extend(translated_text_1_prompt)

    assert len(translated_text_samples) == no_samples * len(user_prompt_translate)
    user_prompt_check_prepared = fill_multi_gen_check_prompt(user_prompt_check, translated_text_samples)
    # print(user_prompt_check_prepared)
    corrected_text = None
    try_counter_check = 0
    temp = cfg.task_config.temperature_judge
    class CorrectedTranslationOutput(BaseModel):
        translation_final: str = Field(description=f"Corrected translated text into {target_lang}.")

    while corrected_text is None: 
        try:
            try_counter_check += 1
            if try_counter_check > 1: # if model fails to format translated output well then regenerate
                temp = 0.7
            elif try_counter_check > 5:
                print('Failed to translate question: ', original_text)
                return None
            
            # corrected_text = prompt_openai_model_structured(system_prompt, user_prompt_check_prepared, target_lang, cfg, CorrectedTranslationOutput, temp, True)
            corrected_text = prompt_llm_model(system_prompt, user_prompt_check_prepared, cfg, temp, True, CorrectedTranslationOutput, "structured")
        except Exception as e:
            print('Error for text: ', original_text, ';', e)

    if answer_list and len(corrected_text) != len(original_text):
        corrected_text = resample_text_list(corrected_text, original_text, target_lang, cfg)
    # print(corrected_text)
    return corrected_text


def translate_using_trank(original_text, target_lang, cfg):
    """
    Translate text using Translation Ranking (T-RANK) method.

    Multi-round competitive ranking with position rotation to reduce bias.
    Best for complex texts where subtle errors need to be caught.

    Process:
    1. Sample N candidates
    2. Rank in N rotated combinations
    3. Average ranks to find best
    4. Final correction step

    Model calls: 2N + 1

    Args:
        original_text: Text to translate (string or list)
        target_lang: Target language name
        cfg: Configuration with n_samples, multi_prompt settings

    Returns:
        [corrected_text, all_ranks, raw_ranks]
    """
    no_samples = cfg.task_config.n_samples

    user_prompt_translate, user_prompt_check = get_prompt_template(original_text, target_lang, 'TRANK', cfg)
    system_prompt = f"You are a professional translator with degree in Linguistics that translates English texts into {target_lang}."
    temp = cfg.task_config.temperature_translator 

    answer_list = False
    if type(original_text) == list: # default assumed type == str
        answer_list = True
    
    translated_text_samples = []

    for prompt in user_prompt_translate:
        
        translated_text_1_prompt = []
        attempts = 0
        while len(translated_text_1_prompt) < no_samples:     
            translated_text = None 
            while translated_text is None:
                try:
                    if attempts > 3:
                        translated_text = "fail"
                        attempts = 0
                        # print('failed to translate text: ', original_text)
                        break
                    # translated_text = prompt_openai_model_structured(system_prompt, prompt, target_lang, cfg, None, temp, False)
                    translated_text = prompt_llm_model(system_prompt, prompt, cfg, temp, False, None, "structured")
                except Exception as e:
                    print(e)
                    continue
                attempts += 1
            translated_text_1_prompt.append(translated_text)
        translated_text_samples.extend(translated_text_1_prompt)
        
    assert len(translated_text_samples) == no_samples * len(user_prompt_translate)
    user_prompt_check_prepared = fill_multi_gen_check_prompt(user_prompt_check, translated_text_samples)
    # print(user_prompt_check_prepared)
    translated_text = None
    try_counter_check = 0
    final_correction_attempts = 0
    max_final_correction_attempts = 3
    combinations = generate_pos_combinations(no_samples*len(user_prompt_translate)) # len(combinations) = no_samples
    random.shuffle(combinations)
    all_ranks = defaultdict(list) 
    raw_ranks = []
    while translated_text is None:
        try:
            best_corrected_text = []
            for i in range(len(combinations)):
                indices = combinations[i]
                ranks = None
                
                current_texts = [translated_text_samples[i] for i in indices]
                
                try_counter_check = 0
                translated_text = None

                user_prompt_check_prepared = fill_multi_gen_check_prompt(user_prompt_check, current_texts)
                while ranks is None:
                    try:
                        try_counter_check += 1
                        temp = cfg.task_config.temperature_judge if try_counter_check <= 1 else 0.5
                        if try_counter_check > 5:
                            break
                        corrected_output = prompt_llm_model(system_prompt, user_prompt_check_prepared, cfg, temp, True, None, "base")
                        parsed_dict = extract_corrected_translation_trank(corrected_output, cfg)
                        if parsed_dict is not None:
                            if "best_translation" in parsed_dict and "rankings_list" in parsed_dict:
                                best_corrected_text.append(parsed_dict["best_translation"])
                                ranking = parsed_dict["rankings_list"]
                            else:
                                # If keys are missing, skip this iteration
                                ranking = []
                                continue
                        else:
                            ranking = parse_ranks(corrected_output)
                            if ranking is not None:
                                best_corrected_text.append(translated_text_samples[indices[ranking.index(1)]])
                            else:
                                ranking = []
                                continue
                        if all(r == 1 for r in ranking): # special case for equal ranking for all responses
                            rank_dict = {i: 1 for i in range(1, len(ranking) + 1)}
                        elif len(ranking) < no_samples: # avoid having missing ranks for some responses
                            ranks = None
                            continue
                        else:
                            rank_dict = {response: rank for rank, response in enumerate(ranking, start=1)}
                        ranks = [rank_dict[response] for response in range(1, len(ranking) + 1)]

                        for pos, rank_val in enumerate(ranks):
                            orig_index = indices[pos]
                            all_ranks[orig_index].append(rank_val)
                        raw_ranks.append(ranks)
                        break
                    except Exception as e:
                        continue
            
            avg_ranks = {}
            for idx in range(len(translated_text_samples)):
                if all_ranks[idx]:
                    avg_ranks[idx] = sum(all_ranks[idx]) / len(all_ranks[idx])
                else:
                    avg_ranks[idx] = None
            if len(avg_ranks) == 0:
                print('No ranks found')
                return [None, None, None]
            best_index = min(avg_ranks, key=lambda i: avg_ranks[i])
            translated_text = translated_text_samples[best_index]
            winning_candidates = {}
            for idx, ranks in all_ranks.items():
                if ranks and 1 in ranks and avg_ranks[idx] is not None:
                    winning_candidates[idx] = avg_ranks[idx]

            if winning_candidates:
                best_index = min(winning_candidates, key=lambda i: winning_candidates[i])
    
                candidate_ranks = all_ranks[best_index]
                win_iterations = [i for i, rank in enumerate(candidate_ranks) if rank == 1]
    
                selected_iteration = None
                for iteration in win_iterations:
                    corrected_text = best_corrected_text[iteration]
                    if corrected_text != translated_text_samples[best_index]:
                        selected_iteration = iteration
                        break

                if selected_iteration is None:
                    selected_iteration = win_iterations[0]
                
                translated_text = best_corrected_text[selected_iteration]

            # now do the final check of selected best ranked candidate and fix if needed

            class FinalTranslationOutput(BaseModel):
                summary: str = Field(description=f"Summary of provided candidates and if correction of best selection is needed.")
                if not answer_list:
                    translation_final: str = Field(description=f"Corrected final translation in {target_lang}.")
                else:
                    translation_final: List[str] = Field(description=f"Corrected final translation in {target_lang}.")

            final_check_prompt = trank_get_final_prompt_template(original_text, translated_text, target_lang, cfg)
            try:
                corrected_output_dict = prompt_llm_model(system_prompt, final_check_prompt, cfg, 0.1, True, FinalTranslationOutput, "structured")
                # Handle both dict/object and str return types
                if isinstance(corrected_output_dict, str):
                    corrected_text = corrected_output_dict
                else:
                    corrected_text = corrected_output_dict.translation_final
                if answer_list and len(corrected_text) != len(original_text):
                    corrected_text = resample_text_list(corrected_text, corrected_text, original_text, cfg)
            except Exception as e:
                final_correction_attempts += 1
                print(f'Final correction failed: {e}')
                if final_correction_attempts >= max_final_correction_attempts:
                    print('Final correction failed too many times. Returning best available.')
                    return [translated_text, all_ranks, raw_ranks]
                # Otherwise, retry only the correction step, not the whole ranking
                continue

        except Exception as e:
            print('Error for text: ', original_text, ';', e)
            continue
        
    return [corrected_text, all_ranks, raw_ranks]


def translate_using_best_of_n(original_text, target_lang, cfg):
    """
    Translate text using Best-of-N (BoN) sampling method.

    Samples N translations at higher temperature, scores each 1-10,
    and selects the highest-scored translation.

    Note: Lower quality than USI/T-RANK due to LLM limitations in
    numerical scoring and positional bias.

    Model calls: N + 1

    Args:
        original_text: Text to translate (string or list)
        target_lang: Target language name
        cfg: Configuration with n_samples setting

    Returns:
        translated_text (best scored translation)
    """
    translated_text = ''
    no_samples = cfg.task_config.n_samples

    user_prompt_translate, user_prompt_check = get_prompt_template(original_text, target_lang, 'BoN', cfg)
    system_prompt = f"You are a professional translator with degree in Linguistics that translates English texts into {target_lang}."
    temp = cfg.task_config.temperature_translator 
    translated_text_samples = []

    answer_list = False
    if type(original_text) == list: # default assumed type == str
        answer_list = True

    while len(translated_text_samples) < no_samples:     
        translated_text = None 
        while translated_text is None:
            try:
                # translated_text = prompt_openai_model_structured(system_prompt, user_prompt_translate, target_lang, cfg, None, temp, False)
                translated_text = prompt_llm_model(system_prompt, user_prompt_translate, cfg, temp, False, None, "structured")
            except Exception as e:
                print('Error for question: ', original_text, ';', e)
        translated_text_samples.append(translated_text)
    
    class ScoredOutput(BaseModel):
        scores_list: list[int] = Field(description=f"List of score numbers.")

    user_prompt_check_prepared = fill_multi_gen_check_prompt(user_prompt_check, translated_text_samples)
    translated_text = None
    try_counter_check = 0
    temp = cfg.task_config.temperature_judge
    while translated_text is None: 
        try:
            try_counter_check += 1
            if try_counter_check > 1: # if model fails to format translated output well then regenerate
                temp = 0.7
            elif try_counter_check > 5:
                print('Failed to translate text: ', original_text)
                return None
            # corrected_output = prompt_openai_model(system_prompt, user_prompt_check_prepared, cfg, temp, True)
            corrected_output = prompt_llm_model(system_prompt, user_prompt_check_prepared, cfg, temp, True, ScoredOutput, "structured", "parsing")
            scores = corrected_output.scores_list
            translated_text = translated_text_samples[scores.index(max(scores))]
            if answer_list and len(translated_text) != len(original_text):
                translated_text = resample_text_list(translated_text, original_text, target_lang, cfg)
        except Exception as e:
            print('Error for text: ', original_text, ';', e)

    return translated_text