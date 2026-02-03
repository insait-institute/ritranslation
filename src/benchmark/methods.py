"""
Benchmark Translation Methods
=============================

This module implements four translation methods for benchmark question-answer
pairs, as described in the paper "Recovered in Translation".

Methods:
    1. Self-Correction (SC): Simple 0-shot translation with optional verification
    2. Universal Self-Improvement (USI): Multi-candidate generation with fusion
    3. Translation Ranking (T-RANK): Multi-round competitive ranking (proposed)
    4. Best-of-N (BoN): Independent scoring of N candidates

Key findings from the paper:
- USI is more suitable for short and simple dataset translation
- T-RANK shows better performance for benchmarks with complex question structures
- T-RANK's competitive ranking helps identify subtle translation errors that
  other methods fail to correct

Model Calls per Entry:
- SC: 1-2 calls (translation + optional check)
- USI: N + 1 calls (N candidates + fusion)
- BoN: N + 1 calls (N candidates + scoring)
- T-RANK: 2N + 1 calls (N candidates + N rankings + final correction)

Reference:
    Yukhymenko, H., Alexandrov, A., & Vechev, M. (2025).
    Recovered in Translation: Efficient Pipeline for Automated Translation
    of Benchmarks and Datasets.
"""

import openai
import random
import warnings
from collections import defaultdict
import time
import langcodes
from pydantic import BaseModel, Field
from typing import List
import huggingface_hub
from .utils import generate_pos_combinations, extract_choices, extract_output, extract_corrected_translation_trank, parse_ranks, get_prompt_template, fill_multi_gen_check_prompt, prompt_openai_model, prompt_openai_model_structured, resample_answers, trank_get_final_prompt_template
from .model_factory import prompt_llm_model
warnings.filterwarnings("ignore")


def translate_using_sc(choices, question, target_lang, cfg=None):
    """
    Translate using Self-Correction (SC) method.

    Simple 0-shot prompting with optional self-check stage. After translation,
    the model (in a new chat with no history) can evaluate and correct the
    result with respect to the original text content.

    Best for: Large text translation into high-resource languages where
    translation capabilities are sufficient.

    Args:
        choices: Answer options (list of strings or single string)
        question: The question text to translate
        target_lang: Target language name
        cfg: Configuration with agent_check flag

    Returns:
        Without check: [translated_question, translated_choices]
        With check: [[translated_q, translated_c], [corrected_q, corrected_c]]
    """
    check = cfg.task_config.agent_check

    user_prompt_translate, user_prompt_check = get_prompt_template(question, choices, target_lang, 'SC', cfg)
    system_prompt = f"You are a professional translator with degree in Linguistics that translates English texts into {target_lang}."
    temp = cfg.task_config.temperature_translator
    try_counter_translate = 0
    translated_question = translated_choices = None 

    answer_str = False
    if type(choices) == str: # default assumed type == list
        answer_str = True
    class TranslationOutput(BaseModel):
        question_final: str = Field(description=f"Translated question text into {target_lang}.")
        if answer_str:
            answers_final: str = Field(description=f"Translated answer in {target_lang}.")
        else:
            answers_final: List[str] = Field(description=f"Translated answer options list in {target_lang}.")

    while translated_question is None or translated_choices is None:
        try:
            try_counter_translate += 1
            if try_counter_translate > 1: # if model fails to format translated output well then regenerate
                temp = 0.5
            # translated_question, translated_choices = prompt_openai_model_structured(system_prompt, user_prompt_translate, target_lang, cfg, TranslationOutput, temp, False)
            translated_dict = prompt_llm_model(system_prompt, user_prompt_translate, cfg, temp, False, TranslationOutput, "structured")
            translated_question = translated_dict.question_final
            translated_choices = translated_dict.answers_final
        except Exception as e:
            print('Error for question: ', question, ';', e)
            continue
    if check:
        class CorrectedTranslationOutput(BaseModel):
            question_final: str = Field(description=f"Corrected translated question text into {target_lang}.")
            if answer_str:
                answers_final: str = Field(description=f"Corrected translated answer in {target_lang}.")
            else:
                answers_final: List[str] = Field(description=f"Corrected translated answer options list in {target_lang}.")

        user_prompt_check_prepared = user_prompt_check.replace('<translated_question>', translated_question).replace('<translated_choices>', str(translated_choices))
        corrected_question = None
        corrected_choices = None
        try_counter_check = 0
        temp = cfg.task_config.temperature_judge
        while corrected_question is None or corrected_choices is None: 
            try:
                try_counter_check += 1
                if try_counter_check > 1: # if model fails to format translated output well then regenerate
                    temp = 0.7
                elif try_counter_check > 5:
                    print('Failed to translate question: ', question)
                    return [[None, None], [None, None]]
                # corrected_question, corrected_choices = prompt_openai_model_structured(system_prompt, user_prompt_check_prepared, target_lang, cfg, CorrectedTranslationOutput, temp, True)
                corrected_dict = prompt_llm_model(system_prompt, user_prompt_check_prepared, cfg, temp, True, CorrectedTranslationOutput, "structured")
                corrected_question = corrected_dict.question_final
                corrected_choices = corrected_dict.answers_final
                if not answer_str and len(corrected_choices) != len(choices):
                    corrected_choices = resample_answers(corrected_choices, corrected_question, choices, cfg)
            except Exception:
                continue

        return [translated_question, translated_choices], [corrected_question, corrected_choices]
        
    else:
        return [translated_question, translated_choices]
    

def translate_using_usi(choices, question, target_lang, cfg=None):
    """
    Translate using Universal Self-Improvement (USI) method.

    Building on Universal Self-Consistency and Fusion-of-N, this method samples
    N candidate translations using higher temperature, then presents them to an
    evaluator LLM with instructions to combine the candidates into the best
    version according to specified criteria.

    The method operates on the principle that the most consistent translation
    is not necessarily the best, and uses fusion to combine the best features
    from multiple candidates.

    Model calls: N + 1 (N candidates + 1 fusion call)

    Best for: Short and simple translations; cost-efficient for low-resource
    languages where it can successfully address language-specific features.

    Args:
        choices: Answer options (list of strings or single string)
        question: The question text to translate
        target_lang: Target language name
        cfg: Configuration with n_samples, multi_prompt settings

    Returns:
        [corrected_question, corrected_choices]
    """
    no_samples = cfg.task_config.n_samples

    user_prompt_translate, user_prompt_check = get_prompt_template(question, choices, target_lang, 'USI', cfg)
    system_prompt = f"You are a professional translator with degree in Linguistics that translates English texts into {target_lang}."

    temp = cfg.task_config.temperature_translator
    counter_failed_prompts = 0

    translated_question_samples = []
    translated_choices_samples = []

    answer_str = False
    if type(choices) == str: # default assumed type == list
        answer_str = True
    class TranslationOutput(BaseModel):
        question_final: str = Field(description=f"Translated question text into {target_lang}.")
        if answer_str:
            answers_final: str = Field(description=f"Translated answer in {target_lang} (string).")
        else:
            answers_final: List[str] = Field(description=f"Translated answer options list in {target_lang} (list).")

    for prompt in user_prompt_translate:
        
        translated_q_1_prompt = []
        translated_a_1_prompt = []
        while len(translated_q_1_prompt) < no_samples or len(translated_a_1_prompt) < no_samples:
            translated_question = translated_choices = None 
            counter_n_candidate_fails = 0
            while translated_question is None or translated_choices is None:
                try:
                    if counter_n_candidate_fails > 5:
                        translated_q_1_prompt.append('skip')
                        if answer_str:
                            translated_a_1_prompt.append('skip')
                        else:
                            translated_a_1_prompt.append(['skip'])
                        break
                    translated_dict = prompt_llm_model(system_prompt, prompt, cfg, temp, False, TranslationOutput, "structured")
                    translated_question = translated_dict.question_final
                    translated_choices = translated_dict.answers_final
                    if "skip" in translated_question:
                        counter_n_candidate_fails += 1
                except Exception as e:
                    print('Error for question: ', question, ';', e)
                    counter_n_candidate_fails += 1  # <-- Add this line
                    continue
            translated_q_1_prompt.append(translated_question)
            translated_a_1_prompt.append(translated_choices)
        translated_question_samples.extend(translated_q_1_prompt)
        translated_choices_samples.extend(translated_a_1_prompt)

    assert len(translated_question_samples) == no_samples * len(user_prompt_translate) == len(translated_choices_samples)

    user_prompt_check_prepared = fill_multi_gen_check_prompt(user_prompt_check, translated_question_samples, translated_choices_samples)
    # print(user_prompt_check_prepared)
    corrected_question = None
    corrected_choices = None
    try_counter_check = 0
    temp = cfg.task_config.temperature_judge
    class CorrectedTranslationOutput(BaseModel):
        question_final: str = Field(description=f"Corrected translated question text into {target_lang}.")
        if answer_str:
            answers_final: str = Field(description=f"Corrected translated answer in {target_lang}.")
        else:
            answers_final: List[str] = Field(description=f"Corrected translated answer options list in {target_lang}.")

    while corrected_question is None or corrected_choices is None: 
        try:
            try_counter_check += 1
            if try_counter_check > 1: # if model fails to format translated output well then regenerate
                temp = 0.7
            elif try_counter_check > 3:
                print('Failed to translate question: ', question)
                return [None, None]
            # corrected_question, corrected_choices = prompt_openai_model_structured(system_prompt, user_prompt_check_prepared, target_lang, cfg, CorrectedTranslationOutput, temp, True)
            corrected_dict = prompt_llm_model(system_prompt, user_prompt_check_prepared, cfg, temp, True, CorrectedTranslationOutput, "structured")
            corrected_question = corrected_dict.question_final
            corrected_choices = corrected_dict.answers_final
            if not answer_str and len(corrected_choices) != len(choices) and counter_failed_prompts >= 1:
                corrected_choices = resample_answers(corrected_choices, corrected_question, choices, cfg)
        except Exception:
            continue
        
    # print(corrected_question, corrected_choices)
    return [corrected_question, corrected_choices]


def translate_using_trank(choices, question, target_lang, cfg=None):
    """
    Translate using Translation Ranking (T-RANK) method.

    Our proposed method that employs multi-prompt candidate sampling and
    multi-round competitive ranking to enhance error detection. Key innovation:
    candidates are systematically presented in different positional orders
    across rounds to reduce positional bias in LLM judgment.

    Process:
    1. Sample N diverse translations with high temperature
    2. Create N ranking combinations (position rotation)
    3. For each combination, judge ranks candidates
    4. Calculate average ranks across all combinations
    5. Select best-ranked candidate
    6. Final correction step with all candidates visible

    The competitive ranking approach enables more sophisticated reasoning from
    non-reasoning models, successfully identifying subtle errors that other
    methods fail to correct.

    Model calls: 2N + 1 (N candidates + N rankings + 1 final correction)

    Best for: Benchmark translation with complex question structures;
    highest quality when cost is not a primary concern.

    Args:
        choices: Answer options (list of strings or single string)
        question: The question text to translate
        target_lang: Target language name
        cfg: Configuration with n_samples, multi_prompt settings

    Returns:
        [corrected_question, corrected_choices, all_ranks, raw_ranks]
    """
    no_samples = cfg.task_config.n_samples
    user_prompt_translate, user_prompt_check = get_prompt_template(question, choices, target_lang, 'TRANK', cfg)
    system_prompt = f"You are a professional translator with degree in Linguistics that translates English texts into {target_lang}."
    temp = cfg.task_config.temperature_translator

    translated_question_samples = []
    translated_choices_samples = []

    answer_str = False
    if type(choices) == str: # default assumed type == list
        answer_str = True
    class TranslationOutput(BaseModel):
        question_final: str = Field(description=f"Corrected translated question text into {target_lang}.")
        if answer_str:
            answers_final: str = Field(description=f"Corrected translated answer in {target_lang}.")
        else:
            answers_final: List[str] = Field(description=f"Corrected translated answer options list in {target_lang}.")

    for prompt in user_prompt_translate:
        
        translated_q_1_prompt = []
        translated_a_1_prompt = []
        while len(translated_q_1_prompt) < no_samples or len(translated_a_1_prompt) < no_samples:     
            translated_question = translated_choices = None
            while translated_question is None or translated_choices is None:
                try:
                    # translated_question, translated_choices = prompt_openai_model_structured(system_prompt, prompt, target_lang, cfg, TranslationOutput, temp, False)
                    translated_dict = prompt_llm_model(system_prompt, prompt, cfg, temp, False, TranslationOutput, "structured")
                    translated_question = translated_dict.question_final
                    translated_choices = translated_dict.answers_final
                except Exception as e:
                    print('Error for question: ', question, ';', e)
                    continue
            translated_q_1_prompt.append(translated_question)
            translated_a_1_prompt.append(translated_choices)
        translated_question_samples.extend(translated_q_1_prompt)
        translated_choices_samples.extend(translated_a_1_prompt)

    assert len(translated_question_samples) == no_samples * len(user_prompt_translate) == len(translated_choices_samples)

    user_prompt_check_prepared = fill_multi_gen_check_prompt(user_prompt_check, translated_question_samples, translated_choices_samples)
    corrected_translated_question = None
    corrected_translated_choices = None
    try_counter_check = 0
    combinations = generate_pos_combinations(no_samples*len(user_prompt_translate)) # len(combinations) = no_samples
    random.shuffle(combinations)
    all_ranks = defaultdict(list) 
    raw_ranks = []
    final_correction_attempts = 0
    max_final_correction_attempts = 3
    while corrected_translated_question is None or corrected_translated_choices is None:
        try:
            best_corrected_question = []
            best_corrected_answers = []
            for i in range(len(combinations)):
                indices = combinations[i]
                ranks = None
                ranking = None
                
                current_questions = [translated_question_samples[i] for i in indices]
                current_answers = [translated_choices_samples[i] for i in indices]
                
                try_counter_check = 0
                translated_question = None
                translated_choices = None

                user_prompt_check_prepared = fill_multi_gen_check_prompt(user_prompt_check, current_questions, current_answers)
                while ranks is None:
                    try:
                        try_counter_check += 1
                        temp = cfg.task_config.temperature_judge if try_counter_check <= 1 else 0.5
                        if try_counter_check > 5:
                            break

                        corrected_output = prompt_llm_model(system_prompt, user_prompt_check_prepared, cfg, temp, True, None, "base")
                        parsed_dict = extract_corrected_translation_trank(corrected_output, cfg)
                        if parsed_dict is not None:
                            if (
                                "best_translation" in parsed_dict and
                                "rankings_list" in parsed_dict and
                                isinstance(parsed_dict["best_translation"], dict) and
                                "best_translated_question" in parsed_dict["best_translation"] and
                                "best_translated_answers" in parsed_dict["best_translation"]
                            ):
                                best_corrected_question.append(parsed_dict["best_translation"]["best_translated_question"])
                                best_corrected_answers.append(parsed_dict["best_translation"]["best_translated_answers"])
                                ranking = parsed_dict["rankings_list"]
                            else:
                                # If keys are missing, skip this iteration
                                continue
                        else:
                            ranking = parse_ranks(corrected_output)
                            if ranking is not None:
                                best_corrected_question.append(translated_question_samples[indices[ranking.index(1)]])
                                best_corrected_answers.append(translated_choices_samples[indices[ranking.index(1)]])
                            else:
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
            for idx in range(len(translated_question_samples)):
                if all_ranks[idx]:
                    avg_ranks[idx] = sum(all_ranks[idx]) / len(all_ranks[idx])
                else:
                    avg_ranks[idx] = None

            if len(avg_ranks) == 0:
                print('No ranks found')
                return [None, None, None, None]
            best_index = min(avg_ranks, key=lambda i: avg_ranks[i])
            translated_question = translated_question_samples[best_index]
            translated_choices = translated_choices_samples[best_index]
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
                    corrected_question = best_corrected_question[iteration]
                    if corrected_question != translated_question_samples[best_index]:
                        selected_iteration = iteration
                        break
                
                if selected_iteration is None:
                    selected_iteration = win_iterations[0]
                
                translated_question = best_corrected_question[selected_iteration]
                translated_choices = best_corrected_answers[selected_iteration]

            # now do the final check of selected best ranked candidate and fix if needed

            class FinalTranslationOutput(BaseModel):
                summary: str = Field(description=f"Summary of provided candidates and if correction of best selection is needed.")
                corrected_question_final: str = Field(description=f"Corrected final translated question text into {target_lang}.")
                if answer_str:
                    corrected_answers_final: str = Field(description=f"Corrected final translated answer in {target_lang}.")
                else:
                    corrected_answers_final: List[str] = Field(description=f"Corrected final translated answer options list in {target_lang}.")
          
            final_check_prompt = trank_get_final_prompt_template(question, choices, translated_question, translated_choices, target_lang, cfg)
            try:
                corrected_output_dict = prompt_llm_model(system_prompt, final_check_prompt, cfg, 0.1, True, FinalTranslationOutput, "structured")
                # print('\n', translated_question, translated_choices, '#####', corrected_output_dict.summary, '\n')
                corrected_translated_question = corrected_output_dict.corrected_question_final
                corrected_translated_choices = corrected_output_dict.corrected_answers_final
                if not answer_str and len(corrected_translated_choices) != len(choices):
                    corrected_translated_choices = resample_answers(corrected_translated_choices, corrected_translated_question, choices, cfg)
            except Exception as e:
                final_correction_attempts += 1
                print(f'Final correction failed: {e}')
                if final_correction_attempts >= max_final_correction_attempts:
                    print('Final correction failed too many times. Returning best available.')
                    return [translated_question, translated_choices, all_ranks, raw_ranks]
                # Otherwise, retry only the correction step, not the whole ranking
                continue
        except Exception as e:
            print(f'Ranking or correction block failed: {e}')
            break

    return [corrected_translated_question, corrected_translated_choices, all_ranks, raw_ranks]


def translate_using_best_of_n(choices, question, target_lang, cfg=None):
    """
    Translate using Best-of-N (BoN) sampling method.

    Drawing from test-time compute scaling methods, this approach samples N
    translations at higher temperature for diversity, then prompts the LLM
    to score candidates 1-10 based on specified criteria, selecting the
    highest-scored translation.

    While cost-effective and language-agnostic, this method yields lower
    quality than T-RANK and USI, as LLMs exhibit limitations in numerical
    scoring and positional bias (favoring earlier candidates).

    Model calls: N + 1 (N candidates + 1 scoring call)

    Best for: Cost-effective translation when language-agnostic approach
    is needed and highest quality is not required.

    Args:
        choices: Answer options (list of strings or single string)
        question: The question text to translate
        target_lang: Target language name
        cfg: Configuration with n_samples setting

    Returns:
        [translated_question, translated_choices]
    """
    translated_question = ''
    translated_choices = []
    no_samples = cfg.task_config.n_samples

    user_prompt_translate, user_prompt_check = get_prompt_template(question, choices, target_lang, 'BoN', cfg)
    system_prompt = f"You are a professional translator with degree in Linguistics that translates English texts into {target_lang}."
    temp = cfg.task_config.temperature_translator

    translated_question_samples = []
    translated_choices_samples = []

    answer_str = False
    if type(choices) == str: # default assumed type == list
        answer_str = True
    class TranslationOutput(BaseModel):
        question_final: str = Field(description=f"Translated question text into {target_lang}.")
        if answer_str:
            answers_final: str = Field(description=f"Translated answer in {target_lang}.")
        else:
            answers_final: List[str] = Field(description=f"Translated answer options list in {target_lang}.")

    while len(translated_question_samples) < no_samples or len(translated_choices_samples) < no_samples:     
        translated_question = translated_choices = None 
        while translated_question is None or translated_choices is None:
            try:
                # translated_question, translated_choices = prompt_openai_model_structured(system_prompt, user_prompt_translate, target_lang, cfg, TranslationOutput, temp, False)
                translated_dict = prompt_llm_model(system_prompt, user_prompt_translate, cfg, temp, False, TranslationOutput, "structured")
                translated_question = translated_dict.question_final
                translated_choices = translated_dict.answers_final
            except Exception:
                continue
        translated_question_samples.append(translated_question)
        translated_choices_samples.append(translated_choices)
  
    user_prompt_check_prepared = fill_multi_gen_check_prompt(user_prompt_check, translated_question_samples, translated_choices_samples)
    translated_question = None
    translated_choices = None
    try_counter_check = 0
    temp = cfg.task_config.temperature_judge
    class ScoredOutput(BaseModel):
        scores_list: list[int] = Field(description=f"List of score numbers.")
    while translated_question is None or translated_choices is None: 
        try:
            try_counter_check += 1
            if try_counter_check > 1: # if model fails to format translated output well then regenerate
                temp = 0.7
            elif try_counter_check > 5:
                print('Failed to translate question: ', question)
                return [None, None]
            # corrected_output = prompt_openai_model(system_prompt, user_prompt_check_prepared, cfg, temp, True)
            scored_output = prompt_llm_model(system_prompt, user_prompt_check_prepared, cfg, temp, True, ScoredOutput, "structured", "parsing")
            # scores = extract_choices(corrected_output)
            scores = scored_output.scores_list
            translated_question = translated_question_samples[scores.index(max(scores))]
            translated_choices = translated_choices_samples[scores.index(max(scores))]
            if not answer_str and len(translated_choices) != len(choices):
                translated_choices = resample_answers(translated_choices, translated_question, choices, cfg)
        except Exception as e:
            print('Error for question: ', question, ';', e)
            continue

    return [translated_question, translated_choices]
