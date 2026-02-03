import openai
import os
import json
import warnings
import ast
import time
from pydantic import BaseModel, Field
from typing import List
import re
import huggingface_hub
from credentials import openai_api_key
warnings.filterwarnings("ignore")

openai.api_key = openai_api_key

def load_prompt_from_file(path: str):
    try:
        with open(path, "r") as f:
            prompt = f.read()

    except Exception:
        raise Exception(f'File <{path}> either does not exist or is corrupted.')

    return prompt

def generate_pos_combinations(n):

    rotations = []
    indices = list(range(n))
    for i in range(len(indices)):
        rotation = indices[-i:] + indices[:-i]
        rotations.append(rotation)

    return rotations

def split_text(text):

    lines = [line.strip() for line in text.split('\n') if line.strip()]

    if len(lines) > 1:
        last_line = lines[-1]
        sentences = re.split(r'(?<=[.!?])\s+', last_line)
        last_sentence = sentences[-1]
        main_text = "\n".join(lines[:-1])
    else:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 1:
            main_text = " ".join(sentences[:-1])
            last_sentence = sentences[-1]
        else:
            main_text = ""
            last_sentence = text

    return main_text, last_sentence


def prepare_base_translate_prompt(original_text, prompt, target_lang):

    prompt = prompt.replace('<original_text>', original_text.replace('\n', ' ')).replace('<target_language>', target_lang)
    
    return prompt

def get_prompt_template(original_text, target_lang, method_type, cfg):

    few_shot = cfg.task_config.few_shot
    multi_prompt = cfg.task_config.multi_prompt

    if few_shot:
        if method_type == 'SC':
            few_shot_prompt = load_prompt_from_file('./src/dataset/prompts/few_shot_sc.txt')
        elif method_type == 'TRANK':
            few_shot_prompt = "## Examples \n" + load_prompt_from_file('./src/dataset/prompts/few_shot_multi.txt')
        elif method_type == 'USI':
            few_shot_prompt = "## Examples \n" + load_prompt_from_file('./src/dataset/prompts/few_shot_usi.txt')
        else:
            few_shot_prompt = ''
    else:
        few_shot_prompt = ''

    if method_type == 'SC':
        translate_prompt = load_prompt_from_file('./src/dataset/prompts/base_prompt_translate.txt')
        translate_prompt = prepare_base_translate_prompt(original_text, translate_prompt, target_lang)
        check_prompt = load_prompt_from_file('./src/dataset/prompts/self_correction_check.txt')
        check_prompt = check_prompt.replace('<original_text>', original_text).replace('<target_language>', target_lang)
        check_prompt = check_prompt.replace('<few-shot_prompt>', few_shot_prompt)

    elif method_type == 'USI':
        if multi_prompt:
            translate_prompt = []
            prompt_files = cfg.task_config.prompt_files
            for file in prompt_files:
                prompt = load_prompt_from_file(f"./src/dataset/prompts/mq_base_translation_prompts/{file}")
                prompt = prepare_base_translate_prompt(original_text, prompt, target_lang)
                translate_prompt.append(prompt)
        else:
            translate_prompt = load_prompt_from_file('./src/dataset/prompts/base_prompt_translate.txt')
            translate_prompt = [prepare_base_translate_prompt(original_text, translate_prompt, target_lang)]
        check_prompt = load_prompt_from_file('./src/dataset/prompts/universal_self_improvement.txt')
        check_prompt = check_prompt.replace('<original_text>', original_text).replace('<target_language>', target_lang)

    elif method_type == 'TRANK':
        if multi_prompt:
            translate_prompt = []
            prompt_files = cfg.task_config.prompt_files
            for file in prompt_files:
                prompt = load_prompt_from_file(f"./src/dataset/prompts/mq_base_translation_prompts/{file}")
                prompt = prepare_base_translate_prompt(original_text, prompt, target_lang)
                prompt = prompt.replace('<few-shot_prompt>', few_shot_prompt)
                translate_prompt.append(prompt)
        else:
            translate_prompt = load_prompt_from_file('./src/dataset/prompts/base_prompt_translate.txt')
            translate_prompt = [prepare_base_translate_prompt(original_text, translate_prompt, target_lang)]
        check_prompt = load_prompt_from_file('./src/dataset/prompts/trank_rank_n.txt')
        check_prompt = check_prompt.replace('<original_text>', original_text).replace('<target_language>', target_lang)

    elif method_type == 'BoN':
        translate_prompt = load_prompt_from_file('./src/dataset/prompts/base_prompt_translate.txt')
        translate_prompt = prepare_base_translate_prompt(original_text, translate_prompt, target_lang)
        check_prompt = load_prompt_from_file('./src/dataset/prompts/best_of_n_scoring.txt')
        check_prompt = check_prompt.replace('<original_text>', original_text).replace('<target_language>', target_lang)

    else:
        raise Exception('Select a valid prompting method type (SC/USI/TRANK/BoN).')

    return translate_prompt, check_prompt


def fill_multi_gen_check_prompt_og(prompt, text_samples):
    
    samples_prompt = ''
    for i in range(len(text_samples)):
        samples_prompt += f'Response {i+1}: \n Translation: ' + text_samples[i] + '\n'
    prompt = prompt.replace('<responses>', samples_prompt)
    
    return prompt

def fill_multi_gen_check_prompt(prompt, text_samples):

    responses_str_list = []
    for i in range(len(text_samples)):
        text = text_samples[i]
        dict_entry = (
            '{\n'
            f'  "response_id": {i+1},\n'
            f'  "translated_text": """{text}""",\n'
            '}'
        )
        responses_str_list.append(dict_entry)
    
    responses_str = ",\n".join(responses_str_list)
    prompt = prompt.replace("<responses>", responses_str)

    return prompt

def prompt_openai_model(system_prompt, user_prompt, cfg, temp=0.1, judge=False):
    
    if cfg == None:
        model_name = "gpt-4o-mini-2024-07-18"
        reason_lvl = None
    else:
        if judge:
            model_name = cfg.judge_model.name
            reason_lvl = cfg.judge_model.reasoning_level
        else:
            model_name = cfg.translation_model.name
            reason_lvl = cfg.translation_model.reasoning_level
    
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_prompt} 
    ]
    output = None
    while output is None:
        try:
            if "o3" in model_name or "o1" in model_name:
                response = openai.chat.completions.create(
                    model=model_name, 
                    messages=messages, 
                    reasoning_effort=reason_lvl,
                    temperature=temp 
                )
            else:
                response = openai.chat.completions.create(
                    model=model_name, 
                    messages=messages, 
                    temperature=temp 
                )
            output = response.choices[0].message.content
        except openai.RateLimitError:
                time.sleep(30)
                continue
    return output


def prompt_openai_model_structured(system_prompt, user_prompt, target_lang, cfg, schema=None, temp=0.1, judge=False):

    class TranslationOutput(BaseModel):
        translation_final: str = Field(description=f"Translated text into {target_lang}.")

    if schema == None:
        schema = TranslationOutput

    if cfg == None:
        model_name = "gpt-4o-mini-2024-07-18"
        reason_lvl = None
    else:
        if judge:
            model_name = cfg.judge_model.name
            reason_lvl = cfg.judge_model.reasoning_level
        else:
            model_name = cfg.translation_model.name
            reason_lvl = cfg.translation_model.reasoning_level
        
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_prompt} 
    ]
    output = None
    while output is None:
        try:
            if "o3" in model_name or "o1" in model_name:
                response = openai.Client(api_key=openai.api_key).beta.chat.completions.parse(
                    model=model_name, 
                    messages=messages, 
                    temperature=temp,
                    reasoning_effort=reason_lvl,
                    response_format=schema
                )
            else:
                response = openai.Client(api_key=openai.api_key).beta.chat.completions.parse(
                    model=model_name, 
                    messages=messages, 
                    temperature=temp,
                    response_format=schema
                )
            output = response.choices[0].message.parsed
        except openai.RateLimitError:
                time.sleep(30)
                continue
        
    return output.translation_final


def parse_ranks(text):

    pattern = r'(?i)\*{0,2}"rankings_list":?\*{0,2}:?\s*(\[[^\]]+\])'
    match = re.search(pattern, text)
    if match:
        list_str = match.group(1)  # this gets the string "[2, 3, 5, 4, 1]"
        try:
            ranks = ast.literal_eval(list_str)
        except Exception as e:
            ranks = None
    else:
        ranks = None

    return ranks

def extract_output(text):
    """
    Extracts the question text after 'Translation:'.
    
    text: a string of the form:
      "Translation: some_translation"
    """
    
    pattern = r"(?i)\*{0,2}Translation\*{0,2}:\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        translation_str = match.group(1).strip()
        return translation_str
    else:
        return None, None
    
def extract_choices(text):
    """
    Extracts the question text after 'Question:' and the list of answers after 'Answers:'.
    
    text: a string of the form:
      "Answers: ['answer1', "ans'we: "wdfdefwr"2", ...]"
    """
    pattern = r"Score[s]?:\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        choices_str = match.group(1).strip()
        try:
            choices_list = ast.literal_eval(choices_str)
            return choices_list
        except Exception:
            return None
        
def extract_corrected_translation_trank(text):

    match = re.search(r'(\{.*\})', text, re.DOTALL)
    data = None
    counter = 0
    while data is None:
        counter += 1
        if counter > 1:
            return None
        if match:
            json_str = match.group(1)
            
            try:
                data_as_python = ast.literal_eval(json_str)

                corrected_json_str = json.dumps(data_as_python, ensure_ascii=False, indent=2)

                data = json.loads(corrected_json_str)
                if "rankings_list" not in data:
                    data = llm_aided_dict_parsing(text)
            except Exception as e:
                data = llm_aided_dict_parsing(text)
                continue
        else:
            try:
                data = llm_aided_dict_parsing(text)
            except:
                print("Failed to extract JSON dict.")
                continue
    return data

def llm_aided_dict_parsing(text):

    class TranslationOutput(BaseModel):
        rankings_list: List[int] = Field(description="Array containing final rankings of each response ID ranked in order according to the evalautions.")
        best_translation: str = Field(description="The best translation text.")
        summary: str = Field(description="Summary of model's reasoning and evaluation of the responses.")

    user_prompt = f'''Your task is to extract dictionary in JSON format from your previous reply. The dictionary should contain the following keys: 'rankings_list', 'best_translation' and 'reasoning'. The 'rankings_list' key should contain a list of integers representing the final rankings of each response. The 'best_translation' key should contain the best translation text. The 'reasoning' key should contain the complete detailed reasoning text analyzing quality, correctness, grammar, etc. of every response candidate. Here is the output to extract the dictionary from: \n
    {text}'''
    messages = [
        {"role": "system", "content": 'You are a helpful assistant.'}, 
        {"role": "user", "content": user_prompt} 
    ]
    output = None
    counter = 0
    while output is None:
        counter += 1
        if counter > 3:
            return None
        try:
            response = openai.Client(api_key=openai.api_key).beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18", 
                messages=messages, 
                temperature=0.6,
                response_format=TranslationOutput
            )
            output = response.choices[0].message.parsed
            if [key for key in ['rankings_list', 'best_translation'] if (not hasattr(output, key))]:
                output = None
        except Exception as e:
                time.sleep(30)
                continue
    return None if output is None else output.dict()