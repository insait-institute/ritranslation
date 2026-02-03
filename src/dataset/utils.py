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
from .model_factory import prompt_llm_model
warnings.filterwarnings("ignore")

openai.api_key = openai_api_key

def load_prompt_from_file(path: str):
    try:
        with open(path, "r") as f:
            prompt = f.read()

    except Exception:
        raise Exception(f'File <{path}> either does not exist or is corrupted.')

    return prompt

def get_data_key(data, dotted_path, default=None):
    
    keys = dotted_path.split(".")
    current = data

    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        elif isinstance(current, list):
            try:
                index = int(key)
                current = current[index]
            except (ValueError, IndexError):
                return default
        else:
            return default

        if current is None:
            return default

    return current

def generate_pos_combinations(n):

    rotations = []
    indices = list(range(n))
    for i in range(len(indices)):
        rotation = indices[-i:] + indices[:-i]
        rotations.append(rotation)

    return rotations

def split_text_into_chunks(text, max_words=10000):
    """
    Splits the input text into chunks with each chunk having up to max_words,
    ensuring splits occur at sentence boundaries.
    """
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_word_count + word_count > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def prepare_base_translate_prompt(original_text, prompt, target_lang):

    # prompt = prompt.replace('<original_text>', original_text.replace('\n', ' ')).replace('<target_language>', target_lang)
    prompt = prompt.replace('<original_text>', original_text).replace('<target_language>', target_lang)
    
    return prompt

def get_prompt_template(original_text, target_lang, method_type, cfg):

    few_shot = cfg.task_config.few_shot
    multi_prompt = cfg.task_config.multi_prompt
    original_text = str(original_text)

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

        translation_prompt_file = cfg.task_config.translation_prompt_file
        if cfg.task_config.judge_prompt_file != None:
            judge_prompt_file = cfg.task_config.judge_prompt_file
        else:
            judge_prompt_file = './src/dataset/prompts/self_correction_check.txt'

        translate_prompt = load_prompt_from_file(translation_prompt_file)
        translate_prompt = prepare_base_translate_prompt(original_text, translate_prompt, target_lang)
        check_prompt = load_prompt_from_file(judge_prompt_file)
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
            translation_prompt_file = cfg.task_config.translation_prompt_file
            translate_prompt = load_prompt_from_file(translation_prompt_file)
            translate_prompt = [prepare_base_translate_prompt(original_text, translate_prompt, target_lang)]
        
        if cfg.task_config.judge_prompt_file != None:
            judge_prompt_file = cfg.task_config.judge_prompt_file
        else:
            judge_prompt_file = './src/dataset/prompts/universal_self_improvement.txt'

        check_prompt = load_prompt_from_file(judge_prompt_file)
        check_prompt = check_prompt.replace('<original_text>', original_text).replace('<target_language>', target_lang)

    elif method_type == 'TRANK':
        if multi_prompt:
            translate_prompt = []
            prompt_files = cfg.task_config.prompt_files
            for file in prompt_files:
                prompt = load_prompt_from_file(f"./src/dataset/prompts/mq_base_translation_prompts/{file}")
                prompt = prepare_base_translate_prompt(original_text, prompt, target_lang)
                translate_prompt.append(prompt)
        else:
            translation_prompt_file = cfg.task_config.translation_prompt_file
            translate_prompt = load_prompt_from_file(translation_prompt_file)
            translate_prompt = [prepare_base_translate_prompt(original_text, translate_prompt, target_lang)]
        if cfg.task_config.judge_prompt_file != None:
            judge_prompt_file = cfg.task_config.judge_prompt_file
        else:
            judge_prompt_file = './src/dataset/prompts/trank_rank_n.txt'
        check_prompt = load_prompt_from_file(judge_prompt_file)
        check_prompt = check_prompt.replace('<original_text>', original_text).replace('<target_language>', target_lang).replace('<few-shot_prompt>', few_shot_prompt)

    elif method_type == 'BoN':
        translation_prompt_file = cfg.task_config.translation_prompt_file
        if cfg.task_config.judge_prompt_file != None:
            judge_prompt_file = cfg.task_config.judge_prompt_file
        else:
            judge_prompt_file = './src/dataset/prompts/best_of_n_scoring.txt'
        translate_prompt = load_prompt_from_file(translation_prompt_file)
        translate_prompt = prepare_base_translate_prompt(original_text, translate_prompt, target_lang)
        check_prompt = load_prompt_from_file(judge_prompt_file)
        check_prompt = check_prompt.replace('<original_text>', original_text).replace('<target_language>', target_lang)

    else:
        raise Exception('Select a valid prompting method type (SC/USI/TRANK/BoN).')

    return translate_prompt, check_prompt

def trank_get_final_prompt_template(original_text, best_text, target_lang, cfg):

    few_shot = cfg.task_config.few_shot

    if few_shot:
        few_shot_prompt = "## Examples \n" + load_prompt_from_file('./src/dataset/prompts/few_shot_rank.txt')
    else:
        few_shot_prompt = ''

    final_fix_prompt = load_prompt_from_file('./src/dataset/prompts/trank_rank_best_candidate_correction.txt')
    final_fix_prompt = final_fix_prompt.replace('<original_text>', original_text).replace('<selected_text>', best_text).replace('<target_language>', target_lang).replace('<few-shot_prompt>', few_shot_prompt)

    return final_fix_prompt

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
                    reasoning_effort=reason_lvl
                )
            else:
                response = openai.chat.completions.create(
                    model=model_name, 
                    messages=messages, 
                    temperature=temp 
                )
            output = response.choices[0].message.content
        except openai.RateLimitError:
                time.sleep(1)
                continue
    
    return output


def prompt_openai_model_structured(system_prompt, user_prompt, target_lang, cfg, schema=None, temp=0.1, judge=False):

    class TranslationOutput(BaseModel):
        translation_final: str = Field(description=f"Translated text string into {target_lang}.")

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
    counter = 0
    while output is None:
        counter += 1
        if counter > 2:
            print('Could not parse response for prompt: ', user_prompt)
            return "skip"
        try:
            if "o3" in model_name or "o1" in model_name:
                response = openai.Client(api_key=openai.api_key).beta.chat.completions.parse(
                    model=model_name, 
                    messages=messages, 
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
        except Exception:
                time.sleep(0.1)
                continue
        
    return output.translation_final

def resample_text_list(output, original_text, target_lang, cfg):
    
    system_prompt = 'You are a helpful assistant who is proficient in many languages.'
    user_prompt = '''Your task is to fix the translation from English to <target_language>. Here is the original list text:
    Original text: <original_text>
    Now I give you the translated text list to fix:
    Translated text: <translated_text>
    Fix the translated text - check if any of the list entries were missed and then translate them. Check if the length of original and translated arrays match - fix if translated array has more values than original one. If the value has an original translation already done, then just repeat it. Think about grammatical cases, declination and corectness of tenses. Format your output in the next form:
    Answers: [list of fixed translated answers]'''
    user_prompt = user_prompt.replace('<target_language>', target_lang).replace('<translated_text>', str(output)).replace('<original_text>', str(original_text))
    temp = 0.5
    choices = None 

    class CorrectedTranslationOutput(BaseModel):
        translation_final: List[str] = Field(description=f"Fixed translated list of text strings into {target_lang}.")

    while choices is None:
        try:
            # fixed_output = prompt_openai_model(system_prompt, user_prompt, temp = temp)
            choices = prompt_llm_model(system_prompt, user_prompt, cfg, temp, False, CorrectedTranslationOutput, "structured")
            # choices = extract_choices(fixed_output)
        except:
            if temp <= 0.9:
                temp += 0.1
            continue

    return choices

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
        
def extract_corrected_translation_trank(text, cfg):

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
                    data = llm_aided_dict_parsing(text, cfg)
            except Exception as e:
                data = llm_aided_dict_parsing(text, cfg)
                continue
        else:
            try:
                data = llm_aided_dict_parsing(text, cfg)
            except:
                print("Failed to extract JSON dict.")
                continue
    return data

def llm_aided_dict_parsing(text, cfg):

    class TranslationOutput(BaseModel):
        rankings_list: List[int] = Field(description="Array containing final rankings of each response ID ranked in order according to the evalautions.")
        best_translation: str = Field(description="The best translation text.")
        summary: str = Field(description="Summary of model's reasoning and evaluation of the responses.")

    user_prompt = f'''Your task is to extract dictionary in JSON format from your previous reply. The dictionary should contain the following keys: 'rankings_list', 'best_translation' and 'reasoning'. The 'rankings_list' key should contain a list of integers representing the final rankings of each response. The 'best_translation' key should contain the best translation text. The 'reasoning' key should contain the complete detailed reasoning text analyzing quality, correctness, grammar, etc. of every response candidate. Here is the output to extract the dictionary from: \n
    {text}'''
    system_prompt = 'You are a helpful assistant.'
    
    output = None
    counter = 0
    while output is None:
        counter += 1
        if counter > 3:
            return None
        try:
            output = prompt_llm_model(system_prompt, user_prompt, cfg, 0.6, True, TranslationOutput, "structured", "parsing")
            if [key for key in ['rankings_list', 'best_translation'] if (not hasattr(output, key))]:
                output = None
        except Exception as e:
                time.sleep(1)
                continue
    return None if output is None else output.dict()