import openai
from openai import OpenAI
import os
import json
import warnings
import ast
import time
from pydantic import BaseModel, Field
from typing import List, get_type_hints
import re
import huggingface_hub
from credentials import openai_api_key, google_api_key, together_api_key, openrouter_api_key
from google import genai
from google.genai.types import GenerateContentConfig
from google.genai import types
import together
warnings.filterwarnings("ignore")

openai.api_key = openai_api_key
os.environ["TOGETHER_API_KEY"] = together_api_key

def prompt_llm_model(system_prompt, user_prompt, cfg, temp, judge, schema, mode, task=None):
    
    if mode == "base":
        if cfg.translation_model.provider == "openai":
            output = prompt_openai_model(system_prompt, user_prompt, cfg, temp, judge)
        elif cfg.translation_model.provider == "google":
            output = prompt_google_model(system_prompt, user_prompt, cfg, temp, judge)
        elif cfg.translation_model.provider == "together":
            output = prompt_together_model(system_prompt, user_prompt, cfg, temp, judge)
        elif cfg.translation_model.provider == "local":
            output = prompt_vllm_model(system_prompt, user_prompt, cfg, temp, judge)
        elif cfg.translation_model.provider == "openrouter":
            output = prompt_openrouter_model(system_prompt, user_prompt, cfg, temp, judge)
        else:
            raise "Please specify a valid model provider"
    elif mode == "structured":
        if cfg.translation_model.provider == "openai":
            output_dict = prompt_openai_model_structured(system_prompt, user_prompt, cfg, schema, temp, judge)
            if task == "parsing":
                output = output_dict
            else:
                output = output_dict.translation_final
        elif cfg.translation_model.provider == "google":
            output_dict = prompt_google_model_structured(system_prompt, user_prompt, cfg, schema, temp, judge)
            if task == "parsing":
                output = output_dict
            else:
                output = output_dict.translation_final
        elif cfg.translation_model.provider == "together":
            output_dict = prompt_together_model_structured(system_prompt, user_prompt, cfg, schema, temp, judge)
            if task == "parsing":
                output = output_dict
            else:
                output = output_dict.translation_final
        elif cfg.translation_model.provider == "local":
            output_dict = prompt_vllm_model_structured(system_prompt, user_prompt, cfg, schema, temp, judge)
            if task == "parsing":
                output = output_dict
            else:
                output = output_dict.translation_final
        elif cfg.translation_model.provider == "openrouter":
            output_dict = prompt_openrouter_model_structured(system_prompt, user_prompt, cfg, schema, temp, judge)
            if task == "parsing":
                output = output_dict
            else:
                output = output_dict.translation_final
        else:
            raise "Please specify a valid model provider"

    return output

### OpenAI utils ###

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
        except Exception as e:
            print(e)
            time.sleep(0.1)
            continue
    
    return output


def prompt_openai_model_structured(system_prompt, user_prompt, cfg, schema=None, temp=0.1, judge=False):

    target_lang = cfg.task_config.target_language

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
            
        except Exception as e:
            print(e)
            time.sleep(0.1)
            continue
        
    return output


### Google (Gemini API) utils ###

def prompt_google_model(system_prompt, user_prompt, cfg, temp=0.1, judge=False):
    
    client = genai.Client(api_key=google_api_key)
    model_name = "gemini-2.0-flash-001" # default model to override
    output = None
    reasoning_config = None
    n_tries = 0

    if judge:
        if cfg.judge_model.name != None:
            model_name = cfg.judge_model.name
    else:    
        if cfg.translation_model.name != None:
            model_name = cfg.translation_model.name

    if "2.5" in model_name:
        reasoning_config = types.ThinkingConfig(thinking_budget=0)


    while output is None:        
        try:
            n_tries += 1
            if n_tries > 3:
                print(f'Skipped value for safety reasons for {user_prompt}')
                return "skip"
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temp,
                    thinking_config=reasoning_config
                )
            )
            output = response.text
        except Exception as e:
            time.sleep(0.1)
            print(e)


    return output


def prompt_google_model_structured(system_prompt, user_prompt, cfg, schema=None, temp=0.1, judge=False):

    target_lang = cfg.task_config.target_language

    class TranslationOutput(BaseModel):
        translation_final: str = Field(description=f"Translated text string into {target_lang}.")

    if schema == None:
        schema = TranslationOutput

    client = genai.Client(api_key=google_api_key)
    model_name = "gemini-2.0-flash-001" # default model to override
    output = None
    reasoning_config = None
    n_tries = 0

    if judge:
        if cfg.judge_model.name != None:
            model_name = cfg.judge_model.name
    else:    
        if cfg.translation_model.name != None:
            model_name = cfg.translation_model.name

    if "gemini-2.5" in model_name:
        reasoning_config = types.ThinkingConfig(thinking_budget=0)

    while output is None:        
        try:
            n_tries += 1
            if n_tries > 3:
                print(f'Skipped value for safety reasons for {user_prompt}')
                return schema(translation_final="skip")
            
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temp,
                    response_mime_type='application/json',
                    response_schema=schema,
                    thinking_config=reasoning_config
                )
            )
            
            output = response.parsed

        except Exception as e:
            time.sleep(0.1)
            print(e)

    return output


### TogetherAI utils ###

def prompt_together_model(system_prompt, user_prompt, cfg, temp=0.1, judge=False):
    
    if cfg == None:
        model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    else:
        if judge:
            model_name = cfg.judge_model.name
        else:
            model_name = cfg.translation_model.name
    
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_prompt} 
    ]
    
    output = None
    while output is None:
        try:
            response = together.Together().chat.completions.create(
                    model=model_name, 
                    messages=messages, 
                    temperature=temp 
                )
            output = response.choices[0].message.content
        except Exception as e:
                time.sleep(1)
                print(e)
                continue
    
    return output


def prompt_together_model_structured(system_prompt, user_prompt, cfg, schema=None, temp=0.1, judge=False):

    target_lang = cfg.task_config.target_language

    class TranslationOutput(BaseModel):
        translation_final: str = Field(description=f"Translated text string into {target_lang}.")

    if schema == None:
        schema = TranslationOutput

    if cfg == None:
        model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    else:
        if judge:
            model_name = cfg.judge_model.name
        else:
            model_name = cfg.translation_model.name
        
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
            response = together.Together().chat.completions.create(
                    model=model_name, 
                    messages=messages, 
                    temperature=temp,
                    response_format={
                        "type": "json_object",
                        "schema": schema.model_json_schema(),
                    },
                )
            output = schema(**json.loads(response.choices[0].message.content))
        except Exception as e:
                time.sleep(0.1)
                print(e)
                continue
        
    return output

### OpenRouter utils ###

def prompt_openrouter_model(system_prompt, user_prompt, cfg, temp=0.1, judge=False):

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        )  

    if cfg == None:
        model_name = "openai/gpt-4o-mini-2024-07-18"
        reason_lvl = None
    else:
        if judge:
            model_name = cfg.judge_model.name
            reason_lvl = cfg.judge_model.reasoning_level
        else:
            model_name = cfg.translation_model.name
    
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_prompt} 
    ]
    
    output = None
    while output is None:
        try:
            response = client.chat.completions.create(
                messages=messages, 
                temperature=temp ,
                model=model_name,
            )
            output = response.choices[0].message.content
        except Exception as e:
            time.sleep(0.1)
            print(e)
            continue

    return output


def prompt_openrouter_model_structured(system_prompt, user_prompt, cfg, schema=None, temp=0.1, judge=False):

    target_lang = cfg.task_config.target_language
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        )

    class TranslationOutput(BaseModel):
        translation_final: str = Field(description=f"Translated text string into {target_lang}.")

    if schema == None:
        schema = TranslationOutput

    if cfg == None:
        model_name = "openai/gpt-4o-mini-2024-07-18"
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
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=messages, 
                temperature=temp,
                response_format=schema,
            )
            output = response.choices[0].message.parsed
        except Exception as e:
            print(e)
            time.sleep(0.1)
            continue
        
    return output

### Local inference (vLLM) utils ###

def prompt_vllm_model(system_prompt, user_prompt, cfg, temp=0.1, judge=False):

    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="token-abc123",
        )
    
    if cfg == None:
        raise "Please specify a valid model name"
    else:
        if judge:
            model_name = cfg.judge_model.name
        else:
            model_name = cfg.translation_model.name
    
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_prompt} 
    ]
    
    output = None
    while output is None:
        try:
            response = client.chat.completions.create(
                model=model_name, 
                messages=messages, 
                temperature=temp 
            )
            output = response.choices[0].message.content
        except Exception as e:
            time.sleep(0.1)
            print(e)
            continue
    
    return output


def prompt_vllm_model_structured(system_prompt, user_prompt, cfg, schema=None, temp=0.1, judge=False):

    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="token-abc123",
        )

    target_lang = cfg.task_config.target_language

    class TranslationOutput(BaseModel):
        translation_final: str = Field(description=f"Translated text string into {target_lang}.")

    if schema == None:
        schema = TranslationOutput

    if cfg == None:
        raise "Please specify a valid model name"
    else:
        if judge:
            model_name = cfg.judge_model.name
        else:
            model_name = cfg.translation_model.name
        
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
            response = client.chat.completions.parse(
                model=model_name, 
                messages=messages, 
                temperature=temp,
                response_format=schema
            )
            output = response.choices[0].message.parsed
            
        except Exception:
            time.sleep(0.1)
            print(e)
            continue
        
    return output

### Other utils ###

def llm_aided_dict_parsing_old(text):

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
                time.sleep(1)
                continue
    return None if output is None else output.dict()



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