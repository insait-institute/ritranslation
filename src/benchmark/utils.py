import openai
import os
import warnings
import json
import ast
import time
import langcodes
from pydantic import BaseModel, Field
from typing import List
import re
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

def split_question(text):

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


def prepare_base_translate_prompt(question, choices, prompt, target_lang):

    prompt = prompt.replace('<original_question>', question).replace('<original_answers>', str(choices)).replace('<target_language>', target_lang)
    # prompt = prompt.replace('<original_question>', question.replace('\n', ' ')).replace('<original_answers>', str(choices).replace('\n', ' ')).replace('<target_language>', target_lang)
    
    return prompt

def get_prompt_template(question, choices, target_lang, method_type, cfg):

    few_shot = cfg.task_config.few_shot
    multi_prompt = cfg.task_config.multi_prompt

    if few_shot:
        if method_type == 'SC':
            few_shot_prompt = load_prompt_from_file('./src/benchmark/prompts/few_shot_sc.txt')
        elif method_type == 'TRANK':
            few_shot_prompt = "## Examples \n" + load_prompt_from_file('./src/benchmark/prompts/few_shot_rank.txt')
        elif method_type == 'USI':
            few_shot_prompt = "## Examples \n" + load_prompt_from_file('./src/benchmark/prompts/few_shot_usi.txt')
        else:
            few_shot_prompt = ''
    else:
        few_shot_prompt = ''

    if method_type == 'SC':

        translation_prompt_file = cfg.task_config.translation_prompt_file
        if cfg.task_config.judge_prompt_file != None:
            judge_prompt_file = cfg.task_config.judge_prompt_file
        else:
            judge_prompt_file = './src/benchmark/prompts/self_correction_check.txt'
        
        translate_prompt = load_prompt_from_file(translation_prompt_file)
        translate_prompt = prepare_base_translate_prompt(question, choices, translate_prompt, target_lang)
        check_prompt = load_prompt_from_file(judge_prompt_file)
        check_prompt = check_prompt.replace('<original_question>', question).replace('<original_answers>', str(choices)).replace('<target_language>', target_lang)
        check_prompt = check_prompt.replace('<few-shot_prompt>', few_shot_prompt)

    elif method_type == 'USI':
        if multi_prompt:
            translate_prompt = []
            prompt_files = cfg.task_config.prompt_files
            for file in prompt_files:
                prompt = load_prompt_from_file(f"./src/benchmark/prompts/mq_base_translation_prompts/{file}")
                prompt = prepare_base_translate_prompt(question, choices, prompt, target_lang)
                translate_prompt.append(prompt)
        else:
            translation_prompt_file = cfg.task_config.translation_prompt_file
            translate_prompt = load_prompt_from_file(translation_prompt_file)
            translate_prompt = [prepare_base_translate_prompt(question, choices, translate_prompt, target_lang)]

        if cfg.task_config.judge_prompt_file != None:
            judge_prompt_file = cfg.task_config.judge_prompt_file
        else:
            judge_prompt_file = './src/benchmark/prompts/universal_self_improvement.txt'
        
        check_prompt = load_prompt_from_file(judge_prompt_file)
        check_prompt = check_prompt.replace('<original_question>', question).replace('<original_answers>', str(choices)).replace('<target_language>', target_lang)

    elif method_type == 'TRANK':
        if multi_prompt:
            translate_prompt = []
            prompt_files = cfg.task_config.prompt_files
            for file in prompt_files:
                prompt = load_prompt_from_file(f"./src/benchmark/prompts/mq_base_translation_prompts/{file}")
                prompt = prepare_base_translate_prompt(question, choices, prompt, target_lang)
                translate_prompt.append(prompt)
        else:
            translation_prompt_file = cfg.task_config.translation_prompt_file
            translate_prompt = load_prompt_from_file(translation_prompt_file)
            translate_prompt = [prepare_base_translate_prompt(question, choices, translate_prompt, target_lang)]

        if cfg.task_config.judge_prompt_file != None:
            judge_prompt_file = cfg.task_config.judge_prompt_file
        else:
            judge_prompt_file = './src/benchmark/prompts/trank_rank_n.txt'

        check_prompt = load_prompt_from_file(judge_prompt_file)
        check_prompt = check_prompt.replace('<original_question>', question).replace('<original_answers>', str(choices)).replace('<target_language>', target_lang).replace('<few-shot_prompt>', few_shot_prompt)
        
    elif method_type == 'BoN':

        translation_prompt_file = cfg.task_config.translation_prompt_file
        if cfg.task_config.judge_prompt_file != None:
            judge_prompt_file = cfg.task_config.judge_prompt_file
        else:
            judge_prompt_file = './src/benchmark/prompts/best_of_n_scoring.txt'

        translate_prompt = load_prompt_from_file(translation_prompt_file)
        translate_prompt = prepare_base_translate_prompt(question, choices, translate_prompt, target_lang)
        check_prompt = load_prompt_from_file(judge_prompt_file)
        check_prompt = check_prompt.replace('<original_question>', question).replace('<original_answers>', str(choices)).replace('<target_language>', target_lang)

    else:
        raise Exception('Select a valid prompting method type (SC/USI/TRANK/BoN).')

    return translate_prompt, check_prompt

def trank_get_final_prompt_template(question, choices, best_question, best_choices, target_lang, cfg):

    few_shot = cfg.task_config.few_shot

    if few_shot:
        few_shot_prompt = "## Examples \n" + load_prompt_from_file('./src/benchmark/prompts/few_shot_rank.txt')
    else:
        few_shot_prompt = ''

    final_fix_prompt = load_prompt_from_file('./src/benchmark/prompts/trank_rank_best_candidate_correction.txt')
    final_fix_prompt = final_fix_prompt.replace('<original_question>', question).replace('<original_answers>', str(choices)).replace('<selected_question>', best_question).replace('<selected_answers>', str(best_choices)).replace('<target_language>', target_lang).replace('<few-shot_prompt>', few_shot_prompt)


    return final_fix_prompt


def fill_multi_gen_check_prompt(prompt, question_samples, answer_samples):

    responses_str_list = []
    for i, (question, answer) in enumerate(zip(question_samples, answer_samples), start=1):
        dict_entry = (
            '{\n'
            f'  "response_id": {i},\n'
            f'  "translated_question": """{question}""",\n'
            f'  "translated_answers": {answer}\n'
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
                time.sleep(30)
                continue
    return output


def prompt_openai_model_structured(system_prompt, user_prompt, target_lang, cfg, schema=None, temp=0.1, judge=False):

    class BaseTranslationOutput(BaseModel):
        question_final: str = Field(description=f"Translated question text into {target_lang}.")
        answers_final: List[str] = Field(description=f"Translated answer options list in {target_lang}.")

    if schema == None:
        schema = BaseTranslationOutput

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
            return "skip", ["skip"]
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
                    max_tokens=10000,
                    response_format=schema
                )
            output = response.choices[0].message.parsed
        except Exception as e:
            time.sleep(0.1)
            continue
        
    return output.question_final, output.answers_final


def resample_answers(output, question, choices, cfg):

    if cfg.task_config.target_language:
        target_lang = cfg.task_config.target_language
    else:
        raise "Please specify target language"
    
    class CorrectedAnswerOutput(BaseModel):
            answers_list: List[str] = Field(description=f"Corrected translated answer options list in {target_lang}.")


    system_prompt = 'You are a helpful assistant who is proficient in many languages.'
    user_prompt = '''Your task is to fix the translation from English to <target_language>. Here is the original translation:
    Original answers: <original_answers>
    Now I give you the translated answers and question for context to fix:
    Translated question: <translated_question>
    Translated answers: <translated answers>
    Fix the translated answers - check if any of the answer options were missed and then translate them with the respect to the question context (as if the answer is completing the question). If an answer has an original translation, then just repeat it. Think about grammatical cases, declination and corectness of tenses. Format your output in the next form:
    Answers: [list of fixed translated answers]'''
    user_prompt = user_prompt.replace('<target_language>', target_lang).replace('<translated_question>', question).replace('<translated_answers>', str(output)).replace('<original_answers>', str(choices))
    temp = 0.5

    while choices is None:
        try:
            # fixed_output = prompt_openai_model(system_prompt, user_prompt, temp = temp)
            fixed_output = prompt_llm_model(system_prompt, user_prompt, cfg, temp, True, CorrectedAnswerOutput, "structured")
            choices = fixed_output.answers_list
            # choices = extract_choices(fixed_output)
        except:
            if temp <= 0.9:
                temp += 0.1
            continue

    return choices


def parse_list_llm(answers_og):

    messages = [
        {"role": "system", "content": "You are a helpful assistant."}, # check
        {"role": "user", "content": f"""The following text is in Ukrainian, it contains a list of strings. Your task is to extract a list of strings from list fixed for JSON list format. Keep in mind that inside a string value there can be quotes " or apostrophes inside words ' which can break parsing. String items in the list should be separated by comma and highlighted by ', apostrophes ' or ’ inside words should be replaced with \' so it would not break parsing. 
        Replacing apostrophes inside words with "\'" is important to maintain the working foramt of the string value, same goes for any other special characters which can break the string separation. List entries should have closing string separators like ' or ".
        For example, '["Поточне значення y "так званій групі" однакове.", "Нуль.", "Один.", "Середнє значення y за періодом вибірки.", "Олігар\'хічна."]' should be ['Поточне значення y "так званій групі" однакове.', 'Нуль.', 'Один.', 'Середнє значення y за періодом вибірки.', 'Олігар\'хічна.']
        Now I give you the list to fix: {answers_og}
        Output only the resulting fixed string and nothing else.
        """
        } 
    ]
    ans_list = None
    try_counter = 0
    while ans_list is None:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=messages,
                max_tokens=2000,
                temperature=0.4
            )
            ans_output = response.choices[0].message.content
            try:
                ans_output = re.sub(r"(?<=[^\W\d_])'(?=[^\W\d_])", r"\\'", ans_output, flags=re.UNICODE)
                ans_list = ast.literal_eval(ans_output)
            except (SyntaxError, ValueError):
                ans_output = ans_output.replace("['", '["').replace("']", '"]').replace("', ", '", ').replace(", '", ', "').replace("'", "\'")
                ans_output = ans_output.replace('["', "['").replace('"]', "']").replace('", ', "', ").replace(', "', ", '").replace(".]", "']")
                # special cases
                ans_output = re.sub(r'''(?<=[\w])'(?=[\w])''', '\'', ans_output, flags=re.UNICODE)
                ans_output = re.sub(r'''(?<=[\w])’(?=[\w])''', '\'', ans_output, flags=re.UNICODE)
                ans_output = re.sub(r'''(?<=[\w])"'(?=[\w])''', '\'', ans_output, flags=re.UNICODE)
                ans_output = re.sub(r'''(?<=[\w])""(?=[\w])''', '\"', ans_output, flags=re.UNICODE) 
                ans_output = re.sub(r'''(?<=[\w])'"(?=[\w])''', '\'', ans_output, flags=re.UNICODE) 
                ans_list = ast.literal_eval(ans_output)
        except Exception as e:
            try_counter += 1
            if try_counter < 5: # if model fails to format translated output well then generate again
                continue
            else: 
                print('Manual extraction failed')
                ans_list = None
                break
    return ans_list

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
    Extracts the question text after 'Question:' and the list of answers after 'Answers:'.
    
    text: a string of the form:
      "Question: some_question
       Answers: ['answer1', "ans'we: "wdfdefwr"2", ...]"
    """
    
    pattern = r"Question:\s*(.*?)\s*Answer[s]?:\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        question_str = match.group(1).strip()
        answers_og = match.group(2).strip()
        try:
            answers_og = re.sub(r"(?<=[^\W\d_])'(?=[^\W\d_])", r"\\'", answers_og, flags=re.UNICODE)
            answers_list = ast.literal_eval(answers_og)
        except Exception:
            answers_list = parse_list_llm(answers_og)
            pass
        return question_str, answers_list
    else:
        return None, None
    
def extract_choices(text):
    """
    Extracts the question text after 'Question:' and the list of answers after 'Answers:'.
    
    text: a string of the form:
      "Answers: ['answer1', "ans'we: "wdfdefwr"2", ...]"
    """

    pattern = r"Answer[s]?:\s*(.*)"
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
            except Exception as e:
                print("Failed to extract JSON dict:", e)
                continue
    return data

def llm_aided_dict_parsing(text, cfg):

    class BestTranslation(BaseModel):
        best_translated_question: str = Field(description="The best translation question.")
        best_translated_answers: List[str] = Field(description="Answer options list from the best translation.")

    class TranslationOutput(BaseModel):
        rankings_list: List[int] = Field(description="Array containing final rankings of each response ID ranked in order according to the evalautions.")
        best_translation: BestTranslation
        summary: str = Field(description="Summary of model's reasoning and evaluation of the responses.")

    user_prompt = f'''Your task is to extract dictionary in JSON format from your previous reply. The dictionary should contain the following keys: 'rankings_list', 'best_translation' and 'reasoning'. The 'rankings_list' key should contain a list of integers representing the final rankings of each response. The 'best_translation' key should contain a dictionary with keys 'question' and 'answers'. The 'question' key should contain the best translation question and the 'answers' key should contain a list of the best translation answers. The 'reasoning' key should contain the complete detailed reasoning text analyzing quality, correctness, grammar, etc. of every response candidate. Here is the output to extract the dictionary from: \n
    {text}'''
    system_prompt = 'You are a helpful assistant.'
    # messages = [
    #     {"role": "system", "content": system_prompt}, 
    #     {"role": "user", "content": user_prompt} 
    # ]
    output = None
    counter = 0
    while output is None:
        counter += 1
        if counter > 3:
            print('failed to parse ranks, trying to regenerate')
            return None
        try:
            # response = openai.Client(api_key=openai.api_key).beta.chat.completions.parse(
            #     model="gpt-4o-mini-2024-07-18", 
            #     messages=messages, 
            #     temperature=0.6,
            #     response_format=TranslationOutput
            # )
            output = prompt_llm_model(system_prompt, user_prompt, cfg, 0.6, True, TranslationOutput, "structured")
            # output = response.choices[0].message.parsed
            if [key for key in ['rankings_list', 'best_translation'] if (not hasattr(output, key))]:
                output = None
        except Exception as e:
            print(e)
            time.sleep(1)
            continue
    return None if output is None else output.dict()