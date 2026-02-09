import os
import time
import json
import openai
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
from pydantic import BaseModel, Field
from multiprocessing import Pool, cpu_count
import warnings
import sys
from pathlib import Path
# Ensure project root is in sys.path for imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from credentials import openai_api_key, openrouter_api_key
warnings.filterwarnings("ignore")

openai.api_key = openai_api_key

def load_json_data(input_json_path):
    """
    Loads the translation JSON file and matches it with the sample_id from the HF MMLU dataset (we do not have sample_id in original cais dataset)
    """

    og_dataset = load_dataset("cais/mmlu", "all")
    mmlu_dataset = og_dataset["test"]
    subject_counters = defaultdict(int)
    match_dict = {}
    for row in mmlu_dataset:
        subject = row["subject"]
        sample_id = f"{subject}/test/{subject_counters[subject]}"
        subject_counters[subject] += 1
        normalized_question = row["question"].replace("\n", " ").strip()
        choices = row["choices"]
        first_choice = choices[0] if choices else ""
        match_key = (subject, normalized_question, first_choice)
        match_dict[match_key] = sample_id

    with open(input_json_path, "r", encoding="utf-8") as f_in:
        data = json.load(f_in)

    for entry in data:
        entry_subject = entry["subject"]
        normalized_question = entry["question"].replace("\n", " ").strip()
        entry_first_choice = entry["choices"][0] if entry["choices"] else ""
        match_key = (entry_subject, normalized_question, entry_first_choice)
        sample_id = match_dict.get(match_key, None)
        if sample_id is None:
            print("no match found")
        entry["sample_id"] = sample_id

    return data

def build_reference_lookup(ref_entries):
    
    ref_lookup = {}
    for record in ref_entries:
        ref_lookup[record["sample_id"]] = record
    return ref_lookup


def load_prompt_from_file(path: str):

    try:
        with open(path, "r") as f:
            prompt = f.read()

    except Exception:
        raise Exception('File either does not exist or is corrupted.')

    return prompt


def prepare_base_eval_prompts(src, ref_1, ref_2):

    prompt = load_prompt_from_file("./prompts/eval/judge_prompt.txt")
    prompt = prompt.replace('{og_text}', src)
    prompt_1 = prompt.replace('{output_1}', ref_1).replace('{output_2}', ref_2)
    prompt_2 = prompt.replace('{output_1}', ref_2).replace('{output_2}', ref_1)
    
    return prompt_1, prompt_2


def llm_aided_dict_parsing_old(text):
    
    class EvaluationOutput(BaseModel):
        analysis_of_A: str = Field(description="Analysis of Translation A")
        analysis_of_B: str = Field(description="Analysis of Translation B")
        reason_of_A_equals_B: str = Field(description="Analysis where Translation A and B perform equally well")
        reason_of_A_better_than_B: str = Field(description="Analysis where Translation A is better than Translation B")
        reason_of_B_better_than_A: str = Field(description="Analysis where Translation B is better than Translation A")
        choice: str = Field(description="A+, B+ or T=")
    
    class ChoiceOutput(BaseModel):
        choice: str = Field(description="A+, B+ or T=")

    user_prompt = f'''Your task is to extract dictionary in JSON format from your previous reply. The dictionary should contain the following keys: 'analysis_of_A', 'analysis_of_B', 'reason_of_A_equals_B', 'reason_of_A_better_than_B', 'reason_of_B_better_than_A' and 'choice'. Please correct it in a way so that the output would be a valid JSON dictionary. Here is the output to extract the dictionary from: \n
    {text}'''
    messages = [
        {"role": "system", "content": 'You are a helpful assistant.'}, 
        {"role": "user", "content": user_prompt} 
    ]
    messages_extra = [
        {"role": "system", "content": 'You are a helpful assistant.'}, 
        {"role": "user", "content": f'''Your task is to extract dictionary in JSON format from your previous reply. The dictionary should contain the key 'choice'. Here is the output to extract the dictionary from: \n {text}'''} 
    ]
    output = None
    no_tries = 0
    while output is None:
        try:
            no_tries += 1
            if no_tries > 5:
                response = openai.Client(api_key=openai.api_key).beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18", 
                messages=messages_extra, 
                temperature=0.7,
                response_format=ChoiceOutput)
                output = response.choices[0].message.parsed
                print(output)
                if output:
                    choice = output.choice
                else:
                    choice = ""
                output_dict = {"analysis_of_A": "", "analysis_of_B": "", "reason_of_A_equals_B": "", "reason_of_A_better_than_B": "", "reason_of_B_better_than_A": "", "choice": choice}
                return output_dict
            response = openai.Client(api_key=openai.api_key).beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18", 
                messages=messages, 
                temperature=0.7,
                response_format=EvaluationOutput
            )
            output = response.choices[0].message.parsed
        except Exception as e:
                time.sleep(30)
                print(e)
                continue
    return None if output is None else output.dict()


def query_llm_judge(prompt):
    """
    Query the OpenAI API with the given prompt and return the response as a dictionary using Stuctured Outputs.
    """

    system_prompt = '''You are an expert evaluator. Your task is to evaluate the quality of the translations from English into Romanian generated by two AI models. We will provide you with the user query and a pair of generated translations (Translation A and Translation B). You should first read the user query carefully for analyzing the task, and then evaluate the quality of the translations based on and rules provided below. Your response should be a valid json.'''
    response_dict = None
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        )  

    class EvaluationOutput(BaseModel):
        analysis_of_A: str = Field(description="Analysis of Translation A")
        analysis_of_B: str = Field(description="Analysis of Translation B")
        reason_of_A_equals_B: str = Field(description="Analysis where Translation A and B perform equally well")
        reason_of_A_better_than_B: str = Field(description="Analysis where Translation A is better than Translation B")
        reason_of_B_better_than_A: str = Field(description="Analysis where Translation B is better than Translation A")
        choice: str = Field(description="A+, B+ or T=")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    temp = 0.1
    tries = 0
    schema = EvaluationOutput
    while response_dict is None:
        tries += 1
        if tries > 5:
            print('fail')
            response_dict = {"analysis_of_A": "", "analysis_of_B": "", "reason_of_A_equals_B": "", "reason_of_A_better_than_B": "", "reason_of_B_better_than_A": "", "choice": "T="}
        try:
            response = client.beta.chat.completions.parse(
                model="google/gemini-2.5-flash",
                messages=messages, 
                temperature=temp,
                response_format=schema,
            )
            parsed = response.choices[0].message.parsed
            # Ensure always a dict, not a Pydantic object
            if hasattr(parsed, 'dict'):
                response_dict = parsed.dict()
            elif isinstance(parsed, dict):
                response_dict = parsed
            else:
                response_dict = dict(parsed)
        except Exception as e:
            # print(e)
            time.sleep(0.5)
    return response_dict


def evaluate_bidirectional_item(item):
    """
    For given evaluation item, query the LLM judge model bidirectionally and return the final decision.
    """
    sample_id = item["sample_id"]

    src_full = "Original question: " + item["question_src"].strip()
    if item.get("choices_src"):
        src_full += "\nOriginal answer options: " + ", ".join(item["choices_src"])

    mt_full = "Translated question: " + item["question_mt"].strip()
    if item.get("choices_mt"):
        mt_full += "\nTranslated answer options: " + ", ".join(item["choices_mt"])

    ref_full = "Translated question: " + item["question_ref"].strip()
    if item.get("choices_ref"):
        ref_full += "\nTranslated answer options: " + ", ".join(item["choices_ref"])
    eval_prompt_1, eval_prompt_2 = prepare_base_eval_prompts(src_full, mt_full, ref_full)

    response_normal = query_llm_judge(eval_prompt_1)
    response_swapped = query_llm_judge(eval_prompt_2)

    normal_result = response_normal["choice"]
    swapped_result = response_swapped["choice"]
    # check final result for A+ B+ T= and map to 0/1/2 scheme
    # 0: draw, 1: our wins, 2: reference wins
    if normal_result == "A+" and swapped_result == "B+":
        final_decision = "1"
    elif normal_result == "B+" and swapped_result == "A+":
        final_decision = "2"
    elif normal_result == "B+" and swapped_result == "T=":
        final_decision = "2"
    elif normal_result == "A+" and swapped_result == "T=":
        final_decision = "1"
    elif normal_result == "T=" and swapped_result == "A+":
        final_decision = "2"
    elif normal_result == "T=" and swapped_result == "B+":
        final_decision = "1"
    else:
        final_decision = "0"
        
    return {
        "sample_id": sample_id,
        "normal_response": normal_result,
        "swapped_response": swapped_result,
        "normal_output": response_normal,
        "swapped_output": response_swapped,
        "final_decision": final_decision
    }


def evaluate_dataset_bidirectional(eval_items):
    """
    Evaluates a list of evaluation items bidirectionally using multiprocessing.
    """
    
    num_workers = max(1, cpu_count()-4)  # burn the CPUs
    print(f"Starting parallel evaluation with {num_workers} worker(s)...")
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(evaluate_bidirectional_item, eval_items), total=len(eval_items)))
    return results


def main():

    language = "ro"
    method = "TRANK"
    
    json_filename = f"./data/cais_mmlu_all_test_{language}_{method}.json" 
    translated_entries = load_json_data(json_filename)
    print(f"Loaded {len(translated_entries)} translated entries from '{json_filename}'.")

    hf_dataset = load_dataset("CohereLabs/Global-MMLU", language)
    ref_entries = hf_dataset["test"]
    print(f"Loaded {len(ref_entries)} reference entries from Global-MMLU.")

    ref_lookup = build_reference_lookup(ref_entries)

    eval_items = []
    for entry in translated_entries:
        sample_id = entry["sample_id"]
        if sample_id not in ref_lookup:
            print(f"[WARNING] No reference found for sample_id: {sample_id}")
            continue

        ref_record = ref_lookup[sample_id]

        question_src = entry["question"].strip()
        choices_src = entry["choices"] if entry.get("choices") is not None else []

        question_mt = entry["question_translated"].strip()
        choices_mt = entry["choices_translated"] if entry.get("choices_translated") is not None else []

        question_ref = ref_record["question"]
        choices_ref = []
        for key in ["option_a", "option_b", "option_c", "option_d"]:
            if ref_record.get(key, ""):
                choices_ref.append(ref_record.get(key, ""))

        eval_items.append({
            "sample_id": sample_id,
            "question_src": question_src,
            "choices_src": choices_src,
            "question_mt": question_mt,
            "choices_mt": choices_mt,
            "question_ref": question_ref,
            "choices_ref": choices_ref
        })

    print(f"Prepared {len(eval_items)} evaluation items (each with combined question text and answer options).\n")

    print("Starting bidirectional evaluation of candidate translations using judge...")
    results = evaluate_dataset_bidirectional(eval_items)

    mt_wins = 0
    ref_wins = 0
    draws = 0

    for r in results:
        decision = r["final_decision"]
        if decision == "1":
            mt_wins += 1
        elif decision == "2":
            ref_wins += 1
        elif decision == "0":
            draws += 1

    print("\nOverall Evaluation Summary:")
    print(f"  {method} (our) wins (final decision '1'): {mt_wins}")
    print(f"  Reference Translation (Global-MMLU) wins (final decision '2'): {ref_wins}")
    print(f"  Draws: {draws}")

    with open(f"./data/quality_evaluations/llm_as_a_judge/mmlu_{language}_{method}_judge_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    main()