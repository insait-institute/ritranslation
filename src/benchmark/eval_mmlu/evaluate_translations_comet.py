import json
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
from comet import download_model, load_from_checkpoint
from datasets import load_dataset

def load_json_data(input_json_path):

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
        if sample_id == None: 
            print('no match found')
        entry["sample_id"] = sample_id

    return data


def main():

    language = "et"
    print(f"Starting evaluation for language: {language}")

    print("Downloading and loading COMET model...")
    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)

    json_filename = f"./data/cais_mmlu_all_test_{language}_TRANK.json"  
    translated_entries = load_json_data(json_filename)
    print(f"Loaded {len(translated_entries)} translated entries from {json_filename}.")

    print("Loading reference dataset...")
    hf_dataset = load_dataset("CohereForAI/Global-MMLU", language)
    ref_entries = hf_dataset['test']
    print(f"Loaded {len(ref_entries)} reference entries from the HF dataset.")

    ref_dict = {}
    for record in ref_entries:
        sample_id = record["sample_id"]
        ref_dict[sample_id] = record

    question_eval_data = []

    for entry in translated_entries:
        sample_id = entry["sample_id"]
        if sample_id not in ref_dict:
            print(f"[WARNING] Reference not found for sample_id: {sample_id}")
            continue

        ref_record = ref_dict[sample_id]

        ref_choices = [
            ref_record.get("option_a", ""),
            ref_record.get("option_b", ""),
            ref_record.get("option_c", ""),
            ref_record.get("option_d", "")
        ]

        original_question = entry["question"].strip()
        choices = entry["choices"]
        src_question = "Question: " + original_question + "\nChoices: " + ", ".join(choices)
        mt_question = "Question: " + entry["question_translated"].strip() + "\nChoices: " + ", ".join(entry["choices_translated"])
        ref_question = "Question: " + ref_record["question"] + "\nChoices: " + ", ".join(ref_choices)

        question_eval_data.append({
            "src": src_question,
            "mt": mt_question,
            "ref": ref_question
        })

    print(f"Prepared {len(question_eval_data)} question evaluation entries.")

    print("\nEvaluating question translations...")
    question_eval_output = model.predict(question_eval_data, batch_size=100, gpus=1)
    print("Question-level system score:", question_eval_output.system_score)

    results = {
                "question_eval_output_scores": question_eval_output.scores,
                "question_eval_output_system_score": question_eval_output.system_score,
                "question_eval_output_error_spans": question_eval_output.metadata.error_spans
              }

    with open(f"./data/quality_evaluations/COMET_ref/mmlu_{language}_TRANK_COMET_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    main()