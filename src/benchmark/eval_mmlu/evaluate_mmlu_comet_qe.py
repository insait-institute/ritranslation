import json
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
from comet import download_model, load_from_checkpoint

def load_and_combine_data(
    json_filename,
):

    with open(json_filename, "r", encoding="utf-8") as f:
        json_data = json.load(f)


    eval_data = []
    for field in ['question', 'choices']:
        for entry in json_data:
            example = {}              
            example["src"] = str(entry[field]) 
        
            if f"{field}_corrected" in entry and entry[f"{field}_corrected"]:
                example["mt"] = str(entry[f"{field}_corrected"])
            else:
                example["mt"] = str(entry[f"{field}_translated"]
)
            eval_data.append(example)

    return eval_data


def main():

    print("Downloading and loading COMET model...")
    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl") # QE
    model = load_from_checkpoint(model_path)

    language = "ro"
    print(f"Starting evaluation for language: {language}")

    # methods = ["USI", "TRANK"]
    methods = ["USI"]
    for method in methods:
        print(f"Evaluating method: {method}")
        json_filename = f"data/cais_mmlu_all_test_{language}_{method}.json"  
        question_eval_data = load_and_combine_data(json_filename)
        print(f"Prepared {len(question_eval_data)} question evaluation entries.")

        # split_1 = question_eval_data[:int(len(question_eval_data) * 0.5)]
        # split_2 = question_eval_data[int(len(question_eval_data) * 0.5):]

        # question_eval_output_1 = model.predict(split_1, batch_size=20, gpus=0)
        # question_eval_output_2 = model.predict(split_2, batch_size=20, gpus=0)
        # # merge output.scores lists into one list
        # question_eval_output = question_eval_output_1.scores.append(question_eval_output_2.scores) 
        # system_score = (question_eval_output_1.system_score + question_eval_output_2.system_score) / 2
        question_eval_output = model.predict(question_eval_data, batch_size=100, gpus=1)
        print(f"Question-level system score method {method}:", question_eval_output.system_score)
        results = {
                "question_eval_output_scores": question_eval_output.scores,
                "question_eval_output_system_score": question_eval_output.system_score,
                }
        with open(f"data/quality_evaluations/COMET_qe/mmlu_{language}_{method}_eval_qe.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    main()