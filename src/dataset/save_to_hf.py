import json
import re
from datasets import load_dataset
from credentials import hf_token

def modify_nested_key(entry, key):

    keys = key.split(".")
    key_1 = keys[0]
    key_2 = keys[1]
    key_dict = entry.get(key_1, {})
    replacement = entry.get(f"{key}_translated", "")
    key_dict[key_2] = replacement
    entry.pop(f"{key}_translated", None)

    return entry

def push_data_to_hf(cfg, bench_name, subset, lang, output_dir):

    split_names = cfg.task_config.dataset.split
    method = cfg.task_config.method
    fields = cfg.task_config.fields
    hf_files = {}

    for split_name in split_names:
        
        input_filename = f"{output_dir}/{bench_name}_{subset}_{split_name}_{lang}_{method}.json"
        with open(input_filename, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        for entry in dataset:

            if "ranks" in entry:
                entry.pop("ranks", None)
            if "raw_ranks" in entry:
                entry.pop("raw_ranks", None)
            
            for field in fields:
                if f"{field}_translated" in entry:
                    if "." in field:
                        entry = modify_nested_key(entry, field)
                    else:
                        entry[field] = entry.pop(f"{field}_translated")
                        entry.pop(f"{field}_translated", None)

        output_filename = f"{output_dir}/{bench_name}_{subset}_{split_name}_{lang}_clean.jsonl"
        hf_files[split_name] = output_filename
        with open(output_filename, "w", encoding="utf-8") as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Modified clean dataset saved to {output_filename}")

        repo_id = f"{bench_name}_{lang}" 

        with open(output_filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                try:
                    _ = json.loads(line)
                except Exception as e:
                    print(f"Error parsing line {i}: {e}")

    dataset = load_dataset("json", data_files=hf_files)
    if len(subset) > 0:
        dataset.push_to_hub(repo_id, subset, token=hf_token, private=True)
    else:
        dataset.push_to_hub(repo_id, token=hf_token, private=True)
    print(f"Uploaded {output_filename} to repo {repo_id}")