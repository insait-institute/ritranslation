# Configuration Guide

This directory contains YAML configuration files for the translation pipeline. The framework supports two modes: **BENCHMARK** and **DATASET**, each with specific parameters tailored for their respective formats.

## Directory Structure

```
configs/
├── benchmark/              # Benchmark translation configs
│   ├── ARC/               # AI2 Reasoning Challenge
│   ├── Hellaswag/         # Commonsense reasoning
│   ├── MMLU/              # Massive Multitask Language Understanding
│   ├── Winogrande/        # Coreference resolution
│   └── ...
└── dataset/               # Dataset translation configs
    ├── FLORES/            # FLORES-101 evaluation set
    ├── Hellaswag/         # Hellaswag as parallel corpus
    └── WMT/               # WMT translation datasets
```

## Quick Start

```bash
# Translate a benchmark
python run.py --config_path configs/benchmark/MMLU/bench_mmlu_bg.yaml

# Translate a dataset
python run.py --config_path configs/dataset/WMT/dataset_wmt_uk.yaml
```

---

## Configuration Schema

### Top-Level Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | string | Task type: `"BENCHMARK"` or `"DATASET"` |
| `output_dir` | string | Directory to save translation results |
| `translation_model` | object | Model configuration for translation |
| `judge_model` | object | Model configuration for evaluation/ranking |
| `timeout` | float | Timeout between API requests (seconds) |

### Model Configuration

```yaml
translation_model:
  name: "gpt-4o-mini-2024-07-18"    # Model name/ID
  provider: "openai"                 # Provider: openai, google, together, openrouter, vllm
  device: "auto"                     # Device for local models
  reasoning_level: "low"             # For OpenAI o1/o3 models: low, medium, high
```

**Supported Providers:**

| Provider | Example Models |
|----------|---------------|
| `openai` | gpt-4o, gpt-4o-mini, o1, o3-mini |
| `google` | gemini-2.0-flash, gemini-2.5-flash |
| `together` | meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo |
| `openrouter` | anthropic/claude-3-sonnet |
| `vllm` | Any locally served model |

---

## Task Configuration Parameters

### Common Parameters (Both Modes)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_language` | string | required | Target language name (e.g., "Bulgarian", "Ukrainian") |
| `method` | string | required | Translation method: `SC`, `USI`, `BoN`, `TRANK` |
| `temperature_translator` | float | 0.5 | Temperature for candidate sampling (0-1) |
| `temperature_judge` | float | 0.1 | Temperature for evaluation/correction (0-1) |
| `max_workers` | bool | false | Use maximum available CPU cores for parallelization |
| `num_workers` | int | 1 | Number of parallel workers (if max_workers=false) |
| `n_samples` | int | 3 | Number of translation candidates to generate |
| `agent_check` | bool | false | Enable additional LLM verification (SC method only) |
| `few_shot` | bool | false | Include few-shot examples in prompts |
| `multi_prompt` | bool | false | Use multiple prompt templates (USI, TRANK) |
| `save_to_hf` | bool | false | Push results to HuggingFace Hub |

### Method Selection Guide

| Method | Use Case | Quality | Cost |
|--------|----------|---------|------|
| `SC` | Quick translation, high-resource languages | ⭐⭐ | $ |
| `USI` | Short texts, balanced quality/cost | ⭐⭐⭐⭐ | $$ |
| `BoN` | Language-agnostic baseline | ⭐⭐⭐ | $$ |
| `TRANK` | Complex benchmarks, highest quality | ⭐⭐⭐⭐⭐ | $$$ |

> [!TIP]
> We recommend using USI for datasets in easier domains or for the good trade-off between cost and quality. For the best quality, especially for particularly technical texts or benchmarks(!) we recommend using T-RANK

---

## Benchmark Mode

### Benchmark-Specific Parameters

```yaml
task_config:
  benchmark:
    name: "cais/mmlu"           # HuggingFace dataset name
    subset: ["all"]             # List of subsets to translate
    split: ["test"]             # List of splits to translate
    n_entries: null             # Sample N entries (null = all)

  question_fields: ["question"]  # Fields to treat as questions
  answer_fields: ["choices"]     # Fields to treat as answer options
```

### Field Mapping

The pipeline supports nested dictionary access using dot notation:

```yaml
# For nested structures like {"messages": {"text": "..."}}
question_fields: ["messages.text"]
```

### Complete Benchmark Example

```yaml
task: "BENCHMARK"
output_dir: "src/benchmark/data"

translation_model:
  name: "gpt-4o-mini-2024-07-18"
  provider: "openai"

judge_model:
  name: "gpt-4o-mini-2024-07-18"
  provider: "openai"

task_config:
  benchmark:
    name: "allenai/ai2_arc"
    subset: ["ARC-Challenge"]
    split: ["test"]
    n_entries: null

  target_language: "Ukrainian"
  method: "TRANK"
  temperature_translator: 0.5
  temperature_judge: 0.1
  max_workers: true
  num_workers: 4
  n_samples: 5
  question_fields: ["question"]
  answer_fields: ["choices.text"]
  agent_check: false
  few_shot: false
  multi_prompt: false
  save_to_hf: false
```

---

## Dataset Mode

### Dataset-Specific Parameters

```yaml
task_config:
  dataset:
    name: "gsarti/flores_101"    # HuggingFace dataset name
    subset: ["eng"]              # List of subsets
    split: ["devtest"]           # List of splits
    n_entries: null              # Sample N entries (null = all)

  fields: ["sentence"]           # Fields to translate
```

### Complete Dataset Example

```yaml
task: "DATASET"
output_dir: "src/dataset/data/flores/uk"

translation_model:
  name: "gpt-4o-mini-2024-07-18"
  provider: "openai"

judge_model:
  name: "gpt-4o-mini-2024-07-18"
  provider: "openai"

task_config:
  dataset:
    name: "gsarti/flores_101"
    subset: ["eng"]
    split: ["devtest"]
    n_entries: null

  target_language: "Ukrainian"
  method: "USI"
  temperature_translator: 0.7
  temperature_judge: 0.1
  max_workers: true
  num_workers: 4
  n_samples: 5
  fields: ["sentence"]
  agent_check: true
  few_shot: false
  multi_prompt: true
  save_to_hf: false
```

---

## Advanced Options

### Custom Prompts

Override default prompt files:

```yaml
task_config:
  # For multi-prompt methods (USI, TRANK)
  prompt_files:
    - "mq_base_translation_prompts/ukrainian/example_base_1.txt"
    - "mq_base_translation_prompts/ukrainian/example_base_2.txt"

  # For single-prompt methods
  translation_prompt_file: "./src/benchmark/prompts/custom_translate.txt"
  judge_prompt_file: "./src/benchmark/prompts/custom_judge.txt"
```

### Reasoning Models (OpenAI o1/o3)

```yaml
translation_model:
  name: "o3-mini"
  provider: "openai"
  reasoning_level: "medium"  # low, medium, high
```

### Local vLLM Inference

```yaml
translation_model:
  name: "your-model-name"
  provider: "vllm"
  device: "cuda:0"
```

Start the vLLM server first:
```bash
bash src/common_utils/serve_local_vllm.sh
```

---

## Output Files

Translation results are saved with the following naming convention:

```
{output_dir}/{dataset_name}_{subset}_{split}_{language}_{method}.json
```

Example:
```
src/benchmark/data/cais_mmlu_all_test_bg_TRANK.json
```

---

## Supported Languages

The framework has been tested with:

- Bulgarian (bg)
- Estonian (et)
- Greek (el)
- Lithuanian (lt)
- Romanian (ro)
- Slovak (sk)
- Turkish (tr)
- Ukrainian (uk)

To add a new language, create prompt templates in:
```
src/benchmark/prompts/mq_base_translation_prompts/<language>/
```
