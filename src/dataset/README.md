# Dataset Translation Module

This module implements the dataset translation pipeline for translating text datasets (e.g., FLORES, WMT) from English to target languages.

## Overview

Unlike benchmark translation, dataset translation processes individual text fields without the question-answer structure. This is suitable for:

- Machine translation evaluation datasets (FLORES, WMT24++)
- Parallel corpora creation
- General text translation tasks

## Module Structure

```
dataset/
├── methods.py          # Translation method implementations (SC, USI, BoN, T-RANK)
├── model_factory.py    # Multi-provider LLM interface (OpenAI, Gemini, Together, etc.)
├── utils.py            # Prompt loading, text processing, output parsing
├── save_to_hf.py       # HuggingFace Hub upload utilities
├── prompts/            # Prompt templates
│   ├── base_prompt_translate.txt
│   ├── self_correction_check.txt
│   ├── universal_self_improvement.txt
│   ├── trank_rank_n.txt
│   ├── best_of_n_scoring.txt
│   └── mq_base_translation_prompts/   # Multi-prompt templates by language
└── data/               # Output directory for translated datasets
    ├── flores/         # FLORES translation results
    ├── wmt/            # WMT translation results
```

## Key Files

### methods.py

Implements four translation methods:

| Method | Description | Model Calls | Best For |
|--------|-------------|-------------|----------|
| **SC** | Self-Correction with optional verification | 1-2 | High-resource languages, large texts |
| **USI** | Universal Self-Improvement with fusion | N+1 | Short texts, cost-efficient |
| **BoN** | Best-of-N with independent scoring | N+1 | Language-agnostic approach |
| **T-RANK** | Multi-round competitive ranking | 2N+1 | Complex texts, highest quality |

### model_factory.py

Unified interface for multiple LLM providers:

- **OpenAI**: GPT-4o, GPT-4o-mini, o1, o3-mini
- **Google Gemini**: gemini-2.0-flash, gemini-2.5-flash
- **TogetherAI**: Llama, Mistral, and other open-weight models
- **OpenRouter**: Multi-model aggregation
- **vLLM**: Local inference server

### utils.py

Utility functions for:

- Prompt template loading and variable substitution
- Text chunking for large documents (>10k words)
- Output parsing and extraction
- Ranking combination generation for T-RANK

## Usage

Dataset translation is invoked via the main `run.py`:

```bash
python run.py --config_path configs/dataset/FLORES/dataset_flores_bg.yaml
```

Or programmatically:

```python
from src.initialization import read_config_from_yaml
from src.translate_dataset import run_dataset_translation

cfg = read_config_from_yaml("configs/dataset/WMT/dataset_wmt_uk.yaml")
run_dataset_translation(cfg)
```

## Configuration Example

```yaml
task: "DATASET"
output_dir: "src/dataset/data/flores/bg"

translation_model:
  name: "gpt-4o-mini-2024-07-18"
  provider: "openai"

task_config:
  dataset:
    name: "gsarti/flores_101"
    subset: ["eng"]
    split: ["devtest"]

  target_language: "Bulgarian"
  method: "USI"              # Recommended for short dataset texts
  fields: ["sentence"]
  temperature_translator: 0.5
  temperature_judge: 0.1
  n_samples: 5
```

## Output Format

Translated datasets are saved as JSON with the following structure:

```json
[
  {
    "sentence": "Original English text",
    "sentence_translated": "Translated text in target language",
    // For T-RANK method:
    "sentence_ranks": {"0": [1, 2], "1": [2, 1]},
    "sentence_raw_ranks": [[1, 2], [2, 1]]
  }
]
```

## Adding New Languages

1. Create language-specific prompts in `prompts/mq_base_translation_prompts/<language>/`
2. Add few-shot examples if needed
3. Create a configuration file in `configs/dataset/`
