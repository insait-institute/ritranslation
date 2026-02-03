# Benchmark Translation Module

This module implements the benchmark translation pipeline for translating question-answer benchmarks (e.g., MMLU, ARC, Hellaswag, Winogrande) from English to target languages.

## Overview

Benchmark translation differs from dataset translation in a crucial way: **questions and answer options must be translated within the same prompt context** to preserve semantic relationships and prevent contextual misleading during evaluation.

This is especially important for:

- Sentence completion tasks (Hellaswag, Winogrande)
- Multiple-choice QA (MMLU, ARC)
- Tasks where answer options have grammatical dependencies on the question

## Module Structure

```
benchmark/
├── methods.py          # Translation method implementations (SC, USI, BoN, T-RANK)
├── model_factory.py    # Multi-provider LLM interface
├── utils.py            # Prompt loading, text processing, output parsing
├── save_to_hf.py       # HuggingFace Hub upload utilities
├── prompts/            # Prompt templates
│   ├── base_prompt_translate.txt
│   ├── self_correction_check.txt
│   ├── universal_self_improvement.txt
│   ├── trank_rank_n.txt
│   ├── best_of_n_scoring.txt
│   ├── few_shot_*.txt              # Few-shot examples
│   └── mq_base_translation_prompts/  # Multi-prompt templates by language
│       ├── bulgarian/
│       ├── estonian/
│       ├── greek/
│       ├── lithuanian/
│       ├── romanian/
│       ├── slovak/
│       ├── turkish/
│       └── ukrainian/
├── eval_mmlu/          # Evaluation scripts
│   ├── evaluate_translations_comet.py      # COMET reference-based evaluation
│   ├── evaluate_mmlu_comet_qe.py           # Quality estimation (reference-free)
│   ├── evaluate_translations_llm_judge.py  # LLM-as-judge evaluation
│   ├── compare_two_translations.py         # Side-by-side comparison
│   └── manual_evaluation.py                # Gradio web interface
└── data/               # Output directory for translated benchmarks
```

## Key Files

### methods.py

Implements four translation methods optimized for QA pairs:

| Method | Description | Model Calls | Best For |
|--------|-------------|-------------|----------|
| **SC** | Self-Correction with optional verification | 1-2 | Quick translations, prototyping |
| **USI** | Universal Self-Improvement with fusion | N+1 | Balanced quality/cost |
| **BoN** | Best-of-N with independent scoring | N+1 | Language-agnostic approach |
| **T-RANK** | Multi-round competitive ranking | 2N+1 | Complex benchmarks, highest quality |

**Key finding from the paper**: T-RANK shows better performance for benchmarks with complex question structures, as its competitive ranking approach helps identify subtle translation errors that other methods fail to correct.

### model_factory.py

Unified interface supporting:

- **OpenAI**: GPT-4o, GPT-4o-mini, o1/o3 reasoning models
- **Google Gemini**: gemini-2.0-flash, gemini-2.5-flash (with thinking config)
- **TogetherAI**: Open-weight models (Llama, Mistral)
- **OpenRouter**: Multi-model aggregation
- **vLLM**: Local inference server

### eval_mmlu/

Evaluation toolkit for assessing translation quality:

```bash
# COMET reference-based evaluation
python src/benchmark/eval_mmlu/evaluate_translations_comet.py

# Quality estimation (no reference needed)
python src/benchmark/eval_mmlu/evaluate_mmlu_comet_qe.py

# LLM-as-judge pairwise comparison
python src/benchmark/eval_mmlu/evaluate_translations_llm_judge.py

# Manual evaluation with Gradio UI
python src/benchmark/eval_mmlu/manual_evaluation.py
```

## Usage

Benchmark translation is invoked via the main `run.py`:

```bash
python run.py --config_path configs/benchmark/MMLU/bench_mmlu_bg.yaml
```

Or programmatically:

```python
from src.initialization import read_config_from_yaml
from src.translate_benchmark import run_benchmark_translation

cfg = read_config_from_yaml("configs/benchmark/ARC/bench_arc_uk.yaml")
run_benchmark_translation(cfg)
```

## Configuration Example

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
    name: "cais/mmlu"
    subset: ["all"]
    split: ["test"]
    n_entries: null

  target_language: "Bulgarian"
  method: "TRANK"              # Recommended for benchmarks
  temperature_translator: 0.5
  temperature_judge: 0.1
  n_samples: 5
  question_fields: ["question"]
  answer_fields: ["choices"]
  multi_prompt: false
```

## Output Format

Translated benchmarks are saved as JSON:

```json
[
  {
    "question": "Original English question",
    "choices": ["A", "B", "C", "D"],
    "answer": 0,
    "question_translated": "Translated question",
    "choices_translated": ["A_bg", "B_bg", "C_bg", "D_bg"],
    // For T-RANK method:
    "ranks": {"0": [1, 2, 1], "1": [2, 1, 2]},
    "raw_ranks": [[1, 2], [2, 1], [1, 2]]
  }
]
```

## Supported Benchmarks

| Benchmark | HuggingFace Name | Fields |
|-----------|------------------|--------|
| MMLU | `cais/mmlu` | question, choices |
| ARC | `allenai/ai2_arc` | question, choices |
| Hellaswag | `Rowan/hellaswag` | ctx, endings |
| Winogrande | `allenai/winogrande` | sentence, option1, option2 |

**Note:** we recommend translating specific Hellaswag columns separately and then merging datasets - look for Hellaswag configs in both `benchmark` and `dataset` folders. 

You can easily configure your own benchmark using our system - just change HF dataset name and columns to translate!

## Adding New Languages

1. Create language-specific prompts in `prompts/mq_base_translation_prompts/<language>/`
2. Add few-shot examples in `prompts/few_shot_*.txt`
3. Create configuration files in `configs/benchmark/`
