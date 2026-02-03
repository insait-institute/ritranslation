"""
Configuration and Initialization Module
=======================================

This module defines the configuration schema for the translation framework
using Pydantic models. It supports two main task types:

- BENCHMARK: For translating question-answer benchmarks (MMLU, ARC, etc.)
- DATASET: For translating text datasets (FLORES, WMT, etc.)

Configuration is read from YAML files and validated using Pydantic.

Classes:
    Task: Enum for task types (BENCHMARK, DATASET)
    ModelConfig: LLM model configuration (provider, name, etc.)
    SourceDataConfig: Dataset/benchmark source configuration
    BENCHConfig: Benchmark-specific translation settings
    DATAConfig: Dataset-specific translation settings
    Config: Main configuration container

Functions:
    read_config_from_yaml: Parse and validate YAML configuration file
"""

import yaml
from pydantic import ValidationError
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel as PBM
from pydantic import Extra, Field


class Task(Enum):
    BENCHMARK = "BENCHMARK"
    DATASET = "DATASET"


class ModelConfig(PBM):

    name: str = Field("gpt-4o-mini-2024-07-18",
        description="Name of the model"
    )
    tokenizer_name: Optional[str] = Field(
        None, description="Name of the tokenizer to use"
    )
    provider: str = Field("openai",
        description="Provider of the model"
    )
    device: str = Field(
        "auto", description="Device to use for the model (only used for local models)"
    )
    max_workers: int = Field(
        1, description="Number of workers to use for parallel generation"
    )
    reasoning_level: str = Field(
        "low", description="Reasoning level (for OpenAI reasoning models)"
    )
    model_template: str = Field(
        default="{prompt}",
        description="Template to use for the model (only used for local models)"
    )
    prompt_template: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the prompt"
    )

class SourceDataConfig(PBM):
    
    name: Optional[str] = Field(
        None, description="Name of the dataset/benchmark to use"
    )
    subset: Optional[list] = Field(
        None, description="Subset of the dataset/benchmark to use"
    )
    split: Optional[list] = Field(
        None, description="Split of the dataset/benchmark to use"
    )
    n_entries: Optional[int] = Field(
        None, description="Number of entries to sample from dataset/benchmark"
    )


class BENCHConfig(PBM):

    benchmark: SourceDataConfig = Field(
        default=None, description="Model to use for translation"
    )

    target_language: str = Field(
        default=None,
        description="Language to translate to"
    )

    method: str = Field(
        default=None,
        description="Translation method to use. Select from [SC, USC, TRANK, BoN]"
    )

    temperature_translator: Optional[float] = Field(
        0.5, description="Temperature value for translator model (during sampling of translation candidate(s))"
    )

    temperature_judge: Optional[float] = Field(
        0.1, description="Temperature value for judge model (during correction or evaluation of translation candidate(s))"
    )

    max_workers: bool = Field(
        False, description="Whether to use maximum number of workers (no. of chips - 1) for parallel generation by default or not"
    )

    num_workers: int = Field(
        1, description="Number of workers to use for parallel generation"
    )

    question_fields: list = Field(
        default=["question"],
        description="The list of question fields to translate"
    )

    answer_fields: list = Field(
        default=["answer"],
        description="The list of answer fields to translate"
    )

    agent_check: bool = Field(
        default=False,
        description="If set to True, then the agent translation check will be performed"
    )

    few_shot: bool = Field(
        default=False,
        description="If this argument is set as True then few-shot prompt will be used"
    )

    n_samples: int = Field(
        3, description="Number of translations to generate for each input"
    )

    multi_prompt: bool = Field(
        default=False,
        description="If set to True then multiple prompts will be used"
    )

    save_to_hf: bool = Field(
        default=False,
        description="If set to True then dataset will be pushed to Hugging Face"
    )

    prompt_files: list = Field(
        default=["src/benchmark/prompts/base_prompt_translate.txt"],
        description="The file paths to prompts for multi-prompt USC and TRANK methods."
    )

    translation_prompt_file: str = Field(
        default="./src/benchmark/prompts/base_prompt_translate.txt",
        description="The file path to translation prompt in single-prompt setting."
    )

    judge_prompt_file: str = Field(
        default=None,
        description="The file path to judging prompt for SC, BoN, USC and TRANK methods."
    )

    class Config:
        extra = Extra.forbid


class DATAConfig(PBM):

    dataset: SourceDataConfig = Field(
        default=None, description="Model to use for translation"
    )

    target_language: str = Field(
        default=None,
        description="Language to translate to"
    )

    method: str = Field(
        default=None,
        description="Translation method to use. Select from [SC, USC, TRANK, BoN]"
    )

    temperature_translator: float = Field(
        0.5, description="Temperature value for translator model (during sampling of translation candidate(s))"
    )

    temperature_judge: float = Field(
        0.1, description="Temperature value for judge model (during correction or evaluation of translation candidate(s))"
    )

    max_workers: bool = Field(
        False, description="Whether to use maximum number of workers (no. of chips - 1) for parallel generation by default or not"
    )

    num_workers: int = Field(
        1, description="Number of workers to use for parallel generation"
    )

    fields: list = Field(
        default=["text"],
        description="The features which to translate"
    )

    agent_check: bool = Field(
        default=False,
        description="If set to True, then the agent translation check will be performed"
    )

    few_shot: bool = Field(
        default=False,
        description="If this argument is set as True then few-shot prompt will be used"
    )

    n_samples: int = Field(
        3, description="Number of translations to generate for each input"
    )

    multi_prompt: bool = Field(
        default=False,
        description="If set to True then multiple prompts will be used"
    )

    save_to_hf: bool = Field(
        default=False,
        description="If set to True then dataset will be pushed to Hugging Face"
    )

    prompt_files: list = Field(
        default=["src/dataset/prompts/base_prompt_translate.txt"],
        description="The file paths to prompts for multi-prompt USC and TRANK methods."
    )

    translation_prompt_file: str = Field(
        default="./src/dataset/prompts/base_prompt_translate.txt",
        description="The file path to translation prompt in single-prompt setting."
    )

    judge_prompt_file: str = Field(
        default=None,
        description="The file path to judging prompt for SC, BoN, USC and TRANK methods."
    )

    class Config:
        extra = Extra.forbid


class Config(PBM):

    output_dir: str = Field(
        default=None, description="Directory to store the translations in"
    )
    task: Task = Field(
        default=None, description="Task to run", choices=list(Task.__members__.values())
    )
    task_config: ( BENCHConfig | DATAConfig ) = Field(
        default=None, description="Config for the task"
    )
    translation_model: ModelConfig = Field(
        default=None, description="Model to use for translation"
    )
    judge_model: ModelConfig = Field(
        default=None, description="Model to use for evaluating/ranking translations"
    )
    timeout: int = Field(
        0.5, description="Timeout in seconds between requests for API restrictions"
    )

def read_config_from_yaml(path: str) -> Config:
    """
    Read and validate a YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Config: Validated configuration object.

    Raises:
        yaml.YAMLError: If the YAML file is malformed.
        ValidationError: If the configuration doesn't match the schema.

    Example:
        >>> cfg = read_config_from_yaml("configs/benchmark/MMLU/bench_mmlu_bg.yaml")
        >>> print(cfg.task)  # Task.BENCHMARK
    """
    with open(path, "r") as stream:
        try:
            yaml_obj = yaml.safe_load(stream)
            print(yaml_obj)
            cfg = Config(**yaml_obj)
            return cfg
        except (yaml.YAMLError, ValidationError) as exc:
            print(exc)
            raise exc
