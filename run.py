"""
Recovered in Translation: Main Entry Point
==========================================

This is the main entry point for the translation framework. It reads a YAML
configuration file and dispatches to either benchmark or dataset translation
pipelines based on the task type specified.

Usage:
    python run.py --config_path configs/benchmark/MMLU/bench_mmlu_bg.yaml
    python run.py --config_path configs/dataset/WMT/dataset_wmt_uk.yaml

For more details, see the README.md file.

Reference:
    Yukhymenko, H., Alexandrov, A., & Vechev, M. (2025).
    Recovered in Translation: Efficient Pipeline for Automated Translation
    of Benchmarks and Datasets.
"""

import argparse
import warnings
from src.initialization import read_config_from_yaml, Task
from src.translate_benchmark import run_benchmark_translation
from src.translate_dataset import run_dataset_translation
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/dataset/WMT/dataset_wmt_uk.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    cfg = read_config_from_yaml(args.config_path)
    print("==================================== Starting Translation ====================================")
    if cfg.task == Task.BENCHMARK:
        run_benchmark_translation(cfg)
    elif cfg.task == Task.DATASET:
        run_dataset_translation(cfg)

    else:
        raise NotImplementedError(f"Task {cfg.task} not implemented")