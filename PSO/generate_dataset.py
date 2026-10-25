"""Compatibility entrypoint for the configured dataset-preparation workflow.

Dataset settings live in prepare_dataset/config.yaml, not PSO/config.yaml.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PSO.runtime import repository_root, run_python

if __name__ == '__main__':
    run_python(repository_root() / 'prepare_dataset/generate_dataset_main.py')
