import yaml
import argparse
from pathlib import Path
from dataclasses import dataclass

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion Training")
    parser.add_argument("--config", type=str, default="configs/train_params.yaml", help="Path to config file")
    args = parser.parse_args()
    return args
