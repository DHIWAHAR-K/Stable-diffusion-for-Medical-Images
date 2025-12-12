import csv
import os
import wandb
import sys
from datetime import datetime

class Console:
    @staticmethod
    def print(text):
        print(text)

    @staticmethod
    def rule(title="", char="─", width=80):
        print(f"{char * width}")
        if title:
            print(f"  {title}")
        print(f"{char * width}")

    @staticmethod
    def box(title, content_dict=None, text=None, width=80):
        print(f"{'=' * width}")
        print(f"  {title.upper()}")
        print(f"{'=' * width}")
        if content_dict:
            for k, v in content_dict.items():
                print(f"    {k:<25} {v}")
        if text:
            print(f"  {text}")
        print(f"{'=' * width}")

    @staticmethod
    def log_epoch_summary(epoch, total_epochs, metrics, width=80):
        print("\n")
        print(f"{'─' * width}")
        print(f"  Epoch {epoch}/{total_epochs} Summary")
        print(f"{'─' * width}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k:<20} {v:.6f}")
            else:
                print(f"    {k:<20} {v}")
        print("\n")

class MetricLogger:
    def __init__(self, log_dir, project_name=None, config=None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_file = os.path.join(log_dir, "metrics.csv")
        self.project_name = project_name
        self.config = config
        
        # Initialize CSV
        self.fieldnames = ['epoch', 'step', 'loss', 'lr', 'grad_norm']
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def init_wandb(self):
        if self.project_name:
            wandb.init(project=self.project_name, config=self.config)

    def log(self, metrics):
        # Save to CSV
        row = {k: metrics.get(k, None) for k in self.fieldnames}
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
            
        # Log to WandB
        if wandb.run:
            wandb.log(metrics)

    def finish(self):
        if wandb.run:
            wandb.finish()
