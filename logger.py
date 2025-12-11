import csv
import os
import wandb
from datetime import datetime

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
        # Metrics is a dict
        # Print to console
        # log_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        # print(f"Log: {log_str}")

        # Save to CSV
        # Ensure all fields are present (fill with None if missing)
        row = {k: metrics.get(k, None) for k in self.fieldnames}
        
        # Open in append mode each time to be safe against crashes
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
            
        # Log to WandB
        if wandb.run:
            wandb.log(metrics)

    def finish(self):
        if wandb.run:
            wandb.finish()
