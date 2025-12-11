import os
import torch
import shutil

class CheckpointManager:
    def __init__(self, checkpoint_dir, accelerator):
        self.checkpoint_dir = checkpoint_dir
        self.accelerator = accelerator
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_loss = float('inf')

    def save(self, is_best, epoch=None):
        # Save 'last' state
        save_path = os.path.join(self.checkpoint_dir, "last")
        self.accelerator.save_state(save_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best")
            # If exists, remove old best to save space (optional, but good for large models)
            if os.path.exists(best_path):
                shutil.rmtree(best_path)
            shutil.copytree(save_path, best_path)
            print(f"New best model saved at epoch {epoch}!")

    def load_best(self):
         # Logic to load best model if needed for inference
         pass
