import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image

def plot_training_curves(csv_file, output_dir):
    """Generates plots for Loss and Learning Rate from the metrics CSV."""
    if not os.path.exists(csv_file):
        print(f"No CSV found at {csv_file}, skipping plots.")
        return

    try:
        df = pd.read_csv(csv_file)
        
        # Loss Curve
        plt.figure(figsize=(10, 5))
        plt.plot(df['step'], df['loss'], label='Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss per Step')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
        plt.close()

        # LR Curve
        if 'lr' in df.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(df['step'], df['lr'], label='Learning Rate', color='orange')
            plt.xlabel('Steps')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'lr_curve.png'))
            plt.close()

        # Gradient Norm Curve
        if 'grad_norm' in df.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(df['step'], df['grad_norm'], label='Gradient Norm', color='green')
            plt.xlabel('Steps')
            plt.ylabel('Gradient Norm')
            plt.title('Gradient Norm per Step')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'grad_norm_curve.png'))
            plt.close()

        # Epoch vs Loss (Aggregated)
        if 'epoch' in df.columns and 'loss' in df.columns:
            epoch_loss = df.groupby('epoch')['loss'].mean()
            plt.figure(figsize=(10, 5))
            plt.plot(epoch_loss.index, epoch_loss.values, label='Avg Epoch Loss', color='red', marker='o')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Average Loss per Epoch')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'epoch_loss_curve.png'))
            plt.close()

    except Exception as e:
        print(f"Error plotting curves: {e}")

def save_sample_grid(images, path, rows=1):
    """Saves a list of PIL images as a grid."""
    if not images:
        return
        
    num_images = len(images)
    cols = (num_images + rows - 1) // rows
    
    w, h = images[0].size
    grid = Image.new('RGB', (cols * w, rows * h))
    
    for i, img in enumerate(images):
        grid.paste(img, ((i % cols) * w, (i // cols) * h))
        
    grid.save(path)
