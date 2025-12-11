import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from diffusers import AutoencoderKL
from torchvision import transforms
from PIL import Image
from device import get_device

class LatentAnalyzer:
    def __init__(self, vae_path, device=None):
        self.device = device if device else get_device()
        self.vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae").to(self.device)
        self.vae.eval()
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def encode_images(self, image_paths):
        latents_list = []
        print(f"Encoding {len(image_paths)} images...")
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                pixel_val = self.transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    latents = self.vae.encode(pixel_val).latent_dist.sample()
                latents_list.append(latents.cpu().flatten().numpy())
            except Exception as e:
                print(f"Failed to load {path}: {e}")
        return np.array(latents_list)

    def plot_tsne(self, real_latents, fake_latents, save_path):
        print("Computing t-SNE...")
        all_latents = np.concatenate([real_latents, fake_latents], axis=0)
        
        # PCA first to reduce dimensionality (recommended before t-SNE for high dim data)
        pca = PCA(n_components=50)
        pca_result = pca.fit_transform(all_latents)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, verbose=1)
        tsne_results = tsne.fit_transform(pca_result)
        
        n_real = len(real_latents)
        
        plt.figure(figsize=(10, 10))
        plt.scatter(tsne_results[:n_real, 0], tsne_results[:n_real, 1], c='blue', label='Real', alpha=0.5)
        plt.scatter(tsne_results[n_real:, 0], tsne_results[n_real:, 1], c='red', label='Synthetic', alpha=0.5)
        plt.legend()
        plt.title("t-SNE of VAE Latents (Real vs Synthetic)")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved t-SNE plot to {save_path}")

if __name__ == "__main__":
    # Test stub
    pass
