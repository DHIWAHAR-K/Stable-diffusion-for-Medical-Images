import torch
import torchmetrics
from cleanfid import fid
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore
from device import get_device

class MedicalMetrics:
    def __init__(self, device=None):
        self.device = device if device else get_device()
        # Image Quality / Fidelity
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.device)
        
        # Distribution (Diversity)
        self.inception_score = InceptionScore().to(self.device)
        
        # Text-Image Alignment
        self.clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    def compute_fid(self, real_path, fake_path):
        """
        Computes FID between two folders using clean-fid.
        Robust standard for research.
        """
        score = fid.compute_fid(real_path, fake_path, device=torch.device(self.device))
        return score
    
    def compute_kid(self, real_path, fake_path):
        """
        Computes Kernel Inception Distance (KID). Unbiased estimator.
        """
        score = fid.compute_kid(real_path, fake_path, device=torch.device(self.device))
        return score

    def compute_batch_metrics(self, real_images, fake_images, prompts=None):
        """
        Computes metrics that work on batches of tensors.
        real_images, fake_images: (B, C, H, W) in [0, 1]
        prompts: List of strings for CLIP score
        """
        results = {}
        
        # Pixel/Structural
        results['ssim'] = self.ssim_metric(fake_images, real_images).item()
        results['psnr'] = self.psnr_metric(fake_images, real_images).item()
        results['ms_ssim'] = self.ms_ssim_metric(fake_images, real_images).item()
        
        # Perceptual
        results['lpips'] = self.lpips_metric(fake_images, real_images).item()
        
        # Diversity (IS requires update step usually, but here we can compute on batch if large enough, 
        # or handle accumulation outside. torchmetrics IS accumulates.
        # For simplicity in this wrapper, we assume accumulation is handled by caller or we update/compute here).
        self.inception_score.update((fake_images * 255).byte()) # Expects [0, 255] uint8
        is_mean, is_std = self.inception_score.compute()
        results['inception_score'] = is_mean.item()
        self.inception_score.reset() # Reset for next batch
        
        # Semantic (CLIP)
        if prompts:
            # CLIP Score expects (B, C, H, W) in [0, 255]? No, usually [0, 1] or specific norm. 
            # Torchmetrics CLIP score handles raw images (0-255 uint8) or tensors.
            # Let's use (fake_images * 255).int()
            self.clip_score.update((fake_images * 255).to(torch.uint8), prompts)
            results['clip_score'] = self.clip_score.compute().item()
            self.clip_score.reset()
            
        return results

if __name__ == "__main__":
    # Test stub
    pass
