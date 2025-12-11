from torchvision import transforms
import torch

class MedicalAugmentation:
    def __init__(self, resolution=512, center_crop=True):
        self.resolution = resolution
        self.center_crop = center_crop
        
    def get_train_transforms(self):
        """
        Returns a composition of transforms suitable for medical images.
        For mammograms, flip is generally safe.
        Elastic deformations could be added here if needed.
        """
        return transforms.Compose([
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(self.resolution),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)), # Slight affine to add robustness
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
    def get_val_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
