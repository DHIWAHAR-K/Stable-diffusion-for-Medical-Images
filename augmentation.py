from torchvision import transforms
import torch

class MedicalAugmentation:
    def __init__(self, resolution=512, center_crop=True):
        self.resolution = resolution
        self.center_crop = center_crop

    def get_train_transforms(self):
        """
        Optimized transforms for medical images with faster preprocessing.
        Using BICUBIC for better quality and PILToTensor for speed.
        """
        transform_list = [
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        ]

        if self.center_crop:
            transform_list.append(transforms.CenterCrop(self.resolution))
        else:
            transform_list.append(transforms.RandomCrop(self.resolution))

        # Lighter augmentation for faster processing
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            # Removed RandomAffine for speed - uncomment if needed
            # transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            transforms.PILToTensor(),  # Faster than ToTensor()
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5], [0.5]),
        ])

        return transforms.Compose(transform_list)

    def get_val_transforms(self):
        """Optimized validation transforms with minimal overhead."""
        return transforms.Compose([
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(self.resolution),
            transforms.PILToTensor(),  # Faster than ToTensor()
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5], [0.5]),
        ])
