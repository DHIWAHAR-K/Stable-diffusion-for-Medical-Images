from datasets import load_dataset
from torchvision import transforms
import torch

def get_train_dataset(data_dir, tokenizer, resolution=512, center_crop=True):
    dataset = load_dataset("imagefolder", data_dir=data_dir)
    
    train_transforms = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip() if True else transforms.Lambda(lambda x: x), # Hardcoded random flip for now or pass config
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def transforms_fn(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        inputs = tokenizer(
            [ex for ex in examples["text"]], 
            max_length=tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = inputs.input_ids
        return examples

    train_dataset = dataset["train"].with_transform(transforms_fn)
    return train_dataset

def get_dataloader(train_dataset, batch_size, num_workers=0):
    return torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
