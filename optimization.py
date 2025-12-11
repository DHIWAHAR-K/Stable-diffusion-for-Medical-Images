import os
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from augmentation import MedicalAugmentation

class MedicalDataset(Dataset):
    """
    Standard Dataset: Loads images and process them on-the-fly.
    No more caching latents to RAM.
    """
    def __init__(self, data_dir, tokenizer, split="train", resolution=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.split = split
        
        self.aug = MedicalAugmentation(resolution=resolution)
        self.transforms = self.aug.get_train_transforms()
        
        print(f"Loading dataset from {data_dir} for split: {split}...")
        try:
             # Load robustly
             print(f"Loading dataset metadata from {data_dir}/metadata.jsonl...")
             full_ds = load_dataset("json", data_files=os.path.join(data_dir, "metadata.jsonl"), split="train") # HF 'split' arg is for files, not our logical split
             
             # Filter by logical split column
             print(f"Filtering for '{split}' split...")
             self.dataset = full_ds.filter(lambda x: x['split'] == split)
             
             def resolve_image_path(examples):
                 # If path is absolute, use it. If relative, join with data_dir.
                 def get_path(f):
                     if os.path.isabs(f):
                         return f
                     return os.path.abspath(os.path.join(data_dir, f))
                     
                 return {"image": [get_path(f) for f in examples["file_name"]]}
             
             self.dataset = self.dataset.map(resolve_image_path, batched=True)
             
             from datasets import Image
             self.dataset = self.dataset.cast_column("image", Image())
             print(f"Successfully loaded {len(self.dataset)} examples for {split}.")
             
        except Exception as e:
             print(f"Dataset load error: {e}")
             raise e

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"].convert("RGB")
        text = example["text"]
        
        # 1. Transform Image -> Pixel Values
        pixel_values = self.transforms(image) # Returns tensor [3, H, W]
        
        # 2. Tokenize Text -> Input IDs
        inputs = self.tokenizer(
            text, 
            max_length=self.tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": inputs.input_ids.squeeze(0)
        }

def get_dataloader(data_dir, tokenizer, batch_size, resolution=512, num_workers=4, split="train"):
    dataset = MedicalDataset(data_dir, tokenizer, split=split, resolution=resolution)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"), # Shuffle only for training
        num_workers=num_workers,
        pin_memory=True
    )
