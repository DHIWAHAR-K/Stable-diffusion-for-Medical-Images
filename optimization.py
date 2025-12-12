import os
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from augmentation import MedicalAugmentation
from PIL import Image
import io

class MedicalDataset(Dataset):
    """
    Optimized Dataset with faster image loading and preprocessing.
    Uses pre-tokenized captions for speed.
    """
    def __init__(self, data_dir, tokenizer, split="train", resolution=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.split = split

        self.aug = MedicalAugmentation(resolution=resolution)
        if self.split == "train":
            self.transforms = self.aug.get_train_transforms()
        else:
            self.transforms = self.aug.get_val_transforms()

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

             from datasets import Image as HFImage
             self.dataset = self.dataset.cast_column("image", HFImage())

             # Pre-tokenize all captions for speed
             print(f"Pre-tokenizing captions...")
             def tokenize_captions(examples):
                 inputs = self.tokenizer(
                     examples["text"],
                     max_length=self.tokenizer.model_max_length,
                     padding="max_length",
                     truncation=True,
                     return_tensors="pt"
                 )
                 return {"input_ids": inputs.input_ids}

             self.dataset = self.dataset.map(tokenize_captions, batched=True, batch_size=1000)
             print(f"Successfully loaded {len(self.dataset)} examples for {split}.")

        except Exception as e:
             print(f"Dataset load error: {e}")
             raise e

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # Fast image loading
        image = example["image"]
        if not isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            image = image.convert("RGB")

        # Transform Image -> Pixel Values (already optimized transforms)
        pixel_values = self.transforms(image)

        # Use pre-tokenized input_ids
        input_ids = torch.tensor(example["input_ids"])

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids
        }

def collate_fn(examples):
    """Custom collate function for faster batching"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids
    }

def get_dataloader(data_dir, tokenizer, batch_size, resolution=512, num_workers=4, split="train"):
    dataset = MedicalDataset(data_dir, tokenizer, split=split, resolution=resolution)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
        collate_fn=collate_fn
    )
