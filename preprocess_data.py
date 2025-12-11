import os
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import pandas as pd

# Paths
DATA_DIR = "data/vindr"
SPLIT_FILE = os.path.join(DATA_DIR, "dataset_splits.csv")
PROCESSED_DIR = "data/processed"
METADATA_FILE = os.path.join(PROCESSED_DIR, "metadata.jsonl")

IMAGE_SIZE = (512, 512)

def normalize_image(image_array):
    """Normalize image array to 0-255."""
    if image_array.max() == image_array.min():
        return np.zeros_like(image_array, dtype=np.uint8)
    
    image_array = image_array.astype(float)
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255.0
    return image_array.astype(np.uint8)

def preprocess():
    if not os.path.exists(SPLIT_FILE):
        print(f"Error: Split file not found at {SPLIT_FILE}")
        return

    df = pd.read_csv(SPLIT_FILE)
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    metadata_entries = []
    
    print("Starting preprocessing...")
    success_count = 0
    error_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        file_path = row['file_path'] # Relative path from data/vindr/ e.g. train/High_Density/0/0_L_...
        full_dicom_path = os.path.join(DATA_DIR, file_path)
        
        density_class = row['density_class']
        
        # New relative path for processed image
        # Keeping structure: train/High_Density/0/image.png
        # file_path ends with .dicom, change to .png
        rel_png_path = file_path.replace('.dicom', '.png')
        full_png_path = os.path.join(PROCESSED_DIR, rel_png_path)
        
        # Create directory
        os.makedirs(os.path.dirname(full_png_path), exist_ok=True)
        
        try:
            # Read DICOM
            ds = pydicom.dcmread(full_dicom_path)
            pixel_array = ds.pixel_array
            
            # Normalize
            normalized_array = normalize_image(pixel_array)
            
            # Convert to PIL and Resize
            image = Image.fromarray(normalized_array)
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            # Save
            image.save(full_png_path)
            
            # Add metadata
            # Prompt: "mammogram of density {class}"
            entry = {
                "file_name": rel_png_path,
                "text": f"mammogram of density {density_class}"
            }
            metadata_entries.append(entry)
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {full_dicom_path}: {e}")
            error_count += 1

    # Save metadata.jsonl
    print(f"Saving metadata to {METADATA_FILE}...")
    with open(METADATA_FILE, 'w') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry) + '\n')

    print("Preprocessing complete.")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    preprocess()
