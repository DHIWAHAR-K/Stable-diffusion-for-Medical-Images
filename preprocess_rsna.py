import os
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import pandas as pd
# from prompts import MedicalPromptGenerator # Can reuse or just string format here since RSNA metadata is rich

# Paths
RSNA_DIR = "rsna-breast-cancer-detection"
TRAIN_CSV = os.path.join(RSNA_DIR, "train.csv")
PROCESSED_DIR = "data/rsna_processed"
METADATA_FILE = os.path.join(PROCESSED_DIR, "metadata.jsonl")

IMAGE_SIZE = (512, 512)

def normalize_image(image_array):
    """Normalize image array to 0-255."""
    if image_array.max() == image_array.min():
        return np.zeros_like(image_array, dtype=np.uint8)
    
    image_array = image_array.astype(float)
    # Simple Min-Max normalization
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255.0
    
    # Handle photometric interpretation (some DICOMs are inverted)
    # For now, let's assume standard MONOCHROME2 (0 is black). 
    # If MONOCHROME1 (0 is white), we might need to invert. 
    # RSNA is a mix. Let's try to detect if possible, but for simplicity just normalize.
    
    return image_array.astype(np.uint8)

def preprocess_rsna():
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: csv file not found at {TRAIN_CSV}")
        return

    df = pd.read_csv(TRAIN_CSV)
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    metadata_entries = []
    # RSNA Structure: rsna-breast-cancer-detection/train_images/{patient_id}/{image_id}.dcm
    
    print("Starting RSNA preprocessing...")
    success_count = 0
    error_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing RSNA images"):
        # Filter out rows with missing density
        if pd.isna(row['density']):
            continue

        patient_id = str(row['patient_id'])
        image_id = str(row['image_id'])
        laterality = row['laterality'] # L or R
        view = row['view'] # CC or MLO usually
        age = row['age']
        cancer = row['cancer'] # 0 or 1
        density_grade = row['density'] # A, B, C, D
        
        dicom_rel_path = f"train_images/{patient_id}/{image_id}.dcm"
        full_dicom_path = os.path.join(RSNA_DIR, dicom_rel_path)
        
        # New relative path for processed image (Flat structure in rsna_processed)
        png_filename = f"{patient_id}_{image_id}.png"
        full_png_path = os.path.join(PROCESSED_DIR, png_filename)
        
        try:
            # Read DICOM
            ds = pydicom.dcmread(full_dicom_path)
            pixel_array = ds.pixel_array
            
            # Simple Inversion Check (MONOCHROME1 means 0=White, we want 0=Black usually for PNG)
            # But standard X-rays are usually white bones/dense on black bg.
            # actually strict pydicom handling is best.
            if hasattr(ds, "PhotometricInterpretation"):
                 if ds.PhotometricInterpretation == "MONOCHROME1":
                     pixel_array = np.amax(pixel_array) - pixel_array
            
            # Normalize
            normalized_array = normalize_image(pixel_array)
            
            # Convert to PIL and Resize
            image = Image.fromarray(normalized_array)
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            # Save
            image.save(full_png_path)
            
            # --- Prompt Generation (Matching Vindr Style) ---
            # 1. View
            view_str = f"{view} view" if pd.notna(view) else "mammogram"
            
            # 2. Laterality
            lat_str = "Left" if laterality == 'L' else "Right"
            
            # 3. Density Text (Mapping from prompts.py)
            density_map_class = {
                'A': 'low density',
                'B': 'low density',
                'C': 'high density', 
                'D': 'high density'
            }
            density_desc = density_map_class.get(density_grade, "medical")
            density_text = f"{density_desc} tissue (Density {density_grade})"
            
            # 4. Findings (Cancer)
            cancer_text = "positive for cancer" if cancer == 1 else "negative for cancer"
            
            # "A CC view mammogram of the Left breast showing high density tissue (Density C) with findings positive for cancer."
            prompt = f"A {view_str} mammogram of the {lat_str} breast showing {density_text} with findings {cancer_text}."
            
            entry = {
                "file_name": png_filename, 
                "text": prompt,
                "density_class": density_grade
            }
            metadata_entries.append(entry)
            
            success_count += 1
            
        except Exception as e:
            # print(f"Error processing {full_dicom_path}: {e}") 
            error_count += 1

    # Save metadata.jsonl
    print(f"Saving metadata to {METADATA_FILE}...")
    with open(METADATA_FILE, 'w') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry) + '\n')

    print("RSNA Preprocessing complete.")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    preprocess_rsna()
