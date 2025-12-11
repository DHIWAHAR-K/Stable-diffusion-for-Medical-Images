import pandas as pd
import os
import shutil
from tqdm import tqdm

# Paths
DATA_DIR = "vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
IMAGES_DIR = f"{DATA_DIR}/images"
SPLIT_FILE = "dataset_splits.csv"
OUTPUT_DIR = "dataset_organized"

def sanitize(text):
    """Sanitize text for filename compatibility."""
    if pd.isna(text):
        return "NA"
    return str(text).replace(" ", "_").replace("/", "-")

def get_density_class(density):
    if density in ['DENSITY A', 'DENSITY B']:
        return 'Low_Density'
    elif density in ['DENSITY C', 'DENSITY D']:
        return 'High_Density'
    return 'Unknown_Density'

def organize_images():
    print(f"Loading split data from {SPLIT_FILE}...")
    try:
        df = pd.read_csv(SPLIT_FILE)
    except FileNotFoundError:
        print(f"Error: File not found at {SPLIT_FILE}")
        return

    # Create output directories
    for split in ['train', 'val', 'test']:
        for density_class in ['High_Density', 'Low_Density']:
            os.makedirs(os.path.join(OUTPUT_DIR, split, density_class), exist_ok=True)

    print("Starting image organization...")
    
    success_count = 0
    error_count = 0
    missing_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        study_id = row['study_id']
        patient_id = row['patient_id'] # Use mapped numeric ID
        image_id = row['image_id']
        split = row['split']
        breast_density = row['breast_density']
        
        # Determine density class
        density_class = get_density_class(breast_density)
        
        # Patient details for filename
        laterality = sanitize(row['laterality'])
        view = sanitize(row['view_position'])
        birads = sanitize(row['breast_birads'])
        density = sanitize(breast_density)
        
        # Construct paths
        src_path = os.path.join(IMAGES_DIR, study_id, f"{image_id}.dicom")
        
        # New filename: {patient_id}_{laterality}_{view_position}_{breast_birads}_{breast_density}_{image_id}.dicom
        new_filename = f"{patient_id}_{laterality}_{view}_{birads}_{density}_{image_id}.dicom"
        
        # Destination folder: dataset_organized/{split}/{density_class}/{patient_id}/
        dest_folder = os.path.join(OUTPUT_DIR, split, density_class, str(patient_id))
        os.makedirs(dest_folder, exist_ok=True)
        
        dest_path = os.path.join(dest_folder, new_filename)
        
        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dest_path)
                success_count += 1
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
                error_count += 1
        else:
            # Try without .dicom extension just in case
            src_path_no_ext = os.path.join(IMAGES_DIR, study_id, image_id)
            if os.path.exists(src_path_no_ext):
                 try:
                    shutil.copy2(src_path_no_ext, dest_path)
                    success_count += 1
                 except Exception as e:
                    print(f"Error copying {src_path_no_ext}: {e}")
                    error_count += 1
            else:
                missing_count += 1

    print("\nOrganization complete.")
    print(f"Successfully copied: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Missing files: {missing_count}")
    
    # Verification of counts
    print("\nVerifying output counts:")
    for split in ['train', 'val', 'test']:
        for density_class in ['High_Density', 'Low_Density']:
            class_dir = os.path.join(OUTPUT_DIR, split, density_class)
            count = sum([len(files) for r, d, files in os.walk(class_dir)])
            
            # Get expected count
            expected_densities = ['DENSITY A', 'DENSITY B'] if density_class == 'Low_Density' else ['DENSITY C', 'DENSITY D']
            expected = len(df[(df['split'] == split) & (df['breast_density'].isin(expected_densities))])
            
            print(f"{split}/{density_class}: Found {count}, Expected {expected}")

if __name__ == "__main__":
    organize_images()
