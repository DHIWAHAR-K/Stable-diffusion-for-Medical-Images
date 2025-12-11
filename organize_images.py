import pandas as pd
import os
import shutil
from tqdm import tqdm

# Paths
DATA_DIR = "vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
IMAGES_DIR = f"{DATA_DIR}/images"
SPLIT_FILE = "dataset_splits.csv"
OUTPUT_DIR = "data/vindr" # Updated output directory
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "dataset_splits.csv")

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

    print("Starting image organization and metadata update...")
    
    success_count = 0
    error_count = 0
    missing_count = 0
    
    # Lists to store new column data
    density_classes = []
    file_paths = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        study_id = row['study_id']
        patient_id = row['patient_id']
        image_id = row['image_id']
        split = row['split']
        breast_density = row['breast_density']
        
        # Determine density class
        density_class = get_density_class(breast_density)
        density_classes.append(density_class)
        
        # Patient details for filename
        laterality = sanitize(row['laterality'])
        view = sanitize(row['view_position'])
        birads = sanitize(row['breast_birads'])
        density = sanitize(breast_density)
        
        # New filename
        new_filename = f"{patient_id}_{laterality}_{view}_{birads}_{density}_{image_id}.dicom"
        
        # Destination folder and path
        dest_folder = os.path.join(OUTPUT_DIR, split, density_class, str(patient_id))
        os.makedirs(dest_folder, exist_ok=True)
        
        dest_path = os.path.join(dest_folder, new_filename)
        
        # Relative path for CSV
        rel_path = os.path.join(split, density_class, str(patient_id), new_filename)
        file_paths.append(rel_path)
        
        # Construct source path
        src_path = os.path.join(IMAGES_DIR, study_id, f"{image_id}.dicom")
        
        # Copy file if it doesn't exist
        if not os.path.exists(dest_path):
            if os.path.exists(src_path):
                try:
                    shutil.copy2(src_path, dest_path)
                    success_count += 1
                except Exception as e:
                    print(f"Error copying {src_path}: {e}")
                    error_count += 1
            else:
                # Try without .dicom extension
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
        else:
            # File already exists, verify success count? No, just skip copy.
            # Should we count it as success? Since we are re-running for CSV update, finding it there is success.
            success_count += 1

    # Update DataFrame with new columns
    df['density_class'] = density_classes
    df['file_path'] = file_paths
    
    # Save updated CSV
    print(f"\nSaving updated metadata to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nOrganization complete.")
    print(f"Files processed/verified: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Missing files: {missing_count}")
    
    # Verification of counts matches CSV
    print("\nVerifying output counts:")
    for split in ['train', 'val', 'test']:
        for density_class in ['High_Density', 'Low_Density']:
            class_dir = os.path.join(OUTPUT_DIR, split, density_class)
            count = sum([len(files) for r, d, files in os.walk(class_dir)])
            
            # Get expected count from UPDATED df
            expected = len(df[(df['split'] == split) & (df['density_class'] == density_class)])
            
            print(f"{split}/{density_class}: Found {count}, Expected {expected}")

if __name__ == "__main__":
    organize_images()
