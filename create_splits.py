import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Paths
DATA_DIR = "vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
ANNOTATIONS_FILE = f"{DATA_DIR}/breast-level_annotations.csv"
OUTPUT_FILE = "dataset_splits.csv"

def create_splits():
    print(f"Loading data from {ANNOTATIONS_FILE}...")
    try:
        df = pd.read_csv(ANNOTATIONS_FILE)
    except FileNotFoundError:
        print(f"Error: File not found at {ANNOTATIONS_FILE}")
        return

    # Check existing splits
    print("Initial split distribution:")
    print(df['split'].value_counts())
    
    # Separate test set (preserve original test set)
    test_df = df[df['split'] == 'test'].copy()
    train_val_df = df[df['split'] == 'training'].copy()
    
    # Get unique study_ids for splitting
    # We split by study_id to ensure no patient leakage (assuming study_id maps to patient session)
    
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    split = splitter.split(train_val_df, groups=train_val_df['study_id'])
    train_inds, val_inds = next(split)
    
    train_df = train_val_df.iloc[train_inds].copy()
    val_df = train_val_df.iloc[val_inds].copy()
    
    # Assign new split labels
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Combine back
    final_df = pd.concat([train_df, val_df, test_df])
    
    # Generate patient_id mapping
    # Sort unique study_ids to ensure deterministic mapping
    unique_study_ids = sorted(final_df['study_id'].unique())
    study_id_to_patient_id = {study_id: i for i, study_id in enumerate(unique_study_ids)}
    
    # Map to numeric string with padding (e.g., 00001)
    # Using 5 digits to cover all 5000 studies
    final_df['patient_id'] = final_df['study_id'].map(lambda x: f"{study_id_to_patient_id[x]:05d}")
    
    print("\nFinal split distribution:")
    print(final_df['split'].value_counts())
    
    print(f"\nTotal samples: {len(final_df)}")
    print(f"Total unique patients (mapped): {len(unique_study_ids)}")
    print(f"Example mapping: {unique_study_ids[0]} -> {study_id_to_patient_id[unique_study_ids[0]]:05d}")
    
    # Verification
    train_studies = set(train_df['study_id'])
    val_studies = set(val_df['study_id'])
    test_studies = set(test_df['study_id'])
    
    leakage_tv = train_studies.intersection(val_studies)
    leakage_tt = train_studies.intersection(test_studies)
    leakage_vt = val_studies.intersection(test_studies)
    
    print("\nVerification (Leakage check by study_id):")
    print(f"Train/Val leakage: {len(leakage_tv)} studies")
    print(f"Train/Test leakage: {len(leakage_tt)} studies")
    print(f"Val/Test leakage: {len(leakage_vt)} studies")
    
    if len(leakage_tv) == 0 and len(leakage_tt) == 0 and len(leakage_vt) == 0:
        print("SUCCESS: No leakage detected.")
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved splits to {OUTPUT_FILE}")
    else:
        print("ERROR: Leakage detected! Splits not saved.")

if __name__ == "__main__":
    create_splits()
