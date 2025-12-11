import os
import json
import shutil

# Paths
VINDR_METADATA = "data/vindr_processed/metadata.jsonl"
RSNA_METADATA = "data/rsna_processed/metadata.jsonl"

MERGED_DIR = "data/merged"
MERGED_METADATA = os.path.join(MERGED_DIR, "metadata.jsonl")

def merge_datasets():
    print("Merging dataset metadata...")
    os.makedirs(MERGED_DIR, exist_ok=True)
    
    merged_entries = []
    
    # 1. Load Vindr
    if os.path.exists(VINDR_METADATA):
        print(f"Loading Vindr metadata from {VINDR_METADATA}...")
        with open(VINDR_METADATA, 'r') as f:
            for line in f:
                entry = json.loads(line)
                # Convert relative path to absolute or relative to MERGED_DIR
                # entry['file_name'] is like "train/...png" relative to vindr_processed
                # We want absolute path to be safe
                abs_path = os.path.abspath(os.path.join(os.path.dirname(VINDR_METADATA), entry['file_name']))
                entry['file_name'] = abs_path
                entry['source'] = 'vindr'
                merged_entries.append(entry)
    else:
        print("Warning: Vindr metadata not found.")

    # 2. Load RSNA
    if os.path.exists(RSNA_METADATA):
        print(f"Loading RSNA metadata from {RSNA_METADATA}...")
        with open(RSNA_METADATA, 'r') as f:
            for line in f:
                entry = json.loads(line)
                # entry['file_name'] is "id.png" relative to rsna_processed
                abs_path = os.path.abspath(os.path.join(os.path.dirname(RSNA_METADATA), entry['file_name']))
                entry['file_name'] = abs_path
                entry['source'] = 'rsna'
                merged_entries.append(entry)
    else:
        print("Warning: RSNA metadata not found.")
        
    print(f"Total merged examples: {len(merged_entries)}")
    
    # 3. Save Merged
    print(f"Saving merged metadata to {MERGED_METADATA}...")
    with open(MERGED_METADATA, 'w') as f:
        for entry in merged_entries:
            f.write(json.dumps(entry) + '\n')
            
    print("Merge complete.")
    
    # 4. Create a dummy 'train' folder link if mostly needed? 
    # Actually, our Dataset class in optimization.py loads metadata and uses 'file_name' directly.
    # It filters by startswith("train/")... wait.
    # Vindr files start with "train/..."
    # RSNA files do not.
    # We need to ensure the Dataset class can handle absolute paths and doesn't filter out RSNA.
    
if __name__ == "__main__":
    merge_datasets()
