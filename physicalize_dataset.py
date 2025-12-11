import os
import json
import shutil
from tqdm import tqdm

# Config
INPUT_METADATA = "data/merged/metadata.jsonl"
OUTPUT_DIR = "data/vindr_rsna"
OUTPUT_METADATA_JSONL = os.path.join(OUTPUT_DIR, "metadata.jsonl")
OUTPUT_METADATA_JSON = os.path.join(OUTPUT_DIR, "metadata.json")

def consolidate_and_physicalize():
    if not os.path.exists(INPUT_METADATA):
        print(f"Error: {INPUT_METADATA} not found.")
        return

    print(f"Reading metadata from {INPUT_METADATA}...")
    entries = []
    with open(INPUT_METADATA, 'r') as f:
        for line in f:
            entries.append(json.loads(line))
            
    print(f"Loaded {len(entries)} entries. Starting physical consolidation to {OUTPUT_DIR}...")
    
    # Create output directories
    for split in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
        
    new_entries = []
    
    for entry in tqdm(entries, desc="Copying images"):
        original_path = entry['file_name']
        split = entry.get('split', 'train') # Default to train if missing
        
        # Prepare destination filename
        # We want to flatten the structure within the split folder to avoid deep nesting issues
        # But we must avoid name collisions.
        # Vindr: .../train/High_Density/PID/filename.png
        # RSNA: .../PID_ImageID.png
        # Strategy: Use unique filename.
        
        basename = os.path.basename(original_path)
        
        # If RSNA (source=rsna), basename is PID_ID.png (Unique)
        # If Vindr, basename is PID_L_CC_...hash.png (Unique due to hash)
        # So basename should be safe.
        
        dest_rel_path = os.path.join(split, basename)
        dest_abs_path = os.path.abspath(os.path.join(OUTPUT_DIR, dest_rel_path))
        
        # Copy File
        try:
            shutil.copy2(original_path, dest_abs_path)
            
            # Update Metadata Entry
            new_entry = entry.copy()
            new_entry['file_name'] = dest_rel_path # Relative to new data_dir is cleaner for portability?
            # Actually optimization.py expects absolute or relative to data_dir.
            # Let's verify optimization.py logic:
            # "return os.path.abspath(os.path.join(data_dir, f))"
            # So relative path "train/image.png" works perfectly.
            
            new_entries.append(new_entry)
            
        except Exception as e:
            print(f"Failed to copy {original_path}: {e}")
            
    # Save New Metadata
    print(f"Saving new metadata to {OUTPUT_METADATA_JSONL}...")
    with open(OUTPUT_METADATA_JSONL, 'w') as f:
        for entry in new_entries:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Saving new metadata to {OUTPUT_METADATA_JSON}...")
    with open(OUTPUT_METADATA_JSON, 'w') as f:
        json.dump(new_entries, f, indent=2)
        
    print("Consolidation complete.")

if __name__ == "__main__":
    consolidate_and_physicalize()
