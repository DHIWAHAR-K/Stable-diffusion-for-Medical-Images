import os
import json
import shutil
from tqdm import tqdm

# Config
# We use the merged metadata to get the split info for RSNA files
MERGED_METADATA = "data/merged/metadata.jsonl"
RSNA_DIR = "data/rsna_processed"

def organize_rsna():
    if not os.path.exists(MERGED_METADATA):
        print(f"Error: {MERGED_METADATA} not found.")
        return

    print(f"Reading splits from {MERGED_METADATA}...")
    rsna_split_map = {}
    
    with open(MERGED_METADATA, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('source') == 'rsna':
                # Entry filename is absolute path to rsna_processed/file.png
                # We extract basename to match files in RSNA_DIR
                basename = os.path.basename(entry['file_name'])
                split = entry.get('split', 'train')
                rsna_split_map[basename] = split
                
    print(f"Found {len(rsna_split_map)} RSNA entries to organize.")
    
    # Create split directories
    for split in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(RSNA_DIR, split), exist_ok=True)
        
    # Iterate and Move
    print("Moving files...")
    moved_count = 0
    missing_count = 0
    
    # We iterate through the map to ensure we only move known files
    for filename, split in tqdm(rsna_split_map.items()):
        src_path = os.path.join(RSNA_DIR, filename)
        
        # Check if file exists (it should, unless already moved)
        if not os.path.exists(src_path):
            # Check if it's already in the target folder?
            dest_path = os.path.join(RSNA_DIR, split, filename)
            if os.path.exists(dest_path):
                # Already moved
                continue
            else:
                missing_count += 1
                continue
        
        dest_path = os.path.join(RSNA_DIR, split, filename)
        
        try:
            shutil.move(src_path, dest_path)
            moved_count += 1
        except Exception as e:
            print(f"Error moving {filename}: {e}")
            
    print(f"Organization complete.")
    print(f"Moved: {moved_count}")
    print(f"Missing/Already Moved: {missing_count}")
    
    # Clean up merged directory
    merged_dir = "data/merged"
    if os.path.exists(merged_dir):
        print(f"Removing {merged_dir}...")
        shutil.rmtree(merged_dir)
        print("Removed.")

if __name__ == "__main__":
    organize_rsna()
