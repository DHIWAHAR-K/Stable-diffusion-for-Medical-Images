import os
import json
import random
import re
from collections import defaultdict

# Paths
MERGED_METADATA_IN = "data/merged/metadata.jsonl"
OUTPUT_METADATA = "data/merged/metadata.jsonl" # Overwrite or new file? Let's overwrite responsibly or create new.
# User asked for "metadata.json", let's make satisfied.
OUTPUT_JSON = "data/merged/metadata.json"

def create_splits():
    if not os.path.exists(MERGED_METADATA_IN):
        print("Error: Merged metadata not found.")
        return

    print("Reading merged metadata...")
    all_entries = []
    with open(MERGED_METADATA_IN, 'r') as f:
        for line in f:
            all_entries.append(json.loads(line))
            
    # Containers
    vindr_entries = []
    rsna_entries = []
    
    for entry in all_entries:
        if entry.get('source') == 'vindr':
            vindr_entries.append(entry)
        else:
            rsna_entries.append(entry)
            
    print(f"Found {len(vindr_entries)} Vindr examples.")
    print(f"Found {len(rsna_entries)} RSNA examples.")
    
    # 1. Process Vindr (Respecting existing folder structure splits)
    # Path format: .../data/vindr_processed/{split}/...
    # We allow "test" folder to map to "test" split, etc.
    
    vindr_split_counts = defaultdict(int)
    
    for entry in vindr_entries:
        path = entry['file_name']
        if '/train/' in path:
            entry['split'] = 'train'
        elif '/test/' in path:
            entry['split'] = 'test'
        elif '/val/' in path:
            entry['split'] = 'validation'
        else:
            # Fallback for weird paths?
            # Maybe it's "train" if not specified? 
            # dataset_splits.csv usually covers train/test. 
            # If unsure, assign to train? Or print warning?
            # Let's assume train if not test.
            entry['split'] = 'train'
            
        vindr_split_counts[entry['split']] += 1
        
    print("Vindr existing splits:", dict(vindr_split_counts))

    # 2. Process RSNA (Split by Patient ID)
    # Filename: {patient_id}_{image_id}.png
    
    rsna_patients = defaultdict(list)
    for entry in rsna_entries:
        fname = os.path.basename(entry['file_name'])
        # 10038_1967300488.png
        patient_id = fname.split('_')[0]
        entry['patient_id'] = patient_id # Enriched metadata
        rsna_patients[patient_id].append(entry)
        
    patient_ids = list(rsna_patients.keys())
    random.seed(42)
    random.shuffle(patient_ids)
    
    # Split Ratios: 80% Train, 10% Val, 10% Test
    n_total = len(patient_ids)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    # Rest test
    
    train_patients = set(patient_ids[:n_train])
    val_patients = set(patient_ids[n_train:n_train+n_val])
    test_patients = set(patient_ids[n_train+n_val:])
    
    rsna_split_counts = defaultdict(int)
    
    for pid in patient_ids:
        entries = rsna_patients[pid]
        if pid in train_patients:
            split_label = 'train'
        elif pid in val_patients:
            split_label = 'validation'
        else:
            split_label = 'test'
            
        for e in entries:
            e['split'] = split_label
            rsna_split_counts[split_label] += 1
            
    print("RSNA generated splits:", dict(rsna_split_counts))
    
    # 3. Combine and Save
    final_entries = vindr_entries + rsna_entries
    
    print(f"Saving {len(final_entries)} entries to {MERGED_METADATA_IN}...")
    with open(MERGED_METADATA_IN, 'w') as f:
        for entry in final_entries:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Also saving to {OUTPUT_JSON} as requested...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_entries, f, indent=2)
        
    print("Splits created successfully.")

if __name__ == "__main__":
    create_splits()
