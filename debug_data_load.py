
from datasets import load_dataset, Image
import os

data_dir = "data/vindr_processed"

try:
    # Method 1: Robust JSON Load
    print("Attempting JSON load strategy...")
    ds = load_dataset("json", data_files=os.path.join(data_dir, "metadata.jsonl"), split="train")
    print(f"Columns: {ds.column_names}")
    
    # Check first item to see path
    print(f"Sample path: {ds[0]['file_name']}")
    
    # Fix paths to be absolute or relative to CWD
    def update_path(examples):
        return {"image": [os.path.abspath(os.path.join(data_dir, f)) for f in examples["file_name"]]}
    
    ds = ds.map(update_path, batched=True)
    ds = ds.cast_column("image", Image())
    
    print("Columns after Processing:", ds.column_names)
    print("Sample Item:", ds[0])
    print("Success!")
    
except Exception as e:
    print(f"Failed: {e}")
