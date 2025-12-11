import argparse
import os
import subprocess
import json
import torch
from metrics import MedicalMetrics
from inference import main as infer_main # Or run via subprocess properly

def run_inference(model_path, output_dir, num_samples=500):
    print(f"Generating {num_samples} samples for {model_path}...")
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= num_samples:
        print(f"Output dir {output_dir} already has images. Skipping generation.")
        return

    cmd = [
        "python", "inference.py",
        "--model_path", model_path,
        "--output_dir", output_dir,
        "--num_samples", str(num_samples),
        "--batch_size", "1"
    ]
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser(description="Multi-Model Benchmark Comparison")
    parser.add_argument("--finetuned_path", type=str, required=True, help="Path to finetuned pipeline")
    parser.add_argument("--real_data_dir", type=str, default="data/processed/train/High_Density/0", help="Reference real images")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define models to compare
    baselines = {
        "SD-v1-4": "CompVis/stable-diffusion-v1-4",
        "SD-v1-5": "runwayml/stable-diffusion-v1-5",
        "SD-v2-1": "stabilityai/stable-diffusion-2-1" 
    }
    
    results = {}
    
    # 0. Setup Metrics
    print("Initializing metrics...")
    metrics = MedicalMetrics(device="mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Evaluate Baselines
    for name, model_id in baselines.items():
        print(f"\n--- Benchmarking {name} ({model_id}) ---")
        out_path = os.path.join(args.output_dir, f"samples_{name}")
        
        # Generate
        try:
            run_inference(model_id, out_path, args.num_samples)
            
            # Compute Metrics
            print(f"Computing FID for {name}...")
            fid_score = metrics.compute_fid(args.real_data_dir, out_path)
            
            results[name] = {
                "model_id": model_id,
                "fid": fid_score
            }
        except Exception as e:
            print(f"Failed to benchmark {name}: {e}")
            results[name] = {"error": str(e)}

    # 2. Evaluate Ours
    print(f"\n--- Benchmarking Finetuned (Ours) ---")
    out_path_ours = os.path.join(args.output_dir, "samples_finetuned")
    run_inference(args.finetuned_path, out_path_ours, args.num_samples)
    
    print(f"Computing FID for Finetuned...")
    fid_ours = metrics.compute_fid(args.real_data_dir, out_path_ours)
    results["Finetuned"] = {
        "model_id": args.finetuned_path,
        "fid": fid_ours
    }
    
    # 3. Save Report
    print("\nFinal Benchmark Report:")
    print(json.dumps(results, indent=2))
    
    with open(os.path.join(args.output_dir, "comparison_report.json"), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
