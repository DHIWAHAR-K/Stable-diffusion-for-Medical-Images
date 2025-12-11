import argparse
import os
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from PIL import Image
from device import get_device

def parse_args():
    parser = argparse.ArgumentParser(description="Bulk Inference for Research Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model pipeline or 'runwayml/stable-diffusion-v1-5' for baseline")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")
    parser.add_argument("--num_samples", type=int, default=1000, help="Total number of images to generate")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompts", type=str, nargs='+', default=["mammogram of density High_Density", "mammogram of density Low_Density"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = get_device()
    print(f"Using device: {device}")
    
    print(f"Loading model from {args.model_path}...")
    pipeline = StableDiffusionPipeline.from_pretrained(args.model_path, safety_checker=None)
    pipeline.to(device)
    
    # Disable progress bar for pipeline, use our own
    pipeline.set_progress_bar_config(disable=True)
    
    total_generated = 0
    pbar = tqdm(total=args.num_samples)
    
    torch.manual_seed(args.seed)
    
    while total_generated < args.num_samples:
        curr_batch = min(args.batch_size, args.num_samples - total_generated)
        
        # Interleave prompts
        batch_prompts = [args.prompts[i % len(args.prompts)] for i in range(total_generated, total_generated + curr_batch)]
        
        with torch.autocast(device): # Attempt autocast
            images = pipeline(batch_prompts, num_inference_steps=50).images
            
        for i, img in enumerate(images):
            idx = total_generated + i
            prompt_tag = "high" if "High_Density" in batch_prompts[i] else "low"
            save_path = os.path.join(args.output_dir, f"{idx:05d}_{prompt_tag}.png")
            img.save(save_path)
            
        total_generated += curr_batch
        pbar.update(curr_batch)
        
    pbar.close()
    print(f"Successfully generated {total_generated} images in {args.output_dir}")

if __name__ == "__main__":
    main()
