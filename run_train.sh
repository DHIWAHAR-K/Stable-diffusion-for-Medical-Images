#!/bin/bash

# Initialize conda for shell interaction
eval "$(conda shell.zsh hook)"

# Activate environment
conda activate stable_diffusion

# Run training
# Adjust max_train_steps as needed.
echo "Starting training on MPS..."
python3 train.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --data_dir="data/processed" \
  --output_dir="sd-vindr-model" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --max_train_steps=2000 \
  --mixed_precision="fp16" 

echo "Training complete. Model saved to sd-vindr-model/"
