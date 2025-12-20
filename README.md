# Stable Diffusion for Medical Images

Fine-tuning Stable Diffusion for medical image generation using RSNA and VinDr mammography datasets with memory-efficient attention mechanisms and optimized training pipeline.

## Overview

This project implements a fine-tuning pipeline for Stable Diffusion v1.5 specifically adapted for medical imaging applications. The implementation focuses on generating high-quality mammography images while incorporating memory-efficient training optimizations and robust data preprocessing.

### Key Features

- **Medical Dataset Support**: Integrated preprocessing for RSNA and VinDr mammography datasets
- **Memory-Efficient Training**: Optimized attention mechanisms and gradient accumulation for limited GPU memory
- **Flexible Configuration**: YAML-based configuration system for easy experimentation
- **Comprehensive Monitoring**: Built-in logging, checkpointing, and validation sampling
- **Production-Ready**: Includes data augmentation, early stopping, and mixed precision training

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- PyTorch 2.0+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DHIWAHAR-K/Stable-diffusion-for-Medical-Images.git
cd Stable-diffusion-for-Medical-Images
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Directory Structure

Organize your data in the following structure:
```
data/vindr_rsna/
├── train/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── val/
    ├── image_001.png
    └── ...
```

### Preprocessing Scripts

The repository includes several preprocessing utilities:

- `preprocess_data.py`: General preprocessing for medical images
- `preprocess_rsna.py`: RSNA-specific preprocessing
- `merge_datasets.py`: Merge multiple datasets
- `create_splits.py`: Create train/validation splits
- `organize_images.py`: Organize images into proper directory structure

Example usage:
```bash
python preprocess_data.py --input_dir /path/to/raw/data --output_dir data/vindr_rsna
python create_splits.py --data_dir data/vindr_rsna --train_ratio 0.8
```

## Training

### Basic Training

Start training with default configuration:
```bash
python train.py --config configs/train_params.yaml
```

### Configuration

Edit `configs/train_params.yaml` to customize training parameters:

```yaml
model:
  pretrained_model_name_or_path: "runwayml/stable-diffusion-v1-5"

data:
  data_dir: "data/vindr_rsna"
  resolution: 512
  train_batch_size: 16
  dataloader_num_workers: 8

training:
  num_train_epochs: 50
  learning_rate: 1.0e-5
  mixed_precision: "fp16"  # Use fp16 for memory efficiency
  gradient_accumulation_steps: 2
  checkpointing_steps: 500
  validation_epochs: 5
```

### Key Training Parameters

- **learning_rate**: Controls the step size during optimization (default: 1e-5)
- **train_batch_size**: Number of samples per batch (adjust based on GPU memory)
- **gradient_accumulation_steps**: Accumulate gradients over multiple batches
- **mixed_precision**: Use "fp16" or "bf16" to reduce memory usage
- **checkpointing_steps**: Save model checkpoint every N steps

### Memory Optimization

For GPUs with limited memory, adjust these parameters:

```yaml
data:
  train_batch_size: 4  # Reduce batch size
  
training:
  gradient_accumulation_steps: 8  # Increase accumulation
  mixed_precision: "fp16"  # Enable mixed precision
```

## Inference

Generate images using the trained model:

```bash
python inference.py \
  --model_path ./checkpoints/checkpoint-5000 \
  --prompt "mammography image showing normal breast tissue" \
  --num_images 4 \
  --output_dir ./generated_images
```

## Project Structure

```
├── configs/
│   └── train_params.yaml    # Training configuration
├── train.py                 # Main training script
├── inference.py             # Image generation script
├── engine.py                # Training engine and loop
├── models.py                # Model loading utilities
├── data.py                  # Dataset and dataloader
├── attention.py             # Memory-efficient attention
├── optimization.py          # Optimizer and scheduler
├── augmentation.py          # Data augmentation
├── checkpoint.py            # Checkpoint management
├── logger.py                # Logging utilities
├── metrics.py               # Evaluation metrics
└── visualization.py         # Visualization tools
```

## Monitoring Training

Training logs are saved to the `logs/` directory. Monitor progress with:

```bash
tail -f logs/training.log
```

Checkpoints are saved to the output directory specified in the config.

## Advanced Features

### Custom Prompts

Modify `prompts.py` to define custom prompt templates for your medical imaging domain.

### Latent Analysis

Use `latent_analyzer.py` to analyze the latent space representation of medical images.

### Benchmarking

Run `benchmark.py` to evaluate model performance and generation quality metrics.

## Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce `train_batch_size` in config
2. Increase `gradient_accumulation_steps`
3. Enable mixed precision training (`mixed_precision: "fp16"`)
4. Reduce `resolution` to 256 or 384

### Slow Training

1. Increase `dataloader_num_workers`
2. Use faster storage (SSD/NVMe)
3. Enable mixed precision training
4. Increase batch size if memory allows

## Citation

If you use this code in your research, please cite:

```bibtex
@software{stable_diffusion_medical,
  author = {DHIWAHAR-K},
  title = {Stable Diffusion for Medical Images},
  year = {2025},
  url = {https://github.com/DHIWAHAR-K/Stable-diffusion-for-Medical-Images}
}
```

## License

This project is provided for research and educational purposes.

## Acknowledgments

- Stable Diffusion by Stability AI and RunwayML
- RSNA Breast Cancer Detection dataset
- VinDr Mammography dataset
- HuggingFace Diffusers library
