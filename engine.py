import os
import math
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from datetime import datetime

from models import load_models
from optimization import get_dataloader
from logger import MetricLogger
from visualization import plot_training_curves, save_sample_grid
from checkpoint import CheckpointManager

class Trainer:
    def __init__(self, config):
        self.config = config
        self.model_config = config['model']
        self.training_config = config['training']
        self.data_config = config['data']
        self.output_config = config['output']
        
        # Setup directories (Split Top-Level Structure)
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H-%M-%S")
        exp_name = config.get('experiment_name', 'default')
        
        # Consistent path suffix for all folers
        run_path = os.path.join(date_str, time_str, exp_name)
        
        # Create separate top-level folders in the main directory
        self.results_dir = os.path.join("results", run_path)
        self.models_dir = os.path.join("models", run_path)
        self.samples_dir = os.path.join("samples", run_path)
        self.graphs_dir = os.path.join("graphs", run_path)
        
        for d in [self.models_dir, self.samples_dir, self.graphs_dir, self.results_dir]:
            os.makedirs(d, exist_ok=True)

        # Accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            mixed_precision=self.training_config['mixed_precision'],
            log_with="wandb"
        )
        
        set_seed(self.training_config['seed'])
        
        # Logging
        self.logger = MetricLogger(self.results_dir, project_name=config.get('project_name'), config=config)
        self.checkpoint_manager = CheckpointManager(self.models_dir, self.accelerator)

        # Load Models
        self.noise_scheduler, self.tokenizer, self.text_encoder, self.vae, self.unet = load_models(
            self.model_config['pretrained_model_name_or_path']
        )
        
        # Move frozen models
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
            
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        
        # Enable gradient checkpointing
        self.unet.enable_gradient_checkpointing()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.training_config['learning_rate'],
            betas=(config['optimization']['adam_beta1'], config['optimization']['adam_beta2']),
            weight_decay=config['optimization']['adam_weight_decay'],
            eps=config['optimization']['adam_epsilon'],
        )
        
        # Dataset (Standard Loader)
        self.train_dataloader = get_dataloader(
            self.data_config['data_dir'], 
            self.tokenizer, 
            batch_size=self.data_config['train_batch_size'],
            resolution=self.data_config['resolution'],
            num_workers=self.data_config['dataloader_num_workers']
        )
        # self.train_dataset = ... (embedded in dataloader now)
        
        # Scheduler
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.training_config['gradient_accumulation_steps'])
        self.max_train_steps = self.training_config.get('max_train_steps')
        if not self.max_train_steps:
            self.max_train_steps = self.training_config['num_train_epochs'] * num_update_steps_per_epoch
            
        self.lr_scheduler = get_scheduler(
            self.training_config['lr_scheduler'],
            optimizer=self.optimizer,
            num_warmup_steps=self.training_config['lr_warmup_steps'] * self.training_config['gradient_accumulation_steps'],
            num_training_steps=self.max_train_steps,
        )

        # Prepare
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

    def train(self):
        if self.accelerator.is_main_process:
            self.logger.init_wandb()
            print(f"Training Started. Outputs will be saved to separate folders (results, models, samples, graphs) with suffix: {os.path.join(datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%H-%M-%S'))}")

        global_step = 0
        best_loss = float('inf')
        
        self.unet.train()
        
        progress_bar = tqdm(range(self.max_train_steps), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(self.training_config['num_train_epochs']):
            epoch_loss = 0.0
            grad_norm = None
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # 1. Get Latents (On-the-fly Encoding)
                    pixel_values = batch["pixel_values"].to(self.accelerator.device, dtype=self.weight_dtype)
                    
                    with torch.no_grad():
                        latents = self.vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * self.vae.config.scaling_factor

                    # Noise
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

                    # Add Noise
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Text Embeddings
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                    # Predict
                    model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Loss
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError("Unknown prediction type")

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.unet.parameters(), self.training_config['max_grad_norm'])
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # Logging
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    epoch_loss += loss.item()
                    
                    logs = {
                        "loss": loss.item(), 
                        "lr": self.lr_scheduler.get_last_lr()[0], 
                        "grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
                        "step": global_step, 
                        "epoch": epoch
                    }
                    progress_bar.set_postfix(**logs)
                    
                    if self.accelerator.is_main_process:
                         self.logger.log(logs)

                if global_step >= self.max_train_steps:
                    break
            
            # Epoch End
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            
            if self.accelerator.is_main_process:
                # Validation / Sampling
                if epoch % self.training_config['validation_epochs'] == 0:
                    self.generate_validation_samples(epoch)
                    
                # Save Checkpoint
                is_best = avg_epoch_loss < best_loss
                if is_best:
                    best_loss = avg_epoch_loss
                
                self.checkpoint_manager.save(is_best, epoch=epoch)
                
                # Plot
                plot_training_curves(self.logger.csv_file, self.graphs_dir)

        self.logger.finish()
        
        # Save final model as a pipeline
        if self.accelerator.is_main_process:
            self.save_pipeline()

    def generate_validation_samples(self, epoch):
        print(f"Generating samples for epoch {epoch}...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_config['pretrained_model_name_or_path'],
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.accelerator.unwrap_model(self.unet),
            tokenizer=self.tokenizer,
            scheduler=self.noise_scheduler,
            safety_checker=None,
            feature_extractor=None
        )
        pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        
        prompts = ["mammogram of density High_Density", "mammogram of density Low_Density"] * 2
        images = []
        for prompt in prompts:
            # Safer to use no_grad and let accelerator handle device/dtype if not explicit autocast
            with torch.no_grad(): 
                 img = pipeline(prompt, num_inference_steps=20).images[0]
                 images.append(img)
        
        save_path = os.path.join(self.samples_dir, f"epoch_{epoch}.png")
        save_sample_grid(images, save_path, rows=2)
        
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_pipeline(self):
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_config['pretrained_model_name_or_path'],
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.accelerator.unwrap_model(self.unet),
            tokenizer=self.tokenizer,
            scheduler=self.noise_scheduler,
        )
        # Save to 'pipeline' inside the models directory
        pipeline.save_pretrained(os.path.join(self.models_dir, "pipeline"))
