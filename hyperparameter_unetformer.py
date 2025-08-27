import os
import sys
import logging
from pathlib import Path
from itertools import product
import torch
import io
import numpy as np
import albumentations as albu  # Added missing import
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.biodiversity_dataset import *
from geoseg.models.UNetFormer import UNetFormer
from tools.utils import Lookahead
from tools.utils import process_model_params
from contextlib import redirect_stdout
from tqdm import tqdm
import traceback 
from io import StringIO
import atexit
import re
from tools.metric import Evaluator
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

num_classes = 6
max_epoch = 30

# Hyperparameter configurations
LR = [5e-4, 6e-4, 2e-4]  # Base learning rates: default 6e-4
BACKBONE_LR = [5e-5, 1e-4]  # Backbone learning rates default 6e-5
BATCH_SIZES = [16]
EPOCHS = [50, 75]
WEIGHT_DECAYS = [1e-2]
BACKBONE_WEIGHT_DECAYS = [1e-2]
SCALE = [1.0]

# Dataset configurations with path mappings (following hyperparameter_tuning.py format)
DATASETS = {
    'biodiversity': {'name': 'Biodiversity Dataset', 'code': 'biodiversity', 'path': 'Biodiversity/Train'},
}

# Add these class names after the hyperparameter configurations
CLASS_NAMES = ['Background', 'Forest land', 'Grassland', 'Cropland', 'Settlement', 'Seminatural Grassland']

def setup_checkpoint_dir(dataset_code, lr, backbone_lr, wd, backbone_wd, epochs, batch_size, scale):
    """Create and return checkpoint directory for specific configuration"""
    # Following hyperparameter_tuning.py naming convention
    dir_name = f"{dataset_code}L{lr:.0e}BL{backbone_lr:.0e}W{wd:.0e}BW{backbone_wd:.0e}B{batch_size}E{epochs}S{scale:.2f}"
    checkpoint_dir = Path('C:/Users/Admin/anaconda3/envs/GeoSeg-Kathe/model_weights/biodiversity') / dir_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir

def get_removal_list():
    """Read folder names from remove_checkpoints.txt that should be skipped"""
    removal_list_path = Path('./remove_checkpoints.txt')
    if not removal_list_path.exists():
        logging.warning("remove_checkpoints.txt not found, no configurations will be skipped")
        return set()
    
    try:
        with open(removal_list_path, 'r') as f:
            return {line.strip() for line in f if line.strip()}
    except Exception as e:
        logging.error(f"Error reading removal list: {e}")
        return set()

def save_run_output(output_text, checkpoint_dir):
    """Save the run output to output.txt"""
    with open(checkpoint_dir / 'output.txt', 'w') as f:
        f.write(output_text)

class SafeOutputCapture:
    def __init__(self):
        self.buffer = StringIO()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        # Add logging capture
        self.log_handler = logging.StreamHandler(self.buffer)
        self.log_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        
    def write(self, text):
        self.buffer.write(text)
        self.original_stdout.write(text)
        
    def flush(self):
        self.buffer.flush()
        self.original_stdout.flush()
        
    def get_output(self):
        return self.buffer.getvalue()
    
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        # Add logging handler when entering context
        logging.getLogger().addHandler(self.log_handler)
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        # Remove logging handler when exiting context
        logging.getLogger().removeHandler(self.log_handler)
        self.buffer.close()

def extract_latest_validation_score(output_text):
    """Extract the most recent validation score from the output"""
    matches = re.findall(r'INFO: Validation Dice score: (\d+\.\d+)', output_text)
    return float(matches[-1]) if matches else 0.0

def cleanup_wandb():
    """This can be removed entirely"""
    pass

def run_training_configuration(dataset_path, checkpoint_dir, lr, backbone_lr, batch_size, epochs, 
                             weight_decay, backbone_weight_decay, scale, config_details):
    """Run training with specific configuration and capture output"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = UNetFormer(num_classes=6)  # Fixed: using consistent num_classes=6
    model = model.to(device=device)
    
    with SafeOutputCapture() as output:
        try:
            sys.stdout.write("="*80 + "\n")
            sys.stdout.write(f"Configuration Details:\n")
            sys.stdout.write("="*80 + "\n")
            sys.stdout.write(config_details + "\n")
            sys.stdout.write("="*80 + "\n\n")
            
            # Setup train dataset and loader
            train_dataset = BiodiversityTrainDataset(
                transform=train_aug,
                data_root=dataset_path
            )
            
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=True,
                shuffle=True,
                drop_last=True
            )
            
            # Setup validation dataset and loader
            val_dataset = biodiversity_val_dataset
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=batch_size,  # Using same batch size as training
                num_workers=0,
                shuffle=False,
                pin_memory=True,
                drop_last=False
            )
            
            # Setup optimizer and scheduler - FIXED: use 'model' instead of 'net'
            layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
            net_params = process_model_params(model, layerwise_params=layerwise_params)  # Changed from 'net' to 'model'
            base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
            optimizer = Lookahead(base_optimizer)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)  # Changed from max_epoch to epochs
            
            # Setup loss
            loss_fn = UnetFormerLoss(ignore_index=0)
            
            # Save detailed configuration at start of training
            config_log = f"""
            Dataset: {dataset_path}
            Learning Rate: {lr}
            Backbone Learning Rate: {backbone_lr}
            Batch Size: {batch_size}
            Epochs: {epochs}
            Weight Decay: {weight_decay}
            Backbone Weight Decay: {backbone_weight_decay}
            Scale: {scale}
            Checkpoint Directory: {checkpoint_dir}
            """
            with open(checkpoint_dir / 'config.txt', 'w') as f:
                f.write(config_log)
            
            # Setup evaluator for metrics
            evaluator = Evaluator(num_class=6)
            
            # Setup checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='checkpoint_epoch{epoch:02d}',
                save_top_k=3,  # Save the best 3 models
                monitor='val_loss',
                mode='min',
                save_last=True
            )
            
            # Setup logger
            logger = CSVLogger(str(checkpoint_dir), name='training_logs')
            
            # Create a LightningModule wrapper for the model
            class LitModel(pl.LightningModule):
                def __init__(self, model, loss_fn, optimizer, scheduler):
                    super().__init__()
                    self.model = model
                    self.loss_fn = loss_fn
                    self.optimizer = optimizer
                    self.scheduler = scheduler
                    self.evaluator = Evaluator(num_class=6)
                    self.class_names = CLASS_NAMES  # Add class names
                
                def forward(self, x):
                    return self.model(x)
                
                def training_step(self, batch, batch_idx):
                    images, masks = batch['img'], batch['gt_semantic_seg']
                    outputs = self(images)
                    loss = self.loss_fn(outputs, masks)
                    self.log('train_loss', loss)
                    return {"loss": loss}
                
                def validation_step(self, batch, batch_idx):
                    images, masks = batch['img'], batch['gt_semantic_seg']
                    outputs = self(images)
                    loss = self.loss_fn(outputs, masks)
                    self.log('val_loss', loss)
                    
                    # Calculate metrics
                    pred = outputs.data.cpu().numpy()
                    target = masks.cpu().numpy()
                    pred = np.argmax(pred, axis=1)
                    self.evaluator.add_batch(target, pred)
                    return loss
                
                def on_validation_epoch_end(self):
                    iou_per_class = self.evaluator.Intersection_over_Union()
                    f1_per_class = self.evaluator.F1()
                    
                    scores = {
                        'mIoU': np.nanmean(iou_per_class),
                        'F1': np.nanmean(f1_per_class),
                        'OA': self.evaluator.OA()
                    }
                    
                    # Log overall metrics
                    self.log_dict(scores)
                    
                    # Create per-class metrics dictionary
                    class_metrics = {}
                    for name, iou, f1 in zip(self.class_names, iou_per_class, f1_per_class):
                        class_metrics[f'{name}_IoU'] = iou
                        class_metrics[f'{name}_F1'] = f1
                    
                    # Log per-class metrics
                    self.log_dict(class_metrics)
                    
                    # Format detailed logging string
                    class_metrics_str = "\n".join([
                        f"'{name}': {iou:.4f}" 
                        for name, iou in zip(self.class_names, iou_per_class)
                    ])
                    
                    print(f"""
Epoch: {self.current_epoch}
Val mIoU: {scores['mIoU']:.4f}
Val F1: {scores['F1']:.4f}
Val OA: {scores['OA']:.4f}
Per-class IoU:
{class_metrics_str}
""")
                    
                    self.evaluator.reset()

                def on_train_epoch_end(self):
                    iou_per_class = self.evaluator.Intersection_over_Union()
                    f1_per_class = self.evaluator.F1()
                    
                    scores = {
                        'mIoU': np.nanmean(iou_per_class),
                        'F1': np.nanmean(f1_per_class),
                        'OA': self.evaluator.OA()
                    }
                    
                    # Log overall metrics
                    self.log_dict({'train_' + k: v for k, v in scores.items()})
                    
                    # Format class-specific metrics
                    class_metrics_str = "\n".join([
                        f"'{name}': {iou:.4f}" 
                        for name, iou in zip(self.class_names, iou_per_class)
                    ])
                    
                    print(f"""
Epoch: {self.current_epoch}
Train mIoU: {scores['mIoU']:.4f}
Train F1: {scores['F1']:.4f}
Train OA: {scores['OA']:.4f}
Per-class IoU:
{class_metrics_str}
""")
                    
                    self.evaluator.reset()
                
                def configure_optimizers(self):
                    return {
                        "optimizer": self.optimizer,
                        "lr_scheduler": {
                            "scheduler": self.scheduler,
                            "interval": "epoch",
                        },
                    }
            
            # Create LightningModule instance
            lit_model = LitModel(model, loss_fn, optimizer, lr_scheduler)
            
            # Setup trainer with disabled progress bar
            trainer = pl.Trainer(
                max_epochs=epochs,
                callbacks=[checkpoint_callback],
                logger=logger,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                enable_progress_bar=False  # Disable progress bar
            )
            
            # Train the model
            trainer.fit(
                lit_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader
            )
            
        except Exception as e:
            print(f"Training failed with error: {str(e)}")
            print("\nFull traceback:")
            print(traceback.format_exc())
        finally:
            cleanup_wandb()
            
        return output.get_output()

def is_training_completed(checkpoint_dir, epochs):
    """Check if training was already completed for this configuration"""
    # Check for PyTorch Lightning checkpoint files
    final_checkpoint = checkpoint_dir / 'last.ckpt'
    return final_checkpoint.exists()

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Load the removal list
    removal_list = get_removal_list()
    if removal_list:
        logging.info(f"Loaded {len(removal_list)} configurations to skip")
    
    # Create all possible combinations of hyperparameters
    configs = list(product(
        DATASETS.items(),
        LR,
        BACKBONE_LR,
        BATCH_SIZES,
        EPOCHS,
        WEIGHT_DECAYS,
        BACKBONE_WEIGHT_DECAYS,
        SCALE
    ))
    total_combinations = len(configs)
    logging.info(f"Total number of combinations to try: {total_combinations}")
    
    try:
        # Register global cleanup
        atexit.register(cleanup_wandb)
        
        for idx, ((dataset_key, dataset_info), lr, backbone_lr, batch_size, epochs, weight_decay, backbone_weight_decay, scale) in enumerate(configs, 1):
            logging.info(f"\n{'='*80}")
            logging.info(f"Running combination {idx}/{total_combinations}")
            logging.info(f"{'='*80}")
            
            # Setup directories for current configuration
            checkpoint_dir = setup_checkpoint_dir(
                dataset_info['code'], lr, backbone_lr, weight_decay, 
                backbone_weight_decay, epochs, batch_size, scale
            )
            
            # Check if this configuration should be skipped based on folder name
            if checkpoint_dir.name in removal_list:
                logging.info(f"Skipping configuration {checkpoint_dir.name} as it's in the removal list")
                continue
            
            # Check if this combination was already completed
            if is_training_completed(checkpoint_dir, epochs):
                logging.info(f"Training already completed for this configuration. Skipping...")
                continue
            
            logging.info(f"""
            Configuration details:
            Dataset: {dataset_info['name']} ({dataset_key})
            Learning Rate: {lr}
            Backbone Learning Rate: {backbone_lr}
            Batch Size: {batch_size}
            Epochs: {epochs}
            Weight Decay: {weight_decay}
            Backbone Weight Decay: {backbone_weight_decay}
            Checkpoint Directory: {checkpoint_dir}
            """)
            
            # Update data directories for current dataset using the path field
            dataset_base = f'./data/{dataset_info["path"]}'
            
            if not Path(dataset_base).exists():
                logging.error(f"Dataset directory {dataset_base} not found! Skipping this combination.")
                continue
                
            logging.info("Starting training for this combination...")
            
            # Format configuration details
            config_details = f"""Dataset: {dataset_info['name']} ({dataset_key})
Learning Rate: {lr}
Backbone Learning Rate: {backbone_lr}
Batch Size: {batch_size}
Epochs: {epochs}
Weight Decay: {weight_decay}
Backbone Weight Decay: {backbone_weight_decay}
Scale: {scale}
Checkpoint Directory: {checkpoint_dir}"""
            
            # Run training with config details
            try:
                output = run_training_configuration(
                    dataset_base,
                    checkpoint_dir,
                    lr,
                    backbone_lr,
                    batch_size,
                    epochs,
                    weight_decay,
                    backbone_weight_decay,
                    scale,  # Add scale parameter here
                    config_details
                )
                
                # Save output
                save_run_output(output, checkpoint_dir)
                logging.info(f"Training completed and saved to {checkpoint_dir}")
                
            except Exception as e:
                logging.error(f"Error during training: {str(e)}")
                cleanup_wandb()  # Ensure wandb is cleaned up after error
                continue
            
            # Clean up (removed wandb.finish() since it's handled by cleanup_wandb)
            torch.cuda.empty_cache()
            
            logging.info(f"Completed combination {idx}/{total_combinations}")
    finally:
        # Final cleanup
        cleanup_wandb()
        # Deregister cleanup function
        atexit.unregister(cleanup_wandb)

if __name__ == '__main__':
    main()