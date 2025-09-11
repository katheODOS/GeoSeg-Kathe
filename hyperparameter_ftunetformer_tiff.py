import os
import sys
import logging
from pathlib import Path
from itertools import product
import torch
import io
import numpy as np
import albumentations as albu
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.biodiversity_tiff_dataset import *
from geoseg.models.FTUNetFormer import ft_unetformer
from tools.utils import Lookahead
from tools.utils import process_model_params
from contextlib import redirect_stdout
from tqdm import tqdm
import traceback 
from io import StringIO
import atexit
import re
from tools.metric import Evaluator
import heapq

num_classes = 6
max_epoch = 30

LR = [4e-4, 5e-4, 6e-4]
BACKBONE_LR = [4e-5, 5e-5, 6e-5]
BATCH_SIZES = [16]
EPOCHS = [30]
WEIGHT_DECAYS = [1e-2, 1e-1]
BACKBONE_WEIGHT_DECAYS = [1e-2]
SCALE = [1.0]

# Dataset configurations with path mappings (following hyperparameter_tuning.py format)
DATASETS = {
    'biodiversity': {'name': 'Biodiversity Dataset Tiff', 'code': 'biodiversity_tiff', 'path': 'Biodiversity_tiff/Train'},
}

class BestCheckpointTracker:
    """Track the best checkpoints based on validation mIoU"""
    def __init__(self, keep_top_k=2):
        self.keep_top_k = keep_top_k
        self.best_checkpoints = []  # Min heap to keep track of best checkpoints
        
    def update(self, epoch, val_miou, checkpoint_path):
        """Update the best checkpoints list"""
        if len(self.best_checkpoints) < self.keep_top_k:
            heapq.heappush(self.best_checkpoints, (val_miou, epoch, checkpoint_path))
        else:
            # If current score is better than the worst in our list
            if val_miou > self.best_checkpoints[0][0]:
                # Remove the worst checkpoint file
                worst_score, worst_epoch, worst_path = heapq.heappop(self.best_checkpoints)
                if worst_path.exists():
                    worst_path.unlink()
                # Add the new best checkpoint
                heapq.heappush(self.best_checkpoints, (val_miou, epoch, checkpoint_path))
    
    def get_best_checkpoints(self):
        """Return list of best checkpoints sorted by score (descending)"""
        return sorted(self.best_checkpoints, key=lambda x: x[0], reverse=True)

def setup_checkpoint_dir(dataset_code, lr, backbone_lr, wd, backbone_wd, epochs, batch_size, scale):
    """Create and return checkpoint directory for specific configuration"""
    # Following hyperparameter_tuning.py naming convention
    dir_name = f"{dataset_code}L{lr:.0e}BL{backbone_lr:.0e}W{wd:.0e}BW{backbone_wd:.0e}B{batch_size}E{epochs}S{scale:.2f}"
    checkpoint_dir = Path('C:/Users/Admin/anaconda3/envs/GeoSeg-Kathe/model_weights/biodiversity_tiff_ftunetformer') / dir_name
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

def save_run_output(output_lines, checkpoint_dir):
    """Save the run output to output.txt in the specified format"""
    with open(checkpoint_dir / 'output.txt', 'w') as f:
        f.write('\n'.join(output_lines))

class SafeOutputCapture:
    def __init__(self):
        self.output_lines = []
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def add_line(self, line):
        """Add a line to the output"""
        self.output_lines.append(line)
        print(line)  # Also print to console
        
    def get_output_lines(self):
        return self.output_lines
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        pass

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
    
    # Initialize FTUNetFormer model
    model = ft_unetformer(
        num_classes=num_classes,
        decoder_channels=256,
        pretrained=True,
        freeze_stages=-1
    )
    model = model.to(device=device)
    
    # Use loss as defined in the original config
    loss_fn = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=0),
                       DiceLoss(smooth=0.05, ignore_index=0), 1.0, 1.0)
    use_aux_loss = False
    
    # Initialize checkpoint tracker
    best_tracker = BestCheckpointTracker(keep_top_k=2)
    
    with SafeOutputCapture() as output:
        try:
            # Add configuration header
            output.add_line("=" * 80)
            output.add_line("Configuration Details:")
            output.add_line("=" * 80)
            for line in config_details.split('\n'):
                if line.strip():
                    output.add_line(line)
            output.add_line("=" * 80)
            
            # Setup train dataset and loader
            train_dataset = BiodiversityTiffTrainDataset(
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
            val_dataset = biodiversity_tiff_val_dataset
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
            
            # Save configuration
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
            Model: FTUNetFormer
            """
            with open(checkpoint_dir / 'config.txt', 'w') as f:
                f.write(config_log)
            
            # Setup evaluator for metrics
            evaluator = Evaluator(num_class=6)
            
            # Class names for logging
            class_names = ['Background', 'Forest land', 'Grassland', 'Cropland', 'Settlement', 'Seminatural Grassland']
            
            # Training loop implementation
            for epoch in range(epochs):
                model.train()
                total_train_loss = 0
                train_evaluator = Evaluator(num_class=6)
                
                for batch in train_loader:
                    images = batch['img'].to(device)
                    masks = batch['gt_semantic_seg'].to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    
                    # Handle outputs - model returns tuple of (main_out, aux_out)
                    if isinstance(outputs, tuple):
                        main_out = outputs[0]
                    else:
                        main_out = outputs
                        
                    loss = loss_fn(outputs, masks)
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    
                    # Calculate training metrics using main output
                    pred = main_out.data.cpu().numpy()
                    target = masks.cpu().numpy()
                    pred = np.argmax(pred, axis=1)
                    train_evaluator.add_batch(target, pred)
                
                # Calculate average training loss and metrics
                train_loss = total_train_loss / len(train_loader)
                train_iou_scores = train_evaluator.Intersection_over_Union()
                train_f1_scores = train_evaluator.F1()
                train_oa_score = train_evaluator.OA()
                
                # Validation phase
                model.eval()
                total_val_loss = 0
                evaluator.reset()
                
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch['img'].to(device)
                        masks = batch['gt_semantic_seg'].to(device)
                        
                        outputs = model(images)
                        
                        # Handle outputs - model returns tuple of (main_out, aux_out)
                        if isinstance(outputs, tuple):
                            main_out = outputs[0]
                        else:
                            main_out = outputs
                            
                        val_loss = loss_fn(outputs, masks)
                        total_val_loss += val_loss.item()
                        
                        # Get predictions for metrics using main output
                        pred = main_out.data.cpu().numpy()
                        target = masks.cpu().numpy()
                        pred = np.argmax(pred, axis=1)
                        evaluator.add_batch(target, pred)
                
                # Calculate validation metrics
                val_loss = total_val_loss / len(val_loader)
                val_iou_scores = evaluator.Intersection_over_Union()
                val_f1_scores = evaluator.F1()
                val_oa_score = evaluator.OA()
                
                val_miou = np.nanmean(val_iou_scores)
                val_f1 = np.nanmean(val_f1_scores)
                train_miou = np.nanmean(train_iou_scores)
                train_f1 = np.nanmean(train_f1_scores)
                
                # Log validation metrics in the desired format
                output.add_line(f"Epoch: {epoch}")
                output.add_line(f"Val mIoU: {val_miou:.4f}")
                output.add_line(f"Val F1: {val_f1:.4f}")
                output.add_line(f"Val OA: {val_oa_score:.4f}")
                output.add_line("Per-class IoU:")
                for name, iou in zip(class_names, val_iou_scores):
                    if np.isnan(iou):
                        output.add_line(f"'{name}': nan")
                    else:
                        output.add_line(f"'{name}': {iou:.4f}")
                
                # Log training metrics in the desired format
                output.add_line(f"Epoch: {epoch}")
                output.add_line(f"Train mIoU: {train_miou:.4f}" if not np.isnan(train_miou) else "Train mIoU: nan")
                output.add_line(f"Train F1: {train_f1:.4f}" if not np.isnan(train_f1) else "Train F1: nan")
                output.add_line(f"Train OA: {train_oa_score:.4f}")
                output.add_line("Per-class IoU:")
                for name, iou in zip(class_names, train_iou_scores):
                    if np.isnan(iou):
                        output.add_line(f"'{name}': nan")
                    else:
                        output.add_line(f"'{name}': {iou:.4f}")
                
                # Create checkpoint data
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_miou': val_miou,
                    'val_f1': val_f1,
                    'val_oa': val_oa_score,
                    'config': {
                        'lr': lr,
                        'backbone_lr': backbone_lr,
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'weight_decay': weight_decay,
                        'backbone_weight_decay': backbone_weight_decay,
                        'scale': scale
                    }
                }
                
                # Save temporary checkpoint for tracking best ones
                temp_checkpoint_path = checkpoint_dir / f'temp_epoch{epoch + 1:02d}.ckpt'
                torch.save(checkpoint_data, temp_checkpoint_path)
                
                # Update best checkpoint tracker
                best_tracker.update(epoch + 1, val_miou, temp_checkpoint_path)
                
                # Always save the last checkpoint
                torch.save(checkpoint_data, checkpoint_dir / 'last.ckpt')
                
                # Update learning rate
                lr_scheduler.step()
            
            # After training, rename the best checkpoints to meaningful names
            best_checkpoints = best_tracker.get_best_checkpoints()
            for i, (score, epoch, temp_path) in enumerate(best_checkpoints):
                if temp_path.exists():
                    new_name = f'{i+1}_epoch{epoch:02d}_miou{score:.4f}.ckpt'
                    new_path = checkpoint_dir / new_name
                    temp_path.rename(new_path)
            
            # Clean up any remaining temporary checkpoints
            for temp_file in checkpoint_dir.glob('temp_epoch*.ckpt'):
                if temp_file.exists():
                    temp_file.unlink()
            
        except Exception as e:
            output.add_line(f"Training failed with error: {str(e)}")
            output.add_line("\nFull traceback:")
            output.add_line(traceback.format_exc())
        finally:
            cleanup_wandb()
            
        return output.get_output_lines()

def is_training_completed(checkpoint_dir, epochs):
    """Check if training was already completed for this configuration"""
    last_checkpoint = checkpoint_dir / 'last.ckpt'
    return last_checkpoint.exists()

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
                output_lines = run_training_configuration(
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
                save_run_output(output_lines, checkpoint_dir)
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