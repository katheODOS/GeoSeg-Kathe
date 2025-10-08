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
import heapq
import json

num_classes = 6
max_epoch = 30

LR = [4e-4, 5e-4]
BACKBONE_LR = [4e-5, 5e-5, 6e-5]
BATCH_SIZES = [16]
EPOCHS = [50]
WEIGHT_DECAYS = [5e-2, 1e-1]
BACKBONE_WEIGHT_DECAYS = [1e-2]
SCALE = [1.0]

# Dataset configurations with path mappings
DATASETS = {
    'biodiversity': {'name': 'Biodiversity Dataset Tiff', 'code': 'biodiversity_tiff', 'path': 'Biodiversity_tiff/Train'},
}

# ============================================================================
# NEW FUNCTIONS FOR RESUME CAPABILITY
# ============================================================================

def save_training_progress(checkpoint_dir, epoch, config_name):
    """Save the current training progress"""
    progress_file = checkpoint_dir / 'training_progress.json'
    progress_data = {
        'last_completed_epoch': epoch,
        'config_name': config_name,
        'status': 'in_progress'
    }
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def mark_training_complete(checkpoint_dir):
    """Mark training as completed"""
    progress_file = checkpoint_dir / 'training_progress.json'
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        progress_data['status'] = 'completed'
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)

def get_training_progress(checkpoint_dir):
    """Get the last completed epoch for a configuration
    
    Returns:
        tuple: (last_completed_epoch, is_completed)
               last_completed_epoch is -1 if training hasn't started
               is_completed is True if training finished all epochs
    """
    progress_file = checkpoint_dir / 'training_progress.json'
    
    if not progress_file.exists():
        return -1, False
    
    try:
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        
        last_epoch = progress_data.get('last_completed_epoch', -1)
        is_completed = progress_data.get('status', 'in_progress') == 'completed'
        
        return last_epoch, is_completed
    except Exception as e:
        logging.error(f"Error reading progress file: {e}")
        return -1, False

def load_checkpoint_for_resume(checkpoint_dir, epoch, model, optimizer, lr_scheduler):
    """Load checkpoint to resume training from a specific epoch"""
    checkpoint_path = checkpoint_dir / 'last.ckpt'
    
    if not checkpoint_path.exists():
        logging.warning(f"Checkpoint not found at {checkpoint_path}")
        return False
    
    try:
        checkpoint = torch.load(checkpoint_path)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logging.info(f"Successfully loaded checkpoint from epoch {epoch}")
        return True
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        return False

# ============================================================================
# EXISTING FUNCTIONS
# ============================================================================

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
    dir_name = f"{dataset_code}L{lr:.0e}BL{backbone_lr:.0e}W{wd:.0e}BW{backbone_wd:.0e}B{batch_size}E{epochs}S{scale:.2f}"
    checkpoint_dir = Path(r'C:\Users\Admin\anaconda3\envs\GeoSeg-Kathe\model_weights\biodiversity_tiff_ftunetformer_new') / dir_name
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

# ============================================================================
# MODIFIED run_training_configuration WITH RESUME CAPABILITY
# ============================================================================

def run_training_configuration(dataset_path, checkpoint_dir, lr, backbone_lr, batch_size, epochs, 
                             weight_decay, backbone_weight_decay, scale, config_details):
    """Run training with specific configuration and capture output - WITH RESUME CAPABILITY"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check for existing progress
    last_completed_epoch, is_completed = get_training_progress(checkpoint_dir)
    
    if is_completed:
        logging.info(f"Training already completed for this configuration. Skipping...")
        # Load and return the existing output
        output_file = checkpoint_dir / 'output.txt'
        if output_file.exists():
            with open(output_file, 'r') as f:
                return f.read().split('\n')
        return []
    
    # Determine starting epoch
    start_epoch = last_completed_epoch + 1
    
    if start_epoch > 0:
        logging.info(f"RESUMING training from epoch {start_epoch} (last completed: {last_completed_epoch})")
    else:
        logging.info(f"STARTING training from scratch")
    
    # Initialize UNetFormer model
    model = UNetFormer(
        num_classes=num_classes,
        decode_channels=64,
        dropout=0.1,
        backbone_name='swsl_resnet18',
        pretrained=True,
        in_channels=4
    )
    model = model.to(device=device)
    
    # Use UnetFormerLoss as defined in the original config
    loss_fn = UnetFormerLoss(ignore_index=0)
    use_aux_loss = True
    
    # Initialize checkpoint tracker
    best_tracker = BestCheckpointTracker(keep_top_k=2)
    
    with SafeOutputCapture() as output:
        try:
            # Add configuration header (only if starting fresh)
            if start_epoch == 0:
                output.add_line("=" * 80)
                output.add_line("Configuration Details:")
                output.add_line("=" * 80)
                for line in config_details.split('\n'):
                    if line.strip():
                        output.add_line(line)
                output.add_line("=" * 80)
            else:
                # Load previous output
                output_file = checkpoint_dir / 'output.txt'
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        for line in f:
                            output.add_line(line.rstrip())
                output.add_line("=" * 80)
                output.add_line(f"RESUMING TRAINING FROM EPOCH {start_epoch}")
                output.add_line(f"   Last completed epoch: {last_completed_epoch}")
                output.add_line("=" * 80)
            
            # Parse the config_details to get dataset info
            dataset_info = {}
            for line in config_details.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    dataset_info[key.strip()] = value.strip()
            
            # Ensure data paths are correct
            data_root = Path('C:/Users/Admin/anaconda3/envs/GeoSeg-Kathe/data')
            
            # Get dataset path information from DATASETS dictionary
            for dataset_key, info in DATASETS.items():
                if dataset_key in config_details:
                    train_path = data_root / info['path']
                    # Try to use separate validation path
                    val_path = data_root / 'Biodiversity_tiff/Val'
                    break
            else:
                train_path = data_root / 'Biodiversity_tiff/Train'  # Default path
                val_path = data_root / 'Biodiversity_tiff/Val'      # Default val path
            
            # Validate paths exist
            if not train_path.exists():
                output.add_line(f"ERROR: Train directory not found: {train_path}")
                return output.get_output_lines()
                
            if not val_path.exists():
                logging.warning(f"Val directory not found: {val_path}")
                logging.warning("Will use Train dataset for validation")
                val_path = train_path
            else:
                logging.info(f"Using validation data from: {val_path}")
            
            # Setup train dataset and loader
            train_dataset = BiodiversityTiffTrainDataset(
                data_root=str(train_path),
                transform=train_aug,
                mosaic_ratio=0.25
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
            val_dataset = BiodiversityTiffTrainDataset(
                data_root=str(val_path),
                transform=val_aug,
                mosaic_ratio=0.0  # No mosaic for validation
            )
            
            if len(train_dataset) == 0:
                output.add_line(f"ERROR: Training dataset is empty!")
                return output.get_output_lines()
            
            if len(val_dataset) == 0:
                output.add_line(f"ERROR: Validation dataset is empty!")
                return output.get_output_lines()
            
            logging.info(f"Training with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=batch_size,
                num_workers=0,
                shuffle=False,
                pin_memory=True,
                drop_last=False
            )
            
            # Setup optimizer and scheduler
            layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
            net_params = process_model_params(model, layerwise_params=layerwise_params)
            base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
            optimizer = Lookahead(base_optimizer)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            
            # Load checkpoint if resuming
            if start_epoch > 0:
                if load_checkpoint_for_resume(checkpoint_dir, last_completed_epoch, 
                                             model, optimizer, lr_scheduler):
                    output.add_line(f"[OK] Successfully loaded checkpoint from epoch {last_completed_epoch}")
                else:
                    output.add_line(f"[WARNING] Could not load checkpoint, starting from scratch")
                    start_epoch = 0
            
            # Save configuration (only if starting fresh)
            if start_epoch == 0:
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
                Model: UNetFormer
                """
                with open(checkpoint_dir / 'config.txt', 'w') as f:
                    f.write(config_log)
            
            # Setup evaluator for metrics
            evaluator = Evaluator(num_class=6)
            
            # Class names for logging
            class_names = ['Background', 'Forest land', 'Grassland', 'Cropland', 'Settlement', 'Seminatural Grassland']
            
            # Training loop implementation - START FROM start_epoch
            for epoch in range(start_epoch, epochs):
                model.train()
                total_train_loss = 0
                train_evaluator = Evaluator(num_class=6)
                
                for batch in train_loader:
                    images = batch['img'].to(device)
                    masks = batch['gt_semantic_seg'].to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    
                    # Handle outputs
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
                        
                        if isinstance(outputs, tuple):
                            main_out = outputs[0]
                        else:
                            main_out = outputs
                            
                        val_loss = loss_fn(outputs, masks)
                        total_val_loss += val_loss.item()
                        
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
                
                # Log metrics
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
                
                # Save checkpoint
                temp_checkpoint_path = checkpoint_dir / f'temp_epoch{epoch + 1:02d}.ckpt'
                torch.save(checkpoint_data, temp_checkpoint_path)
                
                # Update best checkpoint tracker
                best_tracker.update(epoch + 1, val_miou, temp_checkpoint_path)
                
                # Always save the last checkpoint
                torch.save(checkpoint_data, checkpoint_dir / 'last.ckpt')
                
                # 🔥 IMPORTANT: Save progress after each epoch
                save_training_progress(checkpoint_dir, epoch, config_details)
                
                # Also save output after each epoch so we don't lose logs
                save_run_output(output.get_output_lines(), checkpoint_dir)
                
                # Update learning rate
                lr_scheduler.step()
            
            # Mark training as complete
            mark_training_complete(checkpoint_dir)
            output.add_line("=" * 80)
            output.add_line("[COMPLETED] TRAINING COMPLETED SUCCESSFULLY")
            output.add_line("=" * 80)
            
            # After training, rename the best checkpoints
            best_checkpoints = best_tracker.get_best_checkpoints()
            for i, (score, epoch, temp_path) in enumerate(best_checkpoints):
                if temp_path.exists():
                    new_name = f'{i+1}_epoch{epoch:02d}_miou{score:.4f}.ckpt'
                    new_path = checkpoint_dir / new_name
                    temp_path.rename(new_path)
            
            # Clean up temporary checkpoints
            for temp_file in checkpoint_dir.glob('temp_epoch*.ckpt'):
                if temp_file.exists():
                    temp_file.unlink()
            
        except Exception as e:
            output.add_line(f"[ERROR] Training failed with error: {str(e)}")
            output.add_line("\nFull traceback:")
            output.add_line(traceback.format_exc())
            # Don't mark as complete if there was an error
        finally:
            cleanup_wandb()
            
        return output.get_output_lines()

# ============================================================================
# MODIFIED is_training_completed
# ============================================================================

def is_training_completed(checkpoint_dir, epochs):
    """Check if training was already completed for this configuration"""
    last_completed_epoch, is_completed = get_training_progress(checkpoint_dir)
    return is_completed

# ============================================================================
# MAIN FUNCTION (UNCHANGED)
# ============================================================================

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
                    scale,
                    config_details
                )
                
                # Save output
                save_run_output(output_lines, checkpoint_dir)
                logging.info(f"Training completed and saved to {checkpoint_dir}")
                
            except Exception as e:
                logging.error(f"Error during training: {str(e)}")
                cleanup_wandb()
                continue
            
            # Clean up
            torch.cuda.empty_cache()
            
            logging.info(f"Completed combination {idx}/{total_combinations}")
    finally:
        # Final cleanup
        cleanup_wandb()
        # Deregister cleanup function
        atexit.unregister(cleanup_wandb)

if __name__ == '__main__':
    main()