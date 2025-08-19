import os
import sys
import logging
from pathlib import Path
from itertools import product
import torch
import io
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.biodiversity_dataset import *
from geoseg.models.DCSwin import dcswin_base 
from tools.utils import Lookahead
from tools.utils import process_model_params
from contextlib import redirect_stdout
from tqdm import tqdm
import traceback 
from io import StringIO
import atexit
import re
from tools.metric import Evaluator

# Hyperparameter configurations
LR = [6e-4, 3e-4, 1e-4]  # Base learning rates
BACKBONE_LR = [6e-5, 3e-5, 1e-5]  # Backbone learning rates
BATCH_SIZES = [8, 16, 32]
EPOCHS = [30, 40, 50]
WEIGHT_DECAYS = [1e-2, 1e-3]
BACKBONE_WEIGHT_DECAYS = [1e-2, 1e-3]
SCALE = [0.75, 1.0, 1.25]

# Dataset configurations with path mappings (following hyperparameter_tuning.py format)
DATASETS = {
    'biodiversity': {'name': 'Biodiversity Dataset', 'code': 'biodiversity', 'path': 'data/biodiversity/Train'},
}

def setup_checkpoint_dir(dataset_code, lr, backbone_lr, wd, backbone_wd, epochs, batch_size, scale):
    """Create and return checkpoint directory for specific configuration"""
    # Following hyperparameter_tuning.py naming convention
    dir_name = f"{dataset_code}L{lr:.0e}BL{backbone_lr:.0e}W{wd:.0e}BW{backbone_wd:.0e}B{batch_size}E{epochs}S{scale:.2f}"
    checkpoint_dir = Path('./checkpoints') / dir_name
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
    
    # Initialize model with DCSwin
    model = dcswin_base(num_classes=5, pretrained=True, weight_path='pretrain_weights/stseg_base.pth')
    model = model.to(device=device)
    
    with SafeOutputCapture() as output:
        try:
            sys.stdout.write("="*80 + "\n")
            sys.stdout.write(f"Configuration Details:\n")
            sys.stdout.write("="*80 + "\n")
            sys.stdout.write(config_details + "\n")
            sys.stdout.write("="*80 + "\n\n")
            
            # Setup dataset with proper transforms
            train_transform = [
                albu.HorizontalFlip(p=0.5),
                albu.Normalize()
            ]
            
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
            
            # Setup optimizer and scheduler
            layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
            net_params = process_model_params(model, layerwise_params=layerwise_params)
            base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
            optimizer = Lookahead(base_optimizer)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            
            # Setup loss - using same loss as in dcswin config
            loss_fn = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=0),
                               DiceLoss(smooth=0.05, ignore_index=0), 1.0, 1.0)
            
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
            evaluator = Evaluator(num_classes=5)
            
            # Training loop implementation
            for epoch in range(epochs):
                model.train()
                total_train_loss = 0
                
                for batch in train_loader:
                    images = batch['img'].to(device)
                    masks = batch['gt_semantic_seg'].to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = loss_fn(outputs, masks)
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                
                # Calculate average training loss
                train_loss = total_train_loss / len(train_loader)
                
                # Validation phase
                model.eval()
                total_val_loss = 0
                evaluator.reset()
                
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch['img'].to(device)
                        masks = batch['gt_semantic_seg'].to(device)
                        
                        outputs = model(images)
                        val_loss = loss_fn(outputs, masks)
                        total_val_loss += val_loss.item()
                        
                        # Get predictions for metrics
                        pred = outputs.data.cpu().numpy()
                        target = masks.cpu().numpy()
                        pred = np.argmax(pred, axis=1)
                        evaluator.add_batch(target, pred)
                
                # Calculate validation metrics
                val_loss = total_val_loss / len(val_loader)
                val_score = {
                    'mIoU': evaluator.Mean_Intersection_over_Union(),
                    'F1': evaluator.F1(),
                    'OA': evaluator.Pixel_Accuracy()
                }
                
                # Log metrics to stdout
                logging.info(f"""
                Epoch: {epoch}
                Train Loss: {train_loss:.4f}
                Val Loss: {val_loss:.4f}
                Val mIoU: {val_score['mIoU']:.4f}
                Val F1: {val_score['F1']:.4f}
                Val OA: {val_score['OA']:.4f}
                """)
                
                # Update learning rate
                lr_scheduler.step()
            
        except Exception as e:
            print(f"Training failed with error: {str(e)}")
            print("\nFull traceback:")
            print(traceback.format_exc())
        finally:
            cleanup_wandb()
            
        return output.get_output()

def is_training_completed(checkpoint_dir, epochs):
    """Check if training was already completed for this configuration"""
    final_checkpoint = checkpoint_dir / f'checkpoint_epoch{epochs}.pth'
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
        lr,
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