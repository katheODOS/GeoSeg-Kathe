import os
import sys
import logging
from pathlib import Path
from itertools import product
import torch
import io
import numpy as np
import albumentations as albu
from torch.utils.data import DataLoader, Dataset
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
import random
from PIL import Image
import os.path as osp
import rasterio

# Define ORIGIN_IMG_SIZE before it's used in BiodiversityTiffDataset
ORIGIN_IMG_SIZE = 512

# Define augmentation functions
def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5)
    ]
    return albu.Compose(train_transform, additional_targets={'mask': 'mask'})

def get_validation_augmentation():
    # No special validation augmentation
    return albu.Compose([], additional_targets={'mask': 'mask'})

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    return albu.Compose([])

# Define train and val augmentations
train_aug = get_training_augmentation()
val_aug = get_validation_augmentation()

# Create a unified dataset class that can handle both training and validation
class BiodiversityTiffDataset(Dataset):
    def __init__(self, data_root='../data/Biodiversity_tiff/Train',
                 img_dir='images', mask_dir='masks',
                 img_suffix='.tif', mask_suffix='.png',
                 transform=None, mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE, mode='train'):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform        
        self.mosaic_ratio = mosaic_ratio if mode == 'train' else 0.0
        self.img_size = img_size
        self.mode = mode
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)
        
    def __getitem__(self, index):
        p_ratio = random.random()
        if self.mode == 'train' and p_ratio > self.mosaic_ratio:
            img, mask = self.load_img_and_mask(index)
            if self.transform:
                transformed = self.transform(image=img, mask=mask)
                img, mask = transformed['image'], transformed['mask']
        else:
            img, mask = self.load_img_and_mask(index)
            if self.transform:
                transformed = self.transform(image=img, mask=mask)
                img, mask = transformed['image'], transformed['mask']

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.img_ids[index]
        results = {'img': img, 'gt_semantic_seg': mask, 'img_id': img_id}
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        
        # Filter to only matching files
        img_ids = []
        for img_file in img_filename_list:
            if img_file.endswith('.tif'):
                img_name = str(img_file.split('.')[0])
                mask_file = img_name + self.mask_suffix
                if mask_file in mask_filename_list:
                    img_ids.append(img_name)
        
        print(f"Found {len(img_ids)} matching image-mask pairs in {data_root}")
        return img_ids

    def normalize_image(self, img_data):
        """Normalize image data to 0-1 range for each band"""
        normalized = np.zeros_like(img_data, dtype=np.float32)
        
        for i in range(img_data.shape[2]):
            band = img_data[:, :, i].astype(np.float32)
            # Remove nodata/invalid values for percentile calculation
            valid_pixels = band[~np.isnan(band)]
            valid_pixels = valid_pixels[valid_pixels != 0]  # Remove zeros
            
            if len(valid_pixels) > 0:
                # Use percentile normalization to handle outliers
                p2, p98 = np.percentile(valid_pixels, (2, 98))
                band = np.clip(band, p2, p98)
                band = (band - p2) / (p98 - p2) if p98 > p2 else band
            
            normalized[:, :, i] = band
        
        return normalized

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        
        # Load TIFF with rasterio to handle geospatial data properly
        try:
            with rasterio.open(img_name) as src:
                # Read all bands
                img_data = src.read()  # Shape: (bands, height, width)
                img_data = np.transpose(img_data, (1, 2, 0))  # Shape: (height, width, bands)
                
                # Handle nodata values
                nodata = src.nodata
                if nodata is not None:
                    img_data = np.where(img_data == nodata, 0, img_data)
                
                # Handle NaN values
                img_data = np.where(np.isnan(img_data), 0, img_data)
                
                # Normalize the image
                img_data = self.normalize_image(img_data)
                
                # Keep all 4 channels for 4-channel model
                img = img_data
                
        except Exception as e:
            print(f"Error reading TIFF {img_name}: {e}")
            # Fallback to zeros
            img = np.zeros((512, 512, 4), dtype=np.float32)
        
        # Load mask
        try:
            mask = np.array(Image.open(mask_name).convert('L'))
        except Exception as e:
            print(f"Error reading mask {mask_name}: {e}")
            mask = np.zeros((512, 512), dtype=np.uint8)
        
        return img, mask

num_classes = 6

LR = [4e-4, 5e-4]
BACKBONE_LR = [4e-5, 5e-5, 6e-5]
BATCH_SIZES = [16]
EPOCHS = [45, 50]
WEIGHT_DECAYS = [5e-2, 1e-1]
BACKBONE_WEIGHT_DECAYS = [1e-2]
SCALE = [1.0]

# Dataset configurations with path mappings (following hyperparameter_tuning.py format)
DATASETS = {
    'biodiversity': {
        'name': 'Biodiversity Dataset Tiff', 
        'code': 'biodiversity_tiff', 
        'path': 'Biodiversity_tiff/Train',
        'val_path': 'Biodiversity_tiff/Val'  # Add explicit validation path
    },
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
    checkpoint_dir = Path('C:/Users/Admin/anaconda3/envs/GeoSeg-Kathe/model_weights/biodiversity_tiff_ftunetformer_new') / dir_name
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
    try:
        # Add memory initialization to avoid random initialization errors
        torch.manual_seed(42)  # Set a fixed seed for reproducibility
        
        model = ft_unetformer(
            num_classes=num_classes,
            decoder_channels=256,
            pretrained=True,
            freeze_stages=-1,
            in_channels=4  # Use 4 channels for TIFF images
        )
        model = model.to(device=device)
    except Exception as e:
        logging.error(f"Error initializing model: {str(e)}")
        logging.error(traceback.format_exc())
        raise
    
    # Parse the config_details which is a string, not a dictionary
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
            val_path = data_root / info['val_path']
            break
    else:
        train_path = data_root / 'Biodiversity_tiff/Train'  # Default path
        val_path = data_root / 'Biodiversity_tiff/Val'      # Default val path
      # Validate paths exist
    if not train_path.exists():
        logging.error(f"Train directory not found: {train_path}")
        return []
        
    if not val_path.exists():
        logging.warning(f"Val directory not found: {val_path}")
        logging.warning("Will use Train dataset for validation")
        val_path = train_path
    else:
        logging.info(f"Using validation data from: {val_path}")
    
    # Create directories if they don't exist to ensure proper dataset loading
    for path in [train_path, val_path]:
        for subdir in ['images', 'masks']:
            os.makedirs(path / subdir, exist_ok=True)
        
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
            
            # Create datasets
            train_dataset = BiodiversityTiffDataset(
                data_root=str(train_path),
                mode='train',
                mosaic_ratio=0.0,
                transform=train_aug,
            )
            
            # Create validation dataset using the correct path
            val_dataset = BiodiversityTiffDataset(
                data_root=str(val_path), 
                mode='val',
                transform=val_aug,
            )
            
            logging.info(f"Training with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
            
            # Setup data loaders
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=True,
                shuffle=True,
                drop_last=True
            )
            
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=batch_size,
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
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)  # Changed to just epochs
            
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