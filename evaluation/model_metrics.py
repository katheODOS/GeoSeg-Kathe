import os
import sys
from pathlib import Path
import datetime
import time

# Add project root to Python path
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from geoseg.losses import *
from geoseg.datasets.biodiversity_dataset import *
from torch.utils.data import DataLoader
from tools.metric import Evaluator
from geoseg.models.UNetFormer import UNetFormer
from geoseg.models.DCSwin import dcswin_base

# Class names for biodiversity dataset
CLASS_NAMES = ['Forest land', 'Grassland', 'Cropland', 'Settlement', 'Seminatural Grassland']

def evaluate_model(model_path, device='cuda'):
    """Load and evaluate a model checkpoint"""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize appropriate model based on checkpoint path
    if 'dcswin' in str(model_path).lower():
        model = dcswin_base(
            num_classes=6,
            pretrained=True,
            weight_path='pretrain_weights/stseg_base.pth'
        )
    else:
        model = UNetFormer(num_classes=6)
    
    # Load state dict from Lightning checkpoint
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        # Remove 'model.' prefix if it exists
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)  # Added strict=False
    
    model = model.to(device)
    model.eval()
    
    # Setup validation dataset and loader
    val_dataset = biodiversity_val_dataset
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=True
    )
    
    evaluator = Evaluator(num_class=6)
    confusion_mat = np.zeros((6, 6), dtype=np.int64)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            images, masks = batch['img'].to(device), batch['gt_semantic_seg']
            
            outputs = model(images)
            probs = nn.Softmax(dim=1)(outputs)
            preds = probs.argmax(dim=1).cpu().numpy()
            
            # Update metrics
            for true, pred in zip(masks.numpy(), preds):
                confusion_mat += confusion_matrix(
                    true.flatten(),
                    pred.flatten(),
                    labels=range(6)
                )
                evaluator.add_batch(true, pred)
    
    return evaluator, confusion_mat

def plot_confusion_matrix(confusion_mat, save_dir):
    """Plot and save confusion matrix heatmap"""
    plt.figure(figsize=(12, 10))
    conf_mat_percent = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(conf_mat_percent, annot=True, fmt='.2f', cmap='YlGn',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(save_dir / 'confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_class_metrics(evaluator, save_dir):
    """Plot per-class IoU and F1 scores with enhanced styling"""
    # Get scores excluding background class
    iou_scores = evaluator.Intersection_over_Union()[1:]
    f1_scores = evaluator.F1()[1:]
    
    # Get the model name from save_dir
    model_name = save_dir.parent.name
    
    # Plot IoU scores with enhanced styling
    plt.figure(figsize=(15, 8))
    cmap = plt.get_cmap('YlGn')
    colors = [cmap(i) for i in np.linspace(0.3, 0.9, len(CLASS_NAMES))]
    
    bars = plt.bar(range(len(CLASS_NAMES)), iou_scores, color=colors)
    plt.title(f'Per-Class IoU Scores\n{model_name}', fontsize=16, pad=20)
    plt.xlabel('Class', fontsize=15)
    plt.ylabel('IoU', fontsize=15)
    plt.ylim(0, 1)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_dir / 'class_iou_scores.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot F1 scores with same enhanced styling
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(CLASS_NAMES)), f1_scores, color=colors)
    plt.title(f'Per-Class F1 Scores\n{model_name}', fontsize=16, pad=20)
    plt.xlabel('Class', fontsize=15)
    plt.ylabel('F1', fontsize=15)
    plt.ylim(0, 1)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_dir / 'class_f1_scores.png', bbox_inches='tight', dpi=300)
    plt.close()

def save_metrics_report(evaluator, confusion_mat, save_dir):
    """Save detailed metrics report"""
    # Calculate metrics (excluding background class)
    iou_scores = evaluator.Intersection_over_Union()[1:]  # Skip background
    f1_scores = evaluator.F1()[1:]  # Skip background
    oa_score = evaluator.OA()
    
    # Create report
    with open(save_dir / 'evaluation_report.txt', 'w') as f:
        f.write('=== Evaluation Report ===\n\n')
        
        # Overall metrics (excluding background)
        f.write(f'Overall Accuracy: {oa_score:.4f}\n')
        f.write(f'Mean IoU (excluding background): {np.mean(iou_scores):.4f}\n')
        f.write(f'Mean F1 (excluding background): {np.mean(f1_scores):.4f}\n\n')
        
        # Per-class metrics
        f.write('Per-Class Metrics:\n')
        f.write('-' * 50 + '\n')
        for i, class_name in enumerate(CLASS_NAMES):
            f.write(f'\n{class_name}:\n')
            f.write(f'IoU: {iou_scores[i]:.4f}\n')
            f.write(f'F1: {f1_scores[i]:.4f}\n')
            # Use i+1 for confusion matrix since we skipped background
            f.write(f'Pixels: {confusion_mat[i+1].sum()}\n')

def was_recently_modified(folder_path, hours=0):
    """Check if any files inside a folder were modified within the specified hours"""
    if not folder_path.exists():
        return False, None, None
    
    current_time = time.time()
    time_threshold = current_time - (hours * 60 * 60)
    
    required_files = [
        folder_path / 'confusion_matrix.png',
        folder_path / 'class_iou_scores.png',
        folder_path / 'class_f1_scores.png',
        folder_path / 'evaluation_report.txt'
    ]
    
    for file_path in required_files:
        if file_path.exists():
            mod_time = file_path.stat().st_mtime
            if mod_time > time_threshold:
                mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                threshold_str = datetime.datetime.fromtimestamp(time_threshold).strftime('%Y-%m-%d %H:%M:%S')
                return True, mod_time_str, threshold_str
    
    return False, None, None

def process_checkpoint(checkpoint_path):
    """Process a single checkpoint and generate evaluation results"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create results directory
    results_dir = Path(checkpoint_path).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Evaluate model
    logging.info(f'Evaluating checkpoint: {checkpoint_path}')
    evaluator, confusion_mat = evaluate_model(checkpoint_path, device)
    
    # Generate and save results
    logging.info('Generating visualization and reports...')
    plot_confusion_matrix(confusion_mat, results_dir)
    plot_class_metrics(evaluator, results_dir)
    save_metrics_report(evaluator, confusion_mat, results_dir)
    
    logging.info(f'Results saved in {results_dir}')

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Base directory containing all model checkpoints
    base_dir = Path('C:/Users/Admin/anaconda3/envs/GeoSeg-Kathe/model_weights/biodiversity')
    
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Create results directory path
        results_dir = model_dir / 'results'
        
        # Check if results were recently generated
        recently_modified, mod_time, threshold = was_recently_modified(results_dir)
        if recently_modified:
            logging.info(f"Skipping {model_dir.name} - results were recently generated at {mod_time}")
            continue
            
        # Look for checkpoint files
        checkpoint_files = list(model_dir.glob('*.ckpt'))
        if not checkpoint_files:
            logging.warning(f'No checkpoint files found in {model_dir}')
            continue
            
        # Process the last checkpoint
        last_checkpoint = checkpoint_files[-1]
        try:
            process_checkpoint(last_checkpoint)
        except Exception as e:
            logging.error(f'Error processing {last_checkpoint}: {str(e)}')
            continue

if __name__ == '__main__':
    main()
