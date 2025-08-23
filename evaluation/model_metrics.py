import os
import sys
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from geoseg.losses import *
from geoseg.datasets.biodiversity_dataset import *
from torch.utils.data import DataLoader
from tools.metric import Evaluator

# Class names for biodiversity dataset
CLASS_NAMES = ['Forest land', 'Grassland', 'Cropland', 'Settlement', 'Seminatural Grassland']

def evaluate_model(model_path, device='cuda'):
    """Load and evaluate a model checkpoint"""
    # Load model and checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint['model'] if isinstance(checkpoint, dict) else checkpoint
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
    """Plot per-class IoU and F1 scores"""
    iou_scores = evaluator.Intersection_over_Union()
    f1_scores = evaluator.F1()
    
    # Plot IoU scores
    plt.figure(figsize=(12, 6))
    plt.bar(CLASS_NAMES, iou_scores)
    plt.title('Per-Class IoU Scores')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_dir / 'class_iou_scores.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot F1 scores
    plt.figure(figsize=(12, 6))
    plt.bar(CLASS_NAMES, f1_scores)
    plt.title('Per-Class F1 Scores')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_dir / 'class_f1_scores.png', bbox_inches='tight', dpi=300)
    plt.close()

def save_metrics_report(evaluator, confusion_mat, save_dir):
    """Save detailed metrics report"""
    iou_scores = evaluator.Intersection_over_Union()
    f1_scores = evaluator.F1()
    oa_score = evaluator.OA()
    
    with open(save_dir / 'evaluation_report.txt', 'w') as f:
        f.write('=== Evaluation Report ===\n\n')
        
        # Overall metrics
        f.write(f'Overall Accuracy: {oa_score:.4f}\n')
        f.write(f'Mean IoU: {np.mean(iou_scores):.4f}\n')
        f.write(f'Mean F1: {np.mean(f1_scores):.4f}\n\n')
        
        # Per-class metrics
        f.write('Per-Class Metrics:\n')
        f.write('-' * 50 + '\n')
        for i, class_name in enumerate(CLASS_NAMES):
            f.write(f'\n{class_name}:\n')
            f.write(f'IoU: {iou_scores[i]:.4f}\n')
            f.write(f'F1: {f1_scores[i]:.4f}\n')
            f.write(f'Pixels: {confusion_mat[i].sum()}\n')

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
