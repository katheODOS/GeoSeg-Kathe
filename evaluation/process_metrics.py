import os
import json
from pathlib import Path
import re
import numpy as np

def extract_metrics(output_file):
    """Extract metrics and calculate mIoU excluding background class"""
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Find F1 and OA scores
            f1_matches = re.findall(r'Val F1: (\d+\.\d+)', content)
            oa_matches = re.findall(r'Val OA: (\d+\.\d+)', content)
            
            # Find per-class IoU values
            iou_pattern = r"'([^']+)':\s+(\d+\.\d+)"
            class_ious = re.findall(iou_pattern, content)
            
            if class_ious and f1_matches and oa_matches:
                # Convert to dictionary of class name to IoU value
                iou_dict = {name: float(value) for name, value in class_ious}
                
                # Calculate mIoU excluding background class
                non_background_ious = [
                    iou_dict[class_name] for class_name in iou_dict 
                    if class_name != 'Background'
                ]
                miou = np.mean(non_background_ious)
                
                return {
                    'mIoU': miou,
                    'F1': float(f1_matches[-1]),
                    'OA': float(oa_matches[-1]),
                    'per_class_iou': iou_dict
                }
            else:
                print(f"Missing metrics in {output_file}")
    except Exception as e:
        print(f"Error processing {output_file}: {str(e)}")
    return None

def process_metrics():
    """Process all biodiversity model folders and create sorted metric files"""
    base_dir = Path('C:/Users/Admin/anaconda3/envs/GeoSeg-Kathe/model_weights/biodiversity')
    metrics_dict = {
        'mIoU': {},
        'F1': {},
        'OA': {}
    }
    
    # Process each folder
    for folder in base_dir.iterdir():
        if folder.is_dir():
            output_file = folder / 'output.txt'
            if output_file.exists():
                metrics = extract_metrics(output_file)
                if metrics:
                    metrics_dict['mIoU'][folder.name] = metrics['mIoU']
                    metrics_dict['F1'][folder.name] = metrics['F1']
                    metrics_dict['OA'][folder.name] = metrics['OA']
    
    if not metrics_dict['mIoU']:
        print("No metrics found!")
        return
    
    # Save individual metric JSON files
    for metric in ['mIoU', 'F1', 'OA']:
        json_path = base_dir / f'{metric.lower()}_scores.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict[metric], f, indent=4)
    
    # Create sorted metrics text file
    with open(base_dir / 'validation_metrics_sorted.txt', 'w', encoding='utf-8') as f:
        for metric in ['mIoU', 'F1', 'OA']:
            sorted_scores = sorted(metrics_dict[metric].items(), key=lambda x: x[1], reverse=True)
            f.write(f"\n{metric} Scores (sorted):\n")
            f.write("="*50 + "\n")
            for folder_name, score in sorted_scores:
                f.write(f"{folder_name}: {score:.4f}\n")
            f.write("\n")
    
    print(f"Processed {len(metrics_dict['mIoU'])} folders")
    print("Created miou_scores.json, f1_scores.json, oa_scores.json")
    print("Created validation_metrics_sorted.txt with all metrics")

if __name__ == '__main__':
    process_metrics()
