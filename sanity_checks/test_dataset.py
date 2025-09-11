# test_dataset.py
import sys
import torch
sys.path.append('.')
from geoseg.datasets.biodiversity_tiff_dataset import BiodiversityTiffTrainDataset
from torch.utils.data import DataLoader
import numpy as np

# Create dataset
dataset = BiodiversityTiffTrainDataset(
    data_root='../data/Biodiversity_tiff/Train',  # Go up one level
    transform=None  
)

print(f"Dataset has {len(dataset)} samples")

# Test loading a few samples
for i in range(min(3, len(dataset))):
    try:
        sample = dataset[i]
        img = sample['img'] 
        mask = sample['gt_semantic_seg']
        
        print(f"\nSample {i}:")
        print(f"  Image shape: {img.shape}")
        print(f"  Image dtype: {img.dtype}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Unique mask values: {torch.unique(mask)}")
        
        # Check number of channels
        if img.shape[0] == 4:
            print("4 channels detected!")
        else:
            print(f"Expected 4 channels, got {img.shape[0]}")
            
    except Exception as e:
        print(f"Error loading sample {i}: {e}")
        import traceback
        traceback.print_exc()