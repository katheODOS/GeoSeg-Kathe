# test_dataloader.py
import sys
sys.path.append('.')
from torch.utils.data import DataLoader
from geoseg.datasets.biodiversity_tiff_dataset import *

# Create dataset with transforms - Fix the path
dataset = BiodiversityTiffTrainDataset(
    data_root='../data/Biodiversity_tiff/Train',  # Go up one level
    transform=train_aug
)
# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0  # Start with 0 for debugging
)

# Test loading a batch
try:
    for batch_idx, batch in enumerate(loader):
        img = batch['img']
        mask = batch['gt_semantic_seg']
        
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {img.shape}")  # Should be [B, 4, H, W]
        print(f"  Masks shape: {mask.shape}")   # Should be [B, H, W]
        
        if img.shape[1] == 4:
            print("  ✅ Batch has 4 channels!")
        else:
            print(f"  ⚠️ Expected 4 channels, got {img.shape[1]}")
        
        if batch_idx >= 2:  # Test first 3 batches
            break
            
except Exception as e:
    print(f"❌ Error in dataloader: {e}")
    import traceback
    traceback.print_exc()