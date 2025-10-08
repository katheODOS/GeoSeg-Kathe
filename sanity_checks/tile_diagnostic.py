#!/usr/bin/env python3
"""
FIXED: Properly loads PyTorch Lightning checkpoints for 4-band FTUNetFormer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import rasterio

from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [11, 246, 210]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [250, 62, 119]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [168, 232, 84]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [242, 180, 92]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [116, 116, 116]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [255, 214, 33]
    return mask_rgb


def img_writer(inp):
    (mask, mask_id, rgb) = inp
    if rgb:
        mask_name = mask_id + '.png'
        mask_tif = label2rgb(mask)
        mask_tif = cv2.cvtColor(mask_tif, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_name, mask_tif)
    else:
        mask_tif = mask.astype(np.uint8)
        mask_name = mask_id + '.png'
        cv2.imwrite(mask_name, mask_tif)


class SimpleImageDataset(Dataset):
    """Dataset that loads 4-band TIFF images"""
    
    def __init__(self, image_dir, expected_channels=4):
        self.image_dir = Path(image_dir)
        self.expected_channels = expected_channels
        
        # Find TIFF files
        self.image_paths = list(self.image_dir.glob('*.tif')) + list(self.image_dir.glob('*.tiff'))
        
        print(f"Found {len(self.image_paths)} images")
        if len(self.image_paths) > 0:
            print(f"Example: {self.image_paths[0].name}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_id = img_path.stem
        
        try:
            # Load TIFF with rasterio
            with rasterio.open(img_path) as src:
                img_data = src.read()  # (bands, height, width)
                nodata_value = src.nodata  # Get the nodata value from metadata
                
                # Take first channels
                if img_data.shape[0] >= self.expected_channels:
                    img_data = img_data[:self.expected_channels]
                else:
                    # Pad if needed
                    padded = np.zeros((self.expected_channels, img_data.shape[1], img_data.shape[2]), 
                                     dtype=img_data.dtype)
                    padded[:img_data.shape[0]] = img_data
                    img_data = padded
                
                # Create a mask for valid pixels (no NaN, no nodata)
                valid_mask = np.ones((img_data.shape[1], img_data.shape[2]), dtype=bool)
                for i in range(img_data.shape[0]):
                    band_mask = ~np.isnan(img_data[i]) & ~np.isinf(img_data[i])
                    if nodata_value is not None:
                        band_mask &= (img_data[i] != nodata_value)
                    valid_mask &= band_mask
                
                # Count valid pixels
                valid_pixel_count = valid_mask.sum()
                total_pixels = valid_mask.size
                valid_ratio = valid_pixel_count / total_pixels
                
                # Normalize each band to 0-1, COMPLETELY ISOLATING invalid pixels
                img_normalized = np.zeros((img_data.shape[0], img_data.shape[1], img_data.shape[2]), 
                                         dtype=np.float32)
                
                for i in range(img_data.shape[0]):
                    band = img_data[i].astype(np.float32)
                    
                    # Create output array initialized to 0
                    band_normalized = np.zeros_like(band, dtype=np.float32)
                    
                    # Get valid pixels for THIS band only
                    valid_pixels = band[valid_mask]
                    
                    if len(valid_pixels) > 10:  # Need at least some valid pixels
                        # Compute percentiles ONLY on valid pixels
                        p2, p98 = np.percentile(valid_pixels, (2, 98))
                        
                        if p98 > p2:
                            # Normalize ONLY the valid pixels
                            # This completely avoids touching NaN/nodata
                            valid_band_values = band[valid_mask]
                            normalized_valid = np.clip(valid_band_values, p2, p98)
                            normalized_valid = (normalized_valid - p2) / (p98 - p2)
                            
                            # Put normalized values back into valid positions
                            band_normalized[valid_mask] = normalized_valid
                            # Invalid positions stay at 0
                            
                            img_normalized[i] = band_normalized
                        else:
                            # All valid pixels same value - set to 0.5
                            band_normalized[valid_mask] = 0.5
                            img_normalized[i] = band_normalized
                    else:
                        # Not enough valid pixels - leave as zeros
                        img_normalized[i] = band_normalized
                
                # Convert to CHW format and tensor
                img_tensor = torch.from_numpy(img_normalized).float()
            
            return {
                'img': img_tensor,
                'img_id': img_id,
                'img_path': str(img_path),
                'valid_pixel_ratio': valid_mask.sum() / valid_mask.size  # Track how much is valid
            }
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            dummy_img = torch.zeros(self.expected_channels, 256, 256)
            return {
                'img': dummy_img, 
                'img_id': f"error_{idx}", 
                'img_path': str(img_path),
                'valid_pixel_ratio': 0.0
            }


def load_checkpoint_properly(checkpoint_path, num_classes=6, input_channels=4):
    """
    CRITICAL FIX: Properly load checkpoint (handles multiple formats)
    """
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint file
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    
    # Debug: show what keys are in checkpoint
    print(f"✓ Checkpoint keys: {list(checkpoint.keys())}")
    
    # Extract state_dict - handle different checkpoint formats
    state_dict = None
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"✓ Found 'state_dict' with {len(state_dict)} keys")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"✓ Found 'model_state_dict' with {len(state_dict)} keys")
    else:
        print("❌ ERROR: No state_dict found!")
        print(f"Available keys: {list(checkpoint.keys())}")
        return None
    
    # Show sample keys to understand the structure
    sample_keys = list(state_dict.keys())[:5]
    print(f"Sample keys: {sample_keys}")
    
    # Create model
    from geoseg.models.FTUNetFormer import ft_unetformer
    
    model = ft_unetformer(
        num_classes=num_classes,
        decoder_channels=256,
        in_channels=input_channels,
        pretrained=False  # Don't load pretrained weights
    )
    
    # Clean up state_dict keys - remove 'net.' or 'model.' prefix if present
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # Remove common prefixes
        new_key = key
        if key.startswith('net.'):
            new_key = key[4:]  # Remove 'net.'
        elif key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.'
        
        cleaned_state_dict[new_key] = value
    
    print(f"Cleaned state_dict sample: {list(cleaned_state_dict.keys())[:3]}")
    
    # Load weights into model
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    
    if len(missing_keys) == 0 and len(unexpected_keys) == 0:
        print("✅ Perfect match! All weights loaded successfully")
    else:
        print(f"⚠️  Loaded with {len(missing_keys)} missing, {len(unexpected_keys)} unexpected keys")
        if missing_keys:
            print(f"Missing keys (first 5): {missing_keys[:5]}")
        if unexpected_keys:
            print(f"Unexpected keys (first 5): {unexpected_keys[:5]}")
    
    return model


def test_model_on_sample(model, dataset):
    """Test model on first image to verify it works"""
    print("\n🧪 Testing model on sample image...")
    
    sample = dataset[0]
    dummy_input = sample['img'].unsqueeze(0)  # Add batch dimension
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Input range: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")
    
    model.eval()
    with torch.no_grad():
        try:
            test_output = model(dummy_input)
            
            if isinstance(test_output, (list, tuple)):
                output_shape = test_output[0].shape
            else:
                output_shape = test_output.shape
            
            print(f"  Output shape: {output_shape}")
            print("✅ Model test successful!")
            return True
            
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-w", "--checkpoint_path", type=Path, required=True, help="Path to checkpoint (.ckpt)")
    arg("-i", "--input_path", type=Path, required=True, help="Path to input directory")
    arg("-o", "--output_path", type=Path, required=True, help="Path to save results")
    arg("-t", "--tta", default=None, choices=[None, "d4", "lr"], help="Test time augmentation")
    arg("--rgb", action='store_true', help="Output RGB masks")
    arg("--num-classes", type=int, default=6, help="Number of classes")
    arg("--force-channels", type=int, default=4, choices=[3, 4, 8], help="Input channels")
    return parser.parse_args()


def main():
    args = get_args()
    
    # Validate paths
    if not args.checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {args.checkpoint_path}")
        return
    
    if not args.input_path.exists():
        print(f"❌ Input directory not found: {args.input_path}")
        return
    
    args.output_path.mkdir(exist_ok=True, parents=True)
    
    print("=" * 70)
    print("FIXED 4-BAND FTUNETFORMER INFERENCE")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint_path.name}")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Channels: {args.force_channels}")
    print("=" * 70)
    
    # Create dataset
    print("\n📁 Loading dataset...")
    dataset = SimpleImageDataset(args.input_path, args.force_channels)
    
    if len(dataset) == 0:
        print("❌ No images found!")
        return
    
    # Load model with proper checkpoint handling
    print("\n🔧 Loading model...")
    model = load_checkpoint_properly(
        args.checkpoint_path,
        num_classes=args.num_classes,
        input_channels=args.force_channels
    )
    
    if model is None:
        print("❌ Failed to load model!")
        return
    
    # Test model
    if not test_model_on_sample(model, dataset):
        print("❌ Model test failed - aborting")
        return
    
    # Move to GPU
    model.cuda()
    model.eval()
    
    # Setup TTA if requested
    if args.tta == "lr":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        model = tta.SegmentationTTAWrapper(model, transforms)
        print("✓ LR TTA enabled")
    elif args.tta == "d4":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)
        print("✓ D4 TTA enabled")
    
    # Create dataloader
    test_loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    # Process images
    print(f"\n🔄 Processing {len(dataset)} images...")
    results = []
    tiles_with_low_valid_data = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing")):
            try:
                img_cuda = batch['img'].cuda()
                valid_ratios = batch.get('valid_pixel_ratio', [1.0] * len(batch['img_id']))
                
                # Forward pass
                raw_predictions = model(img_cuda)
                if isinstance(raw_predictions, (list, tuple)):
                    raw_predictions = raw_predictions[0]
                
                # Convert to class predictions
                predictions = nn.Softmax(dim=1)(raw_predictions).argmax(dim=1)
                
                # Collect results
                for i in range(predictions.shape[0]):
                    mask = predictions[i].cpu().numpy()
                    img_id = batch['img_id'][i]
                    valid_ratio = valid_ratios[i].item() if hasattr(valid_ratios[i], 'item') else valid_ratios[i]
                    
                    # Track tiles with low valid data
                    if valid_ratio < 0.5:
                        tiles_with_low_valid_data.append((img_id, valid_ratio))
                    
                    results.append((mask, str(args.output_path / img_id), args.rgb))
                    
            except Exception as e:
                print(f"\n❌ Error in batch {batch_idx}: {e}")
                if batch_idx == 0:
                    import traceback
                    traceback.print_exc()
                continue
    
    # Write results
    if results:
        print(f"\n💾 Saving {len(results)} results...")
        
        # Report tiles with low valid data
        if tiles_with_low_valid_data:
            print(f"\n⚠️  Warning: {len(tiles_with_low_valid_data)} tiles have <50% valid pixels:")
            for tile_name, ratio in tiles_with_low_valid_data[:10]:  # Show first 10
                print(f"  - {tile_name}: {ratio*100:.1f}% valid")
            if len(tiles_with_low_valid_data) > 10:
                print(f"  ... and {len(tiles_with_low_valid_data) - 10} more")
        
        t0 = time.time()
        mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
        t1 = time.time()
        
        print(f"\n✅ Completed in {t1-t0:.2f} seconds")
        print(f"✅ Results saved to: {args.output_path}")
        
        # Show examples
        example_files = list(args.output_path.glob('*.png'))[:3]
        if example_files:
            print("\nExample outputs:")
            for f in example_files:
                print(f"  - {f.name}")
    else:
        print("❌ No results generated!")


if __name__ == "__main__":
    main()