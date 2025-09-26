#!/usr/bin/env python3
"""
Optimized checkpoint loader for biodiversity segmentation model
Specifically designed to work with PyTorch Lightning checkpoints
"""

import sys
import os

# Add the parent directory to Python path
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
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [11, 246, 210] #ignore index
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [250, 62, 119] #forestland
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [168, 232, 84] #grassland
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [242, 180, 92] #cropland
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [116, 116, 116] #settlement
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [255, 214, 33] #seminatural grassland
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
    """Simple dataset that loads any images from a directory"""
    
    def __init__(self, image_dir, expected_channels=3):
        self.image_dir = Path(image_dir)
        self.expected_channels = expected_channels
        
        # Find ALL image files
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
        
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(list(self.image_dir.glob(ext)))
            self.image_paths.extend(list(self.image_dir.glob(ext.upper())))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in self.image_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
        self.image_paths = unique_paths
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
        if len(self.image_paths) > 0:
            print(f"Example files: {[p.name for p in self.image_paths[:3]]}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_id = img_path.stem
        
        try:
            # Check file extension to decide how to load
            if img_path.suffix.lower() in ['.tif', '.tiff']:
                # Load TIFF with rasterio
                with rasterio.open(img_path) as src:
                    img_data = src.read()  # Shape: (bands, height, width)
                    
                    # Debug info for first image
                    if idx == 0:
                        print(f"\nDebug - First image loaded: {img_path.name}")
                        print(f"  Original shape: {img_data.shape}")
                        print(f"  Expected channels: {self.expected_channels}")
                    
                    # Take exactly what the model expects
                    if img_data.shape[0] >= self.expected_channels:
                        img_data = img_data[:self.expected_channels]
                    else:
                        # Pad if needed
                        padded = np.zeros((self.expected_channels, img_data.shape[1], img_data.shape[2]), dtype=img_data.dtype)
                        padded[:img_data.shape[0]] = img_data
                        img_data = padded
                    
                    # Normalize to 0-1 range
                    img_data = self.normalize_image(img_data)
                    img_tensor = torch.from_numpy(img_data).float()
            
            else:
                # Load RGB image (PNG, JPG, etc.)
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                # Convert to CHW format and normalize
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                
                # If model expects more than 3 channels, pad with zeros
                if self.expected_channels > 3:
                    padding = torch.zeros(self.expected_channels - 3, img_tensor.shape[1], img_tensor.shape[2])
                    img_tensor = torch.cat([img_tensor, padding], dim=0)
            
            return {
                'img': img_tensor,
                'img_id': img_id,
                'img_path': str(img_path)
            }
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy data to avoid crashing
            dummy_img = torch.zeros(self.expected_channels, 256, 256)
            return {
                'img': dummy_img,
                'img_id': f"error_{idx}",
                'img_path': str(img_path)
            }
    
    def normalize_image(self, img_data):
        """Normalize multi-band image to 0-1 range"""
        normalized = np.zeros_like(img_data, dtype=np.float32)
        
        for i in range(img_data.shape[0]):
            band = img_data[i].astype(np.float32)
            # Remove nodata/invalid values
            valid_pixels = band[~np.isnan(band)]
            valid_pixels = valid_pixels[valid_pixels != 0]
            
            if len(valid_pixels) > 0:
                # Use percentile normalization
                p2, p98 = np.percentile(valid_pixels, (2, 98))
                band = np.clip(band, p2, p98)
                if p98 > p2:
                    band = (band - p2) / (p98 - p2)
                else:
                    band = np.zeros_like(band)
            else:
                band = np.zeros_like(band)
            
            normalized[i] = band
        
        return normalized


def inspect_checkpoint(checkpoint_path):
    """Inspect a checkpoint file and print detailed info about its structure"""
    print(f"\n🔍 INSPECTING CHECKPOINT: {checkpoint_path}")
    print("=" * 60)
    
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        
        # Print top-level keys
        print("Top-level keys:", list(checkpoint.keys()))
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"State dict contains {len(state_dict)} keys")
            
            # Print a few sample keys
            print("Sample keys:")
            for i, k in enumerate(list(state_dict.keys())[:5]):
                print(f"  {k}: {state_dict[k].shape}")
            
            # Look for model architecture clues
            for key in state_dict.keys():
                if 'patch_embed.proj.weight' in key:
                    weight = state_dict[key]
                    print(f"Found input layer: {key} with shape {weight.shape}")
                    print(f"This model expects {weight.shape[1]} input channels")
        
        if 'pytorch-lightning_version' in checkpoint:
            print(f"PyTorch Lightning version: {checkpoint['pytorch-lightning_version']}")
        
        return True
    except Exception as e:
        print(f"Error inspecting checkpoint: {e}")
        return False


def detect_model_input_channels(checkpoint_path):
    """Detect how many input channels the model expects"""
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        
        # Check for state_dict (PyTorch Lightning format)
        if 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
            
            # Look for input layer
            for key, tensor in model_state_dict.items():
                if 'patch_embed.proj.weight' in key and len(tensor.shape) == 4:
                    input_channels = tensor.shape[1]
                    print(f"✓ Model expects {input_channels} input channels (from {key})")
                    return input_channels
        
        print("❌ Could not determine input channels")
        return None
        
    except Exception as e:
        print(f"❌ Error reading checkpoint: {e}")
        return None


def create_model(num_classes=6, input_channels=3):
    """Create model with specified parameters"""
    try:
        from geoseg.models.FTUNetFormer import ft_unetformer
        
        model = ft_unetformer(
            num_classes=num_classes,
            decoder_channels=256,
            in_channels=input_channels,
            pretrained=False
        )
        
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        raise


def load_checkpoint_weights(model, checkpoint_path):
    """Load weights from checkpoint into model"""
    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        
        # Handle PyTorch Lightning checkpoint format
        if 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
            print(f"Found state_dict with {len(model_state_dict)} keys")
            
            # Clean up state_dict keys - remove 'net.' prefix if present
            cleaned_state_dict = {}
            for k, v in model_state_dict.items():
                if k.startswith('net.'):
                    cleaned_state_dict[k[4:]] = v
                else:
                    cleaned_state_dict[k] = v
            
            print(f"Cleaned state_dict has {len(cleaned_state_dict)} keys")
            print(f"First few model keys: {list(model.state_dict().keys())[:3]}")
            print(f"First few checkpoint keys: {list(cleaned_state_dict.keys())[:3]}")
            
            missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        else:
            print("❌ No state_dict found in checkpoint")
            return False
        
        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            print("✓ All weights loaded perfectly")
        else:
            print(f"⚠️  Loaded with {len(missing_keys)} missing, {len(unexpected_keys)} unexpected keys")
            if len(missing_keys) > 0:
                print(f"   First few missing keys: {missing_keys[:5]}")
            if len(unexpected_keys) > 0:
                print(f"   First few unexpected keys: {unexpected_keys[:5]}")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-w", "--checkpoint_path", type=Path, required=True, help="Path to checkpoint file (.ckpt)")
    arg("-i", "--input_path", type=Path, required=True, help="Path to input directory with images")
    arg("-o", "--output_path", type=Path, required=True, help="Path where to save results")
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("--rgb", help="Output RGB masks", action='store_true')
    arg("--num-classes", type=int, default=6, help="Number of classes")
    arg("--force-channels", type=int, choices=[3, 4], help="Force number of input channels")
    arg("--inspect", action='store_true', help="Just inspect the checkpoint without running inference")
    return parser.parse_args()


def main():
    args = get_args()
    
    # Validate inputs
    if not args.checkpoint_path.exists():
        print(f"❌ Checkpoint file not found: {args.checkpoint_path}")
        return
    
    print("="*60)
    print("BIODIVERSITY INFERENCE - OPTIMIZED FOR PYTORCH LIGHTNING")
    print("="*60)
    
    # Inspect checkpoint if requested
    if args.inspect:
        inspect_checkpoint(args.checkpoint_path)
        return
    
    # Detect model requirements
    if args.force_channels:
        input_channels = args.force_channels
        print(f"✓ Forced to {input_channels} channels")
    else:
        input_channels = detect_model_input_channels(args.checkpoint_path)
        if input_channels is None:
            print("💡 Try --force-channels 3 or --force-channels 4")
            return
    
    if not args.input_path.exists():
        print(f"❌ Input directory not found: {args.input_path}")
        return
    
    args.output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\nSetting up {input_channels}-channel inference pipeline...")
    
    # Create dataset - it will load whatever images are in the directory
    dataset = SimpleImageDataset(args.input_path, input_channels)
    if len(dataset) == 0:
        print(f"❌ No images found in {args.input_path}!")
        return
    
    # Create model
    print(f"Creating {input_channels}-channel model...")
    try:
        model = create_model(args.num_classes, input_channels)
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return
    
    # Load weights
    print("Loading checkpoint weights...")
    if not load_checkpoint_weights(model, args.checkpoint_path):
        print("❌ Failed to load weights properly, trying to continue anyway...")
    
    # Test model
    print("Testing model...")
    try:
        sample = dataset[0]
        dummy_input = sample['img'].unsqueeze(0)  # Add batch dimension
        model.eval()
        with torch.no_grad():
            test_output = model(dummy_input)
        print(f"✓ Model test successful!")
        print(f"  Input: {dummy_input.shape}")
        if isinstance(test_output, (list, tuple)):
            print(f"  Output: {test_output[0].shape}")
        else:
            print(f"  Output: {test_output.shape}")
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Setup for inference
    model.cuda()
    model.eval()
    
    # Setup TTA
    if args.tta == "lr":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip()
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)
        print("✓ LR TTA enabled")
    elif args.tta == "d4":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False),
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)
        print("✓ D4 TTA enabled")
    
    # Create data loader
    test_loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    # Process images
    print(f"\nProcessing {len(dataset)} images...")
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing")):
            try:
                img_cuda = batch['img'].cuda()
                
                # Get predictions
                raw_predictions = model(img_cuda)
                if isinstance(raw_predictions, (list, tuple)):
                    raw_predictions = raw_predictions[0]
                
                # Convert to class predictions
                predictions = nn.Softmax(dim=1)(raw_predictions).argmax(dim=1)
                
                # Save results
                for i in range(predictions.shape[0]):
                    mask = predictions[i].cpu().numpy()
                    img_id = batch['img_id'][i]
                    results.append((mask, str(args.output_path / img_id), args.rgb))
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                if batch_idx == 0:
                    import traceback
                    traceback.print_exc()
                continue
    
    # Write results
    if results:
        print(f"\nSaving {len(results)} results...")
        t0 = time.time()
        mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
        t1 = time.time()
        
        print(f'✓ Completed in {t1-t0:.2f} seconds')
        print(f'✓ Results saved to: {args.output_path}')
        
        # Show examples
        example_files = list(args.output_path.glob('*.png'))[:3]
        for f in example_files:
            print(f'   {f.name}')
    else:
        print("❌ No results generated!")


if __name__ == "__main__":
    main()