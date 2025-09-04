import os
import sys
import numpy as np
from PIL import Image
import rasterio
from pathlib import Path
import os.path as osp

# Add the project root to path
sys.path.append('.')

try:
    from geoseg.datasets.transform import *
    from geoseg.datasets.biodiversity_tiff_dataset import *
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def test_single_file(img_path, mask_path, transform_func=None):
    """Test loading and transforming a single image-mask pair"""
    print(f"\nTesting file pair:")
    print(f"  Image: {img_path}")
    print(f"  Mask:  {mask_path}")
    
    try:
        # Load image
        print("\n1. Loading image...")
        if str(img_path).endswith('.tif'):
            try:
                with rasterio.open(img_path) as src:
                    img_data = src.read()  # Shape: (bands, height, width)
                    img_data = np.transpose(img_data, (1, 2, 0))  # Shape: (height, width, bands)
                    print(f"   Rasterio - Shape: {img_data.shape}, dtype: {img_data.dtype}")
                    print(f"   Rasterio - Data range: {img_data.min()} to {img_data.max()}")
                    print(f"   Rasterio - Nodata: {src.nodata}")
                    
                    # Handle nodata
                    if src.nodata is not None:
                        img_data = np.where(img_data == src.nodata, 0, img_data)
                    
                    # Normalize if needed
                    if img_data.dtype != np.uint8:
                        # Simple normalization for testing
                        for i in range(img_data.shape[2]):
                            band = img_data[:, :, i]
                            if band.max() > band.min():
                                band_min, band_max = band.min(), band.max()
                                img_data[:, :, i] = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
                    
                    img = Image.fromarray(img_data.astype(np.uint8))
                    
            except Exception as e:
                print(f"   Rasterio failed: {e}, trying PIL...")
                img = Image.open(img_path)
        else:
            img = Image.open(img_path)
        
        print(f"   Image loaded - Mode: {img.mode}, Size: {img.size}")
        
        # Load mask
        print("\n2. Loading mask...")
        mask = Image.open(mask_path).convert('L')
        print(f"   Mask loaded - Mode: {mask.mode}, Size: {mask.size}")
        
        # Check sizes
        print(f"\n3. Size check:")
        print(f"   Image size: {img.size} (width, height)")
        print(f"   Mask size:  {mask.size} (width, height)")
        size_match = img.size == mask.size
        print(f"   Sizes match: {size_match}")
        
        if not size_match:
            print(f"   ERROR: Size mismatch detected!")
            return False
        
        # Test the problematic transform
        print(f"\n4. Testing crop augmentation...")
        crop_aug = Compose([
            RandomScale(scale_list=[1.0], mode='value'),  # No scaling for test
            SmartCropV1(crop_size=256, max_ratio=0.75, ignore_index=0, nopad=False)
        ])
        
        try:
            img_cropped, mask_cropped = crop_aug(img, mask)
            print(f"   Crop successful!")
            print(f"   Cropped image size: {img_cropped.size}")
            print(f"   Cropped mask size: {mask_cropped.size}")
            
            # Test full transform if provided
            if transform_func:
                print(f"\n5. Testing full transform...")
                img_transformed, mask_transformed = transform_func(img, mask)
                print(f"   Transform successful!")
                print(f"   Final shapes - img: {img_transformed.shape}, mask: {mask_transformed.shape}")
            
            return True
            
        except Exception as e:
            print(f"   Crop failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            
            # Try to debug the exact line that fails
            print(f"\n   Debugging crop failure...")
            try:
                print(f"   - Creating RandomScale...")
                scale_aug = RandomScale(scale_list=[1.0], mode='value')
                img_scaled, mask_scaled = scale_aug(img, mask)
                print(f"   - Scale successful: img {img_scaled.size}, mask {mask_scaled.size}")
                
                print(f"   - Creating SmartCropV1...")
                smart_crop = SmartCropV1(crop_size=256, max_ratio=0.75, ignore_index=0, nopad=False)
                print(f"   - Testing direct crop call...")
                img_final, mask_final = smart_crop(img_scaled, mask_scaled)
                print(f"   - Smart crop successful!")
                
            except Exception as debug_e:
                print(f"   - Debug failed at: {debug_e}")
            
            return False
            
    except Exception as e:
        print(f"Loading failed: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

def find_and_test_problematic_files(data_root, num_files=10):
    """Find and test files from the dataset"""
    # Convert to absolute path and resolve any symlinks
    data_root = Path(data_root).resolve()
    img_dir = data_root / 'images'
    mask_dir = data_root / 'masks'
    
    print(f"\nLooking for data in:")
    print(f"  Images: {img_dir}")
    print(f"  Masks:  {mask_dir}")
    
    if not img_dir.exists() or not mask_dir.exists():
        print(f"\nERROR: Dataset directories not found!")
        print(f"Please ensure the following directories exist:")
        print(f"  Images: {img_dir} (exists: {img_dir.exists()})")
        print(f"  Masks:  {mask_dir} (exists: {mask_dir.exists()})")
        print(f"\nNote: The directory structure should be:")
        print(f"  <data_root>/")
        print(f"    ├── images/")
        print(f"    └── masks/")
        return
    
    # Find matching files
    img_files = list(img_dir.glob('*.tif'))
    print(f"Found {len(img_files)} image files")
    
    successful = 0
    failed = 0
    
    for i, img_file in enumerate(img_files[:num_files]):
        mask_file = mask_dir / (img_file.stem + '.png')
        
        if not mask_file.exists():
            print(f"\nSkipping {img_file.name} - no matching mask")
            continue
            
        print(f"\n{'='*60}")
        print(f"Testing file {i+1}/{min(num_files, len(img_files))}: {img_file.name}")
        print(f"{'='*60}")
        
        success = test_single_file(img_file, mask_file, train_aug)
        
        if success:
            successful += 1
            print(f"✓ SUCCESS")
        else:
            failed += 1
            print(f"✗ FAILED")
            
            # Stop after first failure to debug
            print(f"\nStopping at first failure for debugging.")
            print(f"Problematic file: {img_file}")
            break
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {successful} successful, {failed} failed out of {successful + failed} tested")
    print(f"{'='*60}")

def test_dataset_loading():
    """Test the actual dataset class loading"""
    print(f"\n{'='*60}")
    print(f"TESTING DATASET CLASS")
    print(f"{'='*60}")
    
    try:
        # Create dataset instance with resolved path
        data_path = Path('data/Biodiversity_tiff/train').resolve()
        print(f"Looking for dataset in: {data_path}")
        
        # Use the factory function instead
        dataset = create_biodiversity_dataset(
            data_root=str(data_path),
            is_train=True
        )
        
        print(f"Dataset created with {len(dataset)} items")
        
        if len(dataset) == 0:
            print("ERROR: Dataset is empty!")
            return
        
        # Test loading first few items
        for i in range(min(3, len(dataset))):
            print(f"\nTesting dataset item {i}...")
            try:
                item = dataset[i]
                print(f"  Success! Item keys: {item.keys()}")
                if 'img' in item:
                    print(f"  Image shape: {item['img'].shape}")
                if 'gt_semantic_seg' in item:
                    print(f"  Mask shape: {item['gt_semantic_seg'].shape}")
            except Exception as e:
                print(f"  Failed: {e}")
                print(f"  Error type: {type(e).__name__}")
                
                # Get the problematic file info
                if hasattr(dataset, 'img_ids') and i < len(dataset.img_ids):
                    print(f"  Problematic file ID: {dataset.img_ids[i]}")
                
                break
                
    except Exception as e:
        print(f"Dataset creation failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test dataset loading and transforms')
    parser.add_argument('--data-root', type=str, 
                       help='Path to dataset root (contains "images" and "masks" subdirectories)')
    parser.add_argument('--num-files', type=int, default=5,
                       help='Number of files to test')
    parser.add_argument('--single-file', nargs=2, metavar=('IMG', 'MASK'),
                       help='Test single file pair (img_path mask_path)')
    
    args = parser.parse_args()
    
    if args.single_file:
        print("Testing single file pair...")
        test_single_file(args.single_file[0], args.single_file[1], train_aug)
    else:
        print("Testing dataset files...")
        find_and_test_problematic_files(args.data_root, args.num_files)
        test_dataset_loading()