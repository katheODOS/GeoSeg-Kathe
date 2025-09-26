#!/usr/bin/env python3
"""
Calculate dataset statistics while properly handling NaN values
"""

import os
import sys
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import argparse

def load_tiff_image_robust(filepath, verbose=False):
    """Load and preprocess a TIFF image, handling NaN values gracefully"""
    try:
        with rasterio.open(filepath) as src:
            img = src.read()  # Shape: (bands, height, width)
            
            if verbose:
                print(f"  Raw shape: {img.shape}, dtype: {img.dtype}")
                print(f"  Raw range: {img.min()} to {img.max()}")
            
            img = np.transpose(img, (1, 2, 0))  # Convert to (height, width, bands)
            
            # Handle nodata values
            if src.nodata is not None:
                nodata_count = (img == src.nodata).sum()
                if verbose and nodata_count > 0:
                    print(f"  Replacing {nodata_count} nodata values ({src.nodata}) with NaN")
                img = np.where(img == src.nodata, np.nan, img)
            
            # Choose normalization based on data range (for non-NaN values)
            finite_mask = np.isfinite(img)
            if finite_mask.any():
                finite_data = img[finite_mask]
                original_max = finite_data.max()
                original_min = finite_data.min()
                
                if verbose:
                    print(f"  Finite data range: {original_min} to {original_max}")
                
                if original_max <= 1 and original_min >= 0:
                    # Data already in [0,1] range
                    img_norm = img.astype(np.float32)
                    if verbose:
                        print(f"  Data already in [0,1] range")
                elif original_max <= 255:
                    # Likely 8-bit data
                    img_norm = img.astype(np.float32) / 255.0
                    if verbose:
                        print(f"  Scaling by 255 (8-bit data)")
                elif original_max <= 10000:
                    # Typical for some satellite data
                    img_norm = img.astype(np.float32) / 10000.0
                    if verbose:
                        print(f"  Scaling by 10000")
                elif original_max <= 65535:
                    # 16-bit data
                    img_norm = img.astype(np.float32) / 65535.0
                    if verbose:
                        print(f"  Scaling by 65535 (16-bit data)")
                else:
                    # Use percentile-based normalization
                    p99 = np.percentile(finite_data, 99)
                    img_norm = img.astype(np.float32) / p99
                    if verbose:
                        print(f"  Using 99th percentile normalization: {p99}")
                
                # Clip to [0,1] but preserve NaN values
                img_norm = np.clip(img_norm, 0, 1)
                
                # Restore NaN values where they were originally
                img_norm[~finite_mask] = np.nan
                
            else:
                # All values are NaN/inf - return as is
                img_norm = img.astype(np.float32)
                if verbose:
                    print(f"  Warning: No finite values found in image")
            
            if verbose:
                finite_after = np.isfinite(img_norm)
                if finite_after.any():
                    finite_vals = img_norm[finite_after]
                    print(f"  After normalization - finite range: {finite_vals.min():.6f} to {finite_vals.max():.6f}")
                    print(f"  Valid pixels: {finite_after.sum():,}, NaN pixels: {(~finite_after).sum():,}")
                else:
                    print(f"  After normalization: No finite values (all NaN/Inf)")
            
            return img_norm
            
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def calculate_dataset_statistics_robust(data_root, sample_size=None, min_valid_pixels=1000):
    """
    Calculate statistics while handling NaN values properly
    
    Args:
        data_root: Path to dataset root
        sample_size: Number of files to sample (None = use all)
        min_valid_pixels: Minimum number of valid (non-NaN) pixels per file to include
    """
    data_root = Path(data_root)
    img_dir = data_root / 'images'
    
    if not img_dir.exists():
        raise ValueError(f"Images directory not found: {img_dir}")
    
    tiff_files = list(img_dir.glob('*.tif'))
    print(f"Found {len(tiff_files)} TIFF files")
    
    if sample_size and sample_size < len(tiff_files):
        import random
        tiff_files = random.sample(tiff_files, sample_size)
        print(f"Using random sample of {len(tiff_files)} files")
    
    # Initialize accumulators for each channel
    channel_values = [[] for _ in range(4)]  # Assuming 4 bands
    files_processed = 0
    files_with_valid_data = 0
    total_valid_pixels = 0
    
    print("Processing files...")
    for tiff_path in tqdm(tiff_files):
        try:
            img = load_tiff_image_robust(tiff_path, verbose=False)
            
            if img is None:
                print(f"Failed to load: {tiff_path.name}")
                continue
                
            files_processed += 1
            
            # Check expected number of channels
            if img.shape[2] != 4:
                print(f"Warning: {tiff_path.name} has {img.shape[2]} channels, expected 4. Skipping.")
                continue
            
            # Count valid pixels per channel
            valid_pixels_per_channel = []
            file_has_valid_data = False
            
            for c in range(4):
                channel_data = img[:, :, c]
                finite_mask = np.isfinite(channel_data)
                valid_count = finite_mask.sum()
                valid_pixels_per_channel.append(valid_count)
                
                if valid_count >= min_valid_pixels:
                    # Extract valid values for this channel
                    valid_values = channel_data[finite_mask]
                    channel_values[c].extend(valid_values)
                    file_has_valid_data = True
            
            if file_has_valid_data:
                files_with_valid_data += 1
                total_valid_pixels += sum(valid_pixels_per_channel)
                print(f"✓ {tiff_path.name}: Valid pixels per band: {valid_pixels_per_channel}")
            else:
                print(f"⚠ {tiff_path.name}: Insufficient valid pixels per band: {valid_pixels_per_channel} (min: {min_valid_pixels})")
                
        except Exception as e:
            print(f"Error processing {tiff_path}: {e}")
            continue
    
    print(f"\nProcessing Summary:")
    print(f"  Files attempted: {len(tiff_files)}")
    print(f"  Files loaded: {files_processed}")
    print(f"  Files with valid data: {files_with_valid_data}")
    print(f"  Total valid pixels: {total_valid_pixels:,}")
    
    if files_with_valid_data == 0:
        print("ERROR: No files with sufficient valid data found!")
        return None, None
    
    # Calculate statistics from collected valid pixels
    means = []
    stds = []
    
    print(f"\nPer-channel statistics:")
    for c in range(4):
        if len(channel_values[c]) > 0:
            channel_array = np.array(channel_values[c])
            mean_val = np.mean(channel_array)
            std_val = np.std(channel_array)
            means.append(mean_val)
            stds.append(std_val)
            
            print(f"  Channel {c}: {len(channel_values[c]):,} valid pixels, "
                  f"mean={mean_val:.6f}, std={std_val:.6f}, "
                  f"range=[{channel_array.min():.6f}, {channel_array.max():.6f}]")
        else:
            print(f"  Channel {c}: No valid pixels found!")
            means.append(np.nan)
            stds.append(np.nan)
    
    return np.array(means), np.array(stds)

def main():
    parser = argparse.ArgumentParser(description='Calculate dataset statistics with robust NaN handling')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to dataset root (contains "images" subdirectory)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of random images to sample (default: use all)')
    parser.add_argument('--min-valid-pixels', type=int, default=1000,
                       help='Minimum valid pixels per file to include in statistics')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Optional file to save results')
    
    args = parser.parse_args()
    
    try:
        mean, std = calculate_dataset_statistics_robust(
            args.data_root, 
            args.sample_size, 
            args.min_valid_pixels
        )
        
        if mean is not None and std is not None:
            print("\n" + "="*60)
            print("FINAL DATASET STATISTICS")
            print("="*60)
            print(f"Mean per channel: {mean}")
            print(f"Std per channel:  {std}")
            
            # Check if we got valid statistics
            if not np.isnan(mean).any() and not np.isnan(std).any():
                print("\nFor use in albumentations:")
                print(f"albu.Normalize(")
                print(f"    mean={mean.tolist()},")
                print(f"    std={std.tolist()}")
                print(f")")
                
                if args.output_file:
                    with open(args.output_file, 'w') as f:
                        f.write("# Dataset normalization statistics (calculated from valid pixels only)\n")
                        f.write(f"MEAN = {mean.tolist()}\n")
                        f.write(f"STD = {std.tolist()}\n")
                        f.write(f"\n# Albumentations format:\n")
                        f.write(f"# albu.Normalize(mean={mean.tolist()}, std={std.tolist()})\n")
                    print(f"\nStatistics saved to: {args.output_file}")
                    
            else:
                print("\n⚠ WARNING: Some channels have no valid data!")
                print("This indicates severe data quality issues.")
        else:
            print("\n❌ Failed to calculate statistics - no usable data found!")
            print("\nThis suggests your TIFF files are completely corrupted.")
            print("Recommendations:")
            print("1. Check the original source files")
            print("2. Verify your data preprocessing pipeline")
            print("3. Re-generate the TIFF files from source data")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())