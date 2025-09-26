#!/usr/bin/env python3
"""
Calculate dataset statistics from all TIFF files, handling NaN values properly
"""

import os
import sys
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import argparse

def load_tiff_normalized(filepath):
    """Load and normalize TIFF image, returning normalized data with NaN preservation"""
    try:
        with rasterio.open(filepath) as src:
            img = src.read()  # Shape: (bands, height, width)
            img = np.transpose(img, (1, 2, 0))  # Convert to (height, width, bands)
            
            # Handle nodata values
            if src.nodata is not None:
                img = np.where(img == src.nodata, np.nan, img)
            
            # Determine normalization strategy based on finite (non-NaN) values
            finite_mask = np.isfinite(img)
            if finite_mask.any():
                finite_data = img[finite_mask]
                data_max = finite_data.max()
                
                # Choose normalization based on data range
                if data_max <= 1 and finite_data.min() >= 0:
                    # Already normalized
                    img_norm = img.astype(np.float32)
                elif data_max <= 255:
                    # 8-bit data
                    img_norm = img.astype(np.float32) / 255.0
                elif data_max <= 10000:
                    # Typical satellite data scaling
                    img_norm = img.astype(np.float32) / 10000.0
                elif data_max <= 65535:
                    # 16-bit data
                    img_norm = img.astype(np.float32) / 65535.0
                else:
                    # Use 99th percentile normalization for extreme values
                    p99 = np.percentile(finite_data, 99)
                    img_norm = img.astype(np.float32) / p99
                
                # Clip values but preserve NaN
                img_norm = np.clip(img_norm, 0, 1)
                img_norm[~finite_mask] = np.nan
                
                return img_norm
            else:
                # All NaN - return as is
                return img.astype(np.float32)
                
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def calculate_statistics_from_all_files(data_root, max_files=None):
    """
    Calculate mean and std from all valid pixels across all files
    
    Args:
        data_root: Path to dataset root
        max_files: Maximum number of files to process (None = all)
    """
    data_root = Path(data_root)
    img_dir = data_root / 'images'
    
    if not img_dir.exists():
        raise ValueError(f"Images directory not found: {img_dir}")
    
    tiff_files = list(img_dir.glob('*.tif'))
    print(f"Found {len(tiff_files)} TIFF files")
    
    if max_files:
        tiff_files = tiff_files[:max_files]
        print(f"Processing first {len(tiff_files)} files")
    
    # Collect all valid pixel values for each channel
    print("Collecting valid pixel values from all files...")
    
    all_valid_pixels = [[] for _ in range(4)]  # 4 channels
    total_files_processed = 0
    total_files_with_valid_data = 0
    
    for tiff_path in tqdm(tiff_files, desc="Processing files"):
        img = load_tiff_normalized(tiff_path)
        
        if img is None or img.shape[2] != 4:
            continue
            
        total_files_processed += 1
        file_has_valid_data = False
        
        # Process each channel
        for c in range(4):
            channel_data = img[:, :, c]
            finite_mask = np.isfinite(channel_data)
            
            if finite_mask.any():
                valid_values = channel_data[finite_mask]
                all_valid_pixels[c].extend(valid_values.tolist())
                file_has_valid_data = True
        
        if file_has_valid_data:
            total_files_with_valid_data += 1
    
    print(f"\nProcessing Summary:")
    print(f"  Files processed: {total_files_processed}")
    print(f"  Files with valid data: {total_files_with_valid_data}")
    
    if total_files_with_valid_data == 0:
        print("ERROR: No files with valid data found!")
        return None, None
    
    # Calculate statistics
    means = []
    stds = []
    
    print(f"\nCalculating statistics per channel:")
    
    for c in range(4):
        if len(all_valid_pixels[c]) > 0:
            # Convert to numpy array for efficient calculation
            channel_array = np.array(all_valid_pixels[c], dtype=np.float32)
            
            mean_val = np.mean(channel_array)
            std_val = np.std(channel_array)
            
            means.append(mean_val)
            stds.append(std_val)
            
            print(f"  Channel {c}: {len(channel_array):,} valid pixels")
            print(f"    Mean: {mean_val:.6f}")
            print(f"    Std:  {std_val:.6f}")
            print(f"    Range: [{channel_array.min():.6f}, {channel_array.max():.6f}]")
            
            # Clear memory
            del channel_array
        else:
            print(f"  Channel {c}: No valid pixels found!")
            means.append(np.nan)
            stds.append(np.nan)
        
        # Clear memory
        all_valid_pixels[c].clear()
    
    return np.array(means), np.array(stds)

def calculate_statistics_streaming(data_root, max_files=None):
    """
    Memory-efficient streaming calculation of mean and std using Welford's algorithm
    """
    data_root = Path(data_root)
    img_dir = data_root / 'images'
    
    if not img_dir.exists():
        raise ValueError(f"Images directory not found: {img_dir}")
    
    tiff_files = list(img_dir.glob('*.tif'))
    print(f"Found {len(tiff_files)} TIFF files")
    
    if max_files:
        tiff_files = tiff_files[:max_files]
        print(f"Processing first {len(tiff_files)} files")
    
    # Initialize Welford's algorithm variables for each channel
    counts = np.zeros(4, dtype=np.int64)
    means = np.zeros(4, dtype=np.float64)
    m2s = np.zeros(4, dtype=np.float64)  # Sum of squares of deviations
    
    total_files_processed = 0
    total_files_with_valid_data = 0
    
    print("Streaming calculation (memory efficient)...")
    
    for tiff_path in tqdm(tiff_files, desc="Processing files"):
        img = load_tiff_normalized(tiff_path)
        
        if img is None or img.shape[2] != 4:
            continue
            
        total_files_processed += 1
        file_has_valid_data = False
        
        # Process each channel using Welford's online algorithm
        for c in range(4):
            channel_data = img[:, :, c]
            finite_mask = np.isfinite(channel_data)
            
            if finite_mask.any():
                valid_values = channel_data[finite_mask].astype(np.float64)
                file_has_valid_data = True
                
                # Welford's online algorithm for mean and variance
                for value in valid_values:
                    counts[c] += 1
                    delta = value - means[c]
                    means[c] += delta / counts[c]
                    delta2 = value - means[c]
                    m2s[c] += delta * delta2
        
        if file_has_valid_data:
            total_files_with_valid_data += 1
    
    print(f"\nProcessing Summary:")
    print(f"  Files processed: {total_files_processed}")
    print(f"  Files with valid data: {total_files_with_valid_data}")
    
    if total_files_with_valid_data == 0:
        print("ERROR: No files with valid data found!")
        return None, None
    
    # Calculate final statistics
    stds = np.sqrt(m2s / (counts - 1))  # Sample standard deviation
    
    print(f"\nFinal Statistics:")
    for c in range(4):
        if counts[c] > 0:
            print(f"  Channel {c}: {counts[c]:,} valid pixels")
            print(f"    Mean: {means[c]:.6f}")
            print(f"    Std:  {stds[c]:.6f}")
        else:
            print(f"  Channel {c}: No valid pixels")
            means[c] = np.nan
            stds[c] = np.nan
    
    return means.astype(np.float32), stds.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description='Calculate comprehensive dataset statistics')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to dataset root (contains "images" subdirectory)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (default: all)')
    parser.add_argument('--method', choices=['streaming', 'batch'], default='streaming',
                       help='Calculation method: streaming (memory efficient) or batch (faster)')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Optional file to save results')
    
    args = parser.parse_args()
    
    try:
        print(f"Using {args.method} calculation method...")
        
        if args.method == 'streaming':
            mean, std = calculate_statistics_streaming(args.data_root, args.max_files)
        else:
            mean, std = calculate_statistics_from_all_files(args.data_root, args.max_files)
        
        if mean is not None and std is not None and not np.isnan(mean).all():
            print("\n" + "="*60)
            print("FINAL DATASET NORMALIZATION STATISTICS")
            print("="*60)
            
            # Format for display
            mean_list = [f"{x:.6f}" if not np.isnan(x) else "NaN" for x in mean]
            std_list = [f"{x:.6f}" if not np.isnan(x) else "NaN" for x in std]
            
            print(f"Mean per channel: {mean_list}")
            print(f"Std per channel:  {std_list}")
            
            # Check if we have valid statistics for all channels
            valid_channels = ~np.isnan(mean) & ~np.isnan(std)
            
            if valid_channels.all():
                print("\n✓ All channels have valid statistics!")
                print("\nReplace the placeholder values in your config with:")
                print("="*50)
                print("# In your training config (e.g., config/biodiversity_tiff/ftunetformer.py)")
                print("albu.Normalize(")
                print(f"    mean={mean.tolist()},")
                print(f"    std={std.tolist()}")
                print(")")
                print("="*50)
                
                print(f"\nComparison with current placeholder values:")
                print(f"Current: mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5]")
                print(f"Actual:  mean={mean.tolist()}, std={std.tolist()}")
                
            elif valid_channels.any():
                valid_count = valid_channels.sum()
                print(f"\n⚠ Warning: Only {valid_count}/4 channels have valid statistics")
                print("You may have data quality issues in some channels.")
            
            # Save to file if requested
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write("# Dataset normalization statistics calculated from all valid pixels\n")
                    f.write(f"# Calculated using {args.method} method\n")
                    f.write(f"# Files processed: {args.max_files or 'all'}\n\n")
                    f.write(f"MEAN = {mean.tolist()}\n")
                    f.write(f"STD = {std.tolist()}\n\n")
                    f.write("# Use in albumentations:\n")
                    f.write(f"# albu.Normalize(mean={mean.tolist()}, std={std.tolist()})\n")
                print(f"\nStatistics saved to: {args.output_file}")
                
        else:
            print("\n❌ Failed to calculate valid statistics!")
            print("This indicates severe data quality issues across your dataset.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())