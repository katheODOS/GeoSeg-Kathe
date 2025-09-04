import os
import numpy as np
from PIL import Image
import rasterio
import pandas as pd
from pathlib import Path
import cv2
from collections import defaultdict

class DatasetDiagnostics:
    def __init__(self, data_root, img_dir='images', mask_dir='masks', 
                 img_suffix='.tif', mask_suffix='.png'):
        self.data_root = Path(data_root)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.issues = []
        self.stats = defaultdict(list)
        
    def check_file_pairs(self):
        """Check if image and mask files are properly paired"""
        img_path = self.data_root / self.img_dir
        mask_path = self.data_root / self.mask_dir
        
        print(f"Checking files in:")
        print(f"  Images: {img_path}")
        print(f"  Masks:  {mask_path}")
        
        if not img_path.exists():
            print(f"ERROR: Image directory doesn't exist: {img_path}")
            return
        if not mask_path.exists():
            print(f"ERROR: Mask directory doesn't exist: {mask_path}")
            return
        
        # Get all image and mask files
        img_files = set([f.stem for f in img_path.glob(f"*{self.img_suffix}")])
        mask_files = set([f.stem for f in mask_path.glob(f"*{self.mask_suffix}")])
        
        print(f"\nFound {len(img_files)} image files and {len(mask_files)} mask files")
        
        # Check for missing pairs
        missing_masks = img_files - mask_files
        missing_images = mask_files - img_files
        
        if missing_masks:
            print(f"\nImages without masks ({len(missing_masks)}):")
            for name in sorted(missing_masks):
                print(f"  - {name}{self.img_suffix}")
                
        if missing_images:
            print(f"\nMasks without images ({len(missing_images)}):")
            for name in sorted(missing_images):
                print(f"  - {name}{self.mask_suffix}")
        
        # Return matched pairs
        matched_pairs = img_files & mask_files
        print(f"\nMatched pairs: {len(matched_pairs)}")
        return sorted(matched_pairs)
    
    def analyze_single_image(self, img_id):
        """Analyze a single image-mask pair"""
        img_path = self.data_root / self.img_dir / f"{img_id}{self.img_suffix}"
        mask_path = self.data_root / self.mask_dir / f"{img_id}{self.mask_suffix}"
        
        result = {
            'id': img_id,
            'img_exists': img_path.exists(),
            'mask_exists': mask_path.exists(),
            'img_readable': False,
            'mask_readable': False,
            'size_match': False,
            'img_info': {},
            'mask_info': {},
            'issues': []
        }
        
        # Analyze image
        if result['img_exists']:
            try:
                # Try with rasterio first (for geospatial data)
                with rasterio.open(img_path) as src:
                    result['img_info'] = {
                        'format': 'GeoTIFF',
                        'shape': (src.height, src.width, src.count),
                        'dtype': src.dtypes[0],
                        'crs': str(src.crs) if src.crs else None,
                        'nodata': src.nodata,
                        'bounds': src.bounds,
                        'transform': src.transform,
                        'bands': src.count,
                        'band_dtypes': src.dtypes
                    }
                    
                    # Check for actual data
                    sample_data = src.read(1, window=rasterio.windows.Window(0, 0, min(100, src.width), min(100, src.height)))
                    result['img_info']['has_data'] = not np.all(sample_data == (src.nodata or 0))
                    result['img_info']['data_range'] = (sample_data.min(), sample_data.max())
                    
                result['img_readable'] = True
                
            except Exception as e:
                result['issues'].append(f"Rasterio failed: {e}")
                # Try with PIL as fallback
                try:
                    with Image.open(img_path) as img:
                        result['img_info'] = {
                            'format': img.format,
                            'mode': img.mode,
                            'size': img.size,  # (width, height)
                            'shape': (img.size[1], img.size[0], len(img.getbands())),  # (height, width, channels)
                        }
                    result['img_readable'] = True
                except Exception as e2:
                    result['issues'].append(f"PIL also failed: {e2}")
        
        # Analyze mask
        if result['mask_exists']:
            try:
                with Image.open(mask_path) as mask:
                    mask_array = np.array(mask)
                    unique_values = np.unique(mask_array)
                    
                    result['mask_info'] = {
                        'format': mask.format,
                        'mode': mask.mode,
                        'size': mask.size,  # (width, height)
                        'shape': mask_array.shape,  # (height, width) or (height, width, channels)
                        'unique_values': unique_values.tolist(),
                        'num_classes': len(unique_values),
                        'data_range': (mask_array.min(), mask_array.max())
                    }
                result['mask_readable'] = True
                
            except Exception as e:
                result['issues'].append(f"Mask reading failed: {e}")
        
        # Check size compatibility
        if result['img_readable'] and result['mask_readable']:
            img_shape = result['img_info'].get('shape', result['img_info'].get('size', (0, 0)))
            mask_shape = result['mask_info']['shape']
            
            # Convert to (height, width) format for comparison
            if 'size' in result['img_info']:  # PIL format: (width, height)
                img_hw = (result['img_info']['size'][1], result['img_info']['size'][0])
            else:  # Rasterio format: (height, width, channels)
                img_hw = (img_shape[0], img_shape[1])
            
            if len(mask_shape) == 2:  # Grayscale mask
                mask_hw = mask_shape
            else:  # Multi-channel mask
                mask_hw = mask_shape[:2]
            
            result['size_match'] = img_hw == mask_hw
            result['img_hw'] = img_hw
            result['mask_hw'] = mask_hw
            
            if not result['size_match']:
                result['issues'].append(f"Size mismatch: img {img_hw} vs mask {mask_hw}")
        
        return result
    
    def run_full_diagnostics(self):
        """Run complete diagnostics on the dataset"""
        print("="*60)
        print("DATASET DIAGNOSTICS")
        print("="*60)
        
        # Check file pairing
        matched_pairs = self.check_file_pairs()
        
        if not matched_pairs:
            print("No matched pairs found. Exiting.")
            return
        
        print(f"\nAnalyzing {len(matched_pairs)} image-mask pairs...")
        print("-"*60)
        
        # Analyze each pair
        all_results = []
        problem_files = []
        
        for i, img_id in enumerate(matched_pairs):
            if i % 10 == 0:
                print(f"Processing {i+1}/{len(matched_pairs)}...")
            
            result = self.analyze_single_image(img_id)
            all_results.append(result)
            
            # Collect statistics
            if result['img_readable']:
                img_info = result['img_info']
                if 'shape' in img_info:
                    self.stats['img_shapes'].append(img_info['shape'])
                    self.stats['img_bands'].append(img_info['shape'][2] if len(img_info['shape']) > 2 else 1)
                if 'dtype' in img_info:
                    self.stats['img_dtypes'].append(str(img_info['dtype']))
            
            if result['mask_readable']:
                mask_info = result['mask_info']
                self.stats['mask_shapes'].append(mask_info['shape'])
                self.stats['mask_classes'].append(mask_info['num_classes'])
                self.stats['unique_values'].extend(mask_info['unique_values'])
            
            # Track problematic files
            if result['issues'] or not result['size_match']:
                problem_files.append(result)
        
        # Print summary
        self.print_summary(all_results, problem_files)
        
        return all_results, problem_files
    
    def print_summary(self, all_results, problem_files):
        """Print diagnostic summary"""
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        total_files = len(all_results)
        readable_imgs = sum(1 for r in all_results if r['img_readable'])
        readable_masks = sum(1 for r in all_results if r['mask_readable'])
        size_matches = sum(1 for r in all_results if r['size_match'])
        
        print(f"Total files analyzed: {total_files}")
        print(f"Readable images: {readable_imgs}/{total_files}")
        print(f"Readable masks: {readable_masks}/{total_files}")
        print(f"Size matches: {size_matches}/{total_files}")
        print(f"Problem files: {len(problem_files)}")
        
        # Image statistics
        if self.stats['img_shapes']:
            unique_shapes = list(set(tuple(s) for s in self.stats['img_shapes']))
            print(f"\nImage shapes found: {len(unique_shapes)}")
            for shape in sorted(unique_shapes):
                count = sum(1 for s in self.stats['img_shapes'] if tuple(s) == shape)
                print(f"  {shape}: {count} files")
        
        if self.stats['img_bands']:
            unique_bands = list(set(self.stats['img_bands']))
            print(f"\nBand counts: {sorted(unique_bands)}")
            for bands in sorted(unique_bands):
                count = self.stats['img_bands'].count(bands)
                print(f"  {bands} bands: {count} files")
        
        if self.stats['img_dtypes']:
            unique_dtypes = list(set(self.stats['img_dtypes']))
            print(f"\nImage dtypes: {unique_dtypes}")
        
        # Mask statistics  
        if self.stats['mask_shapes']:
            unique_mask_shapes = list(set(self.stats['mask_shapes']))
            print(f"\nMask shapes found: {len(unique_mask_shapes)}")
            for shape in sorted(unique_mask_shapes):
                count = sum(1 for s in self.stats['mask_shapes'] if s == shape)
                print(f"  {shape}: {count} files")
        
        if self.stats['unique_values']:
            unique_classes = sorted(list(set(self.stats['unique_values'])))
            print(f"\nMask class values found: {unique_classes}")
        
        # Problem files details
        if problem_files:
            print(f"\nPROBLEM FILES ({len(problem_files)}):")
            print("-"*40)
            for result in problem_files[:20]:  # Show first 20
                print(f"\n{result['id']}:")
                if not result['size_match']:
                    print(f"  Size mismatch: img {result.get('img_hw')} vs mask {result.get('mask_hw')}")
                for issue in result['issues']:
                    print(f"  Issue: {issue}")
            
            if len(problem_files) > 20:
                print(f"\n... and {len(problem_files) - 20} more problem files")
    
    def save_detailed_report(self, results, filename='dataset_diagnostics.csv'):
        """Save detailed results to CSV"""
        rows = []
        for result in results:
            row = {
                'file_id': result['id'],
                'img_readable': result['img_readable'],
                'mask_readable': result['mask_readable'],
                'size_match': result['size_match'],
                'issues_count': len(result['issues']),
                'issues': '; '.join(result['issues']),
            }
            
            # Add image info
            if result['img_readable']:
                img_info = result['img_info']
                row.update({
                    'img_format': img_info.get('format', ''),
                    'img_shape': str(img_info.get('shape', '')),
                    'img_dtype': str(img_info.get('dtype', '')),
                    'img_bands': img_info.get('bands', img_info.get('shape', [0,0,0])[2] if len(img_info.get('shape', [])) > 2 else 1),
                    'img_has_data': img_info.get('has_data', ''),
                    'img_nodata': img_info.get('nodata', ''),
                })
            
            # Add mask info
            if result['mask_readable']:
                mask_info = result['mask_info']
                row.update({
                    'mask_format': mask_info.get('format', ''),
                    'mask_shape': str(mask_info.get('shape', '')),
                    'mask_classes': mask_info.get('num_classes', ''),
                    'mask_values': str(mask_info.get('unique_values', '')),
                    'mask_range': str(mask_info.get('data_range', '')),
                })
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"\nDetailed report saved to: {filename}")


def run_diagnostics(data_root, img_dir='images', mask_dir='masks', 
                   img_suffix='.tif', mask_suffix='.png'):
    """Convenience function to run diagnostics"""
    diagnostics = DatasetDiagnostics(data_root, img_dir, mask_dir, img_suffix, mask_suffix)
    results, problem_files = diagnostics.run_full_diagnostics()
    
    # Save detailed report
    report_name = f"diagnostics_{Path(data_root).name}.csv"
    diagnostics.save_detailed_report(results, report_name)
    
    return results, problem_files


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose dataset issues')
    parser.add_argument('--data-root', required=True, help='Path to dataset root')
    parser.add_argument('--img-dir', default='images', help='Image directory name')
    parser.add_argument('--mask-dir', default='masks', help='Mask directory name')
    parser.add_argument('--img-suffix', default='.tif', help='Image file suffix')
    parser.add_argument('--mask-suffix', default='.png', help='Mask file suffix')
    
    args = parser.parse_args()
    
    results, problems = run_diagnostics(
        args.data_root, args.img_dir, args.mask_dir, 
        args.img_suffix, args.mask_suffix
    )
    
    print(f"\nDiagnostics complete! Found {len(problems)} problematic files.")