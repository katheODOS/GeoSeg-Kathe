import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# FIX PYTORCH 2.6+ LOADING ISSUES FIRST
import torch
from torch.serialization import safe_globals, add_safe_globals
import numpy as np
try:
    import numpy._core.multiarray as multiarray
    add_safe_globals([multiarray.scalar, np.dtype, np.float64])
    print("✓ Fixed PyTorch 2.6+ loading compatibility")
except Exception as e:
    print(f"Warning: Could not fix PyTorch loading: {e}")

import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
from geoseg.datasets.biodiversity_tiff_dataset import BiodiversityTiffTestDataset
from tools.cfg import py2cfg

from torch import nn
from torch.utils.data import DataLoader
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
        mask_name_tif = mask_id + '.tif'
        mask_tif = label2rgb(mask)
        mask_tif = cv2.cvtColor(mask_tif, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_name_tif, mask_tif, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    else:
        mask_tif = mask.astype(np.uint8)
        mask_name_tif = mask_id + '.tif'
        cv2.imwrite(mask_name_tif, mask_tif, [cv2.IMWRITE_TIFF_COMPRESSION, 1])


def safe_load_checkpoint(checkpoint_path, config=None):
    """Safely load checkpoint with PyTorch 2.6+ compatibility"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Ensure all numpy types are added to safe_globals list
    try:
        add_safe_globals([
            multiarray.scalar, 
            np.dtype, 
            np.float64,
            np.dtypes.Float64DType,
            np.float32, 
            np.int64, 
            np.int32
        ])
    except Exception as e:
        print(f"Warning: Could not add all numpy types to safe_globals: {e}")
    
    try:
        # Method 1: Try with Lightning's load_from_checkpoint (handles the PyTorch fix internally)
        if config:
            model = Supervision_Train.load_from_checkpoint(
                str(checkpoint_path), 
                config=config,
                map_location='cpu'
            )
        else:
            model = Supervision_Train.load_from_checkpoint(
                str(checkpoint_path),
                map_location='cpu'
            )
        print("✓ Model loaded successfully with Lightning")
        return model
        
    except Exception as e:
        if "weights_only" in str(e) or "UnpicklingError" in str(e):
            print(f"PyTorch loading issue detected. Trying manual fix...")
            
            # Method 2: Manual loading with safe_globals
            try:
                with safe_globals([multiarray.scalar, np.dtype, np.float64]):
                    if config:
                        model = Supervision_Train.load_from_checkpoint(
                            str(checkpoint_path), 
                            config=config,
                            map_location='cpu'
                        )
                    else:
                        model = Supervision_Train.load_from_checkpoint(
                            str(checkpoint_path),
                            map_location='cpu'
                        )
                print("✓ Model loaded with safe_globals fix")
                return model
            except Exception as e2:
                print(f"Safe globals method failed: {e2}")
                
                # Method 3: Last resort - try with weights_only=False (security risk but trusted checkpoint)
                try:
                    print("Attempting to load with weights_only=False (only use with trusted checkpoints)...")
                    if config:
                        checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
                        model = Supervision_Train(config)
                        model.load_state_dict(checkpoint['state_dict'], strict=False)
                    else:
                        model = Supervision_Train.load_from_checkpoint(
                            str(checkpoint_path),
                            map_location='cpu',
                            _load_weights_only=False  # Lightning's parameter
                        )
                    print("✓ Model loaded with weights_only=False")
                    return model
                except Exception as e3:
                    print(f"Final attempt failed: {e3}")
        
        print(f"All loading methods failed: {e}")
        raise e


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-w", "--checkpoint_path", type=Path, required=True, help="Path to specific checkpoint file (.ckpt)")
    arg("-i", "--input_path", type=Path, help="Path to custom input directory (overrides config test dataset)")
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("--rgb", help="whether output rgb masks", action='store_true')
    arg("--val", help="whether eval validation set", action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    args.output_path.mkdir(exist_ok=True, parents=True)

    # Check if the checkpoint file exists
    if not args.checkpoint_path.exists():
        print(f"Error: Checkpoint file {args.checkpoint_path} does not exist!")
        return
    
    # Load config
    config = None
    if args.config_path and args.config_path.exists():
        print(f"Loading config from: {args.config_path}")
        try:
            config = py2cfg(args.config_path)
            print("✓ Config loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    
    # Load model with PyTorch 2.6+ fix
    try:
        model = safe_load_checkpoint(args.checkpoint_path, config)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Test 4-channel input
    print("Testing 4-channel input compatibility...")
    try:
        dummy_input = torch.randn(1, 4, 256, 256)
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'net'):
                test_output = model.net(dummy_input)
            else:
                test_output = model(dummy_input)
        print("✓ Model accepts 4-channel input!")
    except Exception as e:
        print(f"❌ Model failed 4-channel test: {e}")
        print("This suggests the model wasn't properly configured for 4-channel input")
        return
    
    model.cuda()
    model.eval()
    
    # Setup TTA
    if args.tta == "lr":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip()
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False),
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)

    # Setup dataset
    if args.input_path:
        if not args.input_path.exists():
            print(f"Error: Input directory {args.input_path} does not exist!")
            return
        print(f"Using custom input directory: {args.input_path}")
        test_dataset = BiodiversityTiffTestDataset(data_root=str(args.input_path))
        print(f"Found {len(test_dataset)} images to process")
    elif config and hasattr(config, 'test_dataset'):
        test_dataset = config.test_dataset
    else:
        print("Error: Please provide input directory with -i argument or ensure config has test_dataset")
        return

    if len(test_dataset) == 0:
        print("Error: No images found in dataset!")
        return

    # Setup validation if requested
    if args.val:
        from tools.metric import Evaluator
        evaluator = Evaluator(num_class=6)
        evaluator.reset()

    # Process images
    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        
        results = []
        for batch_idx, input in enumerate(tqdm(test_loader, desc="Processing images")):
            # Debug first batch
            if batch_idx == 0:
                print(f"\nFirst batch debug info:")
                print(f"  Keys: {input.keys()}")
                print(f"  Image shape: {input['img'].shape}")
                print(f"  Image dtype: {input['img'].dtype}")
                print(f"  Image device: {input['img'].device}")
                print(f"  Image range: [{input['img'].min():.3f}, {input['img'].max():.3f}]")
                
                if input['img'].shape[1] != 4:
                    print(f"❌ WARNING: Expected 4 channels, got {input['img'].shape[1]}")
                    print("This indicates the dataset is not loading 4-channel images!")
                else:
                    print("✓ Dataset is correctly loading 4-channel images")
            
            try:
                # Get predictions
                img_cuda = input['img'].cuda()
                if hasattr(model, 'net'):
                    raw_predictions = model.net(img_cuda)
                else:
                    raw_predictions = model(img_cuda)
                
                # Handle different output formats
                if isinstance(raw_predictions, (list, tuple)):
                    raw_predictions = raw_predictions[0]  # Take main output
                
                image_ids = input["img_id"]
                img_type = input.get('img_type', 'tif')
                
                if args.val and 'gt_semantic_seg' in input:
                    masks_true = input['gt_semantic_seg']

                raw_predictions = nn.Softmax(dim=1)(raw_predictions)
                predictions = raw_predictions.argmax(dim=1)

                for i in range(predictions.shape[0]):
                    mask = predictions[i].cpu().numpy()
                    mask_name = image_ids[i] if isinstance(image_ids[i], str) else str(image_ids[i])
                    
                    if args.val and 'gt_semantic_seg' in input:
                        evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
                    
                    results.append((mask, str(args.output_path / mask_name), args.rgb))
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                if batch_idx == 0:  # Show full error for first batch
                    import traceback
                    traceback.print_exc()
                continue

    # Print validation results if requested
    if args.val and 'evaluator' in locals():
        try:
            iou_per_class = evaluator.Intersection_over_Union()
            f1_per_class = evaluator.F1()
            OA = evaluator.OA()
            
            classes = ['Background', 'Forest land', 'Grassland', 'Cropland', 'Settlement', 'Seminatural Grassland']
            
            print("\nValidation Results:")
            for class_name, class_iou, class_f1 in zip(classes, iou_per_class, f1_per_class):
                print('F1_{}:{:.4f}, IOU_{}:{:.4f}'.format(class_name, class_f1, class_name, class_iou))
            print('Overall - F1:{:.4f}, mIOU:{:.4f}, OA:{:.4f}'.format(
                np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA))
        except Exception as e:
            print(f"Validation computation failed: {e}")

    # Write results
    if results:
        print(f"\nWriting {len(results)} prediction masks...")
        t0 = time.time()
        mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
        t1 = time.time()
        img_write_time = t1 - t0
        print('Images writing took: {:.2f} seconds'.format(img_write_time))
        print(f"Results saved to: {args.output_path}")
    else:
        print("❌ No results generated!")


if __name__ == "__main__":
    main()