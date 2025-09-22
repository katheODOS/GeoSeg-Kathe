import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

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
    #PLEASE NOTE THAT MASK_CONVERT IS OFF FROM SETTLEMENT ONWARDS. IN UNETKATHE WE HAVE CLASS 4 (WATER BODY) AND CLASS 6 (OTHER).
    #when adding these classes,m if the geoseg proves useful, please add them to the label2rgb function with their appropriate associated number. ALSO AD THEM TO 'datasets/biodiversity_dataset.py'
    return mask_rgb


def img_writer(inp):
    (mask,  mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label2rgb(mask)
        mask_tif = cv2.cvtColor(mask_tif, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-w", "--checkpoint_path", type=Path, required=True, help="Path to specific checkpoint file (.ckpt)")
    arg("-i", "--input_path", type=Path, help="Path to custom input directory (overrides config test dataset)")
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"]) ## lr is flip TTA, d4 is multi-scale TTA
    arg("--rgb", help="whether output rgb masks", action='store_true')
    arg("--val", help="whether eval validation set", action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    # Check if the checkpoint file exists
    if not args.checkpoint_path.exists():
        print(f"Error: Checkpoint file {args.checkpoint_path} does not exist!")
        return
    
    # Load model from the specified checkpoint path instead of config default
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    try:
        # Create model first
        model = Supervision_Train(config)
        
        # Try loading with weights_only=False first
        try:
            checkpoint = torch.load(str(args.checkpoint_path), map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Failed to load with weights_only=False: {str(e)}")
            try:
                # Try with safe_globals context manager
                with torch.serialization.safe_globals(['numpy._core.multiarray.scalar']):
                    checkpoint = torch.load(str(args.checkpoint_path), map_location='cpu')
            except Exception as e:
                print(f"Failed to load with safe_globals: {str(e)}")
                # Last attempt with default settings
                checkpoint = torch.load(str(args.checkpoint_path), map_location='cpu')
        
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint is not a dictionary")
            
        # Try different possible state dict keys
        state_dict = None
        for key in ['state_dict', 'model_state', 'model_state_dict', 'net_state_dict']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"Found state dict under key: {key}")
                break
        
        if state_dict is None:
            # If no known keys found, check if the checkpoint itself is the state dict
            if any(k.startswith('net.') or k.startswith('model.') for k in checkpoint.keys()):
                state_dict = checkpoint
                print("Using checkpoint directly as state dict")
            else:
                raise ValueError("Could not find state dict in checkpoint")
        
        # Fix key names if needed
        fixed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key.replace('model.', 'net.', 1)
                fixed_state_dict[new_key] = value
            else:
                fixed_state_dict[key] = value
        
        # Try loading with both strict and non-strict
        try:
            model.load_state_dict(fixed_state_dict, strict=True)
            print("Successfully loaded checkpoint with strict=True")
        except Exception:
            model.load_state_dict(fixed_state_dict, strict=False)
            print("Successfully loaded checkpoint with strict=False")
            
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return
    
    model.cuda()
    model.eval()
    
    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                # tta.VerticalFlip(),
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.test_dataset
    if args.val:
        evaluator = Evaluator(num_class=config.num_classes)
        evaluator.reset()
        test_dataset = config.val_dataset
    
    # Override with custom input directory if provided
    if args.input_path:
        if not args.input_path.exists():
            print(f"Error: Input directory {args.input_path} does not exist!")
            return
        print(f"Using custom input directory: {args.input_path}")
        
        # Determine the correct data_root path
        # If user provided the full path to images_png, get the parent directories
        if args.input_path.name == 'images_png' and args.input_path.parent.name == 'Rural':
            data_root = str(args.input_path.parent.parent)  # Go up two levels to get Test_2
        else:
            data_root = str(args.input_path)
        
        # Create a custom test dataset with the specified directory
        from geoseg.datasets.biodiversity_dataset import BiodiversityTestDataset
        test_dataset = BiodiversityTestDataset(data_root=data_root)
        print(f"Found {len(test_dataset)} images to process")

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            raw_predictions = model(input['img'].cuda())

            image_ids = input["img_id"]
            if args.val:
                masks_true = input['gt_semantic_seg']

            img_type = input['img_type']

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                mask_name = image_ids[i]
                mask_type = img_type[i]
                if args.val:
                    if not os.path.exists(os.path.join(args.output_path, mask_type)):
                        os.mkdir(os.path.join(args.output_path, mask_type))
                    evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
                    results.append((mask, str(args.output_path / mask_type / mask_name), args.rgb))
                else:
                    results.append((mask, str(args.output_path / mask_name), args.rgb))
    if args.val:
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        OA = evaluator.OA()
        for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
            print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
        print('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA))

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))


if __name__ == "__main__":
    main()

    # sample usage: python GeoSeg-Kathe/biodiversity_test_direct.py -c GeoSeg-Kathe/config/biodiversity/unetformer.py -w "C:\Users\Admin\anaconda3\envs\GeoSeg-Kathe\model_weights\biodiversity\biodiversityL5e-04BL1e-04W1e-02BW1e-02B16E75S1.00\last.ckpt" -i "C:\Users\Admin\anaconda3\envs\GeoSeg-Kathe\data\Biodiversity\Test_2\Rural\images_png" -o "C:\Users\Admin\anaconda3\envs\GeoSeg-Kathe\predictions\best\config3" --rgb