import argparse
from pathlib import Path
import glob
from PIL import Image
import ttach as tta
import cv2
import numpy as np
import torch
import albumentations as albu
from catalyst.dl import SupervisedRunner
from skimage.morphology import remove_small_holes, remove_small_objects
from tools.cfg import py2cfg
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_supervision import *
import random
import os


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def building_to_rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 0]
    return mask_rgb


def pv2rgb(mask):  # Potsdam and vaihingen
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def landcoverai_to_rgb(mask):
    w, h = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(w, h, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [233, 193, 133]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 255, 0]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def uavid2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [128, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 64, 128]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 128, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [128, 128, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [64, 0, 128]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [192, 0, 192]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [64, 64, 0]
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [0, 0, 0]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb

def biodiversity2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    # BGR format for cv2.imwrite (note: swapped R and B values)
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [210, 246, 11]   # BGR: ignore index
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [119, 62, 250]  # BGR: forestland  
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [84, 232, 168]  # BGR: grassland
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [92, 180, 242]  # BGR: cropland
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [116, 116, 116] # BGR: settlement (gray = same)
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [33, 214, 255]  # BGR: seminatural grassland
    return mask_rgb

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--image_path", type=Path, required=True, help="Path to  huge image folder")
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("-ph", "--patch-height", help="height of patch size", type=int, default=512)
    arg("-pw", "--patch-width", help="width of patch size", type=int, default=512)
    arg("-b", "--batch-size", help="batch size", type=int, default=2)
    arg("-d", "--dataset", help="dataset", default="biodiversity", choices=["pv", "landcoverai", "uavid", "building", "biodiversity"])
    return parser.parse_args()


def get_img_padded(image, patch_size):
    oh, ow = image.shape[0], image.shape[1]
    rh, rw = oh % patch_size[0], ow % patch_size[1]

    width_pad = 0 if rw == 0 else patch_size[1] - rw
    height_pad = 0 if rh == 0 else patch_size[0] - rh
    # print(oh, ow, rh, rw, height_pad, width_pad)
    h, w = oh + height_pad, ow + width_pad

    pad = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right',
                           border_mode=0, value=[0, 0, 0])(image=image)
    img_pad = pad['image']
    return img_pad, height_pad, width_pad


class InferenceDataset(Dataset):
    def __init__(self, tile_list=None, transform=albu.Normalize()):
        self.tile_list = tile_list
        self.transform = transform

    def __getitem__(self, index):
        img = self.tile_list[index]
        img_id = index
        aug = self.transform(image=img)
        img = aug['image']
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        results = dict(img_id=img_id, img=img)
        return results

    def __len__(self):
        return len(self.tile_list)


def make_dataset_for_one_huge_image(img_path, patch_size):
    """
    Process a single large image by splitting it into patches
    """
    print(f"Loading image from: {img_path}")
    
    # Check if file exists
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    # Load the image
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image from: {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Original image shape: {img.shape}")
    
    tile_list = []
    image_pad, height_pad, width_pad = get_img_padded(img.copy(), patch_size)
    print(f"Padded image shape: {image_pad.shape}")

    output_height, output_width = image_pad.shape[0], image_pad.shape[1]
    
    # Calculate number of patches
    num_patches_h = output_height // patch_size[0]
    num_patches_w = output_width // patch_size[1]
    total_patches = num_patches_h * num_patches_w
    print(f"Will create {total_patches} patches ({num_patches_h}x{num_patches_w})")

    for x in range(0, output_height, patch_size[0]):
        for y in range(0, output_width, patch_size[1]):
            image_tile = image_pad[x:x+patch_size[0], y:y+patch_size[1]]
            tile_list.append(image_tile)

    dataset = InferenceDataset(tile_list=tile_list)
    return dataset, width_pad, height_pad, output_width, output_height, image_pad, img.shape


def main():
    args = get_args()
    seed_everything(322)
    patch_size = (args.patch_height, args.patch_width)
    
    print(f"Processing image: {args.image_path}")
    print(f"Patch size: {patch_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_path}")
    
    # Load config and model
    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name+'.ckpt'), 
        config=config
    )

    model.cuda()
    model.eval()

    # Set up TTA if requested
    if args.tta == "lr":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip()
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.Scale(scales=[0.75, 1, 1.25, 1.5, 1.75]),
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)

    # Create output directory
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Process the single image specified by -i parameter
    img_path = str(args.image_path)
    img_name = os.path.basename(img_path)
    
    try:
        # Create dataset from the single image
        dataset, width_pad, height_pad, output_width, output_height, img_pad, img_shape = \
            make_dataset_for_one_huge_image(img_path, patch_size)
        
        print(f"Created dataset with {len(dataset)} patches")
        
        # Initialize output mask
        output_mask = np.zeros(shape=(output_height, output_width), dtype=np.uint8)
        output_tiles = []
        
        # Process patches
        with torch.no_grad():
            dataloader = DataLoader(
                dataset=dataset, 
                batch_size=args.batch_size,
                drop_last=False, 
                shuffle=False,
                num_workers=0  # Set to 0 to avoid multiprocessing issues
            )
            
            print(f"Processing {len(dataloader)} batches...")
            
            for batch_idx, input_batch in enumerate(tqdm(dataloader, desc="Processing patches")):
                # Get predictions
                raw_predictions = model(input_batch['img'].cuda())
                raw_predictions = nn.Softmax(dim=1)(raw_predictions)
                predictions = raw_predictions.argmax(dim=1)
                image_ids = input_batch['img_id']

                # Store results
                for i in range(predictions.shape[0]):
                    mask = predictions[i].cpu().numpy()
                    output_tiles.append((mask, image_ids[i].cpu().numpy()))

        print("Reconstructing full image from patches...")
        
        # Reconstruct the full mask from patches
        k = 0
        for m in range(0, output_height, patch_size[0]):
            for n in range(0, output_width, patch_size[1]):
                if k < len(output_tiles):
                    output_mask[m:m + patch_size[0], n:n + patch_size[1]] = output_tiles[k][0]
                    k += 1

        # Remove padding to get back to original size
        output_mask = output_mask[:img_shape[0], :img_shape[1]]
        print(f"Final mask shape: {output_mask.shape}")

        # Convert to RGB based on dataset type
        if args.dataset == 'landcoverai':
            output_mask_rgb = landcoverai_to_rgb(output_mask)
        elif args.dataset == 'pv':
            output_mask_rgb = pv2rgb(output_mask)
        elif args.dataset == 'uavid':
            output_mask_rgb = uavid2rgb(output_mask)
        elif args.dataset == 'building':
            output_mask_rgb = building_to_rgb(output_mask)
        elif args.dataset == 'biodiversity':
            output_mask_rgb = biodiversity2rgb(output_mask)
        else:
            output_mask_rgb = output_mask

        # Save the result
        output_path = os.path.join(args.output_path, img_name)
        cv2.imwrite(output_path, output_mask_rgb)
        print(f"Saved result to: {output_path}")
        
        # Print some statistics
        unique_classes = np.unique(output_mask)
        print(f"Classes found in prediction: {unique_classes}")
        
    except Exception as e:
        print(f"Error processing image {img_path}: {str(e)}")
        raise


if __name__ == "__main__":
    main()