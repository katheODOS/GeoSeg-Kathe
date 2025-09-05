from .transform import *
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import matplotlib.patches as mpatches
from PIL import Image, ImageOps
import random
import rasterio


CLASSES = ('Forest land', 'Grassland', 'Cropland', 'Settlement', 'Seminatural Grassland')

PALETTE = [[250, 62, 119], [168, 232, 84], [242, 180, 92], [116, 116, 116], [255, 214, 33]]

ORIGIN_IMG_SIZE = (512, 512)
INPUT_IMG_SIZE = (512, 512)
TEST_IMG_SIZE = (512, 512)


def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    # multi-scale training and crop
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=0, nopad=False)])
    img, mask = crop_aug(img, mask)

    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


class BiodiversityTiffTrainDataset(Dataset):
    def __init__(self, data_root='data/Biodiversity_tiff/Train',
                 img_dir='images', mask_dir='masks',
                 img_suffix='.tif', mask_suffix='.png',
                 transform=train_aug, mosaic_ratio=0.25,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio:
            img, mask = self.load_img_and_mask(index)
            if self.transform:
                img, mask = self.transform(img, mask)
        else:
            img, mask = self.load_mosaic_img_and_mask(index)
            if self.transform:
                img, mask = self.transform(img, mask)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.img_ids[index]
        results = {'img': img, 'gt_semantic_seg': mask, 'img_id': img_id}
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        
        # Filter to only matching files
        img_ids = []
        for img_file in img_filename_list:
            if img_file.endswith('.tif'):
                img_name = str(img_file.split('.')[0])
                mask_file = img_name + self.mask_suffix
                if mask_file in mask_filename_list:
                    img_ids.append(img_name)
        
        print(f"Found {len(img_ids)} matching image-mask pairs")
        return img_ids

    def normalize_image(self, img_data):
        """Normalize image data to 0-1 range for each band"""
        normalized = np.zeros_like(img_data, dtype=np.float32)
        
        for i in range(img_data.shape[2]):
            band = img_data[:, :, i].astype(np.float32)
            # Remove nodata/invalid values for percentile calculation
            valid_pixels = band[~np.isnan(band)]
            valid_pixels = valid_pixels[valid_pixels != 0]  # Remove zeros
            
            if len(valid_pixels) > 0:
                # Use percentile normalization to handle outliers
                p2, p98 = np.percentile(valid_pixels, (2, 98))
                band = np.clip(band, p2, p98)
                band = (band - p2) / (p98 - p2) if p98 > p2 else band
            
            normalized[:, :, i] = band
        
        return normalized

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        
        # Load TIFF with rasterio to handle geospatial data properly
        try:
            with rasterio.open(img_name) as src:
                # Read all bands
                img_data = src.read()  # Shape: (bands, height, width)
                img_data = np.transpose(img_data, (1, 2, 0))  # Shape: (height, width, bands)
                
                # Handle nodata values
                nodata = src.nodata
                if nodata is not None:
                    img_data = np.where(img_data == nodata, 0, img_data)
                
                # Handle NaN values
                img_data = np.where(np.isnan(img_data), 0, img_data)
                
                # Normalize the image
                img_data = self.normalize_image(img_data)
                
                # Convert to 0-255 range and uint8
                img_data = (img_data * 255).clip(0, 255).astype(np.uint8)
                
                # Handle 4-band to 3-band conversion (take first 3 bands)
                if img_data.shape[2] == 4:
                    img_data = img_data[:, :, :3]  # Take RGB channels
                
                img = Image.fromarray(img_data)
                
        except Exception as e:
            print(f"Error reading TIFF {img_name}: {e}")
            # Fallback to PIL
            img = Image.open(img_name)
            if img.mode != 'RGB':
                img = img.convert('RGB')
        
        # Load mask
        mask = Image.open(mask_name).convert('L')
        
        # Ensure same size
        if img.size != mask.size:
            print(f"Size mismatch for {img_id}: img {img.size}, mask {mask.size}")
            # Resize mask to match image
            mask = mask.resize(img.size, Image.NEAREST)
        
        return img, mask

    def load_mosaic_img_and_mask(self, index):
        # Simplified mosaic - just return single image for now
        return self.load_img_and_mask(index)


class BiodiversityTiffTestDataset(Dataset):
    def __init__(self, data_root='data/Biodiversity_tiff/Test',
                 img_dir='images', img_suffix='.tif',
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir)

    def __getitem__(self, index):
        img = self.load_img(index)
        
        img_array = np.array(img)
        aug = albu.Normalize()(image=img_array)
        img = aug['image']
        
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img_id = self.img_ids[index]
        results = {
            'img': img, 
            'img_id': img_id,
            'img_type': 'tif'
        }
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        img_ids = [str(id.split('.')[0]) for id in img_filename_list if id.endswith('.tif')]
        return img_ids

    def load_img(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        
        try:
            with rasterio.open(img_name) as src:
                img_data = src.read()
                img_data = np.transpose(img_data, (1, 2, 0))
                
                # Handle nodata
                nodata = src.nodata
                if nodata is not None:
                    img_data = np.where(img_data == nodata, 0, img_data)
                
                # Handle NaN values
                img_data = np.where(np.isnan(img_data), 0, img_data)
                
                # Normalize
                img_data = self.normalize_image(img_data)
                img_data = (img_data * 255).clip(0, 255).astype(np.uint8)
                
                # Convert to 3-band
                if img_data.shape[2] == 4:
                    img_data = img_data[:, :, :3]
                
                return Image.fromarray(img_data)
        except:
            img = Image.open(img_name)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
    
    def normalize_image(self, img_data):
        """Same normalization as training dataset"""
        normalized = np.zeros_like(img_data, dtype=np.float32)
        
        for i in range(img_data.shape[2]):
            band = img_data[:, :, i].astype(np.float32)
            valid_pixels = band[~np.isnan(band)]
            valid_pixels = valid_pixels[valid_pixels != 0]
            
            if len(valid_pixels) > 0:
                p2, p98 = np.percentile(valid_pixels, (2, 98))
                band = np.clip(band, p2, p98)
                band = (band - p2) / (p98 - p2) if p98 > p2 else band
            
            normalized[:, :, i] = band
        
        return normalized


# Create validation dataset conditionally
try:
    val_path = osp.join('data', 'Biodiversity_tiff', 'Val')
    if os.path.exists(val_path):
        biodiversity_tiff_val_dataset = BiodiversityTiffTrainDataset(
            data_root=val_path,
            mosaic_ratio=0.0,
            transform=val_aug
        )
    else:
        print("Warning: Val directory not found, will use Train dataset for validation")
        biodiversity_tiff_val_dataset = None
except Exception as e:
    print(f"Warning: Could not create validation dataset: {e}")
    biodiversity_tiff_val_dataset = None


def show_img_mask_seg(seg_path, img_path, mask_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        mask = cv2.imread(f'{mask_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask).convert('P')
        mask.putpalette(np.array(PALETTE, dtype=np.uint8))
        mask = np.array(mask.convert('RGB'))
        img_id = str(seg_id.split('.')[0])+'.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE ' + img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(mask)
        ax[i, 1].set_title('Mask True ' + seg_id)
        ax[i, 2].set_axis_off()
        ax[i, 2].imshow(img_seg)
        ax[i, 2].set_title('Mask Predict ' + seg_id)
        ax[i, 2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_seg(seg_path, img_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        img_id = str(seg_id.split('.')[0])+'.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE '+img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(img_seg)
        ax[i, 1].set_title('Seg IMAGE '+seg_id)
        ax[i, 1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_mask(img, mask, img_id):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(np.array(PALETTE, dtype=np.uint8))
    mask = np.array(mask.convert('RGB'))
    ax1.imshow(img)
    ax1.set_title('RS IMAGE ' + str(img_id)+'.png')
    ax2.imshow(mask)
    ax2.set_title('Mask ' + str(img_id)+'.png')
    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')