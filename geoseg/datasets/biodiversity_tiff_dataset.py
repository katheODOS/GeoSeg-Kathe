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
import cv2
import tifffile


CLASSES = ('Forest land', 'Grassland', 'Cropland', 'Settlement', 'Seminatural Grassland')

PALETTE = [[250, 62, 119], [168, 232, 84], [242, 180, 92], [116, 116, 116], [255, 214, 33]]


ORIGIN_IMG_SIZE = (512, 512)
INPUT_IMG_SIZE = (512, 512)
TEST_IMG_SIZE = (512, 512)


def get_training_transform():
    train_transform = [
        # albu.Resize(height=1024, width=1024),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
        # albu.RandomRotate90(p=0.5),
        # albu.OneOf([
        #     albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
        #     albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25)
        # ], p=0.25),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)

def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)

def train_aug(img, mask):
    # Convert PIL images to numpy arrays first
    img = np.array(img)
    mask = np.array(mask)
    
    # Ensure mask is 2D
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    # Ensure both are same size
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Convert back to PIL for crop_aug
    img = Image.fromarray(img)
    mask = Image.fromarray(mask)
    
    # multi-scale training and crop
    crop_aug = Compose([
        RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
        SmartCropV1(crop_size=256, max_ratio=0.75, ignore_index=0, nopad=False)
    ])
    
    try:
        img, mask = crop_aug(img, mask)
    except Exception as e:
        print(f"Error in crop_aug: {e}")
        print(f"Image size: {img.size}, Mask size: {mask.size}")
        raise e

    # Convert to numpy for albumentations
    img = np.array(img)
    mask = np.array(mask)
    
    # Apply albumentations transforms
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    
    return img, mask

def val_aug(img, mask):
    # Convert PIL images to numpy arrays
    img = np.array(img)
    mask = np.array(mask)
    
    # Ensure mask is 2D
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    # Handle the normalization
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
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0]) for id in img_filename_list]
        return img_ids

    
    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        
        # Read TIFF with BGRNIR channels
        img = tifffile.imread(img_name)
        
        # Normalize and convert to uint8
        if img.dtype in [np.float32, np.float64]:
            img = np.clip(img, 0, 1)  # Ensure values are in [0,1]
            img = (img * 255).astype(np.uint8)
        
        # Rearrange from BGRNIR to RGB by taking only BGR and reversing
        if img.shape[-1] == 4:
            img = img[:, :, :3]  # Take only BGR channels
            img = img[:, :, ::-1]  # Reverse BGR to RGB
        
        # Handle grayscale case
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
                
        img = Image.fromarray(img).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        
        # Ensure same size
        if img.size != mask.size:
            mask = mask.resize(img.size, Image.NEAREST)
        
        return img, mask

    def load_mosaic_img_and_mask(self, index):

        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        strat_y = h // 4
        
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)

        return img, mask

biodiversity_tiff_val_dataset = BiodiversityTiffTrainDataset(data_root='data/Biodiversity_tiff/Val', mosaic_ratio=0.0,
                                        transform=val_aug)


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
        img = np.array(img)
        aug = albu.Normalize()(image=img)
        img = aug['image']
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img_id = self.img_ids[index]
        results = {'img': img, 'img_id': img_id}
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        img_ids = [str(id.split('.')[0]) for id in img_filename_list]
        return img_ids

    def load_img(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        img = Image.open(img_name).convert('RGB')
        return img

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

def check_image_mask_sizes(data_root='data/Biodiversity_tiff/Val', 
                          img_dir='images', 
                          mask_dir='masks',
                          img_suffix='.tif', 
                          mask_suffix='.png'):
    """Check for size mismatches between images and their corresponding masks."""
    
    img_dir_path = osp.join(data_root, img_dir)
    mask_dir_path = osp.join(data_root, mask_dir)
    
    mismatched_pairs = []
    
    for img_name in os.listdir(img_dir_path):
        if img_name.endswith(img_suffix):
            # Get corresponding mask name
            mask_name = img_name.replace(img_suffix, mask_suffix)
            
            # Full paths
            img_path = osp.join(img_dir_path, img_name)
            mask_path = osp.join(mask_dir_path, mask_name)
            
            # Read image and mask
            img = tifffile.imread(img_path)
            mask = np.array(Image.open(mask_path))
            
            # Get dimensions
            img_h, img_w = img.shape[:2]
            mask_h, mask_w = mask.shape[:2]
            
            if (img_h, img_w) != (mask_h, mask_w):
                mismatched_pairs.append({
                    'image': img_name,
                    'mask': mask_name,
                    'img_size': (img_h, img_w),
                    'mask_size': (mask_h, mask_w)
                })
    
    # Print results
    if mismatched_pairs:
        print("\nFound mismatched dimensions:")
        print("-" * 80)
        for pair in mismatched_pairs:
            print(f"Image: {pair['image']}")
            print(f"Mask:  {pair['mask']}")
            print(f"Image size:  {pair['img_size']}")
            print(f"Mask size:   {pair['mask_size']}")
            print("-" * 80)
    else:
        print("\nAll images and masks have matching dimensions!")
    
    return mismatched_pairs