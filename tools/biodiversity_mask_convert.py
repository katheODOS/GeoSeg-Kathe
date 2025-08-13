import glob
import os
import numpy as np
import cv2
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import random

SEED = 322
ignore_index = 0 

CLASSES = ('Background','Forest land', 'Grassland', 'Cropland', 'Settlement', 'Seminatural Grassland')

PALETTE = [[11, 246, 210],[250, 62, 119], [168, 232, 84],
           [242, 180, 92], [116, 116, 116], [255, 214, 33]]


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", default="data/LoveDA/Train/Rural/masks_png")
    parser.add_argument("--output-mask-dir", default="data/LoveDA/Train/Rural/masks_png_convert")
    return parser.parse_args()


def convert_label(mask):
    # Keep background as 0 or set to ignore_index
    mask_copy = mask.copy()
    mask_copy[mask == 0] = ignore_index
    return mask_copy

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

def patch_format(inp):
    (mask_path, masks_output_dir) = inp
    # print(mask_path, masks_output_dir)
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    label = convert_label(mask)
    rgb_label = label2rgb(label.copy())
    rgb_label = cv2.cvtColor(rgb_label, cv2.COLOR_RGB2BGR)
    out_mask_path_rgb = os.path.join(masks_output_dir + '_rgb', "{}.png".format(mask_filename))
    cv2.imwrite(out_mask_path_rgb, rgb_label)

    out_mask_path = os.path.join(masks_output_dir, "{}.png".format(mask_filename))
    cv2.imwrite(out_mask_path, label)


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    masks_dir = args.mask_dir
    masks_output_dir = args.output_mask_dir
    mask_paths = glob.glob(os.path.join(masks_dir, "*.png"))

    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
        os.makedirs(masks_output_dir + '_rgb')

    inp = [(mask_path, masks_output_dir) for mask_path in mask_paths]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(patch_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images spliting spends: {} s'.format(split_time))


