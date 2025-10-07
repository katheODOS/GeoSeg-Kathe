#!/usr/bin/env python3
"""
Debug script to check validation dataset existence and content
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(levelname)s: %(message)s')

def check_dataset_paths():
    # Base data path
    data_root = Path('C:/Users/Admin/anaconda3/envs/GeoSeg-Kathe/data')
    
    # Check Train directory
    train_path = data_root / 'Biodiversity_tiff/Train'
    if train_path.exists():
        train_images = list((train_path / 'images').glob('*.tif'))
        train_masks = list((train_path / 'masks').glob('*.tif'))
        logging.info(f"Train directory exists: {train_path}")
        logging.info(f"  Found {len(train_images)} images and {len(train_masks)} masks")
    else:
        logging.error(f"Train directory not found: {train_path}")
    
    # Check Val directory
    val_path = data_root / 'Biodiversity_tiff/Val'
    if val_path.exists():
        # Check images subdirectory
        val_images_dir = val_path / 'images'
        if val_images_dir.exists():
            val_images = list(val_images_dir.glob('*.tif'))
            logging.info(f"Val images directory exists: {val_images_dir}")
            logging.info(f"  Found {len(val_images)} validation images")
        else:
            logging.error(f"Val images directory not found: {val_images_dir}")
        
        # Check masks subdirectory
        val_masks_dir = val_path / 'masks'
        if val_masks_dir.exists():
            val_masks = list(val_masks_dir.glob('*.tif'))
            logging.info(f"Val masks directory exists: {val_masks_dir}")
            logging.info(f"  Found {len(val_masks)} validation masks")
        else:
            logging.error(f"Val masks directory not found: {val_masks_dir}")
    else:
        logging.error(f"Val directory not found: {val_path}")

if __name__ == "__main__":
    check_dataset_paths()