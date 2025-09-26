#!/usr/bin/env python3
"""
Script to debug PyTorch Lightning checkpoint loading issues
"""

import torch
import os
from pathlib import Path

checkpoint_path = r"C:\Users\Admin\anaconda3\envs\GeoSeg-Kathe\GeoSeg-Kathe\model_weights\biodiversity_tiff4\ftunetformer-512-crop-ms-e45\last.ckpt"

def inspect_checkpoint(checkpoint_path):
    """Inspect a PyTorch Lightning checkpoint to understand its structure and version info"""
    print(f"Inspecting checkpoint: {checkpoint_path}")
    print("=" * 60)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: Checkpoint file does not exist: {checkpoint_path}")
        return False
    
    try:
        # Load checkpoint with different methods
        print("Method 1: Loading with weights_only=False (unsafe but works for trusted checkpoints)")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print("\n📋 CHECKPOINT CONTENTS:")
        print("-" * 40)
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"  {key}: dict with {len(checkpoint[key])} keys")
                if key == 'hyper_parameters' and len(checkpoint[key]) < 20:
                    for subkey, subval in checkpoint[key].items():
                        print(f"    {subkey}: {type(subval)} = {str(subval)[:100]}")
            else:
                print(f"  {key}: {type(checkpoint[key])}")
        
        # Check Lightning version info
        print("\n🔍 VERSION INFORMATION:")
        print("-" * 40)
        if 'pytorch-lightning_version' in checkpoint:
            print(f"  Checkpoint Lightning Version: {checkpoint['pytorch-lightning_version']}")
        else:
            print("  ❌ Missing 'pytorch-lightning_version' key")
        
        if 'lr_schedulers' in checkpoint:
            print(f"  Has lr_schedulers: {len(checkpoint['lr_schedulers'])}")
        
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        
        if 'global_step' in checkpoint:
            print(f"  Global Step: {checkpoint['global_step']}")
            
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"  State Dict Keys: {len(state_dict)}")
            
            # Look for model architecture clues
            print("\n🏗️ MODEL ARCHITECTURE CLUES:")
            print("-" * 40)
            backbone_keys = [k for k in state_dict.keys() if 'backbone' in k][:5]
            if backbone_keys:
                print(f"  Backbone keys (first 5): {backbone_keys}")
            
            net_keys = [k for k in state_dict.keys() if 'net.' in k][:5] 
            if net_keys:
                print(f"  Net keys (first 5): {net_keys}")
            
            # Check for input layer to determine expected channels
            for key in state_dict.keys():
                if 'patch_embed.proj.weight' in key or 'conv1.weight' in key:
                    weight_shape = state_dict[key].shape
                    print(f"  Input layer '{key}': {weight_shape}")
                    print(f"    Expected input channels: {weight_shape[1]}")
                    break
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        print(f"Error type: {type(e)}")
        
        # Try loading as raw tensor dict
        try:
            print("\nMethod 2: Trying to load as raw PyTorch state dict...")
            raw_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            print(f"✓ Loaded as raw dict with keys: {list(raw_checkpoint.keys())}")
            return True
        except Exception as e2:
            print(f"❌ Raw loading also failed: {e2}")
        
        return False

def check_current_versions():
    """Check current PyTorch and Lightning versions"""
    print("\n🐍 CURRENT ENVIRONMENT:")
    print("=" * 60)
    
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    try:
        import pytorch_lightning as pl
        print(f"PyTorch Lightning version: {pl.__version__}")
    except ImportError:
        print("❌ PyTorch Lightning not installed")
    
    try:
        import lightning as L
        print(f"Lightning version: {L.__version__}")
    except ImportError:
        print("❌ Lightning not installed")
    
    try:
        import timm
        print(f"timm version: {timm.__version__}")
    except ImportError:
        print("❌ timm not installed")

def suggest_fixes(checkpoint_path):
    """Suggest potential fixes based on the checkpoint inspection"""
    print("\n🔧 POTENTIAL FIXES:")
    print("=" * 60)
    
    print("1. DOWNGRADE PYTORCH LIGHTNING:")
    print("   pip install pytorch-lightning==1.9.0")
    print("   # Try versions 1.6-1.9 if the checkpoint is old")
    
    print("\n2. MANUAL MODEL LOADING:")
    print("   # Load checkpoint manually and extract state_dict")
    print("   checkpoint = torch.load(path, weights_only=False)")
    print("   model = create_model()")
    print("   model.load_state_dict(checkpoint['state_dict'])")
    
    print("\n3. CONVERT CHECKPOINT:")
    print("   # Create a script to convert old checkpoint to new format")
    print("   # This involves recreating the model and saving a new checkpoint")
    
    print("\n4. USE COMPATIBLE LIGHTNING VERSION:")
    print("   # Check what version was used to create the checkpoint")
    print("   # Install that exact version")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python checkpoint_debug.py <checkpoint_path>")
        print("\nExample:")
        print('python checkpoint_debug.py "model_weights/biodiversity_tiff4/ftunetformer-512-crop-ms-e45/last.ckpt"')
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    check_current_versions()
    
    success = inspect_checkpoint(checkpoint_path)
    
    if not success:
        suggest_fixes(checkpoint_path)
    else:
        print("\n✅ Checkpoint loaded successfully! The issue might be in your loading code.")

