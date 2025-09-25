import torch

# Replace with your actual checkpoint path
checkpoint_path = r"C:\Users\Admin\anaconda3\envs\GeoSeg-Kathe\GeoSeg-Kathe\model_weights\biodiversity_tiff4\ftunetformer-512-crop-ms-e45\last.ckpt"

checkpoint = torch.load(checkpoint_path, map_location='cpu')
print("Checkpoint keys:", list(checkpoint.keys()))

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    
    print(f"\nSearching for the FIRST conv layer (backbone input layer)...")
    
    # Look specifically for backbone layers that process raw input
    backbone_conv_layers = []
    for key, tensor in state_dict.items():
        if ('backbone' in key.lower() and 'conv' in key.lower() and 'weight' in key 
            and hasattr(tensor, 'shape') and len(tensor.shape) == 4):
            backbone_conv_layers.append((key, tensor.shape))
    
    # Sort by key name to get the earliest layers first
    backbone_conv_layers.sort()
    
    print(f"\nFound {len(backbone_conv_layers)} backbone conv layers:")
    for i, (key, shape) in enumerate(backbone_conv_layers[:5]):  # Show first 5
        print(f"  {i+1}. {key}: {shape}")
        if i == 0:  # The very first one should show input channels
            print(f"      -> INPUT CHANNELS: {shape[1]} ⭐")
    
    # Also look for patch_embed layers (common in transformers)
    print(f"\nLooking for patch embedding layers...")
    for key, tensor in state_dict.items():
        if ('patch_embed' in key.lower() and 'weight' in key 
            and hasattr(tensor, 'shape') and len(tensor.shape) == 4):
            print(f"  {key}: {tensor.shape}")
            print(f"      -> INPUT CHANNELS: {tensor.shape[1]} ⭐")
    
    # Show some example keys to understand the structure
    print(f"\nFirst 15 state_dict keys for reference:")
    for i, key in enumerate(list(state_dict.keys())[:15]):
        print(f"  {i+1:2d}. {key}")