# test_multispectral.py
import torch
import sys
sys.path.append('.')
from geoseg.models.UNetFormer import UNetFormer

# Test the model with 4-band input
model = UNetFormer(
    num_classes=6,
    in_channels=4,  # 4-band input
    pretrained=True
)

# Create dummy 4-band input
batch_size = 2
channels = 4
height = 256
width = 256
dummy_input = torch.randn(batch_size, channels, height, width)

# Test forward pass
try:
    output = model(dummy_input)
    if model.training:
        print(f"Training mode - Output shapes: {output[0].shape}, {output[1].shape}")
    else:
        print(f"Eval mode - Output shape: {output.shape}")
    print(" Model accepts 4-band input successfully!")
except Exception as e:
    print(f"Error: {e}")