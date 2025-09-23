def dcswin_base(pretrained=True, num_classes=4, in_chans=3, weight_path='pretrain_weights/stseg_base.pth'):
    # pretrained weights are load from official repo of Swin Transformer
    model = DCSwin(encoder_channels=(128, 256, 512, 1024),
                   num_classes=num_classes,
                   embed_dim=128,
                   depths=(2, 2, 18, 2),
                   num_heads=(4, 8, 16, 32),
                   frozen_stages=2,
                   in_chans=in_chans)  # Add in_chans parameter
    
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        
        # Handle input channel adaptation for pretrained weights
        if in_chans != 3 and 'backbone.patch_embed.proj.weight' in old_dict:
            pretrained_weight = old_dict['backbone.patch_embed.proj.weight']
            old_dict['backbone.patch_embed.proj.weight'] = adapt_input_conv_weight(
                pretrained_weight, in_chans
            )
        
        model_dict.update(old_dict)
        model.load_state_dict(model_dict, strict=False)  # Use strict=False for channel mismatches
    return model


def dcswin_small(pretrained=True, num_classes=4, in_chans=3, weight_path='pretrain_weights/stseg_small.pth'):
    model = DCSwin(encoder_channels=(96, 192, 384, 768),
                   num_classes=num_classes,
                   embed_dim=96,
                   depths=(2, 2, 18, 2),
                   num_heads=(3, 6, 12, 24),
                   frozen_stages=2,
                   in_chans=in_chans)  # Add in_chans parameter
    
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        
        # Handle input channel adaptation for pretrained weights
        if in_chans != 3 and 'backbone.patch_embed.proj.weight' in old_dict:
            pretrained_weight = old_dict['backbone.patch_embed.proj.weight']
            old_dict['backbone.patch_embed.proj.weight'] = adapt_input_conv_weight(
                pretrained_weight, in_chans
            )
        
        model_dict.update(old_dict)
        model.load_state_dict(model_dict, strict=False)
    return model


def dcswin_tiny(pretrained=True, num_classes=4, in_chans=3, weight_path='pretrain_weights/stseg_tiny.pth'):
    model = DCSwin(encoder_channels=(96, 192, 384, 768),
                   num_classes=num_classes,
                   embed_dim=96,
                   depths=(2, 2, 6, 2),
                   num_heads=(3, 6, 12, 24),
                   frozen_stages=2,
                   in_chans=in_chans)  # Add in_chans parameter
    
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        
        # Handle input channel adaptation for pretrained weights
        if in_chans != 3 and 'backbone.patch_embed.proj.weight' in old_dict:
            pretrained_weight = old_dict['backbone.patch_embed.proj.weight']
            old_dict['backbone.patch_embed.proj.weight'] = adapt_input_conv_weight(
                pretrained_weight, in_chans
            )
        
        model_dict.update(old_dict)
        model.load_state_dict(model_dict, strict=False)
    return model


def adapt_input_conv_weight(pretrained_weight, in_chans):
    """Adapt pretrained RGB conv weights to different number of input channels"""
    if in_chans == 3:
        return pretrained_weight
    
    # pretrained_weight shape: (out_channels, 3, kernel_h, kernel_w)
    out_channels = pretrained_weight.shape[0]
    kernel_size = pretrained_weight.shape[2:]
    
    if in_chans == 1:
        # For grayscale: average the RGB weights
        adapted_weight = pretrained_weight.mean(dim=1, keepdim=True)
    elif in_chans < 3:
        # For fewer channels: take subset of RGB weights
        adapted_weight = pretrained_weight[:, :in_chans, :, :]
    else:
        # For more channels: repeat and scale RGB weights
        repeat_factor = (in_chans + 2) // 3  # Ceiling division
        expanded_weight = pretrained_weight.repeat(1, repeat_factor, 1, 1)
        adapted_weight = expanded_weight[:, :in_chans, :, :]
        # Scale to preserve magnitude
        adapted_weight = adapted_weight * (3.0 / in_chans)
    
    return adapted_weight