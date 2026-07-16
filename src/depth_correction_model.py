"""
depth_correction_model.py
==========================
RGB-guided IR depth correction network - SKELETON.

Architecture:
  RGB branch  ---\
                   >-- Cross-Attention Fusion (per scale) --> Decoder --> corrected depth
  IR/Depth branch-/

This is a skeleton: encoders are small conv stacks (swap for ResNet/etc.
later). The goal right now is only to prove the shapes line up and a full
forward pass runs end-to-end - not to hit any accuracy target yet.
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1), # convolution: scan the image with a 3x3 filter, downsample by stride    
        nn.BatchNorm2d(out_ch), #maintain the mean and variance of the output to stabilize training
        nn.ReLU(inplace=True), #make all the negative values zero and keep the positive values unchanged
    )


class Encoder(nn.Module):
    """3-stage downsampling encoder, used for both RGB and IR branches
    (same structure, different input channel count)."""

    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        c = base_channels
        self.stage1 = conv_block(in_channels, c)    # shrink to 1/2
        self.stage2 = conv_block(c, c * 2)          # 1/4
        self.stage3 = conv_block(c * 2, c * 4)      # 1/8

    def forward(self, x):
        f1 = self.stage1(x) #raw picture
        f2 = self.stage2(f1) #based on f1 - process the picture to get more abstract features
        f3 = self.stage3(f2) #based on f2 - process the picture to get more abstract features
        return f1, f2, f3  # multi-scale features, shallow -> deep
    # the encoder sees the larger picture and extracts features at multiple scales, 
    # which will be used later in the fusion and decoding stages.

class CrossAttentionFusion(nn.Module):
    """Fuses RGB and IR features via multi-head cross-attention: IR features
    attend to RGB features (RGB "guides" the correction).

    Full spatial attention is O((H*W)^2), so this is only affordable at the
    deepest (smallest) scale. Shallow scales use ConcatFusion instead - see
    below.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, rgb_feat, ir_feat):
        b, c, h, w = ir_feat.shape

        rgb_seq = rgb_feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        ir_seq = ir_feat.flatten(2).permute(0, 2, 1)     # (B, H*W, C)

        attended, _ = self.attn(query=ir_seq, key=rgb_seq, value=rgb_seq)
        # IR asks the RGB features for attention
        fused_seq = self.norm(ir_seq + attended)
        # add the answer to the original IR features, then normalize

        fused = fused_seq.permute(0, 2, 1).reshape(b, c, h, w)
        return self.proj_out(fused)


class ConcatFusion(nn.Module):
    """Cheap fusion for the shallow (high-resolution) scales, where full
    attention would be too expensive: concatenate RGB + IR channels, then
    a 1x1 conv mixes them back down to the original channel count."""

    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, rgb_feat, ir_feat):
        return self.proj(torch.cat([rgb_feat, ir_feat], dim=1))
    # connect the RGB and IR features by concatenating them along the channel dimension,
    # then use a 1x1 convolution to reduce the channel count back to the original 64 to 32


class Decoder(nn.Module):
    """Upsamples fused multi-scale features back to full resolution,
    with skip connections from shallower fused scales (U-Net style)."""

    def __init__(self, base_channels=32):
        super().__init__()
        c = base_channels
        self.up3 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)   # /8 -> /4
        self.merge3 = conv_block(c * 2 + c * 2, c * 2, stride=1)
        # enlarge the feature map from 1/8 to 1/4 of the original size, 
        # then merge it with the corresponding fused features from the encoder
        self.up2 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)       # /4 -> /2
        self.merge2 = conv_block(c + c, c, stride=1)
        
        self.up1 = nn.ConvTranspose2d(c, c, kernel_size=2, stride=2)           # /2 -> /1
        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)  # 1-channel corrected depth

    def forward(self, fused1, fused2, fused3):
        x = self.up3(fused3)
        x = self.merge3(torch.cat([x, fused2], dim=1))

        x = self.up2(x)
        x = self.merge2(torch.cat([x, fused1], dim=1))
        
        x = self.up1(x)
        return self.out_conv(x)
    # borrow the fused features from the encoder at each scale to help reconstruct the final output


class DepthCorrectionNet(nn.Module):
    """Full model: RGB encoder + IR encoder -> per-scale cross-attention
    fusion -> decoder -> corrected depth map (same H, W as input)."""

    def __init__(self, base_channels=32, num_heads=4):
        super().__init__()
        self.rgb_encoder = Encoder(in_channels=3, base_channels=base_channels)
        self.ir_encoder = Encoder(in_channels=1, base_channels=base_channels)
        # create two separate encoders for RGB and IR inputs, each producing multi-scale features
        c = base_channels
        self.fuse1 = ConcatFusion(channels=c)                                # /2  - cheap
        self.fuse2 = ConcatFusion(channels=c * 2)                            # /4  - cheap
        self.fuse3 = CrossAttentionFusion(channels=c * 4, num_heads=num_heads)  # /8 - attention
        # create fusion modules for each scale: shallow scales use concatenation, 
        # while the deepest scale uses cross-attention
        self.decoder = Decoder(base_channels=base_channels)
        # merge the fused features from all scales and reconstruct the final corrected depth map
    def forward(self, rgb, ir):
        rgb1, rgb2, rgb3 = self.rgb_encoder(rgb)
        ir1, ir2, ir3 = self.ir_encoder(ir)
        #put the RGB and IR inputs through their respective encoders to get multi-scale features
        fused1 = self.fuse1(rgb1, ir1)
        fused2 = self.fuse2(rgb2, ir2)
        fused3 = self.fuse3(rgb3, ir3)
        # fuse the features from both modalities at each scale
        corrected_depth = self.decoder(fused1, fused2, fused3)
        return corrected_depth
        #bring the fused features into the decoder to produce the final corrected depth map

if __name__ == "__main__":
    # Smoke test: confirm shapes line up end-to-end with a fake batch.
    # Using a smaller resolution than the real 1280x720 D415 output so this
    # runs fast on CPU - swap in real data / real resolution once the
    # data-loading + ground-truth pipeline is ready.
    batch_size = 2
    h, w = 256, 256

    rgb = torch.randn(batch_size, 3, h, w)
    ir = torch.randn(batch_size, 1, h, w)

    model = DepthCorrectionNet(base_channels=32, num_heads=4)
    out = model(rgb, ir)

    print(f"rgb input:  {tuple(rgb.shape)}")
    print(f"ir input:   {tuple(ir.shape)}")
    print(f"output:     {tuple(out.shape)}")
    assert out.shape == (batch_size, 1, h, w), "output shape must match input H, W"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {n_params:,}")
    print("Forward pass OK - shapes line up.")