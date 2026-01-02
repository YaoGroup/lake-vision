"""
Attention mechanism options for spatial feature enhancement.

This module contains various attention mechanisms 
that can be applied to feature maps to enhance important spatial regions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialCBAM(nn.Module):
    """
    Spatial-only CBAM attention module.

    Applies spatial attention to emphasize important spatial locations
    while suppressing less relevant regions. Useses both average and max pooling
    across channels to capture spatial importance.

    Args:
        in_channels (int): number of input channels (kept for API consistency, not used)

    Input:
        x: [B, T, C, H, W] tensor of input sequences
            B: batch size
            T: time steps
            C: channels
            H: height
            W: width

    Output:
        [B, T, C, H, W] output tensor with spatial attention applied

    Example:
        >>> attn = SpatialCBAM(in_channels=32, kernel_size=7)
        >>> x = torch.randn(16, 153, 32, 64, 64) # example input tensor [B=16, T=153, C=32, H=64, W=64]
        >>> out = attn(x) # output tensor [B=16, T=153, C=32, H=64, W=64] (same size output)

    Reference:
        Based on spatial attention from Woo et al., "CBAM: Convolutaional Block Attention Module" ECCV 2018
    """

    def __init__(self, in_channels, kernel_size=7):
        super(SpatialCBAM, self).__init__()

        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, but got {kernel_size}")
        
        padding = kernel_size // 2

        # convolution to combine avg and max pooled features
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input features.

        Process:
            1. pool across channels using avg and max pooling
            2. concatenate pooled features
            3. apply convolution to generate attn map
            4. appl sigmoid and multiply with input
        """
        B, T, C, H, W = x.shape

        # flatten time into batch -> [B*T, C, H, W]
        x = x.reshape(B*T, C, H, W)

        # compute spatial attention map using max and average pooling across channels
        avg_pool = torch.mean(x, dim=1, keepdim=True)       # [B*T, 1, H, W]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)     # [B*T, 1, H, W]
        attn_input = torch.cat([avg_pool, max_pool], dim=1) # [B*T, 2, H, W]
        attn_map = self.sigmoid(self.conv(attn_input))      # [B*T, 1, H, W]

        # apply attention map to input
        x = x * attn_map  # [B*T, C, H, W]

        # reshape separating batch and time
        x = x.reshape(B, T, C, H, W)

        return x
        
class FullCBAM(nn.Module):
    """
    Full CBAM attention module (channel + spatial).

    Args:
        in_channels (int): number of input channels
        reduction_ratio (int): reduction ratio for channel attention MLP (default: 16)
        kernel_size (int): kernel size (default: 7)

    Input/Output:
        [B, T, C, H, W] tensor of input/output sequences
            B: batch size
            T: time steps
            C: channels
            H: height
            W: width
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(FullCBAM, self).__init__()

        # channel attention
        reduced = max(in_channels // reduction_ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, 1, bias=False),
        )

        # spatial attention
        self.spatial = SpatialCBAM(in_channels, kernel_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B*T, C, H, W)

        # channel attention
        avg_out = self.mlp(self.avg_pool(x_flat))
        max_out = self.mlp(self.max_pool(x_flat))
        attn_chan = self.sigmoid(avg_out + max_out)
        x_flat = x_flat * attn_chan

        # spatial attention
        x = x_flat.reshape(B, T, C, H, W)
        x = self.spatial(x)

        return x