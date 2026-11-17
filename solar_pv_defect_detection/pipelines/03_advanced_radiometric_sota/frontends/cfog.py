"""
Channel Features of Oriented Gradients (CFOG) - Differentiable PyTorch Implementation.
Based on Ye et al. (IEEE TGRS 2019): "A Robust Matching Method for Multi-Modal Remote Sensing
Images Based on Channel Features of Oriented Gradients".

Maps thermal infrared radiance into N directional orientation channels,
providing strict invariance against Nonlinear Radiometric Differences (NRD).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CFOGExtractor(nn.Module):
    """
    Differentiable CFOG Layer for Thermal Image Invariance.
    Input: (B, 1, H, W) single-channel LWIR thermal image.
    Output: (B, num_bins, H, W) multi-channel directional gradient representation.
    """
    def __init__(self, num_bins=8, kernel_size=5, sigma=1.2):
        super().__init__()
        self.num_bins = num_bins
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Sobel gradient filters for horizontal and vertical directions
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # 2D Gaussian smoothing kernel to enforce local structural coherence
        gauss_1d = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        gauss_1d = torch.exp(-gauss_1d**2 / (2 * sigma**2))
        gauss_2d = torch.outer(gauss_1d, gauss_1d)
        gauss_2d = (gauss_2d / gauss_2d.sum()).unsqueeze(0).unsqueeze(0)
        self.register_buffer("gauss_kernel", gauss_2d)

        # Define bin center angles in [0, pi)
        bin_angles = torch.linspace(0, math.pi, num_bins + 1)[:-1]
        self.register_buffer("bin_angles", bin_angles)

    def forward(self, x):
        if x.shape[1] > 1:
            x_mono = x.mean(dim=1, keepdim=True)
        else:
            x_mono = x

        B, _, H, W = x_mono.shape

        # 1. Compute spatial gradients
        gx = F.conv2d(x_mono, self.sobel_x, padding=1)
        gy = F.conv2d(x_mono, self.sobel_y, padding=1)

        # 2. Compute gradient magnitude and orientation angle
        mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        angle = torch.atan2(gy, gx) % math.pi # Wrap to [0, pi)

        # 3. Soft binning into N orientation channels (trilinear / cosine weighting)
        channels = []
        angle_flat = angle.view(B, 1, H, W)
        mag_flat = mag.view(B, 1, H, W)

        for b_angle in self.bin_angles:
            # Cosine similarity between gradient direction and bin orientation
            weight = torch.clamp(torch.cos(angle_flat - b_angle), min=0.0) ** 2
            channel = mag_flat * weight
            # Apply Gaussian spatial smoothing across each channel
            channel_smooth = F.conv2d(channel, self.gauss_kernel, padding=self.kernel_size // 2)
            channels.append(channel_smooth)

        cfog_tensor = torch.cat(channels, dim=1) # Shape: (B, num_bins, H, W)

        # L2 normalization across channels per pixel to ensure contrast invariance
        norm = torch.sqrt(torch.sum(cfog_tensor**2, dim=1, keepdim=True) + 1e-6)
        cfog_normalized = cfog_tensor / norm

        return cfog_normalized
