"""
Steerable Filters of First- and Second-Order Channels (SFOC & FSFOC) for Thermal Imagery.
Based on Zhu et al. (ISPRS / Remote Sensing, 2022/2023).

Decomposes thermal images into:
1. 1st-Order Steerable Channels (F-SFOC): Captures physical temperature step gradients (defect edges).
2. 2nd-Order Steerable Channels (S-SFOC): Captures thermal dissipation curvature and ridge paths.
3. Fast-SFOC (FSFOC): Separable 1D Gaussian derivative kernels for real-time edge UAV deployment.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_gaussian_derivatives_2d(kernel_size=9, sigma=1.5):
    """Analytical 2D Gaussian kernels and derivatives."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    y, x = torch.meshgrid(coords, coords, indexing="ij")

    g = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    g = g / (2 * math.pi * sigma**2)

    # 1st-order derivatives (Edges)
    gx = -(x / (sigma**2)) * g
    gy = -(y / (sigma**2)) * g

    # 2nd-order derivatives (Ridges / Curvatures)
    gxx = ((x**2 - sigma**2) / (sigma**4)) * g
    gyy = ((y**2 - sigma**2) / (sigma**4)) * g
    gxy = (x * y / (sigma**4)) * g

    return gx, gy, gxx, gyy, gxy


class SFOCExtractor(nn.Module):
    """
    Standard SFOC Layer: Combines 1st- and 2nd-Order Steerable Filter Channels.
    Output: 4-channel Physics-Informed Radiometric Tensor:
      [Raw Thermal, 1st-Order Gradient, 2nd-Order Curvature, Directional Orientation]
    """
    def __init__(self, kernel_size=9, sigma=1.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        gx, gy, gxx, gyy, gxy = get_gaussian_derivatives_2d(kernel_size, sigma)
        filters = torch.stack([gx, gy, gxx, gyy, gxy], dim=0).unsqueeze(1)
        self.register_buffer("weight", filters)

    def forward(self, x):
        if x.shape[1] > 1:
            x_mono = x.mean(dim=1, keepdim=True)
        else:
            x_mono = x

        derivs = F.conv2d(x_mono, self.weight, padding=self.padding)
        gx, gy, gxx, gyy, gxy = derivs.split(1, dim=1)

        # 1st-order gradient magnitude (Defect edge)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)

        # 2nd-order curvature energy (Heat dissipation halo)
        curvature = torch.sqrt(gxx**2 + gyy**2 + 2 * (gxy**2) + 1e-6)

        # Principal thermal orientation angle
        orientation = torch.atan2(gy, gx) / math.pi

        return torch.cat([x_mono, grad_mag, curvature, orientation], dim=1)


class FSFOCExtractor(nn.Module):
    """
    Fast Steerable Filters of Oriented Channels (FSFOC).
    Uses separable 1D steerable kernels to reduce complexity from O(K^2) to O(2K),
    enabling real-time execution (>100 FPS) on edge UAV processors.
    """
    def __init__(self, kernel_size=7, sigma=1.2, num_orientations=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.num_orientations = num_orientations

        # 1D Gaussian and 1st/2nd derivative kernels
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        g1d = torch.exp(-coords**2 / (2 * sigma**2))
        g1d = g1d / g1d.sum()
        dg1d = -coords / (sigma**2) * g1d

        # Register separable buffers
        self.register_buffer("g1d_row", g1d.view(1, 1, 1, kernel_size))
        self.register_buffer("g1d_col", g1d.view(1, 1, kernel_size, 1))
        self.register_buffer("dg1d_row", dg1d.view(1, 1, 1, kernel_size))
        self.register_buffer("dg1d_col", dg1d.view(1, 1, kernel_size, 1))

        # Orientation steering angles
        self.angles = [i * math.pi / num_orientations for i in range(num_orientations)]

    def forward(self, x):
        if x.shape[1] > 1:
            x_mono = x.mean(dim=1, keepdim=True)
        else:
            x_mono = x

        # Separable 1st derivatives: Gx = dg1d_row * g1d_col; Gy = g1d_row * dg1d_col
        gx = F.conv2d(F.conv2d(x_mono, self.dg1d_row, padding=(0, self.padding)), self.g1d_col, padding=(self.padding, 0))
        gy = F.conv2d(F.conv2d(x_mono, self.g1d_row, padding=(0, self.padding)), self.dg1d_col, padding=(self.padding, 0))

        steered_channels = []
        for theta in self.angles:
            # Steerable combination in direction theta: G_theta = cos(theta)*Gx + sin(theta)*Gy
            g_theta = math.cos(theta) * gx + math.sin(theta) * gy
            steered_channels.append(torch.abs(g_theta))

        fsfoc_tensor = torch.cat(steered_channels, dim=1) # (B, num_orientations, H, W)
        # Normalize
        norm = torch.sqrt(torch.sum(fsfoc_tensor**2, dim=1, keepdim=True) + 1e-6)
        return fsfoc_tensor / norm
