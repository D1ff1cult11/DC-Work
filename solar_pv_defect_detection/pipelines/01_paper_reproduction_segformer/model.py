"""
SegFormer (MiT-B0) Architecture for Solar PV Defect Detection.
Exact reproduction of Joshi & Pal (Solar Energy, 2026).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialReductionAttention(nn.Module):
    def __init__(self, dim, num_heads=4, sr_ratio=2, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_spatial = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_reduced = self.sr(x_spatial).reshape(B, C, -1).permute(0, 2, 1)
            x_reduced = self.norm(x_reduced)
            kv = self.kv(x_reduced).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, dropout=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, dropout=dropout)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm(x)
        return x, H, W


class SegFormerPaperModel(nn.Module):
    """
    SegFormer MiT-B0 Model (~3.8M parameters).
    Matches Joshi & Pal (Solar Energy 2026).
    """
    def __init__(self, in_chans=3, num_classes=12, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1], dropout=0.1):
        super().__init__()
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.block1 = TransformerBlock(embed_dims[0], num_heads[0], sr_ratios[0], dropout=dropout)
        self.block2 = TransformerBlock(embed_dims[1], num_heads[1], sr_ratios[1], dropout=dropout)
        self.block3 = TransformerBlock(embed_dims[2], num_heads[2], sr_ratios[2], dropout=dropout)
        self.block4 = TransformerBlock(embed_dims[3], num_heads[3], sr_ratios[3], dropout=dropout)

        self.norm = nn.LayerNorm(embed_dims[3])
        self.head = nn.Sequential(
            nn.Linear(embed_dims[3], 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x, H, W = self.patch_embed1(x)
        x = self.block1(x, H, W)
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, H, W)

        x, H, W = self.patch_embed2(x)
        x = self.block2(x, H, W)
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, H, W)

        x, H, W = self.patch_embed3(x)
        x = self.block3(x, H, W)
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, H, W)

        x, H, W = self.patch_embed4(x)
        x = self.block4(x, H, W)

        x = self.norm(x)
        x_pooled = x.mean(dim=1)
        return self.head(x_pooled)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
