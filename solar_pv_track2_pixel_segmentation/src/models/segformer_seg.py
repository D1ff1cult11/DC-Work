"""
Customized SegFormer Segmentation Architecture
Direct implementation of Table 2 & Section 3.3 in:
Joshi & Pal, "Lightweight transformer driven segmentation of defects in solar PV thermal imagery",
Solar Energy 308 (2026) 114402.

Parameters: ~3.1M
Encoder: Hierarchical Mix-Transformer (MiT-B0 inspired)
Decoder: 4-stage Multi-Layer Perceptron (MLP) with Conv2d classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class OverlapPatchEmbed(nn.Module):
    """Overlapped patch embedding to preserve spatial gradient continuity."""
    def __init__(self, in_chans=1, embed_dim=32, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        # (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    """Efficient self-attention with spatial reduction ratio R."""
    def __init__(self, dim, num_heads=1, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr is not None:
            x_spatial = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_reduced = self.sr(x_spatial).reshape(B, C, -1).permute(0, 2, 1)
            x_reduced = self.norm(x_reduced)
            kv = self.kv(x_reduced).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MixFFN(nn.Module):
    """Mix-FFN with 3x3 depthwise convolution (implicit positional encoding)."""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.dwconv(x)
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, drop=drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class MLPDecoderStage(nn.Module):
    """Linear projection stage in MLP decoder."""
    def __init__(self, in_dim, out_dim=192):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: (B, N, C) -> (B, N, out_dim)
        return self.proj(x)


class CustomizedSegFormer(nn.Module):
    """
    Paper's Customized SegFormer for PV Thermal Segmentation:
      - 4-Stage Encoder: dims=[32, 64, 128, 192], depths=[2, 2, 3, 3], heads=[1, 2, 4, 6]
      - MLP Decoder: hidden_dim=192, Conv1x1 output -> 3 classes
    """
    def __init__(self, in_chans=1, num_classes=3, decoder_dim=192):
        super().__init__()
        self.num_classes = num_classes

        # Stage 1: 1/4 resolution
        self.patch_embed1 = OverlapPatchEmbed(in_chans, 32, patch_size=7, stride=4)
        self.block1 = nn.ModuleList([TransformerBlock(32, 1, sr_ratio=8) for _ in range(2)])
        self.norm1 = nn.LayerNorm(32)

        # Stage 2: 1/8 resolution
        self.patch_embed2 = OverlapPatchEmbed(32, 64, patch_size=3, stride=2)
        self.block2 = nn.ModuleList([TransformerBlock(64, 2, sr_ratio=4) for _ in range(2)])
        self.norm2 = nn.LayerNorm(64)

        # Stage 3: 1/16 resolution
        self.patch_embed3 = OverlapPatchEmbed(64, 128, patch_size=3, stride=2)
        self.block3 = nn.ModuleList([TransformerBlock(128, 4, sr_ratio=2) for _ in range(3)])
        self.norm3 = nn.LayerNorm(128)

        # Stage 4: 1/32 resolution
        self.patch_embed4 = OverlapPatchEmbed(128, 192, patch_size=3, stride=2)
        self.block4 = nn.ModuleList([TransformerBlock(192, 6, sr_ratio=1) for _ in range(3)])
        self.norm4 = nn.LayerNorm(192)

        # MLP Decoder (Paper Table 2)
        self.linear_c1 = MLPDecoderStage(32, decoder_dim)
        self.linear_c2 = MLPDecoderStage(64, decoder_dim)
        self.linear_c3 = MLPDecoderStage(128, decoder_dim)
        self.linear_c4 = MLPDecoderStage(192, decoder_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.linear_pred = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, x):
        orig_H, orig_W = x.shape[2], x.shape[3]

        # Stage 1
        x1, H1, W1 = self.patch_embed1(x)
        for blk in self.block1:
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1)

        # Stage 2
        x2, H2, W2 = self.patch_embed2(x1.permute(0, 2, 1).reshape(-1, 32, H1, W1))
        for blk in self.block2:
            x2 = blk(x2, H2, W2)
        x2 = self.norm2(x2)

        # Stage 3
        x3, H3, W3 = self.patch_embed3(x2.permute(0, 2, 1).reshape(-1, 64, H2, W2))
        for blk in self.block3:
            x3 = blk(x3, H3, W3)
        x3 = self.norm3(x3)

        # Stage 4
        x4, H4, W4 = self.patch_embed4(x3.permute(0, 2, 1).reshape(-1, 128, H3, W3))
        for blk in self.block4:
            x4 = blk(x4, H4, W4)
        x4 = self.norm4(x4)

        # Decoder: Project & Upsample to 1/4 resolution (H1, W1)
        B = x.shape[0]
        c1 = self.linear_c1(x1).permute(0, 2, 1).reshape(B, -1, H1, W1)
        c2 = self.linear_c2(x2).permute(0, 2, 1).reshape(B, -1, H2, W2)
        c2 = F.interpolate(c2, size=(H1, W1), mode="bilinear", align_corners=False)

        c3 = self.linear_c3(x3).permute(0, 2, 1).reshape(B, -1, H3, W3)
        c3 = F.interpolate(c3, size=(H1, W1), mode="bilinear", align_corners=False)

        c4 = self.linear_c4(x4).permute(0, 2, 1).reshape(B, -1, H4, W4)
        c4 = F.interpolate(c4, size=(H1, W1), mode="bilinear", align_corners=False)

        # Fuse
        fused = self.linear_fuse(torch.cat([c1, c2, c3, c4], dim=1))
        logits = self.linear_pred(fused)

        # Upsample to input resolution
        out = F.interpolate(logits, size=(orig_H, orig_W), mode="bilinear", align_corners=False)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
