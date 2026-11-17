"""
Classical U-Net Baseline Architecture for Solar PV Thermal Segmentation.
Provides canonical CNN benchmark comparison matching Section 4.5.2 in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetSeg(nn.Module):
    def __init__(self, in_chans=1, num_classes=3, base_ch=32):
        super().__init__()
        self.inc = DoubleConv(in_chans, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch, base_ch * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch * 2, base_ch * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch * 4, base_ch * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch * 8, base_ch * 16))

        self.up1 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.conv_up1 = DoubleConv(base_ch * 16, base_ch * 8)

        self.up2 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.conv_up2 = DoubleConv(base_ch * 8, base_ch * 4)

        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.conv_up3 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up4 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.conv_up4 = DoubleConv(base_ch * 2, base_ch)

        self.outc = nn.Conv2d(base_ch, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = self.conv_up1(torch.cat([x, x4], dim=1))

        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x3], dim=1))

        x = self.up3(x)
        x = self.conv_up3(torch.cat([x, x2], dim=1))

        x = self.up4(x)
        x = self.conv_up4(torch.cat([x, x1], dim=1))

        logits = self.outc(x)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
