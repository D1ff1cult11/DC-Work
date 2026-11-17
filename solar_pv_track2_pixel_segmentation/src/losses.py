"""
Track 2 Segmentation Losses
Provides composite Cross-Entropy + Soft Dice Loss for handling extreme pixel-level class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    """
    Multi-class Soft Dice Loss.
    Optimizes directly for spatial mask overlap (IoU/Dice).
    """
    def __init__(self, num_classes=3, smooth=1e-5, ignore_bg=False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, logits, targets):
        # logits: (B, C, H, W)
        # targets: (B, H, W)
        probs = F.softmax(logits, dim=1)
        # One-hot encode targets: (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        start_c = 1 if self.ignore_bg else 0
        dice_loss = 0.0
        n_classes = self.num_classes - start_c

        for c in range(start_c, self.num_classes):
            p_c = probs[:, c, :, :].contiguous().view(-1)
            t_c = targets_one_hot[:, c, :, :].contiguous().view(-1)

            intersection = (p_c * t_c).sum()
            union = (p_c * p_c).sum() + (t_c * t_c).sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1.0 - dice)

        return dice_loss / max(n_classes, 1)


class CompositeSegmentationLoss(nn.Module):
    """
    L_total = L_WCE + lambda * L_Dice
    Combines pixel-wise class discrimination with global shape/overlap optimization.
    """
    def __init__(self, weight=None, dice_weight=1.0, num_classes=3):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.dice_loss = SoftDiceLoss(num_classes=num_classes, ignore_bg=False)
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        loss_ce = self.ce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        return loss_ce + self.dice_weight * loss_dice
