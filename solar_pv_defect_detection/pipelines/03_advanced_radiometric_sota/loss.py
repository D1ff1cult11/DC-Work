"""
Physics-Guided Dual-Domain Boundary Loss for Thermal Defect Segmentation.
Penalizes predictions spilling from true 1st-order defect edges into 2nd-order heat dissipation halos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadiometricBoundaryLoss(nn.Module):
    """
    Combines Focal/CrossEntropy Loss with SFOC-guided boundary regularization.
    """
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha # Weight for task classification
        self.beta = beta   # Weight for boundary alignment
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets, sfoc_tensor=None):
        """
        logits: (B, C, H, W) or (B, C)
        targets: (B, H, W) or (B,)
        sfoc_tensor: (B, 4, H, W) [Raw, Gradient, Curvature, Orientation]
        """
        task_loss = self.ce(logits, targets)

        if sfoc_tensor is not None and len(logits.shape) == 4:
            # Dense segmentation mode: apply boundary alignment loss
            # sfoc_tensor[:, 1] is 1st-order edge energy; sfoc_tensor[:, 2] is 2nd-order curvature
            edge_energy = sfoc_tensor[:, 1:2]
            curvature_energy = sfoc_tensor[:, 2:3]

            probs = F.softmax(logits, dim=1)
            # Defect probability channel (assuming class 1+ are defects)
            defect_prob = probs[:, 1:].sum(dim=1, keepdim=True)

            # High defect probability on 2nd-order curvature (diffuse halo) without 1st-order edge is penalized
            halo_spill = defect_prob * curvature_energy * (1.0 - edge_energy)
            boundary_loss = halo_spill.mean()

            return self.alpha * task_loss + self.beta * boundary_loss

        return task_loss
