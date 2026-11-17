"""
Few-Shot Object Counting (FSOC) & Prompt-Guided Defect Localization Module.
Takes K user-provided defect exemplar patches (support set) and localizes/counts
all matching defective cells across the entire gigawatt solar farm image (query set).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FewShotDefectCounter(nn.Module):
    """
    FSOC Module: Density Map Estimator via Cross-Correlation.
    Integrates density over space to output exact count of failing cells,
    while peak activations provide spatial prompt seeds for pixel segmentation.
    """
    def __init__(self, feature_dim=256, hidden_dim=128):
        super().__init__()
        self.feature_dim = feature_dim

        # Correlation projection
        self.corr_proj = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Density Regression Head
        self.density_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU() # Density is strictly non-negative
        )

    def compute_similarity_map(self, query_feat, exemplar_feats):
        """
        query_feat: (B, C, H, W)
        exemplar_feats: (B, K, C) average pooled feature vectors of K exemplars
        """
        B, C, H, W = query_feat.shape
        # Normalize features
        q_norm = F.normalize(query_feat, p=2, dim=1) # (B, C, H, W)
        e_norm = F.normalize(exemplar_feats, p=2, dim=2) # (B, K, C)

        # Average exemplar vectors across K support examples
        prototype = e_norm.mean(dim=1, keepdim=True) # (B, 1, C)

        # Reshape query to (B, C, H*W)
        q_flat = q_norm.view(B, C, -1)
        sim_flat = torch.bmm(prototype, q_flat) # (B, 1, H*W)
        sim_map = sim_flat.view(B, 1, H, W) # (B, 1, H, W)

        return sim_map

    def forward(self, query_feat, exemplar_feats):
        """
        Returns:
          density_map: (B, 1, H, W)
          count: (B,) total predicted defect count across the solar array
        """
        sim_map = self.compute_similarity_map(query_feat, exemplar_feats)
        corr_feat = self.corr_proj(sim_map)
        density_map = self.density_head(corr_feat)

        # Count is spatial integral of density
        predicted_count = density_map.sum(dim=(2, 3)).squeeze(1)

        return density_map, predicted_count
