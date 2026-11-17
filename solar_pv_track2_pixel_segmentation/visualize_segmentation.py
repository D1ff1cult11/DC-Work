"""
Track 2 Visual Overlay Generator
Produces side-by-side [Original Image | Ground Truth Mask | Predicted Overlay] figures
matching Figure 6 in Joshi & Pal (Solar Energy, 2026).
"""

import os
import sys
import glob
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.dataset import SolarThermalSegmentationDataset, CLASSES, NUM_CLASSES
from src.models.segformer_seg import CustomizedSegFormer
from src.models.segnext_msca_seg import SegNeXtMSCASeg
from src.models.unet_seg import UNetSeg

def create_color_mask(mask_2d):
    """0: Black/Background, 1: Red (Hotspot), 2: Blue (Snail Trail)"""
    H, W = mask_2d.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    color_mask[mask_2d == 1] = [255, 0, 0]    # Red for Hotspot
    color_mask[mask_2d == 2] = [0, 100, 255]  # Blue for Snail Trail
    return color_mask

def blend_overlay(orig_img, color_mask, alpha=0.45):
    """Blends color mask onto grayscale or RGB thermal image."""
    if len(orig_img.shape) == 2:
        orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
    else:
        orig_rgb = orig_img.copy()

    has_defect = (color_mask.sum(axis=2) > 0)
    blended = orig_rgb.copy()
    blended[has_defect] = cv2.addWeighted(color_mask, alpha, orig_rgb, 1 - alpha, 0)[has_defect]
    return blended

def visualize_comparison_grid(data_dir, results_dir, out_dir, num_samples=4, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    test_ds = SolarThermalSegmentationDataset(data_dir, split="test", img_size=(224, 224), augment=False)

    # Load trained models if available
    models = {}
    model_constructors = {
        "SegFormer (Paper)": (CustomizedSegFormer, os.path.join(results_dir, "segformer", "best_segformer.pt")),
        "SegNeXt MSCA (Ours)": (SegNeXtMSCASeg, os.path.join(results_dir, "segnext", "best_segnext.pt")),
        "U-Net Baseline": (UNetSeg, os.path.join(results_dir, "unet", "best_unet.pt"))
    }

    for name, (ctor, ckpt) in model_constructors.items():
        if os.path.exists(ckpt):
            m = ctor(in_chans=1, num_classes=NUM_CLASSES)
            m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            m.to(device).eval()
            models[name] = m
            print(f"[+] Loaded checkpoint for {name}")

    if not models:
        print("[!] No trained checkpoints found yet. Visualizing ground truth samples only.")

    n_cols = 2 + len(models)
    fig, axes = plt.subplots(num_samples, n_cols, figsize=(4 * n_cols, 3.5 * num_samples), dpi=300)
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(num_samples):
        img_tensor, mask_tensor, sid = test_ds[i]
        img_np = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
        mask_np = mask_tensor.numpy()

        # Col 0: Raw Thermal
        axes[i, 0].imshow(img_np, cmap="gray")
        axes[i, 0].set_title(f"Thermal Input ({sid[:12]})", fontsize=10)
        axes[i, 0].axis("off")

        # Col 1: Ground Truth Mask
        gt_color = create_color_mask(mask_np)
        gt_overlay = blend_overlay(img_np, gt_color)
        axes[i, 1].imshow(gt_overlay)
        axes[i, 1].set_title("Ground Truth Mask", fontsize=10, fontweight="bold", color="darkgreen")
        axes[i, 1].axis("off")

        # Cols 2+: Model Predictions
        col_idx = 2
        for m_name, model in models.items():
            with torch.no_grad():
                inp = img_tensor.unsqueeze(0).to(device)
                pred_logits = model(inp)
                pred_mask = torch.argmax(pred_logits, dim=1).squeeze().cpu().numpy()

            pred_color = create_color_mask(pred_mask)
            pred_overlay = blend_overlay(img_np, pred_color)
            axes[i, col_idx].imshow(pred_overlay)
            axes[i, col_idx].set_title(f"{m_name} Prediction", fontsize=10)
            axes[i, col_idx].axis("off")
            col_idx += 1

    out_file = os.path.join(out_dir, "segmentation_predictions_overlay.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[+] Visual overlay comparison grid saved to: {out_file}")

if __name__ == "__main__":
    data_dir = r"e:\Dev Projects\DC Work\solar_pv_track2_pixel_segmentation\data"
    res_dir = r"e:\Dev Projects\DC Work\solar_pv_track2_pixel_segmentation\benchmark_results"
    out_dir = r"e:\Dev Projects\DC Work\solar_pv_track2_pixel_segmentation\docs\figures"
    visualize_comparison_grid(data_dir, res_dir, out_dir, num_samples=4)
