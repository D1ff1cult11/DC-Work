"""
Track 2 Segmentation Metrics
Calculates mIoU, Dice Score (F1 per pixel), Pixel Accuracy, and Confusion Matrices.
"""

import numpy as np
import torch

def compute_segmentation_metrics(all_preds, all_targets, num_classes=3, class_names=None):
    """
    Computes dense semantic segmentation metrics:
      - Mean IoU (mIoU) and per-class IoU
      - Mean Dice / F1 and per-class Dice
      - Pixel Accuracy
      - Cohen's Kappa
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    # Flatten arrays
    preds = np.asarray(all_preds).reshape(-1)
    targets = np.asarray(all_targets).reshape(-1)

    # Confusion matrix: rows=targets, cols=preds
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    total_pixels = cm.sum()
    correct_pixels = np.trace(cm)
    pixel_accuracy = float(correct_pixels / max(total_pixels, 1))

    # Per-class IoU and Dice
    iou_per_class = {}
    dice_per_class = {}
    ious = []
    dices = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        denom_iou = tp + fp + fn
        denom_dice = 2 * tp + fp + fn

        iou = float(tp / denom_iou) if denom_iou > 0 else 0.0
        dice = float(2 * tp / denom_dice) if denom_dice > 0 else 0.0

        c_name = class_names[c]
        iou_per_class[c_name] = round(iou * 100, 2)
        dice_per_class[c_name] = round(dice * 100, 2)

        ious.append(iou)
        dices.append(dice)

    miou = float(np.mean(ious))
    mean_dice = float(np.mean(dices))

    # Cohen's Kappa calculation
    po = pixel_accuracy
    pe = sum((cm[c, :].sum() / total_pixels) * (cm[:, c].sum() / total_pixels) for c in range(num_classes))
    kappa = float((po - pe) / (1 - pe)) if (1 - pe) != 0 else 0.0

    return {
        "pixel_accuracy": round(pixel_accuracy * 100, 2),
        "miou": round(miou * 100, 2),
        "mean_dice": round(mean_dice * 100, 2),
        "cohen_kappa": round(kappa, 4),
        "iou_per_class": iou_per_class,
        "dice_per_class": dice_per_class,
        "confusion_matrix": cm.tolist()
    }
