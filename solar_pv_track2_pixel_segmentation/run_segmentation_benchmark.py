"""
Track 2 Unified Segmentation Benchmark Runner
Trains and evaluates semantic segmentation models on the 277-image Colombian PV dataset:
  - Model 1: Customized SegFormer (Paper Reproduction, Table 2 & 3)
  - Model 2: SegNeXt MSCA (Our Innovation with strip convolutions)
  - Model 3: Classical U-Net Baseline

Logs mIoU, Dice scores, Pixel Accuracy, Cohen's Kappa, and inference FPS.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.dataset import create_segmentation_dataloaders, CLASSES, NUM_CLASSES
from src.metrics import compute_segmentation_metrics
from src.losses import CompositeSegmentationLoss
from src.models.segformer_seg import CustomizedSegFormer
from src.models.segnext_msca_seg import SegNeXtMSCASeg
from src.models.unet_seg import UNetSeg

def get_model(name):
    if name == "segformer":
        return CustomizedSegFormer(in_chans=1, num_classes=NUM_CLASSES)
    elif name == "segnext":
        return SegNeXtMSCASeg(in_chans=1, num_classes=NUM_CLASSES)
    elif name == "unet":
        return UNetSeg(in_chans=1, num_classes=NUM_CLASSES, base_ch=32)
    else:
        raise ValueError(f"Unknown model architecture: {name}")

def train_segmentation_model(model_name, data_dir, out_base, epochs=20, batch_size=8, lr=1e-4, device="cuda"):
    out_dir = os.path.join(out_base, model_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n====================================================================")
    print(f"[*] Starting Track 2 Run: {model_name.upper()} | Device: {device} | Epochs: {epochs}")
    print(f"====================================================================")

    train_loader, val_loader, test_loader = create_segmentation_dataloaders(
        data_dir, batch_size=batch_size, img_size=(224, 224)
    )

    model = get_model(model_name).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Model Architecture: {model_name} | Trainable Params: {params:,}")

    # Weighted loss: higher weight on scarce hotspots (1.3% of pixels)
    class_weights = torch.tensor([0.25, 4.0, 1.5], dtype=torch.float32).to(device)
    criterion = CompositeSegmentationLoss(weight=class_weights, dice_weight=1.0, num_classes=NUM_CLASSES)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = (device != "cpu" and torch.cuda.is_available())
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_miou = 0.0
    best_ckpt = os.path.join(out_dir, f"best_{model_name}.pt")
    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_dice": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.append(preds)
                val_targets.append(masks.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_preds_arr = np.concatenate(val_preds, axis=0)
        val_targets_arr = np.concatenate(val_targets, axis=0)
        val_metrics = compute_segmentation_metrics(val_preds_arr, val_targets_arr, NUM_CLASSES, CLASSES)

        dur = time.time() - t0
        print(f"  [{model_name}] Epoch {epoch:02d}/{epochs:02d} ({dur:.1f}s) | "
              f"TrLoss: {train_loss:.4f} | VaLoss: {val_loss:.4f} | "
              f"Va-mIoU: {val_metrics['miou']:.2f}% | Va-Dice: {val_metrics['mean_dice']:.2f}%", flush=True)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_metrics["miou"])
        history["val_dice"].append(val_metrics["mean_dice"])

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            torch.save(model.state_dict(), best_ckpt)
            print(f"    --> Best Checkpoint Saved! (Val mIoU: {best_miou:.2f}%)", flush=True)

    # Test Set Evaluation
    print(f"\n[*] Evaluating {model_name} on unseen test set ({len(test_loader.dataset)} images)...")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    model.eval()

    test_preds, test_targets = [], []
    t_start = time.time()
    with torch.no_grad():
        for imgs, masks, _ in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            test_preds.append(preds)
            test_targets.append(masks.cpu().numpy())

    test_dur = time.time() - t_start
    test_fps = len(test_loader.dataset) / test_dur if test_dur > 0 else 0.0

    test_preds_arr = np.concatenate(test_preds, axis=0)
    test_targets_arr = np.concatenate(test_targets, axis=0)
    test_metrics = compute_segmentation_metrics(test_preds_arr, test_targets_arr, NUM_CLASSES, CLASSES)

    test_metrics["model_name"] = model_name
    test_metrics["parameters"] = params
    test_metrics["inference_fps"] = round(test_fps, 1)
    test_metrics["epochs_trained"] = epochs

    print(f"\n====================================================================")
    print(f"[+] TEST RESULTS: {model_name.upper()}")
    print(f"    mIoU:           {test_metrics['miou']}%")
    print(f"    Mean Dice (F1): {test_metrics['mean_dice']}%")
    print(f"    Pixel Accuracy: {test_metrics['pixel_accuracy']}%")
    print(f"    Cohen's Kappa:  {test_metrics['cohen_kappa']}")
    print(f"    Inference FPS:  {test_metrics['inference_fps']} FPS (GPU)")
    print(f"    IoU Per Class:  {test_metrics['iou_per_class']}")
    print(f"    Dice Per Class: {test_metrics['dice_per_class']}")
    print(f"====================================================================\n")

    # Save artifacts
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    np.save(os.path.join(out_dir, "test_preds.npy"), test_preds_arr)
    np.save(os.path.join(out_dir, "test_targets.npy"), test_targets_arr)

    return test_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["all"], help="Models to benchmark: all, segformer, segnext, unet")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    data_dir = r"e:\Dev Projects\DC Work\solar_pv_track2_pixel_segmentation\data"
    out_base = r"e:\Dev Projects\DC Work\solar_pv_track2_pixel_segmentation\benchmark_results"

    targets = ["segformer", "segnext", "unet"] if "all" in args.models else args.models
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}
    for m in targets:
        res = train_segmentation_model(m, data_dir, out_base, epochs=args.epochs,
                                       batch_size=args.batch_size, lr=args.lr, device=device)
        results[m] = res

    # Save master summary
    with open(os.path.join(out_base, "master_segmentation_summary.json"), "w") as f:
        json.dump(results, f, indent=4)
    print("\n[+] All Track 2 benchmark models trained and evaluated successfully!")

if __name__ == "__main__":
    main()
