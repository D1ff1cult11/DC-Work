"""
Training Script for Pipeline 1: Paper Reproduction (SegFormer MiT-B0).
Exact reproduction of Joshi & Pal (Solar Energy, 2026).
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import create_dataloaders, CLASSES
from src.metrics import compute_metrics, print_evaluation_summary
from model import SegFormerPaperModel


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Pipeline 1 (Paper SegFormer MiT-B0) running on: {device}", flush=True)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=(args.img_size, args.img_size),
        val_split=0.15,
        test_split=0.15,
        random_seed=args.seed
    )

    model = SegFormerPaperModel(num_classes=12).to(device)
    params = model.count_parameters()
    print(f"[*] SegFormer Model Loaded: {params:,} trainable parameters.", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    out_dir = os.path.join(args.output_dir, "01_paper_segformer")
    os.makedirs(out_dir, exist_ok=True)
    best_ckpt = os.path.join(out_dir, "best_segformer.pt")
    best_f1 = 0.0

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    total_batches = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss, train_preds, train_targets = 0.0, [], []
        for b_idx, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, targets)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == total_batches:
                print(f"  [Epoch {epoch:02d}/{args.epochs:02d}] Batch [{b_idx+1:03d}/{total_batches:03d}] Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_m = compute_metrics(train_targets, train_preds, class_names=CLASSES)

        # Validation
        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_m = compute_metrics(val_targets, val_preds, class_names=CLASSES)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_m["macro_f1"])
        history["val_f1"].append(val_m["macro_f1"])

        epoch_dur = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_dur:.1f}s) "
              f"Train Loss: {train_loss:.4f} | Train F1: {train_m['macro_f1']*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val F1: {val_m['macro_f1']*100:.2f}%", flush=True)

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            torch.save(model.state_dict(), best_ckpt)
            print(f" --> Best SegFormer checkpoint saved to: {best_ckpt} (Val F1: {best_f1*100:.2f}%)", flush=True)

    # Final Test Set Evaluation
    print("\n[*] Evaluating Best SegFormer Checkpoint on Test Set...", flush=True)
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    test_preds, test_targets = [], []
    t_start = time.time()
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
    test_dur = time.time() - t_start
    fps = len(test_loader.dataset) / test_dur if test_dur > 0 else 0.0

    test_metrics = compute_metrics(test_targets, test_preds, class_names=CLASSES)
    test_metrics["inference_fps"] = fps
    test_metrics["parameters"] = params
    print_evaluation_summary(test_metrics)
    print(f"[*] Inference Speed: {fps:.1f} FPS on {device.type.upper()}", flush=True)

    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    print(f"[*] Complete evaluation reports saved in: {out_dir}", flush=True)


if __name__ == "__main__":
    default_data = os.path.abspath(os.path.join(PROJECT_ROOT, "data/infrared_solar_modules"))
    default_out = os.path.abspath(os.path.join(PROJECT_ROOT, "benchmark_results"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=default_data)
    parser.add_argument("--output_dir", type=str, default=default_out)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
