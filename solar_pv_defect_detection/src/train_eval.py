"""
Training and Evaluation Pipeline for Solar PV Thermal Defect Detection.
Supports Lightweight Transformer and CNN Baselines with Early Stopping and Metric Logging.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import create_dataloaders, CLASSES
from models.lightweight_transformer import build_model
from models.baselines import get_baseline_model
from metrics import compute_metrics, print_evaluation_summary


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    epoch_metrics = compute_metrics(all_targets, all_preds, class_names=CLASSES)
    return avg_loss, epoch_metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_targets, all_preds, class_names=CLASSES)
    return avg_loss, metrics


def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # Build Dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=(args.img_size, args.img_size),
        val_split=0.15,
        test_split=0.15,
        random_seed=args.seed
    )

    # Initialize Model
    if args.model_type == "transformer":
        model = build_model("lightweight_transformer", num_classes=12)
    else:
        model = get_baseline_model(architecture=args.model_type, num_classes=12)

    model = model.to(device)
    print(f"[*] Model {args.model_type} initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.")

    # Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_f1 = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.checkpoint_dir, f"best_{args.model_type}.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train F1: {train_metrics['macro_f1']*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val F1: {val_metrics['macro_f1']*100:.2f}%")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_ckpt_path)
            print(f" --> Best model saved to {best_ckpt_path} (Val F1: {best_val_f1*100:.2f}%)")

    # Final Test Set Evaluation
    print("\n[*] Loading best checkpoint for Test Set Evaluation...")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    
    print_evaluation_summary(test_metrics)

    results_file = os.path.join(args.checkpoint_dir, f"test_results_{args.model_type}.json")
    with open(results_file, "w") as f:
        json.dump(test_metrics, f, indent=4)
    print(f"[*] Detailed test metrics exported to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solar PV Defect Classification")
    parser.add_argument("--data_dir", type=str, default="./data/infrared_solar_modules", help="Path to extracted Kaggle dataset")
    parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "mobilenet_v3_small", "resnet18"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    run_experiment(args)
