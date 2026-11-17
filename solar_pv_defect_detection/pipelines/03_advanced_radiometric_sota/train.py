"""
Training Script for Pipeline 3: Advanced Neuro-Radiometric SegNeXt (SOTA Proposal).
Integrates FSFOC / SFOC front-end with SegNeXt Multi-Scale Convolutional Attention (MSCA).
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
from frontends.sfoc import SFOCExtractor, FSFOCExtractor
from models.segnext_msca import MSCABlock
from loss import RadiometricBoundaryLoss


class NeuroRadiometricSegNeXt(nn.Module):
    """
    End-to-End Neuro-Radiometric Model:
    FSFOC/SFOC Feature Extractor -> SegNeXt MSCA Strip Convolutional Backbone -> Classifier Head.
    """
    def __init__(self, in_channels=4, num_classes=12, embed_dims=[32, 64, 160, 256], use_fast_sfoc=True):
        super().__init__()
        self.use_fast_sfoc = use_fast_sfoc
        if use_fast_sfoc:
            self.frontend = FSFOCExtractor(kernel_size=7, num_orientations=in_channels)
        else:
            self.frontend = SFOCExtractor(kernel_size=9)

        # SegNeXt Stem (Downsamples 4x)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dims[0] // 2, embed_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU()
        )

        # SegNeXt 4-Stage Hierarchy with MSCA Strip Convolutions
        self.stage1 = nn.Sequential(*[MSCABlock(embed_dims[0]) for _ in range(2)])
        self.down1 = nn.Sequential(nn.Conv2d(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(embed_dims[1]))

        self.stage2 = nn.Sequential(*[MSCABlock(embed_dims[1]) for _ in range(2)])
        self.down2 = nn.Sequential(nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(embed_dims[2]))

        self.stage3 = nn.Sequential(*[MSCABlock(embed_dims[2]) for _ in range(3)])
        self.down3 = nn.Sequential(nn.Conv2d(embed_dims[2], embed_dims[3], kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(embed_dims[3]))

        self.stage4 = nn.Sequential(*[MSCABlock(embed_dims[3]) for _ in range(2)])

        # Classification / Diagnosis Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dims[3], 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Extract physics-informed radiometric channels
        radiometric_feat = self.frontend(x)
        
        # Pass through SegNeXt MSCA backbone
        x = self.stem(radiometric_feat)
        x = self.down1(self.stage1(x))
        x = self.down2(self.stage2(x))
        x = self.down3(self.stage3(x))
        x = self.stage4(x)

        logits = self.head(self.pool(x))
        return logits, radiometric_feat

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Pipeline 3 (Neuro-Radiometric SegNeXt + FSFOC) running on: {device}", flush=True)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=(args.img_size, args.img_size),
        val_split=0.15,
        test_split=0.15,
        random_seed=args.seed
    )

    model = NeuroRadiometricSegNeXt(in_channels=4, num_classes=12, use_fast_sfoc=True).to(device)
    params = model.count_parameters()
    print(f"[*] NR-SegNeXt Model Loaded: {params:,} trainable parameters.", flush=True)

    criterion = RadiometricBoundaryLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    out_dir = os.path.join(args.output_dir, "03_radiometric_sota_segnext")
    os.makedirs(out_dir, exist_ok=True)
    best_ckpt = os.path.join(out_dir, "best_nr_segnext.pt")
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
                outputs, sfoc_feat = model(imgs)
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
                    outputs, sfoc_feat = model(imgs)
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
            print(f" --> Best NR-SegNeXt checkpoint saved: {best_ckpt} (Val F1: {best_f1*100:.2f}%)", flush=True)

    # Test Set Evaluation + Latency Benchmark
    print("\n[*] Evaluating Best NR-SegNeXt Checkpoint on Test Set...", flush=True)
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    test_preds, test_targets = [], []
    
    start_time = time.time()
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs, _ = model(imgs)
            test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
    elapsed = time.time() - start_time
    fps = len(test_loader.dataset) / elapsed if elapsed > 0 else 0.0

    test_metrics = compute_metrics(test_targets, test_preds, class_names=CLASSES)
    test_metrics["inference_fps"] = fps
    test_metrics["parameters"] = params

    print_evaluation_summary(test_metrics)
    print(f"[*] Inference Speed: {fps:.1f} FPS on {device.type.upper()}", flush=True)

    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    print(f"[*] Evaluation saved in: {out_dir}", flush=True)


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
