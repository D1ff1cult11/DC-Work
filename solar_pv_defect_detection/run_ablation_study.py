"""
Comprehensive Ablation & Extended Benchmark Runner.

Runs ALL experimental configurations systematically:
  A. Extended Baselines (15 epochs, class-weighted loss)
     A1: MobileNetV3-Small (baseline CNN)
     A2: ResNet18 (deeper baseline CNN)
     A3: SegFormer MiT-B0 (paper reproduction)

  B. Ablation Study (proving each component matters)
     B1: SegNeXt MSCA only (NO steerable front-end) — proves FSFOC matters
     B2: FSFOC front-end + SegNeXt MSCA (our full proposal)
     B3: SFOC (full 2D) + SegNeXt MSCA — accuracy vs FSFOC speed tradeoff
     B4: CFOG front-end + SegNeXt MSCA — alternative gradient descriptor
     B5: FSFOC + SegNeXt MSCA + Focal Loss (gamma=2) — proves loss function matters
     B6: FSFOC 8-orientation + SegNeXt MSCA — more orientations

  C. Analysis & Visualization
     - Confusion matrix heatmaps for all models
     - t-SNE feature embeddings
     - Per-class failure analysis
     - Full ablation table
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "pipelines", "01_paper_reproduction_segformer"))
sys.path.append(os.path.join(PROJECT_ROOT, "pipelines", "03_advanced_radiometric_sota"))

from src.dataset import create_dataloaders, CLASSES
from src.metrics import compute_metrics, print_evaluation_summary
from src.cfog import CFOGExtractor
from src.sfoc import SFOCExtractor, FSFOCExtractor
from src.models.segnext_msca import MSCABlock

# ============================================================================
# FOCAL LOSS (handles class imbalance better than CE)
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss with per-class alpha weighting for extreme class imbalance."""
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_term = alpha_t * focal_term
        if self.reduction == "mean":
            return focal_term.mean()
        return focal_term.sum()


# ============================================================================
# MODEL VARIANTS
# ============================================================================
def build_segnext_only(num_classes=12, embed_dims=[32, 64, 160, 256]):
    """Ablation B1: SegNeXt MSCA backbone alone, no steerable front-end."""
    class SegNeXtOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, embed_dims[0]//2, 3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dims[0]//2), nn.GELU(),
                nn.Conv2d(embed_dims[0]//2, embed_dims[0], 3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dims[0]), nn.GELU()
            )
            self.stage1 = nn.Sequential(*[MSCABlock(embed_dims[0]) for _ in range(2)])
            self.down1 = nn.Sequential(nn.Conv2d(embed_dims[0], embed_dims[1], 3, stride=2, padding=1), nn.BatchNorm2d(embed_dims[1]))
            self.stage2 = nn.Sequential(*[MSCABlock(embed_dims[1]) for _ in range(2)])
            self.down2 = nn.Sequential(nn.Conv2d(embed_dims[1], embed_dims[2], 3, stride=2, padding=1), nn.BatchNorm2d(embed_dims[2]))
            self.stage3 = nn.Sequential(*[MSCABlock(embed_dims[2]) for _ in range(3)])
            self.down3 = nn.Sequential(nn.Conv2d(embed_dims[2], embed_dims[3], 3, stride=2, padding=1), nn.BatchNorm2d(embed_dims[3]))
            self.stage4 = nn.Sequential(*[MSCABlock(embed_dims[3]) for _ in range(2)])
            self.pool = nn.AdaptiveAvgPool2d((1,1))
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(embed_dims[3], 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, num_classes))

        def forward(self, x):
            x = self.stem(x)
            x = self.down1(self.stage1(x))
            x = self.down2(self.stage2(x))
            x = self.down3(self.stage3(x))
            x = self.stage4(x)
            return self.head(self.pool(x))

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    return SegNeXtOnly()


def build_frontend_segnext(frontend_type="fsfoc", num_orientations=4, num_classes=12, embed_dims=[32, 64, 160, 256]):
    """Builds steerable front-end + SegNeXt model for ablation variants B2-B6."""
    class FrontendSegNeXt(nn.Module):
        def __init__(self):
            super().__init__()
            self.frontend_type = frontend_type
            if frontend_type == "fsfoc":
                self.frontend = FSFOCExtractor(kernel_size=7, num_orientations=num_orientations)
                in_ch = num_orientations
            elif frontend_type == "sfoc":
                self.frontend = SFOCExtractor(kernel_size=9)
                in_ch = 4
            elif frontend_type == "cfog":
                self.frontend = CFOGExtractor(num_bins=8)
                in_ch = 8
            else:
                raise ValueError(f"Unknown frontend: {frontend_type}")

            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, embed_dims[0]//2, 3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dims[0]//2), nn.GELU(),
                nn.Conv2d(embed_dims[0]//2, embed_dims[0], 3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dims[0]), nn.GELU()
            )
            self.stage1 = nn.Sequential(*[MSCABlock(embed_dims[0]) for _ in range(2)])
            self.down1 = nn.Sequential(nn.Conv2d(embed_dims[0], embed_dims[1], 3, stride=2, padding=1), nn.BatchNorm2d(embed_dims[1]))
            self.stage2 = nn.Sequential(*[MSCABlock(embed_dims[1]) for _ in range(2)])
            self.down2 = nn.Sequential(nn.Conv2d(embed_dims[1], embed_dims[2], 3, stride=2, padding=1), nn.BatchNorm2d(embed_dims[2]))
            self.stage3 = nn.Sequential(*[MSCABlock(embed_dims[2]) for _ in range(3)])
            self.down3 = nn.Sequential(nn.Conv2d(embed_dims[2], embed_dims[3], 3, stride=2, padding=1), nn.BatchNorm2d(embed_dims[3]))
            self.stage4 = nn.Sequential(*[MSCABlock(embed_dims[3]) for _ in range(2)])
            self.pool = nn.AdaptiveAvgPool2d((1,1))
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(embed_dims[3], 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, num_classes))

        def forward(self, x):
            feat = self.frontend(x)
            x = self.stem(feat)
            x = self.down1(self.stage1(x))
            x = self.down2(self.stage2(x))
            x = self.down3(self.stage3(x))
            x = self.stage4(x)
            return self.head(self.pool(x))

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    return FrontendSegNeXt()


def build_frontend_segformer(frontend_type="fsfoc", num_orientations=4, num_classes=12):
    """Ablation B7: Steerable front-end + SegFormer MiT-B0 backbone (2x2 factorial comparison)."""
    from model import SegFormerPaperModel
    class FrontendSegFormer(nn.Module):
        def __init__(self):
            super().__init__()
            if frontend_type == "fsfoc":
                self.frontend = FSFOCExtractor(kernel_size=7, num_orientations=num_orientations)
                in_ch = num_orientations
            elif frontend_type == "sfoc":
                self.frontend = SFOCExtractor(kernel_size=9)
                in_ch = 4
            elif frontend_type == "cfog":
                self.frontend = CFOGExtractor(num_bins=8)
                in_ch = 8
            else:
                raise ValueError(f"Unknown frontend: {frontend_type}")
            self.backbone = SegFormerPaperModel(in_chans=in_ch, num_classes=num_classes)

        def forward(self, x):
            feat = self.frontend(x)
            return self.backbone(feat)

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    return FrontendSegFormer()


# ============================================================================
# GENERIC TRAINER
# ============================================================================
def compute_class_weights(train_loader, num_classes=12):
    """Compute inverse-frequency class weights from training set."""
    counts = torch.zeros(num_classes)
    for _, targets in train_loader:
        for t in targets:
            counts[t.item()] += 1
    # Inverse frequency, capped
    total = counts.sum()
    weights = total / (num_classes * counts.clamp(min=1))
    weights = weights / weights.sum() * num_classes  # normalize
    return weights


def train_model(model, train_loader, val_loader, test_loader, criterion, 
                out_dir, epochs=15, lr=3e-4, device="cpu", run_name="model"):
    """Generic training loop with progress logging."""
    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[*] Training {run_name} | {params:,} params | {epochs} epochs | device: {device}", flush=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    use_amp = (device != "cpu" and device != torch.device("cpu"))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_ckpt = os.path.join(out_dir, f"best_{run_name}.pt")
    best_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    total_batches = len(train_loader)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_preds, train_targets = 0.0, [], []

        for b_idx, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(imgs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == total_batches:
                print(f"  [{run_name}] Epoch {epoch:02d}/{epochs:02d} Batch {b_idx+1:03d}/{total_batches:03d} Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_m = compute_metrics(train_targets, train_preds, class_names=CLASSES)

        # Validation
        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
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

        dur = time.time() - t0
        print(f"  [{run_name}] Epoch {epoch:02d}/{epochs:02d} ({dur:.1f}s) "
              f"TrLoss: {train_loss:.4f} TrF1: {train_m['macro_f1']*100:.2f}% "
              f"VaLoss: {val_loss:.4f} VaF1: {val_m['macro_f1']*100:.2f}%", flush=True)

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            torch.save(model.state_dict(), best_ckpt)
            print(f"  --> Best checkpoint saved (Val F1: {best_f1*100:.2f}%)", flush=True)

    # Test evaluation
    print(f"\n[*] Evaluating {run_name} on test set...", flush=True)
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    model.eval()
    test_preds, test_targets = [], []
    all_logits = []
    t_start = time.time()
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            all_logits.append(outputs.cpu())
            test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
    test_dur = time.time() - t_start
    fps = len(test_loader.dataset) / test_dur if test_dur > 0 else 0.0

    test_metrics = compute_metrics(test_targets, test_preds, class_names=CLASSES)
    test_metrics["inference_fps"] = fps
    test_metrics["parameters"] = params
    test_metrics["epochs_trained"] = epochs
    test_metrics["run_name"] = run_name
    print_evaluation_summary(test_metrics)
    print(f"[*] {run_name} Inference: {fps:.1f} FPS on {device}", flush=True)

    # Save everything
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Save predictions for failure analysis
    np.save(os.path.join(out_dir, "test_preds.npy"), np.array(test_preds))
    np.save(os.path.join(out_dir, "test_targets.npy"), np.array(test_targets))
    all_logits_cat = torch.cat(all_logits, dim=0)
    torch.save(all_logits_cat, os.path.join(out_dir, "test_logits.pt"))

    print(f"[+] {run_name} complete. Results in: {out_dir}\n", flush=True)
    return test_metrics


# ============================================================================
# MAIN ABLATION RUNNER
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs", nargs="+", default=["all"],
                        help="Which configs to run: all, baselines, ablations, or specific names like B1,B2,A1")
    args = parser.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, "data", "infrared_solar_modules")
    results_base = os.path.join(PROJECT_ROOT, "benchmark_results")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*80}", flush=True)
    print(f"  COMPREHENSIVE ABLATION & BENCHMARK RUNNER", flush=True)
    print(f"  Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}", flush=True)
    print(f"{'='*80}\n", flush=True)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir, batch_size=args.batch_size,
        image_size=(args.img_size, args.img_size), random_seed=args.seed
    )

    # Compute class weights for focal/weighted loss
    class_weights = compute_class_weights(train_loader, num_classes=12).to(device)
    print(f"[*] Class weights computed: {class_weights.cpu().tolist()}", flush=True)

    weighted_ce = nn.CrossEntropyLoss(weight=class_weights)
    focal_loss = FocalLoss(alpha=class_weights.cpu().tolist(), gamma=2.0)
    if device.type == "cuda":
        focal_loss = focal_loss.to(device)
    plain_ce = nn.CrossEntropyLoss()

    configs_to_run = args.configs
    run_all = "all" in configs_to_run
    run_baselines = run_all or "baselines" in configs_to_run
    run_ablations = run_all or "ablations" in configs_to_run

    all_results = {}

    # === A1: MobileNetV3-Small (class-weighted CE) ===
    if run_baselines or "A1" in configs_to_run:
        import torchvision.models as models
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 12)
        r = train_model(m, train_loader, val_loader, test_loader, weighted_ce,
                        os.path.join(results_base, "A1_mobilenetv3_weighted"), args.epochs, args.lr, device, "A1_MobileNetV3_WCE")
        all_results["A1_MobileNetV3_WCE"] = r

    # === A2: ResNet18 (class-weighted CE) ===
    if run_baselines or "A2" in configs_to_run:
        import torchvision.models as models
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 12)
        r = train_model(m, train_loader, val_loader, test_loader, weighted_ce,
                        os.path.join(results_base, "A2_resnet18_weighted"), args.epochs, args.lr, device, "A2_ResNet18_WCE")
        all_results["A2_ResNet18_WCE"] = r

    # === A3: SegFormer MiT-B0 (class-weighted CE) ===
    if run_baselines or "A3" in configs_to_run:
        from model import SegFormerPaperModel
        m = SegFormerPaperModel(num_classes=12)
        r = train_model(m, train_loader, val_loader, test_loader, weighted_ce,
                        os.path.join(results_base, "A3_segformer_weighted"), args.epochs, args.lr, device, "A3_SegFormer_WCE")
        all_results["A3_SegFormer_WCE"] = r

    # === B1: SegNeXt MSCA only (NO steerable front-end) ===
    if run_ablations or "B1" in configs_to_run:
        m = build_segnext_only(num_classes=12)
        r = train_model(m, train_loader, val_loader, test_loader, weighted_ce,
                        os.path.join(results_base, "B1_segnext_only"), args.epochs, args.lr, device, "B1_SegNeXt_Only")
        all_results["B1_SegNeXt_Only"] = r

    # === B2: FSFOC (4-orient) + SegNeXt (our full proposal, class-weighted) ===
    if run_ablations or "B2" in configs_to_run:
        m = build_frontend_segnext("fsfoc", num_orientations=4, num_classes=12)
        r = train_model(m, train_loader, val_loader, test_loader, weighted_ce,
                        os.path.join(results_base, "B2_fsfoc4_segnext_weighted"), args.epochs, args.lr, device, "B2_FSFOC4_SegNeXt_WCE")
        all_results["B2_FSFOC4_SegNeXt_WCE"] = r

    # === B3: SFOC (full 2D) + SegNeXt ===
    if run_ablations or "B3" in configs_to_run:
        m = build_frontend_segnext("sfoc", num_classes=12)
        r = train_model(m, train_loader, val_loader, test_loader, weighted_ce,
                        os.path.join(results_base, "B3_sfoc_segnext"), args.epochs, args.lr, device, "B3_SFOC_SegNeXt")
        all_results["B3_SFOC_SegNeXt"] = r

    # === B4: CFOG + SegNeXt ===
    if run_ablations or "B4" in configs_to_run:
        m = build_frontend_segnext("cfog", num_classes=12)
        r = train_model(m, train_loader, val_loader, test_loader, weighted_ce,
                        os.path.join(results_base, "B4_cfog_segnext"), args.epochs, args.lr, device, "B4_CFOG_SegNeXt")
        all_results["B4_CFOG_SegNeXt"] = r

    # === B5: FSFOC + SegNeXt + Focal Loss (gamma=2) ===
    if run_ablations or "B5" in configs_to_run:
        m = build_frontend_segnext("fsfoc", num_orientations=4, num_classes=12)
        r = train_model(m, train_loader, val_loader, test_loader, focal_loss,
                        os.path.join(results_base, "B5_fsfoc_segnext_focal"), args.epochs, args.lr, device, "B5_FSFOC_SegNeXt_Focal")
        all_results["B5_FSFOC_SegNeXt_Focal"] = r

    # === B6: FSFOC 8-orientation + SegNeXt ===
    if run_ablations or "B6" in configs_to_run:
        m = build_frontend_segnext("fsfoc", num_orientations=8, num_classes=12)
        r = train_model(m, train_loader, val_loader, test_loader, weighted_ce,
                        os.path.join(results_base, "B6_fsfoc8_segnext"), args.epochs, args.lr, device, "B6_FSFOC8_SegNeXt")
        all_results["B6_FSFOC8_SegNeXt"] = r

    # === B7: FSFOC 4-orientation + SegFormer (2x2 factorial check) ===
    if run_ablations or "B7" in configs_to_run:
        m = build_frontend_segformer("fsfoc", num_orientations=4, num_classes=12)
        r = train_model(m, train_loader, val_loader, test_loader, weighted_ce,
                        os.path.join(results_base, "B7_fsfoc_segformer"), args.epochs, args.lr, device, "B7_FSFOC_SegFormer")
        all_results["B7_FSFOC_SegFormer"] = r

    # === FINAL SUMMARY ===
    print(f"\n{'='*120}", flush=True)
    print(f"  COMPLETE ABLATION RESULTS SUMMARY", flush=True)
    print(f"{'='*120}", flush=True)
    print(f"{'Config':<30} | {'Acc':>7} | {'MacroF1':>8} | {'Prec':>7} | {'Recall':>7} | {'Kappa':>7} | {'Params':>10} | {'FPS':>8}", flush=True)
    print("-" * 120, flush=True)
    for name, r in all_results.items():
        print(f"{name:<30} | {r['accuracy']*100:6.2f}% | {r['macro_f1']*100:7.2f}% | {r['macro_precision']*100:6.2f}% | "
              f"{r['macro_recall']*100:6.2f}% | {r['cohens_kappa']:6.4f} | {r['parameters']:>10,} | {r['inference_fps']:7.1f}", flush=True)
    print("=" * 120, flush=True)

    # Save master summary
    summary_path = os.path.join(results_base, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n[+] Ablation summary saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
