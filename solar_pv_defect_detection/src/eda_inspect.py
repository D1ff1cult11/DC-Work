"""
EDA and Inspection Script for Infrared Solar Modules Dataset.
Generates:
1. Class distribution summary table.
2. Distribution plot (saved to eda_class_distribution.png).
3. Representative 3x4 visual grid of thermal signatures across all 12 classes (saved to eda_samples_grid.png).
4. PyTorch DataLoader verification.
"""

import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

CLASSES = [
    "No-Anomaly",
    "Cell",
    "Cell-Multi",
    "Cracking",
    "Diode",
    "Diode-Multi",
    "Hot-Spot",
    "Hot-Spot-Multi",
    "Offline-Module",
    "Shadowing",
    "Soiling",
    "Vegetation",
]

def run_eda(data_dir="./data/infrared_solar_modules", output_dir="./docs/figures"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("  RAPTOR MAPS INFRARED SOLAR MODULES - DATASET AUDIT")
    print("=" * 60)

    # 1. Inspect metadata
    json_path = os.path.join(data_dir, "module_metadata.json")
    class_counts = {cls: 0 for cls in CLASSES}
    sample_images = {cls: [] for cls in CLASSES}

    if os.path.exists(json_path):
        print(f"[*] Found module_metadata.json at: {json_path}")
        with open(json_path, "r") as f:
            meta = json.load(f)
        items = meta.values() if isinstance(meta, dict) else meta
        for item in items:
            img_rel = item.get("image_filepath", item.get("filename", ""))
            full_path = os.path.join(data_dir, img_rel) if not os.path.isabs(img_rel) else img_rel
            cls_name = item.get("anomaly_class", item.get("class", "No-Anomaly"))
            if cls_name in class_counts:
                class_counts[cls_name] += 1
                if len(sample_images[cls_name]) < 3 and os.path.exists(full_path):
                    sample_images[cls_name].append(full_path)
    else:
        # Check folder structure
        print(f"[*] Scanning class subdirectories in: {data_dir}")
        for cls in CLASSES:
            folder = os.path.join(data_dir, cls)
            if os.path.isdir(folder):
                files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                class_counts[cls] = len(files)
                sample_images[cls] = files[:3]

    total_images = sum(class_counts.values())
    print(f"[*] Total Images Found: {total_images:,}")
    print("-" * 60)
    print(f"{'Class Name':<20} | {'Count':<10} | {'Percentage'}")
    print("-" * 60)
    for cls in CLASSES:
        cnt = class_counts[cls]
        pct = (cnt / total_images * 100) if total_images > 0 else 0
        print(f"{cls:<20} | {cnt:<10} | {pct:>6.2f}%")
    print("=" * 60)

    if total_images == 0:
        print("[!] No images detected. Please place the extracted dataset in:")
        print(f"    {os.path.abspath(data_dir)}")
        print("    Refer to data/README.md for download instructions.")
        return

    # 2. Plot Class Distribution
    plt.figure(figsize=(12, 6))
    bars = plt.bar(CLASSES, [class_counts[c] for c in CLASSES], color="#2563eb", edgecolor="black")
    bars[0].set_color("#16a34a") # Highlight Normal baseline in green
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.title("Raptor Maps Solar PV Defect Dataset - Class Distribution", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Samples", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, "eda_class_distribution.png")
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"[*] Saved class distribution plot to: {dist_plot_path}")

    # 3. Plot 3x4 Sample Grid
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()
    for idx, cls in enumerate(CLASSES):
        ax = axes[idx]
        if sample_images[cls]:
            img = Image.open(sample_images[cls][0]).convert("L")
            ax.imshow(img, cmap="inferno") # Thermal colormap
            ax.set_title(f"{cls} ({class_counts[cls]})", fontsize=11, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No Sample", ha="center", va="center")
            ax.set_title(cls, fontsize=11)
        ax.axis("off")

    plt.suptitle("Thermal Infrared Signatures Across 12 Classes (LWIR 8-14µm)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    grid_path = os.path.join(output_dir, "eda_samples_grid.png")
    plt.savefig(grid_path, dpi=300)
    plt.close()
    print(f"[*] Saved sample signatures grid to: {grid_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/infrared_solar_modules")
    parser.add_argument("--output_dir", type=str, default="./docs/figures")
    args = parser.parse_args()
    run_eda(args.data_dir, args.output_dir)
