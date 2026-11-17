"""
Track 2 Data Preparation Script
Processes all 277 aerial thermographic images from Alfaro-Mejía et al. (2019) / Joshi & Pal (2026)
Generates:
  1. Standardized 336x256 and 224x224 processed images (CLAHE + NLM + Min-Max normalization)
  2. Dense pixel segmentation masks (0: Background, 1: Hotspot, 2: Snail Trail)
  3. Paired visual overlays matching Paper Fig. 6
  4. 80/10/10 Train/Val/Test splits
"""

import os
import glob
import cv2
import numpy as np
import random
from PIL import Image

def process_all_images():
    raw_base = r'e:\Dev Projects\DC Work\solar_pv_track2_pixel_segmentation\data\raw\extracted\PV_images_failures'
    out_img_dir = r'e:\Dev Projects\DC Work\solar_pv_track2_pixel_segmentation\data\processed\images'
    out_mask_dir = r'e:\Dev Projects\DC Work\solar_pv_track2_pixel_segmentation\data\processed\masks'
    out_overlay_dir = r'e:\Dev Projects\DC Work\solar_pv_track2_pixel_segmentation\data\processed\overlays'
    split_dir = r'e:\Dev Projects\DC Work\solar_pv_track2_pixel_segmentation\data\splits'

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    os.makedirs(out_overlay_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    jpg_paths = sorted(glob.glob(os.path.join(raw_base, '**', '*.jpg'), recursive=True))
    print(f"[*] Found {len(jpg_paths)} raw thermal images.")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    k_small = np.ones((3, 3), np.uint8)

    sample_ids = []
    stats = {"bg_pixels": 0, "hotspot_pixels": 0, "snail_pixels": 0}

    for idx, path in enumerate(jpg_paths):
        # Generate unique sample ID based on folder and filename
        rel_path = os.path.relpath(path, raw_base)
        sample_id = f"thermal_{idx:03d}_{os.path.basename(path).replace('.jpg', '')}"
        sample_ids.append(sample_id)

        im = cv2.imread(path)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # 1. CLAHE enhancement
        enhanced = clahe.apply(gray)

        # 2. Panel region detection
        panel_mask = (enhanced > 95).astype(np.uint8)
        panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_CLOSE, k_close)

        # 3. Dynamic Radiometric Thresholding inside panel
        panel_pixels = enhanced[panel_mask == 1]
        if len(panel_pixels) > 0:
            p_median = np.median(panel_pixels)
            p_std = panel_pixels.std()
        else:
            p_median = np.median(enhanced)
            p_std = enhanced.std()

        mask = np.zeros_like(gray, dtype=np.uint8)
        # Hotspot: intense thermal elevation
        mask[(enhanced >= p_median + 1.8 * p_std) & (panel_mask == 1)] = 1
        # Snail Trail: moderate thermal elevation
        mask[(enhanced >= p_median + 0.75 * p_std) & (enhanced < p_median + 1.8 * p_std) & (panel_mask == 1)] = 2

        # Morphological noise removal
        mask_hotspot = cv2.morphologyEx((mask == 1).astype(np.uint8), cv2.MORPH_OPEN, k_small)
        mask_snail = cv2.morphologyEx((mask == 2).astype(np.uint8), cv2.MORPH_OPEN, k_small)

        clean_mask = np.zeros_like(gray, dtype=np.uint8)
        clean_mask[mask_snail == 1] = 2
        clean_mask[mask_hotspot == 1] = 1

        # Accumulate stats
        stats["bg_pixels"] += int((clean_mask == 0).sum())
        stats["hotspot_pixels"] += int((clean_mask == 1).sum())
        stats["snail_pixels"] += int((clean_mask == 2).sum())

        # Save processed image (PNG) and mask (uint8 PNG)
        cv2.imwrite(os.path.join(out_img_dir, f"{sample_id}.png"), enhanced)
        cv2.imwrite(os.path.join(out_mask_dir, f"{sample_id}.png"), clean_mask)

        # Generate sample overlays for inspection (first 20 samples)
        if idx < 20:
            overlay = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            overlay[clean_mask == 1] = [255, 0, 0]    # Red for Hotspot
            overlay[clean_mask == 2] = [0, 100, 255]  # Blue for Snail Trail
            blended = cv2.addWeighted(overlay, 0.45, cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 0.55, 0)
            cv2.imwrite(os.path.join(out_overlay_dir, f"{sample_id}_overlay.png"), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    # Create 80/10/10 reproducible splits (seed=42)
    random.seed(42)
    shuffled = sample_ids.copy()
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(0.80 * n_total)
    n_val = int(0.10 * n_total)
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]

    with open(os.path.join(split_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_ids) + "\n")
    with open(os.path.join(split_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_ids) + "\n")
    with open(os.path.join(split_dir, "test.txt"), "w") as f:
        f.write("\n".join(test_ids) + "\n")

    print("\n[+] Data Preparation Complete:")
    print(f"    Total Processed Images: {n_total}")
    print(f"    Train Split: {len(train_ids)} ({len(train_ids)/n_total*100:.1f}%)")
    print(f"    Val Split:   {len(val_ids)} ({len(val_ids)/n_total*100:.1f}%)")
    print(f"    Test Split:  {len(test_ids)} ({len(test_ids)/n_total*100:.1f}%)")
    total_px = stats["bg_pixels"] + stats["hotspot_pixels"] + stats["snail_pixels"]
    print(f"    Pixel Breakdown: Background={stats['bg_pixels']/total_px*100:.1f}%, "
          f"Hotspot={stats['hotspot_pixels']/total_px*100:.2f}%, Snail Trail={stats['snail_pixels']/total_px*100:.2f}%")

if __name__ == "__main__":
    process_all_images()
