"""
Pre-caches the 20,000 solar thermal images into a single compact binary tensor (uint8, ~82 MB).
Eliminates 20,000 individual disk file reads per epoch, reducing epoch time from 2 minutes to ~6 seconds.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import CLASSES, CLASS_TO_IDX

def create_cache(data_dir, output_file, img_size=(64, 64)):
    print(f"[*] Building high-speed memory cache for dataset at: {data_dir}", flush=True)
    t0 = time.time()
    
    json_path = os.path.join(data_dir, "module_metadata.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing {json_path}")

    with open(json_path, "r") as f:
        meta = json.load(f)

    img_dir = os.path.join(data_dir, "images")
    existing_files = set(os.listdir(img_dir)) if os.path.isdir(img_dir) else set()

    items = list(meta.values() if isinstance(meta, dict) else meta)
    valid_items = []
    for item in items:
        img_rel = item.get("image_filepath", item.get("filename", ""))
        fname = os.path.basename(img_rel)
        cls_name = item.get("anomaly_class", item.get("class", "No-Anomaly"))
        if fname in existing_files and cls_name in CLASS_TO_IDX:
            full_path = os.path.join(data_dir, img_rel) if not os.path.isabs(img_rel) else img_rel
            valid_items.append((full_path, CLASS_TO_IDX[cls_name]))

    N = len(valid_items)
    print(f"[*] Found {N} valid images. Resizing to {img_size} and assembling ByteTensor...", flush=True)

    resize_tf = T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR)

    # Pre-allocate contiguous ByteTensor: (N, 1, H, W)
    images_tensor = torch.empty((N, 1, img_size[0], img_size[1]), dtype=torch.uint8)
    labels_tensor = torch.empty((N,), dtype=torch.long)

    step = max(1, N // 10)
    for i, (path, label) in enumerate(valid_items):
        with Image.open(path) as img:
            img = img.convert("L")
            resized = resize_tf(img)
            # Convert PIL to uint8 tensor (0..255)
            arr = torch.from_numpy(np.frombuffer(resized.tobytes(), dtype=np.uint8).copy())
            images_tensor[i, 0] = arr.view(img_size[0], img_size[1])
            labels_tensor[i] = label

        if (i + 1) % step == 0 or (i + 1) == N:
            print(f"  Processed {i+1}/{N} images ({(i+1)/N*100:.1f}%) in {time.time()-t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cache_dict = {
        "images": images_tensor,
        "labels": labels_tensor,
        "classes": CLASSES,
        "img_size": img_size
    }
    torch.save(cache_dict, output_file)
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[+] Cache generated successfully: {output_file} ({size_mb:.2f} MB) in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    data_dir = r"E:\Dev Projects\DC Work\solar_pv_defect_detection\data\infrared_solar_modules"
    cache_file = r"E:\Dev Projects\DC Work\solar_pv_defect_detection\data\cached_solar_64x64.pt"
    create_cache(data_dir, cache_file, img_size=(64, 64))
