"""
Track 2 Segmentation Dataset & DataLoader
Provides paired thermal image and dense 2D pixel mask loading with data augmentations.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

CLASSES = ["Background", "Hotspot", "Snail_Trail"]
NUM_CLASSES = len(CLASSES)

class SolarThermalSegmentationDataset(Dataset):
    def __init__(self, data_dir, split="train", img_size=(224, 224), augment=True):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size  # (H, W)
        self.augment = augment and (split == "train")

        split_file = os.path.join(data_dir, "splits", f"{split}.txt")
        with open(split_file, "r") as f:
            self.sample_ids = [line.strip() for line in f if line.strip()]

        self.img_dir = os.path.join(data_dir, "processed", "images")
        self.mask_dir = os.path.join(data_dir, "processed", "masks")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        img_path = os.path.join(self.img_dir, f"{sid}.png")
        mask_path = os.path.join(self.mask_dir, f"{sid}.png")

        # Load image (single channel grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise FileNotFoundError(f"Missing sample: {img_path} or {mask_path}")

        # Resize to standardized dimensions (e.g. 224x224 matching paper)
        if (img.shape[0], img.shape[1]) != self.img_size:
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

        # Paired Data Augmentations (Section 3.2 in paper)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            # Random vertical flip
            if random.random() > 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)
            # Random 90-degree rotation
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()

        # Min-Max Normalization to [0, 1] float tensor
        img_tensor = torch.from_numpy(img).float() / 255.0
        # Add channel dimension: (1, H, W)
        img_tensor = img_tensor.unsqueeze(0)

        # Target mask as long tensor: (H, W)
        mask_tensor = torch.from_numpy(mask).long()

        return img_tensor, mask_tensor, sid

import random

def create_segmentation_dataloaders(data_dir, batch_size=8, img_size=(224, 224), num_workers=0):
    train_ds = SolarThermalSegmentationDataset(data_dir, split="train", img_size=img_size, augment=True)
    val_ds = SolarThermalSegmentationDataset(data_dir, split="val", img_size=img_size, augment=False)
    test_ds = SolarThermalSegmentationDataset(data_dir, split="test", img_size=img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
