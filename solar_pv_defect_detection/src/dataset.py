"""
PyTorch Dataset and Data Pipeline for Infrared Solar Module Defect Detection.
Supports CLAHE contrast enhancement, radiometric scaling, and stratified splitting.
Dataset: Raptor Maps / Kaggle 'infrared-solar-modules' (20,000 thermal images, 12 classes).
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Exact 12-class taxonomy from Raptor Maps dataset
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

CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}


class CLAHETransform:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to single-channel infrared thermal images to enhance local defect temperature gradients.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        try:
            import cv2
            np_img = np.array(img)
            if len(np_img.shape) == 3 and np_img.shape[2] == 3:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            enhanced = clahe.apply(np_img)
            return Image.fromarray(enhanced)
        except ImportError:
            # Fallback if cv2 is not installed: return original PIL image
            return img


class ExpandChannels:
    """Expands single-channel thermal image to 3 channels for backbone compatibility."""
    def __call__(self, x):
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x[:3, :, :]


def get_thermal_transforms(image_size=(224, 224), use_clahe=True, is_train=True):
    """
    Constructs the thermal image transformation pipeline.
    Preserves thermal gradient dynamics and maps 1-channel thermal to 3-channel standard tensors.
    """
    transform_list = []
    
    if use_clahe:
        transform_list.append(CLAHETransform(clip_limit=2.5, tile_grid_size=(4, 4)))

    transform_list.append(T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR))

    if is_train:
        transform_list.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=10),
        ])

    transform_list.extend([
        T.ToTensor(),
        ExpandChannels(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return T.Compose(transform_list)


class CachedSolarDataset(Dataset):
    """
    In-memory dataset loaded from pre-cached ByteTensor.
    Extremely fast: 0 disk reads, instantaneous batch slicing and augmentations.
    """
    def __init__(self, images_tensor, labels_tensor, is_train=True):
        self.images = images_tensor # (N, 1, H, W) uint8
        self.labels = labels_tensor # (N,) long
        self.is_train = is_train
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].float() / 255.0
        img = img.repeat(3, 1, 1)

        if self.is_train:
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[2])
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[1])

        img = (img - self.mean) / self.std
        return img, self.labels[idx]


class SolarThermalDataset(Dataset):
    """
    Loads cropped infrared solar module images and associated metadata labels.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(path) as img:
            img = img.convert("L") # Ensure single-channel thermal grayscale
            if self.transform is not None:
                img_tensor = self.transform(img)
            else:
                img_tensor = T.ToTensor()(img)

        return img_tensor, torch.tensor(label, dtype=torch.long)


def load_dataset_metadata(data_dir):
    """
    Parses either module_metadata.json, annotations.csv, or folder-based class structures.
    Returns: image_paths list, labels list
    """
    json_path = os.path.join(data_dir, "module_metadata.json")
    image_paths = []
    labels = []

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            meta = json.load(f)
        
        # Format can be dict keyed by image ID or list
        items = meta.values() if isinstance(meta, dict) else meta
        for item in items:
            img_rel = item.get("image_filepath", item.get("filename", ""))
            full_path = os.path.join(data_dir, img_rel) if not os.path.isabs(img_rel) else img_rel
            cls_name = item.get("anomaly_class", item.get("class", "No-Anomaly"))
            
            if os.path.exists(full_path) and cls_name in CLASS_TO_IDX:
                image_paths.append(full_path)
                labels.append(CLASS_TO_IDX[cls_name])
    else:
        # Fallback: scan class subfolders (e.g. data_dir/Cell/, data_dir/No-Anomaly/)
        for cls_name in CLASSES:
            cls_folder = os.path.join(data_dir, cls_name)
            if os.path.isdir(cls_folder):
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                        image_paths.append(os.path.join(cls_folder, fname))
                        labels.append(CLASS_TO_IDX[cls_name])

    return image_paths, labels


def create_dataloaders(data_dir, batch_size=32, image_size=(64, 64), val_split=0.15, test_split=0.15, random_seed=42):
    """
    Creates stratified Train, Validation, and Test DataLoaders.
    Automatically leverages pre-cached memory dataset if present for blazing speed.
    """
    # Check for pre-built cache in data_dir or parent of data_dir
    possible_cache_paths = [
        os.path.join(data_dir, "cached_solar_64x64.pt"),
        os.path.join(os.path.dirname(data_dir), "cached_solar_64x64.pt"),
        os.path.join(os.path.dirname(__file__), "../data/cached_solar_64x64.pt"),
    ]
    cache_path = next((p for p in possible_cache_paths if os.path.exists(p)), None)

    if cache_path is not None and image_size == (64, 64):
        print(f"[*] Fast cache detected at: {cache_path}. Loading memory dataset...", flush=True)
        cached_data = torch.load(cache_path, weights_only=False)
        images = cached_data["images"]
        labels = cached_data["labels"]
        N = len(labels)

        np.random.seed(random_seed)
        indices = np.arange(N)
        labels_np = labels.numpy()

        train_indices, val_indices, test_indices = [], [], []
        for c in range(len(CLASSES)):
            c_idxs = indices[labels_np == c]
            np.random.shuffle(c_idxs)
            n = len(c_idxs)
            n_test = int(n * test_split)
            n_val = int(n * val_split)
            test_indices.extend(c_idxs[:n_test])
            val_indices.extend(c_idxs[n_test : n_test + n_val])
            train_indices.extend(c_idxs[n_test + n_val :])

        train_ds = CachedSolarDataset(images[train_indices], labels[train_indices], is_train=True)
        val_ds = CachedSolarDataset(images[val_indices], labels[val_indices], is_train=False)
        test_ds = CachedSolarDataset(images[test_indices], labels[test_indices], is_train=False)

        pin_mem = torch.cuda.is_available()
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_mem)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)
        return train_loader, val_loader, test_loader

    # Fallback to standard disk loader
    image_paths, labels = load_dataset_metadata(data_dir)
    
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No valid images found in {data_dir}. Please ensure dataset is extracted.")

    # Stratified split using numpy
    np.random.seed(random_seed)
    indices = np.arange(len(image_paths))
    labels_np = np.array(labels)

    train_indices, val_indices, test_indices = [], [], []

    for c in range(len(CLASSES)):
        c_idxs = indices[labels_np == c]
        np.random.shuffle(c_idxs)
        n = len(c_idxs)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        
        test_indices.extend(c_idxs[:n_test])
        val_indices.extend(c_idxs[n_test : n_test + n_val])
        train_indices.extend(c_idxs[n_test + n_val :])

    train_tf = get_thermal_transforms(image_size=image_size, use_clahe=True, is_train=True)
    val_tf = get_thermal_transforms(image_size=image_size, use_clahe=True, is_train=False)

    train_ds = SolarThermalDataset([image_paths[i] for i in train_indices], [labels[i] for i in train_indices], transform=train_tf)
    val_ds = SolarThermalDataset([image_paths[i] for i in val_indices], [labels[i] for i in val_indices], transform=val_tf)
    test_ds = SolarThermalDataset([image_paths[i] for i in test_indices], [labels[i] for i in test_indices], transform=val_tf)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)

    return train_loader, val_loader, test_loader
