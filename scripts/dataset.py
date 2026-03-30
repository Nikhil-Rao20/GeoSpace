"""
dataset.py — SVAMITVADataset + CAS sampler.

Handles:
  - Reading 3-channel or 4-channel GeoTIFF images (always uses first 3 bands)
  - Converting 9-channel raw masks → 7-class single-channel label map
  - Albumentations augmentations (train / val)
  - Class-Aware Sampling (CAS) repeat-factor weights
"""

import os
import json
import math
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


# ── Mask Conversion ──────────────────────────────────────────────────────────

def convert_9ch_to_7class(mask_9ch: np.ndarray) -> np.ndarray:
    """
    Convert the 9-channel raw rasterised mask to a single-channel 7-class label.

    Channel → Class mapping:
        Ch0 > 0                         → 1  Building
        Ch1 > 0  AND  Ch1 != 1          → 2  Road       (code 1 dropped: 4 patches = noise)
        Ch3 > 0                         → 3  Waterbody polygon
        Ch4 > 0                         → 4  Waterbody line
        Ch7 > 0                         → 5  Utility polygon
        Ch8 > 0                         → 6  Bridge
        everything else                 → 0  Background

    Priority (last write wins — physically motivated):
        Background → Building → Road → Waterbody_poly → Waterbody_line
                   → Utility_poly → Bridge

    Args:
        mask_9ch: np.ndarray shape (9, H, W), dtype uint8

    Returns:
        label: np.ndarray shape (H, W), dtype uint8, values in [0, 6]
    """
    label = np.zeros((mask_9ch.shape[1], mask_9ch.shape[2]), dtype=np.uint8)

    label[mask_9ch[0] > 0]                                    = 1  # Building
    label[(mask_9ch[1] > 0) & (mask_9ch[1] != 1)]             = 2  # Road (drop code 1)
    label[mask_9ch[3] > 0]                                    = 3  # Waterbody polygon
    label[mask_9ch[4] > 0]                                    = 4  # Waterbody line
    label[mask_9ch[7] > 0]                                    = 5  # Utility polygon
    label[mask_9ch[8] > 0]                                    = 6  # Bridge

    return label


# ── Augmentations ────────────────────────────────────────────────────────────

def get_train_transforms(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Augmentation pipeline for training.
    All transforms are applied identically to image and mask.
    Colour jitter is applied to image only (albumentations handles this).
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Colour augmentation — mild for drone orthophotos
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.4
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3
        ),
        # Blur — simulates focus variation
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # Dropout — regularisation, forces context reasoning
        A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32,
            min_holes=1, min_height=8, min_width=8,
            fill_value=0, mask_fill_value=0, p=0.2
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Validation: normalise only — no spatial or colour augmentation."""
    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# ── Dataset ──────────────────────────────────────────────────────────────────

class SVAMITVADataset(Dataset):
    """
    Loads 512 × 512 image patches and corresponding 9-channel masks.
    Converts masks to 7-class label map on-the-fly.

    Directory structure expected:
        base_dir/
            images/   *.tif   (3-band or 4-band uint8 GeoTIFF)
            masks/    *.tif   (9-band uint8 GeoTIFF, same filename as image)
            train.txt         (one filename per line)
            val.txt
    """

    def __init__(self, base_dir: str, split_file: str, transform=None):
        self.img_dir  = os.path.join(base_dir, "images")
        self.mask_dir = os.path.join(base_dir, "masks")
        self.transform = transform

        with open(split_file) as f:
            self.patches = [ln.strip() for ln in f if ln.strip()]

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        fname = self.patches[idx]

        # ── Load image ────────────────────────────────────────────────────
        with rasterio.open(os.path.join(self.img_dir, fname)) as src:
            img = src.read()[:3]          # (3, H, W) — always discard band 4
        img = img.transpose(1, 2, 0)      # → (H, W, 3) for albumentations, uint8

        # ── Load mask ─────────────────────────────────────────────────────
        with rasterio.open(os.path.join(self.mask_dir, fname)) as src:
            mask_9ch = src.read()         # (9, H, W) uint8
        label = convert_9ch_to_7class(mask_9ch)   # (H, W) uint8

        # ── Augment ───────────────────────────────────────────────────────
        if self.transform is not None:
            out   = self.transform(image=img, mask=label)
            img   = out["image"]   # (3, H, W) float32 tensor, normalised
            label = out["mask"]    # (H, W) tensor
        else:
            img   = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            label = torch.from_numpy(label)

        return img, label.long()


# ── CAS (Class-Aware Sampling) ────────────────────────────────────────────────

def build_cas_sampler(
    base_dir: str,
    train_patches: list,
    num_classes: int   = 7,
    threshold: float   = 0.01,
    max_repeat: float  = 15.0,
    cache_path: str    = None,
) -> WeightedRandomSampler:
    """
    Builds a WeightedRandomSampler using Class-Aware Sampling repeat factors.

    For each training patch, the repeat factor is:
        r(patch) = min(max_repeat, max(1, sqrt(T / f_rarest_class)))

    where:
        T         = CAS threshold (default 0.01)
        f_rarest  = fraction of training patches containing the rarest class
                    present in this patch (background excluded)

    A patch containing only background gets repeat factor 1.
    A patch containing bridge (0.1 % prevalence) gets ~max_repeat.

    The resulting weights are fed to WeightedRandomSampler so that
    rare-class patches are seen more often per epoch without duplicating
    the entire dataset in memory.

    Reference: Ha-Hieu Pham et al., CXR-LT 2026 (arXiv:2602.13430).

    Args:
        base_dir:      dataset root (contains masks/)
        train_patches: list of filenames from train.txt
        num_classes:   number of output classes (7)
        threshold:     CAS threshold T
        max_repeat:    cap on repeat factor to prevent extreme oversampling
        cache_path:    if provided, save/load weights to avoid re-scanning

    Returns:
        WeightedRandomSampler — drop-in replacement for a regular sampler
    """
    mask_dir = os.path.join(base_dir, "masks")

    # ── Try loading from cache ────────────────────────────────────────────
    if cache_path and os.path.exists(cache_path):
        print(f"[CAS] Loading cached weights → {cache_path}")
        with open(cache_path) as f:
            cache = json.load(f)
        weights = cache["weights"]
        _print_cas_stats(weights, cache.get("class_freq", {}))
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )

    # ── Scan all training patches ─────────────────────────────────────────
    print("[CAS] Scanning training patches to compute repeat factors …")
    total = len(train_patches)

    # patch_classes[i] = set of class ids present (excl. background)
    patch_classes       = [set() for _ in range(total)]
    class_patch_count   = np.zeros(num_classes, dtype=np.int64)

    for i, fname in enumerate(tqdm(train_patches, desc="[CAS] scan", ncols=80)):
        mask_path = os.path.join(mask_dir, fname)
        with rasterio.open(mask_path) as src:
            raw = src.read()
        label   = convert_9ch_to_7class(raw)
        present = set(int(c) for c in np.unique(label)) - {0}
        patch_classes[i] = present
        for c in present:
            class_patch_count[c] += 1

    # Class frequency: proportion of patches containing each class
    class_freq      = class_patch_count / total        # (num_classes,)
    class_freq_dict = {i: float(class_freq[i]) for i in range(num_classes)}

    # ── Compute per-patch repeat factor ───────────────────────────────────
    weights = []
    for i in range(total):
        cls_set = patch_classes[i]
        if not cls_set:
            weights.append(1.0)
            continue
        f_min  = min(class_freq[c] for c in cls_set if class_freq[c] > 0)
        repeat = float(min(max_repeat, max(1.0, math.sqrt(threshold / f_min))))
        weights.append(repeat)

    # ── Cache ─────────────────────────────────────────────────────────────
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"weights": weights, "class_freq": class_freq_dict}, f, indent=2)
        print(f"[CAS] Weights cached → {cache_path}")

    _print_cas_stats(weights, class_freq_dict)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def _print_cas_stats(weights, class_freq: dict):
    w = np.array(weights)
    print(
        f"[CAS] Repeat factor — "
        f"min: {w.min():.2f}  max: {w.max():.2f}  mean: {w.mean():.2f}  "
        f"patches > 5×: {(w > 5).sum()}"
    )
    if class_freq:
        print("[CAS] Class frequencies (fraction of patches):")
        names = ["Background", "Building", "Road", "Waterbody_poly",
                 "Waterbody_line", "Utility_poly", "Bridge"]
        for i, freq in class_freq.items():
            name = names[int(i)] if int(i) < len(names) else f"class_{i}"
            print(f"  {name:<20}: {float(freq):.4f}")
