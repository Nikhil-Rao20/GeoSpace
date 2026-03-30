"""
stage2_dataset.py — ROI extraction + classifier datasets for Stage-2.

Workflow:
    1. ROIExtractor scans all training patches, finds connected components
       per class (Building / Road / Waterbody_poly), crops the RGB image
       at each component's bounding box, and saves a flat dataset of
       (crop_path, label) records to a JSON cache file.

    2. ClassifierDataset reads the JSON cache and serves (crop_tensor, label)
       pairs with augmentation.

One-time extraction:  ~10 min for 12,552 patches.
After that, training reads from the cached crops directory.
"""

import os
import json
import numpy as np
import rasterio
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Subclass schemas ──────────────────────────────────────────────────────────

# Building: Ch0 codes → class index
BUILDING_CODE_MAP = {1: 0, 2: 1, 3: 2, 4: 3}
BUILDING_NAMES    = ["RCC", "Tiled", "Tin", "Others"]

# Road: Ch1 codes → class index  (code 1 dropped)
ROAD_CODE_MAP = {2: 0, 3: 1, 4: 2, 5: 3}
ROAD_NAMES    = ["Type-2", "Type-3", "Type-4", "Type-5"]

# Waterbody polygon: Ch3 codes → merged class index
#   code 4            → 0  (dominant large water body)
#   code 1            → 1
#   code 2            → 2
#   codes 3, 5, 6     → 3  (merged "Other")
WATERBODY_CODE_MAP = {4: 0, 1: 1, 2: 2, 3: 3, 5: 3, 6: 3}
WATERBODY_NAMES    = ["Type-4", "Type-1", "Type-2", "Other(3+5+6)"]

CLASSIFIER_CONFIG = {
    "building": {
        "channel":      0,
        "code_map":     BUILDING_CODE_MAP,
        "class_names":  BUILDING_NAMES,
        "num_classes":  4,
        "min_pixels":   50,    # minimum component area to keep
        "crop_size":    128,   # resize crop to this square
    },
    "road": {
        "channel":      1,
        "code_map":     ROAD_CODE_MAP,
        "class_names":  ROAD_NAMES,
        "num_classes":  4,
        "min_pixels":   200,
        "crop_size":    128,
        # Road: crop along the road axis — we take a fixed-size window
        # centred on each component centroid rather than the full bbox
        "centroid_crop": True,
        "centroid_size": 128,
    },
    "waterbody": {
        "channel":      3,
        "code_map":     WATERBODY_CODE_MAP,
        "class_names":  WATERBODY_NAMES,
        "num_classes":  4,
        "min_pixels":   100,
        "crop_size":    128,
    },
}


# ── ROI Extractor ─────────────────────────────────────────────────────────────

class ROIExtractor:
    """
    Extracts and saves labelled crop images for each Stage-2 classifier.

    Saved structure:
        crops_dir/
            building/   *.png   (one per detected component)
            road/       *.png
            waterbody/  *.png
        crops_dir/building_records.json   [{path, label, code, village}, ...]
        crops_dir/road_records.json
        crops_dir/waterbody_records.json
    """

    def __init__(
        self,
        base_dir:  str,
        crops_dir: str,
        split_file: str = None,
    ):
        self.img_dir  = os.path.join(base_dir, "images")
        self.mask_dir = os.path.join(base_dir, "masks")
        self.crops_dir = crops_dir

        if split_file is None:
            split_file = os.path.join(base_dir, "train.txt")
        with open(split_file) as f:
            self.patches = [ln.strip() for ln in f if ln.strip()]

        for cls in CLASSIFIER_CONFIG:
            Path(os.path.join(crops_dir, cls)).mkdir(parents=True, exist_ok=True)

    def extract_all(self, force: bool = False):
        """
        Extract crops for all three classifiers.
        Skips if JSON records already exist unless force=True.
        """
        for cls_name in CLASSIFIER_CONFIG:
            rec_path = os.path.join(self.crops_dir, f"{cls_name}_records.json")
            if os.path.exists(rec_path) and not force:
                n = len(json.load(open(rec_path)))
                print(f"[ROI] {cls_name}: cached ({n} crops) — skipping. "
                      f"Use force=True to re-extract.")
                continue
            self._extract_one_class(cls_name)

    def _extract_one_class(self, cls_name: str):
        cfg     = CLASSIFIER_CONFIG[cls_name]
        ch      = cfg["channel"]
        cmap    = cfg["code_map"]
        minpx   = cfg["min_pixels"]
        csize   = cfg["crop_size"]
        centcrop = cfg.get("centroid_crop", False)
        centsz  = cfg.get("centroid_size", csize)
        out_dir  = os.path.join(self.crops_dir, cls_name)
        records  = []

        print(f"\n[ROI] Extracting {cls_name} crops from {len(self.patches):,} patches …")
        crop_idx = 0

        for fname in tqdm(self.patches, desc=f"[ROI:{cls_name}]", ncols=80):
            # ── Load ──────────────────────────────────────────────────────
            img_path  = os.path.join(self.img_dir,  fname)
            mask_path = os.path.join(self.mask_dir, fname)

            with rasterio.open(img_path) as src:
                img = src.read()[:3].transpose(1, 2, 0)   # (H, W, 3) uint8

            with rasterio.open(mask_path) as src:
                raw_mask = src.read()                      # (9, H, W)

            ch_mask = raw_mask[ch]                         # (H, W) uint8

            # ── Per code in this channel ───────────────────────────────
            for code, label_idx in cmap.items():
                binary = (ch_mask == code).astype(np.uint8) * 255
                if binary.sum() == 0:
                    continue

                # Connected components
                n_comp, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(
                    binary, connectivity=8
                )

                for comp_id in range(1, n_comp):           # skip background (0)
                    area = stats[comp_id, cv2.CC_STAT_AREA]
                    if area < minpx:
                        continue

                    if centcrop:
                        # Fixed-size square centred on centroid
                        cx, cy = int(centroids[comp_id][0]), int(centroids[comp_id][1])
                        half   = centsz // 2
                        x0 = max(0, cx - half); y0 = max(0, cy - half)
                        x1 = min(img.shape[1], x0 + centsz)
                        y1 = min(img.shape[0], y0 + centsz)
                        crop = img[y0:y1, x0:x1]
                    else:
                        # Bounding-box crop
                        x0 = stats[comp_id, cv2.CC_STAT_LEFT]
                        y0 = stats[comp_id, cv2.CC_STAT_TOP]
                        w  = stats[comp_id, cv2.CC_STAT_WIDTH]
                        h  = stats[comp_id, cv2.CC_STAT_HEIGHT]
                        # Add 10 % context padding
                        pad = max(4, int(0.10 * max(w, h)))
                        x0p = max(0, x0 - pad); y0p = max(0, y0 - pad)
                        x1p = min(img.shape[1], x0 + w + pad)
                        y1p = min(img.shape[0], y0 + h + pad)
                        crop = img[y0p:y1p, x0p:x1p]

                    if crop.size == 0:
                        continue

                    # Resize to fixed square
                    crop_resized = cv2.resize(
                        crop, (csize, csize), interpolation=cv2.INTER_LINEAR
                    )

                    # Save
                    village = fname.rsplit("_", 2)[0]
                    crop_name = f"{cls_name}_{crop_idx:07d}.png"
                    crop_path = os.path.join(out_dir, crop_name)
                    cv2.imwrite(crop_path, cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))

                    records.append({
                        "path":    os.path.join(cls_name, crop_name),
                        "label":   int(label_idx),
                        "code":    int(code),
                        "village": village,
                    })
                    crop_idx += 1

        rec_path = os.path.join(self.crops_dir, f"{cls_name}_records.json")
        with open(rec_path, "w") as f:
            json.dump(records, f, indent=2)

        # Print summary
        counter = defaultdict(int)
        for r in records: counter[r["label"]] += 1
        cfg = CLASSIFIER_CONFIG[cls_name]
        print(f"[ROI] {cls_name}: {len(records):,} crops saved → {rec_path}")
        print(f"[ROI]   Class breakdown:")
        for idx, name in enumerate(cfg["class_names"]):
            print(f"[ROI]     {idx} ({name}): {counter[idx]:,}")


# ── Stage-2 Classifier Dataset ────────────────────────────────────────────────

class ClassifierDataset(Dataset):
    """
    Reads from the pre-extracted crop records.

    Args:
        crops_dir:  root directory containing cls_name/ subdirs and JSON files
        cls_name:   "building" | "road" | "waterbody"
        records:    pre-loaded list of record dicts (if None, loads from JSON)
        transform:  albumentations transform
    """

    def __init__(
        self,
        crops_dir:  str,
        cls_name:   str,
        records:    list = None,
        transform   = None,
    ):
        self.crops_dir = crops_dir
        self.cls_name  = cls_name
        self.transform = transform

        if records is None:
            rec_path = os.path.join(crops_dir, f"{cls_name}_records.json")
            with open(rec_path) as f:
                records = json.load(f)
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec   = self.records[idx]
        img_path = os.path.join(self.crops_dir, rec["path"])
        img   = np.array(Image.open(img_path).convert("RGB"))   # (H, W, 3)
        label = int(rec["label"])

        if self.transform:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img, torch.tensor(label, dtype=torch.long)


# ── Augmentations ─────────────────────────────────────────────────────────────

def get_cls_train_transforms(size=128,
                              mean=(0.485,0.456,0.406),
                              std=(0.229,0.224,0.225)):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                           rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.25,
                                   contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                             val_shift_limit=10, p=0.3),
        A.GaussNoise(var_limit=(5, 30), p=0.2),
        A.CoarseDropout(max_holes=4, max_height=size//8,
                        max_width=size//8, p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_cls_val_transforms(mean=(0.485,0.456,0.406),
                            std=(0.229,0.224,0.225)):
    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# ── Weighted sampler for imbalanced classifiers ────────────────────────────────

def build_classifier_sampler(records: list, num_classes: int) -> WeightedRandomSampler:
    """Inverse-frequency weighted sampler per crop."""
    counts   = np.zeros(num_classes, dtype=np.float64)
    for r in records:
        counts[r["label"]] += 1
    counts   = np.where(counts == 0, 1, counts)
    weights  = 1.0 / counts
    sample_w = np.array([weights[r["label"]] for r in records])
    return WeightedRandomSampler(
        weights     = sample_w.tolist(),
        num_samples = len(sample_w),
        replacement = True,
    )


# ── Village-level train/val split for Stage-2 ─────────────────────────────────

def split_records_by_village(
    records:     list,
    val_villages: list = None,
    val_frac:    float = 0.20,
) -> tuple:
    """
    Split crop records into train / val.
    If val_villages provided: all crops from those villages go to val.
    Otherwise: random 80/20 split.
    """
    if val_villages:
        train = [r for r in records if r["village"] not in val_villages]
        val   = [r for r in records if r["village"] in val_villages]
    else:
        np.random.seed(42)
        idx  = np.random.permutation(len(records))
        cut  = int(len(records) * (1 - val_frac))
        train = [records[i] for i in idx[:cut]]
        val   = [records[i] for i in idx[cut:]]
    return train, val