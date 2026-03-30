#!/usr/bin/env python3
"""
PHASE 1: Data Validation & Sanity Check
SVAMITVA Multi-Class Semantic Segmentation
"""

import os
import sys
import random
import json
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/Datasets/IIT Tirupathi"
IMG_DIR = os.path.join(BASE_DIR, "preprocessed_dataset", "images")
MASK_DIR = os.path.join(BASE_DIR, "preprocessed_dataset", "masks")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
VIS_DIR = os.path.join(DOCS_DIR, "phase1_visuals")

ACTIVE_CHANNELS = [0, 1, 3, 4, 7, 8]
DROPPED_CHANNELS = [2, 5, 6]

CHANNEL_NAMES = {
    0: "Building",
    1: "Road",
    2: "Railway (dropped)",
    3: "Waterbody Polygon",
    4: "Waterbody Line",
    5: "Waterbody Point (dropped)",
    6: "Utility Point (dropped)",
    7: "Utility Polygon",
    8: "Bridge",
}

EXPECTED_VALUES = {
    0: {0, 1, 2, 3, 4},
    1: {0, 1, 2, 3, 4, 5},
    3: {0, 1, 2, 3, 4, 5, 6},
    4: {0, 1, 2, 3},
    7: {0, 1},
    8: {0, 1},
}

SUBCLASS_NAMES = {
    0: {0: "Background", 1: "RCC", 2: "Tiled", 3: "Tin", 4: "Others"},
    1: {0: "Background", 1: "Type1", 2: "Type2", 3: "Type3", 4: "Type4", 5: "Type5"},
    3: {0: "Background", 1: "Type1", 2: "Type2", 3: "Type3", 4: "Type4", 5: "Type5", 6: "Type6"},
    4: {0: "Background", 1: "Type1", 2: "Type2", 3: "Type3"},
    7: {0: "Background", 1: "Utility"},
    8: {0: "Background", 1: "Bridge"},
}

# Village state mapping (determined from folder structure)
CG_VILLAGES = [
    "MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO",
    "NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO",
    "SAMLUR_450163_SIYANAR_450164_KUTULNAR_450165_BINJAM_450166_JHODIYAWADAM_450167_ORTHO",
]

PB_VILLAGES = [
    "28996_NADALA_ORTHO",
    "37458_fattu_bhila_ortho_3857",
    "37774_bagga ortho_3857",
    "TIMMOWAL_37695_ORI",
]

# Color maps for visualization per channel
VIS_CMAPS = {
    0: {0: (0,0,0,0), 1: (255,0,0,180), 2: (0,255,0,180), 3: (0,0,255,180), 4: (255,255,0,180)},
    1: {0: (0,0,0,0), 1: (255,128,0,180), 2: (128,0,255,180), 3: (0,255,128,180), 4: (255,0,128,180), 5: (128,255,0,180)},
    3: {0: (0,0,0,0), 1: (0,128,255,180), 2: (0,255,255,180), 3: (128,128,255,180), 4: (255,128,128,180), 5: (0,64,128,180), 6: (128,0,64,180)},
    4: {0: (0,0,0,0), 1: (0,0,255,180), 2: (0,128,255,180), 3: (0,255,255,180)},
    7: {0: (0,0,0,0), 1: (255,0,255,180)},
    8: {0: (0,0,0,0), 1: (255,128,64,180)},
}


def parse_village_id(filename):
    """Extract village_id from patch filename using the naming convention:
    {village_id}_{y}_{x}.tif
    We need to strip the last two numeric segments.
    """
    stem = filename.replace(".tif", "")
    parts = stem.rsplit("_", 2)
    if len(parts) >= 3:
        # Check if last two parts are numeric
        try:
            int(parts[-1])
            int(parts[-2])
            return parts[0] if len(parts) == 3 else "_".join(parts[:-2])
        except ValueError:
            pass
    # Fallback: try splitting from right looking for two consecutive numbers
    # Handle complex village names with numbers in them
    tokens = stem.split("_")
    # Walk from the end, find two consecutive numeric tokens
    for i in range(len(tokens) - 1, 0, -1):
        if tokens[i].isdigit() and tokens[i-1].isdigit():
            return "_".join(tokens[:i-1])
    return stem


def get_village_state(village_id):
    if village_id in CG_VILLAGES:
        return "CG"
    elif village_id in PB_VILLAGES:
        return "PB"
    else:
        return "Unknown"


# ============================================================
# STEP 0: PREPROCESSING SCRIPT ANALYSIS
# ============================================================

def step0_analysis():
    print("\n" + "=" * 60)
    print("STEP 0: PREPROCESSING SCRIPT ANALYSIS")
    print("=" * 60)

    findings = {}

    # Patch naming convention
    findings["naming_convention"] = "{village_id}_{y}_{x}.tif"
    print(f"\nPatch naming convention: {findings['naming_convention']}")
    print("  village_id = ortho filename without .tif extension")
    print("  y = row offset (pixels)")
    print("  x = col offset (pixels)")

    # SHP_GROUP bug
    findings["shp_group_bug"] = {
        "severity": "CRITICAL",
        "description": (
            "SHP_GROUP_1 and SHP_GROUP_2 BOTH point to "
            "'Training/PB_Training/shp-file'. SHP_GROUP_2 should "
            "point to 'Training/CG_Training/shp-file'."
        ),
        "affected_villages": [],
        "detail": (
            "The script sorts ortho .tif files alphabetically and uses "
            "idx < 5 to assign SHP_GROUP_1 vs SHP_GROUP_2. Since both "
            "point to PB shapefiles, ALL villages get PB shapefiles "
            "regardless. CG villages received WRONG annotations."
        ),
    }

    # Determine which villages were affected
    # Sorted order of the 7 processed villages:
    all_villages_sorted = sorted([
        "28996_NADALA_ORTHO",
        "37458_fattu_bhila_ortho_3857",
        "37774_bagga ortho_3857",
        "MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO",
        "NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO",
        "SAMLUR_450163_SIYANAR_450164_KUTULNAR_450165_BINJAM_450166_JHODIYAWADAM_450167_ORTHO",
        "TIMMOWAL_37695_ORI",
    ])

    print("\nAlphabetical sort order used by preprocessing script:")
    for i, v in enumerate(all_villages_sorted):
        group = "SHP_GROUP_1 (PB)" if i < 5 else "SHP_GROUP_2 (PB)"
        actual_state = get_village_state(v)
        status = "WRONG" if actual_state == "CG" else "OK (by coincidence)" if i >= 5 else "OK"
        print(f"  idx {i}: {v} [{actual_state}] → {group} → {status}")
        if actual_state == "CG":
            findings["shp_group_bug"]["affected_villages"].append(v)

    print(f"\n*** CRITICAL BUG: {len(findings['shp_group_bug']['affected_villages'])} CG villages "
          f"received PB shapefiles ***")
    print(f"Affected CG villages:")
    for v in findings["shp_group_bug"]["affected_villages"]:
        print(f"  - {v}")

    # Additional note about shapefile naming
    findings["secondary_bug"] = (
        "CG shapefiles use 'Built_Up_Area_type.shp' (with 'e') while "
        "the script loads 'Built_Up_Area_typ.shp' (without 'e'). Even if "
        "SHP_GROUP_2 were corrected to CG path, the Building layer would "
        "fail to load for CG villages."
    )
    print(f"\nSecondary issue: {findings['secondary_bug']}")

    # Missing villages
    findings["missing_from_preprocessed"] = [
        "PINDORI MAYA SINGH-TUGALWAL_28456_ortho (CG_Training)",
        "BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO (PB_Training)",
    ]
    print(f"\nVillages present in Training/ but NOT in preprocessed_dataset/:")
    for v in findings["missing_from_preprocessed"]:
        print(f"  - {v}")

    return findings


# ============================================================
# STEP 1: DATASET INVENTORY
# ============================================================

def step1_inventory():
    print("\n" + "=" * 60)
    print("STEP 1: DATASET INVENTORY")
    print("=" * 60)

    results = {
        "total_patches": 0,
        "per_village": {},
        "mismatches": [],
        "shape_anomalies_img": [],
        "shape_anomalies_mask": [],
        "total_size_gb": 0.0,
    }

    # List all files
    img_files = sorted(os.listdir(IMG_DIR))
    mask_files = sorted(os.listdir(MASK_DIR))
    img_set = set(img_files)
    mask_set = set(mask_files)

    results["total_patches"] = len(img_files)
    print(f"\nTotal image patches: {len(img_files)}")
    print(f"Total mask patches:  {len(mask_files)}")

    # Check mismatches
    img_only = img_set - mask_set
    mask_only = mask_set - img_set
    if img_only:
        results["mismatches"].extend([f"Image without mask: {f}" for f in sorted(img_only)])
    if mask_only:
        results["mismatches"].extend([f"Mask without image: {f}" for f in sorted(mask_only)])
    print(f"Image-mask mismatches: {len(results['mismatches'])}")
    if results["mismatches"]:
        for m in results["mismatches"][:10]:
            print(f"  {m}")

    # Per-village counts
    village_patches = defaultdict(list)
    for f in img_files:
        vid = parse_village_id(f)
        village_patches[vid].append(f)

    print(f"\nVillages found: {len(village_patches)}")
    print(f"\nPer-village patch counts:")
    for vid in sorted(village_patches.keys()):
        count = len(village_patches[vid])
        state = get_village_state(vid)
        results["per_village"][vid] = {"count": count, "state": state}
        print(f"  {vid} [{state}]: {count} patches")

    # Check shapes (sample all for accuracy, use progress bar)
    print(f"\nVerifying image and mask shapes for all {len(img_files)} patches...")
    total_bytes = 0
    common_files = sorted(img_set & mask_set)

    for fname in tqdm(common_files, desc="Shape check"):
        img_path = os.path.join(IMG_DIR, fname)
        mask_path = os.path.join(MASK_DIR, fname)
        try:
            total_bytes += os.path.getsize(img_path) + os.path.getsize(mask_path)

            with rasterio.open(img_path) as src:
                img_shape = (src.count, src.height, src.width)
                if img_shape != (3, 512, 512) and img_shape != (4, 512, 512):
                    results["shape_anomalies_img"].append((fname, img_shape))

            with rasterio.open(mask_path) as src:
                mask_shape = (src.count, src.height, src.width)
                if mask_shape != (9, 512, 512):
                    results["shape_anomalies_mask"].append((fname, mask_shape))
        except Exception as e:
            results["shape_anomalies_img"].append((fname, f"ERROR: {e}"))

    results["total_size_gb"] = total_bytes / (1024 ** 3)
    print(f"\nTotal dataset size: {results['total_size_gb']:.2f} GB")
    print(f"Image shape anomalies: {len(results['shape_anomalies_img'])}")
    if results["shape_anomalies_img"]:
        for f, s in results["shape_anomalies_img"][:10]:
            print(f"  {f}: {s}")
    print(f"Mask shape anomalies: {len(results['shape_anomalies_mask'])}")
    if results["shape_anomalies_mask"]:
        for f, s in results["shape_anomalies_mask"][:10]:
            print(f"  {f}: {s}")

    results["village_patches"] = dict(village_patches)
    return results


# ============================================================
# STEP 2: MASK VALUE VALIDATION
# ============================================================

def step2_mask_validation(village_patches):
    print("\n" + "=" * 60)
    print("STEP 2: MASK VALUE VALIDATION")
    print("=" * 60)

    results = {
        "per_channel_unique_values": {ch: set() for ch in ACTIVE_CHANNELS},
        "unexpected_values": {},
        "dropped_channel_nonzero": {ch: [] for ch in DROPPED_CHANNELS},
    }

    random.seed(42)

    for vid in sorted(village_patches.keys()):
        patches = village_patches[vid]
        sample_size = min(500, len(patches))
        sampled = random.sample(patches, sample_size)
        print(f"\n  Village {vid}: sampling {sample_size}/{len(patches)} patches")

        for fname in tqdm(sampled, desc=f"  {vid[:30]}"):
            mask_path = os.path.join(MASK_DIR, fname)
            try:
                with rasterio.open(mask_path) as src:
                    mask = src.read()  # (9, 512, 512)

                # Active channels
                for ch in ACTIVE_CHANNELS:
                    uniq = set(np.unique(mask[ch]))
                    results["per_channel_unique_values"][ch] |= uniq

                # Dropped channels
                for ch in DROPPED_CHANNELS:
                    if np.any(mask[ch] != 0):
                        results["dropped_channel_nonzero"][ch].append(fname)
            except Exception as e:
                print(f"    ERROR reading {fname}: {e}")

    # Report
    print(f"\n{'='*40}")
    print("MASK VALUE VALIDATION RESULTS")
    print(f"{'='*40}")

    for ch in ACTIVE_CHANNELS:
        observed = results["per_channel_unique_values"][ch]
        expected = EXPECTED_VALUES[ch]
        unexpected = observed - expected
        results["unexpected_values"][ch] = unexpected
        status = "OK" if not unexpected else f"UNEXPECTED: {unexpected}"
        print(f"\n  Channel {ch} ({CHANNEL_NAMES[ch]}):")
        print(f"    Expected:   {sorted(expected)}")
        print(f"    Observed:   {sorted(observed)}")
        print(f"    Status:     {status}")

    for ch in DROPPED_CHANNELS:
        count = len(results["dropped_channel_nonzero"][ch])
        status = "OK (all zeros)" if count == 0 else f"WARNING: {count} patches have non-zero values"
        print(f"\n  Channel {ch} ({CHANNEL_NAMES[ch]}): {status}")

    # Convert sets to lists for JSON serialization
    serializable = {}
    for ch in ACTIVE_CHANNELS:
        serializable[ch] = {
            "observed": sorted(results["per_channel_unique_values"][ch]),
            "unexpected": sorted(results["unexpected_values"][ch]),
        }
    results["serializable"] = serializable

    return results


# ============================================================
# STEP 3: CLASS DISTRIBUTION ANALYSIS
# ============================================================

def step3_class_distribution(village_patches):
    print("\n" + "=" * 60)
    print("STEP 3: CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # Per-channel, per-class pixel counts
    global_counts = {}
    for ch in ACTIVE_CHANNELS:
        max_classes = max(EXPECTED_VALUES[ch]) + 1
        global_counts[ch] = np.zeros(max_classes, dtype=np.int64)

    # Per-village foreground patch counts
    village_fg_patches = {}
    for vid in village_patches:
        village_fg_patches[vid] = {ch: 0 for ch in ACTIVE_CHANNELS}

    all_files = []
    for vid in sorted(village_patches.keys()):
        for f in village_patches[vid]:
            all_files.append((vid, f))

    print(f"\nScanning all {len(all_files)} patches for class distribution...")

    for vid, fname in tqdm(all_files, desc="Class dist"):
        mask_path = os.path.join(MASK_DIR, fname)
        try:
            with rasterio.open(mask_path) as src:
                mask = src.read()  # (9, 512, 512)

            for ch in ACTIVE_CHANNELS:
                ch_data = mask[ch]
                max_val = max(EXPECTED_VALUES[ch])
                vals, counts = np.unique(ch_data, return_counts=True)
                for v, c in zip(vals, counts):
                    if v <= max_val:
                        global_counts[ch][v] += c
                    # Values beyond expected are still counted at their position if within array
                # Check foreground
                if np.any(ch_data > 0):
                    village_fg_patches[vid][ch] += 1
        except Exception as e:
            print(f"  ERROR reading {fname}: {e}")

    # Report
    results = {
        "global_counts": {},
        "channel_stats": {},
        "sparse_channels": [],
        "village_fg_patches": village_fg_patches,
    }

    total_pixels_per_patch = 512 * 512

    print(f"\n{'='*60}")
    print("CLASS DISTRIBUTION RESULTS")
    print(f"{'='*60}")

    for ch in ACTIVE_CHANNELS:
        counts = global_counts[ch]
        total = counts.sum()
        fg_total = counts[1:].sum()
        bg_total = counts[0]
        fg_ratio = (fg_total / total * 100) if total > 0 else 0

        print(f"\nChannel {ch} ({CHANNEL_NAMES[ch]}):")
        print(f"  {'Class':<15} {'Pixels':>15} {'Percentage':>12}")
        print(f"  {'-'*42}")
        class_pcts = {}
        for cls_id in range(len(counts)):
            pct = (counts[cls_id] / total * 100) if total > 0 else 0
            cls_name = SUBCLASS_NAMES.get(ch, {}).get(cls_id, f"Class_{cls_id}")
            print(f"  {cls_name:<15} {counts[cls_id]:>15,} {pct:>11.4f}%")
            class_pcts[cls_id] = pct

        print(f"  Foreground/Total ratio: {fg_ratio:.4f}%")

        results["global_counts"][ch] = {int(k): int(v) for k, v in enumerate(counts)}
        results["channel_stats"][ch] = {
            "fg_ratio_pct": fg_ratio,
            "bg_pixels": int(bg_total),
            "fg_pixels": int(fg_total),
            "total_pixels": int(total),
            "class_pcts": {int(k): float(v) for k, v in class_pcts.items()},
        }

        if fg_ratio < 0.5:
            results["sparse_channels"].append(ch)
            print(f"  *** WARNING: Foreground < 0.5% — needs aggressive loss weighting ***")

    # Per-village foreground patch report
    print(f"\n{'='*60}")
    print("PER-VILLAGE FOREGROUND PATCH COUNTS")
    print(f"{'='*60}")

    header = f"{'Village':<45}"
    for ch in ACTIVE_CHANNELS:
        header += f" Ch{ch:>2}"
    print(header)
    print("-" * len(header))

    for vid in sorted(village_fg_patches.keys()):
        total_patches = len(village_patches[vid])
        row = f"{vid[:44]:<45}"
        for ch in ACTIVE_CHANNELS:
            fg_count = village_fg_patches[vid][ch]
            row += f" {fg_count:>4}"
        row += f"  (of {total_patches})"
        print(row)

    return results


# ============================================================
# STEP 4: CORRUPT / PROBLEMATIC PATCH DETECTION
# ============================================================

def step4_corrupt_detection(village_patches):
    print("\n" + "=" * 60)
    print("STEP 4: CORRUPT / PROBLEMATIC PATCH DETECTION")
    print("=" * 60)

    results = {
        "nan_inf": [],
        "all_black": [],
        "all_white": [],
        "all_mask_zero": [],
    }

    all_files = []
    for vid in sorted(village_patches.keys()):
        for f in village_patches[vid]:
            all_files.append(f)

    print(f"\nScanning all {len(all_files)} patches for corruptions...")

    for fname in tqdm(all_files, desc="Corrupt check"):
        img_path = os.path.join(IMG_DIR, fname)
        mask_path = os.path.join(MASK_DIR, fname)

        try:
            with rasterio.open(img_path) as src:
                img = src.read().astype(np.float32)

            # NaN / Inf
            if np.any(np.isnan(img)) or np.any(np.isinf(img)):
                results["nan_inf"].append(fname)

            # All black
            if np.all(img == 0):
                results["all_black"].append(fname)

            # All white
            if np.all(img == 255):
                results["all_white"].append(fname)

            # All active mask channels zero
            with rasterio.open(mask_path) as src:
                mask = src.read()

            active_sum = sum(np.sum(mask[ch]) for ch in ACTIVE_CHANNELS)
            if active_sum == 0:
                results["all_mask_zero"].append(fname)

        except Exception as e:
            print(f"  ERROR: {fname}: {e}")
            results["nan_inf"].append(f"{fname} (READ ERROR: {e})")

    total_corrupt = (len(results["nan_inf"]) + len(results["all_black"]) +
                     len(results["all_white"]) + len(results["all_mask_zero"]))

    print(f"\nCorrupt/problematic patch summary:")
    print(f"  NaN/Inf images:            {len(results['nan_inf'])}")
    print(f"  All-black images:          {len(results['all_black'])}")
    print(f"  All-white images:          {len(results['all_white'])}")
    print(f"  All-active-mask-zero:      {len(results['all_mask_zero'])}")
    print(f"  Total flagged:             {total_corrupt}")

    if results["nan_inf"]:
        print(f"\n  NaN/Inf patches:")
        for f in results["nan_inf"][:20]:
            print(f"    {f}")

    if results["all_black"]:
        print(f"\n  All-black patches:")
        for f in results["all_black"][:20]:
            print(f"    {f}")

    if results["all_white"]:
        print(f"\n  All-white patches:")
        for f in results["all_white"][:20]:
            print(f"    {f}")

    if results["all_mask_zero"]:
        print(f"\n  All-active-mask-zero patches (first 20):")
        for f in results["all_mask_zero"][:20]:
            print(f"    {f}")

    return results


# ============================================================
# STEP 5: VISUAL SANITY CHECK
# ============================================================

def step5_visual_check(village_patches):
    print("\n" + "=" * 60)
    print("STEP 5: VISUAL SANITY CHECK")
    print("=" * 60)

    os.makedirs(VIS_DIR, exist_ok=True)
    vis_paths = []

    for vid in sorted(village_patches.keys()):
        village_vis_dir = os.path.join(VIS_DIR, vid.replace(" ", "_"))
        os.makedirs(village_vis_dir, exist_ok=True)

        # Find patches with foreground
        fg_patches = []
        patches = village_patches[vid]
        random.seed(42)
        candidates = random.sample(patches, min(50, len(patches)))

        for fname in candidates:
            mask_path = os.path.join(MASK_DIR, fname)
            try:
                with rasterio.open(mask_path) as src:
                    mask = src.read()
                active_sum = sum(np.sum(mask[ch]) for ch in ACTIVE_CHANNELS)
                if active_sum > 0:
                    fg_patches.append(fname)
                if len(fg_patches) >= 3:
                    break
            except Exception:
                continue

        if not fg_patches:
            # fallback: just pick first 3
            fg_patches = patches[:3]

        print(f"\n  Village: {vid} — generating {len(fg_patches)} visualizations")

        for patch_idx, fname in enumerate(fg_patches):
            try:
                img_path = os.path.join(IMG_DIR, fname)
                mask_path = os.path.join(MASK_DIR, fname)

                with rasterio.open(img_path) as src:
                    img = src.read()  # (C, H, W)
                with rasterio.open(mask_path) as src:
                    mask = src.read()  # (9, H, W)

                # Prepare RGB image (handle 3 or 4 bands)
                if img.shape[0] >= 3:
                    rgb = np.stack([img[0], img[1], img[2]], axis=-1)  # (H, W, 3)
                else:
                    rgb = np.stack([img[0]] * 3, axis=-1)

                # Normalize to 0-255 uint8
                if rgb.dtype != np.uint8:
                    if rgb.max() > 0:
                        rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(np.uint8)
                    else:
                        rgb = rgb.astype(np.uint8)

                # Create figure: 1 RGB + 6 active channels
                n_plots = 1 + len(ACTIVE_CHANNELS)
                fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))

                axes[0].imshow(rgb)
                axes[0].set_title("RGB")
                axes[0].axis("off")

                for i, ch in enumerate(ACTIVE_CHANNELS):
                    ax = axes[i + 1]
                    ax.imshow(rgb, alpha=0.6)

                    ch_mask = mask[ch]
                    if np.any(ch_mask > 0):
                        overlay = np.zeros((*ch_mask.shape, 4), dtype=np.uint8)
                        cmap = VIS_CMAPS[ch]
                        for cls_val, color in cmap.items():
                            if cls_val == 0:
                                continue
                            overlay[ch_mask == cls_val] = color
                        ax.imshow(overlay)

                    ax.set_title(f"Ch{ch}: {CHANNEL_NAMES[ch]}")
                    ax.axis("off")

                    # Legend
                    legend_elements = []
                    for cls_val in sorted(cmap.keys()):
                        if cls_val == 0:
                            continue
                        if cls_val in np.unique(ch_mask):
                            c = np.array(cmap[cls_val][:3]) / 255.0
                            label = SUBCLASS_NAMES.get(ch, {}).get(cls_val, f"Class {cls_val}")
                            legend_elements.append(mpatches.Patch(color=c, label=label))
                    if legend_elements:
                        ax.legend(handles=legend_elements, fontsize=6, loc='lower right')

                plt.suptitle(f"{vid}\n{fname}", fontsize=10)
                plt.tight_layout()

                out_path = os.path.join(village_vis_dir, f"patch_{patch_idx}_{fname.replace('.tif', '.png')}")
                fig.savefig(out_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                vis_paths.append(out_path)
                print(f"    Saved: {out_path}")
            except Exception as e:
                print(f"    ERROR visualizing {fname}: {e}")
                traceback.print_exc()

    return vis_paths


# ============================================================
# STEP 6: TRAIN/VAL SPLIT DESIGN
# ============================================================

def step6_split_design(village_patches, village_fg_patches, class_dist_results):
    print("\n" + "=" * 60)
    print("STEP 6: TRAIN/VAL SPLIT DESIGN")
    print("=" * 60)

    results = {
        "all_villages": [],
        "train_split": [],
        "val_split": [],
    }

    print(f"\nAll villages:")
    village_scores = {}
    for vid in sorted(village_patches.keys()):
        state = get_village_state(vid)
        count = len(village_patches[vid])
        results["all_villages"].append({"id": vid, "state": state, "patches": count})
        print(f"  {vid} [{state}]: {count} patches")

        # Score = patch count + number of active channels with foreground
        channels_with_fg = sum(1 for ch in ACTIVE_CHANNELS if village_fg_patches[vid][ch] > 0)
        score = count * 0.5 + channels_with_fg * 100  # weight channel coverage heavily
        village_scores[vid] = {
            "score": score,
            "patches": count,
            "channels_with_fg": channels_with_fg,
            "state": state,
        }

    # Best CG village for val
    cg_villages = {v: s for v, s in village_scores.items() if s["state"] == "CG"}
    pb_villages = {v: s for v, s in village_scores.items() if s["state"] == "PB"}

    best_cg_val = max(cg_villages.keys(), key=lambda v: cg_villages[v]["score"])
    best_pb_val = max(pb_villages.keys(), key=lambda v: pb_villages[v]["score"])

    results["val_split"] = [best_cg_val, best_pb_val]
    results["train_split"] = [v for v in village_patches.keys() if v not in results["val_split"]]

    print(f"\nVillage representativeness scores:")
    for vid, info in sorted(village_scores.items(), key=lambda x: -x[1]["score"]):
        print(f"  {vid[:60]}: score={info['score']:.0f} "
              f"(patches={info['patches']}, fg_channels={info['channels_with_fg']})")

    print(f"\n{'='*40}")
    print("PROPOSED 7/2 TRAIN-VAL SPLIT")
    print(f"{'='*40}")
    print(f"\nVAL split (2 villages):")
    for vid in results["val_split"]:
        info = village_scores[vid]
        print(f"  {vid} [{info['state']}]: {info['patches']} patches, "
              f"{info['channels_with_fg']} active channels with foreground")

    print(f"\nTRAIN split ({len(results['train_split'])} villages):")
    for vid in sorted(results["train_split"]):
        info = village_scores[vid]
        print(f"  {vid} [{info['state']}]: {info['patches']} patches, "
              f"{info['channels_with_fg']} active channels with foreground")

    return results


# ============================================================
# STEP 7: DOCUMENTATION
# ============================================================

def step7_documentation(step0, step1, step2, step3, step4, step5_paths, step6):
    print("\n" + "=" * 60)
    print("STEP 7: GENERATING DOCUMENTATION")
    print("=" * 60)

    os.makedirs(DOCS_DIR, exist_ok=True)
    report_path = os.path.join(DOCS_DIR, "phase1_report.md")

    lines = []
    lines.append("# Phase 1: Data Validation & Sanity Check Report")
    lines.append("")
    lines.append("**Project:** SVAMITVA Multi-Class Semantic Segmentation")
    lines.append("**Dataset:** MoPR Hackathon – Problem Statement 1")
    lines.append("**Backbone:** SegFormer-B2")
    lines.append("")

    # 1. Executive Summary
    lines.append("## 1. Executive Summary")
    lines.append("")
    total_patches = step1["total_patches"]
    n_villages = len(step1["per_village"])
    n_mismatches = len(step1["mismatches"])
    n_shape_anom = len(step1["shape_anomalies_img"]) + len(step1["shape_anomalies_mask"])
    n_corrupt = (len(step4["nan_inf"]) + len(step4["all_black"]) +
                 len(step4["all_white"]) + len(step4["all_mask_zero"]))
    cg_count = sum(1 for v in step1["per_village"].values() if v["state"] == "CG")
    pb_count = sum(1 for v in step1["per_village"].values() if v["state"] == "PB")

    lines.append(f"- **Total patches:** {total_patches:,}")
    lines.append(f"- **Villages processed:** {n_villages} ({cg_count} CG, {pb_count} PB)")
    lines.append(f"- **Image-mask mismatches:** {n_mismatches}")
    lines.append(f"- **Shape anomalies:** {n_shape_anom}")
    lines.append(f"- **Corrupt/problematic patches:** {n_corrupt}")
    lines.append(f"- **Dataset size:** {step1['total_size_gb']:.2f} GB")
    lines.append(f"- **Critical bugs:** 1 (SHP_GROUP duplication)")
    lines.append("")

    # 2. Preprocessing Bug Findings
    lines.append("## 2. Preprocessing Bug Findings (Step 0)")
    lines.append("")
    lines.append("### CRITICAL BUG: SHP_GROUP Duplication")
    lines.append("")
    lines.append(f"**Severity:** {step0['shp_group_bug']['severity']}")
    lines.append("")
    lines.append(f"**Description:** {step0['shp_group_bug']['description']}")
    lines.append("")
    lines.append(f"**Detail:** {step0['shp_group_bug']['detail']}")
    lines.append("")
    lines.append("**Affected CG villages (received PB shapefiles instead of CG):**")
    for v in step0["shp_group_bug"]["affected_villages"]:
        lines.append(f"- `{v}`")
    lines.append("")
    lines.append(f"### Secondary Issue: Shapefile Naming Mismatch")
    lines.append("")
    lines.append(f"{step0['secondary_bug']}")
    lines.append("")
    lines.append("### Missing Villages")
    lines.append("")
    lines.append("Villages present in Training/ but NOT in preprocessed_dataset/:")
    for v in step0["missing_from_preprocessed"]:
        lines.append(f"- {v}")
    lines.append("")
    lines.append("### Patch Naming Convention")
    lines.append("")
    lines.append(f"Format: `{step0['naming_convention']}`")
    lines.append("")

    # 3. Dataset Inventory Table
    lines.append("## 3. Dataset Inventory (Step 1)")
    lines.append("")
    lines.append("| Village ID | State | Patch Count |")
    lines.append("|---|---|---|")
    for vid in sorted(step1["per_village"].keys()):
        info = step1["per_village"][vid]
        lines.append(f"| {vid} | {info['state']} | {info['count']:,} |")
    lines.append(f"| **TOTAL** | | **{total_patches:,}** |")
    lines.append("")
    if step1["mismatches"]:
        lines.append("### Image-Mask Mismatches")
        for m in step1["mismatches"]:
            lines.append(f"- {m}")
        lines.append("")
    if step1["shape_anomalies_img"]:
        lines.append("### Image Shape Anomalies")
        for f, s in step1["shape_anomalies_img"]:
            lines.append(f"- `{f}`: shape={s}")
        lines.append("")
    if step1["shape_anomalies_mask"]:
        lines.append("### Mask Shape Anomalies")
        for f, s in step1["shape_anomalies_mask"]:
            lines.append(f"- `{f}`: shape={s}")
        lines.append("")

    # 4. Mask Value Validation
    lines.append("## 4. Mask Value Validation (Step 2)")
    lines.append("")
    for ch in ACTIVE_CHANNELS:
        info = step2["serializable"][ch]
        lines.append(f"### Channel {ch}: {CHANNEL_NAMES[ch]}")
        lines.append(f"- Expected values: `{sorted(EXPECTED_VALUES[ch])}`")
        lines.append(f"- Observed values: `{info['observed']}`")
        if info["unexpected"]:
            lines.append(f"- **UNEXPECTED values:** `{info['unexpected']}`")
        else:
            lines.append(f"- Status: OK")
        lines.append("")

    lines.append("### Dropped Channels")
    lines.append("")
    for ch in DROPPED_CHANNELS:
        count = len(step2["dropped_channel_nonzero"][ch])
        if count == 0:
            lines.append(f"- Channel {ch} ({CHANNEL_NAMES[ch]}): All zeros ✓")
        else:
            lines.append(f"- Channel {ch} ({CHANNEL_NAMES[ch]}): **{count} patches with non-zero values**")
    lines.append("")

    # 5. Class Distribution
    lines.append("## 5. Class Distribution Analysis (Step 3)")
    lines.append("")
    for ch in ACTIVE_CHANNELS:
        stats = step3["channel_stats"][ch]
        lines.append(f"### Channel {ch}: {CHANNEL_NAMES[ch]}")
        lines.append("")
        lines.append(f"Foreground/Total ratio: **{stats['fg_ratio_pct']:.4f}%**")
        if ch in step3["sparse_channels"]:
            lines.append(f"**⚠ WARNING: Foreground < 0.5% — needs aggressive loss weighting**")
        lines.append("")
        lines.append("| Class | Pixels | Percentage |")
        lines.append("|---|---|---|")
        counts = step3["global_counts"][ch]
        for cls_id, px_count in sorted(counts.items()):
            pct = stats["class_pcts"][cls_id]
            cls_name = SUBCLASS_NAMES.get(ch, {}).get(cls_id, f"Class_{cls_id}")
            lines.append(f"| {cls_name} | {px_count:,} | {pct:.4f}% |")
        lines.append("")

    # Per-village foreground patches table
    lines.append("### Per-Village Foreground Patch Counts")
    lines.append("")
    ch_headers = " | ".join([f"Ch{ch}" for ch in ACTIVE_CHANNELS])
    lines.append(f"| Village | {ch_headers} | Total Patches |")
    lines.append(f"|---|{'---|' * len(ACTIVE_CHANNELS)}---|")
    for vid in sorted(step3["village_fg_patches"].keys()):
        fg = step3["village_fg_patches"][vid]
        total = len(step1["village_patches"][vid])
        ch_vals = " | ".join([str(fg[ch]) for ch in ACTIVE_CHANNELS])
        lines.append(f"| {vid} | {ch_vals} | {total} |")
    lines.append("")

    # 6. Corrupt Patches
    lines.append("## 6. Corrupt / Problematic Patches (Step 4)")
    lines.append("")
    lines.append(f"- NaN/Inf images: {len(step4['nan_inf'])}")
    lines.append(f"- All-black images: {len(step4['all_black'])}")
    lines.append(f"- All-white images: {len(step4['all_white'])}")
    lines.append(f"- All-active-mask-zero: {len(step4['all_mask_zero'])}")
    lines.append("")
    if step4["all_black"]:
        lines.append("### All-Black Patches")
        for f in step4["all_black"]:
            lines.append(f"- `{f}`")
        lines.append("")
    if step4["all_white"]:
        lines.append("### All-White Patches")
        for f in step4["all_white"]:
            lines.append(f"- `{f}`")
        lines.append("")
    if step4["all_mask_zero"]:
        lines.append("### All-Active-Mask-Zero Patches")
        for f in step4["all_mask_zero"][:50]:
            lines.append(f"- `{f}`")
        if len(step4["all_mask_zero"]) > 50:
            lines.append(f"- ... and {len(step4['all_mask_zero']) - 50} more")
        lines.append("")

    # 7. Visual Sanity Check Paths
    lines.append("## 7. Visual Sanity Check Outputs (Step 5)")
    lines.append("")
    if step5_paths:
        for p in step5_paths:
            rel_path = os.path.relpath(p, BASE_DIR)
            lines.append(f"- `{rel_path}`")
    else:
        lines.append("No visualizations generated.")
    lines.append("")

    # 8. Train/Val Split
    lines.append("## 8. Recommended Train/Val Split (Step 6)")
    lines.append("")
    lines.append("### Validation Split (2 villages)")
    lines.append("")
    for vid in step6["val_split"]:
        info = next(v for v in step6["all_villages"] if v["id"] == vid)
        lines.append(f"- **{vid}** [{info['state']}]: {info['patches']:,} patches")
    lines.append("")
    lines.append(f"### Training Split ({len(step6['train_split'])} villages)")
    lines.append("")
    for vid in sorted(step6["train_split"]):
        info = next(v for v in step6["all_villages"] if v["id"] == vid)
        lines.append(f"- **{vid}** [{info['state']}]: {info['patches']:,} patches")
    lines.append("")

    # 9. Open Issues
    lines.append("## 9. Open Issues")
    lines.append("")
    lines.append("1. **SHP_GROUP Bug (Critical):** All 3 CG villages have been annotated "
                 "with PB shapefiles. The preprocessed masks for CG villages are incorrect. "
                 "Decision needed: re-run preprocessing with corrected SHP_GROUP_2 path "
                 "AND fix the Building shapefile naming for CG (`Built_Up_Area_type` vs "
                 "`Built_Up_Area_typ`).")
    lines.append("2. **Missing Villages:** PINDORI MAYA SINGH-TUGALWAL_28456 (CG) and "
                 "BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO "
                 "(PB) were not preprocessed. Decision needed: should they be included?")
    lines.append("3. **ECW File:** An additional CG ortho exists as ECW format "
                 "(KUTRU_451189_AAKLANKA_451163_ORTHO_3857.ecw). Decision needed: "
                 "convert to TIF and include in training?")

    sparse_ch_names = [f"Ch{ch} ({CHANNEL_NAMES[ch]})" for ch in step3["sparse_channels"]]
    if sparse_ch_names:
        lines.append(f"4. **Sparse Channels:** {', '.join(sparse_ch_names)} have foreground "
                     f"< 0.5%. These will need aggressive loss weighting or oversampling strategy.")

    if step4["all_mask_zero"]:
        lines.append(f"5. **All-Active-Mask-Zero Patches:** {len(step4['all_mask_zero'])} patches "
                     f"have zero values in all active channels. These may only have annotations in "
                     f"dropped channels — review whether they should be kept or removed.")

    lines.append("")

    # 10. Decisions Made
    lines.append("## 10. Decisions Made")
    lines.append("")
    lines.append("1. **Validation village selection:** Chose villages with highest combined "
                 "score of patch count and active channel coverage to maximize representativeness.")
    lines.append("2. **Sampling for Step 2:** Used 500 random patches per village (or all if "
                 "fewer than 500) with seed=42 for reproducibility.")
    lines.append("3. **Village ID parsing:** Used the preprocessing script's naming convention "
                 "`{village_id}_{y}_{x}.tif` to extract village IDs by stripping the last two "
                 "underscore-separated numeric segments.")
    lines.append("4. **State assignment:** Villages were assigned CG/PB based on which "
                 "Training subfolder contains their source orthophoto.")
    lines.append("")

    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\nReport saved to: {report_path}")
    return report_path


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1: DATA VALIDATION & SANITY CHECK")
    print("SVAMITVA Multi-Class Semantic Segmentation")
    print("=" * 60)

    # STEP 0
    step0_results = step0_analysis()

    # STEP 1
    step1_results = step1_inventory()
    village_patches = step1_results["village_patches"]

    # STEP 2
    step2_results = step2_mask_validation(village_patches)

    # STEP 3
    step3_results = step3_class_distribution(village_patches)

    # STEP 4
    step4_results = step4_corrupt_detection(village_patches)

    # STEP 5
    step5_paths = step5_visual_check(village_patches)

    # STEP 6
    step6_results = step6_split_design(
        village_patches,
        step3_results["village_fg_patches"],
        step3_results,
    )

    # STEP 7
    report_path = step7_documentation(
        step0_results, step1_results, step2_results,
        step3_results, step4_results, step5_paths,
        step6_results,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================
    total_patches = step1_results["total_patches"]
    n_villages = len(step1_results["per_village"])
    cg_count = sum(1 for v in step1_results["per_village"].values() if v["state"] == "CG")
    pb_count = sum(1 for v in step1_results["per_village"].values() if v["state"] == "PB")
    n_mismatches = len(step1_results["mismatches"])
    n_shape = len(step1_results["shape_anomalies_img"]) + len(step1_results["shape_anomalies_mask"])
    n_corrupt = (len(step4_results["nan_inf"]) + len(step4_results["all_black"]) +
                 len(step4_results["all_white"]) + len(step4_results["all_mask_zero"]))

    train_str = ", ".join(sorted(step6_results["train_split"]))
    val_str = ", ".join(sorted(step6_results["val_split"]))

    print("\n")
    print("PHASE 1 COMPLETE")
    print("================")
    print(f"Total patches:          {total_patches}")
    print(f"Villages found:         {n_villages}")
    print(f"CG villages:            {cg_count}")
    print(f"PB villages:            {pb_count}")
    print(f"Image-mask mismatches:  {n_mismatches}")
    print(f"Shape anomalies:        {n_shape}")
    print(f"Corrupt patches:        {n_corrupt}")
    print(f"Critical bugs found:    1")
    print(f"Proposed train split:   {train_str}")
    print(f"Proposed val split:     {val_str}")
    print(f"Report saved to:        docs/phase1_report.md")
