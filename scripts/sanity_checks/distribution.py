import rasterio
import numpy as np
import os
from collections import defaultdict

BASE = "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/Datasets/IIT Tirupathi/preprocessed_dataset"
MASK_DIR = os.path.join(BASE, "masks")
TRAIN_TXT = os.path.join(BASE, "train.txt")
VAL_TXT   = os.path.join(BASE, "val.txt")

with open(TRAIN_TXT) as f:
    train = f.read().splitlines()
with open(VAL_TXT) as f:
    val = f.read().splitlines()

all_patches = train + val
total = len(all_patches)
print(f"Scanning all {total} patches for subclass distribution...\n")

# What we're scanning:
# Ch0 - Building:        codes 1,2,3,4
# Ch1 - Road:            codes 2,3,4,5
# Ch3 - Waterbody poly:  codes 2,4,5
# Ch4 - Waterbody line:  codes 2,3
# Ch7 - Utility poly:    code 1
# Ch8 - Bridge:          code 1

CHANNELS = {
    0: "Building",
    1: "Road",
    3: "Waterbody_poly",
    4: "Waterbody_line",
    7: "Utility_poly",
    8: "Bridge",
}

# Per channel per code: patch count + pixel count
# code_patch_count[ch][code] = number of patches containing that code
# code_pixel_count[ch][code] = total pixels of that code across all patches
code_patch_count  = {ch: defaultdict(int) for ch in CHANNELS}
code_pixel_count  = {ch: defaultdict(int) for ch in CHANNELS}

errors = 0
for i, fname in enumerate(all_patches):
    if (i+1) % 3000 == 0:
        print(f"  {i+1}/{total}...")

    mask_path = os.path.join(MASK_DIR, fname)
    if not os.path.exists(mask_path):
        errors += 1
        continue

    try:
        with rasterio.open(mask_path) as src:
            data = src.read()   # (9, 512, 512)

        for ch in CHANNELS:
            ch_data = data[ch]
            unique_codes = np.unique(ch_data)
            for code in unique_codes:
                if code == 0:
                    continue  # skip background
                px_count = int((ch_data == code).sum())
                if px_count > 0:
                    code_patch_count[ch][code]  += 1
                    code_pixel_count[ch][code]  += px_count

    except Exception as e:
        errors += 1

TOTAL_PX = total * 512 * 512
print(f"\nDone. Errors: {errors}")
print("=" * 75)

for ch, ch_name in CHANNELS.items():
    codes = sorted(code_patch_count[ch].keys())
    if not codes:
        print(f"\n{ch_name} (Ch{ch}): NO FOREGROUND AT ALL")
        continue

    print(f"\n{'='*75}")
    print(f"Ch{ch} — {ch_name}")
    print(f"{'Code':<8} {'Patches':>10} {'% Patches':>11} {'Total Px':>14} {'% All Px':>12} {'Avg Px/Patch':>14}")
    print("-" * 75)

    for code in codes:
        n_patches  = code_patch_count[ch][code]
        n_pixels   = code_pixel_count[ch][code]
        pct_patch  = 100.0 * n_patches / total
        pct_px     = 100.0 * n_pixels  / TOTAL_PX
        avg_px     = n_pixels / n_patches if n_patches > 0 else 0
        print(f"{code:<8} {n_patches:>10,} {pct_patch:>10.3f}% {n_pixels:>14,} {pct_px:>11.5f}% {avg_px:>14.1f}")

    # Summary per channel
    total_fg_patches = len(set().union(*[
        [fname for fname in all_patches
         if code_patch_count[ch][code] > 0]
        for code in codes
    ])) if codes else 0

    print(f"\n  All codes combined:")
    all_px = sum(code_pixel_count[ch][c] for c in codes)
    all_pt = max(code_patch_count[ch].values()) if codes else 0
    print(f"  Total fg pixels : {all_px:,}  ({100.0*all_px/TOTAL_PX:.4f}% of dataset)")

print("\n" + "="*75)
print("STAGE 2 CLASSIFIER FEASIBILITY SUMMARY")
print("="*75)
print(f"{'Channel':<20} {'Codes':<20} {'Min patches':>12} {'Max patches':>12} {'Balanced?':>10}")
print("-"*75)
for ch, ch_name in CHANNELS.items():
    codes = sorted(code_patch_count[ch].keys())
    if not codes:
        print(f"{ch_name:<20} {'none':<20} {'—':>12} {'—':>12} {'SKIP':>10}")
        continue
    counts = [code_patch_count[ch][c] for c in codes]
    mn, mx = min(counts), max(counts)
    ratio = mx / mn if mn > 0 else float('inf')
    balanced = "YES" if ratio < 5 else f"NO ({ratio:.0f}x)"
    print(f"{ch_name:<20} {str(codes):<20} {mn:>12,} {mx:>12,} {balanced:>10}")