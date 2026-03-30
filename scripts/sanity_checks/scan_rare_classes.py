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
print(f"Scanning all {total} patches...\n")

# Channels of interest
TARGET_CHS = {7: "Utility"}

stats = {
    ch: {
        "patches_with_fg": 0,
        "total_fg_pixels": 0,
        "unique_vals": set(),
        "max_fg_in_patch": 0,
        "example_patches": [],
    }
    for ch in TARGET_CHS
}

errors = 0
for i, fname in enumerate(all_patches):
    if (i+1) % 2000 == 0:
        print(f"  {i+1}/{total}...")

    mask_path = os.path.join(MASK_DIR, fname)
    if not os.path.exists(mask_path):
        errors += 1
        continue

    try:
        with rasterio.open(mask_path) as src:
            data = src.read()  # (9, 512, 512)

        for ch, name in TARGET_CHS.items():
            ch_data = data[ch]
            uniq = np.unique(ch_data)
            stats[ch]["unique_vals"].update(uniq.tolist())
            fg = int((ch_data > 0).sum())
            if fg > 0:
                stats[ch]["patches_with_fg"] += 1
                stats[ch]["total_fg_pixels"] += fg
                if fg > stats[ch]["max_fg_in_patch"]:
                    stats[ch]["max_fg_in_patch"] = fg
                if len(stats[ch]["example_patches"]) < 5:
                    stats[ch]["example_patches"].append((fname, fg, uniq.tolist()))

    except Exception as e:
        errors += 1

print(f"\nDone. Errors: {errors}")
print("=" * 65)

TOTAL_PX = total * 512 * 512

for ch, name in TARGET_CHS.items():
    s = stats[ch]
    n_fg = s["patches_with_fg"]
    total_fg = s["total_fg_pixels"]
    pct_patches = 100.0 * n_fg / total
    pct_area    = 100.0 * total_fg / TOTAL_PX

    print(f"\nChannel {ch} — {name}")
    print(f"  Unique values ever seen : {sorted(s['unique_vals'])}")
    print(f"  Patches with FG         : {n_fg} / {total}  ({pct_patches:.4f}%)")
    print(f"  Total FG pixels         : {total_fg:,}  ({pct_area:.6f}% of all pixels)")
    print(f"  Max FG pixels in 1 patch: {s['max_fg_in_patch']:,}")
    if n_fg > 0:
        print(f"  Avg FG pixels (when present): {total_fg/n_fg:.1f}")
        print(f"  Example patches:")
        for fname, fg, uniq in s["example_patches"]:
            print(f"    {fname}  |  fg_px={fg}  |  vals={uniq}")
    else:
        print(f"  → ZERO foreground pixels across the entire dataset.")

print("\n" + "="*65)
print("VERDICT:")
for ch, name in TARGET_CHS.items():
    s = stats[ch]
    if s["patches_with_fg"] == 0:
        print(f"  {name}: COMPLETELY EMPTY — safe to drop, generate empty shapefile at export.")
    elif s["total_fg_pixels"] < 500:
        print(f"  {name}: EFFECTIVELY EMPTY ({s['total_fg_pixels']} px total) — treat as dropped.")
    else:
        print(f"  {name}: HAS DATA — keep in model.")