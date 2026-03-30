"""
Reconstruct patch mosaic from preprocessed patches and compare
with the original orthophoto for SAMLUR village (smallest, 641 patches).

Uses the pixel offsets (y, x) from patch filenames to place each patch
at its exact position on a canvas matching the original orthophoto region.
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ──────────────────────────────────────────────────
VILLAGE = "SAMLUR_450163_SIYANAR_450164_KUTULNAR_450165_BINJAM_450166_JHODIYAWADAM_450167_ORTHO"
ORTHO_PATH = f"Training/CG_Training/{VILLAGE}.tif"
IMG_DIR = "preprocessed_dataset/images"
MASK_DIR = "preprocessed_dataset/masks"
PATCH_SIZE = 512
DS = 8  # downsample factor for visualization

# ── Collect patch offsets ───────────────────────────────────
patches = sorted([f for f in os.listdir(IMG_DIR) if f.startswith(VILLAGE)])
print(f"Village: {VILLAGE}")
print(f"Patches: {len(patches)}")

offsets = []
for p in patches:
    parts = p.replace(".tif", "").rsplit("_", 2)
    y, x = int(parts[-2]), int(parts[-1])
    offsets.append((y, x, p))

y_min = min(o[0] for o in offsets)
y_max = max(o[0] for o in offsets) + PATCH_SIZE
x_min = min(o[1] for o in offsets)
x_max = max(o[1] for o in offsets) + PATCH_SIZE

h_native = y_max - y_min
w_native = x_max - x_min
h_ds = h_native // DS
w_ds = w_native // DS
ps_ds = PATCH_SIZE // DS  # patch size downsampled

print(f"Coverage region: y=[{y_min}:{y_max}], x=[{x_min}:{x_max}]")
print(f"Native: {w_native} x {h_native} px")
print(f"Downsampled ({DS}x): {w_ds} x {h_ds} px")

# ── Build reconstructed mosaic ──────────────────────────────
# Canvas: RGB + count (for overlap averaging)
mosaic = np.zeros((h_ds, w_ds, 3), dtype=np.float64)
counts = np.zeros((h_ds, w_ds), dtype=np.float64)

# Also build a mask mosaic (Building channel overlay)
mask_mosaic = np.zeros((h_ds, w_ds), dtype=np.float64)

# Coverage map: 1 where a patch exists, 0 where filtered out
coverage = np.zeros((h_ds, w_ds), dtype=np.uint8)

print("\nReading patches...")
for y, x, pname in tqdm(offsets, desc="Patches"):
    # Pixel position on canvas
    cy = (y - y_min) // DS
    cx = (x - x_min) // DS

    # Read image patch (bands 1-3 = RGB), downsample
    with rasterio.open(os.path.join(IMG_DIR, pname)) as src:
        # Read at reduced resolution
        img = src.read(
            [1, 2, 3],
            out_shape=(3, ps_ds, ps_ds),
        ).astype(np.float64)
    img = np.transpose(img, (1, 2, 0))  # HWC

    # Read mask patch (channel 1 = Building)
    with rasterio.open(os.path.join(MASK_DIR, pname)) as src:
        mask_ch0 = src.read(
            1,
            out_shape=(1, ps_ds, ps_ds),
        ).astype(np.float64).squeeze()

    # Place on canvas (handle edge clipping)
    ey = min(cy + ps_ds, h_ds)
    ex = min(cx + ps_ds, w_ds)
    ph = ey - cy
    pw = ex - cx

    mosaic[cy:ey, cx:ex] += img[:ph, :pw]
    counts[cy:ey, cx:ex] += 1.0
    mask_mosaic[cy:ey, cx:ex] += mask_ch0[:ph, :pw]
    coverage[cy:ey, cx:ex] = 1

# Average overlaps
valid = counts > 0
mosaic[valid] /= counts[valid, np.newaxis]
mask_mosaic[valid] /= counts[valid]

# Normalize to 0-255 uint8
mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)

# ── Read original ortho (same region) ───────────────────────
print("\nReading original ortho (matching region)...")
with rasterio.open(ORTHO_PATH) as src:
    window = Window(x_min, y_min, w_native, h_native)
    ortho = src.read(
        [1, 2, 3],
        window=window,
        out_shape=(3, h_ds, w_ds),
    )
ortho = np.transpose(ortho, (1, 2, 0))  # HWC
ortho = np.clip(ortho, 0, 255).astype(np.uint8)

# ── Plot ────────────────────────────────────────────────────
os.makedirs("docs/patch_reconstruction", exist_ok=True)

# Figure 1: Side-by-side comparison (Original vs Reconstructed)
fig, axes = plt.subplots(1, 2, figsize=(24, 10), dpi=120)

axes[0].imshow(ortho)
axes[0].set_title("Original Orthophoto (cropped to patch region)", fontsize=14)
axes[0].axis("off")

axes[1].imshow(mosaic)
axes[1].set_title(f"Reconstructed from {len(patches)} Patches", fontsize=14)
axes[1].axis("off")

fig.suptitle(
    f"SAMLUR Village — Patch Reconstruction vs Original\n"
    f"Patch size: {PATCH_SIZE}px | Stride: 384px | Downsample: {DS}x | "
    f"Coverage: {len(patches)}/3195 grid cells ({100*len(patches)/3195:.1f}%)",
    fontsize=15, fontweight="bold",
)
plt.tight_layout()
plt.savefig("docs/patch_reconstruction/01_sidebyside.png", bbox_inches="tight")
plt.close()
print("Saved: docs/patch_reconstruction/01_sidebyside.png")

# Figure 2: Coverage map — shows which areas have patches vs gaps
fig, ax = plt.subplots(figsize=(14, 8), dpi=120)
ax.imshow(ortho, alpha=0.4)
ax.imshow(
    np.where(coverage[..., None] == 1, mosaic, 0),
    alpha=0.6,
)
# Draw grid lines for patch boundaries
for y, x, _ in offsets:
    cy = (y - y_min) / DS
    cx = (x - x_min) / DS
    rect = plt.Rectangle(
        (cx, cy), ps_ds, ps_ds,
        linewidth=0.3, edgecolor="cyan", facecolor="none",
    )
    ax.add_patch(rect)

ax.set_title(
    f"Patch Coverage Map — {len(patches)} patches shown with cyan borders\n"
    f"Gaps = regions where all mask channels were zero (no annotations → filtered out)",
    fontsize=13,
)
ax.axis("off")
plt.tight_layout()
plt.savefig("docs/patch_reconstruction/02_coverage_map.png", bbox_inches="tight")
plt.close()
print("Saved: docs/patch_reconstruction/02_coverage_map.png")

# Figure 3: Building mask overlay on reconstructed mosaic
fig, axes = plt.subplots(1, 2, figsize=(24, 10), dpi=120)

axes[0].imshow(mosaic)
axes[0].set_title("Reconstructed Mosaic (RGB)", fontsize=14)
axes[0].axis("off")

# Overlay: mosaic with building mask in red
overlay = mosaic.copy().astype(np.float64)
building_mask = mask_mosaic > 0
overlay[building_mask, 0] = np.clip(overlay[building_mask, 0] + 100, 0, 255)
overlay[building_mask, 1] = overlay[building_mask, 1] * 0.5
overlay[building_mask, 2] = overlay[building_mask, 2] * 0.5
overlay = np.clip(overlay, 0, 255).astype(np.uint8)

axes[1].imshow(overlay)
axes[1].set_title("Building Annotations (Ch0) overlaid in red", fontsize=14)
legend_patches = [
    mpatches.Patch(color="red", alpha=0.6, label="Building footprints"),
]
axes[1].legend(handles=legend_patches, loc="lower right", fontsize=12)
axes[1].axis("off")

fig.suptitle("SAMLUR — Building Mask Overlay on Reconstructed Patches", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("docs/patch_reconstruction/03_building_overlay.png", bbox_inches="tight")
plt.close()
print("Saved: docs/patch_reconstruction/03_building_overlay.png")

# Figure 4: Checkerboard blend — alternating tiles from original and reconstruction
fig, ax = plt.subplots(figsize=(14, 8), dpi=120)
checker_size = 64  # pixels in downsampled space
blend = ortho.copy()
for r in range(0, h_ds, checker_size):
    for c in range(0, w_ds, checker_size):
        # Every other square comes from the mosaic
        if ((r // checker_size) + (c // checker_size)) % 2 == 0:
            er = min(r + checker_size, h_ds)
            ec = min(c + checker_size, w_ds)
            blend[r:er, c:ec] = mosaic[r:er, c:ec]

ax.imshow(blend)
ax.set_title(
    "Checkerboard Blend — alternating tiles from Original (lighter) and Reconstructed (patches)\n"
    "Shows spatial alignment of patches vs original orthophoto",
    fontsize=13,
)
ax.axis("off")
plt.tight_layout()
plt.savefig("docs/patch_reconstruction/04_checkerboard.png", bbox_inches="tight")
plt.close()
print("Saved: docs/patch_reconstruction/04_checkerboard.png")

print(f"\nDone! All plots saved to docs/patch_reconstruction/")
