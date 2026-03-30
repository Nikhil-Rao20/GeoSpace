"""
Reconstruct patch mosaics for ALL villages and save clean images
(no titles, no axes, no text) for report use.

Output structure:
  patch_reconstruction/
    {village_name}_{state}_{total_grid_cells}_{patch_count}/
      orthophoto_original.png
      reconstructed.png
      patch_coverage_map.png
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── All villages: (village_id, ortho_path, state) ───────────
VILLAGES = [
    ("MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO",
     "Training/CG_Training/MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO.tif", "CG"),
    ("NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO",
     "Training/CG_Training/NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO.tif", "CG"),
    ("SAMLUR_450163_SIYANAR_450164_KUTULNAR_450165_BINJAM_450166_JHODIYAWADAM_450167_ORTHO",
     "Training/CG_Training/SAMLUR_450163_SIYANAR_450164_KUTULNAR_450165_BINJAM_450166_JHODIYAWADAM_450167_ORTHO.tif", "CG"),
    ("PINDORI MAYA SINGH-TUGALWAL_28456_ortho",
     "Training/CG_Training/PINDORI MAYA SINGH-TUGALWAL_28456_ortho.tif", "PB"),
    ("BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO",
     "Training/PB_Training/BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO.tif", "CG"),
    ("28996_NADALA_ORTHO",
     "Training/PB_Training/28996_NADALA_ORTHO.tif", "PB"),
    ("37458_fattu_bhila_ortho_3857",
     "Training/PB_Training/37458_fattu_bhila_ortho_3857.tif", "PB"),
    ("37774_bagga ortho_3857",
     "Training/PB_Training/37774_bagga ortho_3857.tif", "PB"),
    ("TIMMOWAL_37695_ORI",
     "Training/PB_Training/TIMMOWAL_37695_ORI.tif", "PB"),
]

IMG_DIR = "preprocessed_dataset/images"
PATCH_SIZE = 512
STRIDE = 384
OUT_ROOT = "patch_reconstruction"


def process_village(village_id, ortho_path, state):
    # Collect patches
    patches = sorted([f for f in os.listdir(IMG_DIR) if f.startswith(village_id + "_")])
    # Filter: ensure the part after village_id matches _Y_X.tif pattern
    valid = []
    for p in patches:
        suffix = p[len(village_id):]  # e.g. "_12345_6789.tif"
        parts = suffix.replace(".tif", "").lstrip("_").split("_")
        if len(parts) >= 2:
            try:
                int(parts[-1])
                int(parts[-2])
                valid.append(p)
            except ValueError:
                pass
    patches = valid

    if not patches:
        print(f"  SKIP: no patches found")
        return

    # Extract y, x offsets
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

    # Total possible grid cells
    y_steps = len(range(y_min, y_max - PATCH_SIZE + 1, STRIDE))
    x_steps = len(range(x_min, x_max - PATCH_SIZE + 1, STRIDE))
    total_grid = y_steps * x_steps

    patch_count = len(patches)

    # Adaptive downsample: keep the longer dimension under ~4000px
    max_dim = max(h_native, w_native)
    ds = max(1, max_dim // 4000)

    h_ds = h_native // ds
    w_ds = w_native // ds
    ps_ds = PATCH_SIZE // ds

    # Output folder
    safe_name = village_id.replace(" ", "_")
    folder_name = f"{safe_name}_{state}_{total_grid}_{patch_count}"
    out_dir = os.path.join(OUT_ROOT, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"  Patches: {patch_count}, Grid: {total_grid}, DS: {ds}x, Canvas: {w_ds}x{h_ds}")

    # ── Build mosaic ────────────────────────────────────────
    mosaic = np.zeros((h_ds, w_ds, 3), dtype=np.float64)
    counts = np.zeros((h_ds, w_ds), dtype=np.float64)
    coverage = np.zeros((h_ds, w_ds), dtype=np.uint8)

    for y, x, pname in tqdm(offsets, desc="  Patches", leave=False):
        cy = (y - y_min) // ds
        cx = (x - x_min) // ds

        with rasterio.open(os.path.join(IMG_DIR, pname)) as src:
            img = src.read([1, 2, 3], out_shape=(3, ps_ds, ps_ds)).astype(np.float64)
        img = np.transpose(img, (1, 2, 0))

        ey = min(cy + ps_ds, h_ds)
        ex = min(cx + ps_ds, w_ds)
        ph = ey - cy
        pw = ex - cx

        mosaic[cy:ey, cx:ex] += img[:ph, :pw]
        counts[cy:ey, cx:ex] += 1.0
        coverage[cy:ey, cx:ex] = 1

    valid = counts > 0
    mosaic[valid] /= counts[valid, np.newaxis]
    mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)

    # ── Read original ortho ─────────────────────────────────
    with rasterio.open(ortho_path) as src:
        window = Window(x_min, y_min, w_native, h_native)
        ortho = src.read([1, 2, 3], window=window, out_shape=(3, h_ds, w_ds))
    ortho = np.clip(np.transpose(ortho, (1, 2, 0)), 0, 255).astype(np.uint8)

    # ── Save: orthophoto_original.png ───────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(w_ds / 100, h_ds / 100), dpi=150)
    ax.imshow(ortho)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.savefig(
        os.path.join(out_dir, "orthophoto_original.png"),
        bbox_inches="tight", pad_inches=0, dpi=150,
    )
    plt.close(fig)

    # ── Save: reconstructed.png ─────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(w_ds / 100, h_ds / 100), dpi=150)
    ax.imshow(mosaic)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.savefig(
        os.path.join(out_dir, "reconstructed.png"),
        bbox_inches="tight", pad_inches=0, dpi=150,
    )
    plt.close(fig)

    # ── Save: patch_coverage_map.png ────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(w_ds / 100, h_ds / 100), dpi=150)
    ax.imshow(ortho, alpha=0.4)
    ax.imshow(
        np.where(coverage[..., None] == 1, mosaic, 0),
        alpha=0.6,
    )
    for y, x, _ in offsets:
        cy = (y - y_min) / ds
        cx = (x - x_min) / ds
        rect = plt.Rectangle(
            (cx, cy), ps_ds, ps_ds,
            linewidth=0.3, edgecolor="cyan", facecolor="none",
        )
        ax.add_patch(rect)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.savefig(
        os.path.join(out_dir, "patch_coverage_map.png"),
        bbox_inches="tight", pad_inches=0, dpi=150,
    )
    plt.close(fig)

    print(f"  Saved to {out_dir}/")


# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    for i, (vid, ortho, state) in enumerate(VILLAGES, 1):
        print(f"\n[{i}/{len(VILLAGES)}] {vid} ({state})")
        process_village(vid, ortho, state)
    print("\nAll villages processed.")
