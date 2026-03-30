"""
infer.py  —  Full inference pipeline for SVAMITVA test villages.

Pipeline:
  1. For each test village .tif:
     a. Extract 512×512 patches with stride 384  (skip if already done)
     b. Run Stage-1 SegFormer-B4  → per-patch class prediction maps
     c. Stitch patches back → full-village segmentation map (COG GeoTIFF)
     d. Extract connected components per class
     e. Run Stage-2 classifiers   → assign subtypes to components
     f. Vectorize all predictions → GeoPackage (.gpkg) per village
     g. Also export as ESRI Shapefiles (.shp) per class per village

Output folder structure:
  outputs/
    {village_name}/
      patches/           ← 512×512 image patches (skip if exist)
      pred_patches/      ← per-patch prediction .npy files
      raster/
        {village}_seg_map.tif   ← COG GeoTIFF, full-village class map
        {village}_conf_map.tif  ← COG GeoTIFF, confidence (max softmax prob)
      vectors/
        {village}_features.gpkg ← GeoPackage with all layers
        shapefiles/
          {village}_buildings.shp
          {village}_roads.shp
          {village}_waterbodies.shp
          {village}_waterbody_lines.shp
          {village}_utility.shp
          {village}_bridges.shp
          {village}_railway.shp  ← always empty (confirmed zero instances)

Usage:
  python3 scripts/infer.py --test-dir /path/to/test/tifs
  python3 scripts/infer.py --test-dir /path/to/test/tifs --output-dir /path/to/outputs
  python3 scripts/infer.py --test-dir /path/to/test/tifs --skip-stage2
"""

import os
import sys
import json
import argparse
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.transform import from_bounds, Affine
from rasterio.crs import CRS
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ── allow running from project root ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config        import Config
from model         import SVAMITVASegFormer
from stage2_model  import SubtypeClassifier
from stage2_dataset import CLASSIFIER_CONFIG, BUILDING_NAMES, ROAD_NAMES, WATERBODY_NAMES

# ── Optional geospatial libs (will warn if missing) ──────────────────────────
try:
    import geopandas as gpd
    import shapely.geometry as sgeom
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    print("[WARN] geopandas / shapely not found. Vector export will be skipped.")
    print("       Install: pip install geopandas shapely fiona")

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

PATCH_SIZE = 512
STRIDE     = 384
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]

# Stage-1 class IDs
CLS_BG          = 0
CLS_BUILDING    = 1
CLS_ROAD        = 2
CLS_WATER_POLY  = 3
CLS_WATER_LINE  = 4
CLS_UTILITY     = 5
CLS_BRIDGE      = 6

# ── Subtype label maps (class_idx → human name) ───────────────────────────
BUILDING_SUBTYPES  = {0: "RCC",    1: "Tiled", 2: "Tin",    3: "Others"}
ROAD_SUBTYPES      = {0: "Type-2", 1: "Type-3", 2: "Type-4", 3: "Type-5"}
WATERBODY_SUBTYPES = {0: "Type-4", 1: "Type-1", 2: "Type-2", 3: "Other"}


# ═════════════════════════════════════════════════════════════════════════════
# Sanity check helpers
# ═════════════════════════════════════════════════════════════════════════════

def sanity(condition: bool, msg: str):
    """Raise RuntimeError with clear message if condition is False."""
    if not condition:
        raise RuntimeError(f"\n[SANITY FAIL] {msg}")


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def check_ckpt(path: str, label: str):
    sanity(os.path.exists(path),
           f"{label} checkpoint not found: {path}\n"
           f"  → Make sure training completed and the path in Config is correct.")
    print(f"  [OK] {label}: {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


def check_tif(path: str) -> dict:
    """Open a GeoTIFF and return basic metadata. Raises on failure."""
    try:
        with rasterio.open(path) as src:
            meta = {
                "width":     src.width,
                "height":    src.height,
                "bands":     src.count,
                "crs":       src.crs,
                "transform": src.transform,
                "dtype":     src.dtypes[0],
            }
        return meta
    except Exception as e:
        raise RuntimeError(f"Cannot open GeoTIFF: {path}\n  Error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# Stage 0: Patch extraction
# ═════════════════════════════════════════════════════════════════════════════

def extract_patches(tif_path: str, patch_dir: str, village_name: str) -> list:
    """
    Tile a full-village orthophoto into 512×512 patches with stride 384.
    Skips if patches already exist (checks for a manifest JSON).

    Returns list of patch metadata dicts:
        {fname, row_off, col_off, transform, crs, height, width}
    """
    manifest_path = os.path.join(patch_dir, "_manifest.json")

    # ── Skip if already done ──────────────────────────────────────────────
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"  [SKIP] Patches already extracted: {len(manifest)} patches  "
              f"→ {patch_dir}")
        return manifest

    os.makedirs(patch_dir, exist_ok=True)
    meta = check_tif(tif_path)
    H, W = meta["height"], meta["width"]
    print(f"  Orthophoto: {W}×{H} px  |  {meta['bands']} bands  "
          f"|  CRS: {meta['crs']}")

    sanity(meta["bands"] >= 3,
           f"Expected ≥3 bands, got {meta['bands']} in {tif_path}")

    # Compute grid
    rows = list(range(0, H - PATCH_SIZE + 1, STRIDE))
    cols = list(range(0, W - PATCH_SIZE + 1, STRIDE))
    # Include last row/col if orthophoto doesn't divide evenly
    if (H - PATCH_SIZE) % STRIDE != 0:
        rows.append(H - PATCH_SIZE)
    if (W - PATCH_SIZE) % STRIDE != 0:
        cols.append(W - PATCH_SIZE)
    rows = sorted(set(rows))
    cols = sorted(set(cols))

    print(f"  Grid: {len(rows)} rows × {len(cols)} cols = "
          f"{len(rows)*len(cols):,} patches")

    manifest = []
    with rasterio.open(tif_path) as src:
        t = src.transform
        for row_off in tqdm(rows, desc=f"  Patching {village_name}", ncols=80):
            for col_off in cols:
                # Read patch (first 3 bands → RGB)
                window = rasterio.windows.Window(col_off, row_off,
                                                  PATCH_SIZE, PATCH_SIZE)
                try:
                    data = src.read(
                        indexes=[1, 2, 3],
                        window=window,
                        boundless=True,
                        fill_value=0,
                    )   # (3, 512, 512) uint8
                except Exception as e:
                    print(f"  [WARN] Read error at ({row_off},{col_off}): {e}")
                    continue

                # Patch geo-transform
                patch_transform = Affine(
                    t.a, t.b, t.c + col_off * t.a,
                    t.d, t.e, t.f + row_off * t.e,
                )

                fname = f"{village_name}_{row_off}_{col_off}.tif"
                fpath = os.path.join(patch_dir, fname)

                with rasterio.open(
                    fpath, "w",
                    driver="GTiff",
                    height=PATCH_SIZE, width=PATCH_SIZE,
                    count=3, dtype="uint8",
                    crs=src.crs,
                    transform=patch_transform,
                ) as dst:
                    dst.write(data)

                manifest.append({
                    "fname":     fname,
                    "row_off":   row_off,
                    "col_off":   col_off,
                    "transform": list(patch_transform)[:6],  # 6-tuple for JSON
                    "crs":       str(src.crs),
                })

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  [OK] Extracted {len(manifest):,} patches → {patch_dir}")
    return manifest


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: Segmentation inference
# ═════════════════════════════════════════════════════════════════════════════

def normalise_patch(img_np: np.ndarray) -> torch.Tensor:
    """
    img_np: (3, H, W) uint8  →  (1, 3, H, W) float32 normalised tensor
    """
    x = img_np.astype(np.float32) / 255.0
    for c, (m, s) in enumerate(zip(MEAN, STD)):
        x[c] = (x[c] - m) / s
    return torch.from_numpy(x).unsqueeze(0)  # (1, 3, H, W)


def run_stage1(
    patch_dir:    str,
    pred_dir:     str,
    manifest:     list,
    model:        torch.nn.Module,
    device:       torch.device,
    batch_size:   int = 8,
) -> dict:
    """
    Run Stage-1 segmentation on all patches.
    Saves per-patch predictions as .npy (uint8 class map + float16 softmax probs).
    Skips patches already predicted.

    Returns: pred_manifest {fname: pred_fname}
    """
    os.makedirs(pred_dir, exist_ok=True)
    pred_manifest_path = os.path.join(pred_dir, "_pred_manifest.json")

    if os.path.exists(pred_manifest_path):
        with open(pred_manifest_path) as f:
            pred_manifest = json.load(f)
        already_done = sum(
            os.path.exists(os.path.join(pred_dir, v["pred_cls"]))
            for v in pred_manifest.values()
        )
        if already_done == len(manifest):
            print(f"  [SKIP] Stage-1 predictions already exist: "
                  f"{len(pred_manifest):,} patches")
            return pred_manifest

    model.eval()
    pred_manifest = {}

    # Batch processing
    batch_imgs   = []
    batch_fnames = []

    def flush_batch():
        if not batch_imgs:
            return
        imgs = torch.cat(batch_imgs, dim=0).to(device)  # (B, 3, H, W)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                logits = model(imgs)                    # (B, 7, H, W)
            probs = F.softmax(logits.float(), dim=1)    # (B, 7, H, W) float32
        cls_map = probs.argmax(dim=1).byte().cpu().numpy()    # (B, H, W) uint8
        conf    = probs.max(dim=1).values.half().cpu().numpy()# (B, H, W) float16

        for i, fname in enumerate(batch_fnames):
            pred_cls  = fname.replace(".tif", "_cls.npy")
            pred_conf = fname.replace(".tif", "_conf.npy")
            np.save(os.path.join(pred_dir, pred_cls),  cls_map[i])
            np.save(os.path.join(pred_dir, pred_conf), conf[i])
            pred_manifest[fname] = {
                "pred_cls":  pred_cls,
                "pred_conf": pred_conf,
            }
        batch_imgs.clear()
        batch_fnames.clear()

    for item in tqdm(manifest, desc="  Stage-1 inference", ncols=80):
        fname = item["fname"]
        img_path = os.path.join(patch_dir, fname)

        if not os.path.exists(img_path):
            print(f"  [WARN] Patch missing: {img_path}")
            continue

        with rasterio.open(img_path) as src:
            img = src.read()[:3]  # (3, H, W) uint8

        batch_imgs.append(normalise_patch(img))
        batch_fnames.append(fname)

        if len(batch_imgs) == batch_size:
            flush_batch()

    flush_batch()  # leftover

    with open(pred_manifest_path, "w") as f:
        json.dump(pred_manifest, f, indent=2)

    print(f"  [OK] Stage-1 done: {len(pred_manifest):,} patches predicted")
    sanity(len(pred_manifest) > 0,
           "Stage-1 produced zero predictions. Check patch_dir and model.")
    return pred_manifest


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1.5: Stitch patches → full-village raster
# ═════════════════════════════════════════════════════════════════════════════

def stitch_village(
    manifest:      list,
    pred_manifest: dict,
    pred_dir:      str,
    raster_dir:    str,
    village_name:  str,
    tif_path:      str,
) -> tuple:
    """
    Reconstruct full-village segmentation map and confidence map from patches.
    Uses max-confidence blending in overlapping regions.

    Returns: (seg_map_path, conf_map_path)
    """
    os.makedirs(raster_dir, exist_ok=True)
    seg_path  = os.path.join(raster_dir, f"{village_name}_seg_map.tif")
    conf_path = os.path.join(raster_dir, f"{village_name}_conf_map.tif")

    if os.path.exists(seg_path) and os.path.exists(conf_path):
        print(f"  [SKIP] Raster already stitched → {seg_path}")
        return seg_path, conf_path

    # Get original orthophoto dimensions and geo info
    with rasterio.open(tif_path) as src:
        H, W       = src.height, src.width
        crs        = src.crs
        transform  = src.transform

    # Accumulators: class votes (7 channels) + confidence sum + hit count
    class_votes  = np.zeros((7, H, W), dtype=np.float32)
    conf_acc     = np.zeros((H, W),    dtype=np.float32)
    hit_count    = np.zeros((H, W),    dtype=np.float32)

    n_stitched = 0
    for item in tqdm(manifest, desc="  Stitching", ncols=80):
        fname = item["fname"]
        if fname not in pred_manifest:
            continue

        pm       = pred_manifest[fname]
        cls_path = os.path.join(pred_dir, pm["pred_cls"])
        if not os.path.exists(cls_path):
            continue

        cls_map = np.load(cls_path).astype(np.float32)  # (H, W)
        r0, c0  = item["row_off"], item["col_off"]
        r1 = min(r0 + PATCH_SIZE, H)
        c1 = min(c0 + PATCH_SIZE, W)
        ph = r1 - r0
        pw = c1 - c0

        # One-hot class votes (simple averaging in overlaps)
        for cls_id in range(7):
            class_votes[cls_id, r0:r1, c0:c1] += (cls_map[:ph, :pw] == cls_id)
        hit_count[r0:r1, c0:c1] += 1.0
        n_stitched += 1

    sanity(n_stitched > 0,
           f"No patches were stitched for {village_name}. "
           f"Check pred_manifest and file paths.")

    # Normalise and take argmax
    hit_count = np.maximum(hit_count, 1.0)
    for c in range(7):
        class_votes[c] /= hit_count

    seg_map  = class_votes.argmax(axis=0).astype(np.uint8)   # (H, W)
    conf_map = class_votes.max(axis=0).astype(np.float32)    # (H, W)

    print(f"  Stitched {n_stitched:,} patches → {H}×{W} map")
    for cid, cname in enumerate(["BG","Building","Road","WaterPoly",
                                   "WaterLine","Utility","Bridge"]):
        px = int((seg_map == cid).sum())
        pct = 100.0 * px / (H * W)
        print(f"    {cname:<12}: {px:>10,} px  ({pct:.3f}%)")

    # Write COG GeoTIFF — segmentation map
    _write_cog(seg_map,  seg_path,  crs, transform, "uint8",   nodata=255)
    _write_cog(conf_map, conf_path, crs, transform, "float32", nodata=-1)

    print(f"  [OK] COG seg map   → {seg_path}")
    print(f"  [OK] COG conf map  → {conf_path}")
    return seg_path, conf_path


def _write_cog(
    array:     np.ndarray,
    out_path:  str,
    crs:       CRS,
    transform: Affine,
    dtype:     str,
    nodata     = None,
):
    """Write a 2D numpy array as a Cloud Optimized GeoTIFF."""
    tmp_path = out_path + ".tmp.tif"
    if array.ndim == 2:
        array = array[np.newaxis, ...]  # (1, H, W)

    with rasterio.open(
        tmp_path, "w",
        driver    = "GTiff",
        height    = array.shape[1],
        width     = array.shape[2],
        count     = array.shape[0],
        dtype     = dtype,
        crs       = crs,
        transform = transform,
        nodata    = nodata,
        compress  = "LZW",
        tiled     = True,
        blockxsize= 512,
        blockysize= 512,
    ) as dst:
        dst.write(array)

    # Convert to COG using GDAL translate
    import subprocess
    result = subprocess.run(
        ["gdal_translate", "-of", "COG",
         "-co", "COMPRESS=LZW",
         "-co", "BLOCKSIZE=512",
         tmp_path, out_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # gdal_translate may not be available — just rename
        os.rename(tmp_path, out_path)
        print(f"  [WARN] gdal_translate not available; saving as regular GeoTIFF")
    else:
        os.remove(tmp_path)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2: Component-level subtype classification
# ═════════════════════════════════════════════════════════════════════════════

def classify_components(
    seg_map_path:      str,
    tif_path:          str,
    classifiers:       dict,
    device:            torch.device,
    crop_size:         int = 128,
    min_pixels:        int = 50,
) -> dict:
    """
    For each Stage-2 class (building, road, waterbody):
      - Find connected components in the stitched seg map
      - Crop the corresponding RGB region from the original ortho
      - Run the classifier → assign subtype label

    Returns:
        results[class_name] = list of {
            bbox: (row0,col0,row1,col1),
            subtype_idx: int,
            subtype_name: str,
            confidence: float,
            centroid: (row, col),
            pixel_count: int,
        }
    """
    with rasterio.open(seg_map_path) as src:
        seg_map = src.read(1)   # (H, W) uint8

    H, W = seg_map.shape

    # We'll read the RGB ortho lazily per crop
    ortho_src = rasterio.open(tif_path)

    results = {}

    STAGE2_CLASSES = {
        "building":  (CLS_BUILDING,   10,  classifiers.get("building")),
        "road":      (CLS_ROAD,        30,  classifiers.get("road")),
        "waterbody": (CLS_WATER_POLY,  20,  classifiers.get("waterbody")),
    }

    for cls_name, (cls_id, min_px, clf) in STAGE2_CLASSES.items():
        if clf is None:
            print(f"  [WARN] No classifier loaded for {cls_name}, skipping subtypes")
            results[cls_name] = []
            continue

        binary = ((seg_map == cls_id).astype(np.uint8) * 255)
        n_comp, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        components = []
        clf.eval()

        crop_batch  = []
        comp_metas  = []

        for comp_id in range(1, n_comp):
            area = stats[comp_id, cv2.CC_STAT_AREA]
            if area < min_px:
                continue

            x0 = stats[comp_id, cv2.CC_STAT_LEFT]
            y0 = stats[comp_id, cv2.CC_STAT_TOP]
            bw = stats[comp_id, cv2.CC_STAT_WIDTH]
            bh = stats[comp_id, cv2.CC_STAT_HEIGHT]
            cx, cy = int(centroids[comp_id][0]), int(centroids[comp_id][1])

            # Bounding box with 10 % padding
            pad = max(4, int(0.10 * max(bw, bh)))
            r0 = max(0, y0 - pad);  r1 = min(H, y0 + bh + pad)
            c0 = max(0, x0 - pad);  c1 = min(W, x0 + bw + pad)

            # Read RGB crop from original ortho
            try:
                window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
                crop   = ortho_src.read(indexes=[1, 2, 3], window=window,
                                        boundless=True, fill_value=0)
                crop   = crop.transpose(1, 2, 0)   # (h, w, 3)
                crop   = cv2.resize(crop, (crop_size, crop_size),
                                    interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                print(f"  [WARN] Crop read error (comp {comp_id}): {e}")
                continue

            # Normalise
            x = crop.astype(np.float32) / 255.0
            for ch, (m, s) in enumerate(zip(MEAN, STD)):
                x[:, :, ch] = (x[:, :, ch] - m) / s
            tensor = torch.from_numpy(x.transpose(2, 0, 1)).float()  # (3, H, W)

            crop_batch.append(tensor)
            comp_metas.append({
                "bbox":        (r0, c0, r1, c1),
                "centroid":    (cy, cx),
                "pixel_count": int(area),
            })

            # Flush batch every 32 crops
            if len(crop_batch) == 32:
                _classify_batch(crop_batch, comp_metas, components, clf,
                                device, cls_name)
                crop_batch.clear()
                comp_metas.clear()

        # Flush remaining
        if crop_batch:
            _classify_batch(crop_batch, comp_metas, components, clf,
                            device, cls_name)

        results[cls_name] = components
        print(f"  {cls_name}: {len(components):,} components classified")

    ortho_src.close()
    return results


def _classify_batch(crop_batch, comp_metas, out_list, clf, device, cls_name):
    batch = torch.stack(crop_batch).to(device)
    with torch.no_grad():
        logits = clf(batch)              # (B, num_classes)
        probs  = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1).cpu().numpy()
    confs = probs.max(dim=1).values.cpu().numpy()

    NAME_MAP = {
        "building":  BUILDING_SUBTYPES,
        "road":      ROAD_SUBTYPES,
        "waterbody": WATERBODY_SUBTYPES,
    }

    for i, meta in enumerate(comp_metas):
        out_list.append({
            **meta,
            "subtype_idx":  int(preds[i]),
            "subtype_name": NAME_MAP[cls_name].get(int(preds[i]), "Unknown"),
            "confidence":   float(confs[i]),
        })


# ═════════════════════════════════════════════════════════════════════════════
# Vector export: GeoPackage + Shapefiles
# ═════════════════════════════════════════════════════════════════════════════

def export_vectors(
    seg_map_path:  str,
    tif_path:      str,
    stage2_results: dict,
    vector_dir:    str,
    village_name:  str,
):
    """
    Vectorise the segmentation map and export as:
      - GeoPackage (.gpkg) with one layer per feature class
      - ESRI Shapefiles (.shp) per class (same as training format)
    """
    if not HAS_GEO:
        print("  [SKIP] geopandas not available, skipping vector export")
        return

    os.makedirs(vector_dir, exist_ok=True)
    shp_dir = os.path.join(vector_dir, "shapefiles")
    os.makedirs(shp_dir, exist_ok=True)

    gpkg_path = os.path.join(vector_dir, f"{village_name}_features.gpkg")

    with rasterio.open(seg_map_path) as src:
        seg_map   = src.read(1)
        crs       = src.crs
        transform = src.transform

    # ── Helper: rasterio polygonize ──────────────────────────────────────
    from rasterio.features import shapes as rio_shapes

    def polygonize_class(binary_mask: np.ndarray):
        """Return list of (geometry, value) for foreground pixels."""
        geoms = []
        for geom_dict, val in rio_shapes(
            binary_mask.astype(np.uint8),
            mask=binary_mask.astype(np.uint8),
            transform=transform,
        ):
            if val == 1:
                geoms.append(sgeom.shape(geom_dict))
        return geoms

    all_layers = {}

    # ── Buildings ────────────────────────────────────────────────────────
    print("  Vectorising: buildings …")
    bld_mask = (seg_map == CLS_BUILDING).astype(np.uint8)
    bld_comps = stage2_results.get("building", [])
    bld_geoms, bld_attrs = _build_polygon_layer(
        bld_mask, bld_comps, transform,
        attr_col="Roof_type", subtype_map=BUILDING_SUBTYPES,
        # Numeric code: RCC=1, Tiled=2, Tin=3, Others=4 (original training codes)
        subtype_code_map={0: 1, 1: 2, 2: 3, 3: 4},
    )
    all_layers["buildings"] = _make_gdf(bld_geoms, bld_attrs, crs)

    # ── Roads ────────────────────────────────────────────────────────────
    print("  Vectorising: roads …")
    road_mask  = (seg_map == CLS_ROAD).astype(np.uint8)
    road_comps = stage2_results.get("road", [])
    road_geoms, road_attrs = _build_polygon_layer(
        road_mask, road_comps, transform,
        attr_col="Road_type", subtype_map=ROAD_SUBTYPES,
        subtype_code_map={0: 2, 1: 3, 2: 4, 3: 5},
    )
    all_layers["roads"] = _make_gdf(road_geoms, road_attrs, crs)

    # ── Waterbody polygons ────────────────────────────────────────────────
    print("  Vectorising: waterbodies …")
    wp_mask  = (seg_map == CLS_WATER_POLY).astype(np.uint8)
    wp_comps = stage2_results.get("waterbody", [])
    wp_geoms, wp_attrs = _build_polygon_layer(
        wp_mask, wp_comps, transform,
        attr_col="Water_Body", subtype_map=WATERBODY_SUBTYPES,
        subtype_code_map={0: 4, 1: 1, 2: 2, 3: 3},
    )
    all_layers["waterbodies"] = _make_gdf(wp_geoms, wp_attrs, crs)

    # ── Waterbody lines (binary, no subtype) ──────────────────────────────
    print("  Vectorising: waterbody lines …")
    wl_mask  = (seg_map == CLS_WATER_LINE).astype(np.uint8)
    wl_geoms = polygonize_class(wl_mask)
    wl_attrs = [{"Water_Body": 0, "type_name": "Line"} for _ in wl_geoms]
    all_layers["waterbody_lines"] = _make_gdf(wl_geoms, wl_attrs, crs)

    # ── Utility polygons (binary, no subtype) ─────────────────────────────
    print("  Vectorising: utility …")
    ut_mask  = (seg_map == CLS_UTILITY).astype(np.uint8)
    ut_geoms = polygonize_class(ut_mask)
    # Centroids as points (DT / OHT / well locations)
    ut_centroids = [g.centroid for g in ut_geoms]
    ut_attrs = [{"Utility_Ty": 1, "type_name": "Utility"} for _ in ut_centroids]
    all_layers["utility"] = _make_gdf(ut_centroids, ut_attrs, crs,
                                       geom_type="Point")

    # ── Bridges ──────────────────────────────────────────────────────────
    print("  Vectorising: bridges …")
    br_mask  = (seg_map == CLS_BRIDGE).astype(np.uint8)
    br_geoms = polygonize_class(br_mask)
    br_attrs = [{"Bridge_typ": 1, "type_name": "Bridge"} for _ in br_geoms]
    all_layers["bridges"] = _make_gdf(br_geoms, br_attrs, crs)

    # ── Railway (always empty) ────────────────────────────────────────────
    all_layers["railway"] = gpd.GeoDataFrame(
        {"Railway_Ty": [], "type_name": [], "geometry": []},
        crs=crs,
    )

    # ── Write GeoPackage (all layers in one file) ─────────────────────────
    print(f"  Writing GeoPackage → {gpkg_path}")
    for layer_name, gdf in all_layers.items():
        if gdf is not None and len(gdf) >= 0:
            try:
                gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG")
            except Exception as e:
                print(f"  [WARN] GPKG write failed for {layer_name}: {e}")

    # ── Write Shapefiles (same structure as training data) ────────────────
    SHP_CONFIG = {
        "buildings":      ("Built_Up_Area_typ.shp",   "Roof_type"),
        "roads":          ("Road.shp",                "Road_type"),
        "waterbodies":    ("Water_Body.shp",          "Water_Body"),
        "waterbody_lines":("Water_Body_Line.shp",     "Water_Body"),
        "utility":        ("Utility_Poly.shp",        "Utility_Ty"),
        "bridges":        ("Bridge.shp",              "Bridge_typ"),
        "railway":        ("Railway.shp",             "Railway_Ty"),
    }
    for layer_name, (shp_fname, _) in SHP_CONFIG.items():
        gdf = all_layers.get(layer_name)
        shp_path = os.path.join(shp_dir, shp_fname)
        try:
            if gdf is not None:
                gdf.to_file(shp_path, driver="ESRI Shapefile")
            print(f"  [OK] SHP: {shp_fname}  ({len(gdf) if gdf is not None else 0} features)")
        except Exception as e:
            print(f"  [WARN] SHP write failed for {shp_fname}: {e}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n  Vector export summary for {village_name}:")
    for layer_name, gdf in all_layers.items():
        n = len(gdf) if gdf is not None else 0
        print(f"    {layer_name:<20}: {n:>6} features")


def _build_polygon_layer(
    binary_mask:      np.ndarray,
    stage2_comps:     list,
    transform:        Affine,
    attr_col:         str,
    subtype_map:      dict,
    subtype_code_map: dict,
):
    """
    Polygonize a binary mask and assign subtype attributes using
    Stage-2 classification results (spatial lookup by centroid).

    Returns: (geoms list, attrs list)
    """
    from rasterio.features import shapes as rio_shapes

    geoms = []
    attrs = []

    # Build a spatial index: pixel centroid → subtype
    comp_lookup = {}   # (cy, cx) → subtype_idx
    for comp in stage2_comps:
        cy, cx = comp["centroid"]
        comp_lookup[(cy, cx)] = comp["subtype_idx"]

    for geom_dict, val in rio_shapes(
        binary_mask,
        mask=binary_mask,
        transform=transform,
    ):
        if val != 1:
            continue
        poly = sgeom.shape(geom_dict)
        geoms.append(poly)

        # Default subtype = 0 (no match)
        subtype_idx = 0

        # Find closest Stage-2 component centroid to this polygon's centroid
        if comp_lookup and stage2_comps:
            pc = poly.centroid
            # Transform geo centroid back to pixel coords
            px_c = int((pc.x - transform.c) / transform.a)
            py_c = int((pc.y - transform.f) / transform.e)

            best_dist = float("inf")
            for comp in stage2_comps:
                cy, cx = comp["centroid"]
                dist = (cy - py_c) ** 2 + (cx - px_c) ** 2
                if dist < best_dist:
                    best_dist = dist
                    subtype_idx = comp["subtype_idx"]

        code      = subtype_code_map.get(subtype_idx, 0)
        type_name = subtype_map.get(subtype_idx, "Unknown")
        attrs.append({attr_col: code, "type_name": type_name})

    return geoms, attrs


def _make_gdf(geoms, attrs, crs, geom_type="Polygon"):
    """Create a GeoDataFrame from lists of geometries and attribute dicts."""
    if not geoms:
        return gpd.GeoDataFrame({"geometry": []}, crs=crs)
    gdf = gpd.GeoDataFrame(attrs, geometry=geoms, crs=crs)
    return gdf


# ═════════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════════

def load_stage1(cfg: Config, device: torch.device) -> torch.nn.Module:
    ckpt_path = os.path.join(cfg.checkpoint_dir, "segformer_b4_main_best.pth")
    check_ckpt(ckpt_path, "Stage-1 (main)")

    model = SVAMITVASegFormer(
        num_classes=cfg.num_classes,
        pretrained=cfg.model_name,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    ep   = ckpt.get("epoch", "?")
    miou = ckpt.get("best_miou", "?")
    print(f"  [OK] Stage-1 loaded  |  epoch {ep}  |  best mIoU(fg) = {miou:.4f}"
          if isinstance(miou, float) else
          f"  [OK] Stage-1 loaded  |  epoch {ep}")
    return model


def load_stage2(cfg: Config, device: torch.device, skip: bool = False) -> dict:
    if skip:
        print("  [SKIP] Stage-2 classifiers not loaded (--skip-stage2)")
        return {}

    classifiers = {}
    cls_config = {
        "building":  ("classifier_building.pth",  4),
        "road":      ("classifier_road.pth",       4),
        "waterbody": ("classifier_waterbody.pth",  4),
    }
    for cls_name, (fname, n_cls) in cls_config.items():
        path = os.path.join(cfg.checkpoint_dir, fname)
        if not os.path.exists(path):
            print(f"  [WARN] {cls_name} classifier not found: {path}")
            continue
        clf = SubtypeClassifier(cls_name=cls_name, num_classes=n_cls).to(device)
        ckpt = torch.load(path, map_location=device,weights_only=False)
        clf.load_state_dict(ckpt["model_state"], strict=True)
        clf.eval()
        print(f"  [OK] {cls_name} classifier  |  "
              f"best acc = {ckpt.get('best_acc', '?'):.4f}"
              if isinstance(ckpt.get("best_acc"), float)
              else f"  [OK] {cls_name} classifier loaded")
        classifiers[cls_name] = clf

    return classifiers


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="SVAMITVA Test Inference Pipeline")
    p.add_argument("--test-dir",    required=True,
                   help="Folder containing test village .tif files")
    p.add_argument("--output-dir",  default="outputs",
                   help="Root output directory (default: outputs/)")
    p.add_argument("--batch-size",  type=int, default=8,
                   help="Stage-1 inference batch size (default: 8)")
    p.add_argument("--skip-stage2", action="store_true",
                   help="Skip Stage-2 classifiers (binary segmentation only)")
    p.add_argument("--village",     default=None,
                   help="Process only this village (by filename stem)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()

    # ── Discover test TIFs ────────────────────────────────────────────────
    section("0. SETUP & DISCOVERY")
    test_dir = args.test_dir
    sanity(os.path.isdir(test_dir),
           f"--test-dir does not exist: {test_dir}")

    tif_files = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
    ])
    sanity(len(tif_files) > 0,
           f"No .tif files found in {test_dir}")

    if args.village:
        tif_files = [f for f in tif_files if args.village in f]
        sanity(len(tif_files) > 0,
               f"No file matching --village '{args.village}' in {test_dir}")

    print(f"  Test dir   : {test_dir}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Villages   : {len(tif_files)}")
    for f in tif_files:
        meta = check_tif(os.path.join(test_dir, f))
        print(f"    {f:<60} {meta['width']}×{meta['height']}  "
              f"{meta['bands']}ch  {meta['crs']}")

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory//1024**3} GB)")

    # ── Load models ───────────────────────────────────────────────────────
    section("1. LOADING MODELS")
    stage1_model = load_stage1(cfg, device)
    classifiers  = load_stage2(cfg, device, skip=args.skip_stage2)

    # ── Per-village pipeline ──────────────────────────────────────────────
    success_log = []
    failure_log = []

    for tif_fname in tif_files:
        tif_path     = os.path.join(test_dir, tif_fname)
        village_name = Path(tif_fname).stem
        village_out  = os.path.join(args.output_dir, village_name)

        patch_dir    = os.path.join(village_out, "patches")
        pred_dir     = os.path.join(village_out, "pred_patches")
        raster_dir   = os.path.join(village_out, "raster")
        vector_dir   = os.path.join(village_out, "vectors")

        try:
            # ── Step 1: Patch extraction ──────────────────────────────────
            section(f"VILLAGE: {village_name}  |  Step 1/4: Patching")
            manifest = extract_patches(tif_path, patch_dir, village_name)
            sanity(len(manifest) > 0,
                   f"Patch extraction produced 0 patches for {village_name}")

            # ── Step 2: Stage-1 inference ─────────────────────────────────
            section(f"VILLAGE: {village_name}  |  Step 2/4: Stage-1 Inference")
            pred_manifest = run_stage1(
                patch_dir, pred_dir, manifest,
                stage1_model, device, batch_size=args.batch_size,
            )
            sanity(len(pred_manifest) > 0,
                   "Stage-1 inference produced 0 predictions.")

            # ── Step 3: Stitch ────────────────────────────────────────────
            section(f"VILLAGE: {village_name}  |  Step 3/4: Stitching")
            seg_path, conf_path = stitch_village(
                manifest, pred_manifest, pred_dir,
                raster_dir, village_name, tif_path,
            )
            sanity(os.path.exists(seg_path),
                   f"Stitched seg map not found: {seg_path}")

            # ── Step 4: Stage-2 + Vector export ──────────────────────────
            section(f"VILLAGE: {village_name}  |  Step 4/4: Stage-2 + Vectorise")
            if not args.skip_stage2 and classifiers:
                stage2_results = classify_components(
                    seg_path, tif_path, classifiers, device
                )
            else:
                # Binary only — empty stage2 results
                stage2_results = {
                    "building":  [],
                    "road":      [],
                    "waterbody": [],
                }

            export_vectors(
                seg_path, tif_path, stage2_results,
                vector_dir, village_name,
            )

            success_log.append(village_name)
            print(f"\n  ✓ {village_name} COMPLETE")
            print(f"    Raster  → {raster_dir}/")
            print(f"    Vectors → {vector_dir}/")

        except Exception as e:
            print(f"\n  ✗ {village_name} FAILED")
            print(f"    Error: {e}")
            traceback.print_exc()
            failure_log.append((village_name, str(e)))

    # ── Final summary ─────────────────────────────────────────────────────
    section("PIPELINE COMPLETE")
    print(f"  Success : {len(success_log)}/{len(tif_files)} villages")
    for v in success_log:
        print(f"    ✓  {v}")
    if failure_log:
        print(f"\n  Failed  : {len(failure_log)} villages")
        for v, err in failure_log:
            print(f"    ✗  {v}")
            print(f"       {err[:120]}")

    # Write summary JSON
    summary_path = os.path.join(args.output_dir, "inference_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "success":  success_log,
            "failed":   [{"village": v, "error": e} for v, e in failure_log],
            "total":    len(tif_files),
        }, f, indent=2)
    print(f"\n  Summary → {summary_path}")


if __name__ == "__main__":
    main()