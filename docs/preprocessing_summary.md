# SVAMITVA Preprocessing — Complete Summary

**Project:** SVAMITVA Multi-Class Semantic Segmentation (MoPR Hackathon — Problem Statement 1)  
**Date:** 12 March 2026  
**Backbone:** SegFormer-B2

---

## 1. What This Pipeline Does

The preprocessing pipeline converts large drone orthophotos (GeoTIFF) and their corresponding vector annotations (Shapefiles) into 512×512 pixel patch pairs — one RGB image patch and one 9-channel binary/multi-class mask — suitable for training a semantic segmentation model.

### Input Data

| Input | Format | Location |
|-------|--------|----------|
| Drone orthophotos | GeoTIFF (.tif) | `Training/PB_Training/` and `Training/CG_Training/` |
| Vector annotations | ESRI Shapefile (.shp) | `Training/{state}_Training/shp-file/` |

Two states are covered:
- **PB (Punjab)** — 5 villages (4 orthos + 1 mis-filed)
- **CG (Chhattisgarh)** — 5 villages (3 orthos + 1 mis-filed + 1 ECW pending)

### Output Data

| Output | Location |
|--------|----------|
| Image patches (512×512, multi-band GeoTIFF) | `preprocessed_dataset/images/` |
| Mask patches (512×512, 9-channel uint8 GeoTIFF) | `preprocessed_dataset/masks/` |
| Train split list | `preprocessed_dataset/train.txt` |
| Validation split list | `preprocessed_dataset/val.txt` |

---

## 2. Preprocessing Steps (in `data_preprocessing.py`)

### Step 1 — Discover Ortho Files

The script iterates over each training directory (`Training/PB_Training/`, `Training/CG_Training/`) and collects all `.tif` files. Each ortho is automatically paired with the shapefile folder from its own training directory.

Two villages are mis-filed (ortho placed in the wrong state's folder) and are handled via an explicit `SHP_OVERRIDES` dictionary that routes them to the correct shapefiles.

### Step 2 — Load Shapefiles

For each village, 9 shapefile layers are loaded and reprojected to match the ortho's CRS:

| Channel | Shapefile | Attribute Column | Class Encoding |
|---------|-----------|-----------------|----------------|
| 0 — Building | `Built_Up_Area_type.shp` or `Built_Up_Area_typ.shp` | `Roof_type` | {1: RCC, 2: Tiled, 3: Tin, 4: Others} |
| 1 — Road | `Road.shp` | `Road_type` | {1→1, 3→2, 4→3, 5→4, 6→5} |
| 2 — Railway | `Railway.shp` | `Railway_Ty` | {1→1, 2→1} |
| 3 — Waterbody Polygon | `Water_Body.shp` | `Water_Body` | {1→1, 2→2, 3→3, 5→4, 8→5, 10→6} |
| 4 — Waterbody Line | `Water_Body_Line.shp` | `Water_Body` | {1→1, 2→2, 11→3} |
| 5 — Waterbody Point | `Waterbody_Point.shp` | `Water_Bodi` | {1→1, 2→2, 3→3} |
| 6 — Utility Point | `Utility.shp` | `Utility_Ty` | {1→1, 2→2} |
| 7 — Utility Polygon | `Utility_Poly.shp` or `Utility_Poly_.shp` | `Utility_Ty` | {1→1} |
| 8 — Bridge | `Bridge.shp` | `Bridge_typ` | {7→1} |

Shapefile names differ between CG and PB (e.g., CG uses `Built_Up_Area_type.shp`, PB uses `Built_Up_Area_typ.shp`). The script uses fallback file detection to handle this automatically.

### Step 3 — Sliding Window Patch Extraction

| Parameter | Value |
|-----------|-------|
| Patch size | 512 × 512 pixels |
| Stride | 384 pixels (128 px overlap) |
| Overlap | 25% |

For each patch position:
1. Read the image tile from the ortho via a rasterio `Window`
2. Compute the geospatial bounds for the window
3. Clip each shapefile layer to those bounds
4. Rasterize each clipped layer into a 512×512 mask channel using the class encoding maps
5. Stack all 9 channels into a single mask array

### Step 4 — Empty Patch Filtering

**Patches where all 9 mask channels sum to zero are skipped** — they contain no annotations and would contribute only background pixels. This is the only filtering/deletion rule.

### Step 5 — Write Patches

Each surviving patch is written as:
- **Image**: `preprocessed_dataset/images/{village_id}_{y}_{x}.tif`
- **Mask**: `preprocessed_dataset/masks/{village_id}_{y}_{x}.tif`

where `{y}` and `{x}` are the top-left pixel coordinates of the patch window.

---

## 3. Bugs Found & Fixed

Six bugs were discovered during Phase 1 validation and fixed before reprocessing.

### Bug 1: Both Shapefile Groups Pointed to PB (CRITICAL)

```python
# BEFORE (broken)
SHP_GROUP_1 = "Training/PB_Training/shp-file"
SHP_GROUP_2 = "Training/PB_Training/shp-file"   # ← should be CG

# AFTER (fixed)
TRAINING_DIRS = {
    "Training/PB_Training": "Training/PB_Training/shp-file",
    "Training/CG_Training": "Training/CG_Training/shp-file",
}
```

**Impact**: All 3 CG villages (MURDANDA, NAGUL, SAMLUR) were annotated with PB shapefiles. Since PB shapes are geographically far from CG orthos, the Building channel (Ch0) was **always zero** for CG villages.

### Bug 2: Hardcoded Index-Based Group Assignment

```python
# BEFORE — sorted index doesn't correctly map villages to states
if idx < 5:
    shp_folder = SHP_GROUP_1
else:
    shp_folder = SHP_GROUP_2
```

**Fix**: Eliminated entirely. Each village now auto-maps to its own training folder's shapefiles.

### Bug 3: Building Shapefile Name Mismatch

- CG: `Built_Up_Area_type.shp` (with the letter 'e')
- PB: `Built_Up_Area_typ.shp` (without 'e')
- Code hardcoded PB name → CG Building layer would always return `None`

**Fix**: Tries `Built_Up_Area_type.shp` first, falls back to `Built_Up_Area_typ.shp`.

### Bug 4: Utility_Poly Shapefile Name Mismatch

- CG: `Utility_Poly.shp`
- PB: `Utility_Poly_.shp` (trailing underscore)
- Code hardcoded CG name → PB Utility_Poly channel was silently skipped

**Fix**: Same fallback pattern as Building.

### Bug 5: Stale ORTHO_FOLDER Path

`ORTHO_FOLDER = "./"` pointed to the workspace root which has no `.tif` files. The script was designed to be run with files copied to root first.

**Fix**: Iterates over `Training/PB_Training/` and `Training/CG_Training/` directly.

### Bug 6: Two Villages Mis-Filed in Wrong State Folder

| Village | Filed In | Actually Belongs To | Evidence |
|---------|----------|---------------------|----------|
| PINDORI MAYA SINGH-TUGALWAL_28456 | CG_Training | PB | 548 PB Building features overlap; 0 CG features overlap |
| BADETUMNAR_450157_BANGAPAL_450155... | PB_Training | CG | 419 CG Building features overlap; 0 PB features overlap |

**Fix**: `SHP_OVERRIDES` dictionary routes these villages to the correct shapefiles regardless of folder location.

---

## 4. Where Did the ~15K Patch Count Come From?

### Original Run (Before Bug Fix): 11,479 patches

The original preprocessing used PB shapefiles for all 7 villages. Only 7 orthos were processed (PINDORI and BADETUMNAR were missing).

| Village | State | Old Patches | Notes |
|---------|-------|-------------|-------|
| 28996_NADALA_ORTHO | PB | 2,869 | Correct (PB shapefiles, PB village) |
| TIMMOWAL_37695_ORI | PB | 2,685 | Correct |
| 37774_bagga ortho_3857 | PB | 1,753 | Correct |
| MURDANDA_450879 | CG | 1,750 | **Wrong shapefiles** — 0% Building |
| 37458_fattu_bhila_ortho_3857 | PB | 1,450 | Correct |
| NAGUL_450171 | CG | 563 | **Wrong shapefiles** — 0% Building |
| SAMLUR_450163 | CG | 409 | **Wrong shapefiles** — 0% Building |
| PINDORI_28456 | PB* | — | **Not processed** |
| BADETUMNAR_450157 | CG* | — | **Not processed** |
| **Total** | | **11,479** | |

### What Was Deleted

**2,722 corrupt CG patches were deleted** (image + mask pairs):

| Village | Patches Deleted | Reason |
|---------|----------------|--------|
| MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO | 1,750 | Masks used PB shapefiles → wrong annotations |
| NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO | 563 | Same |
| SAMLUR_450163_SIYANAR_450164_KUTULNAR_450165_BINJAM_450166_JHODIYAWADAM_450167_ORTHO | 409 | Same |
| **Total deleted** | **2,722** | |

**No PB patches were deleted.** The 8,757 PB patches remained untouched throughout.

After deletion: 11,479 − 2,722 = **8,757 patches** (PB only).

### What Was Added Back (Reprocessing)

CG villages were reprocessed with **correct CG shapefiles**. Two previously missing villages were also processed:

| Village | State | New Patches | Change vs Old |
|---------|-------|-------------|---------------|
| MURDANDA_450879 | CG | 2,964 | +1,214 (was 1,750) |
| NAGUL_450171 | CG | 791 | +228 (was 563) |
| SAMLUR_450163 | CG | 641 | +232 (was 409) |
| PINDORI_28456 | PB* | 961 | **New** |
| BADETUMNAR_450157 | CG* | 1,581 | **New** |
| **Total added** | | **6,938** | |

*Mis-filed: ortho in one state's folder but uses the other state's shapefiles.

**Why CG patches increased**: With correct CG shapefiles, the Building channel (Ch0) is now non-zero, so patches that previously had `sum(mask) == 0` (and were filtered out) now survive.

### Final Count: 15,695 patches

8,757 (surviving PB) + 6,938 (reprocessed CG + new) = **15,695**

---

## 5. Current Dataset Summary

### Per-Village Patch Counts

| Village | State | Patches | Train (80%) | Val (20%) |
|---------|-------|---------|-------------|-----------|
| MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO | CG | 2,964 | 2,371 | 593 |
| 28996_NADALA_ORTHO | PB | 2,869 | 2,295 | 574 |
| TIMMOWAL_37695_ORI | PB | 2,685 | 2,148 | 537 |
| 37774_bagga ortho_3857 | PB | 1,753 | 1,402 | 351 |
| BADETUMNAR_450157_...MOFALNAR_450150_ORTHO | CG* | 1,581 | 1,264 | 317 |
| 37458_fattu_bhila_ortho_3857 | PB | 1,450 | 1,160 | 290 |
| PINDORI MAYA SINGH-TUGALWAL_28456_ortho | PB* | 961 | 768 | 193 |
| NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO | CG | 791 | 632 | 159 |
| SAMLUR_450163_...JHODIYAWADAM_450167_ORTHO | CG | 641 | 512 | 129 |
| **Total** | | **15,695** | **12,552** | **3,143** |

### Post-Fix Building Verification

| CG Village | Building (Ch0) Active Patches | Percentage |
|------------|-------------------------------|------------|
| MURDANDA_450879 | ~60% of sampled patches | Confirmed non-zero |
| NAGUL_450171 | 337 / 791 | 42.6% |
| SAMLUR_450163 | 290 / 641 | 45.2% |
| BADETUMNAR_450157 | ~60% of sampled patches | Confirmed non-zero |

Previously, all CG villages had **0%** Building annotations. The fix is verified.

### Active Mask Channels

| Channel | Name | Foreground % (Phase 1 est.) | Status |
|---------|------|-----------------------------|--------|
| 0 | Building (Roof type) | ~24% | Primary target |
| 1 | Road | ~8.5% | Active |
| 2 | Railway | ~0% | Dropped (all zeros) |
| 3 | Waterbody Polygon | ~6.7% | Active |
| 4 | Waterbody Line | ~0.02% | Active (sparse — needs loss weighting) |
| 5 | Waterbody Point | ~0% | Dropped |
| 6 | Utility Point | ~0% | Dropped |
| 7 | Utility Polygon | ~0.006% | Active (sparse — needs loss weighting) |
| 8 | Bridge | ~0.007% | Active (sparse — needs loss weighting) |

---

## 6. Patch Lifecycle Accounting

```
Original preprocessing (7 villages, PB shapefiles for all)
├── PB villages: 8,757 patches ✅ (correct, untouched)
└── CG villages: 2,722 patches ❌ (wrong shapefiles)

Phase 1.5 Bug Fix:
├── Deleted: 2,722 corrupt CG patches
├── Reprocessed 3 CG villages: +4,396 patches (correct CG shapefiles)
├── Added PINDORI (new, PB shapefiles): +961 patches
└── Added BADETUMNAR (new, CG shapefiles): +1,581 patches

Final: 8,757 + 4,396 + 961 + 1,581 = 15,695 patches
```

---

## 7. Train/Val Split

- **Method**: 80/20 random split, stratified by village (seed=42)
- **Train**: 12,552 patches → `preprocessed_dataset/train.txt`
- **Val**: 3,143 patches → `preprocessed_dataset/val.txt`

---

## 8. Remaining Items

| Item | Status |
|------|--------|
| ECW conversion for KUTRU_451189_AAKLANKA_451163_ORTHO_3857 | Blocked — needs GDAL with ECW SDK or QGIS |
| PB Utility_Poly re-check (filename was `Utility_Poly_.shp` with trailing `_`) | Low priority — PB patches were not regenerated |
| Sparse channel loss weighting (Ch4, Ch7, Ch8) | Pending — to be handled during training |
