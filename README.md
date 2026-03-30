# SVAMITVA — Multi-Class Semantic Segmentation of Drone Orthophotos

> **MoPR Hackathon — Problem Statement 1**
> Pixel-level land-use classification from SVAMITVA drone survey orthophotos using SegFormer-B2.

---

## Overview

This project preprocesses high-resolution drone orthophotos (4 cm/px, EPSG:32643/32644) from the **SVAMITVA** (Survey of Villages Abadi and Mapping with Improvised Technology in Village Areas) program into 512×512 patches paired with 9-channel semantic masks, for training a multi-class segmentation model.

Two states are covered — **Punjab (PB)** and **Chhattisgarh (CG)** — across **9 villages**, producing **15,695 geo-referenced patch pairs**.

<p align="center">
  <img src="docs/readme_assets/nadala_orthophoto.jpg" width="46%" alt="Nadala Orthophoto"/>
  &nbsp;
  <img src="docs/readme_assets/nadala_coverage.jpg" width="46%" alt="Nadala Patch Coverage"/>
</p>
<p align="center"><em>Left: Original orthophoto (Nadala, Punjab). Right: Patch coverage map — cyan borders show extracted patches, dark gaps are annotation-free regions filtered out during preprocessing.</em></p>

---

## Dataset

### Source Data

| Input | Description |
|-------|-------------|
| Drone orthophotos | Multi-band GeoTIFF, 4 cm/px resolution |
| Vector annotations | ESRI Shapefiles (buildings, roads, water bodies, utilities, bridges) |
| States | Punjab (PB) — 5 villages, Chhattisgarh (CG) — 4 villages |

### Preprocessed Output — 15,695 Patches

| Village | State | Patches | Train | Val |
|---------|:-----:|--------:|------:|----:|
| MURDANDA_450879 | CG | 2,964 | 2,371 | 593 |
| 28996_NADALA | PB | 2,869 | 2,295 | 574 |
| TIMMOWAL_37695 | PB | 2,685 | 2,148 | 537 |
| 37774_bagga | PB | 1,753 | 1,402 | 351 |
| BADETUMNAR_450157 | CG | 1,581 | 1,264 | 317 |
| 37458_fattu_bhila | PB | 1,450 | 1,160 | 290 |
| PINDORI_28456 | PB | 961 | 768 | 193 |
| NAGUL_450171 | CG | 791 | 632 | 159 |
| SAMLUR_450163 | CG | 641 | 512 | 129 |
| **Total** | | **15,695** | **12,552** | **3,143** |

### Mask Channels (9 classes)

| Ch | Layer | Attribute | Classes | Foreground % |
|:--:|-------|-----------|:-------:|-------------:|
| 0 | Building | Roof type | RCC, Tiled, Tin, Others | ~24% |
| 1 | Road | Road type | 5 sub-types | ~8.5% |
| 2 | Railway | — | — | ~0% (dropped) |
| 3 | Waterbody (polygon) | Type | 6 sub-types | ~6.7% |
| 4 | Waterbody (line) | Type | 3 sub-types | ~0.02% |
| 5 | Waterbody (point) | — | — | ~0% (dropped) |
| 6 | Utility (point) | — | — | ~0% (dropped) |
| 7 | Utility (polygon) | Type | 1 type | ~0.006% |
| 8 | Bridge | Type | 1 type | ~0.007% |

---

## Patch Extraction Pipeline

```
Orthophoto (160K × 176K px)
    │
    ├── Sliding window: 512×512 px, stride 384 (25% overlap)
    ├── For each window position:
    │     ├── Read image tile from ortho
    │     ├── Clip 9 shapefile layers to tile bounds
    │     ├── Rasterize each layer → 512×512 mask channel
    │     └── Stack into 9-channel mask
    │
    ├── Filter: skip patches where mask is all-zero
    │
    └── Write: {village_id}_{y}_{x}.tif  (image + mask)
```

Each patch retains its full geo-transform (CRS, bounds, resolution) enabling pixel-perfect reconstruction back to the original orthophoto.

<p align="center">
  <img src="docs/readme_assets/sample_patch.png" width="60%" alt="Sample Patch"/>
</p>
<p align="center"><em>Sample 512×512 patch with mask channel overlays (Nadala, Punjab).</em></p>

---

## Patch Reconstruction Verification

To verify spatial correctness, patches are reconstructed back into a mosaic using their geo-coordinates and compared against the original orthophoto.

<p align="center">
  <img src="docs/readme_assets/murdanda_orthophoto.jpg" width="46%" alt="Murdanda Original"/>
  &nbsp;
  <img src="docs/readme_assets/murdanda_reconstructed.jpg" width="46%" alt="Murdanda Reconstructed"/>
</p>
<p align="center"><em>Murdanda (CG) — Original orthophoto vs reconstructed mosaic from 2,964 patches.</em></p>

<p align="center">
  <img src="docs/readme_assets/samlur_orthophoto.jpg" width="30%" alt="Samlur Original"/>
  &nbsp;
  <img src="docs/readme_assets/samlur_reconstructed.jpg" width="30%" alt="Samlur Reconstructed"/>
  &nbsp;
  <img src="docs/readme_assets/samlur_coverage.jpg" width="30%" alt="Samlur Coverage"/>
</p>
<p align="center"><em>Samlur (CG, 641 patches) — Original | Reconstructed | Coverage map. Only 18.7% of the grid has patches; the rest was filtered (no annotations).</em></p>

<p align="center">
  <img src="docs/readme_assets/building_overlay.png" width="60%" alt="Building Overlay"/>
</p>
<p align="center"><em>Building footprint annotations (Channel 0) overlaid in red on reconstructed patches.</em></p>

---

## Repository Structure

```
.
├── README.md
├── .gitignore
│
├── scripts/
│   ├── data_preprocessing.py      # Main preprocessing pipeline
│   ├── phase1_validation.py       # Data validation & sanity checks
│   ├── reprocess_fix.py           # CG village reprocessing (bug fix)
│   ├── reprocess_swapped.py       # Mis-filed village reprocessing
│   ├── reconstruct_patches.py     # Single-village reconstruction demo
│   ├── reconstruct_all.py         # All-village reconstruction
│   └── create_split.py            # Train/val split generator
│
├── docs/
│   ├── readme_assets/             # Optimized images for this README
│   ├── preprocessing_summary.md   # Full preprocessing documentation
│   ├── phase1_report.md           # Phase 1 validation report
│   ├── phase1_5_report.md         # Bug fix & reprocessing report
│   ├── phase1_visuals/            # Per-village sample patch visualizations
│   └── patch_reconstruction/      # SAMLUR reconstruction demo plots
│
├── patch_reconstruction/          # Reconstruction outputs for all 9 villages
│   └── {village}_{state}_{grid}_{patches}/
│       ├── orthophoto_original.png
│       ├── reconstructed.png
│       └── patch_coverage_map.png
│
├── preprocessed_dataset/          # [.gitignore] 41 GB
│   ├── images/                    # 15,695 image patches (512×512 GeoTIFF)
│   ├── masks/                     # 15,695 mask patches (9-ch uint8 GeoTIFF)
│   ├── train.txt                  # 12,552 training patches
│   └── val.txt                    # 3,143 validation patches
│
├── Training/                      # [.gitignore] Raw orthophotos + shapefiles
│   ├── PB_Training/
│   └── CG_Training/
│
└── Testing/                       # [.gitignore] Test orthophotos
```

---

## Bugs Found & Fixed

During Phase 1 validation, **6 critical bugs** were discovered and fixed:

| # | Bug | Impact | Fix |
|:-:|-----|--------|-----|
| 1 | Both shapefile groups pointed to PB | CG villages had 0% Building annotations | Folder-based auto-detection |
| 2 | Hardcoded `idx < 5` group assignment | Wrong state mapping after any file addition | Eliminated — auto-maps from training dir |
| 3 | Building shapefile name mismatch | CG: `Built_Up_Area_type` vs PB: `Built_Up_Area_typ` | Fallback file detection |
| 4 | Utility_Poly name mismatch | CG: `Utility_Poly` vs PB: `Utility_Poly_` | Fallback file detection |
| 5 | Stale `ORTHO_FOLDER` path | No .tif files found in root | Iterates training subdirs |
| 6 | Two villages mis-filed | PINDORI (PB) in CG folder, BADETUMNAR (CG) in PB folder | `SHP_OVERRIDES` dict |

See [docs/preprocessing_summary.md](docs/preprocessing_summary.md) for full details.

---

## Quick Start

### Prerequisites

```bash
pip install rasterio geopandas numpy tqdm matplotlib pillow
```

### Run Preprocessing

```bash
cd scripts/
python data_preprocessing.py
```

### Generate Train/Val Split

```bash
python scripts/create_split.py
```

### Reconstruct Patches for Verification

```bash
python scripts/reconstruct_all.py
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Preprocessing Summary](docs/preprocessing_summary.md) | Complete pipeline documentation, bugs, and patch accounting |
| [Phase 1 Report](docs/phase1_report.md) | Data validation, class distributions, corrupt patch analysis |
| [Phase 1.5 Report](docs/phase1_5_report.md) | Bug fix changelog and reprocessing results |

---

## License

This project uses data from the [SVAMITVA Scheme](https://svamitva.nic.in/), Ministry of Panchayati Raj, Government of India.
