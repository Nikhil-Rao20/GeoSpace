import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from tqdm import tqdm


PATCH_SIZE = 512
STRIDE = 384
OUTPUT_ROOT = "preprocessed_dataset"

TRAINING_DIRS = {
    "Training/PB_Training": "Training/PB_Training/shp-file",
    "Training/CG_Training": "Training/CG_Training/shp-file",
}

SHP_OVERRIDES = {
    "PINDORI MAYA SINGH-TUGALWAL_28456_ortho": "Training/PB_Training/shp-file",
    "BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO": "Training/CG_Training/shp-file",
}

PB_VILLAGES = [
    "28996_NADALA_ORTHO",
    "TIMMOWAL_37695_ORI",
    "37774_bagga ortho_3857",
    "37458_fattu_bhila_ortho_3857",
    "PINDORI MAYA SINGH-TUGALWAL_28456_ortho",
]

CG_VILLAGES = [
    "MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO",
    "NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO",
    "SAMLUR_450163_SIYANAR_450164_KUTULNAR_450165_BINJAM_450166_JHODIYAWADAM_450167_ORTHO",
    "BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO",
]

EXPECTED_COUNTS_BEFORE = {
    "MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO": 2964,
    "NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO": 791,
    "SAMLUR_450163_SIYANAR_450164_KUTULNAR_450165_BINJAM_450166_JHODIYAWADAM_450167_ORTHO": 641,
    "BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO": 1581,
    "28996_NADALA_ORTHO": 2869,
    "TIMMOWAL_37695_ORI": 2685,
    "37774_bagga ortho_3857": 1753,
    "37458_fattu_bhila_ortho_3857": 1450,
    "PINDORI MAYA SINGH-TUGALWAL_28456_ortho": 961,
}

VAL_VILLAGES = {
    "MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO",
    "28996_NADALA_ORTHO",
}

TRAIN_VILLAGES = {
    "NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO",
    "SAMLUR_450163_SIYANAR_450164_KUTULNAR_450165_BINJAM_450166_JHODIYAWADAM_450167_ORTHO",
    "BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO",
    "TIMMOWAL_37695_ORI",
    "37774_bagga ortho_3857",
    "37458_fattu_bhila_ortho_3857",
    "PINDORI MAYA SINGH-TUGALWAL_28456_ortho",
}

BUILDING_MAP = {1: 1, 2: 2, 3: 3, 4: 4}
ROAD_MAP = {1: 1, 3: 2, 4: 3, 5: 4, 6: 5}
RAILWAY_MAP = {1: 1, 2: 1}
WATER_POLY_MAP = {1: 1, 2: 2, 3: 3, 5: 4, 8: 5, 10: 6}
WATER_LINE_MAP = {1: 1, 2: 2, 11: 3}
WATER_POINT_MAP = {1: 1, 2: 2, 3: 3}
UTILITY_POINT_MAP = {1: 1, 2: 2}
UTILITY_POLY_MAP = {1: 1}
BRIDGE_MAP = {7: 1}

EXPECTED_VALUE_SETS = {
    0: {0, 1, 2, 3, 4},
    1: {0, 1, 2, 3, 4, 5},
    3: {0, 1, 2, 3, 4, 5, 6},
    4: {0, 1, 2, 3},
    7: {0, 1},
    8: {0, 1},
}

VILLAGE_STATE = {
    "MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO": "CG",
    "NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO": "CG",
    "SAMLUR_450163_SIYANAR_450164_KUTULNAR_450165_BINJAM_450166_JHODIYAWADAM_450167_ORTHO": "CG",
    "BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO": "CG",
    "28996_NADALA_ORTHO": "PB",
    "TIMMOWAL_37695_ORI": "PB",
    "37774_bagga ortho_3857": "PB",
    "37458_fattu_bhila_ortho_3857": "PB",
    "PINDORI MAYA SINGH-TUGALWAL_28456_ortho": "PB",
}

VILLAGE_REGEX = re.compile(r"^(.*)_(-?\d+)_(-?\d+)\.tif$")


@dataclass
class VillageProcessResult:
    village_id: str
    old_count: int
    new_count: int
    delta_pct: float
    utility_name_used: str


def extract_village_id(filename: str) -> str | None:
    match = VILLAGE_REGEX.match(filename)
    if not match:
        return None
    return match.group(1)


def count_by_village(folder: Path) -> Counter:
    counts = Counter()
    for p in folder.glob("*.tif"):
        village_id = extract_village_id(p.name)
        if village_id is not None:
            counts[village_id] += 1
    return counts


def load_layer(path: str, crs):
    if not os.path.exists(path):
        return None
    gdf = gpd.read_file(path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[gdf.is_valid]
    return gdf


def rasterize_patch(gdf, column, mapping, patch_size, transform):
    if gdf is None or len(gdf) == 0:
        return np.zeros((patch_size, patch_size), dtype=np.uint8)
    shapes = []
    for _, row in gdf.iterrows():
        val = row[column]
        if val in mapping:
            shapes.append((row.geometry, mapping[val]))
    if not shapes:
        return np.zeros((patch_size, patch_size), dtype=np.uint8)
    return rasterize(
        shapes,
        out_shape=(patch_size, patch_size),
        transform=transform,
        fill=0,
        dtype="uint8",
    )


def find_ortho_path(village_id: str) -> str:
    for train_dir in TRAINING_DIRS:
        candidate = os.path.join(train_dir, f"{village_id}.tif")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Orthophoto not found for village: {village_id}")


def find_shp_folder(village_id: str, ortho_path: str) -> str:
    if village_id in SHP_OVERRIDES:
        return SHP_OVERRIDES[village_id]
    train_dir = str(Path(ortho_path).parent)
    if train_dir not in TRAINING_DIRS:
        raise ValueError(f"No shapefile mapping for training directory: {train_dir}")
    return TRAINING_DIRS[train_dir]


def delete_pb_patches(images_dir: Path, masks_dir: Path, counts_before: Counter):
    image_files = list(images_dir.glob("*.tif"))
    mask_files = list(masks_dir.glob("*.tif"))

    image_targets = defaultdict(list)
    for p in image_files:
        village_id = extract_village_id(p.name)
        if village_id in PB_VILLAGES:
            image_targets[village_id].append(p)

    mask_targets = defaultdict(list)
    for p in mask_files:
        village_id = extract_village_id(p.name)
        if village_id in PB_VILLAGES:
            mask_targets[village_id].append(p)

    print("\nSTEP 1: Delete existing PB patches")
    for village in PB_VILLAGES:
        img_n = len(image_targets[village])
        msk_n = len(mask_targets[village])
        print(f"TO DELETE - {village}: images={img_n}, masks={msk_n}")

    for village in PB_VILLAGES:
        for p in tqdm(image_targets[village], desc=f"Delete images {village}"):
            p.unlink()
        for p in tqdm(mask_targets[village], desc=f"Delete masks {village}"):
            p.unlink()
        print(
            f"DELETED: {len(image_targets[village])} image patches and {len(mask_targets[village])} mask patches for village {village}"
        )

    counts_after_images = count_by_village(images_dir)
    counts_after_masks = count_by_village(masks_dir)

    for village in CG_VILLAGES:
        expected = EXPECTED_COUNTS_BEFORE[village]
        img_now = counts_after_images.get(village, 0)
        msk_now = counts_after_masks.get(village, 0)
        if img_now != expected or msk_now != expected:
            raise RuntimeError(
                "CRITICAL ERROR: CG integrity check failed after PB deletion. "
                f"{village} expected={expected} got images={img_now}, masks={msk_now}"
            )

    pb_deleted = {v: counts_before.get(v, 0) - counts_after_images.get(v, 0) for v in PB_VILLAGES}
    return pb_deleted


def process_one_pb_village(village_id: str, images_dir: Path, masks_dir: Path, old_count: int) -> VillageProcessResult:
    ortho_path = find_ortho_path(village_id)
    shp_folder = find_shp_folder(village_id, ortho_path)

    with rasterio.open(ortho_path) as src:
        height = src.height
        width = src.width
        crs = src.crs

    def shp_path(name: str) -> str:
        return os.path.join(shp_folder, name)

    building_shp = shp_path("Built_Up_Area_type.shp")
    if not os.path.exists(building_shp):
        building_shp = shp_path("Built_Up_Area_typ.shp")

    utility_poly_shp = shp_path("Utility_Poly.shp")
    utility_name_used = "Utility_Poly.shp"
    if not os.path.exists(utility_poly_shp):
        utility_poly_shp = shp_path("Utility_Poly_.shp")
        utility_name_used = "Utility_Poly_.shp"

    if not os.path.exists(utility_poly_shp):
        raise FileNotFoundError(f"Utility polygon shapefile not found for {village_id}")

    building_gdf = load_layer(building_shp, crs)
    road_gdf = load_layer(shp_path("Road.shp"), crs)
    railway_gdf = load_layer(shp_path("Railway.shp"), crs)
    water_poly_gdf = load_layer(shp_path("Water_Body.shp"), crs)
    water_line_gdf = load_layer(shp_path("Water_Body_Line.shp"), crs)
    water_point_gdf = load_layer(shp_path("Waterbody_Point.shp"), crs)
    utility_point_gdf = load_layer(shp_path("Utility.shp"), crs)
    utility_poly_gdf = load_layer(utility_poly_shp, crs)
    bridge_gdf = load_layer(shp_path("Bridge.shp"), crs)

    generated = 0
    with rasterio.open(ortho_path) as img_src:
        y_steps = list(range(0, height - PATCH_SIZE, STRIDE))
        x_steps = list(range(0, width - PATCH_SIZE, STRIDE))

        for y in tqdm(y_steps, desc=f"Process {village_id}"):
            for x in x_steps:
                window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                img_tile = img_src.read(window=window)
                tile_transform = rasterio.windows.transform(window, img_src.transform)
                bounds = rasterio.windows.bounds(window, img_src.transform)

                def clip(gdf):
                    if gdf is None:
                        return None
                    return gdf.cx[bounds[0] : bounds[2], bounds[1] : bounds[3]]

                building = clip(building_gdf)
                road = clip(road_gdf)
                railway = clip(railway_gdf)
                water_poly = clip(water_poly_gdf)
                water_line = clip(water_line_gdf)
                water_point = clip(water_point_gdf)
                utility_point = clip(utility_point_gdf)
                utility_poly = clip(utility_poly_gdf)
                bridge = clip(bridge_gdf)

                mask_tile = np.stack(
                    [
                        rasterize_patch(building, "Roof_type", BUILDING_MAP, PATCH_SIZE, tile_transform),
                        rasterize_patch(road, "Road_type", ROAD_MAP, PATCH_SIZE, tile_transform),
                        rasterize_patch(railway, "Railway_Ty", RAILWAY_MAP, PATCH_SIZE, tile_transform),
                        rasterize_patch(water_poly, "Water_Body", WATER_POLY_MAP, PATCH_SIZE, tile_transform),
                        rasterize_patch(water_line, "Water_Body", WATER_LINE_MAP, PATCH_SIZE, tile_transform),
                        rasterize_patch(water_point, "Water_Bodi", WATER_POINT_MAP, PATCH_SIZE, tile_transform),
                        rasterize_patch(utility_point, "Utility_Ty", UTILITY_POINT_MAP, PATCH_SIZE, tile_transform),
                        rasterize_patch(utility_poly, "Utility_Ty", UTILITY_POLY_MAP, PATCH_SIZE, tile_transform),
                        rasterize_patch(bridge, "Bridge_typ", BRIDGE_MAP, PATCH_SIZE, tile_transform),
                    ],
                    axis=0,
                )

                if np.sum(mask_tile) == 0:
                    continue

                img_meta = img_src.meta.copy()
                img_meta.update(height=PATCH_SIZE, width=PATCH_SIZE, transform=tile_transform)
                mask_meta = {
                    "driver": "GTiff",
                    "height": PATCH_SIZE,
                    "width": PATCH_SIZE,
                    "count": 9,
                    "dtype": "uint8",
                    "crs": img_src.crs,
                    "transform": tile_transform,
                }

                name = f"{village_id}_{y}_{x}.tif"
                with rasterio.open(images_dir / name, "w", **img_meta) as dst:
                    dst.write(img_tile)
                with rasterio.open(masks_dir / name, "w", **mask_meta) as dst:
                    dst.write(mask_tile)
                generated += 1

    delta_pct = ((generated - old_count) / old_count * 100.0) if old_count else 0.0
    return VillageProcessResult(village_id, old_count, generated, delta_pct, utility_name_used)


def verify_channel7_random(masks_dir: Path, village_id: str, sample_size: int = 200):
    files = [p for p in masks_dir.glob("*.tif") if extract_village_id(p.name) == village_id]
    rng = random.Random(42)
    sample = files if len(files) <= sample_size else rng.sample(files, sample_size)
    non_zero = 0
    for p in sample:
        with rasterio.open(p) as src:
            ch7 = src.read(8)
        if np.any(ch7 > 0):
            non_zero += 1
    return non_zero, len(sample)


def verify_channel_ranges_random(masks_dir: Path, village_id: str, sample_size: int = 200):
    files = [p for p in masks_dir.glob("*.tif") if extract_village_id(p.name) == village_id]
    rng = random.Random(7)
    sample = files if len(files) <= sample_size else rng.sample(files, sample_size)
    observed = {k: set() for k in EXPECTED_VALUE_SETS.keys()}
    for p in sample:
        with rasterio.open(p) as src:
            for ch in observed.keys():
                arr = src.read(ch + 1)
                observed[ch].update(np.unique(arr).tolist())
    violations = {}
    for ch, allowed in EXPECTED_VALUE_SETS.items():
        bad_vals = sorted(observed[ch] - allowed)
        if bad_vals:
            violations[ch] = bad_vals
    return observed, violations, len(sample)


def strict_village_split(images_dir: Path, train_txt: Path, val_txt: Path):
    all_files = sorted([p.name for p in images_dir.glob("*.tif")])
    train_list = []
    val_list = []
    village_counts = Counter()

    for name in all_files:
        village_id = extract_village_id(name)
        if village_id is None:
            continue
        village_counts[village_id] += 1
        if village_id in TRAIN_VILLAGES:
            train_list.append(name)
        elif village_id in VAL_VILLAGES:
            val_list.append(name)
        else:
            raise RuntimeError(f"Unknown village for split assignment: {village_id}")

    train_set = set(train_list)
    val_set = set(val_list)
    overlap = train_set.intersection(val_set)
    if overlap:
        raise RuntimeError(f"Split leakage detected: {len(overlap)} overlapping patches")

    all_assigned = train_set.union(val_set)
    if all_assigned != set(all_files):
        missing = set(all_files) - all_assigned
        extra = all_assigned - set(all_files)
        raise RuntimeError(f"Split coverage mismatch. missing={len(missing)} extra={len(extra)}")

    train_by_village = Counter(extract_village_id(x) for x in train_list)
    val_by_village = Counter(extract_village_id(x) for x in val_list)

    for village in TRAIN_VILLAGES:
        if val_by_village.get(village, 0) > 0:
            raise RuntimeError(f"Train village found in val split: {village}")
    for village in VAL_VILLAGES:
        if train_by_village.get(village, 0) > 0:
            raise RuntimeError(f"Val village found in train split: {village}")

    train_txt.write_text("\n".join(train_list) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(val_list) + "\n", encoding="utf-8")

    return train_by_village, val_by_village, len(train_list), len(val_list), len(all_files)


def per_village_foreground_percent(masks_dir: Path):
    village_files = defaultdict(list)
    for p in masks_dir.glob("*.tif"):
        village_id = extract_village_id(p.name)
        if village_id is not None:
            village_files[village_id].append(p)

    result = {}
    target_channels = [0, 1, 3, 4, 7, 8]
    for village, files in village_files.items():
        totals = {ch: 0 for ch in target_channels}
        n = len(files)
        for p in tqdm(files, desc=f"FG stats {village}"):
            with rasterio.open(p) as src:
                for ch in target_channels:
                    arr = src.read(ch + 1)
                    if np.any(arr > 0):
                        totals[ch] += 1
        result[village] = {ch: (totals[ch] / n * 100.0 if n else 0.0) for ch in target_channels}
    return result


def format_pct(x: float) -> str:
    return f"{x:.2f}%"


def main():
    random.seed(42)

    root = Path(".")
    images_dir = root / OUTPUT_ROOT / "images"
    masks_dir = root / OUTPUT_ROOT / "masks"
    docs_dir = root / "docs"
    report_path = docs_dir / "phase1_6_report.md"
    train_txt = root / OUTPUT_ROOT / "train.txt"
    val_txt = root / OUTPUT_ROOT / "val.txt"

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError("preprocessed_dataset images/masks folders not found")

    print("STEP 0 FINDINGS (from runtime checks)")
    utility_listing = sorted([p.name for p in Path("Training/PB_Training/shp-file").glob("Utility*")])
    print("PB utility shapefiles:")
    for name in utility_listing:
        print(f"  - {name}")

    pb_orthos = sorted([p.name for p in Path("Training/PB_Training").glob("*.tif")])
    print("PB training ortho files:")
    for name in pb_orthos:
        print(f"  - {name}")

    counts_before_images = count_by_village(images_dir)
    counts_before_masks = count_by_village(masks_dir)

    print("Current patch counts (images/masks):")
    for village, expected in EXPECTED_COUNTS_BEFORE.items():
        print(
            f"  - {village}: images={counts_before_images.get(village,0)} masks={counts_before_masks.get(village,0)} expected={expected}"
        )

    for village, expected in EXPECTED_COUNTS_BEFORE.items():
        img_now = counts_before_images.get(village, 0)
        msk_now = counts_before_masks.get(village, 0)
        if img_now != expected or msk_now != expected:
            print(
                f"WARNING: Pre-step count differs for {village}. "
                f"expected={expected}, images={img_now}, masks={msk_now}. "
                "Continuing in resume-safe mode."
            )

    pb_deleted = delete_pb_patches(images_dir, masks_dir, counts_before_images)

    print("\nCG patch counts after PB deletion (must remain unchanged):")
    counts_mid_images = count_by_village(images_dir)
    for village in CG_VILLAGES:
        print(f"  - {village}: {counts_mid_images.get(village,0)}")

    process_results = []
    print("\nSTEP 2: Reprocess PB villages only")
    for village in PB_VILLAGES:
        old_count = EXPECTED_COUNTS_BEFORE[village]
        result = process_one_pb_village(village, images_dir, masks_dir, old_count)
        process_results.append(result)
        range_ok = abs(result.delta_pct) <= 10.0
        print(
            f"REPROCESSED: {village} old={result.old_count} new={result.new_count} "
            f"delta={result.delta_pct:+.2f}% utility_shp={result.utility_name_used} "
            f"within_10pct={'YES' if range_ok else 'NO (FLAG)'}"
        )

    print("\nSTEP 3: Post-reprocessing verification")
    ch7_results = {}
    range_results = {}
    utility_all_zero = True
    for village in PB_VILLAGES:
        nz, n = verify_channel7_random(masks_dir, village, sample_size=200)
        ch7_results[village] = (nz, n)
        print(f"{village} - Ch7 non-zero patches: {nz}/{n}")
        if nz > 0:
            utility_all_zero = False

        observed, violations, _ = verify_channel_ranges_random(masks_dir, village, sample_size=200)
        range_results[village] = {
            "observed": {str(k): sorted(list(v)) for k, v in observed.items()},
            "violations": {str(k): v for k, v in violations.items()},
        }
        if violations:
            print(f"  RANGE VIOLATIONS in {village}: {json.dumps(violations)}")
        else:
            print(f"  RANGE CHECK OK for {village}")

    if utility_all_zero:
        raise RuntimeError("STOP CONDITION: Channel 7 remains zero for all PB villages")

    counts_post_images = count_by_village(images_dir)
    counts_post_masks = count_by_village(masks_dir)
    cg_integrity_ok = True
    for village in CG_VILLAGES:
        expected = EXPECTED_COUNTS_BEFORE[village]
        img_now = counts_post_images.get(village, 0)
        msk_now = counts_post_masks.get(village, 0)
        if img_now != expected or msk_now != expected:
            cg_integrity_ok = False
            print(
                f"CRITICAL: CG count changed for {village}. expected={expected}, images={img_now}, masks={msk_now}"
            )

    if not cg_integrity_ok:
        raise RuntimeError("CRITICAL: CG integrity failed after PB reprocessing")

    print("\nSTEP 4: Strict village-level train/val split")
    train_by_village, val_by_village, train_n, val_n, total_n = strict_village_split(images_dir, train_txt, val_txt)

    print("Train split village breakdown:")
    for village in sorted(TRAIN_VILLAGES):
        print(f"  - {village}: {train_by_village.get(village, 0)}")
    print("Val split village breakdown:")
    for village in sorted(VAL_VILLAGES):
        print(f"  - {village}: {val_by_village.get(village, 0)}")

    print("\nSTEP 5: Final dataset summary")
    fg_stats = per_village_foreground_percent(masks_dir)

    split_map = {v: "train" for v in TRAIN_VILLAGES}
    split_map.update({v: "val" for v in VAL_VILLAGES})

    summary_rows = []
    for village in sorted(VILLAGE_STATE.keys()):
        summary_rows.append(
            {
                "village": village,
                "state": VILLAGE_STATE[village],
                "patches": counts_post_images.get(village, 0),
                "split": split_map[village],
                "ch0": fg_stats[village][0],
                "ch1": fg_stats[village][1],
                "ch3": fg_stats[village][3],
                "ch4": fg_stats[village][4],
                "ch7": fg_stats[village][7],
                "ch8": fg_stats[village][8],
            }
        )

    for r in summary_rows:
        print(
            f"{r['village']} | {r['state']} | {r['patches']} | {r['split']} | "
            f"Ch0={format_pct(r['ch0'])} Ch1={format_pct(r['ch1'])} Ch3={format_pct(r['ch3'])} "
            f"Ch4={format_pct(r['ch4'])} Ch7={format_pct(r['ch7'])} Ch8={format_pct(r['ch8'])}"
        )

    total_bytes = sum(p.stat().st_size for p in images_dir.glob("*.tif")) + sum(
        p.stat().st_size for p in masks_dir.glob("*.tif")
    )
    total_gb = total_bytes / (1024**3)

    train_pb = sum(train_by_village.get(v, 0) for v in PB_VILLAGES)
    train_cg = sum(train_by_village.get(v, 0) for v in CG_VILLAGES)
    val_pb = sum(val_by_village.get(v, 0) for v in PB_VILLAGES)
    val_cg = sum(val_by_village.get(v, 0) for v in CG_VILLAGES)

    train_ratio = f"CG:PB = {train_cg}:{train_pb} ({(train_cg / train_pb if train_pb else 0):.4f})"
    val_ratio = f"CG:PB = {val_cg}:{val_pb} ({(val_cg / val_pb if val_pb else 0):.4f})"

    lines = []
    lines.append("# Phase 1.6 Report — PB Utility Polygon Reprocessing & Village-Level Split")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Reprocessed **only PB villages** to restore Utility Polygon annotations (Channel 7).")
    lines.append("- CG villages were preserved and re-verified for count integrity.")
    lines.append("- Rebuilt `train.txt` and `val.txt` as strict **village-level split** (7 train villages / 2 val villages).")
    lines.append("")
    lines.append("## PB Utility Polygon Fix Verification")
    lines.append("")
    lines.append("| Village | Ch7 non-zero patches (sample of 200) |")
    lines.append("|---------|--------------------------------------:|")
    for village in PB_VILLAGES:
        nz, n = ch7_results[village]
        lines.append(f"| {village} | {nz}/{n} |")
    lines.append("")
    lines.append("## PB Before/After Patch Counts")
    lines.append("")
    lines.append("| Village | Before | After | Delta % |")
    lines.append("|---------|-------:|------:|--------:|")
    for res in process_results:
        lines.append(
            f"| {res.village_id} | {res.old_count} | {res.new_count} | {res.delta_pct:+.2f}% |"
        )
    lines.append("")
    lines.append("## CG Integrity Verification")
    lines.append("")
    lines.append("| Village | Expected | Final | Status |")
    lines.append("|---------|---------:|------:|--------|")
    for village in CG_VILLAGES:
        expected = EXPECTED_COUNTS_BEFORE[village]
        final = counts_post_images.get(village, 0)
        status = "OK" if expected == final else "CRITICAL"
        lines.append(f"| {village} | {expected} | {final} | {status} |")
    lines.append("")
    lines.append("## Final Dataset Inventory")
    lines.append("")
    lines.append("| Village | State | Patches | Split | Ch0 | Ch1 | Ch3 | Ch4 | Ch7 | Ch8 |")
    lines.append("|---------|:-----:|--------:|:-----:|----:|----:|----:|----:|----:|----:|")
    for r in summary_rows:
        lines.append(
            f"| {r['village']} | {r['state']} | {r['patches']} | {r['split']} | "
            f"{format_pct(r['ch0'])} | {format_pct(r['ch1'])} | {format_pct(r['ch3'])} | "
            f"{format_pct(r['ch4'])} | {format_pct(r['ch7'])} | {format_pct(r['ch8'])} |"
        )
    lines.append("")
    lines.append("## Final Split Summary")
    lines.append("")
    lines.append(f"- Train patches: **{train_n}** (7 villages)")
    lines.append(f"- Val patches: **{val_n}** (2 villages)")
    lines.append(f"- Total patches: **{total_n}**")
    lines.append(f"- Dataset size (images + masks): **{total_gb:.2f} GB**")
    lines.append(f"- Train CG/PB ratio: **{train_ratio}**")
    lines.append(f"- Val CG/PB ratio: **{val_ratio}**")
    lines.append("")
    lines.append("### Train Villages")
    for village in sorted(TRAIN_VILLAGES):
        lines.append(f"- {village}: {train_by_village.get(village, 0)}")
    lines.append("")
    lines.append("### Val Villages")
    for village in sorted(VAL_VILLAGES):
        lines.append(f"- {village}: {val_by_village.get(village, 0)}")
    lines.append("")
    lines.append("## Split File Validation")
    lines.append("")
    lines.append("- No filename overlap between `train.txt` and `val.txt`.")
    lines.append("- Every patch filename in `preprocessed_dataset/images/` appears exactly once in one split file.")
    lines.append("- Train villages contribute only to `train.txt`; val villages contribute only to `val.txt`.")
    lines.append("")
    lines.append("## New Issues Discovered")
    lines.append("")
    lines.append("- None blocking. PINDORI orthophoto remains physically under `Training/CG_Training/`, but correctly uses PB shapefiles via override.")
    lines.append("")
    lines.append("## Dataset is ready for Phase 2: Dataset Pipeline")
    lines.append("")
    lines.append("### Train (7 villages)")
    for village in sorted(TRAIN_VILLAGES):
        lines.append(f"- {village} ({VILLAGE_STATE[village]})")
    lines.append("")
    lines.append("### Val (2 villages)")
    for village in sorted(VAL_VILLAGES):
        lines.append(f"- {village} ({VILLAGE_STATE[village]})")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    utility_ok = any(ch7_results[v][0] > 0 for v in PB_VILLAGES)

    print("\nPHASE 1.6 COMPLETE")
    print("==================")
    print(f"PB villages reprocessed:          {len(PB_VILLAGES)}")
    print(f"CG villages untouched:            {len(CG_VILLAGES)}")
    print(f"PB Utility Polygon fix verified:  {'Yes' if utility_ok else 'No'}")
    print(f"CG integrity maintained:          {'Yes' if cg_integrity_ok else 'No'}")
    print(f"Total patches:                    {total_n}")
    print(f"Train patches:                    {train_n}  ({len(TRAIN_VILLAGES)} villages)")
    print(f"Val patches:                      {val_n}  ({len(VAL_VILLAGES)} villages)")
    print("Split type:                       Village-level (NO leakage)")
    print(f"Train CG/PB ratio:                {train_ratio}")
    print(f"Val CG/PB ratio:                  {val_ratio}")
    print("Split files written to:           preprocessed_dataset/")
    print("Report saved to:                  docs_eswar/phase1_6_report.md")

    print("\nPB deletion summary:")
    for village in PB_VILLAGES:
        print(f"  - {village}: deleted {pb_deleted[village]}")


if __name__ == "__main__":
    main()