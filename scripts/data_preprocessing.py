# ============================================================
# SVAMITVA FULL PREPROCESSING WITH WINDOW-BASED RASTERIZATION
# ============================================================

import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from tqdm import tqdm

# ------------------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------------------

TRAINING_DIRS = {
    "Training/PB_Training": "Training/PB_Training/shp-file",
    "Training/CG_Training": "Training/CG_Training/shp-file",
}

# Override: villages mis-filed in the wrong training folder
SHP_OVERRIDES = {
    "PINDORI MAYA SINGH-TUGALWAL_28456_ortho": "Training/PB_Training/shp-file",
    "BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO": "Training/CG_Training/shp-file",
}

OUTPUT_ROOT = "preprocessed_dataset"

PATCH_SIZE = 512
STRIDE = 384

# ------------------------------------------------------------
# OUTPUT DIRECTORIES
# ------------------------------------------------------------

img_out_dir = os.path.join(OUTPUT_ROOT, "images")
mask_out_dir = os.path.join(OUTPUT_ROOT, "masks")

os.makedirs(img_out_dir, exist_ok=True)
os.makedirs(mask_out_dir, exist_ok=True)

# ------------------------------------------------------------
# GLOBAL CLASS ENCODING MAPS
# ------------------------------------------------------------

BUILDING_MAP = {1:1, 2:2, 3:3, 4:4}
ROAD_MAP = {1:1, 3:2, 4:3, 5:4, 6:5}
RAILWAY_MAP = {1:1, 2:1}
WATER_POLY_MAP = {1:1, 2:2, 3:3, 5:4, 8:5, 10:6}
WATER_LINE_MAP = {1:1, 2:2, 11:3}
WATER_POINT_MAP = {1:1, 2:2, 3:3}
UTILITY_POINT_MAP = {1:1, 2:2}
UTILITY_POLY_MAP = {1:1}
BRIDGE_MAP = {7:1}

# ------------------------------------------------------------
# SHAPEFILE LOADER
# ------------------------------------------------------------

def load_layer(path, crs):

    if not os.path.exists(path):
        return None

    gdf = gpd.read_file(path)

    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[gdf.is_valid]

    return gdf


# ------------------------------------------------------------
# PATCH RASTERIZATION
# ------------------------------------------------------------

def rasterize_patch(gdf, column, mapping, patch_size, transform):

    if gdf is None or len(gdf) == 0:
        return np.zeros((patch_size, patch_size), dtype=np.uint8)

    shapes = []

    for _, row in gdf.iterrows():

        val = row[column]

        if val in mapping:
            shapes.append((row.geometry, mapping[val]))

    if len(shapes) == 0:
        return np.zeros((patch_size, patch_size), dtype=np.uint8)

    mask = rasterize(
        shapes,
        out_shape=(patch_size, patch_size),
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    return mask


# ------------------------------------------------------------
# MAIN PROCESSING
# ------------------------------------------------------------

# Build village list: (ortho_path, shp_folder) from each training dir
villages = []
for train_dir, shp_folder in TRAINING_DIRS.items():
    ortho_dir = os.path.join(train_dir)
    if not os.path.isdir(ortho_dir):
        continue
    for f in sorted(os.listdir(ortho_dir)):
        if f.endswith(".tif"):
            villages.append((os.path.join(ortho_dir, f), shp_folder))

for ortho_path, shp_folder in villages:

    village_id = os.path.splitext(os.path.basename(ortho_path))[0]

    # Apply shapefile override for mis-filed villages
    shp_folder = SHP_OVERRIDES.get(village_id, shp_folder)

    print("\nProcessing:", village_id)

    with rasterio.open(ortho_path) as src:

        height = src.height
        width = src.width
        crs = src.crs

    # --------------------------------------------------------
    # LOAD SHAPEFILES (handle naming differences per state)
    # --------------------------------------------------------

    def shp_path(name):
        return os.path.join(shp_folder, name)

    # Building: CG="Built_Up_Area_type.shp", PB="Built_Up_Area_typ.shp"
    building_shp = shp_path("Built_Up_Area_type.shp")
    if not os.path.exists(building_shp):
        building_shp = shp_path("Built_Up_Area_typ.shp")

    # Utility_Poly: CG="Utility_Poly.shp", PB="Utility_Poly_.shp"
    utility_poly_shp = shp_path("Utility_Poly.shp")
    if not os.path.exists(utility_poly_shp):
        utility_poly_shp = shp_path("Utility_Poly_.shp")

    building_gdf = load_layer(building_shp, crs)
    road_gdf = load_layer(shp_path("Road.shp"), crs)
    railway_gdf = load_layer(shp_path("Railway.shp"), crs)
    water_poly_gdf = load_layer(shp_path("Water_Body.shp"), crs)
    water_line_gdf = load_layer(shp_path("Water_Body_Line.shp"), crs)
    water_point_gdf = load_layer(shp_path("Waterbody_Point.shp"), crs)
    utility_point_gdf = load_layer(shp_path("Utility.shp"), crs)
    utility_poly_gdf = load_layer(utility_poly_shp, crs)
    bridge_gdf = load_layer(shp_path("Bridge.shp"), crs)

    # --------------------------------------------------------
    # PATCH GENERATION
    # --------------------------------------------------------

    with rasterio.open(ortho_path) as img_src:

        y_steps = list(range(0, height - PATCH_SIZE, STRIDE))
        x_steps = list(range(0, width - PATCH_SIZE, STRIDE))

        for y in tqdm(y_steps, desc=village_id):

            for x in x_steps:

                window = Window(x, y, PATCH_SIZE, PATCH_SIZE)

                img_tile = img_src.read(window=window)

                tile_transform = rasterio.windows.transform(
                    window,
                    img_src.transform
                )

                bounds = rasterio.windows.bounds(
                    window,
                    img_src.transform
                )

                def clip(gdf):

                    if gdf is None:
                        return None

                    return gdf.cx[
                        bounds[0]:bounds[2],
                        bounds[1]:bounds[3]
                    ]

                building = clip(building_gdf)
                road = clip(road_gdf)
                railway = clip(railway_gdf)
                water_poly = clip(water_poly_gdf)
                water_line = clip(water_line_gdf)
                water_point = clip(water_point_gdf)
                utility_point = clip(utility_point_gdf)
                utility_poly = clip(utility_poly_gdf)
                bridge = clip(bridge_gdf)

                mask_tile = np.stack([

                    rasterize_patch(building,"Roof_type",BUILDING_MAP,PATCH_SIZE,tile_transform),
                    rasterize_patch(road,"Road_type",ROAD_MAP,PATCH_SIZE,tile_transform),
                    rasterize_patch(railway,"Railway_Ty",RAILWAY_MAP,PATCH_SIZE,tile_transform),
                    rasterize_patch(water_poly,"Water_Body",WATER_POLY_MAP,PATCH_SIZE,tile_transform),
                    rasterize_patch(water_line,"Water_Body",WATER_LINE_MAP,PATCH_SIZE,tile_transform),
                    rasterize_patch(water_point,"Water_Bodi",WATER_POINT_MAP,PATCH_SIZE,tile_transform),
                    rasterize_patch(utility_point,"Utility_Ty",UTILITY_POINT_MAP,PATCH_SIZE,tile_transform),
                    rasterize_patch(utility_poly,"Utility_Ty",UTILITY_POLY_MAP,PATCH_SIZE,tile_transform),
                    rasterize_patch(bridge,"Bridge_typ",BRIDGE_MAP,PATCH_SIZE,tile_transform)

                ], axis=0)

                if np.sum(mask_tile) == 0:
                    continue

                img_meta = img_src.meta.copy()

                img_meta.update(
                    height=PATCH_SIZE,
                    width=PATCH_SIZE,
                    transform=tile_transform
                )

                mask_meta = {
                    "driver":"GTiff",
                    "height":PATCH_SIZE,
                    "width":PATCH_SIZE,
                    "count":9,
                    "dtype":"uint8",
                    "crs":img_src.crs,
                    "transform":tile_transform
                }

                name = f"{village_id}_{y}_{x}.tif"

                img_path = os.path.join(img_out_dir, name)
                mask_path = os.path.join(mask_out_dir, name)

                with rasterio.open(img_path,"w",**img_meta) as dst:
                    dst.write(img_tile)

                with rasterio.open(mask_path,"w",**mask_meta) as dst:
                    dst.write(mask_tile)

print("\nPreprocessing complete.")