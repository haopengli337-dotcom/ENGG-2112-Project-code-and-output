# preprocessing/grid.py

import os
import numpy as np
import pandas as pd
import geopandas as gpd

# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_MAP_DIR = os.path.join(BASE_DIR, "R_data", "Greater Sydney map")
OUT_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Grid setting
# ============================================================

GRID_H = 256
GRID_W = 256

# Greater Sydney approximate bounding box
# lon_min, lat_min, lon_max, lat_max
SYDNEY_BOUNDS = [150.50, -34.25, 151.45, -33.35]

LAYER_NAMES = [
    "roads",
    "primary_roads",
    "residential_roads",
    "commercial",
    "residential",
    "industrial",
    "parks",
    "water",
    "buildings",
]

GIS_SUBLAYERS = [
    "points",
    "lines",
    "multilinestrings",
    "multipolygons",
    "other_relations",
]


# ============================================================
# Coordinate conversion
# ============================================================

def coord_to_grid(x, y, bounds):
    minx, miny, maxx, maxy = bounds

    col = int((x - minx) / (maxx - minx + 1e-9) * (GRID_W - 1))
    row = int((maxy - y) / (maxy - miny + 1e-9) * (GRID_H - 1))

    row = max(0, min(GRID_H - 1, row))
    col = max(0, min(GRID_W - 1, col))

    return row, col


def draw_line(x1, y1, x2, y2, grid, bounds):
    r1, c1 = coord_to_grid(x1, y1, bounds)
    r2, c2 = coord_to_grid(x2, y2, bounds)

    steps = max(abs(r2 - r1), abs(c2 - c1)) + 1

    for i in range(steps):
        t = i / max(steps - 1, 1)
        r = int(round(r1 + t * (r2 - r1)))
        c = int(round(c1 + t * (c2 - c1)))

        if 0 <= r < GRID_H and 0 <= c < GRID_W:
            grid[r, c] = 1.0


# ============================================================
# Rasterisation
# ============================================================

def rasterize_geom(geom, grid, bounds):
    if geom is None or geom.is_empty:
        return

    gtype = geom.geom_type

    try:
        if gtype == "Point":
            r, c = coord_to_grid(geom.x, geom.y, bounds)
            grid[r, c] = 1.0

        elif gtype == "MultiPoint":
            for p in geom.geoms:
                rasterize_geom(p, grid, bounds)

        elif gtype == "LineString":
            coords = list(geom.coords)

            if len(coords) == 1:
                x, y = coords[0]
                r, c = coord_to_grid(x, y, bounds)
                grid[r, c] = 1.0

            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                draw_line(x1, y1, x2, y2, grid, bounds)

        elif gtype == "MultiLineString":
            for line in geom.geoms:
                rasterize_geom(line, grid, bounds)

        elif gtype == "Polygon":
            minx, miny, maxx, maxy = geom.bounds

            r1, c1 = coord_to_grid(minx, maxy, bounds)
            r2, c2 = coord_to_grid(maxx, miny, bounds)

            rmin, rmax = sorted([r1, r2])
            cmin, cmax = sorted([c1, c2])

            grid[rmin:rmax + 1, cmin:cmax + 1] = 1.0

        elif gtype == "MultiPolygon":
            for poly in geom.geoms:
                rasterize_geom(poly, grid, bounds)

    except Exception:
        pass


# ============================================================
# Feature classification
# ============================================================

def clean_value(v):
    if pd.isna(v):
        return ""

    return str(v).strip().lower()


def get_field(row, names):
    for name in names:
        if name in row.index:
            value = clean_value(row[name])
            if value not in ["", "none", "nan"]:
                return value

    return ""


def classify_feature(row, sublayer):
    highway = get_field(row, ["highway"])
    landuse = get_field(row, ["landuse"])
    leisure = get_field(row, ["leisure"])
    natural = get_field(row, ["natural"])
    waterway = get_field(row, ["waterway"])
    building = get_field(row, ["building"])
    shop = get_field(row, ["shop"])
    amenity = get_field(row, ["amenity"])
    railway = get_field(row, ["railway"])

    # ----------------------------
    # Road classification
    # ----------------------------
    if highway in [
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
    ]:
        return "primary_roads"

    if highway in [
        "residential",
        "living_street",
        "service",
        "unclassified",
    ]:
        return "residential_roads"

    if highway not in ["", "none", "nan"]:
        return "roads"

    # ----------------------------
    # Water
    # ----------------------------
    if natural in ["water", "bay", "coastline"] or waterway not in ["", "none", "nan"]:
        return "water"

    # ----------------------------
    # Parks / green areas
    # ----------------------------
    if leisure in [
        "park",
        "garden",
        "playground",
        "recreation_ground",
        "sports_centre",
        "pitch",
        "nature_reserve",
    ]:
        return "parks"

    if landuse in [
        "grass",
        "forest",
        "recreation_ground",
        "village_green",
        "meadow",
        "orchard",
        "cemetery",
    ]:
        return "parks"

    if natural in [
        "wood",
        "grassland",
        "scrub",
        "heath",
        "tree",
    ]:
        return "parks"

    # ----------------------------
    # Land use
    # ----------------------------
    if landuse in ["commercial", "retail"]:
        return "commercial"

    if landuse in ["residential"]:
        return "residential"

    if landuse in ["industrial", "brownfield", "construction", "railway"]:
        return "industrial"

    # ----------------------------
    # Commercial POI
    # ----------------------------
    if shop not in ["", "none", "nan"]:
        return "commercial"

    if amenity in [
        "cafe",
        "restaurant",
        "fast_food",
        "bank",
        "fuel",
        "parking",
        "marketplace",
        "cinema",
        "theatre",
    ]:
        return "commercial"

    # ----------------------------
    # Transport / railway treated as roads base
    # ----------------------------
    if railway not in ["", "none", "nan"]:
        return "roads"

    # ----------------------------
    # Buildings
    # ----------------------------
    if building not in ["", "none", "nan"]:
        return "buildings"

    return None


# ============================================================
# File loading
# ============================================================

def get_map_files():
    files = []

    for root, dirs, names in os.walk(RAW_MAP_DIR):
        for name in names:
            if name.lower().startswith("map"):
                files.append(os.path.join(root, name))

    files = sorted(files)

    if len(files) == 0:
        raise FileNotFoundError(f"No map1-map9 files found in:\n{RAW_MAP_DIR}")

    return files


def read_all_gis_layers(map_files):
    all_gdfs = []

    for path in map_files:
        basename = os.path.basename(path)

        for sublayer in GIS_SUBLAYERS:
            try:
                gdf = gpd.read_file(path, layer=sublayer)

                if gdf is None or len(gdf) == 0:
                    continue

                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")

                gdf = gdf.to_crs("EPSG:4326")

                # clip to Greater Sydney
                gdf = gdf.cx[
                    SYDNEY_BOUNDS[0]:SYDNEY_BOUNDS[2],
                    SYDNEY_BOUNDS[1]:SYDNEY_BOUNDS[3],
                ]

                if len(gdf) == 0:
                    continue

                print(f"Loaded {basename}, layer={sublayer}, rows={len(gdf)}")

                all_gdfs.append((basename, sublayer, gdf))

            except Exception:
                continue

    if len(all_gdfs) == 0:
        raise ValueError("No readable GIS layers found inside Sydney bounds.")

    return all_gdfs


# ============================================================
# Main
# ============================================================

def build_city_layers():
    print("Reading Greater Sydney map files...")
    print("Input folder:", RAW_MAP_DIR)

    map_files = get_map_files()

    print(f"\nFound {len(map_files)} map files:")
    for f in map_files:
        print(" -", os.path.basename(f))

    print("\nReading and clipping GIS sublayers...")
    all_gdfs = read_all_gis_layers(map_files)

    bounds = SYDNEY_BOUNDS

    print("\nUsing fixed Sydney bounds:")
    print(bounds)

    layers = {
        name: np.zeros((GRID_H, GRID_W), dtype=np.float32)
        for name in LAYER_NAMES
    }

    feature_count = {name: 0 for name in LAYER_NAMES}

    print("\nRasterizing features...")

    for basename, sublayer, gdf in all_gdfs:
        for _, row in gdf.iterrows():
            target_layer = classify_feature(row, sublayer)

            if target_layer is None:
                continue

            geom = row.geometry

            if geom is None or geom.is_empty:
                continue

            rasterize_geom(geom, layers[target_layer], bounds)
            feature_count[target_layer] += 1

    print("\nFeature count by layer:")
    for name in LAYER_NAMES:
        active_cells = int(np.sum(layers[name] > 0))
        print(f"{name}: features={feature_count[name]}, active_cells={active_cells}")

    print("\nSaving single map layers...")

    stacked = []

    for name in LAYER_NAMES:
        grid = layers[name]
        stacked.append(grid)

        out_csv = os.path.join(OUT_DIR, f"{name}.csv")

        pd.DataFrame(grid).to_csv(
            out_csv,
            index=False,
            header=False,
        )

        print(f"Saved: {out_csv}")

    tensor = np.stack(stacked, axis=0)

    npz_path = os.path.join(OUT_DIR, "city_map_layers_256.npz")

    np.savez_compressed(
        npz_path,
        X_map=tensor,
        layers=np.array(LAYER_NAMES),
        bounds=np.array(bounds),
        grid_size=np.array([GRID_H, GRID_W]),
    )

    print(f"Saved: {npz_path}")

    flat_data = {}

    for name in LAYER_NAMES:
        flat_data[name] = layers[name].reshape(-1)

    flat_df = pd.DataFrame(flat_data)
    flat_df["grid_id"] = np.arange(GRID_H * GRID_W)
    flat_df["row"] = flat_df["grid_id"] // GRID_W
    flat_df["col"] = flat_df["grid_id"] % GRID_W

    flat_path = os.path.join(OUT_DIR, "city_map_layers_256_flat.csv")
    flat_df.to_csv(flat_path, index=False)

    print(f"Saved: {flat_path}")

    print("\nMap preprocessing complete.")
    print("Final tensor shape:", tensor.shape)
    print("\nOutput files:")
    for name in LAYER_NAMES:
        print(f"- datasets/{name}.csv")
    print("- datasets/city_map_layers_256.npz")
    print("- datasets/city_map_layers_256_flat.csv")


if __name__ == "__main__":
    build_city_layers()