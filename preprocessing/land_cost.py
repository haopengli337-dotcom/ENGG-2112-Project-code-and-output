# preprocessing/land_cost.py

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(OUT_DIR, exist_ok=True)

GRID_H = 256
GRID_W = 256

MAP_LAYER_FILE = os.path.join(OUT_DIR, "city_map_layers_256.npz")

# Based on NSW Valuer General 2025 LGA land value summaries
# Higher score = more expensive land
LGA_COST_TABLE = [
    ("Sydney", 1.00),
    ("Woollahra", 0.96),
    ("Mosman", 0.94),
    ("North Sydney", 0.90),
    ("Ku-ring-gai", 0.86),
    ("Lane Cove", 0.82),
    ("Ryde", 0.76),
    ("Parramatta", 0.70),
    ("Bayside", 0.66),
    ("Georges River", 0.64),
    ("Canterbury-Bankstown", 0.58),
    ("Cumberland", 0.55),
    ("The Hills Shire", 0.52),
    ("Blacktown", 0.46),
    ("Liverpool", 0.42),
    ("Fairfield", 0.40),
    ("Camden", 0.36),
    ("Campbelltown", 0.34),
    ("Penrith", 0.32),
]

def normalise(arr):
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def load_map_layers():
    if not os.path.exists(MAP_LAYER_FILE):
        raise FileNotFoundError(
            f"Missing map tensor:\n{MAP_LAYER_FILE}\nRun preprocessing/grid.py first."
        )

    data = np.load(MAP_LAYER_FILE, allow_pickle=True)
    X_map = data["X_map"]
    layer_names = list(data["layers"])

    layer_dict = {}
    for i, name in enumerate(layer_names):
        layer_dict[str(name)] = X_map[i]

    return layer_dict

def get_layer(layer_dict, name):
    return layer_dict.get(name, np.zeros((GRID_H, GRID_W), dtype=np.float32))

def build_land_layers():
    print("Building land cost and land availability layers...")

    layer_dict = load_map_layers()

    roads = get_layer(layer_dict, "roads")
    primary_roads = get_layer(layer_dict, "primary_roads")
    residential_roads = get_layer(layer_dict, "residential_roads")
    commercial = get_layer(layer_dict, "commercial")
    residential = get_layer(layer_dict, "residential")
    industrial = get_layer(layer_dict, "industrial")
    parks = get_layer(layer_dict, "parks")
    water = get_layer(layer_dict, "water")
    buildings = get_layer(layer_dict, "buildings")

    # ======================================================
    # 1. LGA-level land cost proxy table
    # ======================================================

    cost_df = pd.DataFrame(
        LGA_COST_TABLE,
        columns=["lga", "land_cost_score"]
    )

    cost_df["land_cost_level"] = pd.cut(
        cost_df["land_cost_score"],
        bins=[0, 0.4, 0.65, 0.85, 1.0],
        labels=["low", "medium", "high", "very_high"],
        include_lowest=True,
    )

    cost_table_path = os.path.join(OUT_DIR, "land_cost_lga_proxy.csv")
    cost_df.to_csv(cost_table_path, index=False)
    print("Saved:", cost_table_path)

    # ======================================================
    # 2. Spatial land cost proxy
    # ======================================================
    # Since no LGA boundary shapefile is available yet,
    # cost is approximated using urban intensity:
    # CBD-like/commercial/building-dense areas = higher cost
    # outer industrial/low-density areas = lower cost

    urban_intensity = normalise(
        0.35 * commercial +
        0.30 * buildings +
        0.20 * primary_roads +
        0.10 * roads +
        0.05 * residential
    )

    land_cost_proxy = normalise(
        0.65 * urban_intensity +
        0.25 * commercial +
        0.10 * buildings
    )

    # ======================================================
    # 3. No-build zones
    # ======================================================

    no_build_zone = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    no_build_zone[water > 0] = 1.0
    no_build_zone[parks > 0] = 1.0

    # ======================================================
    # 4. Land availability
    # ======================================================
    # Higher score = easier to use for charging infrastructure

    land_availability = (
        0.40 * industrial +
        0.30 * commercial +
        0.15 * roads +
        0.10 * primary_roads +
        0.05 * residential_roads
    )

    land_availability = normalise(land_availability)
    land_availability[no_build_zone > 0] = 0.0

    # ======================================================
    # 5. Parking / roadside proxy
    # ======================================================

    parking_land_proxy = normalise(
        0.35 * commercial +
        0.25 * industrial +
        0.20 * primary_roads +
        0.15 * roads +
        0.05 * buildings
    )
    parking_land_proxy[no_build_zone > 0] = 0.0

    # ======================================================
    # 6. Construction suitability
    # ======================================================
    # Good = available land + road access + non-water/non-park + not too expensive

    construction_suitability = (
        0.35 * land_availability +
        0.25 * parking_land_proxy +
        0.20 * primary_roads +
        0.10 * commercial +
        0.10 * industrial
    )

    construction_suitability = normalise(construction_suitability)
    construction_suitability[no_build_zone > 0] = 0.0

    # ======================================================
    # 7. Final build constraint
    # ======================================================

    feasible_build_area = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    feasible_build_area[
        (construction_suitability > 0.25) &
        (no_build_zone == 0)
    ] = 1.0

    layers = {
        "land_cost_proxy": land_cost_proxy,
        "land_availability": land_availability,
        "construction_suitability": construction_suitability,
        "no_build_zone": no_build_zone,
        "parking_land_proxy": parking_land_proxy,
        "feasible_build_area": feasible_build_area,
    }

    print("\nLayer summary:")
    for name, grid in layers.items():
        print(
            f"{name}: min={grid.min():.3f}, "
            f"max={grid.max():.3f}, "
            f"mean={grid.mean():.3f}, "
            f"active_cells={(grid > 0).sum()}"
        )

    # ======================================================
    # Save single CSV layers
    # ======================================================

    stacked = []

    for name, grid in layers.items():
        out_path = os.path.join(OUT_DIR, f"{name}.csv")
        pd.DataFrame(grid).to_csv(out_path, index=False, header=False)
        stacked.append(grid)
        print("Saved:", out_path)

    tensor = np.stack(stacked, axis=0)

    npz_path = os.path.join(OUT_DIR, "land_layers_256.npz")

    np.savez_compressed(
        npz_path,
        X_land=tensor,
        layers=np.array(list(layers.keys())),
        grid_size=np.array([GRID_H, GRID_W]),
    )

    print("Saved:", npz_path)

    flat_data = {}

    for name, grid in layers.items():
        flat_data[name] = grid.reshape(-1)

    flat_df = pd.DataFrame(flat_data)
    flat_df["grid_id"] = np.arange(GRID_H * GRID_W)
    flat_df["row"] = flat_df["grid_id"] // GRID_W
    flat_df["col"] = flat_df["grid_id"] % GRID_W

    flat_path = os.path.join(OUT_DIR, "land_layers_256_flat.csv")
    flat_df.to_csv(flat_path, index=False)
    print("Saved:", flat_path)

    print("\nLand cost preprocessing complete.")
    print("Final land tensor shape:", tensor.shape)

if __name__ == "__main__":
    build_land_layers()