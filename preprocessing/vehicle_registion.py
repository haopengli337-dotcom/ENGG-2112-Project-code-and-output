# =========================
# preprocessing/vehicle_registration.py
# Build vehicle registration / EV adoption proxy layers
# =========================

import os
import glob
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(
    BASE_DIR,
    "R_data",
    "Number of vehicle registrations"
)

OUT_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(OUT_DIR, exist_ok=True)

GRID_H = 256
GRID_W = 256

SYDNEY_BOUNDS = [150.50, -34.25, 151.45, -33.35]

TRAFFIC_LAYER_FILE = os.path.join(OUT_DIR, "traffic_layers_256.npz")
LAND_LAYER_FILE = os.path.join(OUT_DIR, "land_layers_256.npz")


# =========================
# Utility functions
# =========================

def robust_normalise(arr, p_low=1, p_high=99):
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if np.all(arr == arr.flat[0]):
        return np.zeros_like(arr, dtype=np.float32)

    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)

    if hi - lo < 1e-9:
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-9:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - mn) / (mx - mn)).astype(np.float32)

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def smooth_layer(arr, sigma=2.0):
    arr = gaussian_filter(arr.astype(np.float32), sigma=sigma)
    return robust_normalise(arr)


def print_stats(name, arr):
    arr = np.nan_to_num(arr)
    print(
        f"{name:35s} "
        f"min={arr.min():.4f} "
        f"max={arr.max():.4f} "
        f"mean={arr.mean():.4f} "
        f"std={arr.std():.4f} "
        f"active={(arr > 1e-6).sum()}"
    )


def get_layer_from_npz(npz_path, data_key_candidates, layer_name_candidates):
    if not os.path.exists(npz_path):
        print(f"[WARNING] Missing file: {npz_path}")
        return np.zeros((GRID_H, GRID_W), dtype=np.float32)

    data = np.load(npz_path, allow_pickle=True)

    data_key = None
    for k in data_key_candidates:
        if k in data.files:
            data_key = k
            break

    if data_key is None:
        print(f"[WARNING] No valid tensor key found in {npz_path}")
        print("Available keys:", data.files)
        return np.zeros((GRID_H, GRID_W), dtype=np.float32)

    X = data[data_key]

    if "layers" not in data.files:
        print(f"[WARNING] No layer names found in {npz_path}")
        return robust_normalise(X[0])

    names = [str(x) for x in list(data["layers"])]

    for target_name in layer_name_candidates:
        if target_name in names:
            idx = names.index(target_name)
            print(f"Loaded layer: {target_name} from {os.path.basename(npz_path)}")
            return robust_normalise(X[idx])

    print(f"[WARNING] None of {layer_name_candidates} found in {npz_path}")
    print("Available layers:", names)

    return robust_normalise(X[0])


# =========================
# Load proxy layers
# =========================

def load_proxy_layers():
    traffic_proxy = get_layer_from_npz(
        TRAFFIC_LAYER_FILE,
        data_key_candidates=["X_traffic", "X", "arr_0"],
        layer_name_candidates=[
            "traffic_daily_average",
            "traffic_volume",
            "traffic_density",
            "daily_average",
        ],
    )

    land_cost_proxy = get_layer_from_npz(
        LAND_LAYER_FILE,
        data_key_candidates=["X_land", "X", "arr_0"],
        layer_name_candidates=[
            "land_cost_proxy",
            "map_commercial",
            "land_construction_suitability",
            "commercial",
        ],
    )

    land_suitability = get_layer_from_npz(
        LAND_LAYER_FILE,
        data_key_candidates=["X_land", "X", "arr_0"],
        layer_name_candidates=[
            "land_construction_suitability",
            "map_commercial",
            "land_cost_proxy",
            "commercial",
        ],
    )

    traffic_proxy = smooth_layer(traffic_proxy, sigma=2.0)
    land_cost_proxy = smooth_layer(land_cost_proxy, sigma=1.5)
    land_suitability = smooth_layer(land_suitability, sigma=1.5)

    print("\nProxy layer statistics:")
    print_stats("traffic_proxy", traffic_proxy)
    print_stats("land_cost_proxy", land_cost_proxy)
    print_stats("land_suitability", land_suitability)

    return traffic_proxy, land_cost_proxy, land_suitability


# =========================
# Registration file handling
# =========================

def collect_registration_files():
    snapshot_dirs = [
        os.path.join(RAW_DIR, "tfnsw_vehicle_registrations_snapshot_2025"),
        os.path.join(RAW_DIR, "tfnsw_vehicle_registrations_snapshot_2026"),
    ]

    files = []

    for d in snapshot_dirs:
        if os.path.exists(d):
            files.extend(glob.glob(os.path.join(d, "*.csv")))

    return sorted(files)


def extract_summary_features(df):
    summary = {}

    numeric_df = df.select_dtypes(include=[np.number])

    summary["rows"] = len(df)

    if not numeric_df.empty:
        summary["numeric_mean"] = float(numeric_df.mean().mean())
        summary["numeric_sum"] = float(numeric_df.sum().sum())
        summary["numeric_max"] = float(numeric_df.max().max())
    else:
        summary["numeric_mean"] = 0.0
        summary["numeric_sum"] = 0.0
        summary["numeric_max"] = 0.0

    return summary


def load_registration_strength():
    files = collect_registration_files()

    if len(files) == 0:
        print(f"[WARNING] No vehicle registration CSV files found in:\n{RAW_DIR}")
        return 1.0, pd.DataFrame()

    print("\nFound registration files:")
    for f in files:
        print(" -", os.path.basename(f))

    summaries = []
    total_rows = 0

    for file in files:
        try:
            df = pd.read_csv(file, low_memory=False)
            summary = extract_summary_features(df)
            summary["file"] = os.path.basename(file)
            summaries.append(summary)
            total_rows += len(df)

            print(
                f"Loaded {os.path.basename(file)} "
                f"rows={len(df)} cols={len(df.columns)}"
            )

        except Exception as e:
            print(f"[WARNING] Skipped {file}")
            print(e)

    summary_df = pd.DataFrame(summaries)

    summary_path = os.path.join(
        OUT_DIR,
        "vehicle_registration_file_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)
    print("\nSaved:", summary_path)
    print("Total registration rows:", total_rows)

    if total_rows <= 0:
        return 1.0, summary_df

    strength = np.log1p(total_rows)
    strength = float(np.clip(strength / 15.0, 0.5, 1.5))

    return strength, summary_df


# =========================
# Main builder
# =========================

def build_vehicle_registration_layers():
    print("Reading vehicle registration data...")
    print("Input folder:", RAW_DIR)

    registration_strength, _ = load_registration_strength()

    traffic_proxy, land_cost_proxy, land_suitability = load_proxy_layers()

    # If one proxy is dead, replace it with the other useful proxy
    if traffic_proxy.std() < 1e-6 and land_suitability.std() > 1e-6:
        print("[WARNING] traffic_proxy is inactive; using land_suitability fallback.")
        traffic_proxy = land_suitability.copy()

    if land_cost_proxy.std() < 1e-6 and land_suitability.std() > 1e-6:
        print("[WARNING] land_cost_proxy is inactive; using land_suitability fallback.")
        land_cost_proxy = land_suitability.copy()

    if land_suitability.std() < 1e-6 and land_cost_proxy.std() > 1e-6:
        print("[WARNING] land_suitability is inactive; using land_cost_proxy fallback.")
        land_suitability = land_cost_proxy.copy()

    urban_activity = robust_normalise(
        0.55 * traffic_proxy +
        0.25 * land_cost_proxy +
        0.20 * land_suitability
    )

    ev_adoption_proxy = robust_normalise(
        registration_strength * (
            0.50 * traffic_proxy +
            0.30 * land_cost_proxy +
            0.20 * land_suitability
        )
    )

    commercial_vehicle_density = robust_normalise(
        0.50 * land_cost_proxy +
        0.35 * urban_activity +
        0.15 * traffic_proxy
    )

    vehicle_registration_density = robust_normalise(
        0.70 * traffic_proxy +
        0.20 * urban_activity +
        0.10 * land_suitability
    )

    private_vehicle_density = robust_normalise(
        0.45 * traffic_proxy +
        0.30 * land_suitability +
        0.25 * (1.0 - land_cost_proxy)
    )

    vehicle_growth_proxy = robust_normalise(
        0.40 * ev_adoption_proxy +
        0.30 * urban_activity +
        0.20 * land_suitability +
        0.10 * (1.0 - land_cost_proxy)
    )

    # Smooth final proxy layers slightly to avoid isolated single-cell noise
    layers = {
        "vehicle_registration_density": smooth_layer(vehicle_registration_density, sigma=1.2),
        "ev_adoption_proxy": smooth_layer(ev_adoption_proxy, sigma=1.2),
        "commercial_vehicle_density": smooth_layer(commercial_vehicle_density, sigma=1.0),
        "private_vehicle_density": smooth_layer(private_vehicle_density, sigma=1.2),
        "vehicle_growth_proxy": smooth_layer(vehicle_growth_proxy, sigma=1.2),
    }

    print("\nFinal vehicle layer summary:")
    for name, grid in layers.items():
        print_stats(name, grid)

    # =========================
    # Save individual CSV layers
    # =========================

    stacked = []

    for name, grid in layers.items():
        out_path = os.path.join(OUT_DIR, f"{name}.csv")

        pd.DataFrame(grid).to_csv(
            out_path,
            index=False,
            header=False
        )

        stacked.append(grid)
        print("Saved:", out_path)

    # =========================
    # Save tensor
    # =========================

    tensor = np.stack(stacked, axis=0).astype(np.float32)

    npz_path = os.path.join(
        OUT_DIR,
        "vehicle_registration_layers_256.npz"
    )

    np.savez_compressed(
        npz_path,
        X_vehicle=tensor,
        layers=np.array(list(layers.keys())),
        bounds=np.array(SYDNEY_BOUNDS),
        grid_size=np.array([GRID_H, GRID_W]),
    )

    print("Saved:", npz_path)

    # =========================
    # Save flat table
    # =========================

    flat_data = {}

    for name, grid in layers.items():
        flat_data[name] = grid.reshape(-1)

    flat_df = pd.DataFrame(flat_data)

    flat_df["grid_id"] = np.arange(GRID_H * GRID_W)
    flat_df["row"] = flat_df["grid_id"] // GRID_W
    flat_df["col"] = flat_df["grid_id"] % GRID_W

    flat_path = os.path.join(
        OUT_DIR,
        "vehicle_registration_layers_256_flat.csv"
    )

    flat_df.to_csv(flat_path, index=False)

    print("Saved:", flat_path)

    print("\nVehicle registration preprocessing complete.")
    print("Final vehicle tensor shape:", tensor.shape)


if __name__ == "__main__":
    build_vehicle_registration_layers()