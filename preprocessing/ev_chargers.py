# preprocessing/make_ev_chargers.py

import os
import re
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_FILE = os.path.join(
    BASE_DIR,
    "R_data",
    "EV Charging Stations",
    "ev_20251216.csv"
)
OUT_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(OUT_DIR, exist_ok=True)

GRID_H = 256
GRID_W = 256

# Must match grid.py
SYDNEY_BOUNDS = [150.50, -34.25, 151.45, -33.35]
# lon_min, lat_min, lon_max, lat_max


def coord_to_grid(lon, lat, bounds):
    minx, miny, maxx, maxy = bounds

    col = int((lon - minx) / (maxx - minx + 1e-9) * (GRID_W - 1))
    row = int((maxy - lat) / (maxy - miny + 1e-9) * (GRID_H - 1))

    row = max(0, min(GRID_H - 1, row))
    col = max(0, min(GRID_W - 1, col))

    return row, col


def parse_power_kw(value):
    if pd.isna(value):
        return 0.0

    text = str(value).lower().replace(" ", "")

    nums = re.findall(r"(\d+\.?\d*)", text)

    if len(nums) == 0:
        return 0.0

    nums = [float(x) for x in nums]

    # Use maximum rating if multiple chargers are listed
    return max(nums)


def classify_speed(row):
    charger_type = str(row.get("Charger_Type", "")).lower()
    power_kw = row.get("power_kw", 0)

    if "upcoming" in charger_type:
        return "upcoming"

    if "dc" in charger_type or power_kw >= 50:
        return "fast"

    return "slow"


def build_ev_charger_layers():
    print("Reading EV charger data...")
    print("Input file:", RAW_FILE)

    df = pd.read_csv(RAW_FILE)

    print("Raw shape:", df.shape)
    print("Columns:", list(df.columns))

    required_cols = ["Latitude", "Longitude", "Number_of_plugs"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=["Latitude", "Longitude"]).copy()

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Number_of_plugs"] = pd.to_numeric(df["Number_of_plugs"], errors="coerce").fillna(1)

    df = df.dropna(subset=["Latitude", "Longitude"]).copy()

    lon_min, lat_min, lon_max, lat_max = SYDNEY_BOUNDS

    df = df[
        (df["Longitude"] >= lon_min) &
        (df["Longitude"] <= lon_max) &
        (df["Latitude"] >= lat_min) &
        (df["Latitude"] <= lat_max)
    ].copy()

    print("After Sydney clipping:", df.shape)

    df["power_kw"] = df["Charger_rating"].apply(parse_power_kw)
    df["speed_class"] = df.apply(classify_speed, axis=1)

    rows = []
    cols = []

    for _, row in df.iterrows():
        r, c = coord_to_grid(row["Longitude"], row["Latitude"], SYDNEY_BOUNDS)
        rows.append(r)
        cols.append(c)

    df["grid_row"] = rows
    df["grid_col"] = cols
    df["grid_id"] = df["grid_row"] * GRID_W + df["grid_col"]

    # =========================
    # Create layers
    # =========================

    station_count = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    plug_count = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    slow_charger_count = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    fast_charger_count = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    upcoming_charger_count = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    total_power_kw = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    for _, row in df.iterrows():
        r = int(row["grid_row"])
        c = int(row["grid_col"])
        plugs = float(row["Number_of_plugs"])
        power_kw = float(row["power_kw"])

        station_count[r, c] += 1
        plug_count[r, c] += plugs
        total_power_kw[r, c] += power_kw * plugs

        if row["speed_class"] == "fast":
            fast_charger_count[r, c] += plugs
        elif row["speed_class"] == "slow":
            slow_charger_count[r, c] += plugs
        elif row["speed_class"] == "upcoming":
            upcoming_charger_count[r, c] += plugs

    layers = {
        "ev_station_count": station_count,
        "ev_plug_count": plug_count,
        "ev_slow_charger_count": slow_charger_count,
        "ev_fast_charger_count": fast_charger_count,
        "ev_upcoming_charger_count": upcoming_charger_count,
        "ev_total_power_kw": total_power_kw,
    }

    print("\nLayer summary:")
    for name, grid in layers.items():
        print(f"{name}: sum={grid.sum():.2f}, active_cells={(grid > 0).sum()}")

    # =========================
    # Save cleaned point data
    # =========================

    cleaned_path = os.path.join(OUT_DIR, "ev_chargers_cleaned.csv")
    df.to_csv(cleaned_path, index=False)
    print("\nSaved:", cleaned_path)

    # =========================
    # Save single layers
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
    # Save combined tensor
    # =========================

    tensor = np.stack(stacked, axis=0)

    npz_path = os.path.join(OUT_DIR, "ev_charger_layers_256.npz")

    np.savez_compressed(
        npz_path,
        X_ev=tensor,
        layers=np.array(list(layers.keys())),
        bounds=np.array(SYDNEY_BOUNDS),
        grid_size=np.array([GRID_H, GRID_W])
    )

    print("Saved:", npz_path)

    # =========================
    # Save flat grid table
    # =========================

    flat_data = {}

    for name, grid in layers.items():
        flat_data[name] = grid.reshape(-1)

    flat_df = pd.DataFrame(flat_data)
    flat_df["grid_id"] = np.arange(GRID_H * GRID_W)
    flat_df["row"] = flat_df["grid_id"] // GRID_W
    flat_df["col"] = flat_df["grid_id"] % GRID_W

    flat_path = os.path.join(OUT_DIR, "ev_charger_layers_256_flat.csv")
    flat_df.to_csv(flat_path, index=False)

    print("Saved:", flat_path)

    print("\nEV charger preprocessing complete.")
    print("Final EV tensor shape:", tensor.shape)


if __name__ == "__main__":
    build_ev_charger_layers()