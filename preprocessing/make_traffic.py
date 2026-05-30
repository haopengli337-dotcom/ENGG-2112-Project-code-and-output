# preprocessing/make_traffic.py

import os
import glob
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAFFIC_BASE_DIR = os.path.join(
    BASE_DIR,
    "R_data",
    "Traffic Flow Data"
)

HOURLY_DIR = os.path.join(
    TRAFFIC_BASE_DIR,
    "road_traffic_counts_hourly_permanent"
)

OUT_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(OUT_DIR, exist_ok=True)

GRID_H = 256
GRID_W = 256

SYDNEY_BOUNDS = [150.50, -34.25, 151.45, -33.35]


def coord_to_grid(lon, lat, bounds):
    minx, miny, maxx, maxy = bounds

    col = int((lon - minx) / (maxx - minx + 1e-9) * (GRID_W - 1))
    row = int((maxy - lat) / (maxy - miny + 1e-9) * (GRID_H - 1))

    row = max(0, min(GRID_H - 1, row))
    col = max(0, min(GRID_W - 1, col))

    return row, col


def find_file(keyword):
    pattern = os.path.join(TRAFFIC_BASE_DIR, f"*{keyword}*.csv")
    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise FileNotFoundError(f"No file found for keyword: {keyword}")

    return matches[0]


def load_hourly_files():
    files = glob.glob(
        os.path.join(
            HOURLY_DIR,
            "road_traffic_counts_hourly_permanent*.csv"
        )
    )

    sample_file = os.path.join(
        TRAFFIC_BASE_DIR,
        "road_traffic_counts_hourly_sample_0.csv"
    )

    if os.path.exists(sample_file):
        files.append(sample_file)

    files = sorted(list(set(files)))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No hourly traffic files found in:\n{HOURLY_DIR}"
        )

    return files


def normalise_grid(grid):
    grid = np.nan_to_num(grid.astype(np.float32), nan=0.0)
    mx = np.nanmax(grid)

    if mx <= 0:
        return grid

    return grid / mx


def smooth_grid(grid, sigma=3):
    smoothed = gaussian_filter(
        grid.astype(np.float32),
        sigma=sigma
    )

    return normalise_grid(smoothed)


def build_traffic_layers():
    print("Reading traffic data...")
    print("Traffic base folder:", TRAFFIC_BASE_DIR)
    print("Hourly folder:", HOURLY_DIR)

    station_file = find_file("station_reference")
    yearly_file = find_file("yearly_summary")
    hourly_files = load_hourly_files()

    print("\nStation reference:", station_file)
    print("Yearly summary:", yearly_file)
    print("Hourly files:")
    for f in hourly_files:
        print(" -", os.path.basename(f))

    stations = pd.read_csv(station_file)

    print("\nStation columns:")
    print(list(stations.columns))

    possible_lat_cols = [
        "wgs84_latitude",
        "latitude",
        "Latitude",
        "lat",
    ]

    possible_lon_cols = [
        "wgs84_longitude",
        "longitude",
        "Longitude",
        "lon",
        "lng",
    ]

    lat_col = None
    lon_col = None

    for col in possible_lat_cols:
        if col in stations.columns:
            lat_col = col
            break

    for col in possible_lon_cols:
        if col in stations.columns:
            lon_col = col
            break

    if lat_col is None or lon_col is None:
        raise ValueError(
            "Cannot find latitude/longitude columns in station reference file."
        )

    if "station_key" not in stations.columns:
        if "station_id" in stations.columns:
            stations["station_key"] = stations["station_id"]
        else:
            raise ValueError("Cannot find station_key or station_id column.")

    stations[lat_col] = pd.to_numeric(
        stations[lat_col],
        errors="coerce"
    )

    stations[lon_col] = pd.to_numeric(
        stations[lon_col],
        errors="coerce"
    )

    stations = stations.dropna(
        subset=[lat_col, lon_col]
    ).copy()

    lon_min, lat_min, lon_max, lat_max = SYDNEY_BOUNDS

    stations = stations[
        (stations[lon_col] >= lon_min)
        & (stations[lon_col] <= lon_max)
        & (stations[lat_col] >= lat_min)
        & (stations[lat_col] <= lat_max)
    ].copy()

    print("Stations after Sydney clipping:", stations.shape)

    rows = []
    cols = []

    for _, row in stations.iterrows():
        r, c = coord_to_grid(
            row[lon_col],
            row[lat_col],
            SYDNEY_BOUNDS,
        )

        rows.append(r)
        cols.append(c)

    stations["grid_row"] = rows
    stations["grid_col"] = cols
    stations["grid_id"] = (
        stations["grid_row"] * GRID_W
        + stations["grid_col"]
    )

    stations["station_key"] = stations["station_key"].astype(str)
    station_keys = set(stations["station_key"])

    hourly_parts = []

    for file in hourly_files:
        print("\nReading:", os.path.basename(file))

        preview = pd.read_csv(file, nrows=5)
        print("Columns:", list(preview.columns))

        hour_cols = []

        for h in range(24):
            candidates = [
                f"hour_{h:02d}",
                f"hour_{h}",
                f"{h:02d}",
                str(h),
            ]

            for c in candidates:
                if c in preview.columns:
                    hour_cols.append(c)
                    break

        daily_col = None

        for c in [
            "daily_total",
            "Daily_Total",
            "total",
            "traffic_volume",
            "volume",
        ]:
            if c in preview.columns:
                daily_col = c
                break

        usecols = ["station_key"]

        if "year" in preview.columns:
            usecols.append("year")

        if daily_col:
            usecols.append(daily_col)

        usecols += hour_cols
        usecols = list(dict.fromkeys(usecols))

        df = pd.read_csv(
            file,
            usecols=lambda c: c in usecols
        )

        if "station_key" not in df.columns:
            print(
                "Skipped because station_key not found:",
                os.path.basename(file)
            )
            continue

        df["station_key"] = df["station_key"].astype(str)
        df = df[df["station_key"].isin(station_keys)].copy()

        if len(df) == 0:
            print("No matching Sydney stations in this file.")
            continue

        if "year" not in df.columns:
            df["year"] = 2024

        for col in df.columns:
            if col not in ["station_key"]:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).fillna(0)

        standard_hour_cols = []

        for h in range(24):
            standard_col = f"hour_{h:02d}"

            candidates = [
                f"hour_{h:02d}",
                f"hour_{h}",
                f"{h:02d}",
                str(h),
            ]

            found = None

            for c in candidates:
                if c in df.columns:
                    found = c
                    break

            if found is not None:
                df[standard_col] = df[found]
            else:
                df[standard_col] = 0

            standard_hour_cols.append(standard_col)

        if daily_col and daily_col in df.columns:
            df["daily_total"] = df[daily_col]
        else:
            df["daily_total"] = df[standard_hour_cols].sum(axis=1)

        keep_cols = (
            ["station_key", "year", "daily_total"]
            + standard_hour_cols
        )

        df = df[keep_cols].copy()
        hourly_parts.append(df)

    if len(hourly_parts) == 0:
        raise ValueError(
            "No hourly records matched Sydney traffic stations."
        )

    hourly = pd.concat(hourly_parts, ignore_index=True)

    print("\nMatched hourly rows:", hourly.shape)

    max_year = int(hourly["year"].max())
    recent_start = max_year - 4

    hourly_recent = hourly[
        hourly["year"] >= recent_start
    ].copy()

    if len(hourly_recent) > 1000:
        hourly = hourly_recent

    print(
        f"Using years: {hourly['year'].min()} "
        f"to {hourly['year'].max()}"
    )
    print("Rows used:", hourly.shape)

    hourly["traffic_daily_average"] = hourly["daily_total"]

    hourly["traffic_morning_peak"] = hourly[
        ["hour_07", "hour_08", "hour_09"]
    ].mean(axis=1)

    hourly["traffic_midday"] = hourly[
        ["hour_11", "hour_12", "hour_13", "hour_14"]
    ].mean(axis=1)

    hourly["traffic_evening_peak"] = hourly[
        ["hour_16", "hour_17", "hour_18"]
    ].mean(axis=1)

    hourly["traffic_night"] = hourly[
        ["hour_00", "hour_01", "hour_02", "hour_03", "hour_04"]
    ].mean(axis=1)

    feature_cols = [
        "traffic_daily_average",
        "traffic_morning_peak",
        "traffic_midday",
        "traffic_evening_peak",
        "traffic_night",
    ]

    traffic_by_station = (
        hourly.groupby("station_key")[feature_cols]
        .mean()
        .reset_index()
    )

    merged = stations.merge(
        traffic_by_station,
        on="station_key",
        how="left"
    )

    for col in feature_cols:
        merged[col] = pd.to_numeric(
            merged[col],
            errors="coerce"
        ).fillna(0)

    layers = {
        "traffic_daily_average": np.zeros(
            (GRID_H, GRID_W),
            dtype=np.float32
        ),
        "traffic_morning_peak": np.zeros(
            (GRID_H, GRID_W),
            dtype=np.float32
        ),
        "traffic_midday": np.zeros(
            (GRID_H, GRID_W),
            dtype=np.float32
        ),
        "traffic_evening_peak": np.zeros(
            (GRID_H, GRID_W),
            dtype=np.float32
        ),
        "traffic_night": np.zeros(
            (GRID_H, GRID_W),
            dtype=np.float32
        ),
        "traffic_station_count": np.zeros(
            (GRID_H, GRID_W),
            dtype=np.float32
        ),
    }

    for _, row in merged.iterrows():
        r = int(row["grid_row"])
        c = int(row["grid_col"])

        layers["traffic_station_count"][r, c] += 1

        for col in feature_cols:
            layers[col][r, c] += float(row[col])

    count = layers["traffic_station_count"]

    for col in feature_cols:
        mask = count > 0
        layers[col][mask] = layers[col][mask] / count[mask]

    norm_layers = {}

    for name, grid in layers.items():
        if name == "traffic_station_count":
            norm_layers[name] = grid
            norm_layers["traffic_station_density"] = smooth_grid(
                grid,
                sigma=2
            )
        else:
            raw_norm = normalise_grid(grid)

            if name == "traffic_daily_average":
                norm_layers[name] = smooth_grid(
                    raw_norm,
                    sigma=4
                )
            elif name in [
                "traffic_morning_peak",
                "traffic_midday",
                "traffic_evening_peak",
            ]:
                norm_layers[name] = smooth_grid(
                    raw_norm,
                    sigma=3
                )
            elif name == "traffic_night":
                norm_layers[name] = smooth_grid(
                    raw_norm,
                    sigma=2
                )
            else:
                norm_layers[name] = raw_norm

    print("\nLayer summary:")
    for name, grid in norm_layers.items():
        print(
            f"{name}: sum={grid.sum():.2f}, "
            f"active_cells={(grid > 0).sum()}"
        )

    cleaned_path = os.path.join(
        OUT_DIR,
        "traffic_stations_cleaned.csv"
    )

    merged.to_csv(cleaned_path, index=False)
    print("\nSaved:", cleaned_path)

    stacked = []

    for name, grid in norm_layers.items():
        out_path = os.path.join(OUT_DIR, f"{name}.csv")

        pd.DataFrame(grid).to_csv(
            out_path,
            index=False,
            header=False,
        )

        stacked.append(grid)

        print("Saved:", out_path)

    tensor = np.stack(stacked, axis=0)

    npz_path = os.path.join(
        OUT_DIR,
        "traffic_layers_256.npz"
    )

    np.savez_compressed(
        npz_path,
        X_traffic=tensor,
        layers=np.array(list(norm_layers.keys())),
        bounds=np.array(SYDNEY_BOUNDS),
        grid_size=np.array([GRID_H, GRID_W]),
    )

    print("Saved:", npz_path)

    flat_data = {}

    for name, grid in norm_layers.items():
        flat_data[name] = grid.reshape(-1)

    flat_df = pd.DataFrame(flat_data)
    flat_df["grid_id"] = np.arange(GRID_H * GRID_W)
    flat_df["row"] = flat_df["grid_id"] // GRID_W
    flat_df["col"] = flat_df["grid_id"] % GRID_W

    flat_path = os.path.join(
        OUT_DIR,
        "traffic_layers_256_flat.csv"
    )

    flat_df.to_csv(flat_path, index=False)

    print("Saved:", flat_path)

    print("\nTraffic preprocessing complete.")
    print("Final traffic tensor shape:", tensor.shape)


if __name__ == "__main__":
    build_traffic_layers()