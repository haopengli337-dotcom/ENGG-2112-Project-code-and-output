# preprocessing/build_static_tensor.py

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
OUT_DIR = DATASETS_DIR

GRID_H = 256
GRID_W = 256

INPUT_FILES = [
    {
        "path": "city_map_layers_256.npz",
        "array_key": "X_map",
        "prefix": "map",
    },
    {
        "path": "ev_charger_layers_256.npz",
        "array_key": "X_ev",
        "prefix": "ev",
    },
    {
        "path": "traffic_layers_256.npz",
        "array_key": "X_traffic",
        "prefix": "traffic",
    },
    {
        "path": "power_grid_layers_256.npz",
        "array_key": "X_power",
        "prefix": "power",
    },
    {
        "path": "land_layers_256.npz",
        "array_key": "X_land",
        "prefix": "land",
    },
    {
        "path": "vehicle_registration_layers_256.npz",
        "array_key": "X_vehicle",
        "prefix": "vehicle",
    },
]


def normalise_channel(x):
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    mn = x.min()
    mx = x.max()

    if mx - mn < 1e-9:
        return np.zeros_like(x, dtype=np.float32)

    return (x - mn) / (mx - mn)


def load_npz_layer_group(config):
    path = os.path.join(DATASETS_DIR, config["path"])
    array_key = config["array_key"]
    prefix = config["prefix"]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input file: {path}")

    data = np.load(path, allow_pickle=True)

    if array_key not in data:
        raise KeyError(f"{array_key} not found in {path}")

    X = data[array_key].astype(np.float32)

    if X.ndim != 3:
        raise ValueError(f"{path} should have shape (C, H, W), got {X.shape}")

    if X.shape[1] != GRID_H or X.shape[2] != GRID_W:
        raise ValueError(
            f"{path} has wrong grid size {X.shape[1:]}, expected {(GRID_H, GRID_W)}"
        )

    if "layers" in data:
        raw_names = [str(x) for x in data["layers"]]
    else:
        raw_names = [f"layer_{i}" for i in range(X.shape[0])]

    layer_names = [f"{prefix}_{name}" for name in raw_names]

    return X, layer_names, path


def build_static_tensor():
    print("Building unified static tensor...")
    print("Datasets folder:", DATASETS_DIR)

    all_arrays = []
    all_layer_names = []
    summary_rows = []

    for config in INPUT_FILES:
        X, names, path = load_npz_layer_group(config)

        print(f"\nLoaded {os.path.basename(path)}")
        print("Shape:", X.shape)

        for i, name in enumerate(names):
            channel = X[i]

            channel_norm = normalise_channel(channel)

            all_arrays.append(channel_norm)
            all_layer_names.append(name)

            summary_rows.append(
                {
                    "layer_index": len(all_layer_names) - 1,
                    "layer_name": name,
                    "source_file": os.path.basename(path),
                    "raw_min": float(np.nanmin(channel)),
                    "raw_max": float(np.nanmax(channel)),
                    "raw_mean": float(np.nanmean(channel)),
                    "norm_min": float(np.nanmin(channel_norm)),
                    "norm_max": float(np.nanmax(channel_norm)),
                    "norm_mean": float(np.nanmean(channel_norm)),
                    "active_cells": int(np.sum(channel_norm > 0)),
                }
            )

    X_static = np.stack(all_arrays, axis=0).astype(np.float32)

    print("\nUnified static tensor complete.")
    print("Final shape:", X_static.shape)

    expected_layers = sum([np.load(os.path.join(DATASETS_DIR, c["path"]), allow_pickle=True)[c["array_key"]].shape[0] for c in INPUT_FILES])
    print("Expected layers:", expected_layers)
    print("Actual layers:", X_static.shape[0])

    if X_static.shape[0] != expected_layers:
        raise ValueError("Layer count mismatch.")

    # =========================
    # Save npz
    # =========================

    out_npz = os.path.join(OUT_DIR, "model_static_layers_256.npz")

    np.savez_compressed(
        out_npz,
        X_static=X_static,
        layers=np.array(all_layer_names),
        grid_size=np.array([GRID_H, GRID_W]),
    )

    print("Saved:", out_npz)

    # =========================
    # Save layer summary
    # =========================

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(OUT_DIR, "model_static_layers_256_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("Saved:", summary_path)

    # =========================
    # Save flat version
    # =========================

    flat_data = {}

    for i, name in enumerate(all_layer_names):
        flat_data[name] = X_static[i].reshape(-1)

    flat_df = pd.DataFrame(flat_data)
    flat_df["grid_id"] = np.arange(GRID_H * GRID_W)
    flat_df["row"] = flat_df["grid_id"] // GRID_W
    flat_df["col"] = flat_df["grid_id"] % GRID_W

    flat_path = os.path.join(OUT_DIR, "model_static_layers_256_flat.csv")
    flat_df.to_csv(flat_path, index=False)

    print("Saved:", flat_path)

    print("\nFinal layer list:")
    for i, name in enumerate(all_layer_names):
        print(f"{i:02d}: {name}")


if __name__ == "__main__":
    build_static_tensor()