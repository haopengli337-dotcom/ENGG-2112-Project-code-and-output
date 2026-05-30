# =========================
# preprocessing/power_grid.py
# Build power grid proxy, capacity, overload risk layers
# =========================

import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "R_data", "Power Grid")
OUT_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(OUT_DIR, exist_ok=True)

GRID_H = 256
GRID_W = 256
SYDNEY_BOUNDS = [150.50, -34.25, 151.45, -33.35]

EV_FILE = os.path.join(RAW_DIR, "sgsc-ev-electric-vehicles-data.csv")
PEAK_EVENTS_FILE = os.path.join(RAW_DIR, "sgsc-ctpeak-events.csv")
PEAK_RESPONSE_FILE = os.path.join(RAW_DIR, "sgsc-ctpeak-event-response.csv")

EV_LAYER_FILE = os.path.join(OUT_DIR, "ev_charger_layers_256.npz")
TRAFFIC_LAYER_FILE = os.path.join(OUT_DIR, "traffic_layers_256.npz")
VEHICLE_LAYER_FILE = os.path.join(OUT_DIR, "vehicle_registration_layers_256.npz")


# =========================
# Basic utilities
# =========================

def safe_read_csv(path):
    if not os.path.exists(path):
        print("[WARNING] Missing file:", path)
        return pd.DataFrame()

    print("Reading:", path)

    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print("[WARNING] Standard read failed:", e)
        try:
            return pd.read_csv(
                path,
                low_memory=False,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as e2:
            print("[WARNING] Python-engine read also failed:", e2)
            return pd.DataFrame()


def robust_normalise(arr, p_low=1, p_high=99):
    arr = np.nan_to_num(
        arr.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if np.all(arr == arr.flat[0]):
        return np.zeros_like(arr, dtype=np.float32)

    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)

    if hi - lo < 1e-9:
        mn = arr.min()
        mx = arr.max()

        if mx - mn < 1e-9:
            return np.zeros_like(arr, dtype=np.float32)

        out = (arr - mn) / (mx - mn)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    arr = np.clip(arr, lo, hi)
    out = (arr - lo) / (hi - lo)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def smooth_layer(arr, sigma=3):
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


def get_npz_layer(npz_path, data_keys, layer_candidates):
    if not os.path.exists(npz_path):
        print("[WARNING] Missing layer file:", npz_path)
        return np.zeros((GRID_H, GRID_W), dtype=np.float32)

    data = np.load(npz_path, allow_pickle=True)

    data_key = None
    for key in data_keys:
        if key in data.files:
            data_key = key
            break

    if data_key is None:
        print("[WARNING] No valid tensor key in:", npz_path)
        print("Available keys:", data.files)
        return np.zeros((GRID_H, GRID_W), dtype=np.float32)

    X = data[data_key]

    if "layers" not in data.files:
        print("[WARNING] No layer names in:", npz_path)
        return robust_normalise(X[0])

    names = [str(x) for x in list(data["layers"])]

    for target in layer_candidates:
        if target in names:
            idx = names.index(target)
            print(f"Loaded layer: {target} from {os.path.basename(npz_path)}")
            return robust_normalise(X[idx])

    print("[WARNING] No candidate layer found in:", npz_path)
    print("Candidates:", layer_candidates)
    print("Available:", names)

    return robust_normalise(X[0])


# =========================
# Load existing spatial proxies
# =========================

def load_existing_proxy_layers():
    ev_power_proxy = get_npz_layer(
        EV_LAYER_FILE,
        data_keys=["X_ev", "X", "arr_0"],
        layer_candidates=[
            "ev_total_power_kw",
            "ev_plug_count",
            "ev_fast_charger_count",
            "ev_slow_charger_count",
        ],
    )

    traffic_proxy = get_npz_layer(
        TRAFFIC_LAYER_FILE,
        data_keys=["X_traffic", "X", "arr_0"],
        layer_candidates=[
            "traffic_daily_average",
            "traffic_volume",
            "traffic_density",
            "daily_average",
        ],
    )

    ev_adoption_proxy = get_npz_layer(
        VEHICLE_LAYER_FILE,
        data_keys=["X_vehicle", "X", "arr_0"],
        layer_candidates=[
            "ev_adoption_proxy",
            "vehicle_registration_density",
            "vehicle_growth_proxy",
            "private_vehicle_density",
        ],
    )

    ev_power_proxy = smooth_layer(ev_power_proxy, sigma=2.5)
    traffic_proxy = smooth_layer(traffic_proxy, sigma=4.0)
    ev_adoption_proxy = smooth_layer(ev_adoption_proxy, sigma=3.0)

    if ev_power_proxy.std() < 1e-6:
        print("[WARNING] EV charger power proxy inactive. Using traffic fallback.")
        ev_power_proxy = traffic_proxy.copy()

    if ev_adoption_proxy.std() < 1e-6:
        print("[WARNING] EV adoption proxy inactive. Using traffic fallback.")
        ev_adoption_proxy = traffic_proxy.copy()

    demand_pressure = robust_normalise(
        0.45 * traffic_proxy +
        0.35 * ev_adoption_proxy +
        0.20 * ev_power_proxy
    )

    demand_pressure = smooth_layer(demand_pressure, sigma=3.0)

    print("\nInput proxy statistics:")
    print_stats("ev_power_proxy", ev_power_proxy)
    print_stats("traffic_proxy", traffic_proxy)
    print_stats("ev_adoption_proxy", ev_adoption_proxy)
    print_stats("demand_pressure", demand_pressure)

    return ev_power_proxy, traffic_proxy, ev_adoption_proxy, demand_pressure


def extract_numeric_summary(df):
    if df.empty:
        return {
            "row_count": 0,
            "numeric_mean": 0.0,
            "numeric_max": 0.0,
            "numeric_sum": 0.0,
        }

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return {
            "row_count": len(df),
            "numeric_mean": 0.0,
            "numeric_max": 0.0,
            "numeric_sum": 0.0,
        }

    return {
        "row_count": len(df),
        "numeric_mean": float(numeric_df.mean().mean()),
        "numeric_max": float(numeric_df.max().max()),
        "numeric_sum": float(numeric_df.sum().sum()),
    }


# =========================
# Main builder
# =========================

def build_power_grid_layers():
    print("Building power grid proxy layers...")
    print("Input folder:", RAW_DIR)

    ev_df = safe_read_csv(EV_FILE)
    events_df = safe_read_csv(PEAK_EVENTS_FILE)
    response_df = safe_read_csv(PEAK_RESPONSE_FILE)

    print("\nRaw data shapes:")
    print("EV grid data:", ev_df.shape)
    print("Peak events:", events_df.shape)
    print("Peak response:", response_df.shape)

    ev_summary = extract_numeric_summary(ev_df)
    events_summary = extract_numeric_summary(events_df)
    response_summary = extract_numeric_summary(response_df)

    summary_df = pd.DataFrame([
        {"dataset": "sgsc_ev_electric_vehicles", **ev_summary},
        {"dataset": "sgsc_ctpeak_events", **events_summary},
        {"dataset": "sgsc_ctpeak_event_response", **response_summary},
    ])

    summary_path = os.path.join(OUT_DIR, "power_grid_raw_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("Saved:", summary_path)

    ev_power_proxy, traffic_proxy, ev_adoption_proxy, demand_pressure = (
        load_existing_proxy_layers()
    )

    peak_event_intensity = float(np.clip(events_summary["row_count"] / 100.0, 0.0, 1.0))
    response_intensity = float(np.clip(response_summary["row_count"] / 100.0, 0.0, 1.0))
    ev_grid_intensity = float(np.clip(ev_summary["row_count"] / 1000.0, 0.0, 1.0))

    # =========================
    # Pressure layers
    # =========================

    grid_ev_pressure = robust_normalise(
        ev_power_proxy * (0.75 + 0.25 * ev_grid_intensity)
    )

    grid_traffic_pressure = robust_normalise(
        traffic_proxy * (0.75 + 0.25 * peak_event_intensity)
    )

    grid_adoption_pressure = robust_normalise(
        ev_adoption_proxy * (0.80 + 0.20 * ev_grid_intensity)
    )

    grid_peak_event_risk = robust_normalise(
        0.65 * demand_pressure +
        0.25 * grid_traffic_pressure +
        0.10 * grid_ev_pressure
    )

    grid_peak_event_risk = smooth_layer(grid_peak_event_risk, sigma=2.0)

    load_pressure = robust_normalise(
        0.35 * grid_ev_pressure +
        0.30 * grid_traffic_pressure +
        0.25 * grid_adoption_pressure +
        0.10 * grid_peak_event_risk
    )

    load_pressure = smooth_layer(load_pressure, sigma=2.5)

    # =========================
    # Capacity and risk
    # Key change:
    # Do NOT allow capacity to become almost 1 everywhere.
    # Capacity is now a heterogeneous proxy.
    # =========================

    base_capacity = robust_normalise(
        0.50 * (1.0 - load_pressure) +
        0.25 * (1.0 - grid_peak_event_risk) +
        0.25 * (1.0 - demand_pressure)
    )

    # Keep capacity realistic: most cells moderate, not all perfect.
    grid_capacity_proxy = np.clip(
        0.25 + 0.65 * base_capacity,
        0.10,
        0.90
    ).astype(np.float32)

    grid_response_capacity = robust_normalise(
        0.65 * grid_capacity_proxy +
        0.35 * (1.0 - grid_peak_event_risk) * (0.5 + 0.5 * response_intensity)
    )

    # Actual risk should increase with demand/load and decrease with capacity.
    grid_overload_risk = robust_normalise(
        0.45 * load_pressure +
        0.30 * demand_pressure +
        0.25 * (1.0 - grid_capacity_proxy)
    )

    grid_overload_risk = smooth_layer(grid_overload_risk, sigma=2.0)

    grid_augmentation_cost_proxy = robust_normalise(
        0.45 * grid_overload_risk +
        0.35 * (1.0 - grid_capacity_proxy) +
        0.20 * demand_pressure
    )

    grid_augmentation_cost_proxy = smooth_layer(
        grid_augmentation_cost_proxy,
        sigma=2.0
    )

    # MW capacity used by constraints/grid.py or optimiser.
    # This is deliberately not almost uniform.
    remaining_capacity_mw = (
        0.30 + 2.20 * grid_capacity_proxy
    ).astype(np.float32)

    layers = {
        "grid_ev_pressure": grid_ev_pressure,
        "grid_traffic_pressure": grid_traffic_pressure,
        "grid_adoption_pressure": grid_adoption_pressure,
        "grid_peak_event_risk": grid_peak_event_risk,
        "grid_response_capacity": grid_response_capacity,
        "grid_capacity_proxy": grid_capacity_proxy,
        "remaining_capacity_mw": remaining_capacity_mw,
        "grid_overload_risk": grid_overload_risk,
        "grid_augmentation_cost_proxy": grid_augmentation_cost_proxy,
        "power_demand_pressure": demand_pressure,
    }

    print("\nLayer summary:")
    for name, grid in layers.items():
        print_stats(name, grid)

    # =========================
    # Save CSV layers
    # =========================

    stacked = []

    for name, grid in layers.items():
        grid = np.nan_to_num(grid.astype(np.float32), nan=0.0)

        out_path = os.path.join(OUT_DIR, f"{name}.csv")
        pd.DataFrame(grid).to_csv(out_path, index=False, header=False)

        stacked.append(grid)
        print("Saved:", out_path)

    # =========================
    # Save NPZ tensor
    # =========================

    tensor = np.stack(stacked, axis=0).astype(np.float32)

    npz_path = os.path.join(OUT_DIR, "power_grid_layers_256.npz")

    np.savez_compressed(
        npz_path,
        X_power=tensor,
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

    flat_path = os.path.join(OUT_DIR, "power_grid_layers_256_flat.csv")
    flat_df.to_csv(flat_path, index=False)

    print("Saved:", flat_path)

    print("\nPower grid preprocessing complete.")
    print("Final power grid tensor shape:", tensor.shape)


if __name__ == "__main__":
    build_power_grid_layers()