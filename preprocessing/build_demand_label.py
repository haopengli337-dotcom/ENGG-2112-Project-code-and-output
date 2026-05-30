# preprocessing/build_demand_label.py

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

GRID_H = 256
GRID_W = 256

STATIC_FILE = os.path.join(DATASETS_DIR, "model_static_layers_256.npz")


def normalise(x):
    x = np.nan_to_num(
        x.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    mn = x.min()
    mx = x.max()

    if mx - mn < 1e-9:
        return np.zeros_like(x, dtype=np.float32)

    return (x - mn) / (mx - mn)


def get_layer(layer_dict, name):
    if name not in layer_dict:
        print(f"[WARNING] Missing layer: {name}")
        return np.zeros((GRID_H, GRID_W), dtype=np.float32)

    return normalise(layer_dict[name])


def save_grid_csv(name, grid):
    out_path = os.path.join(DATASETS_DIR, f"{name}.csv")
    pd.DataFrame(grid).to_csv(out_path, index=False, header=False)
    print("Saved:", out_path)


def build_demand_label():
    print("Building pseudo EV charging demand label...")

    if not os.path.exists(STATIC_FILE):
        raise FileNotFoundError(
            f"Missing static tensor:\n{STATIC_FILE}\n"
            "Please run preprocessing/build_static_tensor.py first."
        )

    data = np.load(STATIC_FILE, allow_pickle=True)

    X_static = data["X_static"]
    layer_names = [str(x) for x in data["layers"]]

    print("Loaded static tensor:", X_static.shape)
    print("Number of layers:", len(layer_names))

    layer_dict = {
        name: X_static[i]
        for i, name in enumerate(layer_names)
    }

    # ======================================================
    # 1. Traffic demand component
    # ======================================================

    traffic_daily = get_layer(layer_dict, "traffic_traffic_daily_average")
    traffic_morning = get_layer(layer_dict, "traffic_traffic_morning_peak")
    traffic_midday = get_layer(layer_dict, "traffic_traffic_midday")
    traffic_evening = get_layer(layer_dict, "traffic_traffic_evening_peak")
    traffic_night = get_layer(layer_dict, "traffic_traffic_night")

    traffic_component = normalise(
        0.35 * traffic_daily +
        0.25 * traffic_morning +
        0.25 * traffic_evening +
        0.10 * traffic_midday +
        0.05 * traffic_night
    )

    # ======================================================
    # 2. EV ownership / vehicle component
    # ======================================================

    ev_adoption = get_layer(layer_dict, "vehicle_ev_adoption_proxy")
    vehicle_density = get_layer(layer_dict, "vehicle_vehicle_registration_density")
    private_vehicle_density = get_layer(layer_dict, "vehicle_private_vehicle_density")
    vehicle_growth = get_layer(layer_dict, "vehicle_vehicle_growth_proxy")

    ev_component = normalise(
        0.40 * ev_adoption +
        0.25 * vehicle_density +
        0.20 * private_vehicle_density +
        0.15 * vehicle_growth
    )

    # ======================================================
    # 3. Land use / activity component
    # ======================================================

    commercial = get_layer(layer_dict, "map_commercial")
    residential = get_layer(layer_dict, "map_residential")
    industrial = get_layer(layer_dict, "map_industrial")
    buildings = get_layer(layer_dict, "map_buildings")
    primary_roads = get_layer(layer_dict, "map_primary_roads")
    roads = get_layer(layer_dict, "map_roads")
    residential_roads = get_layer(layer_dict, "map_residential_roads")

    landuse_component = normalise(
        0.30 * commercial +
        0.25 * residential +
        0.15 * industrial +
        0.15 * buildings +
        0.10 * primary_roads +
        0.05 * residential_roads
    )

    # ======================================================
    # 4. Accessibility component
    # ======================================================

    parking_proxy = get_layer(layer_dict, "land_parking_land_proxy")
    construction_suitability = get_layer(layer_dict, "land_construction_suitability")
    feasible_build_area = get_layer(layer_dict, "land_feasible_build_area")

    accessibility_component = normalise(
        0.35 * primary_roads +
        0.25 * roads +
        0.20 * parking_proxy +
        0.15 * construction_suitability +
        0.05 * feasible_build_area
    )

    # ======================================================
    # 5. Power grid component
    # ======================================================

    grid_capacity = get_layer(layer_dict, "power_grid_capacity_proxy")
    grid_response_capacity = get_layer(layer_dict, "power_grid_response_capacity")
    grid_overload_risk = get_layer(layer_dict, "power_grid_overload_risk")
    grid_peak_risk = get_layer(layer_dict, "power_grid_peak_event_risk")

    grid_component = normalise(
        0.45 * grid_capacity +
        0.25 * grid_response_capacity +
        0.20 * (1.0 - grid_overload_risk) +
        0.10 * (1.0 - grid_peak_risk)
    )

    # ======================================================
    # 6. Existing charger supply gap
    # ======================================================

    existing_station = get_layer(layer_dict, "ev_ev_station_count")
    existing_plug = get_layer(layer_dict, "ev_ev_plug_count")
    existing_fast = get_layer(layer_dict, "ev_ev_fast_charger_count")
    existing_slow = get_layer(layer_dict, "ev_ev_slow_charger_count")
    existing_power = get_layer(layer_dict, "ev_ev_total_power_kw")

    existing_supply = normalise(
        0.20 * existing_station +
        0.25 * existing_plug +
        0.25 * existing_fast +
        0.10 * existing_slow +
        0.20 * existing_power
    )

    charger_gap_component = normalise(1.0 - existing_supply)

    # ======================================================
    # 7. Constraint mask
    # ======================================================

    no_build = get_layer(layer_dict, "land_no_build_zone")
    water = get_layer(layer_dict, "map_water")
    parks = get_layer(layer_dict, "map_parks")
    land_availability = get_layer(layer_dict, "land_land_availability")

    constraint_mask = np.ones((GRID_H, GRID_W), dtype=np.float32)

    constraint_mask[no_build > 0.5] = 0.0
    constraint_mask[water > 0.5] = 0.0
    constraint_mask[parks > 0.5] = 0.0
    constraint_mask[feasible_build_area <= 0.0] = 0.0
    constraint_mask[land_availability <= 0.05] = 0.0

    # ======================================================
    # 8. Final demand score
    # ======================================================

    demand_score_raw = (
        0.24 * traffic_component +
        0.20 * ev_component +
        0.18 * landuse_component +
        0.15 * accessibility_component +
        0.13 * charger_gap_component +
        0.10 * grid_component
    )

    demand_score = normalise(demand_score_raw)
    demand_score = demand_score * constraint_mask
    demand_score = normalise(demand_score)

    # ======================================================
    # 9. Demand class label
    # ======================================================
    # 0 = low / unsuitable
    # 1 = medium
    # 2 = high

    demand_class = np.zeros((GRID_H, GRID_W), dtype=np.int64)
    demand_class[demand_score >= 0.33] = 1
    demand_class[demand_score >= 0.66] = 2

    # ======================================================
    # 10. Save components
    # ======================================================

    components = {
        "demand_score": demand_score,
        "traffic_component": traffic_component,
        "ev_component": ev_component,
        "landuse_component": landuse_component,
        "accessibility_component": accessibility_component,
        "grid_component": grid_component,
        "charger_gap_component": charger_gap_component,
        "constraint_mask": constraint_mask,
    }

    print("\nDemand label summary:")
    print("demand_score min:", float(demand_score.min()))
    print("demand_score max:", float(demand_score.max()))
    print("demand_score mean:", float(demand_score.mean()))
    print("active demand cells:", int((demand_score > 0).sum()))
    print("high demand cells:", int((demand_class == 2).sum()))
    print("medium demand cells:", int((demand_class == 1).sum()))
    print("low demand cells:", int((demand_class == 0).sum()))

    for name, grid in components.items():
        save_grid_csv(name, grid)

    save_grid_csv("demand_class", demand_class)

    out_npz = os.path.join(DATASETS_DIR, "demand_label_256.npz")

    np.savez_compressed(
        out_npz,
        y_demand=demand_score.astype(np.float32),
        y_class=demand_class.astype(np.int64),
        components=np.stack(list(components.values()), axis=0).astype(np.float32),
        component_names=np.array(list(components.keys())),
        grid_size=np.array([GRID_H, GRID_W]),
    )

    print("Saved:", out_npz)

    # ======================================================
    # 11. Save flat version
    # ======================================================

    flat_data = {}

    for name, grid in components.items():
        flat_data[name] = grid.reshape(-1)

    flat_data["demand_class"] = demand_class.reshape(-1)

    flat_df = pd.DataFrame(flat_data)

    flat_df["grid_id"] = np.arange(GRID_H * GRID_W)
    flat_df["row"] = flat_df["grid_id"] // GRID_W
    flat_df["col"] = flat_df["grid_id"] % GRID_W

    flat_path = os.path.join(DATASETS_DIR, "demand_label_256_flat.csv")
    flat_df.to_csv(flat_path, index=False)

    print("Saved:", flat_path)

    print("\nDemand label build complete.")
    print("y_demand shape:", demand_score.shape)
    print("y_class shape:", demand_class.shape)


if __name__ == "__main__":
    build_demand_label()