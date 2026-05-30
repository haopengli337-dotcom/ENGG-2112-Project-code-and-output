# optimizer/nsga2_optimizer.py

import os
import random
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

GRID_H = 256
GRID_W = 256

PRED_FILE = os.path.join(RESULTS_DIR, "cnn_predicted_demand.npz")
STATIC_FILE = os.path.join(DATASETS_DIR, "model_static_layers_256.npz")

RANDOM_SEED = 42

N_STATIONS = 20
POP_SIZE = 50
N_GENERATIONS = 60
CANDIDATE_POOL_SIZE = 500

FAST_CHARGER_POWER = 150.0
SLOW_CHARGER_POWER = 22.0

FAST_CHARGER_COST = 120000
SLOW_CHARGER_COST = 25000
BASE_STATION_COST = 180000

MIN_DISTANCE_CELLS = 8
SOFT_DISTANCE_CELLS = 15
SAFETY_MARGIN = 0.85


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def normalise(x):
    x = np.nan_to_num(
        x.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    mn = x.min()
    mx = x.max()

    if mx - mn < 1e-9:
        return np.zeros_like(x)

    return (x - mn) / (mx - mn)


def load_static_layers():
    data = np.load(STATIC_FILE, allow_pickle=True)
    X_static = data["X_static"]
    layers = [str(x) for x in data["layers"]]

    return {
        name: X_static[i]
        for i, name in enumerate(layers)
    }


def get_layer(layer_dict, name, default_value=0.0, normalise_layer=True):
    if name not in layer_dict:
        print(f"[WARNING] Missing layer: {name}")
        return np.ones((GRID_H, GRID_W), dtype=np.float32) * default_value

    arr = layer_dict[name].astype(np.float32)

    if normalise_layer:
        return normalise(arr)

    return np.nan_to_num(arr, nan=default_value)


def load_inputs():
    if not os.path.exists(PRED_FILE):
        raise FileNotFoundError(
            f"Missing prediction file:\n{PRED_FILE}\n"
            "Run training/train_cnn_baseline.py first."
        )

    pred_data = np.load(PRED_FILE)
    demand = normalise(pred_data["pred_demand"])

    layer_dict = load_static_layers()

    land_cost = get_layer(layer_dict, "land_land_cost_proxy")
    suitability = get_layer(layer_dict, "land_construction_suitability")
    feasible = get_layer(layer_dict, "land_feasible_build_area")
    no_build = get_layer(layer_dict, "land_no_build_zone")

    grid_capacity_proxy = get_layer(layer_dict, "power_grid_capacity_proxy")
    grid_risk = get_layer(layer_dict, "power_grid_overload_risk")

    remaining_capacity_mw = get_layer(
        layer_dict,
        "power_remaining_capacity_mw",
        default_value=2.0,
        normalise_layer=False,
    )

    if np.max(remaining_capacity_mw) <= 1.01:
        remaining_capacity_mw = 0.4 + 2.6 * normalise(remaining_capacity_mw)

    existing_fast = get_layer(layer_dict, "ev_ev_fast_charger_count")
    existing_slow = get_layer(layer_dict, "ev_ev_slow_charger_count")
    existing_supply = normalise(existing_fast + existing_slow)

    return {
        "demand": demand,
        "land_cost": land_cost,
        "suitability": suitability,
        "feasible": feasible,
        "no_build": no_build,
        "grid_capacity": grid_capacity_proxy,
        "remaining_capacity_mw": remaining_capacity_mw,
        "grid_risk": grid_risk,
        "existing_supply": existing_supply,
    }


def station_power_kw(fast, slow):
    return fast * FAST_CHARGER_POWER + slow * SLOW_CHARGER_POWER


def build_candidate_pool(inputs):
    demand = inputs["demand"]
    suitability = inputs["suitability"]
    feasible = inputs["feasible"]
    no_build = inputs["no_build"]
    grid_capacity = inputs["grid_capacity"]
    grid_risk = inputs["grid_risk"]
    existing_supply = inputs["existing_supply"]

    score = (
        0.45 * demand
        + 0.20 * suitability
        + 0.12 * grid_capacity
        + 0.15 * (1.0 - grid_risk)
        + 0.08 * (1.0 - existing_supply)
    )

    score[feasible <= 0] = 0
    score[no_build > 0] = 0

    flat = score.reshape(-1)
    candidate_ids = np.argsort(flat)[-CANDIDATE_POOL_SIZE:][::-1]

    candidates = []

    for gid in candidate_ids:
        r = gid // GRID_W
        c = gid % GRID_W

        if flat[gid] <= 0:
            continue

        candidates.append(
            {
                "grid_id": int(gid),
                "row": int(r),
                "col": int(c),
                "candidate_score": float(flat[gid]),
                "demand": float(demand[r, c]),
                "suitability": float(suitability[r, c]),
                "grid_capacity": float(grid_capacity[r, c]),
                "remaining_capacity_mw": float(inputs["remaining_capacity_mw"][r, c]),
                "grid_risk": float(grid_risk[r, c]),
                "existing_supply": float(existing_supply[r, c]),
            }
        )

    candidate_df = pd.DataFrame(candidates)

    out_path = os.path.join(RESULTS_DIR, "candidate_station_cells.csv")
    candidate_df.to_csv(out_path, index=False)
    print("Saved:", out_path)

    return candidate_df


def cell_distance(candidate_df, idx_a, idx_b):
    ra = candidate_df.iloc[idx_a]["row"]
    ca = candidate_df.iloc[idx_a]["col"]
    rb = candidate_df.iloc[idx_b]["row"]
    cb = candidate_df.iloc[idx_b]["col"]

    return np.sqrt((ra - rb) ** 2 + (ca - cb) ** 2)


def distance_ok(selected_indices, candidate_df, new_idx):
    for idx in selected_indices:
        if cell_distance(candidate_df, idx, new_idx) < MIN_DISTANCE_CELLS:
            return False

    return True


def spacing_penalty(individual, candidate_df):
    penalty = 0.0
    seen = set()

    indices = [gene[0] for gene in individual]

    for idx in indices:
        if idx in seen:
            penalty += 5.0
        seen.add(idx)

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            d = cell_distance(candidate_df, indices[i], indices[j])

            if d < MIN_DISTANCE_CELLS:
                penalty += 3.0 * (MIN_DISTANCE_CELLS - d) / MIN_DISTANCE_CELLS
            elif d < SOFT_DISTANCE_CELLS:
                penalty += 0.5 * (SOFT_DISTANCE_CELLS - d) / SOFT_DISTANCE_CELLS

    return penalty


def create_individual(candidate_df):
    selected = []
    attempts = 0

    while len(selected) < N_STATIONS and attempts < 5000:
        idx = random.randint(0, len(candidate_df) - 1)

        if idx in selected:
            attempts += 1
            continue

        if not distance_ok(selected, candidate_df, idx):
            attempts += 1
            continue

        selected.append(idx)
        attempts += 1

    while len(selected) < N_STATIONS:
        idx = random.randint(0, len(candidate_df) - 1)
        if idx not in selected:
            selected.append(idx)

    individual = []

    for idx in selected:
        fast = random.randint(1, 6)
        slow = random.randint(2, 12)
        individual.append([idx, fast, slow])

    return individual


def repair_individual(individual, candidate_df):
    repaired = []
    selected = []

    for gene in individual:
        idx, fast, slow = gene

        if idx in selected or not distance_ok(selected, candidate_df, idx):
            attempts = 0
            while attempts < 200:
                new_idx = random.randint(0, len(candidate_df) - 1)
                if new_idx not in selected and distance_ok(selected, candidate_df, new_idx):
                    idx = new_idx
                    break
                attempts += 1

        selected.append(idx)
        repaired.append([
            int(idx),
            int(max(1, min(8, fast))),
            int(max(1, min(16, slow))),
        ])

    return repaired


def evaluate(individual, candidate_df, inputs):
    demand = inputs["demand"]
    land_cost = inputs["land_cost"]
    suitability = inputs["suitability"]
    grid_capacity = inputs["grid_capacity"]
    remaining_capacity_mw = inputs["remaining_capacity_mw"]
    grid_risk = inputs["grid_risk"]

    total_coverage = 0.0
    total_cost = 0.0
    total_grid_risk = 0.0
    total_suitability = 0.0

    used_cells = set()

    for gene in individual:
        idx, fast, slow = gene

        row = int(candidate_df.iloc[idx]["row"])
        col = int(candidate_df.iloc[idx]["col"])
        grid_id = int(candidate_df.iloc[idx]["grid_id"])

        used_cells.add(grid_id)

        power_kw = station_power_kw(fast, slow)
        power_mw = power_kw / 1000.0

        local_demand = demand[row, col]
        local_cost = land_cost[row, col]
        local_suitability = suitability[row, col]
        local_grid_capacity = grid_capacity[row, col]
        local_remaining_mw = max(float(remaining_capacity_mw[row, col]), 1e-6)
        local_base_risk = grid_risk[row, col]

        utilisation = power_mw / local_remaining_mw
        overload = max(power_mw - SAFETY_MARGIN * local_remaining_mw, 0.0)

        charger_capacity_score = min(power_kw / 800.0, 1.5)
        coverage = local_demand * charger_capacity_score

        station_cost = (
            BASE_STATION_COST
            + fast * FAST_CHARGER_COST
            + slow * SLOW_CHARGER_COST
        )

        land_adjusted_cost = station_cost * (1.0 + local_cost)
        augmentation_cost = overload * 2.4e6

        cost = land_adjusted_cost + augmentation_cost

        risk_penalty = (
            0.45 * local_base_risk
            + 0.45 * utilisation ** 2
            + 0.08 * (1.0 - local_grid_capacity) * power_mw
            + overload * 12.0
        )

        total_coverage += coverage
        total_cost += cost
        total_grid_risk += risk_penalty
        total_suitability += local_suitability

    spacing = spacing_penalty(individual, candidate_df)

    diversity_bonus = len(used_cells) / N_STATIONS
    avg_suitability = total_suitability / N_STATIONS

    total_grid_risk += spacing * 0.8
    total_cost += spacing * 180000

    objectives = np.array([
        -total_coverage,
        total_cost,
        total_grid_risk,
        -avg_suitability,
        -diversity_bonus,
    ], dtype=np.float64)

    return objectives


def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(objectives):
    n = len(objectives)

    S = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if dominates(objectives[p], objectives[q]):
                S[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0

    while len(fronts[i]) > 0:
        next_front = []

        for p in fronts[i]:
            for q in S[p]:
                domination_count[q] -= 1

                if domination_count[q] == 0:
                    next_front.append(q)

        i += 1
        fronts.append(next_front)

    return fronts[:-1]


def crowding_distance(front, objectives):
    if len(front) == 0:
        return {}

    distances = {idx: 0.0 for idx in front}
    num_objectives = objectives.shape[1]

    for m in range(num_objectives):
        values = objectives[front, m]
        sorted_idx = np.argsort(values)

        distances[front[sorted_idx[0]]] = float("inf")
        distances[front[sorted_idx[-1]]] = float("inf")

        min_val = values[sorted_idx[0]]
        max_val = values[sorted_idx[-1]]

        if max_val - min_val < 1e-12:
            continue

        for i in range(1, len(front) - 1):
            prev_val = values[sorted_idx[i - 1]]
            next_val = values[sorted_idx[i + 1]]

            distances[front[sorted_idx[i]]] += (
                (next_val - prev_val) / (max_val - min_val)
            )

    return distances


def tournament_select(population, objectives):
    i, j = random.sample(range(len(population)), 2)

    if dominates(objectives[i], objectives[j]):
        return population[i]

    if dominates(objectives[j], objectives[i]):
        return population[j]

    return population[i] if random.random() < 0.5 else population[j]


def crossover(parent1, parent2):
    child = []

    for g1, g2 in zip(parent1, parent2):
        child.append(g1.copy() if random.random() < 0.5 else g2.copy())

    return child


def mutate(individual, candidate_df, mutation_rate=0.18):
    child = [gene.copy() for gene in individual]

    for i in range(len(child)):
        if random.random() < mutation_rate:
            mutation_type = random.choice(["location", "fast", "slow"])

            if mutation_type == "location":
                child[i][0] = random.randint(0, len(candidate_df) - 1)

            elif mutation_type == "fast":
                child[i][1] = max(
                    1,
                    min(8, child[i][1] + random.choice([-1, 1]))
                )

            elif mutation_type == "slow":
                child[i][2] = max(
                    1,
                    min(16, child[i][2] + random.choice([-2, -1, 1, 2]))
                )

    return repair_individual(child, candidate_df)


def make_next_generation(combined_pop, combined_obj):
    fronts = fast_non_dominated_sort(combined_obj)

    next_pop = []
    next_obj = []

    for front in fronts:
        if len(next_pop) + len(front) <= POP_SIZE:
            for idx in front:
                next_pop.append(combined_pop[idx])
                next_obj.append(combined_obj[idx])
        else:
            distances = crowding_distance(front, combined_obj)

            sorted_front = sorted(
                front,
                key=lambda idx: distances[idx],
                reverse=True,
            )

            remaining = POP_SIZE - len(next_pop)

            for idx in sorted_front[:remaining]:
                next_pop.append(combined_pop[idx])
                next_obj.append(combined_obj[idx])

            break

    return next_pop, np.array(next_obj)


def individual_to_plan(individual, candidate_df, inputs):
    rows = []

    demand = inputs["demand"]
    land_cost = inputs["land_cost"]
    suitability = inputs["suitability"]
    grid_capacity = inputs["grid_capacity"]
    remaining_capacity_mw = inputs["remaining_capacity_mw"]
    grid_risk = inputs["grid_risk"]

    for rank, gene in enumerate(individual, start=1):
        idx, fast, slow = gene

        r = int(candidate_df.iloc[idx]["row"])
        c = int(candidate_df.iloc[idx]["col"])
        gid = int(candidate_df.iloc[idx]["grid_id"])

        power_kw = station_power_kw(fast, slow)
        power_mw = power_kw / 1000.0

        remaining_mw = max(float(remaining_capacity_mw[r, c]), 1e-6)
        utilisation = power_mw / remaining_mw
        overload_mw = max(power_mw - SAFETY_MARGIN * remaining_mw, 0.0)

        rows.append(
            {
                "station_rank": rank,
                "grid_id": gid,
                "grid_row": r,
                "grid_col": c,
                "fast_chargers": int(fast),
                "slow_chargers": int(slow),
                "estimated_power_kw": float(power_kw),
                "estimated_power_mw": float(power_mw),
                "remaining_capacity_mw": float(remaining_mw),
                "grid_utilisation_ratio": float(utilisation),
                "grid_overload_mw": float(overload_mw),
                "demand_score": float(demand[r, c]),
                "land_cost_proxy": float(land_cost[r, c]),
                "construction_suitability": float(suitability[r, c]),
                "grid_capacity_proxy": float(grid_capacity[r, c]),
                "grid_overload_risk": float(grid_risk[r, c]),
            }
        )

    return pd.DataFrame(rows)


def select_balanced_solution(pareto_df):
    pareto_df = pareto_df.copy()

    pareto_df["score"] = (
        0.38 * pareto_df["coverage_score"].rank(pct=True)
        + 0.22 * (1.0 - pareto_df["total_cost"].rank(pct=True))
        + 0.25 * (1.0 - pareto_df["grid_risk"].rank(pct=True))
        + 0.10 * pareto_df["avg_suitability"].rank(pct=True)
        + 0.05 * pareto_df["diversity"].rank(pct=True)
    )

    return pareto_df.sort_values("score", ascending=False).iloc[0], pareto_df


def select_knee_solution(pareto_df):
    df = pareto_df.copy()

    coverage_norm = normalise(df["coverage_score"].values)
    cost_norm = normalise(df["total_cost"].values)
    risk_norm = normalise(df["grid_risk"].values)

    ideal = np.array([1.0, 0.0, 0.0])

    points = np.vstack([
        coverage_norm,
        cost_norm,
        risk_norm,
    ]).T

    distances = np.linalg.norm(points - ideal, axis=1)

    df["knee_distance"] = distances

    return df.sort_values("knee_distance").iloc[0], df


def make_baseline_solution(candidate_df, mode="top_demand"):
    if mode == "random":
        selected = random.sample(range(len(candidate_df)), N_STATIONS)

    elif mode == "top_demand":
        selected = (
            candidate_df.sort_values("demand", ascending=False)
            .head(N_STATIONS)
            .index
            .tolist()
        )

    else:
        raise ValueError(f"Unknown baseline mode: {mode}")

    individual = []

    for idx in selected:
        individual.append([idx, 4, 8])

    return repair_individual(individual, candidate_df)


def main():
    set_seed(RANDOM_SEED)

    print("Running NSGA-II EV charging station optimiser...")

    inputs = load_inputs()
    candidate_df = build_candidate_pool(inputs)

    if len(candidate_df) < N_STATIONS:
        raise ValueError("Not enough candidate cells for station placement.")

    population = [
        create_individual(candidate_df)
        for _ in range(POP_SIZE)
    ]

    objectives = np.array([
        evaluate(ind, candidate_df, inputs)
        for ind in population
    ])

    history = []

    for gen in range(1, N_GENERATIONS + 1):
        offspring = []

        while len(offspring) < POP_SIZE:
            p1 = tournament_select(population, objectives)
            p2 = tournament_select(population, objectives)

            child = crossover(p1, p2)
            child = mutate(child, candidate_df)

            offspring.append(child)

        offspring_obj = np.array([
            evaluate(ind, candidate_df, inputs)
            for ind in offspring
        ])

        combined_pop = population + offspring
        combined_obj = np.vstack([objectives, offspring_obj])

        population, objectives = make_next_generation(
            combined_pop,
            combined_obj,
        )

        best_coverage = -np.min(objectives[:, 0])
        best_cost = np.min(objectives[:, 1])
        best_risk = np.min(objectives[:, 2])

        history.append(
            {
                "generation": gen,
                "best_coverage": best_coverage,
                "best_cost": best_cost,
                "best_grid_risk": best_risk,
            }
        )

        print(
            f"Gen {gen:03d}/{N_GENERATIONS} | "
            f"coverage={best_coverage:.4f} | "
            f"cost={best_cost:.2f} | "
            f"risk={best_risk:.4f}"
        )

    fronts = fast_non_dominated_sort(objectives)
    best_front = fronts[0]

    pareto_rows = []

    for rank, idx in enumerate(best_front, start=1):
        obj = objectives[idx]

        pareto_rows.append(
            {
                "solution_id": rank,
                "coverage_score": float(-obj[0]),
                "total_cost": float(obj[1]),
                "grid_risk": float(obj[2]),
                "avg_suitability": float(-obj[3]),
                "diversity": float(-obj[4]),
                "population_index": int(idx),
            }
        )

    pareto_df = pd.DataFrame(pareto_rows)

    best_solution_row, pareto_scored = select_balanced_solution(pareto_df)
    knee_solution_row, pareto_scored = select_knee_solution(pareto_scored)

    pareto_path = os.path.join(RESULTS_DIR, "nsga2_pareto_solutions.csv")
    pareto_scored.to_csv(pareto_path, index=False)
    print("\nSaved:", pareto_path)

    history_path = os.path.join(RESULTS_DIR, "nsga2_optimisation_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    print("Saved:", history_path)

    best_idx = int(best_solution_row["population_index"])
    knee_idx = int(knee_solution_row["population_index"])

    best_individual = population[best_idx]
    knee_individual = population[knee_idx]

    plan_df = individual_to_plan(best_individual, candidate_df, inputs)
    knee_plan_df = individual_to_plan(knee_individual, candidate_df, inputs)

    plan_path = os.path.join(RESULTS_DIR, "optimal_station_plan.csv")
    knee_plan_path = os.path.join(RESULTS_DIR, "knee_station_plan.csv")

    plan_df.to_csv(plan_path, index=False)
    knee_plan_df.to_csv(knee_plan_path, index=False)

    print("Saved:", plan_path)
    print("Saved:", knee_plan_path)

    baseline_rows = []

    for mode in ["random", "top_demand"]:
        baseline_individual = make_baseline_solution(candidate_df, mode=mode)
        obj = evaluate(baseline_individual, candidate_df, inputs)

        baseline_rows.append(
            {
                "method": mode,
                "coverage_score": float(-obj[0]),
                "total_cost": float(obj[1]),
                "grid_risk": float(obj[2]),
                "avg_suitability": float(-obj[3]),
                "diversity": float(-obj[4]),
            }
        )

    baseline_rows.append(
        {
            "method": "nsga2_balanced",
            "coverage_score": float(best_solution_row["coverage_score"]),
            "total_cost": float(best_solution_row["total_cost"]),
            "grid_risk": float(best_solution_row["grid_risk"]),
            "avg_suitability": float(best_solution_row["avg_suitability"]),
            "diversity": float(best_solution_row["diversity"]),
        }
    )

    baseline_rows.append(
        {
            "method": "nsga2_knee",
            "coverage_score": float(knee_solution_row["coverage_score"]),
            "total_cost": float(knee_solution_row["total_cost"]),
            "grid_risk": float(knee_solution_row["grid_risk"]),
            "avg_suitability": float(knee_solution_row["avg_suitability"]),
            "diversity": float(knee_solution_row["diversity"]),
        }
    )

    baseline_df = pd.DataFrame(baseline_rows)

    baseline_path = os.path.join(RESULTS_DIR, "nsga2_baseline_comparison.csv")
    baseline_df.to_csv(baseline_path, index=False)
    print("Saved:", baseline_path)

    summary = {
        "selected_solution": "nsga2_balanced",
        "number_of_stations": len(plan_df),
        "total_fast_chargers": int(plan_df["fast_chargers"].sum()),
        "total_slow_chargers": int(plan_df["slow_chargers"].sum()),
        "total_estimated_power_kw": float(plan_df["estimated_power_kw"].sum()),
        "average_demand_score": float(plan_df["demand_score"].mean()),
        "average_land_cost_proxy": float(plan_df["land_cost_proxy"].mean()),
        "average_grid_overload_risk": float(plan_df["grid_overload_risk"].mean()),
        "average_grid_utilisation_ratio": float(plan_df["grid_utilisation_ratio"].mean()),
        "total_grid_overload_mw": float(plan_df["grid_overload_mw"].sum()),
        "coverage_score": float(best_solution_row["coverage_score"]),
        "total_cost": float(best_solution_row["total_cost"]),
        "grid_risk": float(best_solution_row["grid_risk"]),
    }

    summary_path = os.path.join(RESULTS_DIR, "optimal_station_plan_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print("Saved:", summary_path)

    print("\nBest balanced solution:")
    print(best_solution_row)

    print("\nKnee point solution:")
    print(knee_solution_row)

    print("\nBaseline comparison:")
    print(baseline_df)

    print("\nNSGA-II optimisation complete.")


if __name__ == "__main__":
    main()