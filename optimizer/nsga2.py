# =========================
# optimizer/nsga2.py
# NSGA-II multi-objective optimizer
# =========================

import os
import sys
import numpy as np
import pandas as pd

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from constraints.grid import GridConstraintChecker


# =========================
# Load prediction maps
# =========================

def load_predictions():
    if not os.path.exists(config.PREDICTION_FILE):
        raise FileNotFoundError(
            f"Missing prediction file: {config.PREDICTION_FILE}"
        )

    data = np.load(config.PREDICTION_FILE)

    pred = data["prediction"]

    return {
        "demand": pred[0],
        "fast": pred[1],
        "slow": pred[2],
        "grid_risk": pred[3],
        "feasibility": pred[4]
    }


# =========================
# Load input layers
# =========================

def load_layers():
    data = np.load(
        config.TENSOR_FILE,
        allow_pickle=True
    )

    X_static = data["X_static"]
    layer_names = data["layer_names"]

    layers = {}

    for idx, name in enumerate(layer_names):
        layers[str(name)] = X_static[idx]

    return layers


# =========================
# NSGA-II problem
# =========================

class EVChargingProblem(ElementwiseProblem):
    def __init__(
        self,
        prediction,
        layers,
        grid_checker
    ):
        self.prediction = prediction
        self.layers = layers
        self.grid_checker = grid_checker

        self.H, self.W = prediction["demand"].shape

        super().__init__(
            n_var=4,
            n_obj=4,
            n_ieq_constr=4,
            xl=np.array([0, 0, 0, 0]),
            xu=np.array([
                self.H - 1,
                self.W - 1,
                config.MAX_FAST_CHARGERS_PER_SITE,
                config.MAX_SLOW_CHARGERS_PER_SITE
            ]),
            vtype=int
        )

    def _evaluate(self, x, out, *args, **kwargs):
        i = int(x[0])
        j = int(x[1])
        fast_num = int(x[2])
        slow_num = int(x[3])

        demand = self.prediction["demand"][i, j]
        fast_score = self.prediction["fast"][i, j]
        slow_score = self.prediction["slow"][i, j]
        grid_risk = self.prediction["grid_risk"][i, j]
        feasibility = self.prediction["feasibility"][i, j]

        coverage_gap = self.layers["coverage_gap"][i, j]
        accessibility = self.layers["accessibility"][i, j]
        construction_cost = self.layers["construction_cost"][i, j]
        land_availability = self.layers["land_availability"][i, j]

        grid_eval = self.grid_checker.evaluate(
            i=i,
            j=j,
            num_fast=fast_num,
            num_slow=slow_num
        )

        charger_match = (
            fast_num * fast_score
            + slow_num * slow_score
        )

        demand_score = (
            demand
            * coverage_gap
            * accessibility
            * (1.0 + charger_match)
        )

        cost_score = (
            construction_cost
            + grid_eval["augmentation_cost"]
            + 0.1 * fast_num
            + 0.03 * slow_num
        )

        grid_penalty = (
            grid_risk
            + grid_eval["risk_penalty"]
        )

        feasibility_score = (
            feasibility
            * land_availability
            * coverage_gap
        )

        # NSGA-II minimizes objectives
        out["F"] = [
            -demand_score,
            grid_penalty,
            cost_score,
            -feasibility_score
        ]

        # Constraints <= 0
        g1 = 1 - (fast_num + slow_num)
        g2 = grid_eval["overload_mw"]
        g3 = config.MIN_FEASIBILITY - feasibility
        g4 = 0.05 - land_availability

        out["G"] = [g1, g2, g3, g4]


# =========================
# Run NSGA-II
# =========================

def run_optimizer():
    prediction = load_predictions()
    layers = load_layers()

    checker = GridConstraintChecker(
        remaining_capacity_layer=layers["remaining_capacity"],
        grid_risk_layer=layers["grid_risk"],
        augmentation_cost_layer=layers["augmentation_cost"],
        slow_charger_power_kw=config.SLOW_CHARGER_POWER_KW,
        fast_charger_power_kw=config.FAST_CHARGER_POWER_KW
    )

    problem = EVChargingProblem(
        prediction=prediction,
        layers=layers,
        grid_checker=checker
    )

    algorithm = NSGA2(
        pop_size=config.NSGA_POP_SIZE,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination(
        "n_gen",
        config.NSGA_GENERATIONS
    )

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        verbose=True
    )

    if result.X is None:
        raise RuntimeError("NSGA-II found no feasible solution.")

    rows = []

    for idx, x in enumerate(result.X):
        i = int(x[0])
        j = int(x[1])
        fast_num = int(x[2])
        slow_num = int(x[3])

        grid_eval = checker.evaluate(
            i=i,
            j=j,
            num_fast=fast_num,
            num_slow=slow_num
        )

        rows.append({
            "rank_id": idx + 1,
            "grid_i": i,
            "grid_j": j,
            "num_fast_chargers": fast_num,
            "num_slow_chargers": slow_num,
            "demand_score": prediction["demand"][i, j],
            "fast_score": prediction["fast"][i, j],
            "slow_score": prediction["slow"][i, j],
            "grid_risk_score": prediction["grid_risk"][i, j],
            "feasibility_score": prediction["feasibility"][i, j],
            "added_load_mw": grid_eval["added_load_mw"],
            "overload_mw": grid_eval["overload_mw"],
            "risk_penalty": grid_eval["risk_penalty"],
            "augmentation_cost": grid_eval["augmentation_cost"],
            "objective_demand": -result.F[idx][0],
            "objective_grid_risk": result.F[idx][1],
            "objective_cost": result.F[idx][2],
            "objective_feasibility": -result.F[idx][3]
        })

    df = pd.DataFrame(rows)

    df["final_score"] = (
        0.35 * df["objective_demand"]
        + 0.25 * df["objective_feasibility"]
        - 0.20 * df["objective_grid_risk"]
        - 0.20 * df["objective_cost"]
    )

    df = df.sort_values(
        "final_score",
        ascending=False
    ).head(config.MAX_NEW_STATIONS)

    df.to_csv(
        config.OPTIMIZATION_FILE,
        index=False
    )

    print("NSGA-II optimization complete.")
    print("Saved:", config.OPTIMIZATION_FILE)
    print(df)

    return df


if __name__ == "__main__":
    run_optimizer()