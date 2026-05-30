# =========================
# main.py
# Run full EV AI planning pipeline
# =========================

import subprocess
import sys
from pathlib import Path

import config


# Project root
ROOT_DIR = Path(__file__).resolve().parent


def run_step(name, module_name):
    print("\n" + "=" * 60)
    print(f"Running: {name}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=ROOT_DIR
    )

    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {name}")

    print(f"Finished: {name}")


def main():
    config.make_dirs()

    steps = [
        (
            "Build static spatial tensors",
            "preprocessing.build_static_tensor",
        ),
        (
            "Build demand labels",
            "preprocessing.build_demand_label",
        ),
        (
            "Build graph data",
            "preprocessing.graph",
        ),
        (
            "Train CNN / hybrid demand model",
            "train",
        ),
        (
            "Run NSGA-II optimizer",
            "optimizer.nsga2_optimizer",
        ),
        (
            "Generate optimisation figures",
            "vis.plot_optimisation",
        ),
        (
            "Generate prediction figures",
            "vis.plot_prediction",
        ),
    ]

    for name, module_name in steps:
        run_step(name, module_name)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully.")
    print("=" * 60)
    print("Results folder:", config.RESULT_DIR)
    print("Figures folder:", config.FIGURE_DIR)
    print("Model folder:", config.MODEL_DIR)


if __name__ == "__main__":
    main()