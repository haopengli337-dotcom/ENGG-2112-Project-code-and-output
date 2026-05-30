# =========================
# config.py
# Project configuration
# =========================

import os

# =========================
# Root paths
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

# =========================
# Input data files
# =========================

GRID_FILE = os.path.join(DATASET_DIR, "grid_matrix.csv")
CHARGER_FILE = os.path.join(DATASET_DIR, "cleaned_ev_chargers.csv")
EV_FILE = os.path.join(DATASET_DIR, "cleaned_ev_data.csv")
TRAFFIC_FILE = os.path.join(DATASET_DIR, "traffic.csv")

# Optional files
GRID_CAPACITY_FILE = os.path.join(DATASET_DIR, "grid_capacity.csv")
SUBSTATION_FILE = os.path.join(DATASET_DIR, "substations.csv")
FEEDER_FILE = os.path.join(DATASET_DIR, "feeders.csv")
SCADA_FILE = os.path.join(DATASET_DIR, "scada.csv")

# =========================
# Intermediate files
# =========================

TENSOR_FILE = os.path.join(DATASET_DIR, "model_tensors.npz")
GRAPH_FILE = os.path.join(DATASET_DIR, "graph_data.npz")

# =========================
# Output files
# =========================

PREDICTION_FILE = os.path.join(RESULT_DIR, "prediction_maps.npz")
OPTIMIZATION_FILE = os.path.join(RESULT_DIR, "nsga2_plan.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "hybrid_model.pt")

# =========================
# Grid settings
# =========================

GRID_SIZE = 100
TIME_STEPS = 24

# =========================
# Model settings
# =========================

OUTPUT_CHANNELS = 5

CNN_FEATURES = 64
LSTM_HIDDEN = 64
GNN_HIDDEN = 64

LEARNING_RATE = 0.001
EPOCHS = 200

# =========================
# Charger assumptions
# =========================

SLOW_CHARGER_POWER_KW = 22
FAST_CHARGER_POWER_KW = 150

# =========================
# Grid assumptions
# Used when real grid data is missing
# =========================

DEFAULT_GRID_CAPACITY_MW = 5.0
DEFAULT_PEAK_LOAD_MW = 3.0
DEFAULT_CURRENT_LOAD_MW = 2.0
DEFAULT_REMAINING_CAPACITY_MW = 2.0

VOLTAGE_MANAGEMENT_COST_PER_MW = 0.4e6
NETWORK_AUGMENTATION_COST_PER_MW = 2.4e6

# =========================
# NSGA-II settings
# =========================

NSGA_POP_SIZE = 80
NSGA_GENERATIONS = 100

MAX_NEW_STATIONS = 20
MAX_FAST_CHARGERS_PER_SITE = 8
MAX_SLOW_CHARGERS_PER_SITE = 12

MIN_FEASIBILITY = 0.1
TOTAL_BUDGET_AUD = 20e6

# =========================
# Output channel names
# =========================

OUTPUT_NAMES = [
    "demand",
    "fast_suitability",
    "slow_suitability",
    "grid_risk",
    "feasibility"
]

# =========================
# Make directories
# =========================

def make_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


make_dirs()