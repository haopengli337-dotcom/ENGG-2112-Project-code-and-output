# ==========================================
# EV Charging AI Planning System
# ==========================================

PYTHON = .venv/bin/python

# ==========================================
# Full pipeline
# ==========================================

run:
	$(PYTHON) main.py

all: preprocess tensors graph train opt plot


# ==========================================
# Raw data preprocessing
# ==========================================

traffic:
	$(PYTHON) preprocessing/make_traffic.py

population:
	$(PYTHON) preprocessing/population.py

griddata:
	$(PYTHON) preprocessing/power_grid.py

land:
	$(PYTHON) preprocessing/land_cost.py

chargers:
	$(PYTHON) preprocessing/ev_chargers.py


# ==========================================
# Tensor / graph generation
# ==========================================

tensors:
	$(PYTHON) preprocessing/build_static_tensor.py

labels:
	$(PYTHON) preprocessing/build_demand_label.py

graph:
	$(PYTHON) preprocessing/graph.py


# ==========================================
# Model training
# ==========================================

train:
	$(PYTHON) train.py


# ==========================================
# Optimization
# ==========================================

opt:
	$(PYTHON) optimizer/nsga2_optimizer.py


# ==========================================
# Visualisation
# ==========================================

plot:
	$(PYTHON) vis/plot_optimisation.py


# ==========================================
# Clean outputs
# ==========================================

clean:
	rm -rf outputs/figures/*
	rm -rf outputs/results/*
	rm -rf outputs/models/*


# ==========================================
# Install dependencies
# ==========================================

install:
	pip install -r requirements.txt


# ==========================================
# Help
# ==========================================

help:
	@echo ""
	@echo "============ COMMANDS ============"
	@echo ""
	@echo "make all          -> full pipeline"
	@echo "make run          -> run main.py"
	@echo ""
	@echo "make traffic      -> preprocess traffic"
	@echo "make population   -> preprocess population"
	@echo "make griddata     -> preprocess power grid"
	@echo "make land         -> preprocess land cost"
	@echo "make chargers     -> preprocess EV chargers"
	@echo ""
	@echo "make tensors      -> build tensors"
	@echo "make labels       -> build demand labels"
	@echo "make graph        -> build graph"
	@echo ""
	@echo "make train        -> train ML model"
	@echo "make opt          -> run NSGA-II"
	@echo "make plot         -> generate figures"
	@echo ""
	@echo "make clean        -> clean outputs"
	@echo "make install      -> install requirements"