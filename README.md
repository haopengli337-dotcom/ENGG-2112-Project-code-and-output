# AI-Based Urban EV Charging Station Planning System

## Overview

This project develops an AI-based urban EV charging station planning framework using:

- Attention CNN
- ConvLSTM
- Graph Neural Network (GNN)
- NSGA-II multi-objective optimization
- Realistic electricity grid constraints

The framework integrates:

- Urban spatial map data
- EV adoption data
- Traffic flow data
- Existing charger infrastructure
- Power grid capacity constraints

to generate intelligent EV charging infrastructure deployment plans.

---

# System Architecture

```text
Spatial Data + Traffic + Grid Data
                ↓
        Tensor Builder
                ↓
        CNN Spatial Branch
                ↓
        ConvLSTM Temporal Branch
                ↓
        Graph Neural Network
                ↓
         Feature Fusion
                ↓
         Prediction Maps
                ↓
        NSGA-II Optimizer
                ↓
     Final Charging Plan