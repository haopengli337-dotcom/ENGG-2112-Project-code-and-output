# =========================
# preprocessing/graph.py
# Build graph structure for GNN
# Compatible with 256x256 unified static tensor
# =========================

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def grid_to_node(i, j, width):
    return i * width + j


def node_to_grid(node_id, width):
    i = node_id // width
    j = node_id % width
    return i, j


def build_grid_edges(height, width, diagonal=False):
    edges = []

    if diagonal:
        directions = [
            (-1, 0), (1, 0),
            (0, -1), (0, 1),
            (-1, -1), (-1, 1),
            (1, -1), (1, 1)
        ]
    else:
        directions = [
            (-1, 0), (1, 0),
            (0, -1), (0, 1)
        ]

    for i in range(height):
        for j in range(width):
            current = grid_to_node(i, j, width)

            for di, dj in directions:
                ni = i + di
                nj = j + dj

                if 0 <= ni < height and 0 <= nj < width:
                    neighbor = grid_to_node(ni, nj, width)
                    edges.append([current, neighbor])

    return np.array(edges, dtype=np.int64).T


def build_positions(height, width):
    positions = []

    for i in range(height):
        for j in range(width):
            positions.append([i, j])

    return np.array(positions, dtype=np.float32)


def build_node_features(X_static):
    """
    Input:
        X_static shape = (C, H, W)

    Output:
        node_features shape = (H*W, C)
    """

    C, H, W = X_static.shape

    node_features = (
        np.transpose(X_static, (1, 2, 0))
        .reshape(H * W, C)
    )

    return node_features.astype(np.float32)


def load_static_tensor():
    """
    Load new 256x256 unified static tensor.

    Expected file:
        datasets/model_static_layers_256.npz
    """

    tensor_file = os.path.join(
        config.DATASET_DIR,
        "model_static_layers_256.npz"
    )

    if not os.path.exists(tensor_file):
        raise FileNotFoundError(
            f"Missing static tensor file: {tensor_file}"
        )

    tensor_data = np.load(tensor_file, allow_pickle=True)

    print("Loaded tensor file:", tensor_file)
    print("Available keys:", tensor_data.files)

    if "X_static" in tensor_data.files:
        X_static = tensor_data["X_static"]
    elif "static_tensor" in tensor_data.files:
        X_static = tensor_data["static_tensor"]
    elif "tensor" in tensor_data.files:
        X_static = tensor_data["tensor"]
    elif "data" in tensor_data.files:
        X_static = tensor_data["data"]
    else:
        raise KeyError(
            "Cannot find tensor data key. "
            "Expected one of: X_static, static_tensor, tensor, data"
        )

    if "layer_names" in tensor_data.files:
        layer_names = tensor_data["layer_names"]
    else:
        layer_names = np.array(
            [f"layer_{i}" for i in range(X_static.shape[0])]
        )

    return X_static, layer_names, tensor_file


def build_graph():
    X_static, layer_names, tensor_file = load_static_tensor()

    C, H, W = X_static.shape

    node_features = build_node_features(X_static)

    edge_index = build_grid_edges(
        height=H,
        width=W,
        diagonal=False
    )

    positions = build_positions(H, W)

    np.savez_compressed(
        config.GRAPH_FILE,
        node_features=node_features,
        edge_index=edge_index,
        positions=positions,
        layer_names=layer_names,
        height=H,
        width=W,
        source_tensor=tensor_file
    )

    print("\nGraph build complete.")
    print("Input tensor:", X_static.shape)
    print("node_features:", node_features.shape)
    print("edge_index:", edge_index.shape)
    print("positions:", positions.shape)
    print("Saved:", config.GRAPH_FILE)


if __name__ == "__main__":
    build_graph()