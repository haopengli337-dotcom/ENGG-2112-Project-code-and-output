# training/train_cnn_baseline.py

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_OUT_DIR = os.path.join(OUTPUT_DIR, "models")
RESULT_OUT_DIR = os.path.join(OUTPUT_DIR, "results")

os.makedirs(MODEL_OUT_DIR, exist_ok=True)
os.makedirs(RESULT_OUT_DIR, exist_ok=True)

STATIC_FILE = os.path.join(DATASETS_DIR, "model_static_layers_256.npz")
LABEL_FILE = os.path.join(DATASETS_DIR, "demand_label_256.npz")

PATCH_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
VAL_RATIO = 0.2


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


class PatchDataset(Dataset):
    def __init__(self, X, y, patch_size=32, stride=16):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.patch_size = patch_size
        self.samples = []

        _, h, w = X.shape

        for r in range(0, h - patch_size + 1, stride):
            for c in range(0, w - patch_size + 1, stride):
                y_patch = y[r:r + patch_size, c:c + patch_size]

                if np.max(y_patch) <= 0:
                    continue

                self.samples.append((r, c))

        print(f"Dataset patches: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r, c = self.samples[idx]

        x_patch = self.X[
            :,
            r:r + self.patch_size,
            c:c + self.patch_size
        ]

        y_patch = self.y[
            r:r + self.patch_size,
            c:c + self.patch_size
        ]

        return (
            torch.tensor(x_patch, dtype=torch.float32),
            torch.tensor(y_patch[None, :, :], dtype=torch.float32),
        )


class SimpleDemandCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def evaluate_loader(model, loader, loss_fn, device):
    model.eval()

    losses = []
    preds = []
    trues = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)

            losses.append(loss.item())
            preds.append(pred.cpu().numpy().reshape(-1))
            trues.append(y_batch.cpu().numpy().reshape(-1))

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if ss_tot < 1e-12:
        r2 = 0.0
    else:
        r2 = float(1.0 - ss_res / ss_tot)

    return {
        "loss": float(np.mean(losses)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def predict_full_map(model, X, device):
    model.eval()

    with torch.no_grad():
        x = torch.tensor(
            X[None, :, :, :],
            dtype=torch.float32
        ).to(device)

        pred = model(x).cpu().numpy()[0, 0]

    return pred


def compute_full_map_metrics(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if ss_tot < 1e-12:
        r2 = 0.0
    else:
        r2 = float(1.0 - ss_res / ss_tot)

    return mae, rmse, r2


def main():
    set_seed(RANDOM_SEED)

    if not os.path.exists(STATIC_FILE):
        raise FileNotFoundError(f"Missing: {STATIC_FILE}")

    if not os.path.exists(LABEL_FILE):
        raise FileNotFoundError(f"Missing: {LABEL_FILE}")

    static_data = np.load(STATIC_FILE, allow_pickle=True)
    label_data = np.load(LABEL_FILE, allow_pickle=True)

    X_static = static_data["X_static"].astype(np.float32)
    y_demand = label_data["y_demand"].astype(np.float32)

    print("X_static:", X_static.shape)
    print("y_demand:", y_demand.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.backends.mps.is_available():
        device = torch.device("mps")

    print("Using device:", device)

    dataset = PatchDataset(
        X_static,
        y_demand,
        patch_size=PATCH_SIZE,
        stride=16,
    )

    if len(dataset) == 0:
        raise ValueError("No valid training patches found.")

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    print("Train patches:", len(train_dataset))
    print("Validation patches:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = SimpleDemandCNN(
        in_channels=X_static.shape[0]
    ).to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    history = []

    best_val_loss = np.inf
    best_model_state = None

    print("\nTraining CNN baseline...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        val_metrics = evaluate_loader(
            model,
            val_loader,
            loss_fn,
            device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
        }

        history.append(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model_state = model.state_dict()

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_R2={val_metrics['r2']:.4f}"
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model_path = os.path.join(
        MODEL_OUT_DIR,
        "cnn_baseline_demand.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": X_static.shape[0],
            "patch_size": PATCH_SIZE,
            "epochs": EPOCHS,
            "history": history,
            "best_val_loss": best_val_loss,
        },
        model_path,
    )

    print("\nSaved model:", model_path)

    pred_map = predict_full_map(model, X_static, device)

    full_mae, full_rmse, full_r2 = compute_full_map_metrics(
        y_demand,
        pred_map,
    )

    pred_path = os.path.join(
        RESULT_OUT_DIR,
        "cnn_predicted_demand.csv"
    )

    pd.DataFrame(pred_map).to_csv(
        pred_path,
        index=False,
        header=False,
    )

    print("Saved prediction:", pred_path)

    npz_pred_path = os.path.join(
        RESULT_OUT_DIR,
        "cnn_predicted_demand.npz"
    )

    np.savez_compressed(
        npz_pred_path,
        pred_demand=pred_map.astype(np.float32),
        y_true=y_demand.astype(np.float32),
    )

    print("Saved prediction npz:", npz_pred_path)

    history_path = os.path.join(
        RESULT_OUT_DIR,
        "cnn_training_loss.csv"
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    print("Saved training history:", history_path)

    metrics_path = os.path.join(
        RESULT_OUT_DIR,
        "cnn_metrics.csv"
    )

    pd.DataFrame([
        {
            "best_val_loss": best_val_loss,
            "full_map_mae": full_mae,
            "full_map_rmse": full_rmse,
            "full_map_r2": full_r2,
            "num_patches": len(dataset),
            "train_patches": len(train_dataset),
            "val_patches": len(val_dataset),
            "input_channels": X_static.shape[0],
        }
    ]).to_csv(metrics_path, index=False)

    print("Saved metrics:", metrics_path)

    print("\nFinal full-map metrics:")
    print(f"MAE  = {full_mae:.6f}")
    print(f"RMSE = {full_rmse:.6f}")
    print(f"R2   = {full_r2:.6f}")

    print("\nCNN baseline training complete.")


if __name__ == "__main__":
    main()