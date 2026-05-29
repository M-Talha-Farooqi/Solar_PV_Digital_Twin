

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.dataset import get_datasets, make_sequences

ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
LOG_DIR       = ARTIFACTS_DIR / "lstm"
RUNS_DIR      = PROJECT_ROOT / "ml" / "runs" / "lstm"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "models").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)


CONFIG = {
    "window_size":   24,
    "hidden_size":   128,
    "num_layers":    3,
    "dropout":       0.2,
    "batch_size":    512,
    "num_epochs":    60,
    "learning_rate": 1e-3,
    "weight_decay":  1e-5,
    "patience":      10,
    "lr_patience":   5,
    "lr_factor":     0.5,
    "lr_min":        1e-6,
    "grad_clip":     1.0,
}


class SolarLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               dropout=dropout if num_layers > 1 else 0.0,
                               batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mask = y_true > 0.01
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.sum() > 0 else None
    return {"rmse": round(rmse, 6), "mae": round(mae, 6), "r2": round(r2, 6),
            "mape": round(mape, 4) if mape else None}


def eval_loss(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            pred = model(Xb.to(device))
            total += criterion(pred, yb.to(device)).item() * len(yb)
            n += len(yb)
    return total / n


def predict_all(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in loader:
            preds.append(model(Xb.to(device)).cpu().numpy())
    return np.concatenate(preds)


def main():
    print("=" * 60)
    print("  LSTM (PyTorch) — Solar PV Power Generation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[LSTM] Device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, _ = get_datasets(scale=True)

    W = CONFIG["window_size"]
    print(f"[LSTM] Building {W}h sliding-window sequences...")
    X_tr, y_tr = make_sequences(X_train, y_train, W)
    X_v,  y_v  = make_sequences(X_val,   y_val,   W)
    X_te, y_te = make_sequences(X_test,  y_test,  W)
    print(f"  Train: {X_tr.shape}  Val: {X_v.shape}  Test: {X_te.shape}")

    BS = CONFIG["batch_size"]
    tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)),
                           batch_size=BS, shuffle=True,  num_workers=0)
    v_loader  = DataLoader(TensorDataset(torch.tensor(X_v, dtype=torch.float32),  torch.tensor(y_v, dtype=torch.float32)),
                           batch_size=BS, shuffle=False, num_workers=0)
    te_loader = DataLoader(TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.float32)),
                           batch_size=BS, shuffle=False, num_workers=0)

    n_features = X_tr.shape[2]
    model = SolarLSTM(n_features, CONFIG["hidden_size"],
                      CONFIG["num_layers"], CONFIG["dropout"]).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LSTM] Trainable parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"],
                                 weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CONFIG["lr_patience"],
        factor=CONFIG["lr_factor"], min_lr=CONFIG["lr_min"])
    writer = SummaryWriter(log_dir=str(RUNS_DIR))

    epoch_log = []
    best_val  = float("inf")
    best_ep   = 1
    best_state = None
    patience_ctr = 0

    print(f"\n[LSTM] Training for up to {CONFIG['num_epochs']} epochs...")
    t0 = time.perf_counter()

    for ep in range(1, CONFIG["num_epochs"] + 1):
        model.train()
        tr_loss, n = 0.0, 0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()
            tr_loss += loss.item() * len(yb); n += len(yb)
        tr_loss /= n

        v_loss = eval_loss(model, v_loader, criterion, device)
        scheduler.step(v_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        epoch_log.append({"epoch": ep, "train_loss": round(tr_loss, 8),
                           "val_loss": round(v_loss, 8), "lr": lr_now})
        writer.add_scalar("Loss/train", tr_loss, ep)
        writer.add_scalar("Loss/val",   v_loss,  ep)
        writer.add_scalar("LR",         lr_now,  ep)

        if ep % 5 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{CONFIG['num_epochs']}  "
                  f"Train:{tr_loss:.5f}  Val:{v_loss:.5f}  LR:{lr_now:.2e}")

        if v_loss < best_val:
            best_val  = v_loss; best_ep = ep; patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["patience"]:
                print(f"\n  Early stopping at epoch {ep} (best: {best_val:.6f} @ ep {best_ep})")
                break

    train_time = time.perf_counter() - t0
    writer.close()
    model.load_state_dict(best_state)

    t_inf = time.perf_counter()
    y_pred_test = predict_all(model, te_loader, device)
    latency_ms  = (time.perf_counter() - t_inf) * 1000 / len(y_te)
    y_pred_val   = predict_all(model, v_loader,  device)
    y_pred_train = predict_all(model, tr_loader, device)

    tm = compute_metrics(y_tr, y_pred_train)
    vm = compute_metrics(y_v,  y_pred_val)
    em = compute_metrics(y_te, y_pred_test)
    print(f"\n  Training time: {train_time:.1f}s")
    print(f"  [Train] R²={tm['r2']:.4f}  RMSE={tm['rmse']:.4f}")
    print(f"  [Val]   R²={vm['r2']:.4f}  RMSE={vm['rmse']:.4f}  MAPE={vm['mape']:.2f}%")
    print(f"  [Test]  R²={em['r2']:.4f}  RMSE={em['rmse']:.4f}  MAPE={em['mape']:.2f}%")

    
    torch.save({
        "state_dict": best_state,
        "config": {**CONFIG, "input_size": n_features},
        "feature_cols": feature_cols,
        "best_epoch": best_ep, "best_val_loss": best_val,
    }, str(ARTIFACTS_DIR / "models" / "lstm.pt"))
    print(f"  ✓ Model saved")

    
    with open(ARTIFACTS_DIR / "predictions" / "lstm_preds.json", "w") as f:
        json.dump({
            "y_train": y_tr.tolist(), "y_pred_train": y_pred_train.tolist(),
            "y_val":   y_v.tolist(),  "y_pred_val":   y_pred_val.tolist(),
            "y_test":  y_te.tolist(), "y_pred_test":  y_pred_test.tolist(),
        }, f)
    print(f"  ✓ Predictions saved")

    
    training_log = {
        "model": "LSTM",
        "config": {**CONFIG, "input_size": n_features, "total_params": total_params},
        "feature_cols": feature_cols,
        "n_features": n_features,
        "dataset_sizes": {
            "train_sequences": int(len(X_tr)),
            "val_sequences":   int(len(X_v)),
            "test_sequences":  int(len(X_te)),
        },
        "training_progress": epoch_log,   
        "best_epoch": best_ep,
        "best_val_loss": round(best_val, 8),
        "total_epochs_run": len(epoch_log),
        "train_time_s": round(train_time, 3),
        "inference_latency_ms_per_sample": round(latency_ms, 6),
        "tensorboard_dir": str(RUNS_DIR.relative_to(PROJECT_ROOT)),
        "metrics": {"train": tm, "val": vm, "test": em},
    }
    log_path = LOG_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"  ✓ Training log ({len(epoch_log)} epochs) saved")
    print(f"  TensorBoard: tensorboard --logdir ml/runs/")

    print(f"\n✅ LSTM done!  Test R²={em['r2']:.4f}  RMSE={em['rmse']:.4f}")
    return training_log


if __name__ == "__main__":
    main()
