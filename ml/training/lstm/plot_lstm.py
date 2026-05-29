

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

PLOTS_DIR = PROJECT_ROOT / "ml" / "plots" / "lstm"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
COLOR = "#C5453B"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 11.5, "axes.grid": True, "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
})


def load_artifacts():
    log_path  = PROJECT_ROOT / "ml" / "artifacts" / "lstm" / "training_log.json"
    pred_path = PROJECT_ROOT / "ml" / "artifacts" / "predictions" / "lstm_preds.json"
    with open(log_path)  as f: log  = json.load(f)
    with open(pred_path) as f: pred = json.load(f)
    return log, pred


def plot_loss_curve(progress, best_epoch, save_path):
    epochs    = [p["epoch"]      for p in progress]
    tr_loss   = [p["train_loss"] for p in progress]
    val_loss  = [p["val_loss"]   for p in progress]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    
    axes[0].plot(epochs, tr_loss,  label="Train Loss (MSE)", color=COLOR,    lw=1.5)
    axes[0].plot(epochs, val_loss, label="Val Loss (MSE)",   color="#2E86AB", lw=1.5)
    axes[0].axvline(best_epoch, color="green", ls="--", lw=1.5,
                    label=f"Best epoch ({best_epoch})")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss (log scale)")
    axes[0].set_title("LSTM — Loss per Epoch (Log Scale)")
    axes[0].legend()

    
    axes[1].plot(epochs, tr_loss,  label="Train Loss (MSE)", color=COLOR,    lw=1.5)
    axes[1].plot(epochs, val_loss, label="Val Loss (MSE)",   color="#2E86AB", lw=1.5)
    axes[1].axvline(best_epoch, color="green", ls="--", lw=1.5,
                    label=f"Best epoch ({best_epoch})")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MSE Loss")
    axes[1].set_title("LSTM — Loss per Epoch (Linear Scale)")
    axes[1].legend()
    axes[1].text(0.98, 0.92,
                 f"Min Val Loss: {min(val_loss):.5f}\nat epoch {best_epoch}",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("LSTM — Training & Validation Loss Curve", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path); plt.close(fig)
    print(f"  ✓ {save_path.name}")


def plot_lr_schedule(progress, save_path):
    epochs = [p["epoch"] for p in progress]
    lrs    = [p["lr"]    for p in progress]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, lrs, color=COLOR, lw=1.5, marker="o", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate (log scale)")
    ax.set_title("LSTM — Learning Rate Schedule (ReduceLROnPlateau)")
    ax.text(0.98, 0.92,
            "LR decays when validation loss plateaus.\nFinal LR: {:.2e}".format(lrs[-1]),
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    fig.savefig(save_path); plt.close(fig)
    print(f"  ✓ {save_path.name}")


def plot_train_val_comparison(progress, save_path):
    
    epochs   = [p["epoch"]      for p in progress]
    tr_loss  = [p["train_loss"] for p in progress]
    val_loss = [p["val_loss"]   for p in progress]
    gap      = [v - t for t, v in zip(tr_loss, val_loss)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(epochs, tr_loss,  label="Train", color=COLOR,    lw=1.5)
    axes[0].plot(epochs, val_loss, label="Val",   color="#2E86AB", lw=1.5)
    axes[0].set_ylabel("MSE Loss"); axes[0].legend()
    axes[0].set_title("LSTM — Train vs Val Loss")

    axes[1].fill_between(epochs, gap, alpha=0.4, color="#E1A100", label="Val − Train gap")
    axes[1].axhline(0, color="black", lw=1, ls="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Gap (Val − Train)")
    axes[1].set_title("Overfitting Gap per Epoch")
    axes[1].legend()

    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)
    print(f"  ✓ {save_path.name}")


def plot_pred_vs_actual(y_true, y_pred, split_name, save_path, sample=3000):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), min(sample, len(y_true)), replace=False)
    yt, yp = np.array(y_true)[idx], np.array(y_pred)[idx]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(yt, yp, alpha=0.3, s=10, color=COLOR)
    lims = [min(yt.min(), yp.min()) - 0.02, max(yt.max(), yp.max()) + 0.02]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.set_xlabel("Actual Power (kW/kWp)"); ax.set_ylabel("Predicted Power (kW/kWp)")
    ax.set_title(f"LSTM — Predicted vs Actual ({split_name})")
    ax.legend()
    r2 = r2_score(np.array(y_true), np.array(y_pred))
    ax.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax.transAxes,
            fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    fig.savefig(save_path); plt.close(fig)
    print(f"  ✓ {save_path.name}")


def plot_residuals(y_true, y_pred, save_path):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    residuals = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.2, s=8, color=COLOR)
    axes[0].axhline(0, color="red", lw=1.5, ls="--")
    axes[0].set_xlabel("Predicted (kW/kWp)"); axes[0].set_ylabel("Residual")
    axes[0].set_title("LSTM — Residuals vs Fitted")
    axes[1].hist(residuals, bins=80, color=COLOR, alpha=0.75, edgecolor="none")
    axes[1].axvline(0, color="red", lw=1.5, ls="--")
    axes[1].set_xlabel("Residual (kW/kWp)"); axes[1].set_ylabel("Count")
    axes[1].set_title("LSTM — Residual Distribution")
    axes[1].text(0.98, 0.92, f"μ={residuals.mean():.4f}\nσ={residuals.std():.4f}",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)
    print(f"  ✓ {save_path.name}")


def plot_metrics_summary(log, save_path):
    splits  = ["train", "val", "test"]
    metrics = ["rmse", "mae", "r2"]
    values  = {m: [log["metrics"][s][m] for s in splits] for m in metrics}
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ["#5DADE2", "#F39C12", "#E74C3C"]
    for ax, metric in zip(axes, metrics):
        bars = ax.bar(splits, values[metric], color=colors, width=0.5, edgecolor="white")
        for bar, v in zip(bars, values[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(f"LSTM — {metric.upper()}")
        ax.set_ylabel(metric.upper()); ax.set_ylim(0, max(values[metric]) * 1.15)
    fig.suptitle("LSTM — Train / Val / Test Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)
    print(f"  ✓ {save_path.name}")


def main():
    print("=" * 60)
    print("  LSTM — Generating Plots")
    print("=" * 60)
    log, pred = load_artifacts()

    print("\n[LSTM Plots] Generating all plots...")
    progress   = log["training_progress"]
    best_epoch = log["best_epoch"]

    plot_loss_curve(progress, best_epoch, PLOTS_DIR / "training_loss_curve.png")
    plot_lr_schedule(progress, PLOTS_DIR / "lr_schedule.png")
    plot_train_val_comparison(progress, PLOTS_DIR / "train_val_gap.png")
    plot_pred_vs_actual(pred["y_test"], pred["y_pred_test"], "Test Set",
                        PLOTS_DIR / "predicted_vs_actual_test.png")
    plot_pred_vs_actual(pred["y_val"], pred["y_pred_val"], "Validation Set",
                        PLOTS_DIR / "predicted_vs_actual_val.png")
    plot_residuals(pred["y_test"], pred["y_pred_test"], PLOTS_DIR / "residuals.png")
    plot_metrics_summary(log, PLOTS_DIR / "metrics_summary.png")

    print(f"\n✅ All LSTM plots saved to ml/plots/lstm/")
    m = log["metrics"]["test"]
    print(f"   Test → R²={m['r2']:.4f}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}")
    print(f"   TensorBoard logs: tensorboard --logdir ml/runs/")


if __name__ == "__main__":
    main()
