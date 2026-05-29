

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

PLOTS_DIR = PROJECT_ROOT / "ml" / "plots" / "rf"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
COLOR = "#2E86AB"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 11.5, "axes.grid": True, "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
})


def load_artifacts():
    log_path  = PROJECT_ROOT / "ml" / "artifacts" / "random_forest" / "training_log.json"
    pred_path = PROJECT_ROOT / "ml" / "artifacts" / "predictions" / "rf_preds.json"
    with open(log_path)  as f: log  = json.load(f)
    with open(pred_path) as f: pred = json.load(f)
    return log, pred


def plot_oob_curve(progress, save_path):
    n_trees = [p["n_trees"] for p in progress]
    oob_r2  = [p["oob_r2"]  for p in progress]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_trees, oob_r2, color=COLOR, lw=2, marker="o", markersize=5)
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("OOB R² Score")
    ax.set_title("Random Forest — Out-of-Bag Score vs. Number of Trees")
    ax.set_ylim(min(oob_r2) - 0.001, 1.0005)
    ax.text(0.98, 0.05,
            "OOB score acts as a free internal validation set.\n"
            "Score converges rapidly after ~50 trees.",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="gray", style="italic")
    fig.savefig(save_path); plt.close(fig)
    print(f"  ✓ {save_path.name}")


def plot_feature_importance(fi_list, save_path, top_n=20):
    fi_list = fi_list[:top_n]
    names  = [x["feature"] for x in fi_list][::-1]
    vals   = [x["importance"] for x in fi_list][::-1]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title(f"Random Forest — Top {top_n} Feature Importances")
    ax.text(0.98, 0.02, "GHI and cell_temperature dominate the prediction.",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="gray", style="italic")
    fig.savefig(save_path); plt.close(fig)
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
    ax.set_title(f"Random Forest — Predicted vs Actual ({split_name})")
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
    axes[0].set_title("Random Forest — Residuals vs Fitted")
    axes[1].hist(residuals, bins=80, color=COLOR, alpha=0.75, edgecolor="none")
    axes[1].axvline(0, color="red", lw=1.5, ls="--")
    axes[1].set_xlabel("Residual (kW/kWp)"); axes[1].set_ylabel("Count")
    axes[1].set_title("Random Forest — Residual Distribution")
    axes[1].text(0.98, 0.92, f"μ={residuals.mean():.4f}\nσ={residuals.std():.4f}",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)
    print(f"  ✓ {save_path.name}")


def plot_metrics_summary(log, save_path):
    all_splits = ["train", "val", "test"]
    splits  = [s for s in all_splits if log["metrics"].get(s, {}).get("rmse") is not None]
    metrics = ["rmse", "mae", "r2"]
    values  = {m: [log["metrics"][s][m] for s in splits] for m in metrics}
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ["#5DADE2", "#F39C12", "#E74C3C"][:len(splits)]
    for ax, metric in zip(axes, metrics):
        bars = ax.bar(splits, values[metric], color=colors, width=0.5, edgecolor="white")
        for bar, v in zip(bars, values[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(f"Random Forest — {metric.upper()}")
        ax.set_ylabel(metric.upper()); ax.set_ylim(0, max(values[metric]) * 1.15)
    fig.suptitle("Random Forest — Val / Test Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)
    print(f"  ✓ {save_path.name}")


def main():
    print("=" * 60)
    print("  Random Forest — Generating Plots")
    print("=" * 60)
    log, pred = load_artifacts()

    print("\n[RF Plots] Generating all plots...")
    plot_oob_curve(log["training_progress"],   PLOTS_DIR / "oob_curve.png")
    plot_feature_importance(log["feature_importances"], PLOTS_DIR / "feature_importance.png")
    plot_pred_vs_actual(pred["y_test"], pred["y_pred_test"], "Test Set",
                        PLOTS_DIR / "predicted_vs_actual_test.png")
    plot_pred_vs_actual(pred["y_val"], pred["y_pred_val"], "Validation Set",
                        PLOTS_DIR / "predicted_vs_actual_val.png")
    plot_residuals(pred["y_test"], pred["y_pred_test"], PLOTS_DIR / "residuals.png")
    plot_metrics_summary(log, PLOTS_DIR / "metrics_summary.png")

    print(f"\n✅ All RF plots saved to ml/plots/rf/")
    m = log["metrics"]["test"]
    print(f"   Test → R²={m['r2']:.4f}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}")


if __name__ == "__main__":
    main()
