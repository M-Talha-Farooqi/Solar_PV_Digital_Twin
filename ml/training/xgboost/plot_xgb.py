

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

PLOTS_DIR = PROJECT_ROOT / "ml" / "plots" / "xgb"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
COLOR = "#E1A100"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 11.5, "axes.grid": True, "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
})


def load_artifacts():
    log_path  = PROJECT_ROOT / "ml" / "artifacts" / "xgboost" / "training_log.json"
    pred_path = PROJECT_ROOT / "ml" / "artifacts" / "predictions" / "xgb_preds.json"
    with open(log_path)  as f: log  = json.load(f)
    with open(pred_path) as f: pred = json.load(f)
    return log, pred


def plot_training_curve(progress, best_round, save_path):
    rounds      = [p["round"]      for p in progress]
    train_rmse  = [p["train_rmse"] for p in progress]
    val_rmse    = [p["val_rmse"]   for p in progress]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rounds, train_rmse, label="Train RMSE", color=COLOR,    lw=1.5)
    ax.plot(rounds, val_rmse,   label="Val RMSE",   color="#C5453B", lw=1.5)
    ax.axvline(best_round, color="green", ls="--", lw=1.5,
               label=f"Best round ({best_round})")
    ax.set_xlabel("Boosting Round"); ax.set_ylabel("RMSE (kW/kWp)")
    ax.set_title("XGBoost — Train / Validation RMSE per Boosting Round")
    ax.legend()
    ax.text(0.98, 0.92, f"Best Val RMSE: {min(val_rmse):.4f} at round {best_round}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    fig.savefig(save_path); plt.close(fig)
    print(f"  ✓ {save_path.name}")


def plot_feature_importance(fi_list, save_path, top_n=20):
    fi_list = fi_list[:top_n]
    names  = [x["feature"] for x in fi_list][::-1]
    vals   = [x["importance_gain"] for x in fi_list][::-1]
    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(names)))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Average Gain"); ax.set_title(f"XGBoost — Top {top_n} Feature Importances (Gain)")
    ax.text(0.98, 0.02, "Gain = average loss reduction per split.",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="gray", style="italic")
    fig.savefig(save_path); plt.close(fig)
    print(f"  ✓ {save_path.name}")


def plot_shap(log, save_path_prefix):
    try:
        import shap, pickle
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(str(PROJECT_ROOT / "ml" / "artifacts" / "models" / "xgboost.json"))
        pred_path = PROJECT_ROOT / "ml" / "artifacts" / "predictions" / "xgb_preds.json"
        with open(pred_path) as f: pred = json.load(f)
        X_test = np.array(pred["y_test"])  

        
        from ml.training.dataset import get_datasets
        _, _, _, _, X_test_arr, _, feature_cols, _ = get_datasets(scale=False)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_test_arr), min(2000, len(X_test_arr)), replace=False)
        X_sample = X_test_arr[idx]
        dmat     = xgb.DMatrix(X_sample, feature_names=feature_cols)
        explainer    = shap.TreeExplainer(model)
        shap_values  = explainer.shap_values(X_sample)

        
        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols,
                          plot_type="bar", show=False)
        plt.title("XGBoost — SHAP Feature Importance (Mean |SHAP|)")
        plt.tight_layout()
        plt.savefig(str(save_path_prefix) + "_bar.png"); plt.close()
        print(f"  ✓ shap_summary_bar.png")
    except ImportError:
        print("  [SHAP] skipped — pip install shap")
    except Exception as e:
        print(f"  [SHAP] Error: {e}")


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
    ax.set_title(f"XGBoost — Predicted vs Actual ({split_name})")
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
    axes[0].set_title("XGBoost — Residuals vs Fitted")
    axes[1].hist(residuals, bins=80, color=COLOR, alpha=0.75, edgecolor="none")
    axes[1].axvline(0, color="red", lw=1.5, ls="--")
    axes[1].set_xlabel("Residual (kW/kWp)"); axes[1].set_ylabel("Count")
    axes[1].set_title("XGBoost — Residual Distribution")
    axes[1].text(0.98, 0.92, f"μ={residuals.mean():.4f}\nσ={residuals.std():.4f}",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)
    print(f"  ✓ {save_path.name}")


def plot_metrics_summary(log, save_path):
    splits = ["train", "val", "test"]
    metrics = ["rmse", "mae", "r2"]
    values  = {m: [log["metrics"][s][m] for s in splits] for m in metrics}
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ["#5DADE2", "#F39C12", "#E74C3C"]
    for ax, metric in zip(axes, metrics):
        bars = ax.bar(splits, values[metric], color=colors, width=0.5, edgecolor="white")
        for bar, v in zip(bars, values[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(f"XGBoost — {metric.upper()}")
        ax.set_ylabel(metric.upper()); ax.set_ylim(0, max(values[metric]) * 1.15)
    fig.suptitle("XGBoost — Train / Val / Test Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)
    print(f"  ✓ {save_path.name}")


def main():
    print("=" * 60)
    print("  XGBoost — Generating Plots")
    print("=" * 60)
    log, pred = load_artifacts()

    print("\n[XGB Plots] Generating all plots...")
    plot_training_curve(log["training_progress"], log["best_round"],
                        PLOTS_DIR / "training_curve.png")
    plot_feature_importance(log["feature_importances"], PLOTS_DIR / "feature_importance.png")
    plot_shap(log, PLOTS_DIR / "shap_summary")
    plot_pred_vs_actual(pred["y_test"], pred["y_pred_test"], "Test Set",
                        PLOTS_DIR / "predicted_vs_actual_test.png")
    plot_pred_vs_actual(pred["y_val"], pred["y_pred_val"], "Validation Set",
                        PLOTS_DIR / "predicted_vs_actual_val.png")
    plot_residuals(pred["y_test"], pred["y_pred_test"], PLOTS_DIR / "residuals.png")
    plot_metrics_summary(log, PLOTS_DIR / "metrics_summary.png")

    print(f"\n✅ All XGB plots saved to ml/plots/xgb/")
    m = log["metrics"]["test"]
    print(f"   Test → R²={m['r2']:.4f}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}")


if __name__ == "__main__":
    main()
