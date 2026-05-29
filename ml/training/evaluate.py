

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import xgboost as xgb
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.dataset import get_datasets, make_sequences
from ml.training.lstm.train_lstm import SolarLSTM


ARTIFACTS_DIR  = PROJECT_ROOT / "ml" / "artifacts"
COMP_PLOTS_DIR = PROJECT_ROOT / "ml" / "plots" / "comparison"
COMP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


COLORS = {
    "Random Forest": "#2E86AB",
    "XGBoost":       "#E1A100",
    "LSTM":          "#C5453B",
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 11.5, "axes.grid": True, "grid.alpha": 0.35,
    "font.family": "DejaVu Sans",
})



def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mask = y_true > 0.01
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)           if mask.sum() > 0 else None
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}



def bar_chart(names, values, metric_name, unit, save_path, higher_is_better=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = [COLORS[n] for n in names]
    bars = ax.bar(names, values, color=bar_colors, width=0.5, edgecolor="white", linewidth=1.5)

    
    best_idx = int(np.argmax(values) if higher_is_better else np.argmin(values))
    for i, (bar, v) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold" if i == best_idx else "normal")
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)

    direction = "↑ higher is better" if higher_is_better else "↓ lower is better"
    ax.set_ylabel(f"{metric_name} ({unit})  [{direction}]")
    ax.set_title(f"Model Comparison — {metric_name}")
    ax.set_ylim(0, max(values) * 1.18)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path.name}")


def plot_predicted_vs_actual_all(predictions: dict, save_path, sample=2000):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (model_name, (y_true, y_pred)) in zip(axes, predictions.items()):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_true), min(sample, len(y_true)), replace=False)
        yt, yp = y_true[idx], y_pred[idx]
        ax.scatter(yt, yp, alpha=0.25, s=8, color=COLORS[model_name])
        lims = [min(yt.min(), yp.min()) - 0.02, max(yt.max(), yp.max()) + 0.02]
        ax.plot(lims, lims, "r--", lw=1.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual Power (kW/kWp)")
        ax.set_ylabel("Predicted Power (kW/kWp)")
        r2 = r2_score(y_true, y_pred)
        ax.set_title(f"{model_name}\nR² = {r2:.4f}")
        ax.set_aspect("equal")
    fig.suptitle("Predicted vs Actual — All Models (Test Set)", fontsize=15,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path.name}")


def plot_residuals_all(predictions: dict, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for col, (model_name, (y_true, y_pred)) in enumerate(predictions.items()):
        residuals = y_pred - y_true
        c = COLORS[model_name]

        
        axes[0, col].scatter(y_pred, residuals, alpha=0.2, s=6, color=c)
        axes[0, col].axhline(0, color="red", lw=1.5, ls="--")
        axes[0, col].set_xlabel("Predicted (kW/kWp)")
        axes[0, col].set_ylabel("Residual")
        axes[0, col].set_title(f"{model_name} — Residuals vs Fitted")

        
        axes[1, col].hist(residuals, bins=60, color=c, alpha=0.75, edgecolor="none")
        axes[1, col].axvline(0, color="red", lw=1.5, ls="--")
        axes[1, col].set_xlabel("Residual (kW/kWp)")
        axes[1, col].set_ylabel("Count")
        axes[1, col].set_title(f"{model_name} — Residual Distribution")
        axes[1, col].text(0.97, 0.92,
                          f"μ={residuals.mean():.4f}\nσ={residuals.std():.4f}",
                          transform=axes[1, col].transAxes, ha="right", va="top",
                          fontsize=9,
                          bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Residual Analysis — All Models (Test Set)", fontsize=15,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path.name}")


def plot_error_distribution(predictions: dict, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    for model_name, (y_true, y_pred) in predictions.items():
        residuals = y_pred - y_true
        ax.hist(residuals, bins=100, alpha=0.55, label=model_name,
                color=COLORS[model_name], edgecolor="none", density=True)
    ax.axvline(0, color="black", lw=1.5, ls="--")
    ax.set_xlabel("Residual (kW per kWp)")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution Comparison — All Models")
    ax.legend()
    ax.text(0.98, 0.92,
            "Narrower = more consistent predictions.\n"
            "Centred at 0 = unbiased model.",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color="gray", style="italic")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path.name}")


def plot_radar(metrics_dict: dict, save_path):
    
    model_names = list(metrics_dict.keys())
    metric_keys = ["r2", "neg_rmse", "neg_mae", "neg_mape"]
    metric_labels = ["R²", "−RMSE", "−MAE", "−MAPE"]

    
    raw = {}
    for name in model_names:
        m = metrics_dict[name]
        raw[name] = [
            m["r2"],
            -m["rmse"],
            -m["mae"],
            -m["mape"] if m["mape"] is not None else 0,
        ]

    vals_arr = np.array([raw[n] for n in model_names], dtype=float)
    col_min  = vals_arr.min(axis=0)
    col_max  = vals_arr.max(axis=0)
    norm = np.where(col_max - col_min > 1e-9,
                    (vals_arr - col_min) / (col_max - col_min),
                    0.5)

    N = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, name in enumerate(model_names):
        values = norm[i].tolist() + [norm[i][0]]
        ax.plot(angles, values, color=COLORS[name], lw=2, label=name)
        ax.fill(angles, values, color=COLORS[name], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_title("Model Comparison — Radar Chart\n(normalised, higher = better)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path.name}")


def plot_metrics_table(all_results: dict, save_path):
    names   = list(all_results.keys())
    headers = ["Model", "RMSE↓", "MAE↓", "R²↑", "MAPE↓", "Train Time (s)", "Latency (ms/sample)"]
    rows    = []
    for name, r in all_results.items():
        m = r["metrics"]["test"]
        rows.append([
            name,
            f"{m['rmse']:.4f}",
            f"{m['mae']:.4f}",
            f"{m['r2']:.4f}",
            f"{m['mape']:.2f}%" if m['mape'] else "N/A",
            f"{r.get('train_time_s', 'N/A')}",
            f"{r.get('inference_latency_ms_per_sample', 'N/A')}",
        ])

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.0)

    
    for j in range(len(headers)):
        tbl[0, j].set_facecolor("#1B262C")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    
    row_colors = ["#EBF5FB", "#FEF9E7", "#FDEDEC"]
    for i, color in enumerate(row_colors[:len(rows)]):
        for j in range(len(headers)):
            tbl[i + 1, j].set_facecolor(color)

    ax.set_title("Unified Model Performance Summary — Test Set",
                 fontsize=14, fontweight="bold", pad=30)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path.name}")



def main():
    print("=" * 60)
    print("  Unified Evaluation — All Models")
    print("=" * 60)

    
    print("\n[Eval] Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, _ =        get_datasets(scale=False)

    
    X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s, _, _ =        get_datasets(scale=True)
    X_te_seq, y_te_seq = make_sequences(X_test_s, y_test_s, 24)

    predictions = {}   
    all_results = {}

    
    print("\n[Eval] Loading Random Forest...")
    rf_model_path = ARTIFACTS_DIR / "models" / "random_forest.pkl"
    if rf_model_path.exists():
        with open(rf_model_path, "rb") as f:
            rf = pickle.load(f)
        y_pred_rf = rf.predict(X_test).astype(np.float32)
        predictions["Random Forest"] = (y_test, y_pred_rf)

        rf_results_path = ARTIFACTS_DIR / "random_forest" / "training_log.json"
        if rf_results_path.exists():
            with open(rf_results_path) as f:
                all_results["Random Forest"] = json.load(f)
        else:
            m = compute_metrics(y_test, y_pred_rf)
            all_results["Random Forest"] = {"test": m, "train_time_s": "N/A",
                                            "inference_latency_ms_per_sample": "N/A"}
        print(f"  R²={all_results['Random Forest']['metrics']['test']['r2']:.4f}")
    else:
        print("  [!] random_forest.pkl not found — run train_rf.py first")

    
    print("\n[Eval] Loading XGBoost...")
    xgb_model_path = ARTIFACTS_DIR / "models" / "xgboost.json"
    if xgb_model_path.exists():
        xgb_model = xgb.Booster()
        xgb_model.load_model(str(xgb_model_path))
        dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
        y_pred_xgb = xgb_model.predict(dtest).astype(np.float32)
        predictions["XGBoost"] = (y_test, y_pred_xgb)

        xgb_results_path = ARTIFACTS_DIR / "xgboost" / "training_log.json"
        if xgb_results_path.exists():
            with open(xgb_results_path) as f:
                all_results["XGBoost"] = json.load(f)
        else:
            m = compute_metrics(y_test, y_pred_xgb)
            all_results["XGBoost"] = {"test": m, "train_time_s": "N/A",
                                      "inference_latency_ms_per_sample": "N/A"}
        print(f"  R²={all_results['XGBoost']['metrics']['test']['r2']:.4f}")
    else:
        print("  [!] xgboost.json not found — run train_xgb.py first")

    
    print("\n[Eval] Loading LSTM...")
    lstm_results_path = ARTIFACTS_DIR / "lstm" / "training_log.json"
    lstm_preds_path   = ARTIFACTS_DIR / "predictions" / "lstm_preds.json"
    if lstm_results_path.exists() and lstm_preds_path.exists():
        with open(lstm_preds_path) as f:
            lstm_preds = json.load(f)
        with open(lstm_results_path) as f:
            all_results["LSTM"] = json.load(f)
        y_te_seq = np.array(lstm_preds["y_test"])
        y_pred_lstm = np.array(lstm_preds["y_pred_test"])
        predictions["LSTM"] = (y_te_seq, y_pred_lstm)
        print(f"  R²={all_results['LSTM']['metrics']['test']['r2']:.4f}")
    else:
        print("  [!] LSTM predictions/logs not found")

    if not predictions:
        print("\n[!] No models loaded. Train at least one model first.")
        return

    
    metrics_out = {}
    for name, r in all_results.items():
        metrics_out[name] = {
            "test":  r["metrics"]["test"],
            "val":   r.get("metrics", {}).get("val", {}),
            "train_time_s": r.get("train_time_s"),
            "inference_latency_ms_per_sample": r.get("inference_latency_ms_per_sample"),
        }

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n  ✓ metrics.json → {metrics_path.relative_to(PROJECT_ROOT)}")

    
    print("\n[Eval] Generating comparison plots...")
    names = list(predictions.keys())

    rmse_vals  = [all_results[n]["metrics"]["test"]["rmse"]  for n in names]
    mae_vals   = [all_results[n]["metrics"]["test"]["mae"]   for n in names]
    r2_vals    = [all_results[n]["metrics"]["test"]["r2"]    for n in names]
    mape_vals  = [all_results[n]["metrics"]["test"]["mape"] or 0 for n in names]
    lat_vals   = [float(all_results[n].get("inference_latency_ms_per_sample", 0) or 0) for n in names]
    time_vals  = [float(all_results[n].get("train_time_s", 0) or 0) for n in names]

    bar_chart(names, rmse_vals,  "RMSE",      "kW/kWp", COMP_PLOTS_DIR / "rmse_bar.png")
    bar_chart(names, mae_vals,   "MAE",       "kW/kWp", COMP_PLOTS_DIR / "mae_bar.png")
    bar_chart(names, r2_vals,    "R² Score",  "",       COMP_PLOTS_DIR / "r2_bar.png", higher_is_better=True)
    bar_chart(names, mape_vals,  "MAPE",      "%",      COMP_PLOTS_DIR / "mape_bar.png")
    bar_chart(names, lat_vals,   "Inference Latency", "ms/sample", COMP_PLOTS_DIR / "latency_bar.png")
    bar_chart(names, time_vals,  "Training Time",     "s",         COMP_PLOTS_DIR / "training_time_bar.png")

    plot_predicted_vs_actual_all(predictions, COMP_PLOTS_DIR / "predicted_vs_actual_all.png")
    plot_residuals_all(predictions, COMP_PLOTS_DIR / "residuals_all.png")
    plot_error_distribution(predictions, COMP_PLOTS_DIR / "error_distribution_all.png")
    plot_metrics_table(all_results, COMP_PLOTS_DIR / "metrics_table.png")

    test_metrics_only = {n: all_results[n]["metrics"]["test"] for n in names}
    plot_radar(test_metrics_only, COMP_PLOTS_DIR / "radar_chart.png")

    
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<18} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE':>8}")
    print(f"  {'-'*52}")
    for name in names:
        m = all_results[name]["metrics"]["test"]
        print(f"  {name:<18} {m['rmse']:>8.4f} {m['mae']:>8.4f} "
              f"{m['r2']:>8.4f} {(m['mape'] or 0):>7.2f}%")
    print(f"\n✅ Evaluation complete. All plots → ml/plots/comparison/")


if __name__ == "__main__":
    main()
