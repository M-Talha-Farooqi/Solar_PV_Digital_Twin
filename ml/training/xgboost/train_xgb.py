

import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.dataset import get_datasets

ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
LOG_DIR       = ARTIFACTS_DIR / "xgboost"
LOG_DIR.mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "models").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)

CONFIG = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "learning_rate": 0.05, "max_depth": 7, "min_child_weight": 5,
    "subsample": 0.8, "colsample_bytree": 0.8, "gamma": 0.1,
    "reg_alpha": 0.05, "reg_lambda": 1.0, "n_jobs": -1,
    "seed": 42, "verbosity": 0,
}
NUM_BOOST_ROUND = 1000
EARLY_STOPPING  = 50


def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mask = y_true > 0.01
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.sum() > 0 else None
    return {"rmse": round(rmse, 6), "mae": round(mae, 6), "r2": round(r2, 6),
            "mape": round(mape, 4) if mape else None}


def main():
    print("=" * 60)
    print("  XGBoost — Solar PV Power Generation")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, _ = get_datasets(scale=False)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_cols)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=feature_cols)

    evals_result = {}
    print(f"\n[XGB] Training (early stopping after {EARLY_STOPPING} rounds)...")
    t0 = time.perf_counter()
    model = xgb.train(
        CONFIG, dtrain, num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOPPING,
        evals_result=evals_result, verbose_eval=50,
    )
    train_time  = time.perf_counter() - t0
    best_round  = model.best_iteration
    n_rounds    = len(evals_result["train"]["rmse"])
    print(f"  Training time: {train_time:.2f}s  |  Best round: {best_round}  |  Total rounds: {n_rounds}")

    t_inf = time.perf_counter()
    y_pred_test = model.predict(dtest)
    latency_ms  = (time.perf_counter() - t_inf) * 1000 / len(X_test)
    y_pred_val   = model.predict(dval)
    y_pred_train = model.predict(dtrain)

    tm = compute_metrics(y_train, y_pred_train)
    vm = compute_metrics(y_val,   y_pred_val)
    em = compute_metrics(y_test,  y_pred_test)
    print(f"  [Train] R²={tm['r2']:.4f}  RMSE={tm['rmse']:.4f}")
    print(f"  [Val]   R²={vm['r2']:.4f}  RMSE={vm['rmse']:.4f}  MAPE={vm['mape']:.2f}%")
    print(f"  [Test]  R²={em['r2']:.4f}  RMSE={em['rmse']:.4f}  MAPE={em['mape']:.2f}%")

    raw_scores = model.get_score(importance_type="gain")
    fi = []
    for k, v in raw_scores.items():
        try:
            name = feature_cols[int(k[1:])]
        except Exception:
            name = k
        fi.append({"feature": name, "importance_gain": round(float(v), 4)})
    fi.sort(key=lambda x: x["importance_gain"], reverse=True)

    model.save_model(str(ARTIFACTS_DIR / "models" / "xgboost.json"))
    print(f"  ✓ Model saved")

    with open(ARTIFACTS_DIR / "predictions" / "xgb_preds.json", "w") as f:
        json.dump({
            "y_train": y_train.tolist(), "y_pred_train": y_pred_train.tolist(),
            "y_val": y_val.tolist(),     "y_pred_val": y_pred_val.tolist(),
            "y_test": y_test.tolist(),   "y_pred_test": y_pred_test.tolist(),
        }, f)
    print(f"  ✓ Predictions saved")

    training_log = {
        "model": "XGBoost",
        "config": CONFIG,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING,
        "actual_rounds": n_rounds,
        "best_round": best_round,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "dataset_sizes": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
        "training_progress": [
            {"round": r + 1,
             "train_rmse": round(evals_result["train"]["rmse"][r], 6),
             "val_rmse":   round(evals_result["val"]["rmse"][r], 6)}
            for r in range(n_rounds)
        ],
        "feature_importances": fi,
        "train_time_s": round(train_time, 3),
        "inference_latency_ms_per_sample": round(latency_ms, 6),
        "metrics": {"train": tm, "val": vm, "test": em},
    }
    log_path = LOG_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"  ✓ Training log ({n_rounds} rounds) saved")

    print(f"\n✅ XGBoost done!  Test R²={em['r2']:.4f}  RMSE={em['rmse']:.4f}")
    return training_log


if __name__ == "__main__":
    main()
