

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.dataset import get_datasets


ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
LOG_DIR       = ARTIFACTS_DIR / "random_forest"
LOG_DIR.mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "models").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)


CONFIG = {
    "n_estimators":    200,
    "max_depth":       None,
    "min_samples_leaf": 4,
    "max_features":    "sqrt",
    "n_jobs":          -1,
    "oob_score":       True,
    "random_state":    42,
}

OOB_CURVE_RANGE = list(range(10, 210, 10))   


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mask = y_true > 0.01
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)           if mask.sum() > 0 else None
    return {"rmse": round(rmse, 6), "mae": round(mae, 6),
            "r2": round(r2, 6), "mape": round(mape, 4) if mape else None}


def main():
    print("=" * 60)
    print("  Random Forest — Solar PV Power Generation")
    print("=" * 60)

    
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, _ =        get_datasets(scale=False)

    
    print(f"\n[RF] Building OOB curve ({OOB_CURVE_RANGE[0]}–{OOB_CURVE_RANGE[-1]} trees)...")
    oob_progress = []
    for n in OOB_CURVE_RANGE:
        m = RandomForestRegressor(n_estimators=n, oob_score=True,
                                  n_jobs=-1, random_state=42, max_features="sqrt")
        m.fit(X_train, y_train)
        oob_r2 = float(m.oob_score_)
        oob_progress.append({"n_trees": n, "oob_r2": round(oob_r2, 6)})
        print(f"   n_trees={n:3d}  OOB R²={oob_r2:.6f}")

    
    print(f"\n[RF] Training final model ({CONFIG['n_estimators']} trees)...")
    t0 = time.perf_counter()
    rf  = RandomForestRegressor(**CONFIG)
    rf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"  Training time: {train_time:.2f}s  |  OOB R²: {rf.oob_score_:.6f}")

    
    t_inf = time.perf_counter()
    y_pred_test = rf.predict(X_test)
    latency_ms  = (time.perf_counter() - t_inf) * 1000 / len(X_test)

    y_pred_val   = rf.predict(X_val)
    y_pred_train = rf.predict(X_train)

    val_metrics  = compute_metrics(y_val,   y_pred_val)
    test_metrics = compute_metrics(y_test,  y_pred_test)
    train_metrics = compute_metrics(y_train, y_pred_train)

    print(f"\n  [Train] R²={train_metrics['r2']:.4f}  RMSE={train_metrics['rmse']:.4f}")
    print(f"  [Val]   R²={val_metrics['r2']:.4f}  RMSE={val_metrics['rmse']:.4f}  MAPE={val_metrics['mape']:.2f}%")
    print(f"  [Test]  R²={test_metrics['r2']:.4f}  RMSE={test_metrics['rmse']:.4f}  MAPE={test_metrics['mape']:.2f}%")

    
    fi = [{"feature": feature_cols[i], "importance": round(float(v), 8)}
          for i, v in enumerate(rf.feature_importances_)]
    fi.sort(key=lambda x: x["importance"], reverse=True)

    
    model_path = ARTIFACTS_DIR / "models" / "random_forest.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
    print(f"  ✓ Model → {model_path.relative_to(PROJECT_ROOT)}")

    
    preds_path = ARTIFACTS_DIR / "predictions" / "rf_preds.json"
    with open(preds_path, "w") as f:
        json.dump({
            "y_train":      y_train.tolist(),
            "y_pred_train": y_pred_train.tolist(),
            "y_val":        y_val.tolist(),
            "y_pred_val":   y_pred_val.tolist(),
            "y_test":       y_test.tolist(),
            "y_pred_test":  y_pred_test.tolist(),
        }, f)
    print(f"  ✓ Predictions → {preds_path.relative_to(PROJECT_ROOT)}")

    
    training_log = {
        "model":       "RandomForest",
        "config":      CONFIG,
        "feature_cols": feature_cols,
        "n_features":  len(feature_cols),
        "dataset_sizes": {
            "train": int(len(X_train)),
            "val":   int(len(X_val)),
            "test":  int(len(X_test)),
        },
        "training_progress": oob_progress,   
        "final_oob_r2": round(float(rf.oob_score_), 6),
        "feature_importances": fi,
        "train_time_s": round(train_time, 3),
        "inference_latency_ms_per_sample": round(latency_ms, 6),
        "metrics": {
            "train": train_metrics,
            "val":   val_metrics,
            "test":  test_metrics,
        },
    }
    log_path = LOG_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"  ✓ Training log → {log_path.relative_to(PROJECT_ROOT)}")

    print(f"\n✅ Random Forest done!  Test R²={test_metrics['r2']:.4f}  RMSE={test_metrics['rmse']:.4f}")
    return training_log


if __name__ == "__main__":
    main()
