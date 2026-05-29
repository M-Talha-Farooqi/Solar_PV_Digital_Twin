
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.features import engineer_features, get_feature_cols, TARGET_COL

DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "weather.parquet"
SCALER_PATH  = PROJECT_ROOT / "ml" / "artifacts" / "scalers" / "scaler.pkl"

TRAIN_END = "2023-12-31 23:59:59"
VAL_END   = "2024-12-31 23:59:59"


def load_and_prepare() -> pd.DataFrame:
    print(f"[dataset] Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)

    df["location"] = df["location"].astype(str)

    print(f"[dataset] Raw shape: {df.shape}  |  "
          f"Range: {df.index.min()} → {df.index.max()}")

    df = engineer_features(df)

    before = len(df)
    df = df.dropna()
    print(f"[dataset] After feature engineering & NaN drop: {len(df)} rows "
          f"(removed {before - len(df)})")

    return df


def time_split(df: pd.DataFrame):
    idx = df.index.tz_convert("UTC").tz_localize(None)

    train_mask = idx <= TRAIN_END
    val_mask   = (idx > TRAIN_END) & (idx <= VAL_END)
    test_mask  = idx > VAL_END

    train = df[train_mask]
    val   = df[val_mask]
    test  = df[test_mask]

    print(f"[dataset] Split sizes  —  "
          f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    return train, val, test


def build_arrays(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    return X, y


def get_datasets(scale: bool = True):
    df = load_and_prepare()
    feature_cols = get_feature_cols(df)

    train, val, test = time_split(df)

    X_train, y_train = build_arrays(train, feature_cols)
    X_val,   y_val   = build_arrays(val,   feature_cols)
    X_test,  y_test  = build_arrays(test,  feature_cols)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)

        SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        print(f"[dataset] Scaler saved → {SCALER_PATH.relative_to(PROJECT_ROOT)}")

    print(f"[dataset] Feature count: {len(feature_cols)}")
    print(f"[dataset] Features: {feature_cols}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, scaler

def make_sequences(X: np.ndarray, y: np.ndarray, window: int = 24):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


if __name__ == "__main__":
    X_tr, y_tr, X_v, y_v, X_te, y_te, cols, sc = get_datasets()
    print("\n✅ Dataset pipeline verified.")
    print(f"   X_train: {X_tr.shape}  y_train: {y_tr.shape}")
    print(f"   X_val:   {X_v.shape}   y_val:   {y_v.shape}")
    print(f"   X_test:  {X_te.shape}  y_test:  {y_te.shape}")
