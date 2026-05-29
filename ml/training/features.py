
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


NOCT       = 45.0   
T_REF      = 25.0   
T_COEFF    = -0.004 
P_STC      = 1.0    

FEATURE_COLS = [
    "ghi", "dni", "dhi", "clearsky_ghi", "temperature",
    "wind_speed", "humidity", "clearness_index",
    
    "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "clearsky_ratio", "cell_temperature",
    "ghi_lag_1h", "ghi_lag_24h",
    "ghi_rolling_3h", "ghi_rolling_24h",
    
]

TARGET_COL = "power_kw"


def compute_power(df: pd.DataFrame) -> pd.Series:
    
    ghi = df["ghi"].values.astype(float)
    t_amb = df["temperature"].values.astype(float)

    
    t_cell = t_amb + ghi * (NOCT - 20.0) / 800.0

    
    power = (ghi / 1000.0) * P_STC * (1.0 + T_COEFF * (t_cell - T_REF))
    power = np.clip(power, 0.0, None)  
    return pd.Series(power, index=df.index, name=TARGET_COL)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")

    
    df[TARGET_COL] = compute_power(df)

    
    hour    = df.index.hour
    month   = df.index.month
    doy     = df.index.dayofyear

    df["hour_sin"]  = np.sin(2 * np.pi * hour  / 24.0)
    df["hour_cos"]  = np.cos(2 * np.pi * hour  / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["doy_sin"]   = np.sin(2 * np.pi * doy   / 365.25)
    df["doy_cos"]   = np.cos(2 * np.pi * doy   / 365.25)

    
    df["clearsky_ratio"] = df["ghi"] / (df["clearsky_ghi"] + 1e-6)
    df["clearsky_ratio"] = df["clearsky_ratio"].clip(0.0, 1.5)

    
    df["cell_temperature"] = (
        df["temperature"].values
        + df["ghi"].values * (NOCT - 20.0) / 800.0
    )

    
    df["ghi_lag_1h"]  = df.groupby("location")["ghi"].shift(1)
    df["ghi_lag_24h"] = df.groupby("location")["ghi"].shift(24)

    
    df["ghi_rolling_3h"]  = (
        df.groupby("location")["ghi"]
          .transform(lambda x: x.rolling(3,  min_periods=1).mean())
    )
    df["ghi_rolling_24h"] = (
        df.groupby("location")["ghi"]
          .transform(lambda x: x.rolling(24, min_periods=1).mean())
    )

    
    df["location"] = df["location"].astype(str)
    loc_dummies = pd.get_dummies(df["location"], prefix="loc", dtype=float)
    df = pd.concat([df, loc_dummies], axis=1)

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    
    loc_dummy_cols = [c for c in df.columns if c.startswith("loc_")]
    base = [c for c in FEATURE_COLS if c in df.columns]
    return base + loc_dummy_cols


def engineer_simulation_features(user_inputs: dict) -> np.ndarray:
    
    
    ghi = float(user_inputs.get("ghi", 500.0))
    temp = float(user_inputs.get("temperature", 25.0))
    wind = float(user_inputs.get("wind_speed", 2.0))
    hour = float(user_inputs.get("hour", 12.0))
    month = float(user_inputs.get("month", 6.0))
    location = str(user_inputs.get("location", "lahore")).lower()

    
    
    clearsky_ghi = ghi * 1.05 if ghi > 0 else 0.0
    clearness_index = min(ghi / 1000.0, 0.8) if ghi > 0 else 0.0
    
    direct_ratio = min(clearness_index + 0.1, 0.9)
    dni = ghi * direct_ratio
    dhi = ghi * (1.0 - direct_ratio)
    humidity = 40.0

    
    doy = month * 30.0 - 15.0  
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    doy_sin = np.sin(2 * np.pi * doy / 365.25)
    doy_cos = np.cos(2 * np.pi * doy / 365.25)

    
    clearsky_ratio = min(ghi / (clearsky_ghi + 1e-6), 1.5)
    cell_temperature = temp + ghi * (NOCT - 20.0) / 800.0

    
    ghi_lag_1h = ghi * 0.95        
    ghi_lag_24h = ghi * 0.95       
    ghi_rolling_3h = ghi * 0.85    
    
    ghi_rolling_24h = ghi * 0.35   

    
    loc_dhaka = 1.0 if location == "dhaka" else 0.0
    loc_lahore = 1.0 if location == "lahore" else 0.0
    loc_thar_desert = 1.0 if location == "thar_desert" else 0.0

    
    
    
    
    
    vector = [
        ghi, dni, dhi, clearsky_ghi, temp, wind, humidity, clearness_index,
        hour_sin, hour_cos, month_sin, month_cos, doy_sin, doy_cos,
        clearsky_ratio, cell_temperature,
        ghi_lag_1h, ghi_lag_24h, ghi_rolling_3h, ghi_rolling_24h,
        loc_dhaka, loc_lahore, loc_thar_desert
    ]
    
    return np.array([vector], dtype=np.float32)
