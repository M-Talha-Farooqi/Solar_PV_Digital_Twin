import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data_pipeline.schema import ALL_COLUMNS, LOCATIONS, Location

RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MAX_INTERPOLATE_GAP = 4

DROP_LONG_GAPS = True


def load_raw(loc: Location) -> pd.DataFrame:
    path = RAW_DIR / f"{loc.name}.csv"
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    return df


def clean_location(loc: Location) -> pd.DataFrame:
    print(f"\n{'─'*50}")
    print(f"Cleaning: {loc.name}")
    print(f"{'─'*50}")

    df = load_raw(loc)
    original_rows = len(df)

    # 1 ── Replace NASA missing-value sentinel (-999) with NaN ────────────
    missing_before = (df == -999).sum()
    df = df.replace(-999, np.nan)
    df = df.replace(-999.0, np.nan)

    total_missing = missing_before.sum()
    print(f"  Missing-value cells (-999): {total_missing}")
    for col_name, count in missing_before.items():
        if count > 0:
            pct = count / original_rows * 100
            print(f"    {col_name}: {count} ({pct:.1f}%)")

    # 2 ── Interpolate short gaps (≤ MAX_INTERPOLATE_GAP hours) ───────────
    df_interpolated = df.interpolate(
        method="time",
        limit=MAX_INTERPOLATE_GAP,
    )
    filled = df.isna().sum() - df_interpolated.isna().sum()
    print(f"  Interpolated (≤{MAX_INTERPOLATE_GAP}h gaps): {filled.sum()} cells")
    df = df_interpolated

    remaining_nan = df.isna().sum()
    if remaining_nan.sum() > 0:
        print(f"  Remaining NaN after interpolation: {remaining_nan.sum()} cells")
        if DROP_LONG_GAPS:
            before = len(df)
            df = df.dropna()
            dropped = before - len(df)
            print(f"  Dropped {dropped} rows with long-gap NaN")

    # 4 ── Localize timestamps (UTC → local) ─────────────────────────────
    df.index = df.index.tz_localize("UTC").tz_convert(loc.timezone)
    print(f"  Timezone: UTC → {loc.timezone}")

    # 5 ── Add location column ────────────────────────────────────────────
    df["location"] = loc.name

    print(f"  Final shape: {df.shape}")
    return df


def main():
    print("NASA POWER Data Cleaner")
    print(f"Raw dir:       {RAW_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Processed dir: {PROCESSED_DIR.relative_to(PROJECT_ROOT)}")

    frames = []
    for loc in LOCATIONS:
        df = clean_location(loc)
        frames.append(df)

    # ── Concatenate all locations ────────────────────────────────────────
    combined = pd.concat(frames)
    combined = combined.sort_index()
    print(f"\n{'='*50}")
    print(f"Combined shape: {combined.shape}")
    print(f"Locations: {combined['location'].unique().tolist()}")
    print(f"Date range: {combined.index.min()} → {combined.index.max()}")

    # ── Save to parquet ──────────────────────────────────────────────────
    out_path = PROCESSED_DIR / "weather.parquet"
    combined.to_parquet(out_path, engine="pyarrow")
    print(f"\n Saved processed data → {out_path.relative_to(PROJECT_ROOT)}")

    csv_path = PROCESSED_DIR / "weather.csv"
    combined.to_csv(csv_path)
    print(f"   Also saved CSV copy → {csv_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
