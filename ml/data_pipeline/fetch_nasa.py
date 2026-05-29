import json
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data_pipeline.schema import API_PARAMS_CSV, API_TO_COLUMN, LOCATIONS, Location

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"



DEFAULT_START = "20210101"
DEFAULT_END   = "20260520"


def _year_chunks(start: str, end: str):
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")

    while s <= e:
        year_end = datetime(s.year, 12, 31)
        chunk_end = min(year_end, e)
        yield s.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")
        s = datetime(s.year + 1, 1, 1)


def fetch(
    lat: float,
    lon: float,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    params = {
        "parameters": API_PARAMS_CSV,
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
    }

    print(f"  → Requesting {start}–{end} for ({lat}, {lon}) …")
    resp = requests.get(BASE_URL, params=params, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    param_block = data["properties"]["parameter"]

    series_dict = {}
    for api_name, hourly_map in param_block.items():
        col = API_TO_COLUMN.get(api_name, api_name)
        series_dict[col] = {k: v for k, v in hourly_map.items()}

    df = pd.DataFrame(series_dict)

    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    df.index.name = "timestamp"

    return df, data 


def fetch_and_save(loc: Location, start: str = DEFAULT_START, end: str = DEFAULT_END):
    print(f"\n{'='*60}")
    print(f"Fetching: {loc.name} ({loc.lat}°N, {loc.lon}°E)")
    print(f"Date range: {start} → {end}")
    print(f"{'='*60}")

    all_dfs = []
    all_jsons = []

    for chunk_start, chunk_end in _year_chunks(start, end):
        df_chunk, raw_json = fetch(loc.lat, loc.lon, chunk_start, chunk_end)
        all_dfs.append(df_chunk)
        all_jsons.append(raw_json)
        print(f"    chunk {chunk_start}–{chunk_end}: {len(df_chunk)} rows")
        time.sleep(3)

    df = pd.concat(all_dfs)
    df = df.sort_index()

    json_path = RAW_DIR / f"{loc.name}.json"
    with open(json_path, "w") as f:
        json.dump(all_jsons, f)
    print(f"  ✓ Saved raw JSON → {json_path.relative_to(PROJECT_ROOT)}")

    csv_path = RAW_DIR / f"{loc.name}.csv"
    df.to_csv(csv_path)
    print(f"Saved raw CSV  → {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Shape: {df.shape}  |  Date range: {df.index.min()} → {df.index.max()}")

    return df


def main():
    print("NASA POWER Data Fetcher")
    print(f"Parameters: {API_PARAMS_CSV}")
    print(f"Date range: {DEFAULT_START} → {DEFAULT_END}")
    print(f"Locations:  {', '.join(l.name for l in LOCATIONS)}")

    for loc in LOCATIONS:
        fetch_and_save(loc)
        time.sleep(5)

    print("\nAll locations fetched successfully!")
    print(f"Raw data saved in: {RAW_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()

