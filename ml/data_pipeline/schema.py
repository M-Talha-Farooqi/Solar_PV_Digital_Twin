from dataclasses import dataclass, field
from typing import ClassVar


@dataclass(frozen=True)
class WeatherColumn:
    name: str
    api_param: str         
    unit: str
    description: str


# ── Canonical columns ────────────────────────────────────────────────────────

GHI = WeatherColumn(
    name="ghi",
    api_param="ALLSKY_SFC_SW_DWN",
    unit="W/m²",
    description="Global Horizontal Irradiance (all-sky)",
)

DNI = WeatherColumn(
    name="dni",
    api_param="ALLSKY_SFC_SW_DNI",
    unit="W/m²",
    description="Direct Normal Irradiance (all-sky)",
)

DHI = WeatherColumn(
    name="dhi",
    api_param="ALLSKY_SFC_SW_DIFF",
    unit="W/m²",
    description="Diffuse Horizontal Irradiance (all-sky)",
)

CLEARSKY_GHI = WeatherColumn(
    name="clearsky_ghi",
    api_param="CLRSKY_SFC_SW_DWN",
    unit="W/m²",
    description="Clear-sky Global Horizontal Irradiance",
)

TEMPERATURE = WeatherColumn(
    name="temperature",
    api_param="T2M",
    unit="°C",
    description="Air temperature at 2 m",
)

WIND_SPEED = WeatherColumn(
    name="wind_speed",
    api_param="WS10M",
    unit="m/s",
    description="Wind speed at 10 m",
)

HUMIDITY = WeatherColumn(
    name="humidity",
    api_param="RH2M",
    unit="%",
    description="Relative humidity at 2 m",
)

CLEARNESS_INDEX = WeatherColumn(
    name="clearness_index",
    api_param="ALLSKY_KT",
    unit="dimensionless",
    description="Clearness index (Kt)",
)


# ── Convenience collections ──────────────────────────────────────────────────

ALL_COLUMNS: list[WeatherColumn] = [
    GHI, DNI, DHI, CLEARSKY_GHI,
    TEMPERATURE, WIND_SPEED, HUMIDITY, CLEARNESS_INDEX,
]

# Map from NASA API parameter name → our canonical column name
API_TO_COLUMN: dict[str, str] = {c.api_param: c.name for c in ALL_COLUMNS}

# Comma-separated string ready for the API URL
API_PARAMS_CSV: str = ",".join(c.api_param for c in ALL_COLUMNS)


# ── Target locations ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Location:
    name: str
    lat: float
    lon: float
    timezone: str          

LOCATIONS: list[Location] = [
    Location(name="lahore",     lat=31.52, lon=74.36, timezone="Asia/Karachi"),
    Location(name="thar_desert", lat=25.50, lon=69.85, timezone="Asia/Karachi"),   
    Location(name="dhaka",      lat=23.81, lon=90.41, timezone="Asia/Dhaka"),    
]
