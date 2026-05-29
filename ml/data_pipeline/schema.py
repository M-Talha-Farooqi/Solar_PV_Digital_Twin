from dataclasses import dataclass, field
from typing import ClassVar


@dataclass(frozen=True)
class WeatherColumn:
    name: str
    api_param: str         
    unit: str
    description: str




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




ALL_COLUMNS: list[WeatherColumn] = [
    GHI, DNI, DHI, CLEARSKY_GHI,
    TEMPERATURE, WIND_SPEED, HUMIDITY, CLEARNESS_INDEX,
]


API_TO_COLUMN: dict[str, str] = {c.api_param: c.name for c in ALL_COLUMNS}


API_PARAMS_CSV: str = ",".join(c.api_param for c in ALL_COLUMNS)




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




@dataclass(frozen=True)
class SimulationField:
    name: str
    label: str
    unit: str
    min_val: float
    max_val: float
    default_val: float
    description: str




EDITABLE_SIMULATION_INPUTS: list[SimulationField] = [
    SimulationField("ghi", "Global Horizontal Irradiance", "W/m²", 0.0, 1200.0, 500.0, "Primary driver for power generation"),
    SimulationField("temperature", "Ambient Temperature", "°C", -10.0, 55.0, 25.0, "Affects thermal degradation of PV cell"),
    SimulationField("wind_speed", "Wind Speed", "m/s", 0.0, 30.0, 2.0, "Helps cool the PV cell, increasing efficiency"),
    SimulationField("hour", "Time of Day", "hour", 0.0, 23.0, 12.0, "Determines cyclical hour features (hour_sin/cos)"),
    SimulationField("month", "Month of Year", "month", 1.0, 12.0, 6.0, "Determines cyclical seasonal features"),
    SimulationField("model_type", "Machine Learning Model", "categorical", 0.0, 0.0, 0.0, "Select 'all', 'random_forest', 'xgboost', or 'lstm'"),
]


SIMULATION_LOCATIONS: list[str] = [loc.name for loc in LOCATIONS]



SYNTHESIZED_FEATURES: list[str] = [
    "hour_sin", "hour_cos",                 
    "month_sin", "month_cos",               
    "doy_sin", "doy_cos",                   
    "cell_temperature",                     
    "ghi_lag_1h", "ghi_lag_24h",            
    "ghi_rolling_3h", "ghi_rolling_24h",    
    "clearness_index", "clearsky_ratio",    
    "dni", "dhi", "clearsky_ghi",           
    "humidity",                             
    "loc_dhaka", "loc_lahore", "loc_thar_desert" 
]




@dataclass(frozen=True)
class APIResponseSchema:
    model_used: str
    power_kw_rf: float | None
    power_kw_xgb: float | None
    power_kw_lstm: float | None
    power_kw_theoretical: float
    unit: str = "kW/kWp"

