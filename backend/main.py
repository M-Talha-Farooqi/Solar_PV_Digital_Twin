from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import json
import xgboost as xgb
from pathlib import Path
import sys
import uvicorn
import numpy as np
import torch
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.features import engineer_simulation_features
from ml.data_pipeline.schema import APIResponseSchema
from ml.training.lstm.train_lstm import SolarLSTM

app = FastAPI(
    title="Solar PV Prediction API",
    description="Real-time inference API for predicting Solar PV Power Generation (kW/kWp) using NASA Irradiance Data.",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class SimulationRequest(BaseModel):
    ghi: float = Field(default=500.0, description="Global Horizontal Irradiance in W/m² (0-1200)", ge=0.0, le=1400.0)
    temperature: float = Field(default=25.0, description="Ambient Temperature in °C (-10 to 55)", ge=-20.0, le=60.0)
    wind_speed: float = Field(default=2.0, description="Wind Speed in m/s (0-30)", ge=0.0, le=40.0)
    hour: float = Field(default=12.0, description="Time of Day (0-23)", ge=0.0, le=23.9)
    month: float = Field(default=6.0, description="Month of Year (1-12)", ge=1.0, le=12.0)
    location: str = Field(default="lahore", description="Location name (lahore, dhaka, thar_desert)")
    model_type: str = Field(default="all", description="Choose which model to run inference on: 'random_forest', 'xgboost', 'lstm' or 'all'.")


models = {}

@app.on_event("startup")
def load_ml_assets():
    
    try:
        
        scaler_path = PROJECT_ROOT / "ml/artifacts/scalers/scaler.pkl"
        with open(scaler_path, "rb") as f:
            models["scaler"] = pickle.load(f)
            
        
        rf_path = PROJECT_ROOT / "ml/artifacts/models/random_forest.pkl"
        with open(rf_path, "rb") as f:
            models["rf"] = pickle.load(f)
            
        
        xgb_path = PROJECT_ROOT / "ml/artifacts/models/xgboost.json"
        bst = xgb.Booster()
        bst.load_model(str(xgb_path))
        models["xgb"] = bst
        
        
        log_path = PROJECT_ROOT / "ml/artifacts/xgboost/training_log.json"
        with open(log_path) as f:
            xgb_log = json.load(f)
            models["xgb_feature_names"] = xgb_log.get("feature_cols", None)
            
        try:
            lstm_log_path = PROJECT_ROOT / "ml/artifacts/lstm/training_log.json"
            with open(lstm_log_path) as f:
                lstm_params = json.load(f)["config"]
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            lstm_model = SolarLSTM(
                input_size=lstm_params["input_size"],
                hidden_size=lstm_params["hidden_size"],
                num_layers=lstm_params["num_layers"],
                dropout=lstm_params["dropout"]
            ).to(device)
            
            lstm_path = PROJECT_ROOT / "ml/artifacts/models/lstm.pt"
            checkpoint = torch.load(lstm_path, map_location=device)
            lstm_model.load_state_dict(checkpoint["state_dict"])
            lstm_model.eval()
            models["lstm"] = lstm_model
            models["device"] = device
        except Exception as e:
            print(f"⚠️ LSTM Model not loaded (needs retraining): {e}")
            models["lstm"] = None
            
        print(" ML Models and Scaler loaded successfully.")
    except Exception as e:
        print(f" Error loading ML assets: {e}")
        models["error"] = str(e)



from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def read_root():
    
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": "scaler" in models and "rf" in models}

@app.post("/predict")
def predict_power(req: SimulationRequest) -> APIResponseSchema:
    if "error" in models:
        raise HTTPException(status_code=500, detail=f"ML Assets failed to load: {models['error']}")
        
    try:
        user_inputs = req.dict()
        feature_vector = engineer_simulation_features(user_inputs)
        scaled_features = models["scaler"].transform(feature_vector)
        
        model_selection = req.model_type.lower()
        
        NOCT = 45.0
        t_cell = req.temperature + req.ghi * (NOCT - 20.0) / 800.0
        theoretical_power = (req.ghi / 1000.0) * 1.0 * (1.0 + -0.004 * (t_cell - 25.0))
        theoretical_power = max(0.0, round(theoretical_power, 4))
        
        power_rf = None
        power_xgb = None
        power_lstm = None
        
        
        if model_selection in ["all", "random_forest", "rf"]:
            rf_pred = float(models["rf"].predict(scaled_features)[0])
            power_rf = round(max(0.0, rf_pred), 4)
            
        
        if model_selection in ["all", "xgboost", "xgb"]:
            dtest = xgb.DMatrix(scaled_features, feature_names=models["xgb_feature_names"])
            xgb_pred = float(models["xgb"].predict(dtest)[0])
            power_xgb = round(max(0.0, xgb_pred), 4)
            
        
        if model_selection in ["all", "lstm"] and models.get("lstm") is not None:
            seq_features = np.tile(scaled_features, (1, 24, 1))
            seq_tensor = torch.tensor(seq_features, dtype=torch.float32).to(models["device"])
            with torch.no_grad():
                lstm_pred = models["lstm"](seq_tensor).item()
            power_lstm = round(max(0.0, lstm_pred), 4)
            
        return APIResponseSchema(
            model_used=model_selection,
            power_kw_rf=power_rf,
            power_kw_xgb=power_xgb,
            power_kw_lstm=power_lstm,
            power_kw_theoretical=theoretical_power
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
