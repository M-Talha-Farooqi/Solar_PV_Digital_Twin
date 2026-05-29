# Solar PV Digital Twin ☀️

A comprehensive digital twin for a solar photovoltaic (PV) power generation system. This project features a **fully interactive 3D web simulation** that predicts solar power output in real-time. These predictions are driven by three distinct machine learning models (Random Forest, XGBoost, and LSTM) which were trained on real meteorological data fetched directly from the NASA POWER dataset for diverse locations (Dhaka, Lahore, and the Thar Desert). 

It covers an end-to-end lifecycle, ranging from raw data acquisition and physics-based feature engineering to model training and interactive 3D visualization.

---

## 🚀 Key Features

* **Data Acquisition & Pipeline:** Automatically fetches 5.5 years of hourly meteorological data (Irradiance, Temperature, Wind Speed, etc.) from the NASA POWER API.
* **Feature Engineering:** Implements advanced solar physics calculations (e.g., calculating solar zenith angles, clear-sky irradiance models) and handles missing sensor data through time-based interpolation.
* **Exploratory Data Analysis (EDA):** In-depth analysis of weather patterns and solar generation potential across three distinct climates.
* **Machine Learning:** Accurate power generation prediction using multiple ML architectures:
  * **Random Forest:** Baseline ensemble model.
  * **XGBoost:** High-performance gradient boosted trees (with SHAP interpretability).
  * **LSTM:** Deep learning sequence model to capture temporal weather dependencies.

---

## 📂 Project Structure

```text
├── data/
│   ├── raw/             # Raw NASA weather files per city
│   └── processed/       # Cleaned, unified parquet/csv datasets
├── ml/
│   ├── artifacts/       # Saved models weights
│   ├── data_pipeline/   # Scripts to fetch, clean, and standardize data
│   ├── EDA/             # Jupyter notebooks for data visualization
│   ├── plots/           # Evaluation plots (Predicted vs Actual, Residuals, SHAP)
│   ├── runs/            # logs
│   └── training/        # Model training and evaluation scripts (RF, XGB, LSTM)
├── backend/             # FastAPI backend implementation
├── frontend/            # Frontend dashboard
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/M-Talha-Farooqi/Solar_PV_Digital_Twin.git
   cd Solar_PV_Digital_Twin
   ```

2. **Install Git LFS (Required for model files):**
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃‍♂️ Running the Pipeline

You can run individual scripts to re-create the pipeline step-by-step from the project root:

1. **Fetch Raw Data:**
   ```bash
   python ml/data_pipeline/fetch_nasa.py
   ```
2. **Clean & Process Data:**
   ```bash
   python ml/data_pipeline/clean.py
   ```
3. **Train Models:**
   * Random Forest: `python ml/training/random_forest/train_rf.py`
   * XGBoost: `python ml/training/xgboost/train_xgb.py`
   * LSTM: `python ml/training/lstm/train_lstm.py`

---

## 🤝 Contributors

* [M. Talha Farooqi](https://github.com/M-Talha-Farooqi)
* [Shoaib Hamza](https://github.com/theshoaibhamza)
* [Husnain Aslam](https://github.com/heyhusn)
* [Talha Farooq](https://github.com/TalhaFarooq82)
* [Muhammad Talha](https://github.com/MuhammadTalha1404)
