# 🌦️ PMACC Weather Data Science Project

## 📌 Overview
This project analyzes **global weather data** to build forecasting baselines, detect anomalies, and extract feature insights.  
It was completed as part of the **PM Accelerator Data Science Technical Assessment**.  

The pipeline covers:  
1. **Data Cleaning & EDA**  
2. **Univariate Baseline Forecast**  
3. **Multivariate Regression Models**  
4. **Anomalies, Spatial Analysis, & Feature Importance**

---

## 📂 Dataset
**Source:** `GlobalWeatherRepository.csv` (88,468 rows × 41 columns)  

**Key columns:**  
- 🌍 Location: `country`, `location_name`, `latitude`, `longitude`, `timezone`  
- ⏰ Time: `last_updated` (standardized time index)  
- 🌡️ Weather metrics: `temperature_celsius`, `feels_like_celsius`, `humidity`, `pressure_mb`, `wind_kph`, `precip_mm`, `cloud`, `uv_index`  
- 🌫️ Air quality: `air_quality_PM2.5`, `air_quality_PM10`, `Ozone`, `Carbon_Monoxide`, etc.  
- 🌞 Solar/Lunar: `sunrise`, `sunset`, `moon_phase`, `moon_illumination`

---

## 🧑‍💻 Notebooks

### 01 – Data Loading & EDA
**Goal:** Load raw data, clean, standardize, and explore key weather features.  
**Steps:**  
- Removed duplicates, standardized time index (`last_updated`)  
- Filtered representative cities (**London, New York, Tokyo**)  
- Handled missing values (per-city imputation)  
- Clipped outliers using **IQR**  
- Added time features (`dow`, `sin_doy`, `cos_doy`)  
- Generated EDA plots: seasonal trends, histograms, correlation heatmap  

**Output:**  
- `assets/clean_weather.csv` & `assets/clean_weather.parquet`  
- EDA plots (`temperature_ALL_CITIES.png`, `corr_heatmap.png`, etc.)  

---

### 02 – Model Baseline (Univariate)
**Goal:** Establish a simple **performance floor** using lag-1 prediction.  

**Models:**  
- **Naive (lag-1 persistence)**: `y_hat_t = y_{t-1}`  
- **Linear Regression (lag-1)**  

**Metrics:** RMSE, MAE, MAPE, R²  

| Model                   | RMSE  | MAE  | MAPE  | R²    |  
|--------------------------|-------|------|-------|-------|  
| Naive (lag-1)            | ~9.8  | ~8.0 | ~40%  | -0.12 |  
| Linear Regression (lag-1)| ~9.1  | ~7.4 | ~37%  | -0.05 |  

📌 Both baselines underperform a simple mean predictor (negative R²).  

**Next:** Add more lags, seasonal features, and other weather variables.  

---

### 03 – Multivariate Models
**Goal:** Improve predictive power with multiple lags + features.  

**Features:**  
- Lag values: `[1, 2, 3, 7, 14]`  
- Rolling means: `[3, 7]`  
- Seasonal encodings: `sin_doy`, `cos_doy`, day-of-week  
- Weather variables: humidity, pressure, wind, cloud, UV  

**Models:**  
- Ridge Regression (linear, regularized)  
- Random Forest (non-linear, ensemble)  
- Simple Ensemble (average of top 2 models)  

**Results:**  

| Model                 | RMSE | MAE | MAPE  | R²   |  
|------------------------|------|-----|-------|------|  
| Ridge                 | 2.6  | 2.1 | 9.4%  | 0.84 |  
| Random Forest         | 2.4  | 1.9 | 8.6%  | 0.86 |  
| Ensemble (Ridge+RF)   | 2.3  | 1.8 | 8.4%  | 0.86 |  

📌 RMSE improved ~75% compared to univariate. Predictions are within ~2°C.  

---

### 04 – Anomalies, Spatial Analysis & Feature Importance
**Goal:** Assess data quality, location patterns, and model interpretability.  

**Anomalies:**  
- **Univariate** (per feature)  
- **Multivariate** (Isolation Forest)  

Example output:  

| Country | Location | n  | Mean Temp (°C) | Std Dev | Uni. Anom. Rate | Multi. Anom. Rate |  
|---------|----------|----|----------------|---------|-----------------|-------------------|  
| Japan   | Tokyo    | 90 | 27.93          | 4.72    | 0.53            | 0.37              |  
| UK      | London   | 89 | 18.52          | 3.23    | 0.45            | 0.53              |  

**Spatial Insights:**  
- Tokyo → hotter baseline, frequent univariate anomalies  
- London → lower temps but high irregularity  
- New York → intermediate stability  

**Feature Importance (Random Forest):**  
- `lag2`, `lag14` strongest predictors  
- Humidity, pressure also contribute  
- Seasonal (`sin_doy`) confirms cyclical effects  

---

## 📊 Final Insights
- Baseline models showed weak predictive power — fixed with **multivariate regression**.  
- Ensemble learning achieved strong results (**R² ≈ 0.86**).  
- Anomaly detection revealed city-specific weather volatility.  
- Feature importance confirmed domain intuition (lags + humidity + pressure drive forecasts).  

---


⚙️ Tech Stack

- **Language:** Python 3.x  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib  
- **Modeling & ML:**  
  - Scikit-learn (Linear Regression, Ridge, RandomForest, IsolationForest, Pipelines, Preprocessing)  
  - Ensemble methods (manual averaging of models)  
- **Feature Engineering:** Custom functions (`lag features`, `rolling stats`, `seasonal encodings`) via `src.features`  
- **Environment:** Jupyter Notebook, VSCode (for modularized src folder)  
