# 🌦️ PMACC Weather Data Science – Weather Trend Report  

## 1. Executive Summary  
This report highlights temperature and weather trends derived from global datasets.  
The analysis progressed from baseline models to advanced ensemble methods, achieving strong predictive performance.  

**Key findings:**  
- Seasonal temperature cycles are clear and consistent.  
- Precipitation patterns vary significantly across cities.  
- Anomaly detection shows localized weather variability.  
- Ensemble models achieved **RMSE ≈ 2.3 °C** and **R² ≈ 0.86**.  

---

## 2. Dataset & Exploratory Data Analysis  

- **Source:** `GlobalWeatherRepository.csv`  
- **Target variable:** `temperature_celsius`  
- **Resolution:** Daily/hourly across multiple global cities  

### Key EDA Insights  

#### Correlation Analysis  
![Correlation Heatmap](images/corr_heatmap.png)  
- Strong positive correlation between `temperature_celsius` and `feels_like_celsius`.  
- Moderate relationships with humidity and pressure.  
- UV index shows city-specific correlation with temperature.  

#### Variable Distributions  
- **Temperature & Feels Like:**  
  ![Temperature Distribution](images/hist_temperature_celsius.png)  
  ![Feels Like Distribution](images/hist_feels_like_celsius.png)  
  Both follow near-normal distributions with seasonal peaks.  

- **Humidity:**  
  ![Humidity Distribution](images/hist_humidity.png)  
  Skewed toward high levels (70–90%).  

- **Pressure:**  
  ![Pressure Distribution](images/hist_pressure_mb.png)  
  Narrow range centered around 1010–1020 mb.  

- **Precipitation:**  
  ![Precipitation Distribution](images/hist_precip_mm.png)  
  Highly skewed; most days had near-zero rainfall.  

- **Wind & UV Index:**  
  ![Wind Distribution](images/hist_wind_kph.png)  
  ![UV Distribution](images/hist_uv_index.png)  
  Winds typically 5–20 kph; UV index generally low with seasonal spikes.  

- **Cloud Cover:**  
  ![Cloud Distribution](images/hist_cloud.png)  
  Clustered around clear (0%), partly cloudy (~50%), and overcast (75–100%).  

#### Temporal Trends  
- **Temperature Over Time (by City):**  
  ![Temperature Trends](images/temperature_celsius_ALL_CITIES.png)  
  Clear seasonal cycles, with Tokyo showing the strongest seasonality.  

- **Precipitation Over Time (by City):**  
  ![Precipitation Trends](images/precip_mm_ALL_CITIES.png)  
  Irregular and spiky, with Tokyo wetter than London or New York.  

---

## 3. Model Results  

### Baseline Models (Notebook 02 – Univariate Lag)  
| Model                     | RMSE   | MAE   | MAPE (%) | R²    |  
|---------------------------|--------|-------|----------|-------|  
| Naive (lag-1)             | 10.54  | 9.58  | 44.22    | -1.85 |  
| Linear Regression (lag-1) | 8.98   | 7.66  | 32.32    | -1.07 |  

⚠️ High errors and negative R² → lag-only models fail to capture variability.  

---

### Multivariate Models (Notebook 03 – Engineered Features)  
| Model               | RMSE   | MAE   | MAPE (%) | R²   |  
|---------------------|--------|-------|----------|------|  
| Ridge Regression    | 2.32   | 1.80  | 8.73     | 0.86 |  
| Random Forest       | 2.54   | 1.94  | 8.88     | 0.83 |  
| **Ensemble (R+RF)** | **2.31** | **1.79** | **8.47** | **0.86** |  

✅ RMSE reduced by ~75% compared to baselines.  
✅ Ensemble provided the best balance of accuracy and stability.  

---

### Anomaly & Spatial Analysis (Notebook 04)  
| City   | Mean Temp (°C) | Std Dev | Uni-anomaly Rate | Multi-anomaly Rate |  
|--------|----------------|---------|------------------|--------------------|  
| Tokyo  | 27.9           | 4.7     | 53%              | 37%                |  
| London | 18.5           | 3.2     | 45%              | 53%                |  

- **Tokyo:** Higher averages, fewer anomalies.  
- **London:** Cooler temps but higher anomaly rates → more complex variability.  

---

## 4. Weather Trends & Insights  

- **Seasonality:** Strong annual cycles with predictable summer/winter swings.  
- **Drivers:** Lagged temperature, humidity, and pressure were most influential.  
- **City Differences:**  
  - Tokyo → Strong seasonality, moderate anomaly rates.  
  - London → Milder temps, higher multivariate anomalies.  
- **Precipitation:** Highly irregular but city-specific (Tokyo wetter than London).  

---

## 5. Conclusions  

- **Multivariate ensemble models** outperform simple baselines significantly.  
- **Weather anomalies** vary across cities, highlighting regional complexity.  
- **EDA confirms** strong seasonal cycles and key feature correlations.  

---

## 6. Next Steps  

- Fine-tune hyperparameters for Ridge and Random Forest.  
- Evaluate boosting methods (XGBoost, LightGBM, CatBoost).  
- Extend anomaly detection to more cities.  
- Deploy a **lightweight forecasting app** (Streamlit/Flask).  

---
