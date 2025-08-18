# 🌦️ PMACC Weather Data Science Project – Comprehensive Report  

## 1. Introduction & Objectives  
The PMACC Weather Data Science Project aims to analyze global weather data and build predictive models for temperature trends.  

**Objectives:**  
1. Clean and prepare large-scale weather data for modeling.  
2. Explore seasonal and spatial weather trends across global cities.  
3. Develop and evaluate machine learning models for temperature forecasting.  
4. Detect anomalies and highlight city-specific variability.  
5. Provide actionable insights and recommendations for future forecasting systems.  

---

## 2. Dataset  

- **Source:** `GlobalWeatherRepository.csv`  
- **Size:** ~89,000rows (daily/hourly resolution)  
- **Target variable:** `temperature_celsius`  
- **Other variables:** humidity, wind speed, pressure, precipitation, UV index, cloud cover  

---

## 3. Exploratory Data Analysis (Notebook 01)  

### Correlation Analysis  
![Correlation Heatmap](images/corr_heatmap.png)  
- Temperature strongly correlated with feels-like values.  
- Moderate relationships with humidity and pressure.  
- UV index correlations varied across cities.  

### Variable Distributions  
- **Temperature & Feels Like:** Near-normal distributions with seasonal peaks.  
- **Humidity:** Skewed toward high humidity (70–90%).  
- **Pressure:** Centered around 1010–1020 mb.  
- **Precipitation:** Highly skewed; most days near-zero.  
- **Wind & UV:** Wind 5–20 kph; UV generally low with seasonal spikes.  
- **Cloud Cover:** Clustered around clear, partly cloudy, and overcast.  

### Temporal Trends  
- **Temperature Trends by City:**  
  ![Temperature Trends](images/temperature_celsius_ALL_CITIES.png)  
  Clear seasonal cycles (Tokyo strongest).  

- **Precipitation Trends by City:**  
  ![Precipitation Trends](images/precip_mm_ALL_CITIES.png)  
  Irregular and city-dependent, Tokyo wetter than London or New York.  

---

## 4. Methodology  

### 4.1 Data Preparation  
- Cleaned raw dataset → `clean_weather.csv`.  
- Handled missing values, duplicates, and type conversions.  
- Time-indexed by `last_updated`.  

### 4.2 Feature Engineering  
- Lag features: 1, 2, 3, 7, 14 days.  
- Rolling averages: 3-day, 7-day.  
- Seasonal encodings: sine/cosine transformations of day-of-year.  
- Categorical encoding: one-hot encoding of day-of-week.  

### 4.3 Modeling Strategy  
- **Univariate Baseline (Notebook 02):**  
  - Used lag-1 temperature only.  
  - Naive & linear regression baselines.  

- **Multivariate Models (Notebook 03):**  
  - Ridge Regression and Random Forest with engineered features.  

- **Ensemble:**  
  - Averaged Ridge + Random Forest predictions.  

- **Anomaly Detection & Spatial Analysis (Notebook 04):**  
  - Isolation Forest for univariate and multivariate anomaly detection.  
  - City-level anomaly summaries.  

---

## 5. Results  

### 5.1 Baseline Models (Notebook 02)  
| Model                     | RMSE   | MAE   | MAPE (%) | R²    |  
|---------------------------|--------|-------|----------|-------|  
| Naive (lag-1)             | 10.54  | 9.58  | 44.22    | -1.85 |  
| Linear Regression (lag-1) | 8.98   | 7.66  | 32.32    | -1.07 |  

⚠️ Lag-only models were ineffective with high errors and negative R².  

---

### 5.2 Multivariate Models (Notebook 03)  
| Model               | RMSE   | MAE   | MAPE (%) | R²   |  
|---------------------|--------|-------|----------|------|  
| Ridge Regression    | 2.32   | 1.80  | 8.73     | 0.86 |  
| Random Forest       | 2.54   | 1.94  | 8.88     | 0.83 |  
| **Ensemble (R+RF)** | **2.31** | **1.79** | **8.47** | **0.86** |  

✅ RMSE reduced ~75% compared to baselines.  
✅ Ensemble gave best balance of accuracy and robustness.  

---

### 5.3 Anomaly & Spatial Analysis (Notebook 04)  
| Country        | City   | Mean Temp (°C) | Std Dev | Uni-anomaly Rate | Multi-anomaly Rate |  
|----------------|--------|----------------|---------|------------------|--------------------|  
| Japan          | Tokyo  | 27.9           | 4.7     | 53%              | 37%                |  
| United Kingdom | London | 18.5           | 3.2     | 45%              | 53%                |  

- **Tokyo:** Warmer averages, fewer anomalies.  
- **London:** Cooler averages, higher multivariate anomaly rates.  

---

## 6. Visual Insights  

- **Correlation Heatmap:** Identified strongest relationships between features.  
- **Rolling Statistics:** Smoothed seasonal cycles and weather fluctuations.  
- **Feature Importance (Random Forest):** Lagged temperature + seasonal encodings ranked highest.  
- **Spatial Analysis:** Highlighted city-level differences in anomaly rates and variability.  

---

## 7. Conclusions  

- **Lagged + seasonal features** provide strong predictive power.  
- **Multivariate ensemble models** significantly outperform naive baselines.  
- **Anomaly detection** adds interpretability by highlighting city-level variability.  
- Workflow is reproducible: raw data → cleaned dataset → features → models → evaluation.
- Environmental impact analysis (e.g., air quality correlations) was not included due to dataset limitations, but could be incorporated in future work.  

---

## 8. Next Steps  

- Hyperparameter tuning for Ridge (α) and Random Forest (depth, estimators).  
- Explore boosting methods (XGBoost, LightGBM, CatBoost).  
- Scale anomaly detection and spatial insights across more cities.  
- Deploy a **Streamlit/Flask forecasting app** for practical use.  

---

## 9. Appendix (Optional for Reviewers)  
- **Notebooks:**  
  - `01_data_prep_eda.ipynb` – Cleaning & EDA  
  - `02_model_baseline_univariate.ipynb` – Baseline Models  
  - `03_model_multivariate.ipynb` – Multivariate Models  
  - `04_anomalies_spatial_feature_importance.ipynb` – Anomalies & Spatial Analysis  

- **Source Code (`src/`):** Functions for data loading, feature engineering, and modeling.  
- **Reports (`report/`):**  
  - `PMA_Weather_Trend_Report.md` – Trend-focused summary  
  - `PMA_Full_Project_Report.md` – Comprehensive report (this file)  
