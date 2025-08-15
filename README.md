 PMACC Weather Data Analysis

This repository contains my technical assessment for the PM Accelerator AI Engineer Bootcamp – Data Scientist role.
The goal is to analyze weather data, identify patterns and anomalies, and present insights with clear visualizations and reproducible workflows.

📂 Project Structure
pmacc-weather-datasci/
│
├── 📁 src/                # Core Python scripts
│   ├── data.py            # Data loading & preprocessing
│   ├── features.py        # Feature engineering
│   ├── models.py          # ML models setup & training
│   ├── evaluate.py        # Model evaluation
│   └── viz.py             # Visualization utilities
│
├── 📁 notebooks/           # Jupyter notebooks for step-by-step workflow
│   ├── 01_data_prep_eda.ipynb                 # Data preparation & EDA
│   ├── 02_model_baseline_univariate.ipynb     # Baseline models
│   ├── 03_models_multivariate+ensemble.ipynb  # Advanced models
│   └── 04_anomalies_spatial_feature_importance.ipynb  # Anomalies & insights
│
├── 📁 report/              # Final deliverables
│   └── PMA_Weather_Trend_Report.md
│
├── 📁 assets/              # Images, plots, and supporting files
│
├── README.md               # Project overview (this file)
├── requirements.txt        # Python dependencies
└── .gitignore              # Git ignore file

 Objectives

Data Acquisition – Load historical and forecast weather data.

Exploratory Data Analysis (EDA) – Understand trends, correlations, and seasonal variations.

Feature Engineering – Create meaningful variables for model improvement.

Modeling – Build baseline and advanced predictive models.

Anomaly Detection – Identify unusual weather patterns.

Visualization – Present findings using clear, interpretable visuals.

⚙️ Tech Stack

Language: Python 3.x

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn, Plotly

Modeling: Scikit-learn, XGBoost

Environment: Jupyter Notebook

🚀 How to Run

Clone the repository

git clone https://github.com/jibz33on/pmacc-weather-datasci.git
cd pmacc-weather-datasci


Install dependencies

pip install -r requirements.txt


Run notebooks in order:

01_data_prep_eda.ipynb

02_model_baseline_univariate.ipynb

03_models_multivariate+ensemble.ipynb

04_anomalies_spatial_feature_importance.ipynb

📊 Deliverables

Source Code: Modular Python scripts (src/) and Jupyter notebooks (notebooks/)

Report: report/PMA_Weather_Trend_Report.md summarizing insights and recommendations

Visuals: Plots and charts saved in assets/

📌 Notes

This is part of my submission for the Data Scientist technical assessment with PM Accelerator.
