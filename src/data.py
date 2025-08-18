# ==============================================================
# data.py  — simple helpers for this project
# ==============================================================

from pathlib import Path
import numpy as np
import pandas as pd

# Folders (centralized)
DATA_DIR   = Path("../data")
ASSETS_DIR = Path("../assets")


# ---------------- I/O ----------------
def load_raw(file_name="GlobalWeatherRepository.csv"):
    """Read the raw CSV from ../data/."""
    return pd.read_csv(DATA_DIR / file_name)


def save_clean(df, file_name="clean_weather.csv", index=True):
    """Write cleaned/features CSV to ../assets/."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ASSETS_DIR / file_name, index=index)


# ------------- BASIC CLEANING -------------
def clean_basic(df, time_col="last_updated"):
    """
    - lowercase/strip column names
    - parse datetime
    - drop rows with missing time
    - sort by time, drop duplicates
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def keep_columns(df, time_col, city_col, country_col=None, numeric_cols=None, include_geo=True):
    """
    Keep only the columns we need and make numeric cols numeric.
    """
    df = df.copy()
    numeric_cols = numeric_cols or []

    cols = [time_col, city_col]
    if country_col:
        cols.append(country_col)

    if include_geo:
        for g in ["latitude", "longitude"]:
            if g in df.columns:
                cols.append(g)

    for c in numeric_cols:
        if c in df.columns:
            cols.append(c)

    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ------------- CITY FILTER + INDEX -------------
def filter_cities(df, city_col, cities, time_col):
    """
    Keep only the requested cities that actually exist.
    Set the time column as index.
    Returns: (df_indexed, cities_present)
    """
    df = df.copy()
    present = [c for c in cities if c in df[city_col].unique()]
    if present:
        df = df[df[city_col].isin(present)]
    df = df.set_index(time_col).sort_index()
    return df, present


# ------------- MISSING VALUES -------------
def fill_missing(df, city_col, numeric_cols):
    """
    Per city: forward fill, backward fill, then fill remaining NaNs with city median.
    """
    df = (
        df.groupby(city_col, group_keys=False)
          .apply(lambda g: g.ffill().bfill())
          .copy()
    )
    for c in numeric_cols:
        if c in df.columns:
            med = df.groupby(city_col)[c].transform("median")
            df[c] = df[c].fillna(med)
    return df


# ------------- OUTLIERS (IQR CLIP) -------------
def clip_outliers_iqr(df, city_col, numeric_cols, k=1.5):
    """
    Clip numeric columns using Tukey IQR rule per city.
    """
    df = df.copy()
    for c in numeric_cols:
        if c in df.columns:
            q1 = df.groupby(city_col)[c].transform(lambda s: s.quantile(0.25))
            q3 = df.groupby(city_col)[c].transform(lambda s: s.quantile(0.75))
            iqr = q3 - q1
            lo = q1 - k * iqr
            hi = q3 + k * iqr
            df[c] = df[c].clip(lower=lo, upper=hi)
    return df


# ------------- TIME FEATURES -------------
def add_time_features(df, day_period=365.25):
    """
    Assumes the DataFrame index is a datetime index.
    Adds: year, month, dayofyear, dow, sin_doy, cos_doy
    """
    df = df.copy()
    dt = pd.DatetimeIndex(df.index)
    doy = dt.dayofyear

    df["year"] = dt.year
    df["month"] = dt.month
    df["dayofyear"] = doy
    df["dow"] = dt.dayofweek
    df["sin_doy"] = np.sin(2 * np.pi * doy / day_period)
    df["cos_doy"] = np.cos(2 * np.pi * doy / day_period)
    return df
