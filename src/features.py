

import numpy as np
import pandas as pd


# ---------------- basic builders ----------------
def add_lags(df, target="temperature_celsius", lag_list=None):
    """
    Add lag features of the target. Uses past values only.
    """
    df = df.copy()
    lag_list = lag_list or [1, 2, 3, 7, 14]
    for k in lag_list:
        df[f"lag{k}"] = df[target].shift(k)
    return df


def add_rollings(df, target="temperature_celsius", windows=None, shift=1):
    """
    Add rolling mean/std of the (shifted) target to avoid leakage.
    """
    df = df.copy()
    windows = windows or [3, 7]
    for w in windows:
        df[f"roll{w}_mean"] = df[target].shift(shift).rolling(w).mean()
        df[f"roll{w}_std"]  = df[target].shift(shift).rolling(w).std()
    return df


def ensure_dow(df):
    """
    Ensure a day-of-week column (0=Mon ... 6=Sun). Assumes datetime index.
    """
    df = df.copy()
    if "dow" not in df.columns:
        df["dow"] = df.index.dayofweek
    return df


# ---------------- feature lists ----------------
def build_feature_lists(lag_list=None, roll_windows=None, include_raw=True):
    """
    Return (raw_feats, lags, rolls, feature_cols) to keep things consistent.
    """
    lag_list = lag_list or [1, 2, 3, 7, 14]
    roll_windows = roll_windows or [3, 7]

    raw_feats = []
    if include_raw:
        raw_feats = [
            "humidity", "pressure_mb", "wind_kph", "precip_mm",
            "cloud", "uv_index", "sin_doy", "cos_doy",
        ]

    lags = [f"lag{k}" for k in lag_list]
    rolls = [f"roll{w}_mean" for w in roll_windows] + [f"roll{w}_std" for w in roll_windows]

    feature_cols = raw_feats + lags + rolls + ["dow"]
    return raw_feats, lags, rolls, feature_cols


# ---------------- end-to-end maker ----------------
def make_features(
    df,
    target="temperature_celsius",
    lag_list=None,
    roll_windows=None,
    include_raw=True,
    dropna=True,
):
    """
    Build features end-to-end.
    Assumes df is already time-indexed on 'last_updated' and cleaned.
    Returns X, y, and feature_cols.
    """
    df = df.copy()

    # add engineered features
    df = add_lags(df, target=target, lag_list=lag_list)
    df = add_rollings(df, target=target, windows=roll_windows, shift=1)
    df = ensure_dow(df)

    # feature lists
    raw_feats, lags, rolls, feature_cols = build_feature_lists(
        lag_list=lag_list, roll_windows=roll_windows, include_raw=include_raw
    )

    # drop rows with NaNs from lag/rolling creation
    if dropna:
        df = df.dropna(subset=feature_cols + [target]).copy()

    X = df[feature_cols]
    y = df[target]
    return X, y, feature_cols


# ---------------- simple time split ----------------
def time_split(X, y, split=0.8):
    """
    Time-based split (no shuffle).
    """
    n = len(X)
    cut = int(n * split)
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]
    return X_train, X_test, y_train, y_test
