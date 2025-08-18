

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance


# ---------------- metrics ----------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE":  mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2":   r2(y_true, y_pred),
    }


# ---------------- single / multi model eval ----------------
def eval_model(model, X_train, y_train, X_test, y_test, name: str = "model"):
    """
    Fit model, predict on test, return metrics dict and y_hat.
    """
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    met = compute_metrics(y_test, y_hat)
    met["model"] = name
    return met, y_hat


def eval_models(models: dict, X_train, y_train, X_test, y_test):
    """
    Evaluate a dict of {name: model}. Returns (results_df, preds_dict).
    """
    results = []
    preds = {}
    for name, m in models.items():
        met, y_hat = eval_model(m, X_train, y_train, X_test, y_test, name=name)
        results.append(met)
        preds[name] = y_hat
    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    return results_df, preds


# ---------------- simple ensemble ----------------
def ensemble_mean(preds: dict[str, np.ndarray], members: list[str] | None = None) -> np.ndarray:
    """
    Average predictions across the given model names (or all in dict if None).
    """
    keys = members or list(preds.keys())
    arr = np.column_stack([preds[k] for k in keys])
    return arr.mean(axis=1)


# ---------------- residuals & errors ----------------
def residuals_frame(y_true, y_pred) -> pd.DataFrame:
    """
    Return a tidy DataFrame with y_true, y_pred, residuals and error stats.
    Index is preserved from y_true when possible (Series).
    """
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        idx = y_true.index
        y_true_vals = np.asarray(y_true).reshape(-1)
    else:
        idx = None
        y_true_vals = np.asarray(y_true).reshape(-1)

    y_pred_vals = np.asarray(y_pred).reshape(-1)
    df = pd.DataFrame(
        {
            "y_true": y_true_vals,
            "y_pred": y_pred_vals,
            "residual": y_true_vals - y_pred_vals,
        },
        index=idx,
    )
    df["abs_error"] = np.abs(df["residual"])
    # safe percentage error
    df["pct_error"] = (df["abs_error"] / np.clip(np.abs(df["y_true"]), 1e-8, None)) * 100.0
    return df


# ---------------- permutation importance (for Pipeline) ----------------
def _ohe_feature_names_from_cat(preprocessor, cat_step_name: str, cat_input_names: list[str]) -> list[str]:
    """
    Helper to pull one-hot encoded feature names from a fitted OHE in a ColumnTransformer.
    Tries get_feature_names_out; falls back to manual names using categories_ if needed.
    """
    ohe = preprocessor.named_transformers_[cat_step_name]
    # newer sklearn
    if hasattr(ohe, "get_feature_names_out"):
        return list(ohe.get_feature_names_out(cat_input_names))
    # older sklearn
    cats = ohe.categories_
    names = []
    for base, cat_vals in zip(cat_input_names, cats):
        names += [f"{base}_{c}" for c in cat_vals]
    return names


def get_feature_names_from_column_transformer(
    preprocessor,
    num_step_name: str = "num",
    cat_step_name: str = "cat",
    num_input_names: list[str] | None = None,
    cat_input_names: list[str] | None = None,
) -> np.ndarray:
    """
    Try to recover the transformed feature names from a fitted ColumnTransformer.
    If .get_feature_names_out() exists, use it. Otherwise, build names from inputs.
    """
    # best case: sklearn >= 1.1
    if hasattr(preprocessor, "get_feature_names_out"):
        return preprocessor.get_feature_names_out()

    # fallback: build from provided input lists
    if num_input_names is None or cat_input_names is None:
        raise ValueError("Provide num_input_names and cat_input_names for older sklearn versions.")
    num_names = list(num_input_names)
    cat_names = _ohe_feature_names_from_cat(preprocessor, cat_step_name, cat_input_names)
    return np.array(num_names + cat_names)


def permutation_importance_table(
    pipeline,               # fitted sklearn Pipeline with step names: "prep", "model"
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
    # only needed for older sklearn without get_feature_names_out on ColumnTransformer:
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute permutation importance on a fitted Pipeline and return a sorted DataFrame.
    Works with your (num, cat) ColumnTransformer under step name "prep".
    """
    imp = permutation_importance(
        pipeline, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs
    )

    prep = pipeline.named_steps["prep"]
    try:
        feat_names = get_feature_names_from_column_transformer(prep)
    except Exception:
        # fallback requires num_cols and cat_cols
        if num_cols is None or cat_cols is None:
            raise
        feat_names = get_feature_names_from_column_transformer(
            prep,
            num_input_names=num_cols,
            cat_input_names=cat_cols,
        )

    # align in case of any mismatch (defensive)
    k = min(len(feat_names), imp.importances_mean.shape[0])
    df = (
        pd.DataFrame(
            {
                "feature": np.asarray(feat_names)[:k],
                "importance_mean": imp.importances_mean[:k],
                "importance_std": imp.importances_std[:k],
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    return df
