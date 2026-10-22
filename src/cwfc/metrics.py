"""Forecast accuracy metrics used by the baseline and model comparison."""

from __future__ import annotations

import numpy as np


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if a.size == 0:
        raise ValueError("metric inputs must not be empty")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("metric inputs must be finite")
    return a, b


def mae(y_true, y_pred) -> float:
    a, b = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    a, b = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def bias(y_true, y_pred) -> float:
    a, b = _as_arrays(y_true, y_pred)
    return float(np.mean(a - b))


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Symmetric mean absolute percentage error in percent.

    The denominator is ``max(|y| + |p|, eps)``.  The aggregate CPU target is
    around 9%, so the guard is not active on this dataset; it is retained for
    robustness on near-zero series.
    """
    a, b = _as_arrays(y_true, y_pred)
    denom = np.maximum(np.abs(a) + np.abs(b), eps)
    return float(np.mean(2.0 * np.abs(a - b) / denom) * 100.0)


def normalized_mae(y_true, y_pred, baseline_pred) -> float:
    model_mae = mae(y_true, y_pred)
    baseline_mae = mae(y_true, baseline_pred)
    if baseline_mae == 0:
        return 0.0 if model_mae == 0 else float("inf")
    return float(model_mae / baseline_mae)


def mase(y_true, y_pred, y_train, seasonality: int = 1, eps: float = 1e-12) -> float:
    """Mean absolute scaled error.

    The scale is the in-sample mean absolute error of the naive (or seasonal
    naive) one-step forecast on the training history.  Values below 1 mean the
    forecast beats that naive benchmark.
    """
    a, b = _as_arrays(y_true, y_pred)
    train = np.asarray(y_train, dtype=float)
    if train.size <= seasonality:
        raise ValueError("y_train must contain more than ``seasonality`` values")
    scale = float(np.mean(np.abs(train[seasonality:] - train[:-seasonality])))
    if scale <= eps:
        return 0.0 if np.mean(np.abs(a - b)) <= eps else float("inf")
    return float(np.mean(np.abs(a - b)) / scale)


def peak_mask(y_true, quantile: float = 0.75) -> np.ndarray:
    """Boolean mask for the highest ``1 - quantile`` fraction of observations."""
    a = np.asarray(y_true, dtype=float)
    if a.size == 0:
        raise ValueError("y_true must not be empty")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    return a >= float(np.quantile(a, quantile))


def peak_mae(y_true, y_pred, quantile: float = 0.75) -> float:
    a, b = _as_arrays(y_true, y_pred)
    mask = peak_mask(a, quantile=quantile)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(a[mask] - b[mask])))


def calm_mae(y_true, y_pred, quantile: float = 0.75) -> float:
    a, b = _as_arrays(y_true, y_pred)
    mask = ~peak_mask(a, quantile=quantile)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(a[mask] - b[mask])))


def summarize_metrics(
    y_true,
    y_pred,
    *,
    y_train=None,
    baseline_pred=None,
    seasonality: int = 1,
    peak_quantile: float = 0.75,
) -> dict[str, float]:
    """Return a consistent metric dictionary for one forecast window."""
    result = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "peak_mae": peak_mae(y_true, y_pred, quantile=peak_quantile),
        "calm_mae": calm_mae(y_true, y_pred, quantile=peak_quantile),
    }
    if y_train is not None:
        result["mase"] = mase(y_true, y_pred, y_train, seasonality=seasonality)
    if baseline_pred is not None:
        result["normalized_mae"] = normalized_mae(y_true, y_pred, baseline_pred)
    return result


def interval_margins(
    y_true,
    y_pred,
    levels: tuple[float, ...] = (0.8, 0.9),
) -> dict[str, float]:
    """Empirical two-sided interval half-widths from absolute residuals.

    This is a split-conformal-style summary, not a formal finite-sample
    guarantee: the residuals come from rolling-origin validation and are
    therefore dependent.  It is used as an operational uncertainty band.
    """
    a, b = _as_arrays(y_true, y_pred)
    absolute = np.abs(a - b)
    return {f"{level:.2f}": float(np.quantile(absolute, level)) for level in levels}
