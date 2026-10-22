"""Leakage-safe feature construction for univariate forecasting models.

Every feature for a forecast made at origin ``T`` is computed from values with
index ``<= T``.  There are no calendar features because the challenge
timestamps are seconds relative to the start of the Azure trace: the absolute
time of day is unknown and the 11-hour window is shorter than a full day.
Using ``timestamp % 86400`` would impose an unverifiable wall-clock assumption.
"""

from __future__ import annotations

import numpy as np


DEFAULT_LAGS = (1, 2, 3, 6, 12)
DEFAULT_WINDOWS = (3, 6, 12)


def validate_series(values, *, minimum: int = 1) -> np.ndarray:
    series = np.asarray(values, dtype=float)
    if series.ndim != 1:
        raise ValueError("history must be a one-dimensional array")
    if series.size < minimum:
        raise ValueError(f"history must contain at least {minimum} observations")
    if not np.all(np.isfinite(series)):
        raise ValueError("history must contain only finite values")
    return series


def lag_feature_vector(
    values,
    index: int,
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> np.ndarray:
    """Features known at ``index`` (inclusive), used to predict ``index + 1``."""
    series = np.asarray(values, dtype=float)
    if series.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if index < 0 or index >= len(series):
        raise IndexError("index is outside the series")
    max_lag = max(max(lags, default=0), max(windows, default=0))
    if index + 1 < max_lag:
        raise ValueError("not enough history for the requested feature set")

    features: list[float] = []
    for lag in lags:
        features.append(float(series[index - lag + 1]))
    for window in windows:
        window_values = series[index - window + 1 : index + 1]
        features.append(float(np.mean(window_values)))
        features.append(float(np.std(window_values, ddof=0)))
    return np.asarray(features, dtype=float)


def build_supervised(
    values,
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build one-step-ahead training rows without using future information.

    Row ``t`` contains features computed from observations up to ``t - 1`` and
    the target ``values[t]``.  The returned ``min_history`` is the smallest
    history length needed to construct a feature vector at the forecast origin.
    """
    series = validate_series(values, minimum=2)
    max_lag = max(max(lags, default=0), max(windows, default=0))
    min_history = max_lag
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for t in range(max_lag, len(series)):
        rows.append(
            lag_feature_vector(
                series, t - 1, lags=lags, windows=windows
            )
        )
        targets.append(float(series[t]))
    if not rows:
        raise ValueError("not enough history to build supervised rows")
    return np.vstack(rows), np.asarray(targets, dtype=float), min_history


def make_future_features(
    history,
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> np.ndarray:
    """Feature vector at the end of ``history`` for a one-step prediction."""
    series = validate_series(history, minimum=max(max(lags), max(windows)))
    return lag_feature_vector(
        series, len(series) - 1, lags=lags, windows=windows
    )
