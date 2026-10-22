"""Candidate forecasters and the final forecast combination.

The models are deliberately simple.  The aggregate series is short (110
training observations) and very smooth, so a well-regularised combination of
level, smoothing, autoregressive, and seasonal persistence is more defensible
than a high-capacity model.  All models are univariate and receive only the
training history at the forecast origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import (
    DEFAULT_LAGS,
    DEFAULT_WINDOWS,
    build_supervised,
    make_future_features,
    validate_series,
)

ForecastFn = Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class ModelSpec:
    """A named forecaster with the minimum history it needs."""

    name: str
    family: str
    min_history: int
    forecast: ForecastFn
    description: str

    def predict(self, history, horizon: int) -> np.ndarray:
        return self.forecast(np.asarray(history, dtype=float), int(horizon))


def _finalize(prediction, horizon: int, fallback: np.ndarray) -> np.ndarray:
    array = np.asarray(prediction, dtype=float).reshape(-1)
    if array.size != horizon or not np.all(np.isfinite(array)):
        array = np.asarray(fallback, dtype=float).reshape(-1)
    # Bound handling belongs to the target-specific pipeline: CPU utilization
    # is capped at 100, while active VM count is a count and may exceed 100.
    return array


def _persistence_array(history: np.ndarray, horizon: int) -> np.ndarray:
    return np.full(horizon, float(history[-1]), dtype=float)


def forecast_persistence(history, horizon: int) -> np.ndarray:
    series = validate_series(history, minimum=1)
    return _finalize(_persistence_array(series, horizon), horizon, _persistence_array(series, horizon))


def forecast_moving_average(history, horizon: int, window: int = 12) -> np.ndarray:
    series = validate_series(history, minimum=1)
    if window <= 0:
        raise ValueError("window must be positive")
    window = min(window, len(series))
    level = float(np.mean(series[-window:]))
    return _finalize(np.full(horizon, level), horizon, _persistence_array(series, horizon))


def forecast_seasonal_naive(history, horizon: int, season: int = 48) -> np.ndarray:
    series = validate_series(history, minimum=season)
    if season <= 0:
        raise ValueError("season must be positive")
    pattern = series[-season:]
    forecast = np.asarray([pattern[i % season] for i in range(horizon)], dtype=float)
    return _finalize(forecast, horizon, _persistence_array(series, horizon))


def forecast_seasonal_random_walk(
    history, horizon: int, season: int = 48, weight: float = 1.0
) -> np.ndarray:
    """Seasonal random walk anchored at the last observed value.

    ``yhat(T+h) = y(T) + weight * (y(T-s+h) - y(T-s))``.  With ``weight=0``
    this is persistence; with ``weight=1`` it is a seasonal random walk.  The
    anchor prevents a level shift in the previous cycle from being extrapolated
    without limit.
    """
    series = validate_series(history, minimum=season)
    if season <= 0:
        raise ValueError("season must be positive")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    previous = series[-season:]
    anchor = float(series[-1])
    forecast = np.asarray(
        [anchor + weight * (previous[i % season] - previous[0]) for i in range(horizon)],
        dtype=float,
    )
    return _finalize(forecast, horizon, _persistence_array(series, horizon))


def forecast_ses(history, horizon: int, alpha: float | None = None) -> np.ndarray:
    """Simple exponential smoothing with a grid-searched level parameter."""
    series = validate_series(history, minimum=4)
    if alpha is None:
        best_sse = np.inf
        best_alpha = 0.5
        for candidate in np.linspace(0.01, 0.99, 99):
            level = float(series[0])
            sse = 0.0
            for value in series[1:]:
                error = float(value) - level
                sse += error * error
                level = candidate * float(value) + (1.0 - candidate) * level
            if sse < best_sse:
                best_sse = sse
                best_alpha = float(candidate)
        alpha = best_alpha
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    level = float(series[0])
    for value in series[1:]:
        level = alpha * float(value) + (1.0 - alpha) * level
    return _finalize(np.full(horizon, level), horizon, _persistence_array(series, horizon))


def forecast_ar(history, horizon: int, order: int = 6) -> np.ndarray:
    """Autoregressive model on levels, fitted by ordinary least squares.

    Forecasts are produced recursively: each predicted value becomes the
    newest lag for the next step.  The model has an intercept, which lets it
    mean-revert to the training-window level rather than drifting indefinitely.
    """
    series = validate_series(history, minimum=order + 2)
    if order <= 0:
        raise ValueError("order must be positive")

    rows: list[np.ndarray] = []
    targets: list[float] = []
    for t in range(order, len(series)):
        rows.append(series[t - order : t][::-1])
        targets.append(float(series[t]))
    design = np.column_stack([np.ones(len(rows)), np.asarray(rows)])
    coefficients, *_ = np.linalg.lstsq(
        design, np.asarray(targets, dtype=float), rcond=None
    )

    working = list(np.asarray(series, dtype=float))
    prediction: list[float] = []
    for _ in range(horizon):
        lags = np.asarray(working[-order:][::-1], dtype=float)
        value = float(coefficients[0] + lags @ coefficients[1:])
        if not np.isfinite(value):
            value = float(working[-1])
        prediction.append(value)
        working.append(value)
    return _finalize(prediction, horizon, _persistence_array(series, horizon))


def _feature_model_forecast(
    history,
    horizon: int,
    *,
    estimator,
    lags: tuple[int, ...],
    windows: tuple[int, ...],
) -> np.ndarray:
    series = validate_series(
        history, minimum=max(max(lags, default=0), max(windows, default=0)) + 1
    )
    features, target, _ = build_supervised(series, lags=lags, windows=windows)
    estimator.fit(features, target)

    working = list(np.asarray(series, dtype=float))
    prediction: list[float] = []
    for _ in range(horizon):
        vector = make_future_features(
            np.asarray(working), lags=lags, windows=windows
        ).reshape(1, -1)
        value = float(estimator.predict(vector)[0])
        if not np.isfinite(value):
            value = float(working[-1])
        prediction.append(value)
        working.append(value)
    return _finalize(prediction, horizon, _persistence_array(series, horizon))


def forecast_ridge(
    history,
    horizon: int,
    alpha: float = 10.0,
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> np.ndarray:
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    return _feature_model_forecast(
        history, horizon, estimator=estimator, lags=lags, windows=windows
    )


def forecast_hist_gradient_boosting(
    history,
    horizon: int,
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> np.ndarray:
    estimator = HistGradientBoostingRegressor(
        max_iter=250,
        max_depth=2,
        learning_rate=0.04,
        min_samples_leaf=8,
        l2_regularization=2.0,
        random_state=0,
    )
    try:
        return _feature_model_forecast(
            history, horizon, estimator=estimator, lags=lags, windows=windows
        )
    except Exception as exc:  # pragma: no cover - defensive on tiny smoke data
        warnings.warn(
            f"HistGradientBoosting failed ({exc}); falling back to persistence.",
            RuntimeWarning,
            stacklevel=2,
        )
        series = validate_series(history, minimum=1)
        return forecast_persistence(series, horizon)


def forecast_ensemble(
    history,
    horizon: int,
    components: Sequence[tuple[ModelSpec, float]],
) -> np.ndarray:
    """Weighted average of available component forecasts.

    Components whose minimum history is not met are skipped and the remaining
    weights are renormalised.  This keeps the same model definition usable on a
    small smoke dataset without hard-coding the challenge's 110 observations.
    """
    series = validate_series(history, minimum=1)
    predictions: list[np.ndarray] = []
    weights: list[float] = []
    for spec, weight in components:
        if weight <= 0 or len(series) < spec.min_history:
            continue
        try:
            forecast = spec.predict(series, horizon)
        except Exception:
            continue
        if forecast.shape != (horizon,) or not np.all(np.isfinite(forecast)):
            continue
        predictions.append(forecast)
        weights.append(float(weight))
    if not predictions:
        return forecast_persistence(series, horizon)
    stacked = np.vstack(predictions)
    weights_array = np.asarray(weights, dtype=float)
    combined = np.average(stacked, axis=0, weights=weights_array)
    return _finalize(combined, horizon, _persistence_array(series, horizon))


PERSISTENCE = ModelSpec(
    name="persistence",
    family="baseline",
    min_history=1,
    forecast=forecast_persistence,
    description="Last observed value repeated for every horizon.",
)

SEASONAL_NAIVE_12 = ModelSpec(
    name="seasonal_naive_12",
    family="baseline",
    min_history=12,
    forecast=lambda history, horizon: forecast_seasonal_naive(history, horizon, season=12),
    description="Repeat the 1-hour seasonal pattern.",
)

SEASONAL_NAIVE_24 = ModelSpec(
    name="seasonal_naive_24",
    family="baseline",
    min_history=24,
    forecast=lambda history, horizon: forecast_seasonal_naive(history, horizon, season=24),
    description="Repeat the 2-hour seasonal pattern.",
)

SEASONAL_NAIVE_48 = ModelSpec(
    name="seasonal_naive_48",
    family="baseline",
    min_history=48,
    forecast=lambda history, horizon: forecast_seasonal_naive(history, horizon, season=48),
    description="Repeat the 4-hour seasonal pattern.",
)

SEASONAL_RANDOM_WALK_48 = ModelSpec(
    name="seasonal_random_walk_48",
    family="baseline",
    min_history=48,
    forecast=lambda history, horizon: forecast_seasonal_random_walk(
        history, horizon, season=48, weight=0.5
    ),
    description="50/50 blend of persistence and a 4-hour seasonal random walk.",
)

MOVING_AVERAGE_6 = ModelSpec(
    name="moving_average_6",
    family="baseline",
    min_history=6,
    forecast=lambda history, horizon: forecast_moving_average(history, horizon, window=6),
    description="Mean of the last 30 minutes.",
)

MOVING_AVERAGE_12 = ModelSpec(
    name="moving_average_12",
    family="baseline",
    min_history=12,
    forecast=lambda history, horizon: forecast_moving_average(history, horizon, window=12),
    description="Mean of the last hour.",
)

MOVING_AVERAGE_24 = ModelSpec(
    name="moving_average_24",
    family="baseline",
    min_history=24,
    forecast=lambda history, horizon: forecast_moving_average(history, horizon, window=24),
    description="Mean of the last two hours.",
)

SES = ModelSpec(
    name="ses",
    family="statistical",
    min_history=4,
    forecast=forecast_ses,
    description="Simple exponential smoothing with grid-searched alpha.",
)

AR_2 = ModelSpec(
    name="ar_2",
    family="statistical",
    min_history=4,
    forecast=lambda history, horizon: forecast_ar(history, horizon, order=2),
    description="AR(2) on levels with an intercept, recursive multi-step forecast.",
)

AR_6 = ModelSpec(
    name="ar_6",
    family="statistical",
    min_history=8,
    forecast=lambda history, horizon: forecast_ar(history, horizon, order=6),
    description="AR(6) on levels with an intercept, recursive multi-step forecast.",
)

RIDGE_10 = ModelSpec(
    name="ridge_10",
    family="ml",
    min_history=14,
    forecast=lambda history, horizon: forecast_ridge(history, horizon, alpha=10.0),
    description="Ridge regression on lag and rolling-window features.",
)

HIST_GBM = ModelSpec(
    name="hist_gradient_boosting",
    family="ml",
    min_history=14,
    forecast=forecast_hist_gradient_boosting,
    description="Small gradient-boosted tree model on lag and rolling-window features.",
)


def _ensemble(
    name: str,
    components: Sequence[tuple[ModelSpec, float]],
    description: str,
) -> ModelSpec:
    component_specs = tuple(components)
    return ModelSpec(
        name=name,
        family="ensemble",
        min_history=1,
        forecast=lambda history, horizon: forecast_ensemble(
            history, horizon, component_specs
        ),
        description=description,
    )


ENSEMBLE_P_MA12_AR6_S48 = _ensemble(
    "ensemble_p_ma12_ar6_s48",
    (
        (PERSISTENCE, 1.0),
        (MOVING_AVERAGE_12, 1.0),
        (AR_6, 1.0),
        (SEASONAL_NAIVE_48, 1.0),
    ),
    "Equal-weight combination of persistence, 1-hour mean, AR(6), and 4-hour seasonal naive.",
)

ENSEMBLE_P_MA12_S48 = _ensemble(
    "ensemble_p_ma12_s48",
    ((PERSISTENCE, 1.0), (MOVING_AVERAGE_12, 1.0), (SEASONAL_NAIVE_48, 1.0)),
    "Equal-weight combination of persistence, 1-hour mean, and 4-hour seasonal naive.",
)

ENSEMBLE_P_AR6 = _ensemble(
    "ensemble_p_ar6",
    ((PERSISTENCE, 1.0), (AR_6, 1.0)),
    "Equal-weight combination of persistence and AR(6).",
)

ENSEMBLE_P_MA12_AR6 = _ensemble(
    "ensemble_p_ma12_ar6",
    ((PERSISTENCE, 1.0), (MOVING_AVERAGE_12, 1.0), (AR_6, 1.0)),
    "Equal-weight combination of persistence, 1-hour mean, and AR(6).",
)

ENSEMBLE_P_MA12_AR6_S48_RIDGE = _ensemble(
    "ensemble_p_ma12_ar6_s48_ridge",
    (
        (PERSISTENCE, 1.0),
        (MOVING_AVERAGE_12, 1.0),
        (AR_6, 1.0),
        (SEASONAL_NAIVE_48, 1.0),
        (RIDGE_10, 1.0),
    ),
    "Five-way equal-weight combination including ridge regression.",
)


ALL_MODELS: tuple[ModelSpec, ...] = (
    PERSISTENCE,
    MOVING_AVERAGE_6,
    MOVING_AVERAGE_12,
    MOVING_AVERAGE_24,
    SEASONAL_NAIVE_12,
    SEASONAL_NAIVE_24,
    SEASONAL_NAIVE_48,
    SEASONAL_RANDOM_WALK_48,
    SES,
    AR_2,
    AR_6,
    RIDGE_10,
    HIST_GBM,
    ENSEMBLE_P_AR6,
    ENSEMBLE_P_MA12_AR6,
    ENSEMBLE_P_MA12_S48,
    ENSEMBLE_P_MA12_AR6_S48,
    ENSEMBLE_P_MA12_AR6_S48_RIDGE,
)

FINAL_MODEL_NAME = ENSEMBLE_P_MA12_AR6_S48.name


def available_models(history_length: int) -> list[ModelSpec]:
    """Return models that can be fitted with the available history."""
    if history_length <= 0:
        raise ValueError("history_length must be positive")
    return [spec for spec in ALL_MODELS if spec.min_history <= history_length]


def get_model(name: str) -> ModelSpec:
    for spec in ALL_MODELS:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown model: {name}")


def model_table() -> list[dict[str, object]]:
    return [
        {
            "model": spec.name,
            "family": spec.family,
            "min_history": spec.min_history,
            "description": spec.description,
        }
        for spec in ALL_MODELS
    ]
