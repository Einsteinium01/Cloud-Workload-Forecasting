"""Time-aware validation for the forecasting challenge.

The original starter helpers are preserved.  The additional backtest utilities
use expanding-window, rolling-origin evaluation: each forecast is produced from
a history that ends before the validation window, and the validation window is
never passed to a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .data import ACTIVE_VM_COUNT, AGGREGATE_CPU
from .metrics import calm_mae, mae, mase, peak_mae, rmse, smape, bias
from .models import ModelSpec


@dataclass(frozen=True)
class TimeSplit:
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon: int = 288,
    min_train_size: int = 7 * 24 * 12,
    step: int | None = None,
) -> list[TimeSplit]:
    """Return expanding-window splits for evenly sampled time series.

    `horizon=288` is one day at 5-minute cadence.
    """
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive")
    ts = pd.to_datetime(frame[timestamp_col], utc=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted")
    n = len(frame)
    step = horizon if step is None else step
    out: list[TimeSplit] = []
    train_end_idx = min_train_size
    while train_end_idx + horizon <= n:
        train_end = ts.iloc[train_end_idx - 1]
        val_start = ts.iloc[train_end_idx]
        val_end = ts.iloc[train_end_idx + horizon - 1]
        out.append(TimeSplit(train_end, val_start, val_end))
        train_end_idx += step
    return out


def expanding_window_origins(
    n_observations: int,
    *,
    horizon: int,
    min_train: int,
    step: int = 12,
) -> list[int]:
    """Return training-window lengths for rolling-origin evaluation.

    An origin ``o`` means ``history = series[:o]`` and
    ``validation = series[o:o+horizon]``.
    """
    if horizon <= 0 or min_train <= 0 or step <= 0:
        raise ValueError("horizon, min_train, and step must be positive")
    if n_observations < min_train + horizon:
        raise ValueError(
            "not enough observations for the requested min_train and horizon"
        )
    return list(range(min_train, n_observations - horizon + 1, step))


def default_min_train(n_train: int, horizon: int = 24) -> int:
    """A production-like default: use as much history as possible.

    On the challenge, 110 - 24 = 86, so the default is capped at 50 to leave
    multiple rolling origins.  On a smaller smoke dataset the minimum shrinks
    automatically.
    """
    if n_train <= horizon:
        raise ValueError("training window must be longer than the horizon")
    return max(2, min(50, n_train - horizon))


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    origins: list[int]

    @property
    def models(self) -> list[str]:
        return sorted(self.predictions["model"].unique().tolist())


def _validate_series_frame(series: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in series.columns:
        raise KeyError(f"target column not found: {target}")
    frame = series[["timestamp", target]].sort_values(
        "timestamp", kind="mergesort"
    ).reset_index(drop=True)
    frame[target] = pd.to_numeric(frame[target], errors="raise").astype(float)
    if frame[target].isna().any():
        raise ValueError(f"target contains missing values: {target}")
    if not np.all(np.isfinite(frame[target].to_numpy())):
        raise ValueError(f"target contains non-finite values: {target}")
    return frame


def run_backtest(
    series: pd.DataFrame,
    models: Sequence[ModelSpec],
    *,
    target: str,
    horizon: int = 24,
    min_train: int | None = None,
    step: int = 12,
    peak_quantile: float = 0.75,
    baseline_name: str = "persistence",
) -> BacktestResult:
    """Expanding-window backtest with strictly past-only model inputs.

    The holdout values are used only to score forecasts after each model has
    produced them.  No model receives data beyond its forecast origin.
    """
    frame = _validate_series_frame(series, target)
    n = len(frame)
    if n <= horizon:
        raise ValueError("series must be longer than the forecast horizon")
    if min_train is None:
        min_train = default_min_train(n, horizon=horizon)
    origins = expanding_window_origins(
        n, horizon=horizon, min_train=min_train, step=step
    )
    values = frame[target].to_numpy(dtype=float)
    timestamps = frame["timestamp"].to_numpy()

    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for fold, origin in enumerate(origins):
        history = values[:origin]
        actual = values[origin : origin + horizon]
        actual_timestamps = timestamps[origin : origin + horizon]
        train_timestamps = timestamps[:origin]

        baseline_spec = next(
            (spec for spec in models if spec.name == baseline_name), None
        )
        baseline_forecast: np.ndarray | None = None
        if baseline_spec is not None:
            try:
                baseline_forecast = baseline_spec.predict(history, horizon)
            except Exception:  # pragma: no cover - defensive
                baseline_forecast = None

        for spec in models:
            if origin < spec.min_history:
                continue
            try:
                forecast = spec.predict(history, horizon)
            except Exception as exc:  # pragma: no cover - defensive
                metric_rows.append(
                    {
                        "model": spec.name,
                        "fold": fold,
                        "origin_index": origin,
                        "origin_timestamp": int(timestamps[origin - 1]),
                        "error": str(exc),
                    }
                )
                continue

            if target == AGGREGATE_CPU:
                forecast = np.clip(np.asarray(forecast, dtype=float), 0.0, 100.0)
            elif target == ACTIVE_VM_COUNT:
                forecast = np.clip(
                    np.rint(np.asarray(forecast, dtype=float)), 0.0, None
                )

            # Range guard: a forecast that leaves the observed training range
            # is treated as an extrapolation failure for model selection.  This
            # is a training-only check.
            within_range = bool(
                forecast.min() >= history.min() and forecast.max() <= history.max()
            )

            for step_index, (timestamp, truth, predicted) in enumerate(
                zip(actual_timestamps, actual, forecast)
            ):
                prediction_rows.append(
                    {
                        "fold": fold,
                        "origin_index": origin,
                        "origin_timestamp": int(train_timestamps[-1]),
                        "model": spec.name,
                        "step": step_index + 1,
                        "timestamp": int(timestamp),
                        "y_true": float(truth),
                        "y_pred": float(predicted),
                    }
                )

            metric_rows.append(
                {
                    "model": spec.name,
                    "fold": fold,
                    "origin_index": origin,
                    "origin_timestamp": int(train_timestamps[-1]),
                    "n_train": int(origin),
                    "mae": mae(actual, forecast),
                    "rmse": rmse(actual, forecast),
                    "smape": smape(actual, forecast),
                    "bias": bias(actual, forecast),
                    "mase": mase(actual, forecast, history, seasonality=1),
                    "peak_mae": peak_mae(actual, forecast, quantile=peak_quantile),
                    "calm_mae": calm_mae(actual, forecast, quantile=peak_quantile),
                    "normalized_mae": (
                        float(mae(actual, forecast) / mae(actual, baseline_forecast))
                        if baseline_forecast is not None and mae(actual, baseline_forecast) > 0
                        else np.nan
                    ),
                    "within_training_range": within_range,
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    if predictions.empty:
        raise RuntimeError("no model produced a forecast")
    return BacktestResult(predictions=predictions, metrics=metrics, origins=origins)


def summarize_backtest(result: BacktestResult, *, peak_quantile: float = 0.75) -> pd.DataFrame:
    """Aggregate fold metrics into one row per model.

    Pooled MAE/RMSE/sMAPE are computed over all validation points, while
    ``mean_fold_mae``, ``median_fold_mae``, and ``max_fold_mae`` describe
    stability across origins.
    """
    pred = result.predictions
    metric = result.metrics
    if metric.empty:
        raise ValueError("backtest metrics are empty")

    pooled_rows: list[dict[str, object]] = []
    for model, group in pred.groupby("model", sort=False):
        truth = group["y_true"].to_numpy(dtype=float)
        forecast = group["y_pred"].to_numpy(dtype=float)
        pooled_rows.append(
            {
                "model": model,
                "n_predictions": int(len(group)),
                "pooled_mae": mae(truth, forecast),
                "pooled_rmse": rmse(truth, forecast),
                "pooled_smape": smape(truth, forecast),
                "pooled_bias": bias(truth, forecast),
                "pooled_peak_mae": peak_mae(truth, forecast, quantile=peak_quantile),
                "pooled_calm_mae": calm_mae(truth, forecast, quantile=peak_quantile),
            }
        )
    pooled = pd.DataFrame(pooled_rows)

    fold = (
        metric.dropna(subset=["mae"])
        .groupby("model", sort=False)
        .agg(
            n_folds=("fold", "nunique"),
            mean_fold_mae=("mae", "mean"),
            median_fold_mae=("mae", "median"),
            max_fold_mae=("mae", "max"),
            mean_fold_rmse=("rmse", "mean"),
            mean_mase=("mase", "mean"),
            mean_normalized_mae=("normalized_mae", "mean"),
            n_out_of_range_folds=(
                "within_training_range",
                lambda values: int((~values.to_numpy(dtype=bool)).sum()),
            ),
        )
        .reset_index()
    )
    summary = fold.merge(pooled, on="model", how="left")
    summary = summary.sort_values(["mean_fold_mae", "pooled_mae"], kind="mergesort")
    return summary.reset_index(drop=True)


@dataclass
class SelectionResult:
    selected_model: str
    selected_spec: ModelSpec
    table: pd.DataFrame
    reason: str


def select_model(
    summary: pd.DataFrame,
    models: Sequence[ModelSpec],
    *,
    baseline_name: str = "persistence",
    peak_tolerance: float = 0.10,
    max_fold_tolerance: float = 0.10,
) -> SelectionResult:
    """Select a model using training-only backtest evidence.

    Eligibility requires:

    * no fold where the forecast left the observed training range;
    * pooled peak-period MAE no more than ``peak_tolerance`` worse than the
      persistence baseline;
    * worst-fold MAE no more than ``max_fold_tolerance`` worse than persistence.

    Among eligible models, the lowest mean fold MAE wins.  If none qualify, the
    baseline is used.  The tolerances are part of the validation protocol and
    are not tuned on the holdout.
    """
    if summary.empty:
        raise ValueError("summary is empty")
    if baseline_name not in set(summary["model"]):
        raise ValueError(f"baseline {baseline_name!r} is missing from the backtest")

    by_name = {spec.name: spec for spec in models}
    table = summary.copy()
    baseline_row = table.loc[table["model"] == baseline_name].iloc[0]
    baseline_peak = float(baseline_row["pooled_peak_mae"])
    baseline_max_fold = float(baseline_row["max_fold_mae"])
    max_folds = int(table["n_folds"].max())

    table["eligible"] = (
        (table["n_folds"] == max_folds)
        & (table["n_out_of_range_folds"] == 0)
        & (table["pooled_peak_mae"] <= baseline_peak * (1.0 + peak_tolerance))
        & (table["max_fold_mae"] <= baseline_max_fold * (1.0 + max_fold_tolerance))
    )
    eligible = table.loc[table["eligible"]].sort_values(
        ["mean_fold_mae", "pooled_mae"], kind="mergesort"
    )
    if eligible.empty:
        selected_name = baseline_name
        reason = (
            "No candidate passed the range and peak-risk eligibility checks; "
            "using the persistence baseline."
        )
    else:
        best_mean = float(eligible.iloc[0]["mean_fold_mae"])
        baseline_eligible = eligible.loc[eligible["model"] == baseline_name]
        if (
            not baseline_eligible.empty
            and float(baseline_eligible.iloc[0]["mean_fold_mae"]) <= best_mean + 1e-12
        ):
            selected_name = baseline_name
            reason = (
                "Persistence tied for the lowest mean fold MAE among eligible "
                "models and is preferred for simplicity."
            )
        else:
            selected_name = str(eligible.iloc[0]["model"])
            reason = (
                "Lowest mean fold MAE among models that stayed inside the training "
                "range and did not materially worsen peak-period error."
            )

    table["selected"] = table["model"] == selected_name
    table = table.sort_values(
        ["selected", "eligible", "mean_fold_mae"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if selected_name not in by_name:
        selected_name = baseline_name
    return SelectionResult(
        selected_model=selected_name,
        selected_spec=by_name[selected_name],
        table=table,
        reason=reason,
    )
