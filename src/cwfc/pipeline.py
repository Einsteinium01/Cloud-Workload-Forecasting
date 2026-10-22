"""End-to-end forecasting pipeline.

The pipeline is intentionally explicit about the information boundary:

1. load the release and construct the full target series;
2. split the final 24 timestamps off as the holdout;
3. run rolling-origin validation using only the training window;
4. select a model from that validation evidence;
5. fit the selected model on the training window and forecast the holdout
   timestamps.

Holdout target values are not returned unless ``include_holdout=True`` is
explicitly requested for the evaluation stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    ACTIVE_VM_COUNT,
    AGGREGATE_CPU,
    CPU_FILE,
    META_FILE,
    Dataset,
    load_dataset,
    split_holdout,
)
from .metrics import interval_margins, summarize_metrics
from .models import (
    MOVING_AVERAGE_6,
    MOVING_AVERAGE_12,
    PERSISTENCE,
    SEASONAL_NAIVE_12,
    ModelSpec,
    available_models,
)
from .validation import (
    BacktestResult,
    SelectionResult,
    default_min_train,
    run_backtest,
    select_model,
    summarize_backtest,
)


REQUIRED_PREDICTION_COLUMNS = (
    "timestamp",
    "active_vm_count",
    "aggregate_cpu_utilization",
)


@dataclass
class ForecastResult:
    dataset: Dataset
    train: pd.DataFrame
    holdout_actuals: pd.DataFrame | None
    cpu_backtest: BacktestResult
    cpu_summary: pd.DataFrame
    selection: SelectionResult
    active_backtest: BacktestResult
    active_summary: pd.DataFrame
    active_selection: SelectionResult
    predictions: pd.DataFrame
    interval_frame: pd.DataFrame
    intervals: dict[str, float]
    meta: dict[str, Any]


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _active_models(history_length: int) -> list[ModelSpec]:
    """Small, count-appropriate candidate set for active VM count."""
    candidates = [PERSISTENCE, MOVING_AVERAGE_6, MOVING_AVERAGE_12, SEASONAL_NAIVE_12]
    return [spec for spec in candidates if spec.min_history <= history_length]


def run_forecast(
    data_dir: str | Path,
    *,
    horizon: int = 24,
    min_train: int | None = None,
    step: int = 12,
    include_holdout: bool = False,
    strict: bool = True,
) -> ForecastResult:
    """Run validation, model selection, and the final holdout forecast."""
    data_dir = Path(data_dir)
    dataset = load_dataset(data_dir, strict=strict)
    targets = dataset.targets
    train, holdout = split_holdout(targets, horizon=horizon)

    if len(train) <= horizon:
        raise ValueError(
            "The training window must be longer than the forecast horizon."
        )
    if len(train) < 2:
        raise ValueError("Not enough training observations to forecast.")

    models = available_models(len(train))
    if min_train is None:
        min_train = default_min_train(len(train), horizon=horizon)
    # Keep the default validation origins aligned with the challenge protocol:
    # 4 folds for 110 training observations (origins 50, 62, 74, 86).
    cpu_backtest = run_backtest(
        train,
        models,
        target=AGGREGATE_CPU,
        horizon=horizon,
        min_train=min_train,
        step=step,
        baseline_name=PERSISTENCE.name,
    )
    cpu_summary = summarize_backtest(cpu_backtest)
    selection = select_model(cpu_summary, models, baseline_name=PERSISTENCE.name)

    history_cpu = train[AGGREGATE_CPU].to_numpy(dtype=float)
    cpu_forecast = selection.selected_spec.predict(history_cpu, horizon)
    cpu_forecast = np.clip(np.asarray(cpu_forecast, dtype=float), 0.0, 100.0)

    active_models = _active_models(len(train))
    active_backtest = run_backtest(
        train,
        active_models,
        target=ACTIVE_VM_COUNT,
        horizon=horizon,
        min_train=min_train,
        step=step,
        baseline_name=PERSISTENCE.name,
    )
    active_summary = summarize_backtest(active_backtest)
    active_selection = select_model(
        active_summary, active_models, baseline_name=PERSISTENCE.name
    )
    history_active = train[ACTIVE_VM_COUNT].to_numpy(dtype=float)
    active_forecast = active_selection.selected_spec.predict(history_active, horizon)
    active_forecast = np.clip(np.rint(np.asarray(active_forecast, dtype=float)), 0, None)

    selected_predictions = cpu_backtest.predictions.loc[
        cpu_backtest.predictions["model"] == selection.selected_model
    ]
    intervals = interval_margins(
        selected_predictions["y_true"].to_numpy(dtype=float),
        selected_predictions["y_pred"].to_numpy(dtype=float),
        levels=(0.8, 0.9),
    )

    prediction_frame = pd.DataFrame(
        {
            "timestamp": holdout["timestamp"].to_numpy(dtype=np.int64),
            "active_vm_count": active_forecast.astype(np.int64),
            "aggregate_cpu_utilization": cpu_forecast.astype(float),
        }
    )
    interval_frame = prediction_frame[["timestamp", "aggregate_cpu_utilization"]].copy()
    for level, margin in intervals.items():
        interval_frame[f"lower_{level.replace('.', '')}"] = (
            interval_frame["aggregate_cpu_utilization"] - margin
        )
        interval_frame[f"upper_{level.replace('.', '')}"] = (
            interval_frame["aggregate_cpu_utilization"] + margin
        )
    interval_frame = interval_frame.clip(lower=0.0, upper=100.0)

    meta: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "n_observations": int(len(targets)),
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "train_start": int(train["timestamp"].iloc[0]),
        "train_end": int(train["timestamp"].iloc[-1]),
        "holdout_start": int(holdout["timestamp"].iloc[0]),
        "holdout_end": int(holdout["timestamp"].iloc[-1]),
        "horizon": int(horizon),
        "validation_step": int(step),
        "min_train": int(min_train),
        "validation_origins": [int(origin) for origin in cpu_backtest.origins],
        "selected_model": selection.selected_model,
        "selected_model_family": selection.selected_spec.family,
        "selected_model_description": selection.selected_spec.description,
        "selection_reason": selection.reason,
        "active_model": active_selection.selected_model,
        "active_model_description": active_selection.selected_spec.description,
        "intervals": {key: float(value) for key, value in intervals.items()},
        "candidate_models": [spec.name for spec in models],
        "active_candidates": [spec.name for spec in active_models],
    }
    cpu_path = data_dir / CPU_FILE
    meta_path = data_dir / META_FILE
    if cpu_path.exists():
        meta["cpu_file_sha256"] = file_sha256(cpu_path)
    if meta_path.exists():
        meta["meta_file_sha256"] = file_sha256(meta_path)

    holdout_return = holdout.copy() if include_holdout else None
    return ForecastResult(
        dataset=dataset,
        train=train,
        holdout_actuals=holdout_return,
        cpu_backtest=cpu_backtest,
        cpu_summary=cpu_summary,
        selection=selection,
        active_backtest=active_backtest,
        active_summary=active_summary,
        active_selection=active_selection,
        predictions=prediction_frame,
        interval_frame=interval_frame,
        intervals=intervals,
        meta=meta,
    )


def validate_predictions(
    predictions: pd.DataFrame,
    *,
    expected_timestamps: pd.Series | np.ndarray | list[int] | None = None,
    expected_rows: int = 24,
) -> None:
    """Validate the submission contract before writing predictions.csv."""
    if list(predictions.columns) != list(REQUIRED_PREDICTION_COLUMNS):
        raise ValueError(
            "predictions must have exactly these columns, in order: "
            + ", ".join(REQUIRED_PREDICTION_COLUMNS)
        )
    if len(predictions) != expected_rows:
        raise ValueError(
            f"predictions must contain exactly {expected_rows} rows, got {len(predictions)}"
        )
    if predictions["timestamp"].duplicated().any():
        raise ValueError("predictions contain duplicate timestamps")
    if not predictions["timestamp"].is_monotonic_increasing:
        raise ValueError("predictions must be in chronological order")
    if expected_timestamps is not None:
        expected = np.asarray(expected_timestamps, dtype=np.int64)
        actual = predictions["timestamp"].to_numpy(dtype=np.int64)
        if not np.array_equal(actual, expected):
            raise ValueError("prediction timestamps do not match the holdout timestamps")
    numeric = predictions[["active_vm_count", "aggregate_cpu_utilization"]].to_numpy(
        dtype=float
    )
    if not np.all(np.isfinite(numeric)):
        raise ValueError("predictions contain non-finite values")
    if (predictions["active_vm_count"] < 0).any():
        raise ValueError("active_vm_count must be non-negative")
    cpu = predictions["aggregate_cpu_utilization"].to_numpy(dtype=float)
    if np.any(cpu < 0.0) or np.any(cpu > 100.0):
        raise ValueError("aggregate_cpu_utilization must be in [0, 100]")


def write_predictions(
    result: ForecastResult,
    path: str | Path,
    *,
    expected_timestamps: pd.Series | np.ndarray | list[int] | None = None,
    expected_rows: int = 24,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = result.predictions[list(REQUIRED_PREDICTION_COLUMNS)].copy()
    validate_predictions(
        frame, expected_timestamps=expected_timestamps, expected_rows=expected_rows
    )
    frame.to_csv(path, index=False, float_format="%.6f")
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def write_manifest(result: ForecastResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result.meta, handle, indent=2, default=_json_safe, sort_keys=True)
        handle.write("\n")
    return path


@dataclass
class HoldoutEvaluation:
    selected_model: str
    cpu_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    cpu_forecast: pd.DataFrame
    interval_coverage: dict[str, float]
    error_by_horizon: pd.DataFrame


def evaluate_holdout(result: ForecastResult) -> HoldoutEvaluation:
    """Score the frozen forecasts against the public holdout.

    This function is called only after the prediction file has been generated
    and must not feed back into model selection.
    """
    if result.holdout_actuals is None:
        raise ValueError(
            "Holdout values are not available. Re-run with include_holdout=True "
            "for the evaluation stage."
        )
    holdout = result.holdout_actuals
    history_cpu = result.train[AGGREGATE_CPU].to_numpy(dtype=float)
    actual = holdout[AGGREGATE_CPU].to_numpy(dtype=float)
    forecast = result.predictions[AGGREGATE_CPU].to_numpy(dtype=float)

    baseline_spec = PERSISTENCE
    baseline_forecast = baseline_spec.predict(history_cpu, len(actual))
    cpu_metrics = summarize_metrics(
        actual,
        forecast,
        y_train=history_cpu,
        baseline_pred=baseline_forecast,
        seasonality=1,
        peak_quantile=0.75,
    )
    baseline_metrics = summarize_metrics(
        actual,
        baseline_forecast,
        y_train=history_cpu,
        baseline_pred=baseline_forecast,
        seasonality=1,
        peak_quantile=0.75,
    )

    coverage: dict[str, float] = {}
    errors = np.abs(actual - forecast)
    for level_key, margin in result.intervals.items():
        coverage[level_key] = float(np.mean(errors <= margin))

    error_by_horizon = pd.DataFrame(
        {
            "step": np.arange(1, len(actual) + 1),
            "timestamp": holdout["timestamp"].to_numpy(dtype=np.int64),
            "y_true": actual,
            "y_pred": forecast,
            "error": actual - forecast,
            "abs_error": errors,
        }
    )
    error_by_horizon["baseline_pred"] = baseline_forecast
    error_by_horizon["baseline_abs_error"] = np.abs(actual - baseline_forecast)

    cpu_forecast = holdout[["timestamp", AGGREGATE_CPU]].copy()
    cpu_forecast["predicted"] = forecast
    cpu_forecast["persistence"] = baseline_forecast
    cpu_forecast["error"] = actual - forecast
    cpu_forecast["abs_error"] = errors
    for level_key, margin in result.intervals.items():
        cpu_forecast[f"lower_{level_key.replace('.', '')}"] = np.clip(
            forecast - margin, 0.0, 100.0
        )
        cpu_forecast[f"upper_{level_key.replace('.', '')}"] = np.clip(
            forecast + margin, 0.0, 100.0
        )

    return HoldoutEvaluation(
        selected_model=result.selection.selected_model,
        cpu_metrics=cpu_metrics,
        baseline_metrics=baseline_metrics,
        cpu_forecast=cpu_forecast,
        interval_coverage=coverage,
        error_by_horizon=error_by_horizon,
    )
