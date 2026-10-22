from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cwfc.pipeline import (
    REQUIRED_PREDICTION_COLUMNS,
    run_forecast,
    validate_predictions,
    write_predictions,
)


def write_synthetic_dataset(
    directory: Path,
    *,
    n_timestamps: int = 60,
    n_vms: int = 5,
    holdout_offset: float = 0.0,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    timestamps = np.arange(n_timestamps) * 300
    rows = []
    for vm_index in range(n_vms):
        for index, timestamp in enumerate(timestamps):
            value = 8.0 + 2.0 * np.sin(2 * np.pi * index / 24 + vm_index * 0.2)
            value += 0.05 * vm_index
            if index >= n_timestamps - 24:
                value += holdout_offset
            value = float(np.clip(value, 0.0, 100.0))
            rows.append(
                {
                    "timestamp": int(timestamp),
                    "vm_id": f"vm-{vm_index}",
                    "min_cpu": value * 0.8,
                    "max_cpu": min(100.0, value * 1.1),
                    "avg_cpu": value,
                }
            )
    cpu = pd.DataFrame(rows)
    meta = pd.DataFrame({"vm_id": [f"vm-{index}" for index in range(n_vms)]})
    cpu.to_csv(directory / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
    meta.to_csv(directory / "vm_metadata.csv.gz", index=False, compression="gzip")
    return directory


def test_pipeline_produces_valid_24_row_predictions(tmp_path):
    data_dir = write_synthetic_dataset(tmp_path / "data")
    result = run_forecast(data_dir, horizon=24, include_holdout=False)
    predictions = result.predictions

    assert list(predictions.columns) == list(REQUIRED_PREDICTION_COLUMNS)
    assert len(predictions) == 24
    assert predictions["timestamp"].is_monotonic_increasing
    assert predictions["active_vm_count"].tolist() == [5] * 24
    assert np.all(np.isfinite(predictions["aggregate_cpu_utilization"]))
    assert predictions["aggregate_cpu_utilization"].between(0, 100).all()

    expected = result.dataset.targets["timestamp"].tail(24)
    validate_predictions(predictions, expected_timestamps=expected, expected_rows=24)

    output = write_predictions(
        result,
        tmp_path / "predictions.csv",
        expected_timestamps=expected,
        expected_rows=24,
    )
    reloaded = pd.read_csv(output)
    assert list(reloaded.columns) == list(REQUIRED_PREDICTION_COLUMNS)
    assert len(reloaded) == 24


def test_predictions_do_not_depend_on_holdout_values(tmp_path):
    normal = write_synthetic_dataset(tmp_path / "normal", holdout_offset=0.0)
    shifted = write_synthetic_dataset(tmp_path / "shifted", holdout_offset=20.0)

    first = run_forecast(normal, horizon=24, include_holdout=False)
    second = run_forecast(shifted, horizon=24, include_holdout=False)

    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    assert first.selection.selected_model == second.selection.selected_model


def test_validate_predictions_rejects_wrong_schema():
    frame = pd.DataFrame(
        {
            "timestamp": [0, 300],
            "active_vm_count": [1, 1],
            "aggregate_cpu_utilization": [9.0, 9.1],
        }
    )
    with pytest.raises(ValueError):
        validate_predictions(frame, expected_rows=24)
    with pytest.raises(ValueError):
        validate_predictions(frame[["timestamp", "active_vm_count"]], expected_rows=2)
