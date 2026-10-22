import numpy as np
import pandas as pd
import pytest

from cwfc.data import (
    ACTIVE_VM_COUNT,
    AGGREGATE_CPU,
    DataValidationError,
    build_target_series,
    split_holdout,
)


def make_cpu(rows):
    return pd.DataFrame(
        rows,
        columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"],
    )


def test_targets_count_distinct_vms_and_mean_cpu():
    cpu = make_cpu(
        [
            (0, "a", 1.0, 3.0, 2.0),
            (0, "b", 4.0, 8.0, 6.0),
            (300, "a", 2.0, 4.0, 3.0),
            (300, "b", 5.0, 9.0, 7.0),
        ]
    )
    targets, audit, meta = build_target_series(cpu)
    assert meta is None
    assert targets[ACTIVE_VM_COUNT].tolist() == [2, 2]
    assert targets[AGGREGATE_CPU].tolist() == [4.0, 5.0]
    assert audit.checks["n_target_timestamps"] == 2


def test_duplicate_vm_timestamp_is_rejected_in_strict_mode():
    cpu = make_cpu(
        [
            (0, "a", 1.0, 3.0, 2.0),
            (0, "a", 1.0, 3.0, 2.0),
        ]
    )
    with pytest.raises(DataValidationError):
        build_target_series(cpu, strict=True)


def test_missing_avg_cpu_is_excluded_from_mean():
    cpu = make_cpu(
        [
            (0, "a", 1.0, 3.0, 2.0),
            (0, "b", 4.0, 8.0, np.nan),
            (300, "a", 2.0, 4.0, 3.0),
        ]
    )
    targets, audit, _ = build_target_series(cpu, strict=True)
    assert targets.loc[targets.timestamp == 0, ACTIVE_VM_COUNT].iloc[0] == 1
    assert targets.loc[targets.timestamp == 0, AGGREGATE_CPU].iloc[0] == 2.0
    assert audit.checks["n_non_finite_avg_cpu"] == 1


def test_metadata_coverage_mismatch_is_rejected():
    cpu = make_cpu([(0, "a", 1.0, 3.0, 2.0)])
    meta = pd.DataFrame({"vm_id": ["b"]})
    with pytest.raises(DataValidationError):
        build_target_series(cpu, meta, strict=True)


def test_split_holdout_is_chronological_and_does_not_mutate_input():
    targets = pd.DataFrame(
        {
            "timestamp": np.arange(10) * 300,
            ACTIVE_VM_COUNT: np.arange(10) + 1,
            AGGREGATE_CPU: np.linspace(9.0, 10.0, 10),
        }
    )
    train, holdout = split_holdout(targets, horizon=3)
    assert len(train) == 7
    assert len(holdout) == 3
    assert train.timestamp.max() < holdout.timestamp.min()
    assert holdout.timestamp.tolist() == [2100, 2400, 2700]
    assert len(targets) == 10
