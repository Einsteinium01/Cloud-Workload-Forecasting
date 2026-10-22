"""Data loading, validation, and construction of the challenge targets.

The challenge release contains VM-level five-minute CPU readings and VM
metadata.  This module deliberately keeps the raw data separate from the
aggregate targets:

* ``load_dataset`` reads and validates the release.
* ``build_target_series`` constructs ``active_vm_count`` and
  ``aggregate_cpu_utilization`` exactly as defined in ``DATASET.md``.
* ``split_holdout`` performs the required chronological 110/24 split without
  looking at the holdout target values.

No interpolation or forward filling is performed.  Missing CPU readings are
excluded from the aggregate mean and reported in the audit.  A VM is counted
at most once per timestamp because ``nunique`` is used on ``vm_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CPU_FILE = "vm_cpu_readings.csv.gz"
META_FILE = "vm_metadata.csv.gz"
CPU_COLUMNS = ("timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu")
META_COLUMNS = (
    "vm_id",
    "subscription_id",
    "deployment_id",
    "timestamp_vm_created",
    "timestamp_vm_deleted",
    "max_cpu",
    "avg_cpu",
    "p95_max_cpu",
    "vm_category",
    "virtual_core_bucket",
    "memory_gb_bucket",
)
META_REQUIRED_COLUMNS = ("vm_id",)
ACTIVE_VM_COUNT = "active_vm_count"
AGGREGATE_CPU = "aggregate_cpu_utilization"
TARGET_COLUMNS = (ACTIVE_VM_COUNT, AGGREGATE_CPU)


class DataValidationError(ValueError):
    """Raised when the supplied release violates the challenge contract."""


@dataclass
class DataAudit:
    """Serialisable record of every data-quality check performed."""

    checks: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add(self, name: str, value: Any) -> None:
        self.checks[name] = value

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    @property
    def passed(self) -> bool:
        """True when the checks that matter for target construction passed."""
        required = (
            "required_columns_present",
            "no_duplicate_vm_timestamps",
            "vm_observation_counts_complete",
            "metadata_one_row_per_vm",
        )
        return all(bool(self.checks.get(key, False)) for key in required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "warnings": self.warnings,
        }

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")


@dataclass
class Dataset:
    """Raw data plus the two required aggregate target series."""

    cpu: pd.DataFrame
    meta: pd.DataFrame | None
    targets: pd.DataFrame
    audit: DataAudit
    grid_seconds: int = 300

    @property
    def target_series(self) -> pd.DataFrame:
        return self.targets.copy()

    def split(self, horizon: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
        return split_holdout(self.targets, horizon=horizon)

    def write_audit(self, path: str | Path) -> None:
        self.audit.write(path)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DataValidationError(
            f"{name} is missing required column(s): {', '.join(missing)}"
        )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset file not found: {path}. "
            "Download the data-v1 release and extract it into challenge_data/."
        )
    return pd.read_csv(path)


def load_raw(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Read the release files without modifying their contents."""
    data_dir = Path(data_dir)
    cpu_path = data_dir / CPU_FILE
    meta_path = data_dir / META_FILE

    cpu = _read_csv(cpu_path)
    _require_columns(cpu, CPU_COLUMNS, CPU_FILE)

    meta: pd.DataFrame | None
    if meta_path.exists():
        meta = _read_csv(meta_path)
        _require_columns(meta, META_REQUIRED_COLUMNS, META_FILE)
    else:
        meta = None
        # Metadata is not required to construct the two targets.  The analysis
        # stage will report that segment analysis is unavailable.

    return cpu, meta


def _clean_cpu(
    cpu: pd.DataFrame,
    *,
    grid_seconds: int,
    strict: bool,
    audit: DataAudit,
) -> pd.DataFrame:
    frame = cpu.copy()
    audit.add("required_columns_present", True)

    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise")
    frame["vm_id"] = frame["vm_id"].astype(str)
    for column in ("min_cpu", "max_cpu", "avg_cpu"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    audit.add("n_cpu_rows", int(len(frame)))
    audit.add("n_unique_vms", int(frame["vm_id"].nunique()))
    audit.add("n_unique_timestamps", int(frame["timestamp"].nunique()))
    audit.add("timestamp_min", int(frame["timestamp"].min()))
    audit.add("timestamp_max", int(frame["timestamp"].max()))

    non_finite = ~np.isfinite(frame["avg_cpu"].to_numpy(dtype=float))
    audit.add("n_non_finite_avg_cpu", int(non_finite.sum()))
    if non_finite.any():
        audit.warn(
            f"{int(non_finite.sum())} rows had a non-finite avg_cpu and were "
            "excluded from the aggregate target."
        )
        frame = frame.loc[~non_finite].copy()

    outside = (frame["avg_cpu"] < 0.0) | (frame["avg_cpu"] > 100.0)
    audit.add("n_avg_cpu_outside_0_100", int(outside.sum()))
    if outside.any():
        message = (
            f"{int(outside.sum())} rows had avg_cpu outside [0, 100]; "
            "they were excluded from the aggregate target."
        )
        if strict:
            raise DataValidationError(message)
        audit.warn(message)
        frame = frame.loc[~outside].copy()

    order_bad = (
        (frame["min_cpu"] > frame["avg_cpu"] + 1e-9)
        | (frame["avg_cpu"] > frame["max_cpu"] + 1e-9)
    )
    audit.add("n_min_avg_max_order_violations", int(order_bad.sum()))
    if order_bad.any():
        audit.warn(
            f"{int(order_bad.sum())} rows violated min_cpu <= avg_cpu <= max_cpu."
        )

    duplicates = frame.duplicated(["vm_id", "timestamp"], keep=False)
    audit.add("n_duplicate_vm_timestamp_rows", int(duplicates.sum()))
    audit.add("no_duplicate_vm_timestamps", not bool(duplicates.any()))
    if duplicates.any():
        message = (
            f"{int(duplicates.sum())} duplicate (vm_id, timestamp) rows were found."
        )
        if strict:
            raise DataValidationError(message)
        audit.warn(message + " The first row per pair was kept.")
        frame = frame.drop_duplicates(["vm_id", "timestamp"], keep="first")

    frame = frame.sort_values(["timestamp", "vm_id"], kind="mergesort").reset_index(
        drop=True
    )

    counts = frame.groupby("vm_id", sort=False).size()
    audit.add("observations_per_vm_min", int(counts.min()))
    audit.add("observations_per_vm_max", int(counts.max()))
    audit.add("vm_observation_counts_complete", bool(counts.nunique() == 1))
    audit.add("n_observations_per_vm", int(counts.iloc[0]) if len(counts) else 0)

    timestamps = np.sort(frame["timestamp"].unique())
    diffs = np.diff(timestamps)
    cadence_mode = (
        int(pd.Series(diffs).mode().iloc[0]) if len(diffs) else int(grid_seconds)
    )
    audit.add("cadence_seconds_mode", cadence_mode)
    audit.add("n_irregular_intervals", int(np.sum(diffs != grid_seconds)))
    audit.add("max_gap_seconds", int(diffs.max()) if len(diffs) else 0)
    audit.add("regular_five_minute_cadence", bool(np.all(diffs == grid_seconds)))
    if len(diffs) and np.any(diffs != grid_seconds):
        audit.warn(
            "The timestamp grid is not perfectly regular at five-minute cadence. "
            "Observed timestamps are preserved; no values are interpolated."
        )

    return frame


def _clean_metadata(
    meta: pd.DataFrame | None,
    cpu_vm_ids: set[str],
    *,
    strict: bool,
    audit: DataAudit,
) -> pd.DataFrame | None:
    if meta is None:
        audit.add("metadata_available", False)
        audit.add("metadata_one_row_per_vm", False)
        audit.warn("vm_metadata.csv.gz is absent; segment analysis is unavailable.")
        return None

    frame = meta.copy()
    frame["vm_id"] = frame["vm_id"].astype(str)
    audit.add("metadata_available", True)
    audit.add("n_metadata_rows", int(len(frame)))
    audit.add("n_metadata_unique_vms", int(frame["vm_id"].nunique()))

    duplicated = frame["vm_id"].duplicated(keep=False)
    audit.add("n_duplicate_metadata_vm_ids", int(duplicated.sum()))
    if duplicated.any():
        message = f"{int(duplicated.sum())} duplicate metadata vm_id rows were found."
        if strict:
            raise DataValidationError(message)
        audit.warn(message + " The first row per vm_id was kept.")
        frame = frame.drop_duplicates("vm_id", keep="first")

    cpu_ids = set(cpu_vm_ids)
    meta_ids = set(frame["vm_id"])
    missing_meta = sorted(cpu_ids - meta_ids)
    extra_meta = sorted(meta_ids - cpu_ids)
    audit.add("n_vms_without_metadata", len(missing_meta))
    audit.add("n_metadata_without_cpu", len(extra_meta))
    coverage_ok = not missing_meta and not extra_meta
    audit.add("metadata_covers_all_cpu_vms", coverage_ok)
    audit.add("metadata_one_row_per_vm", bool(coverage_ok and not duplicated.any()))
    if not coverage_ok:
        message = (
            "Metadata coverage does not match the CPU VM population "
            f"({len(missing_meta)} CPU VMs without metadata, "
            f"{len(extra_meta)} metadata VMs without CPU readings)."
        )
        if strict:
            raise DataValidationError(message)
        audit.warn(message)

    # A handful of descriptive checks that make the data card claims auditable.
    if "timestamp_vm_created" in frame.columns:
        created = pd.to_numeric(frame["timestamp_vm_created"], errors="coerce")
        audit.add("vm_created_min", int(created.min()))
        audit.add("vm_created_max", int(created.max()))
    if "timestamp_vm_deleted" in frame.columns:
        deleted = pd.to_numeric(frame["timestamp_vm_deleted"], errors="coerce")
        audit.add("vm_deleted_min", int(deleted.min()))
        audit.add("vm_deleted_max", int(deleted.max()))
    if "vm_category" in frame.columns:
        audit.add(
            "vm_category_counts",
            {str(key): int(value) for key, value in frame["vm_category"].value_counts().items()},
        )
    if "virtual_core_bucket" in frame.columns:
        audit.add(
            "virtual_core_bucket_counts",
            {
                str(key): int(value)
                for key, value in frame["virtual_core_bucket"].value_counts().items()
            },
        )

    return frame


def build_target_series(
    cpu: pd.DataFrame,
    meta: pd.DataFrame | None = None,
    *,
    grid_seconds: int = 300,
    strict: bool = True,
) -> tuple[pd.DataFrame, DataAudit, pd.DataFrame | None]:
    """Construct the required aggregate targets from VM-level readings.

    ``active_vm_count`` counts distinct VM IDs with a valid CPU reading at each
    timestamp.  ``aggregate_cpu_utilization`` is the arithmetic mean of the
    VM-level ``avg_cpu`` values at that timestamp, in percent.  The two targets
    are computed on the observed five-minute timestamps; the function does not
    invent observations for missing timestamps.
    """
    audit = DataAudit()
    frame = _clean_cpu(cpu, grid_seconds=grid_seconds, strict=strict, audit=audit)
    meta_clean = _clean_metadata(
        meta, set(frame["vm_id"]), strict=strict, audit=audit
    )

    grouped = (
        frame.groupby("timestamp", sort=True)
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .reset_index()
    )
    grouped["timestamp"] = grouped["timestamp"].astype("int64")
    grouped["active_vm_count"] = grouped["active_vm_count"].astype("int64")
    grouped["aggregate_cpu_utilization"] = grouped[
        "aggregate_cpu_utilization"
    ].astype(float)

    audit.add("n_target_timestamps", int(len(grouped)))
    audit.add("active_vm_count_min", int(grouped["active_vm_count"].min()))
    audit.add("active_vm_count_max", int(grouped["active_vm_count"].max()))
    audit.add(
        "aggregate_cpu_utilization_min",
        float(grouped["aggregate_cpu_utilization"].min()),
    )
    audit.add(
        "aggregate_cpu_utilization_max",
        float(grouped["aggregate_cpu_utilization"].max()),
    )
    audit.add(
        "aggregate_cpu_utilization_mean",
        float(grouped["aggregate_cpu_utilization"].mean()),
    )
    audit.add(
        "aggregate_cpu_utilization_std",
        float(grouped["aggregate_cpu_utilization"].std(ddof=0)),
    )
    audit.add(
        "n_target_timestamps_with_missing_cpu",
        int(grouped["aggregate_cpu_utilization"].isna().sum()),
    )

    # Keep metadata rows aligned with the CPU population; unused rows are
    # reported by ``n_metadata_without_cpu`` above.
    if meta_clean is not None:
        meta_clean = meta_clean.loc[meta_clean["vm_id"].isin(set(frame["vm_id"]))].copy()

    return grouped, audit, meta_clean


def load_dataset(
    data_dir: str | Path,
    *,
    grid_seconds: int = 300,
    strict: bool = True,
) -> Dataset:
    """Load, validate, and aggregate the challenge release."""
    cpu_raw, meta_raw = load_raw(data_dir)
    targets, audit, meta = build_target_series(
        cpu_raw, meta_raw, grid_seconds=grid_seconds, strict=strict
    )
    # Store a cleaned CPU frame for analysis.  The target builder already
    # removed invalid/duplicate rows; rebuild the same view here deterministically.
    cpu = cpu_raw.copy()
    cpu["timestamp"] = pd.to_numeric(cpu["timestamp"], errors="raise").astype("int64")
    cpu["vm_id"] = cpu["vm_id"].astype(str)
    for column in ("min_cpu", "max_cpu", "avg_cpu"):
        cpu[column] = pd.to_numeric(cpu[column], errors="coerce")
    cpu = cpu.dropna(subset=["avg_cpu"])
    cpu = cpu.loc[(cpu["avg_cpu"] >= 0.0) & (cpu["avg_cpu"] <= 100.0)]
    cpu = cpu.drop_duplicates(["vm_id", "timestamp"], keep="first")
    cpu = cpu.sort_values(["timestamp", "vm_id"], kind="mergesort").reset_index(
        drop=True
    )
    if meta is not None:
        merge_columns = [
            column
            for column in (
                "vm_id",
                "vm_category",
                "virtual_core_bucket",
                "memory_gb_bucket",
                "timestamp_vm_created",
                "timestamp_vm_deleted",
            )
            if column in meta.columns
        ]
        cpu = cpu.merge(
            meta[merge_columns],
            on="vm_id",
            how="left",
            validate="many_to_one",
        )
    return Dataset(cpu=cpu, meta=meta, targets=targets, audit=audit, grid_seconds=grid_seconds)


def split_holdout(
    targets: pd.DataFrame,
    *,
    horizon: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split the final ``horizon`` observations from the rest.

    The returned holdout frame contains timestamps and observed values.  The
    prediction pipeline passes only the training frame (and the holdout
    timestamps) to the forecaster, so holdout target values cannot enter model
    fitting or selection.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if len(targets) <= horizon:
        raise DataValidationError(
            f"Need more than {horizon} target observations, got {len(targets)}."
        )
    ordered = targets.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return ordered.iloc[:-horizon].copy(), ordered.iloc[-horizon:].copy()


def target_stats(targets: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for the two required targets."""
    return targets[list(TARGET_COLUMNS)].describe().T
