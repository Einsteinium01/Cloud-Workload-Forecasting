"""Generate the analysis artefacts and figures for the final report.

Example
-------
    python -m cwfc.analyze --data challenge_data --reports-dir reports
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import ACTIVE_VM_COUNT, AGGREGATE_CPU
from .metrics import mae, normalized_mae, peak_mae, rmse, smape
from .models import model_table
from .pipeline import HoldoutEvaluation, ForecastResult, evaluate_holdout, run_forecast
from .validation import summarize_backtest


COLORS = {
    "actual": "#1f1f1f",
    "model": "#1f77b4",
    "baseline": "#d62728",
    "accent": "#2ca02c",
    "muted": "#7f7f7f",
    "shade": "#bdbdbd",
    "orange": "#ff7f0e",
    "purple": "#9467bd",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 150,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.autolayout": False,
        }
    )


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _hours(timestamps: Iterable[float], origin: float) -> np.ndarray:
    return (np.asarray(list(timestamps), dtype=float) - origin) / 3600.0


def markdown_table(frame: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
    """Small dependency-free Markdown table formatter."""
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("" if pd.isna(value) else f"{float(value):{floatfmt}}")
            elif isinstance(value, (bool, np.bool_)):
                values.append("yes" if bool(value) else "no")
            elif pd.isna(value):
                values.append("")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
        if isinstance(value, (pd.Timestamp,)):
            return value.isoformat()
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return value.to_dict(orient="records")
        return value

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=convert, sort_keys=True)
        handle.write("\n")


def _vm_summary(dataset) -> pd.DataFrame:
    vm = (
        dataset.cpu.groupby("vm_id", sort=False)["avg_cpu"]
        .agg(mean_cpu="mean", std_cpu="std", min_cpu="min", max_cpu="max")
        .reset_index()
    )
    if dataset.meta is not None:
        columns = [
            column
            for column in (
                "vm_id",
                "vm_category",
                "virtual_core_bucket",
                "memory_gb_bucket",
            )
            if column in dataset.meta.columns
        ]
        vm = vm.merge(dataset.meta[columns], on="vm_id", how="left")
    return vm


def plot_target_series(targets: pd.DataFrame, split_timestamp: int, path: Path) -> None:
    origin = float(targets["timestamp"].iloc[0])
    hours = _hours(targets["timestamp"], origin)
    split_hour = (split_timestamp - origin) / 3600.0
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    for ax, column, label, color in (
        (axes[0], AGGREGATE_CPU, "Aggregate CPU utilization (%)", COLORS["model"]),
        (axes[1], ACTIVE_VM_COUNT, "Active VMs", COLORS["accent"]),
    ):
        ax.plot(hours, targets[column], color=color, linewidth=1.3, label="observed")
        ax.axvspan(split_hour, hours.max(), color=COLORS["shade"], alpha=0.18, label="holdout")
        ax.axvline(split_hour, color="black", linestyle="--", linewidth=0.9)
        ax.set_ylabel(label)
        ax.legend(loc="best")
    axes[1].set_xlabel("Hours from trace start")
    axes[0].set_title("Required five-minute targets: 110 training observations / 24 holdout observations")
    fig.tight_layout()
    _save(fig, path)


def plot_cpu_distribution(dataset, path: Path) -> None:
    vm = _vm_summary(dataset)
    cpu = dataset.cpu
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))

    ax = axes[0, 0]
    ax.hist(cpu["avg_cpu"], bins=80, color=COLORS["model"], alpha=0.85)
    ax.set_yscale("log")
    ax.axvline(float(cpu["avg_cpu"].mean()), color=COLORS["baseline"], linewidth=1.2, label="mean")
    ax.axvline(float(cpu["avg_cpu"].median()), color=COLORS["accent"], linewidth=1.2, label="median")
    ax.set_title("VM-level five-minute avg_cpu")
    ax.set_xlabel("CPU utilization (%)")
    ax.set_ylabel("Observations (log scale)")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(vm["mean_cpu"], bins=60, color=COLORS["purple"], alpha=0.85)
    ax.set_yscale("log")
    ax.axvline(float(vm["mean_cpu"].mean()), color=COLORS["baseline"], linewidth=1.2, label="mean")
    ax.axvline(float(vm["mean_cpu"].median()), color=COLORS["accent"], linewidth=1.2, label="median")
    ax.set_title("Per-VM mean CPU over the 11.1-hour window")
    ax.set_xlabel("Mean CPU utilization (%)")
    ax.set_ylabel("VMs (log scale)")
    ax.legend()

    ax = axes[1, 0]
    if "vm_category" in vm.columns:
        groups = [
            (str(key), group["mean_cpu"].to_numpy())
            for key, group in vm.groupby("vm_category", sort=False)
        ]
        ax.boxplot(
            [values for _, values in groups],
            tick_labels=[f"{name}\n(n={len(values):,})" for name, values in groups],
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#c6dbef", "edgecolor": COLORS["model"]},
            medianprops={"color": COLORS["baseline"]},
        )
        ax.set_ylabel("Per-VM mean CPU (%)")
        ax.set_title("Per-VM mean by workload category")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Metadata unavailable", ha="center", va="center")

    ax = axes[1, 1]
    if "virtual_core_bucket" in vm.columns:
        groups = [
            (str(key), group["mean_cpu"].to_numpy())
            for key, group in vm.groupby("virtual_core_bucket", sort=False)
        ]
        ax.boxplot(
            [values for _, values in groups],
            tick_labels=[f"{name}\n(n={len(values):,})" for name, values in groups],
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#d9f0d3", "edgecolor": COLORS["accent"]},
            medianprops={"color": COLORS["baseline"]},
        )
        ax.set_ylabel("Per-VM mean CPU (%)")
        ax.set_title("Per-VM mean by virtual-core bucket")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Metadata unavailable", ha="center", va="center")

    fig.suptitle("The aggregate mean hides a heavy-tailed, segment-dependent distribution", y=1.02)
    fig.tight_layout()
    _save(fig, path)


def plot_segment_series(dataset, split_timestamp: int, path: Path) -> None:
    if dataset.meta is None:
        return
    cpu = dataset.cpu
    available = [
        column
        for column in ("vm_category", "virtual_core_bucket", "memory_gb_bucket")
        if column in cpu.columns
    ]
    if not available:
        return
    origin = float(cpu["timestamp"].min())
    split_hour = (split_timestamp - origin) / 3600.0
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharex=True)
    for ax, column in zip(axes, available[:2]):
        pivot = cpu.pivot_table(
            index="timestamp", columns=column, values="avg_cpu", aggfunc="mean"
        )
        for key in pivot.columns:
            ax.plot(
                _hours(pivot.index, origin),
                pivot[key],
                linewidth=1.1,
                label=str(key),
            )
        ax.axvspan(split_hour, _hours(pivot.index, origin).max(), color=COLORS["shade"], alpha=0.15)
        ax.axvline(split_hour, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"Segment mean CPU by {column.replace('_', ' ')}")
        ax.set_xlabel("Hours from trace start")
        ax.set_ylabel("Mean CPU (%)")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, path)


def plot_autocorrelation(train_values: np.ndarray, path: Path) -> None:
    series = np.asarray(train_values, dtype=float)
    max_lag = min(96, len(series) - 2)
    acf = np.asarray(
        [np.corrcoef(series[:-lag], series[lag:])[0, 1] for lag in range(1, max_lag + 1)]
    )
    band = 1.96 / np.sqrt(len(series))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.vlines(np.arange(1, max_lag + 1), 0.0, acf, color=COLORS["model"], linewidth=1.1)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.axhspan(-band, band, color=COLORS["baseline"], alpha=0.12, label="95% white-noise band")
    for lag, label in ((12, "1 h"), (24, "2 h"), (48, "4 h"), (72, "6 h"), (96, "8 h")):
        if lag <= max_lag:
            ax.axvline(lag, color=COLORS["muted"], linestyle=":", linewidth=0.7)
            ax.text(lag, ax.get_ylim()[1] * 0.92, label, ha="center", fontsize=7, color=COLORS["muted"])
    ax.set_xlabel("Lag (five-minute steps)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Training-window autocorrelation: strong short memory and multi-hour persistence")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, path)


def plot_hotspot_counts(dataset, split_timestamp: int, path: Path) -> None:
    cpu = dataset.cpu
    timestamps = np.sort(cpu["timestamp"].unique())
    origin = float(timestamps[0])
    split_hour = (split_timestamp - origin) / 3600.0
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for threshold, color in zip((50, 80, 90, 95), ("#9ecae1", "#6baed6", "#3182bd", "#08519c")):
        counts = (
            cpu.loc[cpu["avg_cpu"] >= threshold]
            .groupby("timestamp")
            .size()
            .reindex(timestamps, fill_value=0)
        )
        ax.plot(_hours(timestamps, origin), counts, linewidth=1.2, color=color, label=f"avg_cpu >= {threshold}%")
    ax.axvspan(split_hour, _hours(timestamps, origin).max(), color=COLORS["shade"], alpha=0.15)
    ax.axvline(split_hour, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Hours from trace start")
    ax.set_ylabel("Number of VMs")
    ax.set_title("Hot-VM breadth remains material even when the aggregate mean is ~9%")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, path)


def _fold_frame(backtest_predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    return backtest_predictions.loc[
        backtest_predictions["model"] == model
    ].sort_values(["fold", "step"])


def plot_validation_folds(
    result: ForecastResult, path: Path
) -> None:
    predictions = result.cpu_backtest.predictions
    selected = result.selection.selected_model
    folds = sorted(predictions["fold"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.4), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, fold in zip(axes, folds[:4]):
        selected_frame = _fold_frame(predictions, selected)
        baseline_frame = _fold_frame(predictions, "persistence")
        selected_frame = selected_frame.loc[selected_frame["fold"] == fold]
        baseline_frame = baseline_frame.loc[baseline_frame["fold"] == fold]
        ax.plot(selected_frame["step"], selected_frame["y_true"], color=COLORS["actual"], linewidth=1.4, marker="o", markersize=2.5, label="actual")
        ax.plot(selected_frame["step"], selected_frame["y_pred"], color=COLORS["model"], linewidth=1.4, label=selected)
        ax.plot(baseline_frame["step"], baseline_frame["y_pred"], color=COLORS["baseline"], linewidth=1.0, linestyle="--", label="persistence")
        origin = int(selected_frame["origin_timestamp"].iloc[0])
        train_n = int(selected_frame["origin_index"].iloc[0])
        ax.set_title(
            f"Fold {int(fold)}: last train t={origin:,} (n={train_n})"
        )
        ax.set_xlabel("Forecast step (5 min)")
        ax.set_ylabel("CPU (%)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Expanding-window validation: four 2-hour forecast windows, always trained on the past", y=1.01)
    fig.tight_layout()
    _save(fig, path)


def plot_model_comparison(selection_table: pd.DataFrame, path: Path) -> None:
    table = selection_table.sort_values("mean_fold_mae", kind="mergesort").head(14)
    table = table.iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6), sharey=True)
    labels = table["model"].tolist()
    colors = [
        COLORS["model"] if bool(selected) else (COLORS["accent"] if bool(eligible) else COLORS["muted"])
        for selected, eligible in zip(table["selected"], table["eligible"])
    ]
    axes[0].barh(labels, table["mean_fold_mae"], color=colors)
    axes[0].set_xlabel("Mean fold MAE (lower is better)")
    axes[0].set_title("Average accuracy across four folds")
    axes[1].barh(labels, table["pooled_peak_mae"], color=colors)
    axes[1].set_xlabel("Pooled peak-period MAE")
    axes[1].set_title("Peak-period accuracy")
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    from matplotlib.patches import Patch

    axes[0].legend(
        handles=[
            Patch(color=COLORS["model"], label="selected"),
            Patch(color=COLORS["accent"], label="eligible"),
            Patch(color=COLORS["muted"], label="ineligible"),
        ],
        loc="lower right",
        fontsize=7,
    )
    fig.suptitle("Model comparison: average and peak-period error trade off against extrapolation risk", y=1.02)
    fig.tight_layout()
    _save(fig, path)


def plot_holdout_forecast(evaluation: HoldoutEvaluation, path: Path) -> None:
    frame = evaluation.cpu_forecast
    x = np.arange(1, len(frame) + 1)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.fill_between(x, frame["lower_090"], frame["upper_090"], color=COLORS["model"], alpha=0.12, label="90% residual interval")
    ax.fill_between(x, frame["lower_080"], frame["upper_080"], color=COLORS["model"], alpha=0.18, label="80% residual interval")
    ax.plot(x, frame[AGGREGATE_CPU], color=COLORS["actual"], linewidth=1.5, marker="o", markersize=3, label="actual holdout")
    ax.plot(x, frame["predicted"], color=COLORS["model"], linewidth=1.6, label="selected forecast")
    ax.plot(x, frame["persistence"], color=COLORS["baseline"], linewidth=1.0, linestyle="--", label="persistence")
    ax.set_xlabel("Holdout step (5 min)")
    ax.set_ylabel("CPU utilization (%)")
    ax.set_title(
        "Two-hour holdout: selected model MAE "
        f"{evaluation.cpu_metrics['mae']:.3f} pp vs persistence "
        f"{evaluation.baseline_metrics['mae']:.3f} pp"
    )
    ax.legend(fontsize=7, ncol=2)
    ax.set_xticks(x[::2])
    fig.tight_layout()
    _save(fig, path)


def plot_error_by_horizon(evaluation: HoldoutEvaluation, path: Path) -> None:
    frame = evaluation.error_by_horizon
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    ax.bar(frame["step"] - 0.2, frame["abs_error"], width=0.4, color=COLORS["model"], label="selected")
    ax.bar(frame["step"] + 0.2, frame["baseline_abs_error"], width=0.4, color=COLORS["baseline"], alpha=0.7, label="persistence")
    ax.set_xlabel("Holdout step (5 min)")
    ax.set_ylabel("Absolute error (pp)")
    ax.set_title("Absolute error by forecast step")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.scatter(frame["y_true"], frame["y_pred"], color=COLORS["model"], label="selected", s=24)
    ax.scatter(frame["y_true"], frame["baseline_pred"], color=COLORS["baseline"], label="persistence", s=18, alpha=0.7)
    low = min(frame["y_true"].min(), frame["y_pred"].min(), frame["baseline_pred"].min()) - 0.05
    high = max(frame["y_true"].max(), frame["y_pred"].max(), frame["baseline_pred"].max()) + 0.05
    ax.plot([low, high], [low, high], color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Actual CPU (%)")
    ax.set_ylabel("Predicted CPU (%)")
    ax.set_title("Actual vs predicted on the holdout")
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, path)


def plot_active_count(dataset, split_timestamp: int, path: Path) -> None:
    targets = dataset.targets
    origin = float(targets["timestamp"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    ax = axes[0]
    ax.plot(_hours(targets["timestamp"], origin), targets[ACTIVE_VM_COUNT], color=COLORS["accent"], linewidth=1.4)
    ax.axvspan((split_timestamp - origin) / 3600.0, _hours(targets["timestamp"], origin).max(), color=COLORS["shade"], alpha=0.15)
    ax.axvline((split_timestamp - origin) / 3600.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Hours from trace start")
    ax.set_ylabel("Active VMs")
    ax.set_title("Active VM count is constant in this complete-trace cohort")
    ax.set_ylim(0, max(1, int(targets[ACTIVE_VM_COUNT].max()) * 1.15))

    ax = axes[1]
    if dataset.meta is not None and "timestamp_vm_deleted" in dataset.meta.columns:
        deleted_hours = (
            pd.to_numeric(dataset.meta["timestamp_vm_deleted"], errors="coerce")
            - targets["timestamp"].max()
        ) / 3600.0
        ax.hist(deleted_hours, bins=30, color=COLORS["purple"], alpha=0.8)
        ax.set_yscale("log")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Deletion timestamp minus end of window (hours)")
        ax.set_ylabel("VMs (log scale)")
        ax.set_title("All deletion timestamps are after the holdout window")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Deletion metadata unavailable", ha="center", va="center")
    fig.tight_layout()
    _save(fig, path)


def build_tables(result: ForecastResult, evaluation: HoldoutEvaluation, reports_dir: Path) -> dict[str, pd.DataFrame]:
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    dataset = result.dataset
    targets = dataset.targets
    vm = _vm_summary(dataset)

    audit_frame = pd.DataFrame(
        [{"check": key, "value": value} for key, value in dataset.audit.checks.items()]
    )
    _write_text(tables_dir / "data_audit.md", markdown_table(audit_frame))

    target_frame = targets[[ACTIVE_VM_COUNT, AGGREGATE_CPU]].describe().T.reset_index()
    target_frame = target_frame.rename(columns={"index": "target"})
    _write_text(tables_dir / "target_stats.md", markdown_table(target_frame))

    segment_rows: list[dict[str, Any]] = []
    for column in ("vm_category", "virtual_core_bucket", "memory_gb_bucket"):
        if column not in vm.columns:
            continue
        grouped = vm.groupby(column, dropna=False)
        for key, group in grouped:
            segment_rows.append(
                {
                    "dimension": column,
                    "segment": str(key),
                    "n_vms": int(len(group)),
                    "share_vms": float(len(group) / len(vm)),
                    "mean_cpu": float(group["mean_cpu"].mean()),
                    "median_cpu": float(group["mean_cpu"].median()),
                    "p90_cpu": float(group["mean_cpu"].quantile(0.9)),
                    "max_cpu": float(group["mean_cpu"].max()),
                }
            )
    segment_frame = pd.DataFrame(segment_rows)
    if not segment_frame.empty:
        _write_text(tables_dir / "segment_stats.md", markdown_table(segment_frame))

    comparison = result.selection.table.copy()
    family_by_model = {row["model"]: row["family"] for row in model_table()}
    comparison.insert(1, "family", comparison["model"].map(family_by_model))
    comparison = comparison[
        [
            "model",
            "family",
            "n_folds",
            "n_out_of_range_folds",
            "mean_fold_mae",
            "median_fold_mae",
            "max_fold_mae",
            "pooled_mae",
            "pooled_rmse",
            "pooled_peak_mae",
            "eligible",
            "selected",
        ]
    ]
    _write_text(tables_dir / "model_comparison.md", markdown_table(comparison))

    holdout_rows = []
    for metric in ("mae", "rmse", "smape", "mase", "normalized_mae", "peak_mae", "calm_mae", "bias"):
        holdout_rows.append(
            {
                "metric": metric,
                "selected": evaluation.cpu_metrics.get(metric, np.nan),
                "persistence": evaluation.baseline_metrics.get(metric, np.nan),
                "improvement_pct": (
                    100.0
                    * (evaluation.baseline_metrics.get(metric, np.nan) - evaluation.cpu_metrics.get(metric, np.nan))
                    / evaluation.baseline_metrics.get(metric, np.nan)
                    if evaluation.baseline_metrics.get(metric, np.nan) not in (0, np.nan)
                    and np.isfinite(evaluation.baseline_metrics.get(metric, np.nan))
                    else np.nan
                ),
            }
        )
    holdout_frame = pd.DataFrame(holdout_rows)
    _write_text(tables_dir / "holdout_metrics.md", markdown_table(holdout_frame))

    error_frame = evaluation.error_by_horizon[
        ["step", "timestamp", "y_true", "y_pred", "error", "abs_error", "baseline_abs_error"]
    ].copy()
    _write_text(tables_dir / "error_by_horizon.md", markdown_table(error_frame))

    cpu = dataset.cpu
    hotspot_rows = []
    for threshold in (50, 80, 90, 95):
        counts = (
            cpu.loc[cpu["avg_cpu"] >= threshold]
            .groupby("timestamp")
            .size()
            .reindex(np.sort(cpu["timestamp"].unique()), fill_value=0)
        )
        hotspot_rows.append(
            {
                "threshold_pct": threshold,
                "mean_vms": float(counts.mean()),
                "min_vms": int(counts.min()),
                "max_vms": int(counts.max()),
                "mean_share": float(counts.mean() / targets[ACTIVE_VM_COUNT].mean()),
                "count_at_window_start": int(counts.iloc[0]),
                "count_at_window_end": int(counts.iloc[-1]),
            }
        )
    hotspot_frame = pd.DataFrame(hotspot_rows)
    _write_text(tables_dir / "hotspot_summary.md", markdown_table(hotspot_frame))

    return {
        "data_audit": audit_frame,
        "target_stats": target_frame,
        "segment_stats": segment_frame,
        "model_comparison": comparison,
        "holdout_metrics": holdout_frame,
        "error_by_horizon": error_frame,
        "hotspot_summary": hotspot_frame,
    }


def build_figures(result: ForecastResult, evaluation: HoldoutEvaluation, figures_dir: Path) -> None:
    _style()
    dataset = result.dataset
    targets = dataset.targets
    split_timestamp = int(result.train["timestamp"].iloc[-1])
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_target_series(targets, split_timestamp, figures_dir / "01_target_series.png")
    plot_cpu_distribution(dataset, figures_dir / "02_cpu_distribution.png")
    plot_segment_series(dataset, split_timestamp, figures_dir / "03_segment_series.png")
    plot_autocorrelation(
        result.train[AGGREGATE_CPU].to_numpy(dtype=float),
        figures_dir / "04_autocorrelation.png",
    )
    plot_hotspot_counts(dataset, split_timestamp, figures_dir / "05_hotspot_counts.png")
    plot_validation_folds(result, figures_dir / "06_validation_folds.png")
    plot_model_comparison(result.selection.table, figures_dir / "07_model_comparison.png")
    plot_holdout_forecast(evaluation, figures_dir / "08_holdout_forecast.png")
    plot_error_by_horizon(evaluation, figures_dir / "09_error_by_horizon.png")
    plot_active_count(dataset, split_timestamp, figures_dir / "10_active_count.png")


def _improvement(selected: dict[str, float], baseline: dict[str, float], metric: str) -> float:
    base = baseline.get(metric, np.nan)
    value = selected.get(metric, np.nan)
    if base is None or not np.isfinite(base) or base == 0:
        return float("nan")
    return float(100.0 * (base - value) / base)


def write_artifacts(
    result: ForecastResult,
    evaluation: HoldoutEvaluation,
    reports_dir: Path,
) -> dict[str, Any]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables = build_tables(result, evaluation, reports_dir)

    result.dataset.write_audit(reports_dir / "data_audit.json")
    result.cpu_backtest.predictions.to_csv(reports_dir / "backtest_predictions.csv", index=False)
    result.cpu_backtest.metrics.to_csv(reports_dir / "backtest_metrics.csv", index=False)
    result.cpu_summary.to_csv(reports_dir / "model_comparison.csv", index=False)
    evaluation.cpu_forecast.to_csv(reports_dir / "holdout_predictions.csv", index=False)
    evaluation.error_by_horizon.to_csv(reports_dir / "holdout_error_by_horizon.csv", index=False)
    result.dataset.targets.to_csv(reports_dir / "target_series.csv", index=False)

    summary = {
        "dataset": result.dataset.audit.to_dict(),
        "selection": {
            "selected_model": result.selection.selected_model,
            "reason": result.selection.reason,
            "selected_description": result.selection.selected_spec.description,
            "active_model": result.active_selection.selected_model,
            "active_reason": result.active_selection.reason,
        },
        "backtest": {
            "origins": result.cpu_backtest.origins,
            "n_folds": int(result.cpu_summary["n_folds"].max()),
            "selected": result.cpu_summary.loc[
                result.cpu_summary["model"] == result.selection.selected_model
            ].to_dict(orient="records")[0],
            "baseline": result.cpu_summary.loc[
                result.cpu_summary["model"] == "persistence"
            ].to_dict(orient="records")[0],
        },
        "holdout": {
            "selected_metrics": evaluation.cpu_metrics,
            "baseline_metrics": evaluation.baseline_metrics,
            "improvement_pct": {
                metric: _improvement(
                    evaluation.cpu_metrics, evaluation.baseline_metrics, metric
                )
                for metric in evaluation.cpu_metrics
            },
            "interval_coverage": evaluation.interval_coverage,
            "intervals": result.intervals,
        },
        "active_vm_count_holdout": {
            "forecast_unique": sorted(
                int(value) for value in result.predictions[ACTIVE_VM_COUNT].unique()
            ),
            "actual_unique": sorted(
                int(value)
                for value in result.holdout_actuals[ACTIVE_VM_COUNT].unique()
            ),
            "mae": float(
                mae(
                    result.holdout_actuals[ACTIVE_VM_COUNT],
                    result.predictions[ACTIVE_VM_COUNT],
                )
            ),
        },
        "predictions": result.predictions.to_dict(orient="records"),
        "run_manifest": result.meta,
        "tables": {name: frame.to_dict(orient="records") for name, frame in tables.items()},
    }
    _write_json(reports_dir / "results.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate EDA, validation diagnostics, and holdout evaluation."
    )
    parser.add_argument("--data", default="challenge_data", help="Directory with the challenge release.")
    parser.add_argument("--reports-dir", default="reports", help="Directory for generated artefacts.")
    parser.add_argument("--horizon", type=int, default=24, help="Holdout length in observations.")
    parser.add_argument("--min-train", type=int, default=None, help="Minimum training window.")
    parser.add_argument("--step", type=int, default=12, help="Step between rolling origins.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports_dir = Path(args.reports_dir)
    result = run_forecast(
        args.data,
        horizon=args.horizon,
        min_train=args.min_train,
        step=args.step,
        include_holdout=True,
    )
    evaluation = evaluate_holdout(result)
    summary = write_artifacts(result, evaluation, reports_dir)
    if not args.no_figures:
        build_figures(result, evaluation, reports_dir / "figures")

    print(f"Reports written to {reports_dir}")
    print(f"Selected model: {result.selection.selected_model}")
    print(
        "Holdout MAE: "
        f"{evaluation.cpu_metrics['mae']:.4f} pp | "
        f"persistence MAE: {evaluation.baseline_metrics['mae']:.4f} pp | "
        f"improvement: {summary['holdout']['improvement_pct']['mae']:.1f}%"
    )
    print(
        "Peak-period MAE: "
        f"{evaluation.cpu_metrics['peak_mae']:.4f} pp | "
        f"persistence: {evaluation.baseline_metrics['peak_mae']:.4f} pp"
    )
    print(f"Intervals: {result.intervals} | coverage: {evaluation.interval_coverage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
