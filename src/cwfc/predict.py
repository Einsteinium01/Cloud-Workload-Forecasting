"""Generate the required ``predictions.csv``.

Example
-------
    python -m cwfc.predict --data challenge_data --output predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .pipeline import run_forecast, write_manifest, write_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forecast the final two-hour holdout of the Azure workload challenge."
    )
    parser.add_argument(
        "--data",
        default="challenge_data",
        help="Directory containing vm_cpu_readings.csv.gz and vm_metadata.csv.gz.",
    )
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Destination for the required prediction file.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Number of final five-minute timestamps to forecast (default: 24).",
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=None,
        help="Minimum training window for rolling-origin validation.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=12,
        help="Step between rolling origins (default: 12 timestamps).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional path for a JSON run manifest.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_forecast(
            args.data,
            horizon=args.horizon,
            min_train=args.min_train,
            step=args.step,
            include_holdout=False,
        )
        expected_timestamps = result.dataset.targets["timestamp"].tail(args.horizon)
        output = write_predictions(
            result,
            args.output,
            expected_timestamps=expected_timestamps,
            expected_rows=args.horizon,
        )
        if args.manifest:
            write_manifest(result, args.manifest)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Wrote {len(result.predictions)} predictions to {output}")
        print(
            "Selected aggregate_cpu_utilization model: "
            f"{result.selection.selected_model}"
        )
        print(f"Validation folds: {result.cpu_backtest.origins}")
        print(
            "Validation pooled MAE: "
            f"{result.cpu_summary.loc[result.cpu_summary['model'] == result.selection.selected_model, 'pooled_mae'].iloc[0]:.4f}"
        )
        print(
            "Holdout timestamps: "
            f"{result.predictions['timestamp'].iloc[0]} .. "
            f"{result.predictions['timestamp'].iloc[-1]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
