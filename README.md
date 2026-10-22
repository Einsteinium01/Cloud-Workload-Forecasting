# Cloud Workload Forecasting — Solution

Forecasting `active_vm_count` and `aggregate_cpu_utilization` for the final 24 five-minute timestamps of the fixed `data-v1` Azure VM release. The repository contains a reproducible pipeline, time-aware validation, the required [`predictions.csv`](predictions.csv), a 4–6 page [final report](reports/final_report.md), and tests.

## Quick start: clean checkout to `predictions.csv`

After installing the package and extracting the release (see below), the single command that produces the submission is:

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv --manifest reports/run_manifest.json
```

It writes exactly 24 rows with columns `timestamp,active_vm_count,aggregate_cpu_utilization`, using only the first 110 observations for fitting and model selection.

## 1. Environment

Python 3.11 or newer is required.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The `dev` extra installs `pytest` and `matplotlib`; the forecasting runtime itself only needs `numpy`, `pandas`, and `scikit-learn`.

Verify the installation:

```bash
python -m pytest -q
```

Expected: `21 passed`.

## 2. Dataset

The challenge uses the fixed **`data-v1`** release of this repository, a curated subset of the Microsoft Azure Public Dataset V2. The release is not committed to the repository; download it from:

<https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1>

Extract `data-v1.tar.gz` so the repository contains:

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

For example:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
```

Verify the checksums before running:

```bash
cd challenge_data
sha256sum -c SHA256SUMS
```

The release contains 10,000 VMs × 134 five-minute observations (1,340,000 rows) plus metadata for all 10,000 VMs. `challenge_data/` is gitignored; do not commit the release files.

## 3. Reproduce the analysis

Regenerate all tables, figures, and machine-readable results:

```bash
python -m cwfc.analyze --data challenge_data --reports-dir reports
```

This runs the same frozen pipeline, scores the holdout for reporting, and writes:

- `reports/results.json` — metrics, selection details, intervals, and the prediction rows;
- `reports/tables/*.md` — data audit, target/segment statistics, model comparison, holdout metrics, per-step errors;
- `reports/figures/*.png` — EDA, validation, and holdout figures;
- `reports/backtest_predictions.csv`, `reports/backtest_metrics.csv`;
- `reports/holdout_predictions.csv`, `reports/holdout_error_by_horizon.csv`.

The analysis command is separate from the prediction command: `cwfc.predict` uses `include_holdout=False`, while `cwfc.analyze` performs the holdout evaluation only after the forecasts are frozen.

## 4. Method summary

### Targets and split

For timestamp `t`:

- `active_vm_count(t)` = number of distinct VM IDs with a valid CPU reading at `t` (each VM counted at most once);
- `aggregate_cpu_utilization(t)` = unweighted mean of VM-level `avg_cpu` at `t`, in percent.

The full 134-point series is constructed first, then split chronologically:

- training: first 110 observations, timestamps `0 … 32,700`;
- holdout: final 24 observations, timestamps `33,000 … 39,900`.

No resampling, interpolation, or forward filling is performed. Missing readings would be excluded from the timestamp and recorded in the audit; none occur in this release.

### Validation

The main comparison uses an expanding-window rolling-origin backtest:

- 4 folds, with training lengths `50, 62, 74, 86`;
- forecast horizon 24 (two hours);
- step between origins 12;
- 96 out-of-sample predictions per model;
- no random shuffling and no future values in any training window.

Metrics: MAE, RMSE, sMAPE, MASE, normalized MAE vs persistence, peak-period MAE (top quartile), calm-period MAE, and bias. The final model is refitted on all 110 training observations and forecasts the 24 holdout timestamps.

### Candidate models

18 univariate candidates:

- baselines: persistence, moving averages (6/12/24), seasonal naive (1h/2h/4h), 4-hour seasonal random walk blended with persistence;
- statistical: simple exponential smoothing, AR(2), AR(6) on levels with an intercept;
- ML: ridge regression and a shallow histogram gradient-boosting model on lags and rolling features;
- ensembles: equal-weight combinations of persistence, the 1-hour mean, AR(6), the 4-hour seasonal naive, and ridge.

Features for the ML models use only observations at or before the forecast origin: lags 1, 2, 3, 6, 12; rolling means and standard deviations over 3, 6, 12 steps. No calendar features are used because timestamps are relative to an anonymized trace start and the 11.1-hour window is shorter than a day.

### Selection rule

The rule is fixed before looking at the holdout:

1. results on all four folds;
2. no forecast outside the observed training range;
3. pooled peak-period MAE at most 10% worse than persistence;
4. worst-fold MAE at most 10% worse than persistence's worst fold;
5. lowest mean fold MAE; persistence wins ties.

The selected model is **`ensemble_p_ar6`**, an equal-weight average of persistence and AR(6). The active-count model is persistence (selected by the tie-break).

## 5. Results

| Metric | Selected | Persistence |
|---|---:|---:|
| CV pooled MAE | 0.1977 | 0.2056 |
| CV pooled RMSE | 0.2358 | 0.2540 |
| Holdout MAE (pp) | 0.1026 | 0.1468 |
| Holdout RMSE (pp) | 0.1378 | 0.1848 |
| Holdout sMAPE (%) | 1.1048 | 1.5851 |
| Holdout MASE | 0.8469 | 1.2119 |
| Holdout peak MAE (pp) | 0.2155 | 0.3085 |
| Holdout normalized MAE | 0.6988 | 1.0000 |

- The holdout MAE is 30.1% lower than persistence; the CV MAE is 3.8% lower.
- Cross-validation is the more conservative estimate: the ensemble lost to persistence in three of four folds and won one large level-shift fold.
- The active-count forecast is exactly 10,000 for all 24 timestamps; the target is structurally constant in this complete-trace cohort.
- The 80% and 90% empirical residual bands are `±0.2968` and `±0.3880` pp; both cover 23 of 24 holdout points.

See [`reports/final_report.md`](reports/final_report.md) for the full analysis, error discussion, and operational recommendation.

## 6. Repository layout

```text
predictions.csv                 required 24-row submission
reports/final_report.md         4–6 page report
reports/results.json            machine-readable results
reports/tables/                 generated Markdown tables
reports/figures/                generated figures
src/cwfc/data.py                loading, audit, targets, chronological split
src/cwfc/features.py            leakage-safe lag/rolling features
src/cwfc/models.py              candidate models and ensembles
src/cwfc/validation.py          rolling-origin backtest and model selection
src/cwfc/pipeline.py            end-to-end forecast, intervals, manifest, evaluation
src/cwfc/predict.py             CLI that writes predictions.csv
src/cwfc/analyze.py             report tables, figures, holdout evaluation
tests/                          unit and pipeline tests
```

## 7. Leakage protocol

The holdout target values are never used for fitting, feature selection, hyperparameter tuning, or model selection:

- `run_forecast` constructs the full target series, then passes only the 110-row training frame to the backtest and final fit;
- every validation origin uses a strictly past history; validation windows are never passed to a model;
- `cwfc.predict` calls `run_forecast(..., include_holdout=False)` and writes predictions before any holdout scoring;
- `cwfc.analyze` runs the holdout evaluation separately, after the predictions are frozen;
- intervals are computed from cross-validation residuals, not holdout residuals.

The only use of the holdout timestamps during prediction is to label the 24 output rows; their target values are not read by the forecasting path.

## 8. Assumptions and limitations

- The release is a complete-trace cohort: all VMs exist for the full window, so `active_vm_count` is constant. This is not evidence that production churn is absent.
- `aggregate_cpu_utilization` is an unweighted mean of per-VM percentages; it is not a core-weighted or host-weighted capacity measure.
- The training window is short (110 observations, 9.2 hours). Daily/weekly seasonality cannot be estimated, and the top candidate models are close enough that selection is uncertain.
- The holdout contains one sharp spike that no univariate model anticipates; empirical intervals cover 23 of 24 points but are not formal guarantees.
- The operational thresholds in the report are illustrative and should be calibrated to SLOs and capacity costs.

## 9. Data source

Microsoft Azure Public Dataset V2: <https://github.com/Azure/AzurePublicDataset>

Challenge release: `data-v1`, <https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1>