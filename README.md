# nerd-cast

A small command-line tool that tracks the monthly **ADP National Employment
Report (NER)** and forecasts the next print. It lets you:

- see the historical numbers (`history`),
- see the prediction for next month (`forecast`),
- understand *why* that prediction was made (`explain`).

The headline ADP print is the **month-over-month change in the seasonally
adjusted employment level**. The bundled data provides employment *levels*, so
the tool models the level and reports both the level and the derived change.

---

## How to run

The project targets **Python 3.14** and uses [`uv`](https://docs.astral.sh/uv/)
for environment and dependency management.

```bash
# 1. Clone, then from the repository root create the environment.
uv venv

# 2. Install the package (and dev tools) in editable mode.
uv pip install -e ".[dev]"

# 3. Run the CLI.
uv run nerd-cast --help
```

If you prefer plain `pip`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
nerd-cast --help
```

### Commands

```bash
# Last 12 months of the headline change (seasonally adjusted).
uv run nerd-cast history

# Last 6 months as raw levels instead of changes.
uv run nerd-cast history --months 6 --metric level

# Next month's forecast with a 95% interval (ETS model by default).
uv run nerd-cast forecast

# The same forecast from the naive baseline.
uv run nerd-cast forecast --model seasonal-naive

# Why the model predicted what it did, plus its backtested accuracy.
uv run nerd-cast explain
```

Every command accepts `--data PATH` to point at a different history CSV, and
`--agg-ris` / `--category` to forecast a slice other than National (for example
`--agg-ris Industry --category Manufacturing`). The default slice is
`National / U.S.`.

Example `forecast` output:

```
Model:        ets-holt
As of:        2026-05-01
Next print:   2026-06-01
Forecast change:  +107,751 jobs
95% interval:    [-182,629, +398,132] jobs
Forecast level:   132,731,751
```

---

## Approach and key tradeoffs

**Forecast the level, report the change.** The seasonally adjusted level
(`NER_SA`) is a smooth, trending series, so it is the easier and more stable
thing to model. The headline change is derived by subtracting the last observed
level. Accuracy, however, is always measured on the *change* — the number users
actually care about — because a fraction-of-a-percent error on a ~132 million
level would otherwise dwarf the ~100 thousand monthly change.

**A classical model over machine learning.** With only ~197 monthly
observations, a classical model is more robust and far more explainable than a
gradient-boosted or deep model. The primary model is **Holt's linear trend
exponential smoothing (ETS)** from `statsmodels`, with a damped trend so the
forecast does not extrapolate a straight line indefinitely. Its level and trend
components, and its smoothing parameters, are surfaced directly in `explain`.

**A baseline that must be beaten.** A **seasonal-naive** forecaster (next change
= mean of the trailing twelve changes) is shipped alongside the ETS model. It is
the reference the real model is judged against, and it doubles as a sanity check.

**One interchangeable model interface.** Every model implements a single
`Forecaster` protocol (`fit` / `predict` / `explain`). The CLI, the backtester,
and the explanation layer never depend on a concrete model, so adding ARIMA or a
machine-learning model later is a drop-in change.

**A single static CSV, not a database.** The dataset is ~26k rows and never
changes at runtime, so it is loaded into a tidy pandas frame with a thin
accessor that returns a date-indexed series for any slice. SQLite would have
added a dependency and complexity for no benefit at this scale.

**Plain-text output, no presentation dependency.** Output is aligned plain text,
which keeps the dependency surface small and makes the tool easy to pipe.

Key tradeoffs accepted for v1: the tool forecasts **National only** (the slice
accessor is already general enough to extend to industries, divisions, and
establishment sizes); the forecast horizon is **one month**; and the prediction
interval is a normal-approximation band from in-sample residual dispersion rather
than a fully simulated predictive distribution.

---

## How forecast accuracy was evaluated

Accuracy is measured with a **walk-forward (rolling-origin) backtest**, the
standard approach for time series. For each month past a minimum training window,
a fresh model is fit on all data *up to* that month and asked to predict the next
month's change; the prediction is compared against the realised change. This
mirrors real use and never lets future data leak into a fit. (Plain k-fold
cross-validation is intentionally avoided — shuffling time-series rows leaks the
future into the past.)

Three metrics are reported, all on the **monthly change**:

- **MAE** — mean absolute error, in jobs (the headline accuracy number).
- **RMSE** — root mean squared error, in jobs (penalises large misses).
- **MASE** — MAE scaled by the error of a one-step random walk on the change;
  below `1.0` means the model beats that naive reference.

### Results

Backtest over the full National history (Jan 2010 – May 2026, 197 monthly
points), default minimum training window of 36 months → **161 folds**:

| Model            | MAE (jobs) | RMSE (jobs) | MASE  |
|------------------|-----------:|------------:|:-----:|
| `ets-holt`       |     88,734 |     166,551 | 1.015 |
| `seasonal-naive` |    151,890 |     290,267 | 1.738 |

**The ETS model lowers mean absolute error by ~42%** versus the seasonal-naive
baseline. The result is stable when the training window is widened to 60 months
(137 folds): ETS MAE 95,986 vs baseline 171,240, a ~44% improvement.

A MASE near `1.0` for ETS is an honest finding: on this already-seasonally-
adjusted, near-random-walk series, the month-to-month change is genuinely hard to
beat against a pure random walk — but ETS clearly and consistently beats the
trailing-mean baseline this tool ships. Reproduce these numbers with:

```bash
uv run nerd-cast explain                 # default 36-month window
uv run nerd-cast explain --min-train 60  # wider window
```

---

## Testing and coverage

The suite uses `pytest` and `coverage`, and the project is linted and formatted
with `ruff`.

```bash
# Run the full test suite.
uv run pytest

# Run the suite under coverage and print a per-file report.
uv run coverage run -m pytest
uv run coverage report

# Optional: write a browsable HTML report to htmlcov/index.html.
uv run coverage html

# Lint and format checks.
uv run ruff check .
uv run ruff format --check .
```

The suite covers the data layer, both models, the backtester and its metric
math (validated against hand-computed values), the explanation assembly, and the
CLI (via `typer`'s `CliRunner`). **Line and branch coverage is 100%**; the only
excluded lines are the `__main__` guard and one defensive numeric guard that is
unreachable given the models' minimum-observation requirements.

---

## Project layout

```
pinwheel-employment-cli/
├── pyproject.toml            # metadata, dependencies, console script, tool config
├── README.md
├── PROMPTS.md                # full log of AI sessions used to build this
├── data/
│   └── ADP_NER_history.csv   # bundled ADP National Employment Report history
├── src/
│   └── nerd_cast/
│       ├── cli.py            # typer app: history / forecast / explain
│       ├── data.py           # load CSV, slice into date-indexed series
│       ├── evaluation.py     # walk-forward backtest + MAE / RMSE / MASE
│       ├── explain.py        # assemble the "why" behind a forecast
│       └── models/
│           ├── base.py       # Forecaster protocol + ForecastResult
│           ├── baseline.py   # seasonal-naive baseline
│           └── statistical.py# ETS (Holt linear trend)
└── tests/                    # pytest suite, 100% coverage
```

The package follows the standard `src/` layout from the
[Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
so it mirrors a production CLI. It is intentionally **not** published to PyPI;
`uv pip install -e .` is all that is needed to run it locally.

---

## What I would build next with another week

- **Component / hierarchical forecasting.** Forecast each industry, census
  division, and establishment-size series and reconcile them against the
  National total, then compare top-down vs bottom-up accuracy. The data and the
  slice accessor already support this.
- **A second model and automatic order selection.** Add SARIMAX / auto-ARIMA
  behind the existing `Forecaster` protocol and let `explain` pick the model
  with the best backtested MASE per slice.
- **Better prediction intervals.** Replace the normal-approximation band with a
  simulated or bootstrapped predictive distribution and report interval
  *coverage* (how often the actual lands inside the stated interval) as a fourth
  accuracy metric.
- **Live data ingestion.** A small fetch layer to pull new ADP releases on a
  schedule, with caching and a `--refresh` flag, so the tool tracks the report
  rather than a static snapshot.
- **Richer history view.** Optional plotting / sparkline output and the ability
  to diff a past forecast against what was actually printed.
