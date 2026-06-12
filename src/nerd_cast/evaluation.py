"""Walk-forward backtesting and accuracy metrics.

Forecast quality is measured with a rolling-origin (walk-forward) backtest: for
each historical month past a minimum training window, a fresh model is fit on
all data up to that month and asked to predict the next month's change. The
prediction is compared against the realised change. This mirrors how the tool is
used in production and never lets future data leak into a fit.

All metrics are computed on the *change* scale - the headline print - rather
than the level, because a tiny percentage error on a ~132 million level would
otherwise dwarf the ~100 thousand change users care about.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from nerd_cast.models.base import Forecaster

ForecasterFactory = Callable[[], Forecaster]


@dataclass(frozen=True)
class BacktestMetrics:
    """Accuracy summary for one model over a walk-forward backtest.

    Attributes:
        model_name: Name of the evaluated model.
        fold_count: Number of one-step forecasts scored.
        mean_absolute_error: Average absolute change error, in jobs.
        root_mean_squared_error: RMSE of the change error, in jobs.
        mean_absolute_scaled_error: MAE divided by the seasonal-naive MAE on the
            same folds; below ``1.0`` means the model beats the naive baseline.
    """

    model_name: str
    fold_count: int
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_absolute_scaled_error: float


def walk_forward(
    level_series: pd.Series,
    build_forecaster: ForecasterFactory,
    *,
    min_train: int = 36,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Run a one-step rolling-origin backtest over the level series.

    Args:
        level_series: Date-indexed seasonally adjusted level series.
        build_forecaster: Zero-argument factory returning a fresh forecaster for
            each fold, so no state leaks between fits.
        min_train: Minimum number of observations in the first training window.
        confidence: Coverage passed through to each forecast.

    Returns:
        A DataFrame indexed by target date with ``predicted_change``,
        ``actual_change`` and ``absolute_error`` columns, one row per fold.

    Raises:
        ValueError: If the series is too short to form a single fold.
    """
    ordered = level_series.sort_index().astype(float)
    if len(ordered) <= min_train:
        raise ValueError(
            f"need more than min_train={min_train} observations to backtest"
        )

    target_dates: list[pd.Timestamp] = []
    predicted_changes: list[float] = []
    actual_changes: list[float] = []

    for split in range(min_train, len(ordered)):
        train = ordered.iloc[:split]
        actual_level = float(ordered.iloc[split])
        last_train_level = float(train.iloc[-1])

        forecaster = build_forecaster()
        forecaster.fit(train)
        forecast = forecaster.predict(confidence=confidence)

        target_dates.append(ordered.index[split])
        predicted_changes.append(forecast.point_change)
        actual_changes.append(actual_level - last_train_level)

    results = pd.DataFrame(
        {
            "predicted_change": predicted_changes,
            "actual_change": actual_changes,
        },
        index=pd.Index(target_dates, name="target_date"),
    )
    results["absolute_error"] = (
        results["predicted_change"] - results["actual_change"]
    ).abs()
    return results


def _seasonal_naive_absolute_errors(actual_changes: pd.Series) -> pd.Series:
    """Absolute errors of a one-step seasonal-naive forecast of the change.

    The naive forecast for each change is simply the previous change, which is
    the standard MASE scaling reference for a one-step horizon.
    """
    return (actual_changes - actual_changes.shift(1)).abs().dropna()


def score(results: pd.DataFrame, model_name: str) -> BacktestMetrics:
    """Reduce backtest folds to summary accuracy metrics.

    Args:
        results: Output of :func:`walk_forward`.
        model_name: Name to attach to the resulting metrics.

    Returns:
        A :class:`BacktestMetrics` describing the model's accuracy.

    Raises:
        ValueError: If ``results`` contains no folds.
    """
    if results.empty:
        raise ValueError("cannot score an empty backtest result")

    absolute_error = results["absolute_error"]
    mean_absolute_error = float(absolute_error.mean())
    root_mean_squared_error = float((absolute_error**2).mean() ** 0.5)

    naive_errors = _seasonal_naive_absolute_errors(results["actual_change"])
    naive_mean_absolute_error = float(naive_errors.mean())
    if naive_mean_absolute_error == 0.0:
        mean_absolute_scaled_error = float("inf")
    else:
        mean_absolute_scaled_error = mean_absolute_error / naive_mean_absolute_error

    return BacktestMetrics(
        model_name=model_name,
        fold_count=int(len(results)),
        mean_absolute_error=mean_absolute_error,
        root_mean_squared_error=root_mean_squared_error,
        mean_absolute_scaled_error=mean_absolute_scaled_error,
    )


def evaluate(
    level_series: pd.Series,
    build_forecaster: ForecasterFactory,
    model_name: str,
    *,
    min_train: int = 36,
    confidence: float = 0.95,
) -> BacktestMetrics:
    """Backtest a forecaster and return its summary metrics in one call."""
    results = walk_forward(
        level_series,
        build_forecaster,
        min_train=min_train,
        confidence=confidence,
    )
    return score(results, model_name)
