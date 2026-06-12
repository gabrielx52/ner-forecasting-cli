"""Assembly of the human-readable explanation for a forecast.

The ``explain`` command answers "why did you predict this?". It combines three
strands of evidence: the model's own decomposition (level and trend), the
forecast itself with its uncertainty band, and the backtested accuracy of the
model versus the naive baseline so the user can judge how much to trust it.
"""

from dataclasses import dataclass

import pandas as pd

from nerd_cast.evaluation import BacktestMetrics, evaluate
from nerd_cast.models.base import Forecaster, ForecastResult
from nerd_cast.models.baseline import SeasonalNaiveForecaster


@dataclass(frozen=True)
class Explanation:
    """Everything needed to justify a single forecast.

    Attributes:
        forecast: The model's next-month forecast.
        drivers: Model-specific drivers from ``Forecaster.explain``.
        model_metrics: Backtest accuracy of the forecasting model.
        baseline_metrics: Backtest accuracy of the seasonal-naive baseline.
    """

    forecast: ForecastResult
    drivers: dict[str, float | str]
    model_metrics: BacktestMetrics
    baseline_metrics: BacktestMetrics


def build_explanation(
    level_series: pd.Series,
    forecaster: Forecaster,
    *,
    min_train: int = 36,
    confidence: float = 0.95,
) -> Explanation:
    """Fit the model, forecast, and backtest both it and the baseline.

    Args:
        level_series: Date-indexed seasonally adjusted level series.
        forecaster: The model to fit and explain. It is fit in place on the full
            series to produce the live forecast.
        min_train: Minimum training window for the backtests.
        confidence: Coverage of the forecast interval.

    Returns:
        A fully populated :class:`Explanation`.
    """
    forecaster.fit(level_series)
    forecast = forecaster.predict(confidence=confidence)
    drivers = forecaster.explain()

    model_metrics = evaluate(
        level_series,
        type(forecaster),
        forecaster.name,
        min_train=min_train,
        confidence=confidence,
    )
    baseline_metrics = evaluate(
        level_series,
        SeasonalNaiveForecaster,
        SeasonalNaiveForecaster.name,
        min_train=min_train,
        confidence=confidence,
    )
    return Explanation(
        forecast=forecast,
        drivers=drivers,
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
    )
