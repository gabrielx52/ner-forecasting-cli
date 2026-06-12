"""Shared contract for forecasters.

Every model in this package fits on a seasonally adjusted *level* series and
predicts the next level. The headline change is derived by subtracting the last
observed level, so the result object carries both representations. Keeping a
single :class:`ForecastResult` shape lets the CLI and the backtester treat the
baseline and the statistical model interchangeably.
"""

from dataclasses import dataclass
from datetime import date
from typing import Protocol, Self, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class ForecastResult:
    """A single-step forecast expressed as both a level and a change.

    Attributes:
        model_name: Human-readable name of the producing model.
        as_of_date: Date of the last observed level used for fitting.
        target_date: Date the forecast applies to (the next print).
        last_level: The last observed seasonally adjusted level.
        point_level: Forecast level for ``target_date``.
        point_change: Forecast month-over-month change (``point_level`` minus
            ``last_level``); this is the headline number.
        lower_change: Lower bound of the change prediction interval.
        upper_change: Upper bound of the change prediction interval.
        confidence: Nominal coverage of the interval, for example ``0.95``.
    """

    model_name: str
    as_of_date: date
    target_date: date
    last_level: float
    point_level: float
    point_change: float
    lower_change: float
    upper_change: float
    confidence: float


@runtime_checkable
class Forecaster(Protocol):
    """Protocol implemented by every forecasting model."""

    name: str

    def fit(self, level_series: pd.Series) -> Self:
        """Fit the model on a date-indexed seasonally adjusted level series."""
        ...

    def predict(self, confidence: float = 0.95) -> ForecastResult:
        """Produce the next-month forecast with a prediction interval."""
        ...

    def explain(self) -> dict[str, float | str]:
        """Return model-specific drivers behind the latest forecast."""
        ...
