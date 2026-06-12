"""Forecasting models exposed behind a common protocol."""

from nerd_cast.models.base import Forecaster, ForecastResult
from nerd_cast.models.baseline import SeasonalNaiveForecaster
from nerd_cast.models.statistical import EtsForecaster

__all__ = [
    "ForecastResult",
    "Forecaster",
    "SeasonalNaiveForecaster",
    "EtsForecaster",
]
