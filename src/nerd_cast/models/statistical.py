"""Exponential smoothing (ETS) forecaster built on statsmodels.

The seasonally adjusted level is a smooth, trending series, so Holt's linear
trend method (exponential smoothing with a trend but no seasonal term, since
seasonality is already removed) is a strong and explainable fit. The model
exposes its smoothing parameters and the decomposed level and trend so the
``explain`` command can describe *why* it produced a given number.
"""

import warnings
from typing import Self

import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters.results import HoltWintersResultsWrapper

from nerd_cast.models.base import ForecastResult
from nerd_cast.models.baseline import _normal_quantile


class EtsForecaster:
    """Holt linear-trend exponential smoothing on the adjusted level.

    Args:
        damped_trend: Damp the trend so long-run forecasts flatten rather than
            extrapolate a straight line indefinitely. Recommended for a mature
            employment level.
    """

    name = "ets-holt"

    def __init__(self, damped_trend: bool = True) -> None:
        """Configure the trend specification used at fit time."""
        self._damped_trend = damped_trend
        self._level_series: pd.Series | None = None
        self._fitted: HoltWintersResultsWrapper | None = None

    def fit(self, level_series: pd.Series) -> Self:
        """Fit Holt's linear trend model to the level series.

        Args:
            level_series: Date-indexed seasonally adjusted level series.

        Raises:
            ValueError: If fewer than ten observations are supplied, which is
                too few to estimate level and trend reliably.
        """
        if len(level_series) < 10:
            raise ValueError("at least ten observations are required to fit ETS")
        self._level_series = level_series.sort_index().astype(float)
        model = ExponentialSmoothing(
            self._level_series,
            trend="add",
            damped_trend=self._damped_trend,
            seasonal=None,
            initialization_method="heuristic",
        )
        # The smoothing parameters often settle on the boundary (alpha or beta
        # near 1.0) because the adjusted level tracks its own recent history
        # closely. statsmodels flags that boundary solution as non-convergent
        # even though the fit is valid, so this expected warning is silenced.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._fitted = model.fit()
        return self

    def predict(self, confidence: float = 0.95) -> ForecastResult:
        """Forecast the next level and derive the headline change.

        The prediction interval is built from the in-sample residual standard
        deviation scaled by the normal quantile, then expressed on the change
        scale (which equals the level interval because the last level is known).

        Args:
            confidence: Nominal coverage of the prediction interval.

        Returns:
            A :class:`ForecastResult` for the month after the last observation.
        """
        if self._fitted is None or self._level_series is None:
            raise RuntimeError("predict called before fit")

        point_level = float(self._fitted.forecast(1).iloc[0])
        last_date = self._level_series.index[-1]
        last_level = float(self._level_series.iloc[-1])
        target_date = last_date + pd.offsets.MonthBegin(1)
        point_change = point_level - last_level

        residual_dispersion = float(pd.Series(self._fitted.resid).std(ddof=1))
        if pd.isna(residual_dispersion):  # pragma: no cover - guard for <2 residuals
            residual_dispersion = 0.0
        half_width = _normal_quantile(confidence) * residual_dispersion

        return ForecastResult(
            model_name=self.name,
            as_of_date=last_date.date(),
            target_date=target_date.date(),
            last_level=last_level,
            point_level=point_level,
            point_change=point_change,
            lower_change=point_change - half_width,
            upper_change=point_change + half_width,
            confidence=confidence,
        )

    def explain(self) -> dict[str, float | str]:
        """Return smoothing parameters and the latest level and trend states."""
        if self._fitted is None:
            raise RuntimeError("explain called before fit")
        parameters = self._fitted.params
        return {
            "method": "Holt linear trend exponential smoothing",
            "damped_trend": str(self._damped_trend),
            "smoothing_level_alpha": float(parameters["smoothing_level"]),
            "smoothing_trend_beta": float(parameters["smoothing_trend"]),
            "final_level": float(self._fitted.level.iloc[-1]),
            "final_trend": float(self._fitted.trend.iloc[-1]),
        }
