"""Seasonal-naive baseline forecaster.

This is the benchmark every serious model must beat. It predicts the next
month-over-month change as the average of the most recent observed changes,
which captures the recent drift in the seasonally adjusted level without any
fitted parameters. Because the series is already seasonally adjusted, a short
trailing mean is a fair, hard-to-beat reference point and supplies the
denominator for the MASE skill score in the evaluation module.
"""

from typing import Self

import pandas as pd

from nerd_cast.models.base import ForecastResult


class SeasonalNaiveForecaster:
    """Predict the next change as the mean of recent changes.

    Args:
        window: Number of trailing monthly changes to average. A full year by
            default, which smooths month-to-month noise in the adjusted level.
    """

    name = "seasonal-naive"

    def __init__(self, window: int = 12) -> None:
        """Store the trailing window length used for the mean change."""
        if window < 1:
            raise ValueError("window must be a positive integer")
        self._window = window
        self._level_series: pd.Series | None = None

    def fit(self, level_series: pd.Series) -> Self:
        """Record the level series used to derive recent changes.

        Args:
            level_series: Date-indexed seasonally adjusted level series.

        Raises:
            ValueError: If fewer than two observations are supplied.
        """
        if len(level_series) < 2:
            raise ValueError("at least two observations are required to fit")
        self._level_series = level_series.sort_index().astype(float)
        return self

    def predict(self, confidence: float = 0.95) -> ForecastResult:
        """Forecast the next change as the trailing mean change.

        The interval width is the standard deviation of the trailing changes
        scaled by the matching normal quantile, giving an empirical band around
        the mean drift.

        Args:
            confidence: Nominal coverage of the prediction interval.

        Returns:
            A :class:`ForecastResult` for the month after the last observation.
        """
        if self._level_series is None:
            raise RuntimeError("predict called before fit")

        changes = self._level_series.diff().dropna()
        trailing_changes = changes.tail(self._window)
        mean_change = float(trailing_changes.mean())
        change_dispersion = float(trailing_changes.std(ddof=1))
        if pd.isna(change_dispersion):
            change_dispersion = 0.0

        last_date = self._level_series.index[-1]
        last_level = float(self._level_series.iloc[-1])
        target_date = last_date + pd.offsets.MonthBegin(1)
        quantile = _normal_quantile(confidence)
        half_width = quantile * change_dispersion

        return ForecastResult(
            model_name=self.name,
            as_of_date=last_date.date(),
            target_date=target_date.date(),
            last_level=last_level,
            point_level=last_level + mean_change,
            point_change=mean_change,
            lower_change=mean_change - half_width,
            upper_change=mean_change + half_width,
            confidence=confidence,
        )

    def explain(self) -> dict[str, float | str]:
        """Return the trailing window and the mean change it produced."""
        if self._level_series is None:
            raise RuntimeError("explain called before fit")
        trailing_changes = self._level_series.diff().dropna().tail(self._window)
        return {
            "method": "mean of trailing month-over-month changes",
            "window_months": float(self._window),
            "mean_change": float(trailing_changes.mean()),
        }


def _normal_quantile(confidence: float) -> float:
    """Return the two-sided normal quantile for a coverage level.

    Args:
        confidence: Nominal coverage such as ``0.95``.

    Returns:
        The standard-normal quantile that places ``confidence`` mass between the
        symmetric bounds (for example ``1.96`` for ``0.95``).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    from scipy.stats import norm

    tail_probability = 1.0 - (1.0 - confidence) / 2.0
    return float(norm.ppf(tail_probability))
