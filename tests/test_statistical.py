"""Tests for the ETS (Holt linear trend) forecaster."""

import pandas as pd
import pytest

from adp_ner.models.base import Forecaster, ForecastResult
from adp_ner.models.statistical import EtsForecaster


def test_ets_satisfies_protocol() -> None:
    """The ETS model is recognised as a Forecaster."""
    assert isinstance(EtsForecaster(), Forecaster)


def test_predict_tracks_linear_trend(linear_level_series: pd.Series) -> None:
    """On a clean linear trend the forecast change is close to the slope."""
    result = EtsForecaster(damped_trend=False).fit(linear_level_series).predict()
    assert isinstance(result, ForecastResult)
    assert result.point_change == pytest.approx(1_000.0, rel=0.05)


def test_damped_trend_is_below_undamped(linear_level_series: pd.Series) -> None:
    """Damping pulls the forecast change below the raw slope."""
    damped = EtsForecaster(damped_trend=True).fit(linear_level_series).predict()
    assert damped.point_change < 1_000.0


def test_target_is_next_month(linear_level_series: pd.Series) -> None:
    """The target date is the month after the final observation."""
    result = EtsForecaster().fit(linear_level_series).predict()
    assert result.target_date.year == 2014
    assert result.target_date.month == 1


def test_interval_brackets_point(linear_level_series: pd.Series) -> None:
    """The prediction interval is ordered around the point forecast."""
    result = EtsForecaster().fit(linear_level_series).predict()
    assert result.lower_change <= result.point_change <= result.upper_change


def test_level_equals_last_plus_change(linear_level_series: pd.Series) -> None:
    """The forecast level is the last level plus the forecast change."""
    result = EtsForecaster().fit(linear_level_series).predict()
    assert result.point_level == pytest.approx(result.last_level + result.point_change)


def test_fit_requires_enough_observations() -> None:
    """Too few observations to estimate level and trend is an error."""
    series = pd.Series(
        range(5), index=pd.date_range("2020-01-01", periods=5, freq="MS")
    ).astype(float)
    with pytest.raises(ValueError, match="ten observations"):
        EtsForecaster().fit(series)


def test_predict_before_fit_raises() -> None:
    """Predicting before fitting is an error."""
    with pytest.raises(RuntimeError, match="before fit"):
        EtsForecaster().predict()


def test_explain_before_fit_raises() -> None:
    """Explaining before fitting is an error."""
    with pytest.raises(RuntimeError, match="before fit"):
        EtsForecaster().explain()


def test_explain_reports_smoothing_parameters(
    linear_level_series: pd.Series,
) -> None:
    """The explanation exposes smoothing parameters and final states."""
    drivers = EtsForecaster().fit(linear_level_series).explain()
    assert "smoothing_level_alpha" in drivers
    assert "smoothing_trend_beta" in drivers
    assert "final_trend" in drivers
    assert drivers["method"].startswith("Holt")
