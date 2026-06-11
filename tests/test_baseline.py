"""Tests for the seasonal-naive baseline forecaster."""

import pandas as pd
import pytest

from adp_ner.models.base import Forecaster, ForecastResult
from adp_ner.models.baseline import SeasonalNaiveForecaster, _normal_quantile


def test_baseline_satisfies_protocol() -> None:
    """The baseline is recognised as a Forecaster."""
    assert isinstance(SeasonalNaiveForecaster(), Forecaster)


def test_predict_returns_mean_change(linear_level_series: pd.Series) -> None:
    """On a constant-change series the forecast equals that change."""
    forecaster = SeasonalNaiveForecaster().fit(linear_level_series)
    result = forecaster.predict()
    assert isinstance(result, ForecastResult)
    assert result.point_change == pytest.approx(1_000.0)
    assert result.point_level == pytest.approx(linear_level_series.iloc[-1] + 1_000.0)


def test_predict_target_is_next_month(linear_level_series: pd.Series) -> None:
    """The target date is the month after the final observation."""
    result = SeasonalNaiveForecaster().fit(linear_level_series).predict()
    assert result.target_date.year == 2014
    assert result.target_date.month == 1
    assert result.as_of_date.month == 12


def test_interval_brackets_point(linear_level_series: pd.Series) -> None:
    """The prediction interval is ordered around the point forecast."""
    result = SeasonalNaiveForecaster().fit(linear_level_series).predict()
    assert result.lower_change <= result.point_change <= result.upper_change


def test_zero_dispersion_collapses_interval(
    linear_level_series: pd.Series,
) -> None:
    """A constant change yields a zero-width interval (no dispersion)."""
    result = SeasonalNaiveForecaster().fit(linear_level_series).predict()
    assert result.lower_change == pytest.approx(result.upper_change)


def test_single_change_has_zero_dispersion() -> None:
    """A two-point series yields one change, so its dispersion is treated as 0."""
    series = pd.Series(
        [100.0, 150.0],
        index=pd.date_range("2020-01-01", periods=2, freq="MS"),
    )
    result = SeasonalNaiveForecaster().fit(series).predict()
    assert result.point_change == pytest.approx(50.0)
    assert result.lower_change == pytest.approx(result.upper_change)


def test_window_controls_lookback() -> None:
    """A short window only averages the most recent changes."""
    dates = pd.date_range("2020-01-01", periods=6, freq="MS")
    values = [0.0, 10.0, 20.0, 30.0, 130.0, 230.0]
    series = pd.Series(values, index=dates)
    result = SeasonalNaiveForecaster(window=2).fit(series).predict()
    assert result.point_change == pytest.approx(100.0)


def test_invalid_window_raises() -> None:
    """A non-positive window is rejected at construction."""
    with pytest.raises(ValueError, match="positive"):
        SeasonalNaiveForecaster(window=0)


def test_fit_requires_two_observations() -> None:
    """A single observation cannot define a change."""
    series = pd.Series([1.0], index=pd.date_range("2020-01-01", periods=1, freq="MS"))
    with pytest.raises(ValueError, match="two observations"):
        SeasonalNaiveForecaster().fit(series)


def test_predict_before_fit_raises() -> None:
    """Predicting before fitting is an error."""
    with pytest.raises(RuntimeError, match="before fit"):
        SeasonalNaiveForecaster().predict()


def test_explain_before_fit_raises() -> None:
    """Explaining before fitting is an error."""
    with pytest.raises(RuntimeError, match="before fit"):
        SeasonalNaiveForecaster().explain()


def test_explain_reports_window_and_mean(linear_level_series: pd.Series) -> None:
    """The explanation surfaces the window and the mean change."""
    drivers = SeasonalNaiveForecaster(window=6).fit(linear_level_series).explain()
    assert drivers["window_months"] == pytest.approx(6.0)
    assert drivers["mean_change"] == pytest.approx(1_000.0)


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(0.95, 1.9600), (0.90, 1.6449), (0.80, 1.2816)],
)
def test_normal_quantile_matches_known_values(
    confidence: float, expected: float
) -> None:
    """The normal quantile matches standard textbook values."""
    assert _normal_quantile(confidence) == pytest.approx(expected, abs=1e-3)


def test_normal_quantile_rejects_out_of_range() -> None:
    """A confidence outside the open unit interval is rejected."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        _normal_quantile(1.5)
