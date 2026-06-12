"""Tests for the explanation assembly."""

import pandas as pd

from nerd_cast.explain import Explanation, build_explanation
from nerd_cast.models.baseline import SeasonalNaiveForecaster
from nerd_cast.models.statistical import EtsForecaster


def test_build_explanation_populates_all_strands(
    linear_level_series: pd.Series,
) -> None:
    """The explanation carries a forecast, drivers, and both metric sets."""
    explanation = build_explanation(linear_level_series, EtsForecaster(), min_train=24)
    assert isinstance(explanation, Explanation)
    assert explanation.forecast.model_name == "ets-holt"
    assert "method" in explanation.drivers
    assert explanation.model_metrics.model_name == "ets-holt"
    assert explanation.baseline_metrics.model_name == "seasonal-naive"


def test_build_explanation_fits_forecaster_in_place(
    linear_level_series: pd.Series,
) -> None:
    """The supplied forecaster is fit and can be explained afterwards."""
    forecaster = SeasonalNaiveForecaster()
    explanation = build_explanation(linear_level_series, forecaster, min_train=24)
    assert explanation.forecast.model_name == "seasonal-naive"
    # The same instance is now fit, so explain works without raising.
    assert forecaster.explain()["mean_change"] is not None


def test_model_beats_baseline_on_real_data(
    real_history_frame: pd.DataFrame,
) -> None:
    """On the real series ETS achieves a lower MAE than the baseline."""
    from nerd_cast.data import get_series

    level_series = get_series(real_history_frame)
    explanation = build_explanation(level_series, EtsForecaster())
    assert (
        explanation.model_metrics.mean_absolute_error
        < explanation.baseline_metrics.mean_absolute_error
    )
