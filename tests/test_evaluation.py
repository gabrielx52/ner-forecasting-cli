"""Tests for the walk-forward backtest and accuracy metrics."""

import pandas as pd
import pytest

from nerd_cast.evaluation import (
    BacktestMetrics,
    evaluate,
    score,
    walk_forward,
)
from nerd_cast.models.baseline import SeasonalNaiveForecaster
from nerd_cast.models.statistical import EtsForecaster


def test_walk_forward_fold_count(linear_level_series: pd.Series) -> None:
    """The number of folds equals observations beyond the training window."""
    results = walk_forward(linear_level_series, SeasonalNaiveForecaster, min_train=24)
    assert len(results) == len(linear_level_series) - 24


def test_walk_forward_columns(linear_level_series: pd.Series) -> None:
    """Each fold records predicted, actual, and absolute-error values."""
    results = walk_forward(linear_level_series, SeasonalNaiveForecaster, min_train=24)
    assert set(results.columns) == {
        "predicted_change",
        "actual_change",
        "absolute_error",
    }


def test_walk_forward_actual_change_matches_series(
    linear_level_series: pd.Series,
) -> None:
    """On the constant-change series every realised change is the slope."""
    results = walk_forward(linear_level_series, SeasonalNaiveForecaster, min_train=24)
    assert results["actual_change"].eq(1_000.0).all()


def test_walk_forward_no_leakage(linear_level_series: pd.Series) -> None:
    """The first target date sits exactly one month past the training window."""
    min_train = 24
    results = walk_forward(
        linear_level_series, SeasonalNaiveForecaster, min_train=min_train
    )
    assert results.index[0] == linear_level_series.index[min_train]


def test_walk_forward_too_short_raises() -> None:
    """A series no longer than the training window cannot be backtested."""
    series = pd.Series(
        range(10), index=pd.date_range("2020-01-01", periods=10, freq="MS")
    ).astype(float)
    with pytest.raises(ValueError, match="more than"):
        walk_forward(series, SeasonalNaiveForecaster, min_train=10)


def test_score_metric_math() -> None:
    """MAE and RMSE match hand-computed values on a crafted result frame."""
    results = pd.DataFrame(
        {
            "predicted_change": [100.0, 100.0, 100.0],
            "actual_change": [110.0, 80.0, 100.0],
            "absolute_error": [10.0, 20.0, 0.0],
        },
        index=pd.date_range("2020-01-01", periods=3, freq="MS"),
    )
    metrics = score(results, "demo")
    assert metrics.mean_absolute_error == pytest.approx(10.0)
    assert metrics.root_mean_squared_error == pytest.approx((500.0 / 3) ** 0.5)
    assert metrics.fold_count == 3


def test_score_mase_against_random_walk() -> None:
    """MASE divides model MAE by the one-step random-walk MAE of the change."""
    results = pd.DataFrame(
        {
            "predicted_change": [100.0, 100.0, 100.0],
            "actual_change": [100.0, 200.0, 100.0],
            "absolute_error": [0.0, 100.0, 0.0],
        },
        index=pd.date_range("2020-01-01", periods=3, freq="MS"),
    )
    # Random-walk errors of the change: |200-100|=100, |100-200|=100 -> mean 100.
    # Model MAE = (0+100+0)/3 = 33.333 -> MASE = 0.3333.
    metrics = score(results, "demo")
    assert metrics.mean_absolute_scaled_error == pytest.approx(1.0 / 3.0)


def test_score_empty_raises() -> None:
    """Scoring an empty result frame is an error."""
    empty = pd.DataFrame(
        {"predicted_change": [], "actual_change": [], "absolute_error": []}
    )
    with pytest.raises(ValueError, match="empty"):
        score(empty, "demo")


def test_score_zero_naive_error_is_infinite() -> None:
    """A degenerate constant change gives an infinite MASE."""
    results = pd.DataFrame(
        {
            "predicted_change": [90.0, 90.0, 90.0],
            "actual_change": [100.0, 100.0, 100.0],
            "absolute_error": [10.0, 10.0, 10.0],
        },
        index=pd.date_range("2020-01-01", periods=3, freq="MS"),
    )
    metrics = score(results, "demo")
    assert metrics.mean_absolute_scaled_error == float("inf")


def test_evaluate_returns_metrics(linear_level_series: pd.Series) -> None:
    """The evaluate helper returns populated BacktestMetrics."""
    metrics = evaluate(linear_level_series, EtsForecaster, "ets-holt", min_train=24)
    assert isinstance(metrics, BacktestMetrics)
    assert metrics.model_name == "ets-holt"
    assert metrics.fold_count == len(linear_level_series) - 24
