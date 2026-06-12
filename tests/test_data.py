"""Tests for the data loading and slicing layer."""

from pathlib import Path

import pandas as pd
import pytest

from nerd_cast.data import (
    EXPECTED_COLUMNS,
    Metric,
    get_series,
    list_categories,
    load_history,
)


def test_load_history_returns_expected_columns(
    real_history_frame: pd.DataFrame,
) -> None:
    """The loaded frame exposes every expected column."""
    for column in EXPECTED_COLUMNS:
        assert column in real_history_frame.columns


def test_load_history_parses_dates_and_numerics(
    real_history_frame: pd.DataFrame,
) -> None:
    """Dates parse to datetimes and level columns are numeric."""
    assert pd.api.types.is_datetime64_any_dtype(real_history_frame["date"])
    assert pd.api.types.is_numeric_dtype(real_history_frame["NER"])
    assert pd.api.types.is_numeric_dtype(real_history_frame["NER_SA"])


def test_load_history_missing_file(tmp_path: Path) -> None:
    """A missing CSV raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_history(tmp_path / "does_not_exist.csv")


def test_load_history_missing_column(tmp_path: Path) -> None:
    """A CSV without an expected column raises ValueError."""
    incomplete = tmp_path / "incomplete.csv"
    incomplete.write_text(
        "timestep,agg_RIS,category,date\nM,National,U.S.,2010-01-01\n"
    )
    with pytest.raises(ValueError, match="missing expected columns"):
        load_history(incomplete)


def test_get_series_level_is_sorted_and_named(
    tidy_history_frame: pd.DataFrame,
) -> None:
    """A level slice is date-sorted, float, and named for its category."""
    series = get_series(tidy_history_frame, metric=Metric.LEVEL)
    assert list(series.index) == sorted(series.index)
    assert series.name == "U.S."
    assert series.iloc[0] == pytest.approx(100_500.0)


def test_get_series_change_is_diff_of_level(
    tidy_history_frame: pd.DataFrame,
) -> None:
    """The change series equals the first difference of the level series."""
    level = get_series(tidy_history_frame, metric=Metric.LEVEL)
    change = get_series(tidy_history_frame, metric=Metric.CHANGE)
    assert len(change) == len(level) - 1
    assert change.iloc[0] == pytest.approx(level.iloc[1] - level.iloc[0])


def test_get_series_raw_uses_ner_column(tidy_history_frame: pd.DataFrame) -> None:
    """Disabling seasonal adjustment reads the raw NER column."""
    series = get_series(
        tidy_history_frame, metric=Metric.LEVEL, seasonally_adjusted=False
    )
    assert series.iloc[0] == pytest.approx(100_000.0)


def test_get_series_selects_requested_slice(
    tidy_history_frame: pd.DataFrame,
) -> None:
    """A non-default slice returns only that category's rows."""
    series = get_series(
        tidy_history_frame, agg_ris="Industry", category="Manufacturing"
    )
    assert len(series) == 2
    assert series.name == "Manufacturing"


def test_get_series_unknown_slice_raises(tidy_history_frame: pd.DataFrame) -> None:
    """An empty slice raises ValueError."""
    with pytest.raises(ValueError, match="No rows"):
        get_series(tidy_history_frame, category="Nowhere")


def test_get_series_irregular_dates_leaves_freq_unset() -> None:
    """Irregular dates cannot infer a frequency, so the index keeps none."""
    rows = [
        ("M", "National", "U.S.", "2010-01-01", 100.0, 100.0),
        ("M", "National", "U.S.", "2010-02-01", 110.0, 110.0),
        ("M", "National", "U.S.", "2010-09-01", 120.0, 120.0),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["timestep", "agg_RIS", "category", "date", "NER", "NER_SA"],
    ).assign(date=lambda inner: pd.to_datetime(inner["date"]))
    series = get_series(frame, metric=Metric.LEVEL)
    assert series.index.freq is None
    assert len(series) == 3


def test_list_categories(tidy_history_frame: pd.DataFrame) -> None:
    """Categories are returned sorted for an aggregation level."""
    assert list_categories(tidy_history_frame, "National") == ["U.S."]
    assert list_categories(tidy_history_frame, "Industry") == ["Manufacturing"]
