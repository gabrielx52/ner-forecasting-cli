"""Shared fixtures for the nerd_cast test suite."""

import pandas as pd
import pytest

from nerd_cast.data import load_history


@pytest.fixture
def linear_level_series() -> pd.Series:
    """Build a clean upward-trending monthly level series with a known slope.

    The level rises by exactly 1,000 each month from a base of 100,000, so the
    month-over-month change is a constant 1,000. Models and metrics can be
    checked against this exact value.
    """
    dates = pd.date_range("2010-01-01", periods=48, freq="MS")
    values = [100_000.0 + 1_000.0 * step for step in range(len(dates))]
    series = pd.Series(values, index=dates, name="U.S.")
    series.index.freq = "MS"
    return series


@pytest.fixture
def tidy_history_frame() -> pd.DataFrame:
    """Build a tiny tidy history frame covering two slices for accessor tests."""
    rows = [
        ("M", "National", "U.S.", "2010-01-01", 100_000.0, 100_500.0),
        ("M", "National", "U.S.", "2010-02-01", 101_000.0, 101_200.0),
        ("M", "National", "U.S.", "2010-03-01", 102_500.0, 102_000.0),
        ("M", "Industry", "Manufacturing", "2010-01-01", 20_000.0, 20_100.0),
        ("M", "Industry", "Manufacturing", "2010-02-01", 20_400.0, 20_300.0),
    ]
    return pd.DataFrame(
        rows,
        columns=["timestep", "agg_RIS", "category", "date", "NER", "NER_SA"],
    ).assign(date=lambda frame: pd.to_datetime(frame["date"]))


@pytest.fixture(scope="session")
def real_history_frame() -> pd.DataFrame:
    """Load the bundled ADP history once for the whole session."""
    return load_history()
