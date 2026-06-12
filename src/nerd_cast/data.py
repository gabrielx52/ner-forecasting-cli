"""Loading and slicing of the ADP National Employment Report history.

The raw CSV stores employment *levels* in a tidy long format. Every row is a
single observation for one aggregation slice (for example National, or the
Manufacturing industry) at one date. Two level columns are provided:

* ``NER``    - the raw (not seasonally adjusted) employment level.
* ``NER_SA`` - the seasonally adjusted employment level.

The headline ADP print is the month-over-month *change* in the seasonally
adjusted level, so this module exposes both the level series and the derived
change series through a single accessor.
"""

from enum import StrEnum
from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "ADP_NER_history.csv"
)

EXPECTED_COLUMNS = ("timestep", "agg_RIS", "category", "date", "NER", "NER_SA")

MONTHLY_TIMESTEP = "M"
NATIONAL_AGG_RIS = "National"
NATIONAL_CATEGORY = "U.S."


class Metric(StrEnum):
    """Selects which series an accessor returns for a slice."""

    LEVEL = "level"
    CHANGE = "change"


def load_history(csv_path: Path | None = None) -> pd.DataFrame:
    """Load the ADP history CSV into a validated tidy DataFrame.

    Args:
        csv_path: Location of the history CSV. Defaults to the bundled data file.

    Returns:
        A DataFrame with parsed ``date`` values and numeric level columns,
        sorted by ``date`` within each slice.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If the CSV is missing any expected column.
    """
    resolved_path = csv_path if csv_path is not None else DEFAULT_DATA_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"ADP history CSV not found at {resolved_path}")

    history = pd.read_csv(resolved_path, parse_dates=["date"])

    missing_columns = [
        column for column in EXPECTED_COLUMNS if column not in history.columns
    ]
    if missing_columns:
        raise ValueError(f"CSV is missing expected columns: {missing_columns}")

    history["NER"] = pd.to_numeric(history["NER"])
    history["NER_SA"] = pd.to_numeric(history["NER_SA"])
    return history.sort_values("date").reset_index(drop=True)


def get_series(
    history: pd.DataFrame,
    *,
    agg_ris: str = NATIONAL_AGG_RIS,
    category: str = NATIONAL_CATEGORY,
    metric: Metric = Metric.LEVEL,
    seasonally_adjusted: bool = True,
    timestep: str = MONTHLY_TIMESTEP,
) -> pd.Series:
    """Return a date-indexed series for one aggregation slice.

    Args:
        history: The tidy DataFrame produced by :func:`load_history`.
        agg_ris: Aggregation level, for example ``"National"`` or ``"Industry"``.
        category: Category within the aggregation level.
        metric: ``Metric.LEVEL`` for the employment level or ``Metric.CHANGE``
            for the month-over-month difference (the headline print).
        seasonally_adjusted: Use the ``NER_SA`` column when ``True``.
        timestep: ``"M"`` for monthly or ``"W"`` for weekly observations.

    Returns:
        A float series indexed by date, sorted ascending. A change series has
        its first (undefined) observation dropped.

    Raises:
        ValueError: If the slice matches no rows.
    """
    value_column = "NER_SA" if seasonally_adjusted else "NER"
    slice_mask = (
        (history["agg_RIS"] == agg_ris)
        & (history["category"] == category)
        & (history["timestep"] == timestep)
    )
    sliced = history.loc[slice_mask, ["date", value_column]]
    if sliced.empty:
        raise ValueError(
            f"No rows for agg_ris={agg_ris!r}, category={category!r}, "
            f"timestep={timestep!r}"
        )

    series = (
        sliced.set_index("date")[value_column]
        .sort_index()
        .astype(float)
        .rename(category)
    )
    if len(series) >= 3:
        inferred_frequency = pd.infer_freq(series.index)
        if inferred_frequency is not None:
            series.index.freq = inferred_frequency
    if metric is Metric.CHANGE:
        series = series.diff().dropna()
    return series


def list_categories(history: pd.DataFrame, agg_ris: str) -> list[str]:
    """Return the sorted unique categories available for an aggregation level."""
    categories = history.loc[history["agg_RIS"] == agg_ris, "category"].unique()
    return sorted(str(category) for category in categories)
