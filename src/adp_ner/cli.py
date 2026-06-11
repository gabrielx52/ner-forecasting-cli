"""Command-line interface for tracking and forecasting the ADP NER print.

Three commands map to the three things a user needs:

* ``history``  - see the historical numbers.
* ``forecast`` - see the prediction for next month.
* ``explain``  - understand why that prediction was made.

Output uses plain aligned text only, so the tool has no presentation
dependencies and stays easy to pipe into other programs.
"""

from pathlib import Path
from typing import Annotated

import typer

from adp_ner.data import (
    NATIONAL_AGG_RIS,
    NATIONAL_CATEGORY,
    Metric,
    get_series,
    load_history,
)
from adp_ner.explain import build_explanation
from adp_ner.models.base import ForecastResult
from adp_ner.models.baseline import SeasonalNaiveForecaster
from adp_ner.models.statistical import EtsForecaster

app = typer.Typer(
    add_completion=False,
    help="Track the ADP National Employment Report and forecast the next print.",
)

DataPathOption = Annotated[
    Path | None,
    typer.Option("--data", help="Path to the ADP history CSV."),
]
AggRisOption = Annotated[
    str,
    typer.Option("--agg-ris", help="Aggregation level, for example National."),
]
CategoryOption = Annotated[
    str,
    typer.Option("--category", help="Category within the aggregation level."),
]
ConfidenceOption = Annotated[
    float,
    typer.Option("--confidence", min=0.5, max=0.999, help="Interval coverage."),
]
MinTrainOption = Annotated[
    int,
    typer.Option("--min-train", min=12, help="Minimum backtest training window."),
]


def _format_jobs(value: float) -> str:
    """Format a jobs figure with a sign and thousands separators."""
    return f"{value:+,.0f}"


def _select_forecaster(model: str) -> SeasonalNaiveForecaster | EtsForecaster:
    """Map a model name to a fresh forecaster instance."""
    if model == EtsForecaster.name:
        return EtsForecaster()
    if model == SeasonalNaiveForecaster.name:
        return SeasonalNaiveForecaster()
    raise typer.BadParameter(
        f"unknown model {model!r}; choose "
        f"{EtsForecaster.name!r} or {SeasonalNaiveForecaster.name!r}"
    )


def _print_forecast(forecast: ForecastResult) -> None:
    """Print the headline forecast block shared by commands."""
    coverage_percent = round(forecast.confidence * 100)
    typer.echo(f"Model:        {forecast.model_name}")
    typer.echo(f"As of:        {forecast.as_of_date}")
    typer.echo(f"Next print:   {forecast.target_date}")
    typer.echo(f"Forecast change:  {_format_jobs(forecast.point_change)} jobs")
    typer.echo(
        f"{coverage_percent}% interval:    "
        f"[{_format_jobs(forecast.lower_change)}, "
        f"{_format_jobs(forecast.upper_change)}] jobs"
    )
    typer.echo(f"Forecast level:   {forecast.point_level:,.0f}")


@app.command()
def history(
    months: Annotated[
        int,
        typer.Option("--months", min=1, help="Number of recent months to show."),
    ] = 12,
    metric: Annotated[
        Metric,
        typer.Option("--metric", help="Show the level or the monthly change."),
    ] = Metric.CHANGE,
    agg_ris: AggRisOption = NATIONAL_AGG_RIS,
    category: CategoryOption = NATIONAL_CATEGORY,
    data: DataPathOption = None,
) -> None:
    """Show recent historical ADP numbers for a slice."""
    history_frame = load_history(data)
    series = get_series(
        history_frame, agg_ris=agg_ris, category=category, metric=metric
    )
    recent = series.tail(months)

    heading = "change (jobs)" if metric is Metric.CHANGE else "level (jobs)"
    typer.echo(f"{agg_ris} / {category} - seasonally adjusted {heading}")
    typer.echo(f"{'date':<12}{heading:>18}")
    for observed_date, value in recent.items():
        formatted = _format_jobs(value) if metric is Metric.CHANGE else f"{value:,.0f}"
        typer.echo(f"{observed_date.date()!s:<12}{formatted:>18}")


@app.command()
def forecast(
    model: Annotated[
        str,
        typer.Option("--model", help="Forecasting model to use."),
    ] = EtsForecaster.name,
    confidence: ConfidenceOption = 0.95,
    agg_ris: AggRisOption = NATIONAL_AGG_RIS,
    category: CategoryOption = NATIONAL_CATEGORY,
    data: DataPathOption = None,
) -> None:
    """Forecast next month's ADP print for a slice."""
    history_frame = load_history(data)
    level_series = get_series(
        history_frame, agg_ris=agg_ris, category=category, metric=Metric.LEVEL
    )
    forecaster = _select_forecaster(model)
    forecaster.fit(level_series)
    _print_forecast(forecaster.predict(confidence=confidence))


@app.command()
def explain(
    model: Annotated[
        str,
        typer.Option("--model", help="Forecasting model to explain."),
    ] = EtsForecaster.name,
    confidence: ConfidenceOption = 0.95,
    min_train: MinTrainOption = 36,
    agg_ris: AggRisOption = NATIONAL_AGG_RIS,
    category: CategoryOption = NATIONAL_CATEGORY,
    data: DataPathOption = None,
) -> None:
    """Explain the forecast: model drivers and backtested accuracy."""
    history_frame = load_history(data)
    level_series = get_series(
        history_frame, agg_ris=agg_ris, category=category, metric=Metric.LEVEL
    )
    forecaster = _select_forecaster(model)
    explanation = build_explanation(
        level_series, forecaster, min_train=min_train, confidence=confidence
    )

    _print_forecast(explanation.forecast)
    typer.echo("")
    typer.echo("Why this number (model drivers):")
    for driver_name, driver_value in explanation.drivers.items():
        if isinstance(driver_value, float):
            typer.echo(f"  {driver_name}: {driver_value:,.4f}")
        else:
            typer.echo(f"  {driver_name}: {driver_value}")

    typer.echo("")
    typer.echo("Backtested accuracy on the monthly change (lower is better):")
    typer.echo("MASE is scaled against a one-step random walk; below 1.0 is skill.")
    typer.echo(f"{'model':<16}{'folds':>7}{'MAE':>14}{'RMSE':>14}{'MASE':>9}")
    for metrics in (explanation.model_metrics, explanation.baseline_metrics):
        typer.echo(
            f"{metrics.model_name:<16}{metrics.fold_count:>7}"
            f"{metrics.mean_absolute_error:>14,.0f}"
            f"{metrics.root_mean_squared_error:>14,.0f}"
            f"{metrics.mean_absolute_scaled_error:>9.3f}"
        )
    typer.echo("")
    model_error = explanation.model_metrics.mean_absolute_error
    baseline_error = explanation.baseline_metrics.mean_absolute_error
    improvement = 1.0 - model_error / baseline_error
    verdict = "beats" if improvement > 0.0 else "does not beat"
    typer.echo(
        f"The {explanation.model_metrics.model_name} model {verdict} the "
        f"{explanation.baseline_metrics.model_name} baseline: "
        f"{improvement:+.1%} mean absolute error."
    )


if __name__ == "__main__":
    app()
