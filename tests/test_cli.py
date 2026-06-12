"""Tests for the command-line interface."""

from typer.testing import CliRunner

from nerd_cast.cli import app

runner = CliRunner()


def test_history_default_runs() -> None:
    """The history command prints a header and recent rows."""
    result = runner.invoke(app, ["history", "--months", "3"])
    assert result.exit_code == 0
    assert "seasonally adjusted" in result.stdout
    assert "2026" in result.stdout


def test_history_level_metric() -> None:
    """The history command can show levels instead of changes."""
    result = runner.invoke(app, ["history", "--months", "2", "--metric", "level"])
    assert result.exit_code == 0
    assert "level (jobs)" in result.stdout


def test_forecast_default_runs() -> None:
    """The forecast command prints a next-print block."""
    result = runner.invoke(app, ["forecast"])
    assert result.exit_code == 0
    assert "Next print" in result.stdout
    assert "Forecast change" in result.stdout


def test_forecast_baseline_model() -> None:
    """The forecast command accepts the baseline model by name."""
    result = runner.invoke(app, ["forecast", "--model", "seasonal-naive"])
    assert result.exit_code == 0
    assert "seasonal-naive" in result.stdout


def test_forecast_unknown_model_errors() -> None:
    """An unknown model name is rejected with a non-zero exit code."""
    result = runner.invoke(app, ["forecast", "--model", "wizardry"])
    assert result.exit_code != 0


def test_explain_runs_and_reports_skill() -> None:
    """The explain command prints drivers and the backtest verdict."""
    result = runner.invoke(app, ["explain", "--min-train", "60"])
    assert result.exit_code == 0
    assert "Why this number" in result.stdout
    assert "MASE" in result.stdout
    assert "baseline" in result.stdout


def test_help_lists_commands() -> None:
    """The top-level help advertises all three commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("history", "forecast", "explain"):
        assert command in result.stdout
