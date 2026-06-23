"""CLI smoke tests using Click's test runner."""

import pytest
from click.testing import CliRunner
from main import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_demo_runs_without_error(runner):
    result = runner.invoke(cli, ["demo", "--no-primer"])
    assert result.exit_code == 0, result.output


def test_demo_shows_dataset_info(runner):
    result = runner.invoke(cli, ["demo", "--no-primer"])
    assert "workers" in result.output.lower() or "dataset" in result.output.lower()


def test_demo_with_specific_epsilon(runner):
    result = runner.invoke(cli, ["demo", "--no-primer", "--epsilon", "1.0"])
    assert result.exit_code == 0


def test_demo_with_multiple_epsilons(runner):
    result = runner.invoke(cli, ["demo", "--no-primer", "--epsilon", "0.1", "--epsilon", "1.0"])
    assert result.exit_code == 0


def test_explore_valid_epsilon(runner):
    result = runner.invoke(cli, ["explore", "--epsilon", "1.0"])
    assert result.exit_code == 0


def test_explore_shows_query_results(runner):
    result = runner.invoke(cli, ["explore", "--epsilon", "1.0"])
    assert "count" in result.output.lower() or "mean" in result.output.lower()


def test_explore_small_epsilon(runner):
    result = runner.invoke(cli, ["explore", "--epsilon", "0.1"])
    assert result.exit_code == 0


def test_explore_large_epsilon(runner):
    result = runner.invoke(cli, ["explore", "--epsilon", "10.0"])
    assert result.exit_code == 0


def test_compare_runs_without_error(runner):
    result = runner.invoke(cli, ["compare"])
    assert result.exit_code == 0


def test_compare_shows_epsilon_values(runner):
    result = runner.invoke(cli, ["compare"])
    assert "0.01" in result.output or "0.1" in result.output


def test_compare_with_custom_epsilons(runner):
    result = runner.invoke(cli, ["compare", "--epsilon", "0.5", "--epsilon", "2.0"])
    assert result.exit_code == 0


def test_budget_command(runner):
    result = runner.invoke(cli, ["budget", "--epsilon", "1.0", "--queries", "5"])
    assert result.exit_code == 0


def test_budget_shows_total(runner):
    result = runner.invoke(cli, ["budget", "--epsilon", "1.0", "--queries", "5"])
    assert "5.0" in result.output or "total" in result.output.lower()


def test_budget_composition_math(runner):
    # 3 queries × ε=2.0 = ε=6.0 total
    result = runner.invoke(cli, ["budget", "--epsilon", "2.0", "--queries", "3"])
    assert "6.0" in result.output


def test_help_shows_commands(runner):
    result = runner.invoke(cli, ["--help"])
    assert "demo" in result.output
    assert "explore" in result.output
    assert "compare" in result.output
    assert "budget" in result.output
