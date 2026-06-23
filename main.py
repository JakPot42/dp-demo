"""
Differential Privacy Demonstration Tool (P48)

CLI commands:
    demo     -- Full interactive demo across all epsilon presets
    explore  -- Deep-dive on a single epsilon value
    compare  -- Side-by-side accuracy table across epsilons
    budget   -- Show privacy budget composition under repeated queries
"""

import click
from rich.console import Console

from claude_analyst import get_explanation
from config import EPSILON_PRESETS, QUERIES
from dataset import CensusDataset
from display import (
    console,
    print_ascii_noise_bar,
    print_banner,
    print_budget_composition,
    print_epsilon_comparison_table,
    print_math_primer,
    print_queries_table,
    print_single_epsilon_detail,
)
from dp_engine import run_all_queries
from risk_analyzer import all_risk_profiles


def _load_data() -> CensusDataset:
    ds = CensusDataset()
    return ds


def _run_epsilon(ds: CensusDataset, epsilon: float) -> tuple:
    summary = ds.summary()
    results = run_all_queries(
        true_count=float(summary["true_count_high_income"]),
        true_mean=summary["true_mean_age"],
        true_proportion=summary["true_proportion_college"],
        n=summary["n"],
        age_min=summary["age_bounds"][0],
        age_max=summary["age_bounds"][1],
        epsilon=epsilon,
    )
    risks = all_risk_profiles(
        epsilon=epsilon,
        true_count=summary["true_count_high_income"],
        true_mean=summary["true_mean_age"],
        true_proportion=summary["true_proportion_college"],
        n=summary["n"],
        age_min=summary["age_bounds"][0],
        age_max=summary["age_bounds"][1],
    )
    explanation = get_explanation(epsilon, results)
    return results, risks, explanation


@click.group()
def cli():
    """Differential Privacy Demonstration Tool.

    Shows the real privacy/accuracy tradeoff using the Laplace mechanism
    (OpenDP library) on a 1,000-record US Census income sample.

    Run 'demo' for the full interactive walkthrough.
    """
    pass


@cli.command()
@click.option(
    "--epsilon", "-e",
    default=None,
    type=float,
    multiple=True,
    help="Epsilon value(s) to demo (default: all 7 presets)",
)
@click.option("--primer/--no-primer", default=True, help="Show math primer")
def demo(epsilon, primer):
    """Full interactive demo: all queries, all epsilons, with Claude explanations."""
    print_banner()

    if primer:
        print_math_primer()

    ds = _load_data()
    summary = ds.summary()
    console.print(
        f"\n[bold]Dataset:[/] {summary['n']} workers, "
        f"true high-earners: {summary['true_count_high_income']}, "
        f"true mean age: {summary['true_mean_age']:.2f} yrs, "
        f"true college-educated: {summary['true_proportion_college']*100:.1f}%\n"
    )

    print_queries_table()
    console.print()

    epsilons = list(epsilon) if epsilon else EPSILON_PRESETS

    # Compute all results for comparison table
    all_results = {}
    for eps in epsilons:
        results, _, _ = _run_epsilon(ds, eps)
        all_results[eps] = results

    print_epsilon_comparison_table(all_results, epsilons)

    # ASCII noise bars for count query (sensitivity=1) at each epsilon
    console.print("\n[bold]Noise scale visualization — Count query (Δf = 1):[/]")
    for eps in epsilons:
        bar = print_ascii_noise_bar(1.0, eps, width=50)
        console.print(f"  ε={eps:5.2f}  {bar}")

    # Show Claude explanation for the middle epsilon if available
    middle = epsilons[len(epsilons) // 2]
    console.print(f"\n[dim]Showing detailed Claude explanation for ε = {middle}...[/]")
    explanation = get_explanation(middle)
    from rich.panel import Panel
    console.print(Panel(
        explanation,
        title=f"[bold cyan]Claude's Explanation — ε = {middle}[/]",
        border_style="cyan",
    ))

    console.print(
        "\n[dim]Run [bold]explore --epsilon VALUE[/] for full detail on any single epsilon.[/]"
    )


@cli.command()
@click.option(
    "--epsilon", "-e",
    required=True,
    type=float,
    help="Privacy budget to explore (e.g. 0.1, 1.0, 5.0)",
)
def explore(epsilon):
    """Deep-dive on a single epsilon: noise, accuracy, re-identification risk, Claude explanation."""
    print_banner()
    ds = _load_data()
    results, risks, explanation = _run_epsilon(ds, epsilon)
    print_single_epsilon_detail(epsilon, results, risks, explanation)


@cli.command()
@click.option(
    "--epsilon", "-e",
    default=None,
    type=float,
    multiple=True,
    help="Epsilon values to compare (default: all presets)",
)
def compare(epsilon):
    """Side-by-side accuracy comparison table across epsilon values."""
    print_banner()
    ds = _load_data()
    epsilons = list(epsilon) if epsilon else EPSILON_PRESETS

    all_results = {}
    for eps in epsilons:
        results, _, _ = _run_epsilon(ds, eps)
        all_results[eps] = results

    print_epsilon_comparison_table(all_results, epsilons)

    console.print("\n[bold]Key takeaway:[/]")
    console.print(
        "  The count query (Δf=1) shows the starkest tradeoff — noise scale = 1/ε.\n"
        "  The mean-age query (Δf=0.073) shows how large n makes DP nearly free.\n"
        "  The proportion query (Δf=0.001) is even cheaper — DP barely affects it.\n"
        "  [dim]Re-identification risk depends on which statistic the attacker targets.[/]"
    )


@cli.command()
@click.option(
    "--epsilon", "-e",
    required=True,
    type=float,
    help="Per-query epsilon budget",
)
@click.option(
    "--queries", "-k",
    required=True,
    type=int,
    help="Number of queries to compose",
)
def budget(epsilon, queries):
    """Show how repeated queries drain the privacy budget (composition theorem)."""
    print_banner()
    print_budget_composition(epsilon, queries)


if __name__ == "__main__":
    cli()
