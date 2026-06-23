"""Rich display helpers for the DP demo CLI."""

import math

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import EPSILON_PRESETS, QUERIES
from dp_engine import QueryResult, privacy_loss_ratio
from risk_analyzer import RiskProfile

console = Console()

PROTECTION_COLORS = {
    "STRONG": "bold green",
    "MODERATE": "yellow",
    "WEAK": "bold yellow",
    "NEGLIGIBLE": "bold red",
}

EPSILON_COLORS = {
    0.01: "bright_green",
    0.1: "green",
    0.5: "cyan",
    1.0: "yellow",
    2.0: "orange3",
    5.0: "red",
    10.0: "bold red",
}


def epsilon_color(epsilon: float) -> str:
    return EPSILON_COLORS.get(epsilon, "white")


def print_banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]Differential Privacy Demonstration Tool[/]\n"
        "[dim]P48 — Defense Portfolio | Dataset: 1994 US Census Bureau (1,000 records)[/]",
        border_style="cyan",
    ))


def print_math_primer() -> None:
    console.print(Panel(
        "[bold]Core definitions[/]\n\n"
        "  [cyan]ε (epsilon)[/]      Privacy budget. Smaller = more private = more noise.\n"
        "  [cyan]Δf (sensitivity)[/] Max query change from adding/removing one person.\n"
        "  [cyan]scale = Δf/ε[/]    Laplace noise scale. Determines how much noise is added.\n\n"
        "[bold]Laplace mechanism[/]  M(D) = f(D) + Lap(0, Δf/ε)\n\n"
        "[bold]ε-DP guarantee[/]    For any two neighboring datasets D, D' (differ by one record):\n"
        "                     P[M(D) ∈ S] ≤ [cyan]e^ε[/] × P[M(D') ∈ S]   for all output sets S\n\n"
        "  This bounds the adversary's [bold]likelihood ratio[/] — how much observing the\n"
        "  noisy output can shift their belief about any single individual.",
        title="[bold]Differential Privacy Math[/]",
        border_style="dim",
    ))


def print_queries_table() -> None:
    t = Table(title="Demonstration Queries", show_lines=True)
    t.add_column("Query", style="bold")
    t.add_column("Description")
    t.add_column("Sensitivity Δf", justify="right")
    t.add_column("Formula", style="dim")

    for q in QUERIES:
        t.add_row(
            q["name"],
            q["description"],
            f"{q['sensitivity']:.6g}",
            q["sensitivity_formula"],
        )
    console.print(t)


def print_epsilon_comparison_table(
    results_by_epsilon: dict[float, dict[str, QueryResult]],
    epsilons: list[float] | None = None,
) -> None:
    epsilons = epsilons or EPSILON_PRESETS

    t = Table(
        title="Privacy / Accuracy Tradeoff Across ε Values",
        show_lines=True,
        caption="Expected absolute error = Δf/ε (Laplace scale). Lower ε = more noise.",
    )
    t.add_column("ε (epsilon)", justify="right")
    t.add_column("e^ε (belief ratio)", justify="right")
    t.add_column("Protection", justify="center")
    t.add_column("Count error (±people)", justify="right")
    t.add_column("Mean age error (±yrs)", justify="right")
    t.add_column("Proportion error (±)", justify="right")

    from risk_analyzer import _protection_level

    for eps in epsilons:
        ratio = privacy_loss_ratio(eps)
        level = _protection_level(eps)
        color = PROTECTION_COLORS[level]
        eps_color = epsilon_color(eps)

        if eps in results_by_epsilon:
            r = results_by_epsilon[eps]
            count_err = f"{r['count_high_income'].expected_abs_error:.3f}"
            mean_err = f"{r['mean_age'].expected_abs_error:.5f}"
            prop_err = f"{r['proportion_college'].expected_abs_error:.6f}"
        else:
            count_err = mean_err = prop_err = "—"

        t.add_row(
            f"[{eps_color}]{eps}[/]",
            f"{ratio:.1f}×",
            f"[{color}]{level}[/]",
            count_err,
            mean_err,
            prop_err,
        )
    console.print(t)


def print_single_epsilon_detail(
    epsilon: float,
    results: dict[str, QueryResult],
    risk_profiles: dict[str, RiskProfile],
    explanation: str,
) -> None:
    ratio = privacy_loss_ratio(epsilon)
    from risk_analyzer import _protection_level
    level = _protection_level(epsilon)
    color = PROTECTION_COLORS[level]
    eps_color = epsilon_color(epsilon)

    console.print(Panel(
        f"[{eps_color}]ε = {epsilon}[/]   "
        f"scale = Δf/ε   belief-ratio = e^{epsilon} = [{color}]{ratio:.2f}×[/]   "
        f"protection = [{color}][bold]{level}[/][/]",
        title="[bold]Epsilon Detail[/]",
        border_style=eps_color,
    ))

    # Query results table
    t = Table(show_lines=True)
    t.add_column("Query")
    t.add_column("True value", justify="right")
    t.add_column("Noisy value", justify="right")
    t.add_column("Noise added", justify="right")
    t.add_column("Exp. abs error", justify="right")
    t.add_column("95% CI half-width", justify="right")
    t.add_column("Relative error", justify="right")

    for qid, r in results.items():
        rel = f"{r.relative_error_pct:.2f}%" if r.relative_error_pct is not None else "—"
        noise_color = "green" if abs(r.noise_added) < r.expected_abs_error else "yellow"
        t.add_row(
            qid.replace("_", " "),
            f"{r.true_value:.4f}",
            f"{r.noisy_value:.4f}",
            f"[{noise_color}]{r.noise_added:+.4f}[/]",
            f"{r.expected_abs_error:.6f}",
            f"±{r.ci_95_half_width:.6f}",
            rel,
        )
    console.print(t)

    # Re-identification risk
    console.print("\n[bold]Re-identification Risk — Alice/John Scenario[/]")
    for qid, rp in risk_profiles.items():
        console.print(Panel(
            rp.alice_scenario + "\n\n[dim]" + rp.what_noise_prevents + "[/]",
            title=f"[bold]{qid.replace('_', ' ')}[/]",
            border_style="dim",
        ))

    # Claude explanation
    console.print(Panel(
        explanation,
        title=f"[bold cyan]Claude's Plain-Language Explanation — ε = {epsilon}[/]",
        border_style="cyan",
    ))


def print_budget_composition(
    epsilon_per_query: float,
    n_queries: int,
) -> None:
    total = epsilon_per_query * n_queries
    from risk_analyzer import _protection_level
    level_per = _protection_level(epsilon_per_query)
    level_total = _protection_level(total)

    console.print(Panel(
        f"  Per-query budget:    ε = {epsilon_per_query} ({level_per})\n"
        f"  Number of queries:  {n_queries}\n"
        f"  [bold]Total budget spent: ε = {total} ({level_total})[/]\n\n"
        f"  Composition theorem (basic): k × ε_per = ε_total\n"
        f"  If each query spends ε={epsilon_per_query}, running {n_queries} queries "
        f"degrades privacy to ε={total}.\n"
        f"  Belief ratio per query: {math.exp(epsilon_per_query):.2f}×\n"
        f"  Belief ratio after all queries: {math.exp(total):.2f}×",
        title="[bold]Privacy Budget Composition[/]",
        border_style="yellow",
    ))


def print_ascii_noise_bar(sensitivity: float, epsilon: float, width: int = 40) -> str:
    """Return an ASCII bar showing noise magnitude relative to sensitivity."""
    scale = sensitivity / epsilon
    # Log scale: scale from epsilon=10 (tiny) to epsilon=0.01 (huge)
    # Normalize: max_scale = sensitivity / 0.01
    max_scale = sensitivity / 0.01
    filled = min(width, int((scale / max_scale) * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] scale={scale:.5g}"
