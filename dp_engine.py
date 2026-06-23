"""
Differential privacy engine using the OpenDP library.

Laplace mechanism formal definition:
    M(D) = f(D) + Lap(0, Δf/ε)

where:
    f(D)  = true query answer on dataset D
    Δf    = global sensitivity (max |f(D) − f(D')| over neighboring D, D')
    ε     = privacy budget (epsilon)
    Lap(0, b) = Laplace distribution with scale b

Privacy guarantee (ε-differential privacy):
    For all neighboring datasets D, D' (differing by one record):
        P[M(D) ∈ S] ≤ e^ε × P[M(D') ∈ S]   for all output sets S

OpenDP validates this guarantee at construction time.
"""

import math
from dataclasses import dataclass

import opendp.prelude as dp

dp.enable_features("contrib")


@dataclass
class QueryResult:
    query_id: str
    epsilon: float
    sensitivity: float
    scale: float            # = Δf / ε
    true_value: float
    noisy_value: float
    noise_added: float
    expected_abs_error: float   # E[|noise|] = scale
    ci_95_half_width: float     # 95% of noise within ±this
    relative_error_pct: float | None  # expected abs error as % of true value


def _make_laplace_measurement(scale: float):
    """Return an OpenDP Laplace measurement on a non-NaN scalar float."""
    return dp.m.make_laplace(
        dp.atom_domain(T=float, nan=False),
        dp.absolute_distance(T=float),
        scale=scale,
    )


def laplace_mechanism(
    true_value: float,
    sensitivity: float,
    epsilon: float,
    query_id: str = "unknown",
) -> QueryResult:
    """
    Apply the Laplace mechanism via OpenDP.

    Args:
        true_value: exact query result on the dataset
        sensitivity: Δf, max change from adding/removing one person
        epsilon: privacy budget (smaller = more private = more noise)
        query_id: identifier for result labeling

    Returns:
        QueryResult with noise statistics
    """
    scale = sensitivity / epsilon
    m = _make_laplace_measurement(scale)
    noisy = float(m(float(true_value)))
    noise = noisy - true_value

    rel_err = None
    if true_value != 0.0:
        rel_err = (scale / abs(true_value)) * 100.0

    return QueryResult(
        query_id=query_id,
        epsilon=epsilon,
        sensitivity=sensitivity,
        scale=scale,
        true_value=true_value,
        noisy_value=noisy,
        noise_added=noise,
        expected_abs_error=scale,
        ci_95_half_width=2.0 * scale,  # 95% of Laplace mass within ±2b
        relative_error_pct=rel_err,
    )


# ── Sensitivity helpers ────────────────────────────────────────────────────

def count_sensitivity() -> float:
    """
    Count query: adding/removing one person changes the count by exactly 1.
    Δf = 1
    """
    return 1.0


def mean_sensitivity(lower: float, upper: float, n: int) -> float:
    """
    Bounded mean query over n records with values in [lower, upper].
    Adding/removing one record changes the mean by at most (upper − lower) / n.
    Δf = (upper − lower) / n
    """
    return (upper - lower) / n


def proportion_sensitivity(n: int) -> float:
    """
    Proportion query over n records.
    Adding/removing one record changes the proportion by at most 1/n.
    Δf = 1 / n
    """
    return 1.0 / n


# ── Noise statistics ───────────────────────────────────────────────────────

def laplace_scale(sensitivity: float, epsilon: float) -> float:
    """b = Δf / ε"""
    return sensitivity / epsilon


def laplace_std(sensitivity: float, epsilon: float) -> float:
    """σ = b√2 = Δf√2 / ε"""
    return laplace_scale(sensitivity, epsilon) * math.sqrt(2)


def laplace_expected_abs_error(sensitivity: float, epsilon: float) -> float:
    """E[|noise|] = b = Δf / ε"""
    return laplace_scale(sensitivity, epsilon)


def laplace_ci_95(sensitivity: float, epsilon: float) -> float:
    """95% of Laplace noise falls within ±2b = ±2Δf/ε."""
    return 2.0 * laplace_scale(sensitivity, epsilon)


def privacy_loss_ratio(epsilon: float) -> float:
    """
    Maximum likelihood ratio e^ε bounding the adversary's belief update.
    An adversary observing the noisy output can multiply their belief
    about any individual's presence by at most this factor.
    """
    return math.exp(epsilon)


def compose_epsilon(epsilons: list[float]) -> float:
    """
    Basic composition theorem: k sequential ε_i-DP mechanisms together
    spend ε = Σ ε_i total privacy budget.
    """
    return sum(epsilons)


def run_all_queries(
    true_count: float,
    true_mean: float,
    true_proportion: float,
    n: int,
    age_min: float,
    age_max: float,
    epsilon: float,
) -> dict[str, QueryResult]:
    """Run all three demonstration queries at a given epsilon."""
    s_count = count_sensitivity()
    s_mean = mean_sensitivity(age_min, age_max, n)
    s_prop = proportion_sensitivity(n)

    return {
        "count_high_income": laplace_mechanism(true_count, s_count, epsilon, "count_high_income"),
        "mean_age": laplace_mechanism(true_mean, s_mean, epsilon, "mean_age"),
        "proportion_college": laplace_mechanism(true_proportion, s_prop, epsilon, "proportion_college"),
    }
