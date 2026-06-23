"""
Tests for dp_engine.py — correct math, OpenDP integration, noise statistics.

Statistical tests use large samples and wide tolerances since Laplace noise
is random. Each statistical assertion should pass with probability > 99.99%.
"""

import math
import statistics

import pytest

from dp_engine import (
    QueryResult,
    compose_epsilon,
    count_sensitivity,
    laplace_ci_95,
    laplace_expected_abs_error,
    laplace_mechanism,
    laplace_scale,
    laplace_std,
    mean_sensitivity,
    privacy_loss_ratio,
    proportion_sensitivity,
    run_all_queries,
)


# ── Sensitivity helpers ────────────────────────────────────────────────────

def test_count_sensitivity_is_one():
    assert count_sensitivity() == 1.0


def test_mean_sensitivity_formula():
    """Δf = (upper - lower) / n"""
    s = mean_sensitivity(17, 90, 1000)
    assert abs(s - 73 / 1000) < 1e-10


def test_mean_sensitivity_shrinks_with_more_records():
    s100 = mean_sensitivity(0, 100, 100)
    s1000 = mean_sensitivity(0, 100, 1000)
    assert s1000 < s100


def test_proportion_sensitivity_formula():
    """Δf = 1/n"""
    assert abs(proportion_sensitivity(1000) - 0.001) < 1e-10


def test_proportion_sensitivity_shrinks_with_n():
    assert proportion_sensitivity(1000) < proportion_sensitivity(100)


# ── Noise statistics ───────────────────────────────────────────────────────

def test_laplace_scale_formula():
    """scale = Δf / ε"""
    assert abs(laplace_scale(1.0, 2.0) - 0.5) < 1e-10


def test_laplace_scale_smaller_epsilon_larger_scale():
    assert laplace_scale(1.0, 0.1) > laplace_scale(1.0, 1.0)


def test_laplace_std_formula():
    """σ = scale × √2"""
    s = laplace_scale(1.0, 1.0)
    assert abs(laplace_std(1.0, 1.0) - s * math.sqrt(2)) < 1e-10


def test_laplace_expected_abs_error_equals_scale():
    """E[|noise|] = scale for Laplace distribution."""
    scale = laplace_scale(1.0, 2.0)
    assert abs(laplace_expected_abs_error(1.0, 2.0) - scale) < 1e-10


def test_laplace_ci_95_is_two_scales():
    """95% of Laplace mass is within ±2×scale."""
    assert abs(laplace_ci_95(1.0, 1.0) - 2.0) < 1e-10


def test_privacy_loss_ratio_formula():
    """Belief ratio = e^ε"""
    assert abs(privacy_loss_ratio(1.0) - math.e) < 1e-10
    assert abs(privacy_loss_ratio(0.0) - 1.0) < 1e-10


def test_privacy_loss_ratio_increases_with_epsilon():
    assert privacy_loss_ratio(2.0) > privacy_loss_ratio(1.0)


def test_compose_epsilon_sum():
    """k queries × ε each = k×ε total."""
    assert abs(compose_epsilon([1.0, 1.0, 1.0]) - 3.0) < 1e-10
    assert abs(compose_epsilon([0.1, 0.2, 0.5]) - 0.8) < 1e-10


# ── Laplace mechanism via OpenDP ───────────────────────────────────────────

def test_laplace_mechanism_returns_query_result():
    r = laplace_mechanism(100.0, 1.0, 1.0, "count_test")
    assert isinstance(r, QueryResult)


def test_laplace_mechanism_fields_populated():
    r = laplace_mechanism(50.0, 1.0, 2.0)
    assert r.epsilon == 2.0
    assert r.sensitivity == 1.0
    assert abs(r.scale - 0.5) < 1e-10
    assert r.true_value == 50.0
    assert r.noise_added == pytest.approx(r.noisy_value - r.true_value, abs=1e-10)


def test_laplace_mechanism_expected_abs_error_matches_scale():
    r = laplace_mechanism(0.0, 1.0, 1.0)
    assert abs(r.expected_abs_error - r.scale) < 1e-10


def test_laplace_mechanism_ci_is_two_scales():
    r = laplace_mechanism(0.0, 1.0, 1.0)
    assert abs(r.ci_95_half_width - 2.0 * r.scale) < 1e-10


def test_laplace_mechanism_relative_error_none_for_zero():
    r = laplace_mechanism(0.0, 1.0, 1.0)
    assert r.relative_error_pct is None


def test_laplace_mechanism_relative_error_computed():
    r = laplace_mechanism(100.0, 1.0, 1.0)
    # scale=1, true=100 → relative = 1%
    assert abs(r.relative_error_pct - 1.0) < 1e-6


def test_laplace_noise_nonzero_almost_always():
    """Probability of exact zero noise from Laplace is measure zero."""
    results = [laplace_mechanism(10.0, 1.0, 1.0) for _ in range(20)]
    noises = [r.noise_added for r in results]
    assert any(n != 0.0 for n in noises)


def test_laplace_noise_mean_near_zero():
    """E[Laplace(0, b)] = 0. Mean of 5000 samples should be close."""
    noises = [laplace_mechanism(0.0, 1.0, 1.0).noise_added for _ in range(5000)]
    mean = statistics.mean(noises)
    assert abs(mean) < 0.15, f"Noise mean {mean:.4f} too far from 0"


def test_laplace_noise_abs_mean_near_scale():
    """E[|Laplace(0, b)|] = b. Using 5000 samples."""
    scale = 2.0  # ε=0.5, Δf=1
    noises = [laplace_mechanism(0.0, 1.0, 0.5).noise_added for _ in range(5000)]
    abs_mean = statistics.mean(abs(n) for n in noises)
    assert abs(abs_mean - scale) < 0.4, f"Abs mean {abs_mean:.3f} too far from scale {scale}"


def test_laplace_noise_larger_at_smaller_epsilon():
    """Smaller ε → more noise. Compare abs noise averages."""
    noises_low = [abs(laplace_mechanism(0.0, 1.0, 0.1).noise_added) for _ in range(500)]
    noises_high = [abs(laplace_mechanism(0.0, 1.0, 5.0).noise_added) for _ in range(500)]
    assert statistics.mean(noises_low) > statistics.mean(noises_high)


# ── run_all_queries ────────────────────────────────────────────────────────

def test_run_all_queries_returns_three_results():
    results = run_all_queries(240.0, 38.6, 0.28, 1000, 17, 90, 1.0)
    assert set(results.keys()) == {"count_high_income", "mean_age", "proportion_college"}


def test_run_all_queries_correct_epsilon():
    results = run_all_queries(240.0, 38.6, 0.28, 1000, 17, 90, 1.0)
    for r in results.values():
        assert r.epsilon == 1.0


def test_run_all_queries_correct_true_values():
    results = run_all_queries(240.0, 38.6, 0.28, 1000, 17, 90, 1.0)
    assert results["count_high_income"].true_value == 240.0
    assert results["mean_age"].true_value == 38.6
    assert results["proportion_college"].true_value == 0.28
