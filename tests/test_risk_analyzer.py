"""Tests for risk_analyzer.py — protection levels, scenarios, math."""

import math
import pytest

from risk_analyzer import (
    RiskProfile,
    _protection_level,
    all_risk_profiles,
    count_risk_profile,
    mean_age_risk_profile,
    proportion_risk_profile,
)


# ── Protection level thresholds ────────────────────────────────────────────

def test_strong_at_low_epsilon():
    assert _protection_level(0.01) == "STRONG"
    assert _protection_level(0.5) == "STRONG"


def test_moderate_at_mid_epsilon():
    assert _protection_level(1.0) == "MODERATE"
    assert _protection_level(1.5) == "MODERATE"


def test_weak_at_high_epsilon():
    assert _protection_level(2.0) == "WEAK"
    assert _protection_level(5.0) == "WEAK"


def test_negligible_at_very_high_epsilon():
    assert _protection_level(6.0) == "NEGLIGIBLE"
    assert _protection_level(10.0) == "NEGLIGIBLE"


# ── Count risk profile ─────────────────────────────────────────────────────

def test_count_risk_profile_returns_risk_profile():
    rp = count_risk_profile(1.0, 240, 1000)
    assert isinstance(rp, RiskProfile)


def test_count_risk_profile_epsilon_stored():
    rp = count_risk_profile(2.0, 240, 1000)
    assert rp.epsilon == 2.0


def test_count_risk_profile_belief_ratio():
    rp = count_risk_profile(1.0, 240, 1000)
    assert abs(rp.belief_ratio - math.exp(1.0)) < 1e-10


def test_count_risk_profile_scenario_nonempty():
    rp = count_risk_profile(1.0, 240, 1000)
    assert len(rp.alice_scenario) > 50


def test_count_risk_profile_what_prevents_nonempty():
    rp = count_risk_profile(0.1, 240, 1000)
    assert len(rp.what_noise_prevents) > 20


def test_count_risk_profile_protection_level_low_epsilon():
    rp = count_risk_profile(0.01, 240, 1000)
    assert rp.protection_level == "STRONG"


def test_count_risk_profile_protection_level_high_epsilon():
    rp = count_risk_profile(10.0, 240, 1000)
    assert rp.protection_level == "NEGLIGIBLE"


# ── Mean-age risk profile ──────────────────────────────────────────────────

def test_mean_age_risk_profile_returns_risk_profile():
    rp = mean_age_risk_profile(1.0, 38.6, 1000, 17, 90)
    assert isinstance(rp, RiskProfile)


def test_mean_age_belief_ratio():
    rp = mean_age_risk_profile(0.5, 38.6, 1000, 17, 90)
    assert abs(rp.belief_ratio - math.exp(0.5)) < 1e-10


def test_mean_age_scenario_mentions_uncertainty():
    rp = mean_age_risk_profile(1.0, 38.6, 1000, 17, 90)
    assert "uncertainty" in rp.alice_scenario.lower() or "years" in rp.alice_scenario.lower()


def test_mean_age_low_epsilon_strong_protection():
    rp = mean_age_risk_profile(0.01, 38.6, 1000, 17, 90)
    assert rp.protection_level == "STRONG"
    # At ε=0.01, uncertainty = 1000 × (73/1000/0.01) = 7300 years — enormous
    assert "far exceeds" in rp.alice_scenario.lower() or "completely" in rp.alice_scenario.lower()


def test_mean_age_high_epsilon_weak_protection():
    rp = mean_age_risk_profile(10.0, 38.6, 1000, 17, 90)
    assert rp.protection_level == "NEGLIGIBLE"


# ── Proportion risk profile ────────────────────────────────────────────────

def test_proportion_risk_profile_returns_risk_profile():
    rp = proportion_risk_profile(1.0, 0.28, 1000)
    assert isinstance(rp, RiskProfile)


def test_proportion_belief_ratio():
    rp = proportion_risk_profile(2.0, 0.28, 1000)
    assert abs(rp.belief_ratio - math.exp(2.0)) < 1e-10


def test_proportion_low_epsilon_hides_individual():
    """At ε=1, scale=0.001, count_uncertainty = n × scale = 1.0 → hides Bob."""
    rp = proportion_risk_profile(1.0, 0.28, 1000)
    # scale = (1/1000)/1 = 0.001; count_uncertainty = 1000×0.001 = 1.0
    assert "1.00" in rp.alice_scenario or "hides" in rp.what_noise_prevents.lower() or "hide" in rp.what_noise_prevents.lower()


def test_proportion_high_epsilon_does_not_hide():
    """At ε=100, scale=0.00001, count_uncertainty=0.01 → Bob can be identified."""
    rp = proportion_risk_profile(100.0, 0.28, 1000)
    assert "identified" in rp.alice_scenario.lower() or "less than 1" in rp.alice_scenario.lower()


# ── all_risk_profiles ──────────────────────────────────────────────────────

def test_all_risk_profiles_returns_three():
    profiles = all_risk_profiles(1.0, 240, 38.6, 0.28, 1000, 17, 90)
    assert set(profiles.keys()) == {"count_high_income", "mean_age", "proportion_college"}


def test_all_risk_profiles_correct_epsilon():
    profiles = all_risk_profiles(2.0, 240, 38.6, 0.28, 1000, 17, 90)
    for rp in profiles.values():
        assert rp.epsilon == 2.0
