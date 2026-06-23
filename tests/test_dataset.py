"""Tests for dataset.py — determinism, bounds, and true query answers."""

import pytest
from dataset import CensusDataset, generate_dataset
from config import DATASET


def test_generate_deterministic():
    """Same seed → same data every time."""
    r1 = generate_dataset(1000)
    r2 = generate_dataset(1000)
    assert r1 == r2


def test_record_count(records):
    assert len(records) == DATASET["n"]


def test_age_within_bounds(records):
    lo, hi = DATASET["age_bounds"]
    for r in records:
        assert lo <= r["age"] <= hi, f"age {r['age']} out of [{lo}, {hi}]"


def test_education_years_in_range(records):
    for r in records:
        assert 1 <= r["education_years"] <= 16


def test_income_binary(records):
    for r in records:
        assert r["income_above_50k"] in (0, 1)


def test_hours_per_week_in_range(records):
    for r in records:
        assert 1 <= r["hours_per_week"] <= 99


def test_high_income_prevalence_approx_24pct(records):
    frac = sum(r["income_above_50k"] for r in records) / len(records)
    # Should be around 24% ± 5%
    assert 0.18 <= frac <= 0.30, f"High-income fraction {frac:.2%} outside [18%, 30%]"


def test_mean_age_approx(records):
    mean_age = sum(r["age"] for r in records) / len(records)
    assert 34 <= mean_age <= 44, f"Mean age {mean_age:.2f} not in expected range"


def test_census_dataset_loads(records):
    ds = CensusDataset(records)
    assert ds.n == len(records)


def test_true_count_high_income(ds, records):
    expected = sum(r["income_above_50k"] for r in records)
    assert ds.true_count_high_income() == expected


def test_true_mean_age(ds, records):
    expected = sum(r["age"] for r in records) / len(records)
    assert abs(ds.true_mean_age() - expected) < 1e-10


def test_true_proportion_college(ds, records):
    threshold = DATASET["college_edu_years"]
    expected = sum(1 for r in records if r["education_years"] >= threshold) / len(records)
    assert abs(ds.true_proportion_college() - expected) < 1e-10


def test_summary_keys(summary):
    assert "n" in summary
    assert "true_count_high_income" in summary
    assert "true_mean_age" in summary
    assert "true_proportion_college" in summary
    assert "age_bounds" in summary


def test_age_bounds_are_within_dataset_bounds(ds):
    lo, hi = ds.age_bounds()
    config_lo, config_hi = DATASET["age_bounds"]
    assert lo >= config_lo
    assert hi <= config_hi


def test_small_dataset_works():
    ds = CensusDataset(generate_dataset(10))
    assert ds.n == 10
    assert 0 <= ds.true_proportion_college() <= 1.0
