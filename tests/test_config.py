"""Tests for config.py — sanity checks on constants and structure."""

import math

from config import DATASET, EPSILON_PRESETS, QUERIES, QUERY_INDEX


def test_epsilon_presets_ordered():
    assert EPSILON_PRESETS == sorted(EPSILON_PRESETS)


def test_epsilon_presets_all_positive():
    for e in EPSILON_PRESETS:
        assert e > 0


def test_epsilon_presets_count():
    assert len(EPSILON_PRESETS) == 7


def test_dataset_n_positive():
    assert DATASET["n"] > 0


def test_dataset_age_bounds_valid():
    lo, hi = DATASET["age_bounds"]
    assert lo < hi
    assert lo > 0


def test_queries_have_required_fields():
    required = {"id", "name", "description", "type", "unit", "sensitivity", "sensitivity_formula"}
    for q in QUERIES:
        assert required.issubset(q.keys()), f"Query {q['id']} missing fields"


def test_query_sensitivities_positive():
    for q in QUERIES:
        assert q["sensitivity"] > 0


def test_count_sensitivity_is_one():
    count_q = QUERY_INDEX["count_high_income"]
    assert count_q["sensitivity"] == 1.0


def test_mean_sensitivity_formula():
    q = QUERY_INDEX["mean_age"]
    expected = (90 - 17) / DATASET["n"]
    assert abs(q["sensitivity"] - expected) < 1e-10


def test_proportion_sensitivity_formula():
    q = QUERY_INDEX["proportion_college"]
    expected = 1.0 / DATASET["n"]
    assert abs(q["sensitivity"] - expected) < 1e-10


def test_query_index_matches_queries():
    for q in QUERIES:
        assert q["id"] in QUERY_INDEX
        assert QUERY_INDEX[q["id"]] is q
