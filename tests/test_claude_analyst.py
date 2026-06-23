"""Tests for claude_analyst.py — DEMO_MODE explanations."""

import pytest
from claude_analyst import get_all_explanations, get_explanation
from config import EPSILON_PRESETS


def test_demo_explanations_exist_for_all_presets():
    explanations = get_all_explanations()
    for eps in EPSILON_PRESETS:
        assert eps in explanations, f"No explanation for ε={eps}"


def test_demo_explanations_nonempty():
    for eps in EPSILON_PRESETS:
        text = get_explanation(eps, demo_mode=True)
        assert len(text) > 100, f"Explanation for ε={eps} too short"


def test_demo_explanation_for_exact_preset():
    text = get_explanation(1.0, demo_mode=True)
    assert "1.0" in text or "e^1" in text or "2.72" in text


def test_demo_explanation_fallback_to_nearest():
    # ε=3.7 is not a preset; should fall back to nearest
    text = get_explanation(3.7, demo_mode=True)
    assert len(text) > 50


def test_demo_explanations_cover_strong_privacy():
    text = get_explanation(0.01, demo_mode=True)
    assert any(word in text.lower() for word in ["strong", "private", "privacy", "large", "noise"])


def test_demo_explanations_cover_weak_privacy():
    text = get_explanation(10.0, demo_mode=True)
    assert any(word in text.lower() for word in ["weak", "negligible", "no protection", "22,026", "little"])


def test_get_all_explanations_is_dict():
    result = get_all_explanations()
    assert isinstance(result, dict)
    assert len(result) == len(EPSILON_PRESETS)
