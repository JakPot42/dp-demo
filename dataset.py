"""
Census income sample dataset.

Generates a 1000-record dataset with statistical properties matching the
1994 US Census Bureau data (UCI ML Repository Adult dataset):
  - Mean age ≈ 38.6, std ≈ 13.6, clamped to [17, 90]
  - ≈24.1% of workers earn >$50K/year
  - Education years 1-16, roughly matching Census education distribution

All records are deterministically generated from seed=42 — same output
every run, no external download required.
"""

import csv
import math
from pathlib import Path

import numpy as np

from config import DATASET

DATA_FILE = Path(__file__).parent / "data" / "census_sample.csv"
SEED = 42


def _education_probabilities() -> list[float]:
    # Rough weights matching UCI Adult edu_num distribution (1–16 years)
    weights = [0.005, 0.01, 0.015, 0.03, 0.08, 0.095, 0.12, 0.065,
               0.21, 0.03, 0.115, 0.01, 0.075, 0.015, 0.01, 0.015]
    total = sum(weights)
    return [w / total for w in weights]


def generate_dataset(n: int = 1000) -> list[dict]:
    rng = np.random.default_rng(SEED)

    # Age: truncated normal matching UCI Adult mean/std
    ages_raw = rng.normal(38.6, 13.6, n)
    ages = np.clip(ages_raw, 17, 90).round().astype(int)

    # Education years 1–16
    edu_probs = _education_probabilities()
    edu_years = rng.choice(range(1, 17), size=n, p=edu_probs)

    # High income (>$50K): ~24.1% prevalence matching UCI Adult
    high_income = rng.binomial(1, 0.241, n)

    # Hours per week: roughly normal around 40
    hours_raw = rng.normal(40.4, 12.3, n)
    hours = np.clip(hours_raw, 1, 99).round().astype(int)

    return [
        {
            "age": int(ages[i]),
            "education_years": int(edu_years[i]),
            "income_above_50k": int(high_income[i]),
            "hours_per_week": int(hours[i]),
        }
        for i in range(n)
    ]


def save_dataset(records: list[dict]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def load_dataset() -> list[dict]:
    if not DATA_FILE.exists():
        records = generate_dataset(DATASET["n"])
        save_dataset(records)
        return records
    with open(DATA_FILE, newline="") as f:
        reader = csv.DictReader(f)
        return [
            {
                "age": int(r["age"]),
                "education_years": int(r["education_years"]),
                "income_above_50k": int(r["income_above_50k"]),
                "hours_per_week": int(r["hours_per_week"]),
            }
            for r in reader
        ]


class CensusDataset:
    def __init__(self, records: list[dict] | None = None):
        self.records = records if records is not None else load_dataset()
        self.n = len(self.records)

    # ── True (non-private) query answers ──────────────────────────────────

    def true_count_high_income(self) -> int:
        return sum(1 for r in self.records if r["income_above_50k"] == 1)

    def true_mean_age(self) -> float:
        return sum(r["age"] for r in self.records) / self.n

    def true_proportion_college(self) -> float:
        college_threshold = DATASET["college_edu_years"]
        return sum(1 for r in self.records if r["education_years"] >= college_threshold) / self.n

    def age_bounds(self) -> tuple[int, int]:
        ages = [r["age"] for r in self.records]
        return min(ages), max(ages)

    def summary(self) -> dict:
        return {
            "n": self.n,
            "true_count_high_income": self.true_count_high_income(),
            "true_mean_age": round(self.true_mean_age(), 2),
            "true_proportion_college": round(self.true_proportion_college(), 4),
            "age_bounds": self.age_bounds(),
        }
