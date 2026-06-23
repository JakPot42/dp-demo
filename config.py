"""
Configuration for the Differential Privacy Demonstration Tool.

Terminology:
    epsilon (ε):    privacy budget — lower = stronger privacy, more noise
    sensitivity:    max change in query result from adding/removing one person
    scale = Δf/ε:  Laplace distribution scale parameter
"""

DEMO_MODE = True

EPSILON_PRESETS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

DATASET = {
    "name": "US Census Income Sample",
    "source": "Derived from 1994 US Census Bureau data (UCI ML Repository Adult dataset)",
    "n": 1000,
    "age_bounds": (17, 90),
    "high_income_threshold": 50_000,
    "college_edu_years": 13,  # ≥13 years = bachelor's degree or higher
}

QUERIES = [
    {
        "id": "count_high_income",
        "name": "Count: High-Income Workers",
        "description": "How many people in this sample earn more than $50,000/year?",
        "type": "count",
        "unit": "people",
        "sensitivity_formula": "Δf = 1  (adding one person changes count by at most 1)",
        "sensitivity": 1.0,
    },
    {
        "id": "mean_age",
        "name": "Mean: Average Worker Age",
        "description": "What is the average age of workers in this sample?",
        "type": "mean",
        "unit": "years",
        "sensitivity_formula": "Δf = (max_age − min_age) / n = (90 − 17) / 1000 = 0.073",
        "sensitivity": (90 - 17) / DATASET["n"],
    },
    {
        "id": "proportion_college",
        "name": "Proportion: College-Educated Workers",
        "description": "What fraction of workers have a college degree or higher?",
        "type": "proportion",
        "unit": "fraction (0–1)",
        "sensitivity_formula": "Δf = 1 / n = 1 / 1000 = 0.001",
        "sensitivity": 1.0 / DATASET["n"],
    },
]

QUERY_INDEX = {q["id"]: q for q in QUERIES}
