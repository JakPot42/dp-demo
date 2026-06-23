"""
Claude explanations of the privacy/accuracy tradeoff at each epsilon.

DEMO_MODE=True: pre-baked explanations cover all 7 standard epsilon presets.
DEMO_MODE=False: calls Claude Haiku with the full query context.
"""

import os
from typing import Optional

import anthropic

from config import DEMO_MODE
from dp_engine import QueryResult

# Pre-baked plain-language explanations for standard epsilon values
_DEMO_EXPLANATIONS: dict[float, str] = {
    0.01: (
        "ε = 0.01 is one of the strongest privacy budgets used in practice — "
        "essentially the same privacy Apple and Google use for keyboard analytics on your phone. "
        "The noise added is so large (scale = Δf × 100) that published statistics are almost "
        "completely uninformative for small datasets. Any single query result could plausibly "
        "be off by orders of magnitude. You are spending only 0.01 'privacy dollars' per query, "
        "so your budget lasts a long time — but analysts get almost nothing useful in return. "
        "This epsilon is appropriate when the underlying data is extremely sensitive (medical records, "
        "location history) and approximate answers are acceptable."
    ),
    0.1: (
        "ε = 0.1 provides strong privacy protection — the kind researchers use when demonstrating "
        "meaningful DP in published papers. Noise scale is Δf × 10, meaning count queries could be "
        "off by ±10–30 people and mean queries by noticeable amounts. The 'Alice attack' is "
        "essentially defeated: an adversary's belief about any individual can shift by at most "
        "e^0.1 ≈ 1.1× per query — barely more than random guessing. The tradeoff: queries on "
        "a 1,000-person dataset might be inaccurate by 5–15%, which is often too imprecise for "
        "policy decisions. Analysts typically need 10,000+ records to get useful answers at this epsilon."
    ),
    0.5: (
        "ε = 0.5 sits in the 'academically reasonable' zone — strong enough for most research "
        "publications, not so strong that results become useless. An adversary observing a single "
        "noisy statistic can update their belief by at most e^0.5 ≈ 1.65×. On 1,000 records, "
        "count queries have a Laplace scale of 2, meaning 95% of results land within ±4 of the "
        "truth. Mean-age queries are very accurate (scale ≈ 0.146 years). The main tradeoff: "
        "proportion queries (fraction with college degree) have scale = 0.002 — also accurate. "
        "This epsilon gives a reasonable balance for aggregate statistics on medium-sized datasets."
    ),
    1.0: (
        "ε = 1.0 is often called the 'DP threshold' — the boundary where practitioners debate "
        "whether privacy protection is still meaningful. An adversary can update their belief "
        "by at most e^1 ≈ 2.72× per query. For count queries on 1,000 people, scale = 1: "
        "95% of answers are within ±2 of the truth, and relative error is about 0.4%. "
        "Mean-age queries (sensitivity = 0.073) are extremely accurate: expected error < 0.1 years. "
        "This is the standard epsilon for the US Census Bureau's 2020 Decennial Census disclosure "
        "avoidance system — a real-world deployment choice that generated significant academic debate."
    ),
    2.0: (
        "ε = 2.0 is considered weak privacy by academic standards but is sometimes used "
        "in practice when data utility is paramount. Adversaries can update beliefs by up to "
        "e^2 ≈ 7.4× per query — meaningful information gain per observation. For count queries, "
        "scale = 0.5: results are typically off by less than 1 person, making individual-level "
        "inference more feasible with repeated queries. Mean-age sensitivity (0.037 years expected "
        "error) and proportion sensitivity (0.0005) are negligible at this budget. "
        "The composition risk is significant: 10 queries at ε=2 spend ε=20 total, which provides "
        "essentially no protection."
    ),
    5.0: (
        "ε = 5.0 provides very weak privacy. The likelihood ratio bound e^5 ≈ 148 means an "
        "adversary observing a single noisy statistic could multiply their belief about any "
        "individual by up to 148×. Count queries (scale = 0.2) are off by less than ±0.5 people "
        "95% of the time — nearly exact. An attacker computing 'mean × n − sum_of_others' to "
        "reconstruct Alice's age gets only ±14.6 years of uncertainty on age data — not enough "
        "protection for a 73-year range. This epsilon is typically only defensible when the "
        "dataset is very large (millions of records) or the data is only moderately sensitive."
    ),
    10.0: (
        "ε = 10.0 is considered negligible privacy protection in most academic contexts. "
        "The likelihood ratio bound e^10 ≈ 22,026 means that observing the noisy output can "
        "shift an adversary's beliefs by more than 22,000×. Count query noise (scale = 0.1) is "
        "so small that results are nearly exact — 95% within ±0.2 of the truth. For mean-age "
        "queries, expected error is only 0.0073 years (less than 3 days). An attacker performing "
        "the 'Alice reconstruction' gets ±7.3 years of uncertainty from the mean-age query alone — "
        "which, combined with other side information (employer records, public profiles), leaves "
        "Alice's privacy effectively unprotected. This epsilon is shown here as a reference point "
        "to illustrate why the choice of epsilon matters enormously."
    ),
}


def get_explanation(
    epsilon: float,
    query_results: Optional[dict[str, QueryResult]] = None,
    demo_mode: Optional[bool] = None,
) -> str:
    """
    Return a plain-language explanation of the privacy/accuracy tradeoff at epsilon.

    Args:
        epsilon: the privacy budget being explained
        query_results: optional actual query outputs for contextual explanation
        demo_mode: override config.DEMO_MODE for testing
    """
    use_demo = demo_mode if demo_mode is not None else DEMO_MODE

    if use_demo:
        # Find the closest preset explanation
        if epsilon in _DEMO_EXPLANATIONS:
            return _DEMO_EXPLANATIONS[epsilon]
        # Nearest preset
        nearest = min(_DEMO_EXPLANATIONS.keys(), key=lambda k: abs(k - epsilon))
        return (
            f"[ε={epsilon} — showing explanation for nearest preset ε={nearest}]\n"
            + _DEMO_EXPLANATIONS[nearest]
        )

    # Live Claude call
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "[ANTHROPIC_API_KEY not set — run with DEMO_MODE=True or set the key]"

    client = anthropic.Anthropic(api_key=api_key)

    context_lines = []
    if query_results:
        for qid, r in query_results.items():
            context_lines.append(
                f"  - {qid}: true={r.true_value:.4f}, noisy={r.noisy_value:.4f}, "
                f"noise_added={r.noise_added:+.4f}, expected_abs_error={r.expected_abs_error:.4f}"
            )
    context_block = "\n".join(context_lines) if context_lines else "  (no query results provided)"

    prompt = f"""You are explaining differential privacy to a non-technical policy analyst.

Current epsilon (privacy budget): ε = {epsilon}

Query results at this epsilon (true vs noisy values on 1,000 Census records):
{context_block}

Key math facts:
- Laplace noise scale = Δf / ε. Smaller ε → larger scale → more noise.
- Privacy guarantee: an adversary observing the noisy output can update their
  belief about any individual by at most e^ε = {2.71828**epsilon:.2f}× per query.
- Composition: k queries each at ε spend k×ε total privacy budget.

Write 4–6 sentences in plain English explaining:
1. What this epsilon level means intuitively (is this strong or weak protection?)
2. How much noise is being added and whether the published statistics are still useful
3. A concrete example of what an attacker CAN and CANNOT learn at this epsilon
4. The key tradeoff a data curator faces at this epsilon

Avoid jargon. Be direct and honest about when protection is weak."""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:
        return f"[Claude API error: {exc}]"


def get_all_explanations(demo_mode: Optional[bool] = None) -> dict[float, str]:
    """Return pre-baked explanations for all standard epsilon presets."""
    return dict(_DEMO_EXPLANATIONS)
