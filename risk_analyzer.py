"""
Re-identification risk analysis for each query type and epsilon.

The ε-DP guarantee (Dwork et al., 2006):
    For neighboring datasets D, D' differing in one person's record,
    and ANY adversary with ANY side information:

        P[M(D) ∈ S] ≤ e^ε × P[M(D') ∈ S]   for all output sets S

This bounds the adversary's LIKELIHOOD RATIO — how much their belief
about any one individual can shift after observing the noisy statistic.
"""

import math
from dataclasses import dataclass


@dataclass
class RiskProfile:
    epsilon: float
    protection_level: str       # STRONG / MODERATE / WEAK / NEGLIGIBLE
    belief_ratio: float         # e^ε — max multiplicative belief update
    alice_scenario: str         # concrete re-identification walkthrough
    what_noise_prevents: str    # one-sentence layman summary


def _protection_level(epsilon: float) -> str:
    if epsilon <= 0.5:
        return "STRONG"
    elif epsilon <= 1.5:
        return "MODERATE"
    elif epsilon <= 5.0:
        return "WEAK"
    return "NEGLIGIBLE"


# ── Per-query risk profiles ────────────────────────────────────────────────

def count_risk_profile(
    epsilon: float,
    true_count: int,
    n: int,
) -> RiskProfile:
    """
    Concrete scenario: John is a high-earner in the dataset.
    The attacker knows everyone else's income status (n−1 people).
    They observe the noisy count.
    """
    scale = 1.0 / epsilon
    ratio = math.exp(epsilon)
    signal_noise_ratio = 1.0 / scale  # signal=1 count, noise=scale

    if scale > 5:
        prevents = (
            "The noise swamps the 1-count signal entirely — the attacker "
            "cannot tell whether John is included or excluded."
        )
        alice = (
            f"John is one of the {true_count} high-earners in this {n}-person dataset. "
            f"The attacker knows everyone else's income status and waits for the "
            f"published count. With ε={epsilon}, the Laplace scale is {scale:.2f}, "
            f"so the observed count could plausibly be anywhere from "
            f"{true_count - 3*scale:.0f} to {true_count + 3*scale:.0f}. "
            f"That 6×{scale:.2f} = {6*scale:.1f}-person uncertainty window makes it "
            f"impossible to confirm John's presence."
        )
    elif scale > 0.3:
        prevents = (
            f"The noise (scale={scale:.2f}) is comparable to the 1-count signal, "
            "giving the attacker limited information about John's presence."
        )
        alice = (
            f"John is a high-earner in this {n}-person sample. The attacker knows "
            f"everyone else. With ε={epsilon}, noise scale={scale:.2f}: the published "
            f"count lands in [{true_count - scale:.1f}, {true_count + scale:.1f}] 63% "
            f"of the time whether John is included OR excluded. The attacker's belief "
            f"update is bounded to {ratio:.1f}×."
        )
    else:
        prevents = (
            f"The noise (scale={scale:.3f}) is tiny relative to the 1-count signal. "
            f"The attacker can detect John's presence with confidence ≤ e^ε = {ratio:.0f}×."
        )
        alice = (
            f"With ε={epsilon}, the Laplace scale is only {scale:.4f} — nearly zero "
            f"noise on a count query. The attacker observing the published count "
            f"can distinguish 'John included' vs 'John excluded' with high confidence. "
            f"This epsilon provides essentially no count-query privacy."
        )

    return RiskProfile(
        epsilon=epsilon,
        protection_level=_protection_level(epsilon),
        belief_ratio=ratio,
        alice_scenario=alice,
        what_noise_prevents=prevents,
    )


def mean_age_risk_profile(
    epsilon: float,
    true_mean: float,
    n: int,
    age_min: int,
    age_max: int,
) -> RiskProfile:
    """
    Concrete scenario: Alice's age is in the dataset.
    The attacker knows all other n−1 ages exactly.
    Without DP: attacker computes alice_age = mean × n − sum_of_others.
    With DP: the noise propagates to ±n × scale uncertainty in Alice's age.
    """
    sensitivity = (age_max - age_min) / n
    scale = sensitivity / epsilon
    ratio = math.exp(epsilon)
    alice_uncertainty = n * scale  # noise in reconstructed Alice's age

    alice = (
        f"Alice's age is in this {n}-person dataset. An attacker knows the exact "
        f"ages of all {n-1} other people. Without DP, they could compute: "
        f"alice_age = published_mean × {n} − sum_of_others, perfectly revealing Alice. "
        f"\n\nWith ε={epsilon}: the published mean has Laplace noise (scale={scale:.5f}). "
        f"When the attacker reverses the mean formula, they get: "
        f"alice_age + {n} × Laplace(0, {scale:.5f}) = alice_age ± ~{alice_uncertainty:.1f} years of uncertainty. "
        f"{'That far exceeds the entire lifespan — Alice is completely protected.' if alice_uncertainty > 73 else f'That is {alice_uncertainty:.1f} years of uncertainty in a [17,90] range.' if alice_uncertainty > 5 else f'That is only {alice_uncertainty:.2f} years — Alice is weakly protected at this epsilon.'}"
    )

    if alice_uncertainty > 73:
        prevents = (
            f"Noise propagation makes Alice's reconstructed age uncertain by "
            f"±{alice_uncertainty:.0f} years — far exceeding the realistic age range."
        )
    elif alice_uncertainty > 10:
        prevents = (
            f"Alice's reconstructed age has ±{alice_uncertainty:.1f} years of uncertainty, "
            f"preventing precise re-identification."
        )
    else:
        prevents = (
            f"Only ±{alice_uncertainty:.2f} years of uncertainty in Alice's reconstructed age — "
            f"attackers with contextual knowledge could narrow this further."
        )

    return RiskProfile(
        epsilon=epsilon,
        protection_level=_protection_level(epsilon),
        belief_ratio=ratio,
        alice_scenario=alice,
        what_noise_prevents=prevents,
    )


def proportion_risk_profile(
    epsilon: float,
    true_proportion: float,
    n: int,
) -> RiskProfile:
    """
    Concrete scenario: Bob has a college degree. Attacker knows n−1 others' education.
    Without DP: proportion × n − known_count reveals Bob's status exactly.
    With DP: noise makes the proportion uncertain.
    """
    sensitivity = 1.0 / n
    scale = sensitivity / epsilon
    ratio = math.exp(epsilon)
    count_uncertainty = n * scale  # = 1/ε (uncertainty in the reconstructed count)

    alice = (
        f"Bob has a college degree (one of ~{true_proportion*n:.0f} in the dataset). "
        f"An attacker knows the education status of all {n-1} others. Without DP, "
        f"they compute bob_status = round(proportion × {n}) − known_count.\n\n"
        f"With ε={epsilon}: the Laplace scale on the proportion is {scale:.6f}. "
        f"When reversed to a count, uncertainty = {n} × {scale:.6f} = {count_uncertainty:.2f} people. "
        f"{'That is enough to hide Bob entirely.' if count_uncertainty > 1 else f'That is less than 1 person — Bob can likely be identified.'}"
    )

    if count_uncertainty >= 1.0:
        prevents = (
            f"Proportion noise creates ±{count_uncertainty:.2f} people of uncertainty "
            "in the count — enough to conceal any one individual's status."
        )
    else:
        prevents = (
            f"Only ±{count_uncertainty:.3f} person of uncertainty — attackers can "
            "likely recover exact counts and identify individuals."
        )

    return RiskProfile(
        epsilon=epsilon,
        protection_level=_protection_level(epsilon),
        belief_ratio=ratio,
        alice_scenario=alice,
        what_noise_prevents=prevents,
    )


def all_risk_profiles(
    epsilon: float,
    true_count: int,
    true_mean: float,
    true_proportion: float,
    n: int,
    age_min: int,
    age_max: int,
) -> dict[str, RiskProfile]:
    return {
        "count_high_income": count_risk_profile(epsilon, true_count, n),
        "mean_age": mean_age_risk_profile(epsilon, true_mean, n, age_min, age_max),
        "proportion_college": proportion_risk_profile(epsilon, true_proportion, n),
    }
