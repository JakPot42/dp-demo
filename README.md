# Differential Privacy Demonstration Tool (P48)

CLI tool showing the real privacy/accuracy tradeoff using the **Laplace mechanism** (OpenDP library) on a 1,000-record US Census income sample.

Most tools *cite* differential privacy. This one *demonstrates* it — with live noise, concrete re-identification scenarios, and plain-language Claude explanations at each epsilon value.

---

## What it demonstrates

Three queries on a 1,000-person Census sample, run at seven epsilon values:

| Query | Sensitivity Δf | What it measures |
|---|---|---|
| Count of high-earners (>$50K/yr) | Δf = 1 | Adding one person changes count by exactly 1 |
| Mean worker age | Δf = (90−17)/1000 = 0.073 | Adding one person changes mean by at most 0.073 |
| Proportion college-educated | Δf = 1/1000 = 0.001 | Adding one person changes proportion by at most 0.001 |

For each epsilon, the tool shows:
- **Noise magnitude**: Laplace scale = Δf/ε
- **Query accuracy**: expected absolute error, 95% confidence interval, relative error
- **Re-identification risk**: concrete Alice/John scenario showing exactly what the noise prevents
- **Claude's explanation**: plain-language tradeoff at that epsilon level

---

## The math (precisely stated)

**ε-Differential Privacy** (Dwork et al., 2006): A randomized mechanism M is ε-differentially private if for all neighboring datasets D, D' differing in one record, and all output sets S:

```
P[M(D) ∈ S] ≤ e^ε × P[M(D') ∈ S]
```

**Laplace mechanism**: `M(D) = f(D) + Lap(0, Δf/ε)`

- `Δf` = global sensitivity (max |f(D) − f(D')| over all neighboring pairs)
- Scale parameter `b = Δf/ε` controls noise magnitude
- `E[|noise|] = b`, `σ = b√2`, 95% of noise within `±2b`
- Privacy cost is `ε` per query; basic composition: k queries spend k×ε total

**Why OpenDP?** The `opendp` library (Harvard OpenDP Initiative) validates the ε-DP guarantee at construction time — this project does not implement the noise mechanism from scratch.

---

## Dataset

1,000 records generated deterministically (seed=42) with statistical properties matching the **1994 US Census Bureau data** (UCI ML Repository Adult dataset):
- Age: approximately N(38.6, 13.6²), clamped to [17, 90]
- High-income (>$50K): ~24% prevalence
- Education years: 1–16, matching published Census distribution

No external download required. Data is generated on first run and cached to `data/census_sample.csv`.

---

## Installation

```bash
pip install opendp numpy rich click anthropic python-dotenv
```

---

## Usage

```bash
# Full interactive demo — all epsilons, all queries, noise bars, Claude explanation
python main.py demo

# Deep-dive on one epsilon value
python main.py explore --epsilon 0.1
python main.py explore --epsilon 1.0
python main.py explore --epsilon 10.0

# Side-by-side accuracy table
python main.py compare

# Privacy budget composition under repeated queries
python main.py budget --epsilon 1.0 --queries 10
```

Windows (for Unicode output):
```powershell
$env:PYTHONUTF8="1"; py main.py demo
```

---

## Example output

```
Privacy / Accuracy Tradeoff Across ε Values
┌─────────┬────────────┬────────────┬──────────────┬────────────────┬──────────────┐
│ ε       │ e^ε ratio  │ Protection │ Count error  │ Mean age error │ Prop. error  │
├─────────┼────────────┼────────────┼──────────────┼────────────────┼──────────────┤
│ 0.01    │ 1.0×       │ STRONG     │ ±100.000     │ ±6.50000 yrs   │ ±0.100000    │
│ 0.1     │ 1.1×       │ STRONG     │ ±10.000      │ ±0.65000 yrs   │ ±0.010000    │
│ 1.0     │ 2.7×       │ MODERATE   │ ±1.000       │ ±0.06500 yrs   │ ±0.001000    │
│ 10.0    │ 22026.5×   │ NEGLIGIBLE │ ±0.100       │ ±0.00650 yrs   │ ±0.000100    │
└─────────┴────────────┴────────────┴──────────────┴────────────────┴──────────────┘
```

**The Alice reconstruction attack** (mean-age query):
An attacker who knows the exact ages of all 999 other people can compute `alice_age = published_mean × 1000 − sum_of_others`. At ε=0.1, Laplace noise with scale=0.65 propagates to ±650 years of uncertainty in Alice's reconstructed age — far exceeding any realistic age range. At ε=10, the uncertainty drops to ±7.3 years, leaving Alice weakly protected.

---

## DEMO_MODE

`DEMO_MODE = True` (default in `config.py`) uses pre-baked Claude explanations for all 7 standard epsilon presets. No API key required.

Set `DEMO_MODE = False` and export `ANTHROPIC_API_KEY` to get live Claude explanations that reference the actual query results for any epsilon value.

---

## Tests

```bash
pytest tests/ -q
# 96 tests, all passing
```

---

## Architecture

| File | Role |
|---|---|
| `config.py` | Epsilon presets, query definitions, sensitivity constants |
| `dataset.py` | Census sample generation (seed=42), true query answers |
| `dp_engine.py` | OpenDP Laplace mechanism, sensitivity helpers, noise statistics |
| `risk_analyzer.py` | Re-identification risk — Alice/John concrete scenarios |
| `claude_analyst.py` | Plain-language tradeoff explanations (DEMO_MODE or live) |
| `display.py` | Rich console tables, noise bars, panels |
| `main.py` | Click CLI: `demo`, `explore`, `compare`, `budget` |

**Key design decision:** The Laplace noise is added by OpenDP, not a manual `numpy.random.laplace` call. OpenDP validates the ε-DP guarantee at construction time — the guarantee is *proven*, not assumed.

---

## What this does NOT do

- Does not implement cryptography or break any real system
- Does not handle "advanced composition" (Rényi DP, zCDP) — basic composition only
- Does not implement the Gaussian mechanism (for (ε,δ)-DP)
- The dataset is a synthetic Census replica for demonstration purposes only

---

## Portfolio context

**P48** in a defense/national security portfolio. Pairs with:
- **P9 RAI Compliance** — DoD AI ethical principles assessment
- **P13 Carta** — platform accountability standard
- **P13 redteam-eval** — LLM red-teaming for compliance tools

Differential privacy is increasingly required in federal data programs (Census Bureau, VA health data, DoD analytics). Understanding the epsilon/accuracy tradeoff is a concrete technical skill, not just a buzzword.

---

GitHub: `JakPot42/dp-demo`
