# AB Testing

> A randomized controlled experiment that compares two (or more) variants of a product/feature by randomly assigning users to each and measuring the effect on a target metric.

| | |
|---|---|
| **Category** | Data Analysis & Statistics |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [Hypothesis Testing](../hypothesis-testing/), [Inferential Statistics](../inferential-statistics/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
You want to know if a new checkout button increases purchases, but you can't run the universe twice. So you split it in two: randomly show half of users the old button (**control**, A) and half the new one (**treatment**, B), then compare outcomes. Randomization is what lets you claim the *difference* was caused by the button, not by who happened to see which version — it's hypothesis testing applied specifically to product/business decisions.

## 2. Formal definition / Key concepts
- **Control (A)** and **treatment (B)** groups, randomly assigned to remove confounding.
- **Primary metric** — the single outcome the test is powered/designed to move (e.g. conversion rate). **Guardrail metrics** — outcomes that must not regress (e.g. latency, unsubscribe rate).
- **Minimum Detectable Effect (MDE)** — the smallest effect size worth detecting; drives the required sample size.
- **Statistical Power** — probability of detecting a true effect of the MDE, typically targeted at 80–90%.
- **Randomization unit** — user, session, or device; must match the unit of analysis to avoid dependence between "independent" observations.
- **Novelty effect / network effects** — biases specific to running experiments on live, interacting user populations.

## 3. Math
Sample size per group for a two-proportion test (approx.):
$$n = \frac{2\,(z_{1-\alpha/2} + z_{1-\beta})^2 \; \bar p (1-\bar p)}{(p_B - p_A)^2}$$
where $\bar p = (p_A+p_B)/2$.

Two-proportion $z$-test statistic:
$$z = \frac{\hat p_B - \hat p_A}{\sqrt{\bar p (1-\bar p)\left(\frac{1}{n_A}+\frac{1}{n_B}\right)}}$$

## 4. How it works
1. **Define the hypothesis and metrics** — primary metric, guardrails, MDE, and $\alpha$/power *before* launch.
2. **Compute required sample size** and expected test duration given traffic.
3. **Randomize** users into A/B (consistently, e.g. via hashing user ID) and instrument logging.
4. **Run for the pre-planned duration** — don't stop early just because it looks significant (peeking problem).
5. **Analyze** with the pre-specified test (e.g. two-proportion $z$-test, $t$-test for continuous metrics), check guardrails, and check for sample ratio mismatch (SRM).
6. **Decide and ship (or don't)** based on the primary metric, guardrails, and practical significance — not just $p < 0.05$.

## 5. When to use / When not to
- ✅ When you can randomize at the right unit and get enough traffic to reach the required sample size in reasonable time.
- ✅ Product/UX changes, pricing, ranking algorithm tweaks, email copy.
- ❌ Low-traffic contexts where the test would take months to reach power — consider a different design (e.g. before/after with causal inference, or a switchback test).
- ❌ When treatment can leak to control (e.g. social/marketplace network effects, shared inventory) — violates the independence assumption (SUTVA); consider cluster/switchback randomization instead.
- ❌ Irreversible or high-risk changes without a way to roll back on guardrail regression.

## 6. Common pitfalls & gotchas
- **Peeking / early stopping** — checking significance repeatedly and stopping as soon as $p<0.05$ massively inflates the false-positive rate; use sequential testing methods if you need to peek.
- **Sample Ratio Mismatch (SRM)** — if the observed A/B split deviates from the intended ratio (e.g. 48/52 instead of 50/50), something is broken in randomization/logging and results are untrustworthy — check with a chi-squared test before trusting the metric.
- **Network interference (SUTVA violation)** — in a marketplace/social product, treating one user can affect control users too (e.g. shared inventory, friend interactions).
- **Novelty/primacy effects** — a new UI can spike engagement short-term simply because it's new, and settle differently over weeks.
- **Multiple metrics / multiple testing** — checking dozens of secondary metrics and reporting whichever moved is p-hacking; pre-register the primary metric.
- **Simpson's paradox** — an effect can reverse when segments are aggregated vs analyzed separately; check for heterogeneous effects across key segments.

## 7. Code
```python
from scipy import stats
import numpy as np

# Two-proportion z-test for conversion rate
def ab_test(conv_a, n_a, conv_b, n_b):
    p_a, p_b = conv_a / n_a, conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value

# Sample Ratio Mismatch check
chi2, srm_p, _, _ = stats.chi2_contingency([[n_a, n_b], [n_a + n_b, n_a + n_b]])
```

## 8. Interview / viva questions
- Q: Why is peeking at A/B test results daily and stopping early a problem?
  - A: Each peek is an additional chance to hit $p<0.05$ by chance; repeated peeking inflates the true false-positive rate well above the nominal α. Either commit to a fixed sample size/duration or use a sequential testing method (e.g. always-valid p-values, SPRT) designed for continuous monitoring.
- Q: Your test shows p=0.01 in favor of the treatment, but the split was 45/55 instead of 50/50. Do you trust it?
  - A: No — that's a Sample Ratio Mismatch, a red flag that randomization or logging is broken (e.g. bot filtering, redirect bugs). Investigate and fix the root cause before trusting any metric from the experiment.
- Q: How do you A/B test a feature on a social network where users interact with each other?
  - A: Standard user-level randomization violates SUTVA because treated and control users interact. Use cluster-based or graph-cluster randomization (e.g. randomize by friend group/geography) or a switchback design instead.

## 9. References
- Kohavi, R., Tang, D., Xu, Y. — *Trustworthy Online Controlled Experiments*.
- Deng, A. et al. — "Continuous Monitoring of A/B Tests without Pain: Optional Stopping in Bayesian Testing" (2016).
- Kohavi, R. et al. — "Trustworthy Online Controlled Experiments: Five Puzzling Outcomes Explained" (2012) (SRM).

---
> _Status: 🟢 done._
