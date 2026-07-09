# AB Testing — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Randomly split users into control/treatment, run for a pre-planned duration, compare the primary metric with a pre-specified test.

## Key formulas
- Sample size (2-proportion): $n = \frac{2(z_{1-\alpha/2}+z_{1-\beta})^2 \bar p(1-\bar p)}{(p_B-p_A)^2}$
- Two-proportion $z$: $z = \frac{\hat p_B - \hat p_A}{\sqrt{\bar p(1-\bar p)(1/n_A+1/n_B)}}$

## Must-know facts
- Randomization removes confounding — that's what lets you claim causation.
- Pre-register primary metric, MDE, α, power, and duration before launch.
- Check Sample Ratio Mismatch (SRM) before trusting results.
- SUTVA violation (network effects) breaks standard user-level randomization.

## Quick decisions
| Situation | Do this |
|---|---|
| Want to peek continuously | Use sequential/always-valid testing, not repeated fixed tests |
| Split looks off (e.g. 45/55) | Check for SRM — investigate before trusting results |
| Users interact (social/marketplace) | Cluster or switchback randomization |
| Effect differs by segment | Check for Simpson's paradox before aggregating |

## Common mistakes
- Stopping the test early as soon as it looks significant.
- Trusting results despite a Sample Ratio Mismatch.
- Cherry-picking whichever secondary metric moved.

## One-liner code
```python
from statsmodels.stats.proportion import proportions_ztest
proportions_ztest([conv_a, conv_b], [n_a, n_b])   # returns (z_stat, p_value)
```
