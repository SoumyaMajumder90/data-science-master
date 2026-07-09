# Hypothesis Testing — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Assume $H_0$ (no effect); compute a test statistic and p-value; reject $H_0$ only if $p \le \alpha$.

## Key formulas
- $z = (\bar x - \mu_0)/(\sigma/\sqrt n)$
- Welch's $t = (\bar x_1 - \bar x_2)/\sqrt{s_1^2/n_1 + s_2^2/n_2}$
- $\chi^2 = \sum (O_i-E_i)^2/E_i$

## Must-know facts
- p-value = P(data this extreme | $H_0$ true) — **not** P($H_0$ true).
- Type I error = α = false positive; Type II error = β = false negative; Power = $1-\beta$.
- Multiple tests inflate false positives — correct with Bonferroni/FDR.
- "Not significant" ≠ "no effect" — could just be underpowered.

## Quick decisions
| Situation | Test |
|---|---|
| Compare 2 means | $t$-test (Welch's if unequal variance) |
| Compare >2 means | ANOVA |
| Categorical association | Chi-squared |
| Non-normal / small n | Mann-Whitney U (non-parametric) |
| Many simultaneous tests | Bonferroni / Benjamini-Hochberg |

## Common mistakes
- Peeking at results repeatedly and stopping early ("p-hacking").
- Treating statistical significance as practical significance.
- Misreading p-value as probability $H_0$ is true.

## One-liner code
```python
stats.ttest_ind(group_a, group_b, equal_var=False)
```
