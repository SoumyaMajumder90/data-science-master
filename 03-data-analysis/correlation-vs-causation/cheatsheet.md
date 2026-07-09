# Correlation vs Causation — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Two variables moving together doesn't mean one causes the other — a confounder, reverse causation, or chance can explain it.

## Key formulas
- Pearson $r = \dfrac{\sum(x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum(x_i-\bar x)^2}\sqrt{\sum(y_i-\bar y)^2}}$
- Causal target: $ATE = \mathbb{E}[Y(1)-Y(0)]$ (potential outcomes)

## Must-know facts
- Correlation is symmetric; causation is directional.
- Randomization (RCT/A-B test) is what licenses a causal claim.
- Confounders, reverse causation, and selection bias all mimic causal-looking correlation.
- $r \approx 0$ only rules out a *linear* relationship, not any relationship.

## Quick decisions
| Situation | Do this |
|---|---|
| Want to predict $Y$ | Correlation is fine as a feature signal |
| Want to intervene on $X$ to change $Y$ | Need a causal claim — experiment or confounder-adjusted design |
| Suspect a confounder $Z$ | Control for $Z$ (regression adjustment, matching, stratification) |
| No randomization possible | Look for natural experiments (IV, RDD, diff-in-diff) |

## Common mistakes
- Claiming causation from an observed correlation alone.
- Mining many variable pairs and citing the highest correlation (spurious by chance).
- Ignoring Simpson's paradox — aggregated vs subgroup correlation can flip sign.

## One-liner code
```python
df[["x", "y"]].corr()
```
