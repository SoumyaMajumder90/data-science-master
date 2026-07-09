# Descriptive Statistics — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Summarize data with central tendency (mean/median/mode), dispersion (std/IQR), and shape (skew/kurtosis) — no inference to a population involved.

## Key formulas
- Mean: $\bar x = \frac{1}{n}\sum x_i$
- Sample variance: $s^2 = \frac{1}{n-1}\sum(x_i-\bar x)^2$
- IQR: $Q_3 - Q_1$
- Skewness: $g_1 = \frac{\frac1n\sum(x_i-\bar x)^3}{s^3}$

## Must-know facts
- Mean is outlier-sensitive; median is robust.
- Sample variance uses $n-1$ (Bessel's correction) to be unbiased.
- Skewness > 0: right tail longer (mean > median).
- Descriptive ≠ inferential — no uncertainty/generalization claims here.

## Quick decisions
| Situation | Do this |
|---|---|
| Skewed data (income, prices) | Report median + IQR |
| Symmetric data | Mean + std is fine |
| Comparing spread across scales | Coefficient of variation ($s/\bar x$) |
| Categorical data | Mode + frequency table |

## Common mistakes
- Reporting only the mean for skewed distributions.
- Using $n$ instead of $n-1$ in the variance denominator for a sample.
- Trusting summary stats without plotting (Anscombe's quartet).

## One-liner code
```python
df.describe()
```
