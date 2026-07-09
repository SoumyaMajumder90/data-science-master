# Exploratory Data Analysis — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Look at the data first — shape, distributions, missingness, relationships — before modeling.

## Key formulas
- Correlation: $r = \text{Cov}(X,Y) / (\sigma_X \sigma_Y)$
- IQR outlier bounds: $[Q_1 - 1.5\,\text{IQR},\ Q_3 + 1.5\,\text{IQR}]$

## Must-know facts
- Univariate → bivariate → multivariate is the usual pass order.
- EDA generates hypotheses; it doesn't confirm them.
- Correlation heatmaps only catch **linear** relationships.
- Check the *mechanism* of missing data, not just the amount.

## Quick decisions
| Situation | Do this |
|---|---|
| New dataset | `.info()`, `.describe()`, `.isna().mean()` first |
| Numeric column | Histogram / KDE + boxplot |
| Categorical column | Value counts / bar chart |
| Two numeric columns | Scatter plot + correlation |
| Large dataset, overplotting | Sample, alpha blend, or hexbin |

## Common mistakes
- Treating patterns found via EDA as statistically validated.
- Missing that a numeric-looking column is really categorical (e.g. IDs, zip codes).
- Overplotting dense scatter plots without alpha/sampling.

## One-liner code
```python
df.describe(include="all"); df.isna().mean()
```
