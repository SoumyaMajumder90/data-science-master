# Data Visualization — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Pick the chart that matches the question (comparison/distribution/relationship/trend), encode with position/length over color/angle, and strip chart junk.

## Key formulas
- Histogram bin width (Freedman–Diaconis): $h = 2\,\text{IQR}(x)/n^{1/3}$
- KDE: $\hat f(x) = \frac{1}{nh}\sum K\!\left(\frac{x-x_i}{h}\right)$

## Must-know facts
- Position/length are perceived most accurately; angle/area/color least.
- Log scale for data spanning orders of magnitude or ratio comparisons.
- Anscombe's quartet: identical summary stats, very different shapes — always plot the data.
- Exploratory plots (for you) vs explanatory plots (for others) have different design goals.

## Quick decisions
| Question | Chart |
|---|---|
| Compare categories | Bar chart |
| Distribution of one variable | Histogram / box / violin |
| Relationship between two numerics | Scatter plot |
| Trend over time | Line chart |
| Composition (few categories) | Stacked bar (not pie) |

## Common mistakes
- Truncated y-axis exaggerating bar differences.
- Rainbow palette on ordered/sequential data.
- Overplotting dense scatter data without alpha/sampling.

## One-liner code
```python
sns.histplot(df["x"], kde=True)
```
