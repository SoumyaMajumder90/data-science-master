# Descriptive Statistics

> Numerical and graphical summaries — central tendency, spread, and shape — that condense a dataset into a handful of interpretable numbers.

| | |
|---|---|
| **Category** | Data Analysis & Statistics |
| **Difficulty** | 🟩 Beginner |
| **Prerequisites** | [Statistics](../../01-foundations/mathematics/statistics/), [Probability](../../01-foundations/mathematics/probability/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
You can't hold ten thousand numbers in your head, but you can hold "average $52, mostly between $30 and $80, with a few outliers up near $500." Descriptive statistics compress a dataset down to its essential shape — where's the center, how spread out is it, is it symmetric or skewed — without making any claim about a broader population (that's inferential statistics' job).

## 2. Formal definition / Key concepts
- **Central tendency** — mean, median, mode: "where's the typical value?"
- **Dispersion** — range, variance, standard deviation, IQR: "how spread out is it?"
- **Shape** — skewness (asymmetry), kurtosis (tailedness).
- **Population vs sample** — a population parameter (e.g. $\mu$) is a fixed truth; a sample statistic (e.g. $\bar{x}$) is an estimate computed from data, and varies from sample to sample.

## 3. Math
- Mean: $\bar{x} = \dfrac{1}{n}\sum_{i=1}^n x_i$
- Sample variance (Bessel-corrected, unbiased): $s^2 = \dfrac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2$
- Standard deviation: $s = \sqrt{s^2}$
- Median: middle value of sorted data (or average of the two middle values if $n$ is even).
- Skewness (Fisher-Pearson): $g_1 = \dfrac{\frac{1}{n}\sum (x_i-\bar x)^3}{s^3}$
- IQR: $\text{IQR} = Q_3 - Q_1$

## 4. How it works
1. Compute **central tendency**: mean (sensitive to outliers), median (robust), mode (for categorical/multimodal data).
2. Compute **dispersion**: variance/std (in squared/original units), IQR (robust to outliers), coefficient of variation ($s/\bar{x}$) for comparing spread across differently-scaled variables.
3. Assess **shape**: histogram + skewness/kurtosis — is the distribution symmetric, and how heavy are the tails?
4. Summarize with `.describe()` / a five-number summary (min, Q1, median, Q3, max) and visualize with a box plot or histogram.

## 5. When to use / When not to
- ✅ As the very first numeric summary of any dataset.
- ✅ When you need robust, at-a-glance comparisons (e.g. median income across regions).
- ✅ To choose the right central-tendency measure for skewed data (median over mean).
- ❌ Don't use the mean alone to describe a skewed distribution (e.g. income, house prices) — report the median too, or the mean will be pulled by outliers.
- ❌ Don't treat descriptive statistics as proof of a population-level claim — that requires inferential statistics with uncertainty quantification.

## 6. Common pitfalls & gotchas
- **Mean is outlier-sensitive**; a single billionaire changes average income drastically, median doesn't move.
- Reporting **variance** in squared units without also giving **std** in original units for interpretability.
- **Bimodal data** — mean and median can both land in a "valley" that no actual observation occupies; always plot the histogram.
- Confusing **population** ($N$ denominator, $\sigma^2$) and **sample** ($n-1$ denominator, $s^2$) variance formulas — using $n$ underestimates variance in a sample.
- Anscombe's quartet / the Datasaurus Dozen: identical descriptive statistics can hide wildly different underlying data — always visualize alongside summarizing.

## 7. Code
```python
import pandas as pd

s = df["price"]
s.mean(), s.median(), s.mode()[0]
s.std(), s.var(), (s.quantile(0.75) - s.quantile(0.25))   # std, var, IQR
s.skew(), s.kurt()
df.describe()   # five-number summary + mean/std for all numeric columns
```

## 8. Interview / viva questions
- Q: When would you report the median instead of the mean?
  - A: When the distribution is skewed or has outliers (income, house prices, latency) — the median better represents the "typical" value.
- Q: Why divide by $n-1$ instead of $n$ for sample variance?
  - A: Using the sample mean $\bar x$ (instead of the true $\mu$) to compute deviations underestimates variance; dividing by $n-1$ (Bessel's correction) makes it an unbiased estimator of population variance.
- Q: Two datasets have the same mean, median, and standard deviation. Are they the same?
  - A: Not necessarily — Anscombe's quartet shows datasets with identical summary statistics but very different shapes/relationships. Always visualize.

## 9. References
- Wasserman, L. — *All of Statistics*, ch. 1–2.
- Anscombe, F. — "Graphs in Statistical Analysis" (1973).
- NIST/SEMATECH e-Handbook of Statistical Methods, ch. 1.

---
> _Status: 🟢 done._
