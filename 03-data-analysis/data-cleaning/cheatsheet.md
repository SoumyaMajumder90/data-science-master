# Data Cleaning — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Detect and fix missing values, duplicates, outliers, and inconsistencies before analysis or modeling — and document every step.

## Key formulas
- Z-score outlier: $z = (x-\mu)/\sigma$, flag $|z|>3$
- IQR outlier bounds: $[Q_1 - 1.5\,\text{IQR},\ Q_3 + 1.5\,\text{IQR}]$

## Must-know facts
- Missingness mechanisms: **MCAR / MAR / MNAR** — determines if imputation is safe.
- Fit imputation/scaling stats on the **train split only**, to avoid leakage.
- A missing-value indicator column can preserve MNAR signal that imputation would erase.
- Outliers aren't automatically errors — investigate before removing.

## Quick decisions
| Situation | Do this |
|---|---|
| Few missing, MCAR | Drop rows |
| Missing correlates with other features | Impute (median/KNN/MICE) + indicator flag |
| Exact duplicate rows | `drop_duplicates()` |
| Near-duplicate records | Fuzzy matching (`rapidfuzz`) |
| Extreme values could be genuine | Investigate before capping/removing |

## Common mistakes
- Imputing using stats computed on the full dataset (train+test leakage).
- Silently dropping rows without logging how many were lost.
- Treating sentinel strings like `"N/A"` or `-999` as valid numeric data.

## One-liner code
```python
df["x"] = df["x"].fillna(df["x"].median()); df = df.drop_duplicates()
```
