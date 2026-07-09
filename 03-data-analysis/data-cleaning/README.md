# Data Cleaning

> The process of detecting and fixing (or removing) corrupt, missing, duplicated, or inconsistent data so that downstream analysis and models aren't corrupted by garbage input.

| | |
|---|---|
| **Category** | Data Analysis & Statistics |
| **Difficulty** | 🟩 Beginner |
| **Prerequisites** | [Exploratory Data Analysis](../exploratory-data-analysis/), [Descriptive Statistics](../descriptive-statistics/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
"Garbage in, garbage out." Real-world data is messy: sensors drop readings, users mistype forms, systems merge inconsistently. Data cleaning is the unglamorous work of making a dataset trustworthy — fixing what can be fixed, flagging what can't, and documenting every decision so it's reproducible.

## 2. Formal definition / Key concepts
- **Missing data** — values absent from the dataset (`NaN`, `NULL`, sentinel values like `-999`).
  - **MCAR** (Missing Completely At Random), **MAR** (Missing At Random, depends on observed data), **MNAR** (Missing Not At Random, depends on the missing value itself) — the mechanism determines whether imputation is safe.
- **Duplicates** — exact or near-duplicate rows/records.
- **Outliers** — values far from the rest of the distribution (may be errors or genuine extremes).
- **Inconsistency** — same entity represented differently (`"NY"` vs `"New York"`), unit mismatches, encoding issues.
- **Schema/type errors** — a numeric column stored as text, wrong date formats.

## 3. Math
- Z-score outlier flag: $z = \dfrac{x - \mu}{\sigma}$, typically flag $|z| > 3$.
- IQR outlier flag: $x < Q_1 - 1.5\,\text{IQR}$ or $x > Q_3 + 1.5\,\text{IQR}$.
- Mean/median imputation replaces a missing $x_i$ with $\bar{x}$ or $\text{median}(x)$ — simple but shrinks variance and biases correlations.

## 4. How it works
1. **Profile** — quantify missingness, duplicates, dtypes, cardinality (usually as part of EDA).
2. **Standardize** — fix types, units, casing, categorical spellings, date formats.
3. **Handle missing values** — drop rows/columns, impute (mean/median/mode, forward-fill, model-based like KNN/MICE), or add a "was missing" indicator flag.
4. **Handle duplicates** — deduplicate exact matches; use fuzzy matching for near-duplicates (e.g. `record_linkage`, `rapidfuzz`).
5. **Handle outliers** — investigate first (error vs genuine); then cap/winsorize, transform (log), or leave in for robust models.
6. **Validate** — re-run profiling; assert constraints (e.g. age ≥ 0) with a schema/validation library.

## 5. When to use / When not to
- ✅ Always, before any serious analysis or model training.
- ✅ As an automated, repeatable pipeline step — not a one-off manual notebook edit.
- ❌ Don't "clean" away genuine signal (e.g. dropping all outliers in fraud data removes the fraud cases you're trying to detect).
- ❌ Don't impute blindly on MNAR data — the fact that it's missing may itself be informative; consider a missingness indicator instead.

## 6. Common pitfalls & gotchas
- **Silent data loss** — dropping NA rows can shrink your dataset drastically without anyone noticing; always log before/after counts.
- **Leaking test information** — computing imputation statistics (mean, mode) on the full dataset before a train/test split leaks test info into training.
- **Over-cleaning** — deleting legitimate extreme values because they "look wrong."
- Treating **sentinel values** (`-999`, `"N/A"` as a string) as valid numbers instead of missing.
- Not **documenting** cleaning steps — makes the pipeline unreproducible and hides assumptions.

## 7. Code
```python
import pandas as pd

df = pd.read_csv("data.csv")

# Standardize types & missing sentinels
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df.replace(["N/A", "-999", ""], pd.NA, inplace=True)

# Missingness indicator + imputation (fit stats on TRAIN split only in practice)
df["age_missing"] = df["age"].isna().astype(int)
df["age"] = df["age"].fillna(df["age"].median())

# Deduplicate
df = df.drop_duplicates(subset=["user_id", "event_time"])

# Outlier cap (winsorize) via IQR
q1, q3 = df["amount"].quantile([0.25, 0.75])
iqr = q3 - q1
df["amount"] = df["amount"].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
```

## 8. Interview / viva questions
- Q: MCAR vs MAR vs MNAR — why does it matter?
  - A: It determines whether simple imputation is valid. MCAR/MAR can often be safely imputed from observed data; MNAR imputation can introduce systematic bias because the missingness itself carries information.
- Q: When would you NOT remove outliers?
  - A: When they're genuine and the task cares about them (fraud, rare disease detection), or when using a model robust to outliers (tree-based methods) — removing them would delete signal.
- Q: How do you avoid data leakage during cleaning?
  - A: Fit any statistic used for cleaning (mean, median, mode, scaler parameters) only on the training split, then apply it to validation/test — never compute on the full dataset before splitting.

## 9. References
- Van Buuren, S. — *Flexible Imputation of Missing Data*.
- Rubin, D. — "Inference and Missing Data" (1976), the original MCAR/MAR/MNAR framework.
- pandas documentation: Working with missing data.

---
> _Status: 🟢 done._
