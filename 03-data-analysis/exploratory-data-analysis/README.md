# Exploratory Data Analysis

> The practice of investigating a dataset — through summary statistics and visualization — to understand its structure, spot anomalies, and form hypotheses before modeling.

| | |
|---|---|
| **Category** | Data Analysis & Statistics |
| **Difficulty** | 🟩 Beginner |
| **Prerequisites** | [Descriptive Statistics](../descriptive-statistics/), [Data Visualization](../data-visualization/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
Before you model data, you should look at it. EDA is the detective work of data science: get a feel for the shape of each variable, how variables relate to each other, and where the data misbehaves — outliers, missing values, weird distributions — before committing to a modeling approach. John Tukey, who coined the term, framed it as generating hypotheses visually rather than confirming them statistically.

## 2. Formal definition / Key concepts
- **Univariate analysis** — examine one variable at a time (distribution, central tendency, spread).
- **Bivariate/multivariate analysis** — examine relationships between two or more variables (correlation, cross-tabs, grouped summaries).
- **Data profiling** — shape, dtypes, missingness, cardinality, duplicates.
- EDA is iterative and open-ended, contrasted with **confirmatory data analysis** (hypothesis testing), which is planned in advance on unseen data.

## 3. Math
EDA doesn't have one governing equation, but it leans on:
- Summary statistics: mean $\bar{x} = \frac{1}{n}\sum x_i$, variance $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$.
- Correlation: $r = \dfrac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$.
- Outlier bounds via IQR: $[Q_1 - 1.5\,\text{IQR},\ Q_3 + 1.5\,\text{IQR}]$, where $\text{IQR} = Q_3 - Q_1$.

## 4. How it works
1. **Understand the shape** — `.shape`, `.dtypes`, `.info()`.
2. **Univariate pass** — histograms/KDEs for numeric columns, value counts/bar charts for categoricals; check `.describe()`.
3. **Missingness & duplicates** — quantify and visualize (e.g. a missingness heatmap).
4. **Bivariate pass** — correlation matrix/heatmap, scatter plots, grouped `groupby().agg()`, box plots by category.
5. **Outlier check** — box plots, z-scores, the IQR rule.
6. **Write down hypotheses** — what looks predictive, what needs cleaning, what needs a transform — to carry into cleaning/feature engineering.

## 5. When to use / When not to
- ✅ At the start of every project, before modeling or dashboarding.
- ✅ After any major data pipeline change, as a sanity check.
- ✅ When debugging a model that performs unexpectedly — re-inspect the data.
- ❌ As a substitute for rigorous hypothesis testing — EDA generates hypotheses, it doesn't confirm them (risk of p-hacking if you test on the same data you explored).
- ❌ On data you haven't secured/anonymized appropriately — profiling can leak sensitive values into notebooks.

## 6. Common pitfalls & gotchas
- **Confirmation bias** — hunting for patterns until you find one, then treating it as validated fact.
- Drawing statistical conclusions (p-values) from patterns discovered during EDA on the *same* data — validate on held-out data instead.
- Ignoring **data types** — a numeric-looking column (e.g. zip code) may actually be categorical.
- Skipping the **missingness mechanism** — is data missing at random, or systematically (which changes how you should handle it)?
- Overplotting on large datasets, hiding density — use sampling, alpha transparency, or hexbin/2D histograms.

## 7. Code
```python
import pandas as pd

df = pd.read_csv("data.csv")
df.info()
df.describe(include="all")
df.isna().mean().sort_values(ascending=False)   # missingness by column

import seaborn as sns
sns.histplot(df["feature"], kde=True)
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
```

## 8. Interview / viva questions
- Q: What's the difference between EDA and confirmatory data analysis?
  - A: EDA is open-ended pattern discovery to generate hypotheses; confirmatory analysis tests a pre-specified hypothesis, ideally on separate data, to avoid bias.
- Q: How would you EDA a dataset with 200 columns?
  - A: Start with automated profiling (dtypes, missingness, cardinality), cluster columns by type, prioritize by relevance to the target (e.g. correlation/mutual information), then drill into the top candidates manually.
- Q: Your correlation heatmap shows nothing above 0.3. Does that mean no relationships exist?
  - A: No — correlation only captures linear relationships; non-linear or interaction effects can be invisible to it. Check scatter plots and mutual information too.

## 9. References
- Tukey, J. — *Exploratory Data Analysis* (1977).
- Wickham, H. & Grolemund, G. — *R for Data Science*, ch. on Exploratory Data Analysis.
- pandas and seaborn official documentation.

---
> _Status: 🟢 done._
