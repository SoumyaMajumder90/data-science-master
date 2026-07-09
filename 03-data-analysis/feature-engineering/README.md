# Feature Engineering

> Transforming raw data into input variables (features) that better expose the underlying patterns to a model — often the single highest-leverage step in a modeling pipeline.

| | |
|---|---|
| **Category** | Data Analysis & Statistics |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [Data Cleaning](../data-cleaning/), [Exploratory Data Analysis](../exploratory-data-analysis/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
A model can only find patterns in the representation you give it. Raw timestamps hide "is this a weekend?"; raw text hides "does this review sound angry?" Feature engineering is translating domain knowledge into columns a model can actually use — often, a simple model with great features beats a complex model with raw ones ("better features beat better algorithms" is a common maxim in applied ML).

## 2. Formal definition / Key concepts
- **Feature transformation** — reshaping existing features (scaling, log transform, binning) without adding new information.
- **Feature creation** — deriving new features from existing ones or domain knowledge (ratios, interactions, date parts, aggregations).
- **Feature encoding** — converting categorical/text/etc. into numeric form a model can consume (one-hot, target/ordinal encoding, embeddings).
- **Feature selection** — choosing a useful subset (covered in depth in [Feature Selection](../../04-machine-learning/advanced/feature-selection/)); feature engineering is about *creating* good candidates first.

## 3. Math
- Standardization (z-score scaling): $x' = \dfrac{x - \mu}{\sigma}$
- Min-max scaling: $x' = \dfrac{x - x_{\min}}{x_{\max} - x_{\min}}$
- Log transform (for right-skewed, positive data): $x' = \log(x + 1)$
- One-hot encoding of a $k$-category variable → $k$ (or $k-1$, to avoid collinearity) binary columns.
- Target encoding (mean encoding) for category $c$: $x'_c = \mathbb{E}[y \mid x = c]$, usually smoothed toward the global mean to avoid overfitting rare categories.

## 4. How it works
1. **Understand the data types** — numeric, categorical, datetime, text, geospatial — each has its own toolbox.
2. **Numeric features** — scale/normalize for distance-based or gradient-based models (KNN, SVM, neural nets); log/Box-Cox transform skewed features; bin continuous variables if the relationship is non-linear and you want interpretability.
3. **Categorical features** — one-hot for low cardinality, target/frequency/embedding encoding for high cardinality; always fit the encoder on train only.
4. **Datetime features** — extract hour/day-of-week/month, cyclic encoding ($\sin$/$\cos$ of the angle) for periodicity, time-since-event features.
5. **Interaction/derived features** — ratios, differences, domain-specific combinations (e.g. `price_per_sqft = price / sqft`).
6. **Validate with the model** — feature engineering is iterative; check whether new features actually improve held-out performance, not just training performance.

## 5. When to use / When not to
- ✅ Always — even "let the model learn it" architectures (deep nets) benefit from sensible input representations.
- ✅ Especially valuable for classical ML (trees, linear models) which don't learn representations automatically the way deep nets do.
- ❌ Don't engineer features using information unavailable at prediction time (future data, target leakage) — see pitfalls below.
- ❌ Don't blindly add every possible interaction — it multiplies dimensionality and overfitting risk; let domain knowledge or feature-importance analysis guide you.

## 6. Common pitfalls & gotchas
- **Target/data leakage** — accidentally including information that wouldn't be available at prediction time (e.g. using a feature computed *after* the outcome, or fitting an encoder on the full dataset including test).
- **Fitting transformers on the full dataset** before train/test split — leaks test distribution info into training (same issue as in [Data Cleaning](../data-cleaning/)).
- **High-cardinality categoricals with one-hot encoding** — explodes dimensionality; use target encoding, hashing, or embeddings instead.
- **Ignoring cyclical structure** — encoding "hour of day" as a raw integer (0–23) tells the model 23 and 0 are far apart, when they're adjacent; use sine/cosine encoding.
- **Over-engineering** — hundreds of hand-crafted features with no validation lift add noise and maintenance burden.

## 7. Code
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Cyclic encoding for hour-of-day
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Log transform for a skewed positive feature
df["log_price"] = np.log1p(df["price"])

# Ratio / derived feature
df["price_per_sqft"] = df["price"] / df["sqft"]

# Fit scaler on TRAIN only, apply to both
scaler = StandardScaler().fit(X_train[["income"]])
X_train["income_scaled"] = scaler.transform(X_train[["income"]])
X_test["income_scaled"] = scaler.transform(X_test[["income"]])
```

## 8. Interview / viva questions
- Q: What's the difference between feature engineering and feature selection?
  - A: Feature engineering *creates* candidate features (transforms, combinations, encodings); feature selection *chooses* which of those (and the raw features) to keep for the model.
- Q: Why use sine/cosine encoding for "hour of day" instead of the raw integer?
  - A: Raw integers impose a false linear ordering (hour 23 and hour 0 look maximally far apart to the model), while sine/cosine encoding preserves the cyclical adjacency — 23:00 and 00:00 end up close in the encoded space.
- Q: How do you encode a categorical feature with 50,000 unique values?
  - A: One-hot would create 50,000 sparse columns — instead use target/mean encoding (with smoothing/regularization to avoid leakage on rare categories), frequency encoding, hashing, or learned embeddings.

## 9. References
- Zheng, A. & Casari, A. — *Feature Engineering for Machine Learning*.
- Kuhn, M. & Johnson, K. — *Feature Engineering and Selection: A Practical Approach for Predictive Models*.
- scikit-learn User Guide: Preprocessing data.

---
> _Status: 🟢 done._
