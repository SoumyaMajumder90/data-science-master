# Cross-Validation

> Resampling the data into multiple train/validation folds so every point is validated on once, giving a lower-variance estimate of generalization than a single split.

| | |
|---|---|
| **Category** | Machine Learning → Fundamentals |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [Train/Test/Validation Split](../train-test-validation-split/), [Bias-Variance Tradeoff](../bias-variance-tradeoff/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
A single validation split gives one score — but that score depends on *which* rows happened to land in the validation set. On small data that luck-of-the-draw noise is large. Cross-validation removes the luck: split the data into $k$ parts, train on $k-1$ and validate on the held-out part, rotate through all $k$, and average. Now every row is used for both training and validation, and you get an estimate plus its variability.

## 2. Formal definition / Key concepts
- **k-fold CV** — partition data into $k$ equal folds; each fold serves once as validation while the other $k-1$ train. Report the mean (and std) of the $k$ scores. Typical $k = 5$ or $10$.
- **Stratified k-fold** — preserves class proportions in each fold; the default for classification.
- **Leave-One-Out (LOOCV)** — $k = n$; nearly unbiased but high variance and expensive.
- **Group k-fold** — keeps all rows from one group in the same fold (no group leakage).
- **Time-series CV** — expanding/rolling window; only ever validate on the future.
- **Nested CV** — an inner loop tunes hyperparameters, an outer loop estimates performance, avoiding selection bias.

## 3. Math
The CV estimate of risk averages the fold losses:
$$\hat{R}_{CV} = \frac{1}{k}\sum_{j=1}^{k} \frac{1}{|F_j|}\sum_{i \in F_j} L\big(y_i, f^{(-j)}(x_i)\big)$$
where $f^{(-j)}$ is the model trained with fold $F_j$ removed. Larger $k$ → less bias (more training data per fit) but higher variance and cost; LOOCV is the extreme with $k=n$.

## 4. How it works
1. (Optionally shuffle and) split indices into $k$ folds.
2. For each fold $j$: fit the **entire pipeline** on the other folds, predict on fold $j$, record the score.
3. Average the $k$ scores → performance estimate; the std across folds signals stability.
4. For tuning, run CV for each hyperparameter setting and pick the best mean score.
5. Refit the final model on all training data before deploying.

## 5. When to use / When not to
- ✅ Small or medium datasets where a single validation split is too noisy.
- ✅ Hyperparameter tuning and model comparison (`GridSearchCV`, `cross_val_score`).
- ✅ When you need a variance estimate, not just a point score.
- ❌ Very large datasets or expensive models — a single held-out set is often enough and far cheaper.
- ❌ Plain k-fold on grouped or temporal data — use group/time-series variants.

## 6. Common pitfalls & gotchas
- **Leakage across folds** — fitting scalers/encoders/feature-selection *outside* the CV loop leaks validation info. Put every transform inside a `Pipeline` so it refits per fold.
- **Wrong splitter** — using `KFold` instead of `StratifiedKFold` (imbalance) or `GroupKFold` / `TimeSeriesSplit` (structure).
- **Tuning + estimating in one loop** — reusing CV both to select and to report is optimistic; use **nested CV**.
- **Shuffling time series** destroys temporal order.
- CV estimates the performance of the *procedure*, not of one specific fitted model.

## 7. Code
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
print(f"AUC: {scores.mean():.3f} +/- {scores.std():.3f}")
```

## 8. Interview / viva questions
- Q: Why is 5-fold CV usually better than one 80/20 split?
  - A: It uses every point for validation and averages over splits, lowering the variance of the estimate — valuable when data is limited.
- Q: What is the bias-variance trade-off in choosing $k$?
  - A: Larger $k$ means more training data per fold (lower bias) but more correlated, higher-variance estimates and higher cost; $k=5$–$10$ is the usual sweet spot.
- Q: How do you avoid leakage in cross-validation?
  - A: Wrap all preprocessing in a pipeline so it is refit within each fold, never on the full dataset.
- Q: What is nested cross-validation for?
  - A: To get an unbiased performance estimate when you're also tuning hyperparameters — the inner loop tunes, the outer loop evaluates.

## 9. References
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 7.5–7.10.
- Cawley & Talbot (2010) — "On Over-fitting in Model Selection…" (nested CV).
- scikit-learn User Guide: Cross-validation.

---
> _Status: 🟢 done._
