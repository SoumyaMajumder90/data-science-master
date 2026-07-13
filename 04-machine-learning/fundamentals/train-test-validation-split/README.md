# Train / Test / Validation Split

> Partitioning data into disjoint sets — **train** to fit, **validation** to tune, **test** to give one honest final estimate of generalization.

| | |
|---|---|
| **Category** | Machine Learning → Fundamentals |
| **Difficulty** | 🟩 Beginner |
| **Prerequisites** | [ML Workflow](../ml-workflow/), [Overfitting & Underfitting](../overfitting-underfitting/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
If you study from the exam's exact questions, your score says nothing about whether you learned. A model that is evaluated on data it trained on is doing exactly that. We hold data back: the model learns on the **training** set, we tune choices against a **validation** set, and we report the **test** set once — the only truly "unseen" exam.

## 2. Formal definition / Key concepts
- **Training set** — used to fit model parameters (weights).
- **Validation (dev) set** — used to choose hyperparameters, features, and model families. Model selection *uses* it, so scores here are mildly optimistic.
- **Test set** — used exactly once, after all decisions are frozen, for an unbiased estimate of $R(f)=\mathbb{E}[L(y,f(x))]$.
- Typical splits: **60/20/20** or **70/15/15**; with lots of data (millions of rows) validation/test can be far smaller in fraction (e.g. 98/1/1).
- With small data, replace a fixed validation set with **cross-validation** (train once, resample folds).

## 3. Math
The test estimate is a sample mean of the per-example loss:
$$\hat{R}_{\text{test}} = \frac{1}{m}\sum_{i=1}^{m} L(y_i, f(x_i))$$
Its standard error shrinks like $\propto 1/\sqrt{m}$, so a tiny test set gives a noisy, unreliable estimate. This is the trade-off: bigger test set = more reliable estimate but less data to train on.

## 4. How it works
1. **Shuffle** (unless the data is temporal), then split into disjoint sets.
2. Fit *all* preprocessing (scalers, encoders, imputers) on **train only**; apply the fitted transforms to validation/test.
3. Train on train, evaluate candidates on validation, pick the winner.
4. Optionally refit the chosen model on train + validation combined.
5. Report performance on the test set — **once**.

## 5. When to use / When not to
- ✅ Always hold out a test set for any model you'll trust or ship.
- ✅ Use **stratified** splits for classification, especially with imbalance.
- ✅ Use a **time-based** split for time series / any forecasting task.
- ❌ Random splitting when rows are correlated (same user, same patient, time-ordered) → leakage. Split by group/time instead.
- ❌ A single fixed validation split on small data — prefer cross-validation.

## 6. Common pitfalls & gotchas
- **Preprocessing before splitting** leaks test information into training (the #1 mistake).
- **Random split of grouped data** — the same entity in train and test inflates scores. Use `GroupShuffleSplit`.
- **Random split of time series** — you train on the future. Use a chronological cutoff.
- **Tuning on the test set** turns it into a validation set; the "final" number is then optimistic.
- **Imbalanced classes** without `stratify` can leave a rare class absent from a split.

## 7. Code
```python
from sklearn.model_selection import train_test_split

# 60 / 20 / 20 stratified split
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

# time series: no shuffling, chronological cutoff
# X_train, X_test = X[:cut], X[cut:]

# grouped data: keep a group entirely on one side
# from sklearn.model_selection import GroupShuffleSplit
```

## 8. Interview / viva questions
- Q: Why do we need a validation set if we already have a test set?
  - A: Every time you use a set to make a decision it becomes optimistically biased; the validation set absorbs that bias so the test set stays a clean, unbiased final estimate.
- Q: How would you split time-series data?
  - A: Chronologically — train on the past, validate/test on the future — never a random shuffle.
- Q: Your rows are multiple visits per patient. What's the risk of a random split?
  - A: The same patient appears in train and test, leaking information; split by patient group instead.
- Q: When is a single validation split a bad idea?
  - A: On small datasets, where one split is high-variance; use cross-validation.

## 9. References
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 7.
- Andrew Ng — *Machine Learning Yearning* (train/dev/test set design).
- scikit-learn User Guide: Cross-validation and `train_test_split`.

---
> _Status: 🟢 done._
