# ML Workflow

> The end-to-end pipeline that turns a business problem into a deployed, monitored model: frame → data → features → train → evaluate → deploy → monitor.

| | |
|---|---|
| **Category** | Machine Learning → Fundamentals |
| **Difficulty** | 🟩 Beginner |
| **Prerequisites** | [Feature Engineering](../../../03-data-analysis/feature-engineering/), [Train/Test/Validation Split](../train-test-validation-split/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
Building a model is a small part of a much longer loop. Most real-world effort goes into **framing the problem correctly**, getting clean data, and making the model survive contact with production. The workflow is a checklist that stops you from skipping the unglamorous-but-critical steps (leakage checks, a proper validation split, monitoring) that decide whether a model is useful or a liability.

## 2. Formal definition / Key concepts
A canonical supervised-learning workflow:
1. **Problem framing** — define the target, the unit of prediction, and a success metric tied to business value.
2. **Data collection & labeling** — gather raw data; establish ground truth.
3. **EDA & cleaning** — understand distributions, handle missingness and outliers.
4. **Feature engineering** — transform raw fields into model inputs.
5. **Split** — carve out train / validation / test *before* fitting anything.
6. **Model selection & training** — fit candidates, tune hyperparameters on validation.
7. **Evaluation** — measure on the held-out test set with the right metric.
8. **Deployment** — serve predictions (batch or online).
9. **Monitoring** — watch for data/concept drift and performance decay; retrain.

It is a **loop**, not a line — insights from later stages send you back to earlier ones.

## 3. Math
No single equation, but the guiding principle is **empirical risk minimization** with an eye on generalization: you minimize training loss $\hat{R}(f) = \frac{1}{n}\sum_i L(y_i, f(x_i))$ while the quantity you actually care about is the expected risk on unseen data $R(f) = \mathbb{E}_{(x,y)}[L(y, f(x))]$. Every workflow choice (splitting, CV, regularization) exists to make $\hat{R}$ a trustworthy estimate of $R$.

## 4. How it works
- **Framing first.** "Reduce churn" becomes "predict $P(\text{cancel within 30 days})$ per active user, optimize PR-AUC, act on the top decile."
- **Baseline early.** A trivial model (majority class, last value, linear) sets the bar; anything complex must beat it.
- **Iterate on features and data, not just models** — usually the biggest wins.
- **Freeze the test set.** Touch it once, at the end. Use validation/CV for all tuning.
- **Ship, then watch.** Deployment is the start of the model's life, not the end. Monitor inputs and outputs; schedule retraining.

## 5. When to use / When not to
- ✅ Any supervised or unsupervised modeling project — the skeleton is universal.
- ✅ As a planning and code-review checklist to catch leakage and metric mismatch.
- ❌ Don't over-engineer a one-off analysis; not every question needs deployment/monitoring.
- ❌ Skipping framing to "just try XGBoost" wastes the most time in the long run.

## 6. Common pitfalls & gotchas
- **Data leakage** — fitting scalers/encoders or selecting features on the full dataset before splitting. Always fit transforms on train only (use a `Pipeline`).
- **Metric mismatch** — optimizing accuracy on imbalanced data, or a metric that doesn't map to business value.
- **Peeking at the test set** repeatedly turns it into a second validation set and inflates estimates.
- **Train/serve skew** — features computed differently in training vs production.
- **No monitoring** — silent performance decay from drift is the most common production failure.

## 7. Code
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

# split FIRST — everything below is fit on train only
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])

# tune/estimate on train via CV; test set stays untouched
print(cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean())
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))   # touched once, at the end
```

## 8. Interview / viva questions
- Q: Where does most of the effort in an ML project actually go?
  - A: Problem framing, data collection/cleaning, and feature engineering — modeling itself is usually a minority of the work.
- Q: How do you prevent data leakage in the workflow?
  - A: Split before any fitting, wrap all preprocessing in a pipeline fit on the training fold only, and audit features for future information.
- Q: Why keep a separate test set if you already do cross-validation?
  - A: CV is used for tuning, so it becomes optimistically biased; an untouched test set gives an unbiased final estimate.
- Q: What happens after deployment?
  - A: Monitor input distributions and prediction quality for drift, and retrain on a schedule or when performance degrades.

## 9. References
- Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, ch. 2 (end-to-end project).
- Google — *Rules of Machine Learning* (Zinkevich), best-practices guide.
- scikit-learn User Guide: Pipelines and composite estimators.

---
> _Status: 🟢 done._
