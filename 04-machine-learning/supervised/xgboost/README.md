# XGBoost

> A fast, regularized implementation of gradient boosting that uses second-order (Newton) optimization, a penalized tree objective, and system-level engineering to win on tabular data.

| | |
|---|---|
| **Category** | Machine Learning → Supervised |
| **Difficulty** | 🟥 Advanced |
| **Prerequisites** | [Gradient Boosting](../gradient-boosting/), [Regularization](../../advanced/regularization/), [Hyperparameter Tuning](../../advanced/hyperparameter-tuning/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
XGBoost ("eXtreme Gradient Boosting") is gradient boosting made industrial. It keeps the core idea — add trees that correct residual errors — but improves it in three ways: it uses the loss's **curvature** (second derivative), not just the slope, to choose better splits; it **regularizes** each tree so the ensemble doesn't overfit; and it's engineered for **speed** (histogram splits, parallelism, cache-awareness). For years it was the default winner of Kaggle tabular competitions.

## 2. Formal definition / Key concepts
- Minimizes a **regularized objective**: training loss + a complexity penalty on trees.
- Uses a **second-order Taylor expansion** of the loss (gradients $g_i$ and Hessians $h_i$) → Newton boosting.
- **Regularization:** $\gamma$ (min split gain), $\lambda$ (L2 on leaf weights), `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`.
- **Sparsity-aware split finding** learns a default direction for missing values.
- **Approximate/histogram** split finding (`tree_method="hist"`) for scalability; GPU support.

## 3. Math
Objective at round $t$ with regularization $\Omega(f) = \gamma T + \tfrac12\lambda\lVert w\rVert^2$ ($T$ = #leaves):
$$\mathcal{L}^{(t)} \approx \sum_i \big[g_i f_t(x_i) + \tfrac12 h_i f_t(x_i)^2\big] + \Omega(f_t)$$
For a fixed tree structure, the optimal leaf weight and the resulting gain of a split are:
$$w_j^* = -\frac{\sum_{i\in j} g_i}{\sum_{i\in j} h_i + \lambda},\qquad
\text{Gain} = \tfrac12\!\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$$

## 4. How it works
1. Start with a base score; compute $g_i, h_i$ for every row from the loss.
2. Grow each tree by choosing splits that maximize the **regularized gain** above (a split is kept only if gain > 0, i.e. exceeds $\gamma$).
3. Set leaf weights to $w_j^*$; add the tree scaled by the learning rate $\eta$.
4. Handle missing values by routing them to the learned default branch.
5. Repeat with **early stopping** on a watchlist; use `hist`/GPU for large data.

## 5. When to use / When not to
- ✅ **Tabular** classification, regression, and ranking where accuracy is the goal.
- ✅ Medium-to-large datasets; native missing-value handling; needs a differentiable loss.
- ✅ Competitions and production tabular pipelines.
- ❌ Unstructured data (images/text/audio) — use deep learning.
- ❌ Tiny datasets or when a simple, fully interpretable model suffices.
- ❌ If training speed on huge/high-cardinality data is critical, **LightGBM** is often faster.

## 6. Common pitfalls & gotchas
- **Too many rounds** overfit — always pair a low `eta` with `early_stopping_rounds` and an eval set.
- **Leakage via early stopping** if the eval set isn't a clean hold-out.
- Key knobs interact: `eta`↔`n_estimators`, `max_depth`/`min_child_weight` (complexity), `subsample`/`colsample_bytree` (variance), `gamma`/`lambda` (regularization).
- **Imbalance:** tune `scale_pos_weight`, not just the threshold.
- **`gain` feature importance** is biased; prefer SHAP values (XGBoost has built-in support).
- Passing a raw label-encoded categorical as numeric imposes a fake order — one-hot or use the native categorical support.

## 7. Code
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=2000, learning_rate=0.03, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, gamma=0.0, eval_metric="auc",
    early_stopping_rounds=50, n_jobs=-1, tree_method="hist")

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print("best iteration:", model.best_iteration)

import shap
explainer = shap.TreeExplainer(model)          # trustworthy importances
shap_values = explainer.shap_values(X_val)
```

## 8. Interview / viva questions
- Q: What does XGBoost add over vanilla gradient boosting?
  - A: A regularized objective (γ, λ), second-order (Newton) split gains using gradients and Hessians, sparsity-aware missing-value handling, and heavy system optimization (histogram splits, parallelism).
- Q: Why use the Hessian?
  - A: The second-order Taylor expansion gives more accurate leaf weights and split gains than gradient-only steps, improving convergence and quality.
- Q: How does it handle missing values?
  - A: Each split learns a default direction, so missing entries are routed consistently without imputation.
- Q: XGBoost vs LightGBM?
  - A: Similar accuracy; LightGBM grows leaf-wise with histogram binning and is usually faster/lower-memory on large, high-cardinality data, while XGBoost's level-wise growth can be more robust on smaller sets.

## 9. References
- Chen & Guestrin (2016) — "XGBoost: A Scalable Tree Boosting System", KDD.
- XGBoost official documentation (parameters, tree methods).
- Lundberg & Lee (2017) — "A Unified Approach to Interpreting Model Predictions" (SHAP).

---
> _Status: 🟢 done._
