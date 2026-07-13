# Random Forest

> An ensemble of decorrelated decision trees — each trained on a bootstrap sample with random feature subsets — averaged to cut variance without raising bias.

| | |
|---|---|
| **Category** | Machine Learning → Supervised |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [Decision Trees](../decision-trees/), [Bias-Variance Tradeoff](../../fundamentals/bias-variance-tradeoff/), [Ensemble Methods](../../advanced/ensemble-methods/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
One decision tree is a smart but jumpy expert — small data changes swing its answer. Ask hundreds of experts, each shown a slightly different slice of data and allowed to consider only a random subset of clues at each decision, then take a vote. The individual mistakes are somewhat independent, so they cancel on average. That's a random forest: **many overfit trees, averaged, become a stable model**.

## 2. Formal definition / Key concepts
- **Bagging (bootstrap aggregating):** train each tree on a bootstrap resample (sample $n$ rows with replacement).
- **Feature bagging:** at each split consider only a random subset of $m$ features ($m \approx \sqrt{p}$ for classification, $p/3$ for regression) — this *decorrelates* the trees, the key innovation over plain bagging.
- **Aggregation:** majority vote (classification) or mean (regression).
- **Out-of-bag (OOB) error:** each tree's ~37% unsampled rows form a free validation set.
- Trees are grown **deep** (low bias, high variance); the ensemble kills the variance.

## 3. Math
Averaging $B$ trees, each with variance $\sigma^2$ and pairwise correlation $\rho$, gives ensemble variance:
$$\text{Var} = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2$$
More trees ($B\uparrow$) shrink the second term; lowering the correlation $\rho$ (via random feature subsets) shrinks the first — which is why feature bagging matters. A row is out-of-bag for a tree with probability $(1-\tfrac1n)^n \to e^{-1}\approx 0.368$.

## 4. How it works
1. For $b = 1..B$: draw a bootstrap sample; grow a tree, choosing each split from a **random subset of $m$ features**.
2. Grow trees deep (little/no pruning).
3. **Predict** by averaging (regression) or voting (classification) across all trees.
4. Estimate generalization with **OOB error**; rank features via impurity or (better) **permutation importance**.

## 5. When to use / When not to
- ✅ A strong, low-tuning **default** for tabular classification/regression.
- ✅ Robust to outliers, mixed feature types, and non-linear interactions; no scaling needed.
- ✅ When you want built-in validation (OOB) and feature importances.
- ❌ Very high-accuracy needs on structured data where **gradient boosting** usually edges it out.
- ❌ Low-latency/low-memory serving (hundreds of deep trees are heavy), or when you need a single interpretable model.

## 6. Common pitfalls & gotchas
- **Not a bias reducer** — bagging cuts variance; if trees underfit, the forest still underfits.
- **Impurity importances are biased** toward high-cardinality/continuous features; use permutation importance or SHAP.
- **Correlated features** split importance among themselves, hiding relevance.
- **Poor extrapolation** — like trees, it can't predict beyond the training target range.
- **Probabilities are averaged votes** — often need calibration.
- More trees never hurt accuracy but cost memory/latency; tune `max_features` and depth for real gains.

## 7. Code
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(
    n_estimators=500, max_features="sqrt", n_jobs=-1,
    oob_score=True, class_weight="balanced", random_state=0)
rf.fit(X_train, y_train)

print("OOB accuracy:", rf.oob_score_)
imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=0)
print("perm importances:", imp.importances_mean)
```

## 8. Interview / viva questions
- Q: How does a random forest differ from plain bagging of trees?
  - A: It also samples a random subset of features at each split, decorrelating the trees so averaging reduces variance more.
- Q: Why grow the trees deep instead of pruning them?
  - A: Deep trees are low-bias/high-variance; averaging removes the variance, so pruning would only add bias.
- Q: What is OOB error?
  - A: Each tree is validated on the ~37% of rows it didn't sample; aggregating gives a cross-validation-like estimate for free.
- Q: Random forest vs gradient boosting?
  - A: RF builds independent deep trees in parallel to reduce variance (hard to overfit, low tuning); boosting builds shallow trees sequentially to reduce bias (often higher accuracy, more tuning).

## 9. References
- Breiman (2001) — "Random Forests", *Machine Learning* 45(1).
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 15.
- scikit-learn User Guide: Ensemble methods — Forests of randomized trees.

---
> _Status: 🟢 done._
