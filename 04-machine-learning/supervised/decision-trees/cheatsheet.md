# Decision Trees — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Recursively split features to reduce impurity; predict via the leaf a sample lands in. Interpretable but high-variance alone.

## Key formulas
- Gini $= 1-\sum_k p_k^2$
- Entropy $= -\sum_k p_k\log_2 p_k$
- Info gain $= I(\text{parent}) - \sum_c \frac{n_c}{n} I(c)$

## Must-know facts
- Greedy top-down CART; **axis-aligned** (staircase) boundaries.
- **No feature scaling** needed.
- Unconstrained trees **overfit** → limit depth/leaf size or prune (`ccp_alpha`).
- High variance → the reason bagging/RF exists.
- Impurity importances are biased toward high-cardinality features.

## Quick decisions
| Situation | Do this |
|---|---|
| Overfitting | ↓`max_depth`, ↑`min_samples_leaf`, prune |
| Class imbalance | `class_weight="balanced"` |
| Want accuracy | Use RF / gradient boosting |
| Reliable importances | Permutation importance |

## Common mistakes
- Growing an unpruned tree and trusting it.
- Reading impurity importances at face value.
- Expecting a regression tree to extrapolate.

## One-liner code
```python
DecisionTreeClassifier(max_depth=4, min_samples_leaf=20).fit(X, y)
```
