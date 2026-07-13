# Random Forest — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Bag many deep, decorrelated trees (bootstrap rows + random features per split) and average. Cuts variance; strong low-tuning tabular default.

## Key formulas
- Ensemble variance: $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$
- Feature subset: $m\approx\sqrt{p}$ (clf), $p/3$ (reg)
- OOB fraction $\approx e^{-1} \approx 37\%$

## Must-know facts
- Random **feature subsets** decorrelate trees — the key over plain bagging.
- Reduces **variance, not bias**; grow trees deep.
- **OOB error** = free validation estimate.
- Impurity importances biased → use **permutation importance**.
- Can't extrapolate beyond training target range.

## Quick decisions
| Situation | Do this |
|---|---|
| Strong tabular baseline | RF with defaults |
| Imbalance | `class_weight="balanced"` |
| Need max accuracy | Try gradient boosting/XGBoost |
| Reliable importances | Permutation / SHAP |
| Latency-sensitive | Fewer/shallower trees or another model |

## Common mistakes
- Expecting it to fix underfitting.
- Trusting impurity importances with correlated features.
- Using raw vote-averaged probabilities without calibration.

## One-liner code
```python
RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=-1, oob_score=True).fit(X, y)
```
