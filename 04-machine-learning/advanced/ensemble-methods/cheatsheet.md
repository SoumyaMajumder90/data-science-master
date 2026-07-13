# Ensemble Methods — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Combine diverse models so errors cancel. Bagging ↓variance, boosting ↓bias, stacking learns the blend.

## Key formulas
- Averaged variance: $\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$
- Boosting additive model: $F_M=\sum_m\nu h_m$

## Must-know facts
- **Bagging** = parallel, independent, ↓variance (RF).
- **Boosting** = sequential, fit residuals, ↓bias (XGBoost).
- **Stacking** = meta-learner on **out-of-fold** base preds.
- Gains need **diverse** (decorrelated) members.
- Costs: compute, latency, interpretability.

## Quick decisions
| Base model problem | Use |
|---|---|
| High variance (deep trees) | Bagging / RF |
| High bias (stumps) | Boosting |
| Many different families | Stacking / voting |
| Need interpretability/latency | Single model |

## Common mistakes
- Combining correlated models (no gain).
- Training meta-learner on in-sample preds (leakage).
- Overcomplex meta-learner.

## One-liner code
```python
StackingClassifier(estimators=base, final_estimator=LogisticRegression(), cv=5).fit(X, y)
```
