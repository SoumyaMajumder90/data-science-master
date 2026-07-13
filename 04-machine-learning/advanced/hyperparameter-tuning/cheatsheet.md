# Hyperparameter Tuning — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Search pre-training dials to maximize CV score. Random/Bayesian beat grid for big spaces. Never tune on the test set.

## Key formulas
- Bi-level: $\theta^*=\arg\max_\theta \frac1k\sum_j \text{score}(f_{\hat w^{(-j)}(\theta)}, \text{fold}_j)$
- Grid cost $=\prod_i m_i$ (exponential in #params).

## Must-know facts
- **Parameters** learned; **hyperparameters** set beforehand.
- **Random search** > grid when few params matter.
- **Bayesian / Hyperband** for expensive models.
- Sample learning rate / $C$ / $\lambda$ on **log scale**.
- Preprocessing inside pipeline; **nested CV** for unbiased estimate.

## Quick decisions
| Situation | Method |
|---|---|
| Few discrete cheap params | Grid |
| Large/continuous space | Random / Bayesian |
| Expensive training | Hyperband / successive halving |
| Need honest estimate | Nested CV |

## Common mistakes
- Tuning on the test set.
- Leakage (preprocessing outside folds).
- Linear-scale sampling of lr/regularization.
- Over-tuning tiny data → validation overfitting.

## One-liner code
```python
RandomizedSearchCV(model, param_dist, n_iter=50, cv=5, scoring="roc_auc", n_jobs=-1).fit(X_train, y_train)
```
