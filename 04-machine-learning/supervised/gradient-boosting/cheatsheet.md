# Gradient Boosting — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Add shallow trees sequentially, each fitting the negative-gradient residuals of the current model. Reduces bias; top tabular accuracy with tuning.

## Key formulas
- Pseudo-residual: $r_{im} = -\partial L(y_i, F(x_i))/\partial F(x_i)$
- Update: $F_m = F_{m-1} + \nu\, h_m$
- Squared loss ⇒ residual $= y_i - F_{m-1}(x_i)$

## Must-know facts
- **Sequential**, bias-reducing (vs RF's parallel variance reduction).
- "Gradient" = gradient descent in **function space**.
- Small **learning rate** + many trees + **early stopping** generalizes best.
- More trees **can overfit** (unlike RF).
- Use histogram-based libs (XGBoost/LightGBM/HistGB) for scale.

## Quick decisions
| Situation | Do this |
|---|---|
| Max tabular accuracy | Boosting, tuned |
| Overfitting | ↓ learning rate, early stop, subsample |
| Large data | LightGBM / HistGradientBoosting |
| Low tuning budget | Random forest instead |
| Noisy labels | Robust loss + subsample |

## Common mistakes
- Too many trees without early stopping.
- High learning rate + deep trees.
- Confusing with AdaBoost (reweighting) or RF (bagging).

## One-liner code
```python
HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1000, early_stopping=True).fit(X, y)
```
