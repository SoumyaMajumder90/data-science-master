# Cross-Validation — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Rotate the validation set through $k$ folds and average — a lower-variance estimate than one split. Keep all preprocessing inside the fold loop.

## Key formulas
- $\hat{R}_{CV}=\frac1k\sum_{j=1}^{k}\frac{1}{|F_j|}\sum_{i\in F_j}L(y_i, f^{(-j)}(x_i))$
- LOOCV = $k=n$.

## Must-know facts
- Default **$k=5$ or $10$**.
- Larger $k$ → lower bias, higher variance and cost.
- Use **StratifiedKFold** (classification), **GroupKFold** (repeated units), **TimeSeriesSplit** (temporal).
- **Nested CV** to tune *and* estimate without optimistic bias.
- Refit on all data after CV before deploying.

## Quick decisions
| Situation | Do this |
|---|---|
| Classification | `StratifiedKFold` |
| Repeated entities | `GroupKFold` |
| Time series | `TimeSeriesSplit` (no shuffle) |
| Tuning + honest estimate | Nested CV |
| Huge data / costly model | Single held-out set |

## Common mistakes
- Preprocessing outside the CV loop (leakage) → wrap in a `Pipeline`.
- Plain `KFold` on imbalanced/grouped/temporal data.
- Using the same CV to select and to report performance.

## One-liner code
```python
cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")
```
