# XGBoost — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Regularized, second-order gradient boosting engineered for speed. The tabular go-to; needs tuning + early stopping.

## Key formulas
- Objective: $\sum_i[g_i f + \tfrac12 h_i f^2] + \gamma T + \tfrac12\lambda\lVert w\rVert^2$
- Leaf weight: $w_j^* = -\frac{\sum_{i\in j} g_i}{\sum_{i\in j} h_i + \lambda}$
- Split gain uses $G^2/(H+\lambda)$ minus $\gamma$.

## Must-know facts
- Uses **gradients + Hessians** (Newton boosting).
- **Regularization** built in: $\gamma$, $\lambda$, depth, `min_child_weight`, subsampling.
- **Sparsity-aware**: learns default direction for missing values.
- Low `eta` + **early stopping** is standard.
- Prefer **SHAP** over `gain` importance.

## Quick decisions
| Situation | Do this |
|---|---|
| Overfitting | ↓`eta`, ↓`max_depth`, ↑`min_child_weight`, subsample |
| Imbalance | `scale_pos_weight` |
| Large/high-cardinality data | Try LightGBM / `tree_method="hist"` |
| Missing values | Leave them — handled natively |
| Categoricals | Native categorical or one-hot (not label-as-numeric) |

## Common mistakes
- Too many rounds, no early stopping.
- Eval set leaking into training.
- Trusting `gain` importance blindly.

## One-liner code
```python
xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, early_stopping_rounds=50).fit(Xtr, ytr, eval_set=[(Xval,yval)])
```
