# Regularization — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Penalize complexity (loss + λ·penalty) to cut variance and overfitting. L1 = sparse/selection, L2 = smooth shrinkage.

## Key formulas
- Ridge (L2): $\min_w \frac1n\lVert y-Xw\rVert_2^2 + \lambda\lVert w\rVert_2^2$
- Lasso (L1): $\min_w \frac1n\lVert y-Xw\rVert_2^2 + \lambda\lVert w\rVert_1$
- Ridge closed form: $\hat w=(X^\top X+\lambda I)^{-1}X^\top y$
- Elastic Net: $\lambda(\alpha\lVert w\rVert_1+(1-\alpha)\lVert w\rVert_2^2)$

## Must-know facts
- **L1** → exact zeros (feature selection); **L2** → shrink all, handles collinearity.
- **Standardize** first; don't penalize the intercept.
- Tune **λ by CV** (log scale).
- sklearn: **`C = 1/λ`** (smaller C = stronger reg).
- Reduces variance, not bias — won't fix underfitting.
- DL analogues: weight decay, dropout, early stopping.

## Quick decisions
| Situation | Do this |
|---|---|
| Many/correlated features | Ridge |
| Want sparsity/selection | Lasso |
| Correlated + want selection | Elastic Net |
| Underfitting | Lower λ / add capacity |
| Neural net | Dropout + weight decay + early stop |

## Common mistakes
- Not scaling features.
- Penalizing the intercept.
- L1 on correlated groups (unstable).
- Misreading sklearn's `C`.

## One-liner code
```python
make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3,3,25))).fit(X_train, y_train)
```
