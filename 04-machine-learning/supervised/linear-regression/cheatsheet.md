# Linear Regression — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Predict a continuous $y$ as $w^\top x + b$; fit by minimizing squared error. Interpretable baseline for regression.

## Key formulas
- Model: $\hat y = w^\top x + b$
- Loss (MSE): $\frac1n\lVert y - Xw\rVert_2^2$
- Normal equations: $\hat w = (X^\top X)^{-1}X^\top y$
- Gradient: $-\frac2n X^\top(y - Xw)$

## Must-know facts
- OLS = MLE under Gaussian noise.
- Under Gauss-Markov assumptions OLS is **BLUE**.
- $w_j$ = change in $y$ per unit $x_j$, others fixed.
- Libraries solve via **QR/SVD**, not literal matrix inversion.
- Ridge/Lasso for many/correlated features.

## Quick decisions
| Situation | Do this |
|---|---|
| Many correlated features | Ridge (L2) |
| Want feature selection | Lasso (L1) |
| Outliers in target | Huber / transform y |
| Non-linear relationship | Add polynomial/log features or use trees |
| $p > n$ or singular $X^\top X$ | Regularize / SVD |

## Common mistakes
- Ignoring multicollinearity (check VIF).
- Reading coefficients as causal.
- Extrapolating beyond the training range.
- Trusting p-values under heteroscedastic residuals.

## One-liner code
```python
LinearRegression().fit(X_train, y_train).predict(X_test)
```
