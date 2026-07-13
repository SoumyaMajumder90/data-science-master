# Feature Selection — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Keep the signal-carrying original features. Filter (fast stats), wrapper (search + model), embedded (Lasso/tree). Select inside CV.

## Key formulas
- Mutual info: $I(X;Y)=\sum p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$
- Lasso: $\min_w \frac1n\lVert y-Xw\rVert_2^2 + \lambda\lVert w\rVert_1$

## Must-know facts
- **Filter**: correlation/MI/χ²/ANOVA — fast, ignores interactions.
- **Wrapper**: RFE/forward-backward — accurate, costly, overfits.
- **Embedded**: L1/tree importance — best balance.
- **Selection must live inside the CV/pipeline** (else leakage).
- Selection ≠ PCA (keeps originals vs new axes).

## Quick decisions
| Situation | Method |
|---|---|
| Quick shortlist | Filter (MI / F-test) |
| Want interactions | Embedded (Lasso) / RFECV |
| Sparse linear model | Lasso |
| Rank importance | Permutation importance / SHAP |

## Common mistakes
- Selecting on the full dataset before splitting (leakage).
- Trusting impurity importances / correlated features.
- Filters dropping jointly-useful features.

## One-liner code
```python
make_pipeline(StandardScaler(), SelectFromModel(LassoCV(cv=5))).fit(X_train, y_train)
```
