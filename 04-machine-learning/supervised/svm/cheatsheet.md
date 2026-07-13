# Support Vector Machines — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Find the max-margin boundary defined by support vectors; kernels give non-linear boundaries. Great in high-dim, poor at huge $n$.

## Key formulas
- Margin $= 2/\lVert w\rVert$
- Primal: $\min \tfrac12\lVert w\rVert^2 + C\sum_i\xi_i$
- RBF: $K(x,x')=\exp(-\gamma\lVert x-x'\rVert^2)$
- Hinge loss: $\max(0, 1-y\,f(x))$ + L2

## Must-know facts
- Only **support vectors** define the boundary.
- **Standardize** features first (kernels are scale-sensitive).
- $C$ = regularization; RBF $\gamma$ = influence radius — both large ⇒ overfit.
- Linear SVM for text/high-dim; RBF for general non-linear.
- No native probabilities (Platt scaling); scales badly with $n$.

## Quick decisions
| Situation | Do this |
|---|---|
| High-dim/sparse (text) | Linear SVM / `LinearSVC` |
| Non-linear, medium data | RBF kernel, tune $C,\gamma$ |
| Millions of rows | Linear SVM (SGD) or trees |
| Imbalance | `class_weight="balanced"` |
| Need probabilities | `probability=True` (slow) or calibrate |

## Common mistakes
- Not scaling features.
- Tuning $C$ and $\gamma$ separately.
- Using kernel SVM on huge datasets.

## One-liner code
```python
make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")).fit(X, y)
```
