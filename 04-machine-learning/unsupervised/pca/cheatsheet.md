# PCA — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Rotate data onto orthogonal axes ordered by variance; keep the top few. Linear, unsupervised dimensionality reduction.

## Key formulas
- Covariance: $\Sigma=\frac{1}{n-1}X^\top X$; eigen: $\Sigma v_k=\lambda_k v_k$
- Explained variance ratio: $\lambda_k/\sum_j\lambda_j$
- SVD: $X=USV^\top$, PCs = columns of $V$, $\lambda_k=s_k^2/(n-1)$
- Projection: $Z = X V_q$

## Must-know facts
- PCs = eigenvectors of covariance = right singular vectors (SVD).
- **Standardize** before PCA; **fit on train only** (leakage).
- PCs are **orthogonal** but **not interpretable**.
- Variance ≠ predictive importance (unsupervised).
- Sensitive to outliers; sign is arbitrary.

## Quick decisions
| Situation | Do this |
|---|---|
| Choose #components | 90–95% cumulative EVR / scree elbow |
| Non-linear structure | t-SNE / UMAP / kernel PCA |
| Need interpretability | Skip PCA (or use feature selection) |
| Different units | Standardize first |

## Common mistakes
- Not scaling features.
- Fitting PCA on all data before the split.
- Assuming low-variance PCs are useless for the target.

## One-liner code
```python
make_pipeline(StandardScaler(), PCA(n_components=0.95)).fit_transform(X_train)
```
