# K-Means — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Partition into $k$ clusters by alternating "assign to nearest centroid" and "recompute centroid means" to minimize inertia. Fast; assumes spherical clusters.

## Key formulas
- Inertia: $J=\sum_j\sum_{x\in C_j}\lVert x-\mu_j\rVert^2$
- Centroid: $\mu_j = \frac{1}{|C_j|}\sum_{x\in C_j}x$

## Must-know facts
- Lloyd's algorithm → **local** optimum → use `n_init` restarts.
- **k-means++** for initialization.
- **Scale** features first; Euclidean distance.
- Must choose $k$ (elbow + silhouette).
- Assumes **spherical, equal-size** clusters; sensitive to outliers.

## Quick decisions
| Situation | Do this |
|---|---|
| Non-spherical/varying density | DBSCAN / spectral |
| Huge data | `MiniBatchKMeans` |
| Choose $k$ | Elbow + silhouette |
| Outliers | k-medoids |
| Categorical data | k-modes / k-prototypes |

## Common mistakes
- Not scaling features.
- Single run (no restarts).
- Forcing spherical clusters on non-convex shapes.

## One-liner code
```python
KMeans(n_clusters=4, init="k-means++", n_init=10).fit_predict(StandardScaler().fit_transform(X))
```
