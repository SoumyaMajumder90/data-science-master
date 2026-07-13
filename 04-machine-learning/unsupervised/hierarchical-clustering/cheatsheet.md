# Hierarchical Clustering — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Merge closest clusters into a dendrogram (agglomerative); cut the tree to get $k$. No need to pre-set $k$, but $O(n^2)$.

## Key formulas
- Single: $\min d(a,b)$; Complete: $\max d(a,b)$; Average: mean pairwise.
- **Ward**: minimize increase in within-cluster SS (needs Euclidean).

## Must-know facts
- **Agglomerative** = bottom-up; dendrogram height = merge distance.
- **Linkage** dictates cluster shape (single chains; Ward/complete compact).
- Deterministic, no $k$ needed — but **$O(n^2)$ memory**.
- **Scale** features first.
- Greedy: merges are irreversible.

## Quick decisions
| Situation | Do this |
|---|---|
| Compact, k-means-like | Ward linkage |
| Elongated clusters | Single linkage (careful: chaining) |
| Text/cosine distance | Average/complete (not Ward) |
| Large data | Sample or use k-means |
| Pick $k$ | Cut at longest dendrogram gap |

## Common mistakes
- Using it on large datasets.
- Pairing Ward with non-Euclidean distance.
- Not scaling features.

## One-liner code
```python
fcluster(linkage(StandardScaler().fit_transform(X), "ward"), t=4, criterion="maxclust")
```
