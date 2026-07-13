# Hierarchical Clustering

> Builds a tree of nested clusters (a dendrogram) by repeatedly merging the closest clusters (agglomerative) or splitting them (divisive) — no need to pre-specify $k$.

| | |
|---|---|
| **Category** | Machine Learning → Unsupervised |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [K-Means](../k-means/), [Descriptive Statistics](../../../03-data-analysis/descriptive-statistics/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
Instead of committing to a fixed number of groups, hierarchical clustering builds a family tree of the data. Start with every point as its own cluster, then repeatedly glue together the two nearest clusters, recording each merge. The result is a **dendrogram** — a tree you can "cut" at any height to get however many clusters you want. It reveals structure at *all* scales, from tight sub-groups to broad super-groups.

## 2. Formal definition / Key concepts
- **Agglomerative (bottom-up):** start with $n$ singletons, merge the closest pair each step until one cluster remains. (Divisive is top-down, rarer.)
- **Dendrogram:** the merge tree; the y-axis is the distance at which clusters joined.
- **Linkage** = how you measure distance *between clusters*:
  - **Single** — nearest pair (can "chain").
  - **Complete** — farthest pair (compact clusters).
  - **Average** — mean pairwise distance.
  - **Ward** — merge that minimizes the increase in within-cluster variance (most popular, k-means-like).
- Requires a **distance metric** (Euclidean, cosine, etc.); Ward needs Euclidean.

## 3. Math
Given a pairwise distance $d(a,b)$, the cluster-to-cluster distances are:
$$
\text{single} = \min_{a\in A, b\in B} d(a,b),\quad
\text{complete} = \max_{a\in A, b\in B} d(a,b),\quad
\text{average} = \frac{1}{|A||B|}\sum_{a\in A}\sum_{b\in B} d(a,b)
$$
Ward's criterion merges the pair minimizing the rise in total within-cluster sum of squares; the Lance–Williams recurrence updates distances after each merge efficiently.

## 4. How it works
1. Compute the pairwise distance matrix.
2. Treat each point as its own cluster.
3. **Merge** the two clusters with the smallest linkage distance; record the merge height.
4. Update distances from the new cluster to all others (Lance–Williams).
5. Repeat until one cluster remains → the full dendrogram.
6. **Cut** the tree at a chosen height or target number of clusters to get flat labels.

## 5. When to use / When not to
- ✅ You don't know $k$ and want to explore structure at multiple granularities.
- ✅ Small-to-medium datasets; you want an interpretable dendrogram.
- ✅ Non-globular clusters (with appropriate linkage) and arbitrary distance metrics (e.g. cosine on text).
- ❌ **Large datasets** — building the distance matrix is $O(n^2)$ memory and $O(n^2\log n)$–$O(n^3)$ time.
- ❌ When you need a fast, streaming, or online method → use k-means / MiniBatch.
- ❌ Very high-dimensional data where distances degrade.

## 6. Common pitfalls & gotchas
- **$O(n^2)$ memory** makes it infeasible for large $n$ — sample or use k-means.
- **Linkage choice changes everything** — single linkage chains clusters together; Ward/complete give compact clusters.
- **Greedy and irreversible** — a bad early merge can't be undone.
- **Not scaling features** distorts distances — standardize first.
- **Ward requires Euclidean** distance; pairing it with cosine is invalid.
- Reading a dendrogram: cut height is a *choice*; the longest vertical gap suggests a natural number of clusters.

## 7. Code
```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

Xs = StandardScaler().fit_transform(X)
Z = linkage(Xs, method="ward")          # merge tree

dendrogram(Z, truncate_mode="level", p=5)
plt.show()

labels = fcluster(Z, t=4, criterion="maxclust")   # cut into 4 clusters
```

## 8. Interview / viva questions
- Q: How is hierarchical clustering different from k-means?
  - A: It builds a full tree of nested clusters without pre-specifying $k$ and is deterministic, but scales poorly ($O(n^2)$) compared with k-means.
- Q: What is linkage and why does it matter?
  - A: It defines distance between clusters (single/complete/average/Ward); it controls cluster shape — single linkage chains, Ward makes compact, variance-minimizing clusters.
- Q: What does the height in a dendrogram represent?
  - A: The distance (or variance increase) at which two clusters were merged; large vertical gaps hint at a natural cluster count.
- Q: When would you not use it?
  - A: On large datasets, because of quadratic memory/time; or when you need online/streaming clustering.

## 9. References
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 14.3.12.
- Ward (1963) — "Hierarchical Grouping to Optimize an Objective Function".
- SciPy documentation: `scipy.cluster.hierarchy`.

---
> _Status: 🟢 done._
