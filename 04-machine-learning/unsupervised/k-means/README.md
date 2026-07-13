# K-Means Clustering

> An unsupervised algorithm that partitions data into $k$ clusters by iteratively assigning points to the nearest centroid and recomputing centroids to minimize within-cluster variance.

| | |
|---|---|
| **Category** | Machine Learning → Unsupervised |
| **Difficulty** | 🟩 Beginner / 🟨 Intermediate |
| **Prerequisites** | [Descriptive Statistics](../../../03-data-analysis/descriptive-statistics/), [PCA](../pca/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
You have a cloud of points and want to group them into $k$ blobs, but nobody labeled them. Drop $k$ "flags" (centroids) at random, assign each point to its nearest flag, then move each flag to the center of the points it caught. Repeat: reassign, recenter, reassign… The flags settle into the natural clumps. K-means is that simple loop, and it's the most widely used clustering method because it's fast and intuitive.

## 2. Formal definition / Key concepts
- **Goal:** partition $n$ points into $k$ clusters minimizing **within-cluster sum of squares (WCSS / inertia)**.
- **Centroid:** the mean of the points in a cluster.
- **Lloyd's algorithm:** alternating *assignment* and *update* steps — a form of coordinate descent that converges to a **local** optimum.
- **k-means++**: smart seeding that spreads initial centroids apart for better, more stable results.
- Assumes **spherical, similarly-sized, convex** clusters and uses Euclidean distance.

## 3. Math
Objective (inertia):
$$J = \sum_{j=1}^{k}\sum_{x\in C_j}\lVert x - \mu_j\rVert^2,\qquad \mu_j = \frac{1}{|C_j|}\sum_{x\in C_j} x$$
Each iteration provably does not increase $J$:
- **Assignment:** $C_j = \{x : j = \arg\min_l \lVert x-\mu_l\rVert^2\}$
- **Update:** $\mu_j \leftarrow$ mean of $C_j$

Minimizing squared Euclidean distance is why the optimal center is the mean.

## 4. How it works
1. Choose $k$; initialize centroids with **k-means++**.
2. **Assign** each point to its nearest centroid.
3. **Update** each centroid to the mean of its assigned points.
4. Repeat 2–3 until assignments stop changing (or max iterations).
5. Because it finds a local optimum, run **several restarts** (`n_init`) and keep the lowest inertia.
6. Pick $k$ via the **elbow method** (inertia vs $k$) or **silhouette score**.

## 5. When to use / When not to
- ✅ Large datasets needing fast, scalable clustering (`MiniBatchKMeans` scales further).
- ✅ Roughly spherical, well-separated, comparably sized clusters.
- ✅ Vector quantization, image color compression, customer segmentation, feature engineering.
- ❌ Non-convex / elongated / varying-density clusters → use **DBSCAN** or spectral clustering.
- ❌ When $k$ is genuinely unknown and no clear elbow exists.
- ❌ Categorical data (means undefined) → use k-modes / k-prototypes.

## 6. Common pitfalls & gotchas
- **Must pick $k$ in advance** — it won't find the "right" number itself.
- **Sensitive to initialization** — always use k-means++ and multiple `n_init` restarts.
- **Not scaling features** lets large-range features dominate distances — standardize first.
- **Assumes spherical, equal-size clusters** — fails on moons/rings/varying densities.
- **Sensitive to outliers** because the mean is not robust; consider k-medoids.
- **Local optima** — the result can vary run to run without enough restarts.
- The elbow can be ambiguous; corroborate with silhouette.

## 7. Code
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

Xs = StandardScaler().fit_transform(X)   # scale first
km = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=0)
labels = km.fit_predict(Xs)

print("inertia:", km.inertia_)
print("silhouette:", silhouette_score(Xs, labels))
```

## 8. Interview / viva questions
- Q: What does k-means optimize, and does it find the global optimum?
  - A: It minimizes within-cluster sum of squares (inertia) via Lloyd's algorithm, which only guarantees a local optimum — hence multiple restarts.
- Q: Why do we use k-means++ initialization?
  - A: Random seeds can give poor, unstable clusters; k-means++ spreads initial centroids apart, improving quality and convergence.
- Q: How do you choose $k$?
  - A: The elbow method (inertia vs $k$) and silhouette score; also domain knowledge and stability across runs.
- Q: When does k-means fail?
  - A: On non-spherical, unequal-size, or varying-density clusters, with outliers, or on unscaled/categorical features.

## 9. References
- MacQueen (1967) — original k-means; Lloyd (1982) — the algorithm.
- Arthur & Vassilvitskii (2007) — "k-means++: The Advantages of Careful Seeding".
- scikit-learn User Guide: Clustering — K-means.

---
> _Status: 🟢 done._
