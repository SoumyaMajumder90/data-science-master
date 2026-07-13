# DBSCAN

> A density-based clustering algorithm that groups points packed closely together and labels points in sparse regions as noise — finding arbitrary-shaped clusters without specifying $k$.

| | |
|---|---|
| **Category** | Machine Learning → Unsupervised |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [K-Means](../k-means/), [KNN](../../supervised/knn/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
K-means forces every point into a round cluster; DBSCAN instead thinks in terms of **crowds**. A cluster is a region where points are densely packed, and you grow it by connecting neighbors of neighbors. Points in the dense core belong to a cluster, points on the fringe get pulled in, and lonely points in empty regions are flagged as **outliers**. This lets DBSCAN discover clusters shaped like moons, rings, or blobs, and it decides the number of clusters on its own.

## 2. Formal definition / Key concepts
- Two parameters: **eps** ($\varepsilon$, neighborhood radius) and **minPts** (minimum points to be dense).
- **Core point:** has ≥ minPts points within $\varepsilon$ (including itself).
- **Border point:** within $\varepsilon$ of a core point but not itself core.
- **Noise point:** neither — an outlier.
- **Density-reachability / connectivity:** clusters are maximal sets of density-connected points.
- Finds clusters of **arbitrary shape** and **any number**; noise is a first-class output.

## 3. Math
The $\varepsilon$-neighborhood of a point $p$:
$$N_\varepsilon(p) = \{q : d(p, q) \le \varepsilon\}$$
$p$ is a **core point** if $|N_\varepsilon(p)| \ge \text{minPts}$. A point $q$ is *directly density-reachable* from core $p$ if $q \in N_\varepsilon(p)$; a cluster is the transitive closure of this relation. Heuristics: **minPts ≈ 2·dim**; choose $\varepsilon$ at the "knee" of the sorted **k-distance plot** ($k=$ minPts).

## 4. How it works
1. For each unvisited point, retrieve its $\varepsilon$-neighborhood.
2. If it's a **core point**, start a cluster and add all density-reachable points (expanding through other cores).
3. **Border points** join the cluster that reached them; **non-reachable** points are labeled noise (may be reclaimed later if reached).
4. Continue until all points are visited.
5. Output: cluster labels plus a `-1` label for noise. (Complexity $O(n\log n)$ with a spatial index, $O(n^2)$ worst case.)

## 5. When to use / When not to
- ✅ Arbitrary-shaped clusters (spatial data, geolocation, moons/rings).
- ✅ You want **automatic outlier detection** and don't know $k$.
- ✅ Clusters of roughly uniform density with clear separation.
- ❌ **Varying-density** clusters — a single $\varepsilon$ can't fit all; use **HDBSCAN** or OPTICS.
- ❌ **High-dimensional** data — distances concentrate, making $\varepsilon$ hard to set.
- ❌ Very large datasets without a spatial index; or when you need every point assigned to a cluster.

## 6. Common pitfalls & gotchas
- **Parameter sensitivity** — results hinge on $\varepsilon$; too small → everything is noise, too large → one giant cluster. Use the k-distance elbow.
- **Varying densities** break the single-$\varepsilon$ assumption — switch to HDBSCAN.
- **Curse of dimensionality** — Euclidean $\varepsilon$ loses meaning in high dimensions.
- **Not scaling features** distorts the neighborhood — standardize first.
- **Border points are ambiguous** — assignment can depend on processing order.
- It doesn't force all points into clusters; noise is expected, not a bug.

## 7. Code
```python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

Xs = StandardScaler().fit_transform(X)

# pick eps from the k-distance elbow (k = minPts)
dist, _ = NearestNeighbors(n_neighbors=5).fit(Xs).kneighbors(Xs)
kdist = np.sort(dist[:, -1])          # plot kdist and read the knee

db = DBSCAN(eps=0.5, min_samples=5).fit(Xs)
labels = db.labels_                    # -1 == noise/outlier
print("clusters:", len(set(labels)) - (1 if -1 in labels else 0))
```

## 8. Interview / viva questions
- Q: What are DBSCAN's two parameters and what do they mean?
  - A: $\varepsilon$ (neighborhood radius) and minPts (density threshold); a point is a core if ≥ minPts points lie within $\varepsilon$.
- Q: How is it better than k-means?
  - A: It finds arbitrary-shaped clusters, doesn't need $k$, and labels outliers as noise instead of forcing them into a cluster.
- Q: How do you choose $\varepsilon$?
  - A: Plot the sorted distance to the minPts-th nearest neighbor and pick the value at the "elbow/knee".
- Q: What's DBSCAN's main weakness and the fix?
  - A: It can't handle clusters of varying density with a single $\varepsilon$; HDBSCAN or OPTICS address that.

## 9. References
- Ester, Kriegel, Sander, Xu (1996) — "A Density-Based Algorithm for Discovering Clusters…", KDD.
- Campello, Moulavi, Sander (2013) — HDBSCAN.
- scikit-learn User Guide: Clustering — DBSCAN.

---
> _Status: 🟢 done._
