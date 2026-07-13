# t-SNE and UMAP

> Non-linear dimensionality-reduction methods that embed high-dimensional data into 2–3D for visualization by preserving local neighborhood structure.

| | |
|---|---|
| **Category** | Machine Learning → Unsupervised |
| **Difficulty** | 🟥 Advanced |
| **Prerequisites** | [PCA](../pca/), [KNN](../../supervised/knn/), [Information Theory](../../../01-foundations/mathematics/information-theory/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
PCA can only rotate and flatten — it misses curved structure. t-SNE and UMAP instead ask: "which points are *neighbors* in high-D?" and then arrange points in 2D so those neighborhoods are preserved, letting tangled manifolds unfold into visible clusters. They're the go-to tools for *seeing* the structure of embeddings, gene expression, or image features — but the pictures are for exploration, not measurement.

## 2. Formal definition / Key concepts
- **t-SNE (t-distributed Stochastic Neighbor Embedding):** converts pairwise distances into probabilities and matches high-D and low-D neighbor distributions by minimizing KL divergence; uses a heavy-tailed (Student-t) kernel in low-D to avoid crowding.
- **UMAP (Uniform Manifold Approximation and Projection):** builds a fuzzy $k$-nearest-neighbor graph and optimizes a low-D layout to preserve it via cross-entropy; grounded in manifold/topology theory.
- Both are **non-linear**, **unsupervised**, primarily for **visualization**; neither preserves global distances reliably.
- Key knobs: t-SNE **perplexity** (~5–50), UMAP **n_neighbors** and **min_dist**.

## 3. Math
**t-SNE** high-D affinities and low-D affinities:
$$p_{j\mid i} \propto \exp\!\left(-\lVert x_i - x_j\rVert^2 / 2\sigma_i^2\right),\qquad
q_{ij} \propto \left(1 + \lVert y_i - y_j\rVert^2\right)^{-1}$$
minimizing $\text{KL}(P\Vert Q) = \sum_{i\ne j} p_{ij}\log\frac{p_{ij}}{q_{ij}}$ by gradient descent. **UMAP** minimizes the cross-entropy between the high-D and low-D fuzzy neighbor graphs; the heavy tail / repulsion is controlled by `min_dist`.

## 4. How it works
1. Optionally **pre-reduce** with PCA (e.g. to 50 dims) to denoise and speed things up.
2. Build local neighbor relationships (perplexity for t-SNE; `n_neighbors` graph for UMAP).
3. Initialize the low-D layout (PCA/spectral init is more stable than random).
4. **Optimize** the embedding by gradient descent (t-SNE: KL; UMAP: cross-entropy).
5. Read the 2D map to spot clusters — but treat cluster sizes and gaps skeptically.

## 5. When to use / When not to
- ✅ **Visualizing** high-dimensional data and embeddings in 2–3D.
- ✅ Exploratory analysis: spotting clusters, batch effects, class separability.
- ✅ **UMAP** when you want speed, scalability, and the option to `transform` new data / feed features downstream.
- ❌ As input features for models without care (t-SNE especially isn't a general-purpose reducer) — prefer PCA/UMAP.
- ❌ When you need to interpret **distances, densities, or global geometry** — these are not faithfully preserved.
- ❌ Quantitative claims from the picture (cluster size ≠ importance).

## 6. Common pitfalls & gotchas
- **Cluster sizes and inter-cluster distances are not meaningful** — t-SNE equalizes densities; don't measure the map.
- **Perplexity / n_neighbors matter a lot** — too small fragments data, too large washes out structure; try several.
- **Random seed changes the layout** — run a few; use fixed seeds for reproducibility.
- **t-SNE doesn't scale** and has no natural out-of-sample transform; UMAP is faster and can embed new points.
- **PCA-preprocess first** for high-D inputs (stability + speed).
- Reading "gaps" as hard separations or apparent clusters in random noise are classic misinterpretations.

## 7. Code
```python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# import umap    # pip install umap-learn

X50 = PCA(n_components=50).fit_transform(X)      # denoise/speed-up first

emb_tsne = TSNE(n_components=2, perplexity=30, init="pca",
                random_state=0).fit_transform(X50)

# emb_umap = umap.UMAP(n_neighbors=15, min_dist=0.1,
#                      random_state=0).fit_transform(X50)
```

## 8. Interview / viva questions
- Q: How do t-SNE/UMAP differ from PCA?
  - A: PCA is linear and preserves global variance; t-SNE/UMAP are non-linear and preserve *local* neighborhoods, revealing curved structure but distorting global geometry.
- Q: Why shouldn't you interpret cluster sizes or distances in a t-SNE plot?
  - A: t-SNE normalizes local densities and uses a heavy-tailed kernel, so it warps sizes and between-cluster distances — only local neighbor structure is trustworthy.
- Q: What does perplexity control?
  - A: The effective number of neighbors each point considers; it balances attention to local vs broader structure.
- Q: When prefer UMAP over t-SNE?
  - A: When you need speed, scalability, better global structure, or the ability to transform new points and use the embedding downstream.

## 9. References
- van der Maaten & Hinton (2008) — "Visualizing Data using t-SNE", JMLR.
- McInnes, Healy, Melville (2018) — "UMAP: Uniform Manifold Approximation and Projection".
- Wattenberg et al. (2016) — "How to Use t-SNE Effectively", Distill.

---
> _Status: 🟢 done._
