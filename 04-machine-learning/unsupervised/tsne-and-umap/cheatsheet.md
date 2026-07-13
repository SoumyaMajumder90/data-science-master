# t-SNE and UMAP — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Non-linear reducers that preserve *local* neighborhoods for 2–3D visualization. Great for seeing structure; don't measure the map.

## Key formulas
- t-SNE high-D: $p_{j|i}\propto\exp(-\lVert x_i-x_j\rVert^2/2\sigma_i^2)$
- t-SNE low-D (Student-t): $q_{ij}\propto(1+\lVert y_i-y_j\rVert^2)^{-1}$
- Objective: minimize $\mathrm{KL}(P\Vert Q)$ (t-SNE) / cross-entropy (UMAP)

## Must-know facts
- Preserve **local** structure, distort **global** distances/sizes.
- t-SNE knob = **perplexity** (~5–50); UMAP = **n_neighbors**, **min_dist**.
- **PCA-preprocess** (~50 dims) first.
- t-SNE: slow, no out-of-sample transform. UMAP: faster, can `transform`.
- Use PCA/spectral **init** + fixed seed for stability.

## Quick decisions
| Situation | Do this |
|---|---|
| Just visualize | t-SNE or UMAP |
| Need speed/scale/new points | UMAP |
| Global geometry matters | PCA instead |
| Features for a model | UMAP/PCA, not t-SNE |

## Common mistakes
- Reading cluster sizes / gaps as meaningful.
- One perplexity/seed only.
- Skipping PCA preprocessing on high-D data.

## One-liner code
```python
TSNE(n_components=2, perplexity=30, init="pca").fit_transform(PCA(50).fit_transform(X))
```
