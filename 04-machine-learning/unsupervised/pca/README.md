# Principal Component Analysis (PCA)

> A linear dimensionality-reduction technique that rotates the data onto new orthogonal axes ordered by variance, so you can keep the few directions that capture most of the information.

| | |
|---|---|
| **Category** | Machine Learning → Unsupervised |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [Linear Algebra](../../../01-foundations/mathematics/linear-algebra/), [Descriptive Statistics](../../../03-data-analysis/descriptive-statistics/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
Imagine a flat, tilted pancake of points floating in 3D. Most of the spread lies in the plane of the pancake; the thickness carries almost no information. PCA finds that plane automatically: it looks for the direction of **greatest variance**, then the next-greatest direction perpendicular to it, and so on. Projecting onto the first couple of these directions compresses the data with minimal loss — useful for visualization, denoising, and speeding up downstream models.

## 2. Formal definition / Key concepts
- **Principal components (PCs):** orthogonal directions (eigenvectors of the covariance matrix) ordered by the variance they capture (eigenvalues).
- **Loadings:** the weights defining each PC in terms of original features.
- **Scores:** the data projected onto the PCs (the new coordinates).
- **Explained variance ratio:** fraction of total variance each PC accounts for.
- Equivalent to the **SVD** of the centered data matrix; PCA is a *linear*, unsupervised, variance-maximizing projection.

## 3. Math
Center the data, form the covariance matrix $\Sigma = \tfrac{1}{n-1}X^\top X$, and eigendecompose:
$$\Sigma v_k = \lambda_k v_k$$
The $k$-th PC is eigenvector $v_k$; it captures variance $\lambda_k$. Explained variance ratio:
$$\text{EVR}_k = \frac{\lambda_k}{\sum_j \lambda_j}$$
Via **SVD** of centered $X = U S V^\top$: columns of $V$ are the PCs, and $\lambda_k = s_k^2/(n-1)$. Projection onto the top $q$ PCs: $Z = X V_q$.

## 4. How it works
1. **Standardize** features (mean-center always; scale to unit variance if units differ).
2. Compute the covariance matrix (or run SVD directly on the centered data — more stable).
3. Get eigenvectors/eigenvalues (PCs and their variances), sorted descending.
4. Choose $q$ components (e.g. enough for 90–95% cumulative variance, or via a scree-plot elbow).
5. **Project** data onto the top $q$ PCs to get the reduced representation.

## 5. When to use / When not to
- ✅ **Dimensionality reduction** before modeling to cut noise, collinearity, and compute.
- ✅ **Visualization** of high-dimensional data in 2–3D (as a first look).
- ✅ **Decorrelating** features, compression, denoising, whitening.
- ❌ When you need **interpretable** features — PCs are opaque linear mixes.
- ❌ **Non-linear** manifolds — use t-SNE/UMAP or kernel PCA/autoencoders.
- ❌ When variance ≠ importance (a low-variance feature can still be predictive) — PCA is unsupervised and ignores the target.

## 6. Common pitfalls & gotchas
- **Not scaling** — PCA is dominated by high-variance/large-unit features; standardize first (unless units are already comparable).
- **Fit on the full dataset before splitting** = leakage; fit PCA on train only.
- **Variance ≠ predictive power** — discarding low-variance PCs can drop signal that matters for the target.
- **PCs aren't interpretable** — don't over-read loadings as "meanings".
- **Sign/rotation ambiguity** — eigenvector signs are arbitrary and can flip between runs.
- Sensitive to **outliers** (variance-based); consider robust PCA.

## 7. Code
```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), PCA(n_components=0.95))  # keep 95% variance
Z = pipe.fit_transform(X_train)                                 # fit on train only

pca = pipe.named_steps["pca"]
print("components kept:", pca.n_components_)
print("cumulative EVR:", np.cumsum(pca.explained_variance_ratio_))
```

## 8. Interview / viva questions
- Q: What does PCA maximize?
  - A: The variance captured by each successive orthogonal component — equivalently it minimizes the reconstruction (projection) error.
- Q: What's the relationship between PCA and SVD?
  - A: PCA is the SVD of the centered data matrix; the right singular vectors are the principal components and squared singular values give the variances.
- Q: Why standardize before PCA?
  - A: Otherwise features with larger scales dominate the variance and hijack the components.
- Q: How do you choose the number of components?
  - A: Cumulative explained variance (e.g. 90–95%) or the elbow of a scree plot, balanced against downstream performance.
- Q: When is PCA a bad idea?
  - A: For non-linear structure, when interpretability is required, or when low-variance directions carry the predictive signal.

## 9. References
- Jolliffe — *Principal Component Analysis* (2nd ed.).
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 14.5.
- scikit-learn User Guide: Decomposition — PCA.

---
> _Status: 🟢 done._
