# k-Nearest Neighbors — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Predict from the $k$ closest training points (vote or average). No training, lazy, distance-based — scale features and mind high dimensions.

## Key formulas
- Euclidean: $d(x,x_i)=\sqrt{\sum_j (x_j-x_{ij})^2}$
- Rule: $\hat y = \arg\max_c \sum_{i\in N_k(x)}\mathbb 1[y_i=c]$
- Distance weight: $w_i = 1/d(x,x_i)$

## Must-know facts
- **Non-parametric, lazy**: training set is the model.
- **Always scale** features.
- Small $k$ = high variance; large $k$ = high bias.
- Prediction is $O(n)$ per query → use KD/Ball-tree or ANN.
- Fails in **high dimensions** (curse of dimensionality).

## Quick decisions
| Situation | Do this |
|---|---|
| Different feature scales | Standardize |
| High dimensionality | PCA / feature selection first |
| Imbalance | `weights="distance"` or resample |
| Large data | Approximate NN index |
| Pick $k$ | Cross-validation (odd for binary) |

## Common mistakes
- Not scaling features.
- Using KNN on many-dimensional data.
- Choosing $k$ without CV.

## One-liner code
```python
make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")).fit(X, y)
```
