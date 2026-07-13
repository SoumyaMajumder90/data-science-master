# k-Nearest Neighbors (KNN)

> A non-parametric, "lazy" model that predicts a point by looking at its $k$ closest training examples and taking their majority vote (or average).

| | |
|---|---|
| **Category** | Machine Learning → Supervised |
| **Difficulty** | 🟩 Beginner |
| **Prerequisites** | [Descriptive Statistics](../../../03-data-analysis/descriptive-statistics/), [Feature Engineering](../../../03-data-analysis/feature-engineering/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
"You are the average of the people closest to you." To classify a new point, KNN finds the $k$ training examples nearest to it and lets them vote. There's no training in the usual sense — the model just *stores* the data and does all the work at prediction time (hence "lazy"). It assumes that nearby points share labels, so it works when the feature space is meaningful and well-scaled.

## 2. Formal definition / Key concepts
- **Non-parametric, instance-based:** no model is fit; the training set *is* the model.
- **Prediction:** for a query $x$, find the $k$ nearest neighbors by a distance metric and return the majority class (classification) or mean/weighted mean (regression).
- **Distance metric:** Euclidean (L2), Manhattan (L1), Minkowski, cosine; must match the data's geometry.
- **$k$** controls smoothness: small $k$ → flexible, high variance; large $k$ → smooth, high bias.
- **Weighting:** uniform, or by inverse distance so closer neighbors count more.

## 3. Math
Euclidean distance and the $k$-NN classification rule:
$$d(x, x_i) = \sqrt{\sum_{j=1}^{p}(x_j - x_{ij})^2},\qquad \hat y = \arg\max_{c}\sum_{i\in N_k(x)} \mathbb{1}[y_i = c]$$
Distance-weighted vote uses weights $w_i = 1/d(x, x_i)$. As $n\to\infty$ with $k\to\infty$, $k/n\to 0$, the 1-NN error is bounded by twice the Bayes error.

## 4. How it works
1. **Scale** all features (distances are dominated by large-range features otherwise).
2. Store the training data (optionally in a **KD-tree** or **Ball-tree** for faster search).
3. For a query, compute distances, find the $k$ smallest, and aggregate their labels.
4. Choose $k$ by cross-validation (often an odd number for binary classification to avoid ties).

## 5. When to use / When not to
- ✅ Small, low-dimensional datasets with a meaningful distance and locally smooth labels.
- ✅ A quick, assumption-light **baseline**; naturally handles multiclass and complex boundaries.
- ✅ Recommendation/similarity tasks (find nearest items).
- ❌ **High-dimensional** data — the curse of dimensionality makes all points nearly equidistant.
- ❌ Large datasets or low-latency serving — prediction is $O(n)$ per query (or needs ANN indexes).
- ❌ Data with many irrelevant features or mixed scales unless carefully engineered.

## 6. Common pitfalls & gotchas
- **Forgetting to scale** — the single biggest KNN mistake; standardize or normalize first.
- **Curse of dimensionality** — distances become meaningless as $p$ grows; reduce dimensions (PCA) or select features.
- **Choosing $k$ poorly** — too small overfits noise, too large blurs the boundary; tune via CV.
- **Class imbalance** — the majority class dominates votes; use distance weighting or resampling.
- **Expensive prediction & memory** — must store all data; use KD/Ball-tree or approximate NN for scale.
- Irrelevant features add noise to every distance — feature selection helps a lot.

## 7. Code
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(weights="distance"))
grid = GridSearchCV(pipe, {"kneighborsclassifier__n_neighbors": range(1, 31, 2)},
                    cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
print("best k:", grid.best_params_)
```

## 8. Interview / viva questions
- Q: Why is KNN called a lazy learner?
  - A: It does no work at training time — it just stores the data and defers all computation to prediction, where it searches for neighbors.
- Q: How does $k$ affect the bias-variance trade-off?
  - A: Small $k$ = flexible, low bias/high variance; large $k$ = smooth, high bias/low variance.
- Q: Why must you scale features?
  - A: Distance is dominated by features with large ranges, so unscaled features silently dictate the neighbors.
- Q: Why does KNN struggle in high dimensions?
  - A: The curse of dimensionality makes distances between points concentrate, so "nearest" becomes almost meaningless.

## 9. References
- Cover & Hart (1967) — "Nearest Neighbor Pattern Classification", IEEE Trans. Information Theory.
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 13.
- scikit-learn User Guide: Nearest Neighbors.

---
> _Status: 🟢 done._
