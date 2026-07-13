# Feature Selection

> Choosing a subset of the most relevant features to improve generalization, cut cost, and aid interpretability — via filter, wrapper, or embedded methods.

| | |
|---|---|
| **Category** | Machine Learning → Advanced |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [Feature Engineering](../../../03-data-analysis/feature-engineering/), [Regularization](../regularization/), [Cross-Validation](../../fundamentals/cross-validation/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
More features aren't always better. Irrelevant or redundant columns add noise, inflate variance, slow training, and make models harder to explain — the "curse of dimensionality". Feature selection keeps the signal-carrying columns and drops the rest. Unlike dimensionality reduction (PCA), which creates new combined axes, selection keeps a subset of the **original, interpretable** features.

## 2. Formal definition / Key concepts
Three families:
- **Filter methods** — score each feature by a statistic independent of any model (correlation, mutual information, χ², ANOVA F, variance threshold). Fast, model-agnostic, ignore interactions.
- **Wrapper methods** — search subsets by training a model and scoring it (forward/backward selection, **Recursive Feature Elimination**). Accurate but expensive and overfitting-prone.
- **Embedded methods** — selection happens *during* training: **L1/Lasso** drives coefficients to zero; tree importances; ElasticNet. Good balance of cost and quality.

Related but different: **dimensionality reduction** (PCA) transforms rather than selects.

## 3. Math
- **Mutual information** between feature $X$ and target $Y$ (captures non-linear dependence):
$$I(X;Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$$
- **Lasso** performs embedded selection via the L1 penalty, whose corner solutions zero out coefficients:
$$\min_w \ \tfrac{1}{n}\lVert y - Xw\rVert_2^2 + \lambda\lVert w\rVert_1$$
- **Pearson correlation** $r$ for linear filter screening.

## 4. How it works
1. Remove trivially useless features (near-zero variance, IDs, duplicates, leaky columns).
2. **Filter** a first pass with a univariate score (fast shortlist).
3. **Embedded** (Lasso / tree importance) or **wrapper** (RFE with CV) to select the working set considering interactions.
4. **Validate inside cross-validation** — the selection step must be refit on each training fold, not on the whole dataset.
5. Confirm the smaller set matches/beats the full set on held-out data; prefer permutation importance for ranking.

## 5. When to use / When not to
- ✅ Many features, especially $p \gg n$ (genomics, text) — reduces overfitting and cost.
- ✅ When interpretability and cheaper inference matter.
- ✅ To remove redundant/collinear or leaky features.
- ❌ When you have few features and lots of data (little to gain).
- ❌ When features are all weakly-but-jointly informative — aggressive pruning can hurt.
- ❌ As a substitute for regularization in deep learning (learned representations handle this).

## 6. Common pitfalls & gotchas
- **Selection leakage** — selecting features on the *entire* dataset before CV/splitting biases scores optimistically. Put selection inside the pipeline/fold.
- **Filter methods miss interactions** — a feature useless alone can be vital in combination.
- **Correlated features** split importance and can be dropped together; check redundancy explicitly.
- **Impurity-based importances are biased** toward high-cardinality features — prefer permutation importance/SHAP.
- **Wrapper search overfits** the validation signal on small data.
- Confusing selection (keep originals) with **extraction/PCA** (new features).

## 7. Code
```python
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Embedded (Lasso) — selection inside a pipeline avoids leakage
sel = make_pipeline(
    StandardScaler(),
    SelectFromModel(LassoCV(cv=5)),      # keep non-zero-coefficient features
)
sel.fit(X_train, y_train)
mask = sel.named_steps["selectfrommodel"].get_support()
print("kept:", X_train.columns[mask].tolist())
```

## 8. Interview / viva questions
- Q: Filter vs wrapper vs embedded — trade-offs?
  - A: Filters are fast and model-agnostic but ignore interactions; wrappers are accurate but costly and overfit-prone; embedded (Lasso/tree) balance cost and quality by selecting during training.
- Q: How does Lasso perform feature selection?
  - A: Its L1 penalty has corner solutions that shrink some coefficients exactly to zero, effectively dropping those features.
- Q: What's the biggest correctness pitfall?
  - A: Selecting features using the whole dataset before splitting/CV — that leaks target info and inflates performance; selection must happen inside each training fold.
- Q: Feature selection vs PCA?
  - A: Selection keeps a subset of original interpretable features; PCA builds new linear combinations, losing interpretability.

## 9. References
- Guyon & Elisseeff (2003) — "An Introduction to Variable and Feature Selection", JMLR.
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 3.3–3.4 (Lasso), 18 (high-dim).
- scikit-learn User Guide: Feature selection.

---
> _Status: 🟢 done._
