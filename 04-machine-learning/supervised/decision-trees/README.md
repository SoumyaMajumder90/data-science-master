# Decision Trees

> A model that predicts by asking a sequence of yes/no questions on features, recursively splitting the data into ever-purer regions.

| | |
|---|---|
| **Category** | Machine Learning → Supervised |
| **Difficulty** | 🟩 Beginner / 🟨 Intermediate |
| **Prerequisites** | [Information Theory](../../../01-foundations/mathematics/information-theory/), [Overfitting & Underfitting](../../fundamentals/overfitting-underfitting/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
A decision tree is a flowchart of questions: "Is age > 30? If yes, is income > 50k? …". Each question splits the data; you keep splitting until each leaf is mostly one class (or has a stable average). Prediction just means following the questions down to a leaf. It mirrors how humans make decisions and needs no feature scaling — but left unchecked it will memorize the training set.

## 2. Formal definition / Key concepts
- A **binary tree** where each internal node tests one feature against a threshold, and each leaf holds a prediction (majority class or mean target).
- Built **greedily, top-down** (CART): at each node pick the split that most reduces impurity.
- **Impurity measures:** Gini and entropy (classification); variance/MSE (regression).
- **Axis-aligned** splits → decision boundaries are staircases of rectangles.
- Controlled by **hyperparameters**: `max_depth`, `min_samples_leaf`, `min_samples_split`, `ccp_alpha` (pruning).

## 3. Math
Node impurity for classification with class proportions $p_k$:
$$\text{Gini} = 1 - \sum_k p_k^2,\qquad \text{Entropy} = -\sum_k p_k \log_2 p_k$$
A split is chosen to maximize the **impurity decrease** (information gain):
$$\Delta I = I(\text{parent}) - \sum_{c\in\{L,R\}} \frac{n_c}{n}\, I(c)$$
For regression, impurity is the node variance and the split minimizes total child MSE.

## 4. How it works
1. Start with all data at the root.
2. For every feature and candidate threshold, compute the weighted child impurity.
3. Pick the split with the largest impurity decrease; partition the data.
4. Recurse on each child until a stopping rule (max depth, min samples, no gain).
5. Optionally **prune** back (cost-complexity, `ccp_alpha`) to reduce overfitting.
6. Predict by routing a sample to its leaf and returning the leaf's majority class / mean.

## 5. When to use / When not to
- ✅ Need an **interpretable**, rule-based model you can visualize and explain.
- ✅ Mixed numeric + categorical features, non-linear relationships, no scaling required.
- ✅ As the base learner for ensembles (random forests, boosting).
- ❌ As a standalone high-accuracy model — single trees are high-variance; prefer ensembles.
- ❌ Smooth/linear relationships or extrapolation — the staircase boundary is a poor fit.

## 6. Common pitfalls & gotchas
- **Overfitting** — an unconstrained tree grows until leaves are pure; always limit depth/leaf size or prune.
- **High variance / instability** — a small data change can reshape the whole tree.
- **Biased impurity-based importances** favor high-cardinality features; prefer permutation importance.
- **Cannot extrapolate** — regression trees predict a constant outside seen ranges.
- **Greedy** splitting is locally, not globally, optimal.
- Class imbalance skews splits; use `class_weight`.

## 7. Code
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

clf = DecisionTreeClassifier(
    max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=0)
clf.fit(X_train, y_train)

plt.figure(figsize=(12, 6))
plot_tree(clf, feature_names=feature_names, filled=True)   # visualize the rules
print("importances:", dict(zip(feature_names, clf.feature_importances_)))
```

## 8. Interview / viva questions
- Q: Gini vs entropy — do they matter?
  - A: Both measure impurity and usually give near-identical trees; Gini is slightly cheaper (no log). The bigger levers are depth and pruning.
- Q: Why do single trees overfit and how do you prevent it?
  - A: They grow until leaves are pure, memorizing noise; limit `max_depth`/`min_samples_leaf` or apply cost-complexity pruning.
- Q: Why are trees called high-variance?
  - A: Small perturbations in the data can change split choices and produce a very different tree — which is exactly why bagging/random forests help.
- Q: How does a regression tree predict?
  - A: It routes the point to a leaf and returns the mean target of the training samples in that leaf (a piecewise-constant function).

## 9. References
- Breiman, Friedman, Olshen, Stone — *Classification and Regression Trees* (CART).
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 9.2.
- scikit-learn User Guide: Decision Trees.

---
> _Status: 🟢 done._
