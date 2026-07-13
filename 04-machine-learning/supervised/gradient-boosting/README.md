# Gradient Boosting

> An ensemble that builds shallow trees **sequentially**, each new tree fitting the residual errors (negative gradient) of the current model, to reduce bias step by step.

| | |
|---|---|
| **Category** | Machine Learning → Supervised |
| **Difficulty** | 🟨 Intermediate / 🟥 Advanced |
| **Prerequisites** | [Decision Trees](../decision-trees/), [Optimization](../../../01-foundations/mathematics/optimization/), [Ensemble Methods](../../advanced/ensemble-methods/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
Instead of averaging many independent trees (random forest), boosting works like a team correcting each other in turn. Start with a rough guess. Look at where it's wrong, and train a small tree that *specializes in those mistakes*. Add a shrunken version of it to the model. Repeat. Each round nudges the predictions toward the truth. Because it directly attacks residual error, boosting typically achieves the **highest accuracy on tabular data** — at the cost of careful tuning.

## 2. Formal definition / Key concepts
- **Additive model:** $F_M(x) = \sum_{m=1}^{M} \nu\, h_m(x)$, built stagewise.
- Each **base learner** $h_m$ is a shallow "weak" regression tree fit to the **negative gradient** of the loss at the current predictions (gradient descent in function space).
- **Learning rate / shrinkage** $\nu$ scales each tree's contribution; smaller $\nu$ needs more trees but generalizes better.
- **Regularization:** tree depth, `subsample` (stochastic GB), L1/L2 on leaf weights, early stopping.
- Works for any differentiable loss (squared error, log loss, quantile, etc.).

## 3. Math
Given loss $L$, at stage $m$ compute pseudo-residuals (negative gradient):
$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$
Fit tree $h_m$ to $\{(x_i, r_{im})\}$, find leaf values by a line search minimizing $L$, then update:
$$F_m(x) = F_{m-1}(x) + \nu\, h_m(x)$$
For squared-error loss the pseudo-residual is exactly the ordinary residual $y_i - F_{m-1}(x_i)$.

## 4. How it works
1. Initialize $F_0$ with a constant (e.g. the mean / log-odds).
2. For $m = 1..M$: compute pseudo-residuals; fit a shallow tree to them; set leaf outputs by minimizing the loss; add $\nu\,h_m$ to the model.
3. (Stochastic GB) fit each tree on a random **row and column subsample** for regularization and speed.
4. Stop when validation loss stops improving (**early stopping**).
5. Predict by summing all trees' contributions.

## 5. When to use / When not to
- ✅ **Tabular / structured data** where top accuracy matters — often the single best off-the-shelf model.
- ✅ Heterogeneous features, non-linearities, interactions; any differentiable objective (ranking, quantiles).
- ✅ When you can afford tuning and slightly longer training.
- ❌ Very large data with tight training budgets and no tuning — a random forest is more forgiving.
- ❌ Unstructured data (images, text, audio) — deep learning dominates.

## 6. Common pitfalls & gotchas
- **Overfits if over-trained** — unlike RF, more trees can hurt; use a small learning rate **with** early stopping on a validation set.
- **Sequential = slower**, harder to parallelize than bagging.
- **Sensitive to hyperparameters** — learning rate, n_estimators, max_depth, and subsample interact (low `lr` ↔ high `n_estimators`).
- **Outliers/noisy labels** get chased by successive trees; use robust losses or subsampling.
- Don't confuse with **AdaBoost** (reweights misclassified points) — gradient boosting generalizes it to arbitrary differentiable losses.
- For big data prefer histogram-based implementations (XGBoost, LightGBM, `HistGradientBoosting`).

## 7. Code
```python
from sklearn.ensemble import GradientBoostingClassifier
# or, faster for large data:
from sklearn.ensemble import HistGradientBoostingClassifier

gb = HistGradientBoostingClassifier(
    learning_rate=0.05, max_depth=3, max_iter=1000,
    early_stopping=True, validation_fraction=0.1, random_state=0)
gb.fit(X_train, y_train)
print("n trees kept:", gb.n_iter_)          # early stopping picked this
print("test acc:", gb.score(X_test, y_test))
```

## 8. Interview / viva questions
- Q: How does gradient boosting differ from random forest?
  - A: RF builds independent deep trees in parallel to reduce variance; boosting builds shallow trees sequentially, each fitting residuals, to reduce bias — usually higher accuracy but more tuning and overfitting risk.
- Q: What does "gradient" refer to?
  - A: Each tree fits the negative gradient of the loss w.r.t. the current predictions — gradient descent in function space.
- Q: Why use a small learning rate?
  - A: Shrinkage makes each step conservative, improving generalization; pair it with more trees and early stopping.
- Q: AdaBoost vs gradient boosting?
  - A: AdaBoost reweights misclassified examples with an exponential loss; gradient boosting is the general framework fitting residuals of any differentiable loss.

## 9. References
- Friedman (2001) — "Greedy Function Approximation: A Gradient Boosting Machine", *Annals of Statistics*.
- Friedman (2002) — "Stochastic Gradient Boosting".
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 10.

---
> _Status: 🟢 done._
