# Ensemble Methods

> Combining many models so their errors partly cancel — bagging reduces variance, boosting reduces bias, and stacking learns how to blend diverse models.

| | |
|---|---|
| **Category** | Machine Learning → Advanced |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [Bias-Variance Tradeoff](../../fundamentals/bias-variance-tradeoff/), [Random Forest](../../supervised/random-forest/), [Gradient Boosting](../../supervised/gradient-boosting/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
A single model has one perspective and its own blind spots. Ask a *committee* of models and average their opinions: if their mistakes are somewhat independent, the errors cancel and the consensus is more accurate and stable than any member. The art is making the members **diverse** (different data, features, or algorithms) so they don't all make the *same* mistake — a committee of clones is no better than one.

## 2. Formal definition / Key concepts
- **Bagging (Bootstrap Aggregating):** train models on bootstrap resamples and average/vote → reduces **variance**. E.g. Random Forest.
- **Boosting:** train models sequentially, each fixing the previous one's errors → reduces **bias**. E.g. AdaBoost, gradient boosting, XGBoost.
- **Stacking (stacked generalization):** train diverse base models, then a **meta-learner** on their out-of-fold predictions.
- **Voting/Averaging:** simple hard/soft vote (classification) or mean (regression) over heterogeneous models.
- Ensembles help most when members are **accurate and diverse** (decorrelated errors).

## 3. Math
Averaging $B$ models each with variance $\sigma^2$ and pairwise error correlation $\rho$:
$$\text{Var}_{\text{avg}} = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2$$
More models shrink the second term; **lower correlation** $\rho$ shrinks the first — hence the emphasis on diversity. Boosting instead forms an additive model $F_M(x) = \sum_m \nu\, h_m(x)$ that greedily reduces the training loss stage by stage.

## 4. How it works
- **Bagging:** bootstrap → fit independent (often deep) learners in parallel → average/vote. Diversity comes from data resampling (+ feature subsets in RF).
- **Boosting:** start weak → repeatedly fit a learner to the current residuals/errors → add it with a small weight. Diversity comes from focusing on hard cases.
- **Stacking:** generate **out-of-fold** predictions from each base model (to avoid leakage) → feed them as features to a meta-model (often simple, e.g. logistic/linear).
- **Blending** is stacking with a single hold-out set instead of CV folds.

## 5. When to use / When not to
- ✅ Squeeze extra accuracy from tabular problems (boosting/stacking win competitions).
- ✅ **Bagging** when your base model is high-variance (deep trees); **boosting** when it's high-bias (stumps).
- ✅ **Stacking/voting** to combine genuinely different model families (trees + linear + kNN + NN).
- ❌ When interpretability, latency, or simplicity matter — ensembles are heavier black boxes.
- ❌ When base models are already low-variance/low-bias or all make the same errors (little to gain).

## 6. Common pitfalls & gotchas
- **No diversity, no benefit** — averaging highly correlated models barely helps.
- **Stacking leakage** — you must use out-of-fold predictions to train the meta-learner, never in-sample ones.
- **Overfitting the meta-learner** — keep it simple; too many/complex base models can memorize.
- **Bagging won't fix bias; boosting can overfit** — match the technique to the failure mode.
- **Cost** — training and serving many models multiplies compute, memory, and latency.
- Diminishing returns: doubling members rarely doubles gains.

## 7. Code
```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

stack = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=300, n_jobs=-1)),
        ("svc", SVC(probability=True)),
    ],
    final_estimator=LogisticRegression(),
    cv=5,                       # out-of-fold preds -> no leakage
    n_jobs=-1,
)
stack.fit(X_train, y_train)
print("test acc:", stack.score(X_test, y_test))
```

## 8. Interview / viva questions
- Q: Bagging vs boosting — what does each reduce?
  - A: Bagging reduces variance by averaging independent high-variance learners; boosting reduces bias by sequentially fitting residuals of weak learners.
- Q: Why is diversity essential in ensembles?
  - A: Averaging only cancels errors that are uncorrelated; identical models share the same mistakes, so the variance-reduction term $\frac{1-\rho}{B}\sigma^2$ vanishes as $\rho\to1$.
- Q: How does stacking avoid leakage?
  - A: The meta-learner is trained on out-of-fold (cross-validated) base predictions, so no base model predicts on data it trained on.
- Q: When would you not use an ensemble?
  - A: When you need interpretability, low latency, or when base models are already strong and make correlated errors.

## 9. References
- Breiman (1996) — "Bagging Predictors"; Breiman (2001) — "Random Forests".
- Freund & Schapire (1997) — AdaBoost; Wolpert (1992) — "Stacked Generalization".
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 8, 10, 15, 16.

---
> _Status: 🟢 done._
