# Hyperparameter Tuning

> Systematically searching the settings you choose *before* training (learning rate, depth, regularization) to maximize validated performance — via grid, random, or Bayesian search.

| | |
|---|---|
| **Category** | Machine Learning → Advanced |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [Cross-Validation](../../fundamentals/cross-validation/), [Regularization](../regularization/), [Model Evaluation Metrics](../../fundamentals/model-evaluation-metrics/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
**Parameters** are learned from data (weights); **hyperparameters** are the dials you set beforehand (tree depth, learning rate, $C$, number of neighbors). They control model capacity and the optimization itself, and the right values are problem-specific. Tuning is the disciplined search for the dial settings that generalize best — measured on validation data, not the training set, and never on the final test set.

## 2. Formal definition / Key concepts
- **Objective:** choose $\theta$ to maximize a cross-validated score $\text{CV-score}(\theta)$ over a search space.
- **Grid search** — exhaustively try every combination on a predefined grid.
- **Random search** — sample combinations at random; more efficient when few hyperparameters matter.
- **Bayesian optimization** — build a surrogate model of score vs hyperparameters and query promising regions (Optuna, Hyperopt, `skopt`).
- **Successive halving / Hyperband** — give many configs small budgets, promote the best (bandit-based, great for expensive models).
- **Search space**: sample scale-sensitive params (learning rate, $C$, $\lambda$) on a **log scale**.

## 3. Math
Formally a **bi-level** optimization: inner loop fits parameters; outer loop tunes hyperparameters.
$$\theta^\* = \arg\max_{\theta\in\Theta}\ \frac{1}{k}\sum_{j=1}^{k}\text{score}\big(f_{\hat w^{(-j)}(\theta)},\ \text{fold}_j\big)$$
Grid search over $d$ dimensions costs $\prod_i m_i$ fits — exponential in $d$. Random search's advantage (Bergstra & Bengio): with only a few *effective* dimensions, random sampling covers them far better per unit compute than a grid.

## 4. How it works
1. Split off a **test set** first; tune only on the rest.
2. Define the search space and a **CV scheme** + scoring metric aligned to the goal.
3. Run the search (grid/random/Bayesian), each candidate scored by cross-validation.
4. Pick the best config; **refit** on all training data.
5. Report performance **once** on the untouched test set. Use **nested CV** if you also need an unbiased estimate of the tuned model.

## 5. When to use / When not to
- ✅ Any model with impactful hyperparameters (boosting, SVM, neural nets, regularized linear models).
- ✅ **Random/Bayesian** search for large or continuous spaces and expensive models.
- ✅ **Grid** search only for a few discrete, cheap parameters.
- ❌ When compute is tight and defaults are already strong (start with sensible defaults + a small random search).
- ❌ Tuning dozens of params on tiny data — you'll overfit the validation set.

## 6. Common pitfalls & gotchas
- **Tuning on the test set** — the cardinal sin; it inflates the final estimate. Use validation/CV, keep test untouched.
- **Leakage in CV** — preprocessing must live inside the pipeline so it refits per fold during the search.
- **Over-tuning / validation overfitting** — with a big search and small data, the "best" config is partly luck; prefer nested CV and simpler models.
- **Linear-scale sampling** of learning rate / regularization wastes the search; use log scale.
- **Grid explosion** — combinatorial cost; random/Bayesian scale better.
- Optimizing the **wrong metric** (accuracy on imbalanced data) tunes toward the wrong model.

## 7. Code
```python
from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier

search = RandomizedSearchCV(
    GradientBoostingClassifier(),
    param_distributions={
        "learning_rate": loguniform(1e-3, 3e-1),   # log scale
        "max_depth": randint(2, 6),
        "n_estimators": randint(100, 800),
        "subsample": [0.6, 0.8, 1.0],
    },
    n_iter=50, cv=5, scoring="roc_auc", random_state=0, n_jobs=-1,
)
search.fit(X_train, y_train)          # test set untouched
print(search.best_params_, search.best_score_)
```

## 8. Interview / viva questions
- Q: Parameters vs hyperparameters?
  - A: Parameters are learned from data during training (weights); hyperparameters are set beforehand and control capacity/optimization (depth, learning rate, $C$).
- Q: Why is random search often better than grid search?
  - A: When only a few hyperparameters matter, random sampling explores those effective dimensions more densely for the same budget, whereas grids waste trials on irrelevant ones.
- Q: How do you avoid overfitting the validation set while tuning?
  - A: Use cross-validation, limit the search size, sample sensibly (log scale), and use nested CV for an unbiased estimate.
- Q: What is Bayesian optimization?
  - A: It fits a surrogate (e.g. Gaussian process/TPE) mapping hyperparameters to score and uses an acquisition function to choose the next promising configuration, needing fewer evaluations than grid/random.

## 9. References
- Bergstra & Bengio (2012) — "Random Search for Hyper-Parameter Optimization", JMLR.
- Li et al. (2017) — "Hyperband: A Novel Bandit-Based Approach…".
- Akiba et al. (2019) — "Optuna: A Next-generation Hyperparameter Optimization Framework".

---
> _Status: 🟢 done._
