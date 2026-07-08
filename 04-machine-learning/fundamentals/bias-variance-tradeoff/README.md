# Bias-Variance Tradeoff

> The core tension in supervised learning: overly simple models miss patterns (**bias**), overly flexible models chase noise (**variance**), and total error is minimized somewhere in between.

| | |
|---|---|
| **Category** | Machine Learning → Fundamentals |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [ML Workflow](../ml-workflow/), [Overfitting & Underfitting](../overfitting-underfitting/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
Imagine hitting a target many times with different training sets. **Bias** is how far your average shot lands from the bullseye — a systematic error from wrong assumptions. **Variance** is how scattered your shots are — sensitivity to which particular data you trained on. You want both low, but reducing one usually raises the other.

## 2. Formal definition / Key concepts
Expected test error at a point decomposes into three parts:
- **Bias²** — error from approximating a complex reality with a simpler model.
- **Variance** — error from sensitivity to the training sample.
- **Irreducible error** — noise you can never remove.

High bias = **underfitting**; high variance = **overfitting**.

## 3. Math
For squared-error loss, expected prediction error decomposes as:
$$\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \underbrace{(\text{Bias}[\hat{f}(x)])^2}_{\text{systematic}} + \underbrace{\text{Var}[\hat{f}(x)]}_{\text{sensitivity}} + \underbrace{\sigma^2}_{\text{noise}}$$

where $\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)$.

## 4. How it works
As model complexity increases (more parameters, deeper trees, higher-degree polynomials):
- Bias **decreases** (model can fit more).
- Variance **increases** (model reacts more to noise).
- Total error traces a **U-shape**; the sweet spot is the minimum.

## 5. When to use / When not to
This is a diagnostic lens, not an algorithm. Use it to reason about *why* a model underperforms:
- ✅ High train error **and** high test error → high bias → add complexity/features.
- ✅ Low train error, high test error → high variance → regularize, get more data, simplify.

## 6. Common pitfalls & gotchas
- Deep learning can break the classic U-curve (**double descent**) — very large models sometimes generalize well despite interpolating the training set.
- More data reduces **variance**, not bias.
- Regularization trades a little bias for a large drop in variance.
- Ensembles (bagging) attack variance; boosting attacks bias.

## 7. Code
```python
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)
)
# Big gap between train and val scores => variance; both low => bias.
```

## 8. Interview / viva questions
- Q: You get 99% train accuracy, 70% test. Diagnosis?
  - A: High variance / overfitting — regularize, add data, or reduce complexity.
- Q: Does more data fix high bias?
  - A: No. It mainly reduces variance; bias needs a more expressive model or better features.
- Q: How do bagging and boosting relate to this tradeoff?
  - A: Bagging (e.g. random forest) reduces variance; boosting reduces bias.

## 9. References
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 2 & 7.
- Belkin et al. (2019) — "Reconciling modern ML and the bias-variance trade-off" (double descent).

---
> _Status: 🟢 done (example note)._
