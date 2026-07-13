# Overfitting & Underfitting

> The two failure modes of learning: **underfitting** is too little capacity to capture the signal, **overfitting** is so much capacity that the model memorizes noise and fails to generalize.

| | |
|---|---|
| **Category** | Machine Learning → Fundamentals |
| **Difficulty** | 🟩 Beginner / 🟨 Intermediate |
| **Prerequisites** | [Bias-Variance Tradeoff](../bias-variance-tradeoff/), [Train/Test/Validation Split](../train-test-validation-split/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
Fitting a model is like drawing a curve through data points. **Underfitting** is a straight line through a clearly curved cloud — too rigid, it misses the pattern and does badly everywhere. **Overfitting** is a wiggly curve threaded through every single point including the noise — it looks perfect on the training data but flails on anything new. The goal is the smooth curve in between that captures the trend and ignores the jitter.

## 2. Formal definition / Key concepts
- **Underfitting (high bias):** model too simple → high training error *and* high test error.
- **Overfitting (high variance):** model too flexible → low training error but a large **generalization gap** (test error ≫ train error).
- **Generalization gap** = test error − train error; it grows with model capacity and shrinks with more data/regularization.
- The classic remedy space: adjust **capacity**, **regularization**, and **data size**.

## 3. Math
Expected test error decomposes into bias, variance, and irreducible noise:
$$\mathbb{E}\big[(y-\hat f(x))^2\big] = \text{Bias}[\hat f(x)]^2 + \text{Var}[\hat f(x)] + \sigma^2$$
Underfitting ⇔ the **bias** term dominates; overfitting ⇔ the **variance** term dominates. Regularization adds a penalty $\lambda\,\Omega(f)$ to the loss, trading a little bias for a large variance reduction:
$$\min_f \ \frac{1}{n}\sum_i L(y_i, f(x_i)) + \lambda\,\Omega(f)$$

## 4. How it works
Diagnose from the two error curves as complexity or training progresses:
- **Both errors high, close together** → underfitting → add capacity/features, train longer, reduce regularization.
- **Train error low, test error much higher** → overfitting → regularize, get more data, simplify, add dropout/early stopping.
- **Learning curve** (error vs training-set size): a large persistent gap → high variance (more data helps); both curves plateau high → high bias (more data won't help).

## 5. When to use / When not to
This is a diagnostic lens applied to every model. Use it to decide the next move:
- ✅ High train + high test error → increase model complexity or engineer better features.
- ✅ Low train, high test error → regularize, gather data, prune features, early-stop.
- ✅ Use validation curves to locate the capacity sweet spot.
- ❌ Don't fight overfitting by blindly shrinking the model if the real issue is too little data.

## 6. Common pitfalls & gotchas
- Judging fit from **training error alone** — it always improves with capacity; only the validation curve reveals overfitting.
- **More data reduces variance, not bias** — it won't rescue an underfit model.
- **Early stopping** is regularization; stopping too late overfits, too early underfits.
- Deep nets can show **double descent** — test error can drop again past the interpolation point, breaking the simple U-curve.
- Data leakage can *hide* overfitting by making the test set look easy.

## 7. Code
```python
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier

depths = range(1, 21)
train, val = validation_curve(
    DecisionTreeClassifier(random_state=0), X, y,
    param_name="max_depth", param_range=depths, cv=5, scoring="accuracy")

# train keeps rising while val peaks then falls => overfitting past the peak depth
best_depth = depths[np.argmax(val.mean(axis=1))]
print("best max_depth:", best_depth)
```

## 8. Interview / viva questions
- Q: How do you tell overfitting from underfitting?
  - A: Compare errors — both high ⇒ underfitting; low train but high test (big gap) ⇒ overfitting.
- Q: Name three ways to reduce overfitting.
  - A: More training data, regularization (L1/L2/dropout), and reducing model complexity or using early stopping.
- Q: Will collecting more data fix underfitting?
  - A: No — underfitting is a bias problem; you need a more expressive model or better features.
- Q: What does a learning curve tell you?
  - A: A large persistent train/val gap signals variance (more data helps); both plateauing high signals bias (more data won't).

## 9. References
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 7.
- Goodfellow, Bengio, Courville — *Deep Learning*, ch. 5 (capacity, overfitting, regularization).
- Belkin et al. (2019) — "Reconciling modern ML and the bias-variance trade-off" (double descent).

---
> _Status: 🟢 done._
