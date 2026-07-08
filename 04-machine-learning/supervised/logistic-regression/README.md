# Logistic Regression

> A linear model for **classification** that maps a weighted sum of features through a sigmoid to produce a probability between 0 and 1.

| | |
|---|---|
| **Category** | Machine Learning → Supervised |
| **Difficulty** | 🟩 Beginner / 🟨 Intermediate |
| **Prerequisites** | [Linear Regression](../linear-regression/), [Probability](../../../01-foundations/mathematics/probability/), [Optimization](../../../01-foundations/mathematics/optimization/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
Linear regression predicts an unbounded number; that's wrong for a yes/no question. Logistic regression squashes the linear score $z = w^\top x + b$ through the **sigmoid** so the output is a probability. You then threshold (usually at 0.5) to decide the class. Despite the name, it's a **classifier**, not a regressor.

## 2. Formal definition / Key concepts
- **Model:** $P(y=1 \mid x) = \sigma(w^\top x + b)$
- **Decision boundary** is linear in the features (a hyperplane where $z = 0$).
- **Log-odds (logit)** are linear: $\log\frac{p}{1-p} = w^\top x + b$. Each weight is the change in log-odds per unit change in that feature.
- Extends to multiclass via **softmax** (multinomial logistic regression).

## 3. Math
Sigmoid:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Trained by minimizing **binary cross-entropy** (negative log-likelihood):
$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\big[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\big]$$

There is no closed-form solution, so weights are found by gradient descent. The gradient has a clean form:
$$\nabla_w \mathcal{L} = \frac{1}{n} X^\top (\hat{p} - y)$$

## 4. How it works
1. Compute linear score $z = w^\top x + b$.
2. Apply sigmoid → probability $\hat{p}$.
3. Compare $\hat{p}$ to a threshold to assign a class.
4. Fit weights by minimizing cross-entropy with (stochastic) gradient descent; add L1/L2 penalties to regularize.

## 5. When to use / When not to
- ✅ Strong, interpretable **baseline** for binary/multiclass problems.
- ✅ When you need calibrated **probabilities** and coefficient interpretability (odds ratios).
- ✅ High-dimensional sparse data (e.g. text) with L1/L2 regularization.
- ❌ When the true boundary is highly non-linear (unless you engineer features).
- ❌ Heavy feature interactions — tree ensembles often win.

## 6. Common pitfalls & gotchas
- **Scale your features** — gradient descent and regularization assume comparable scales.
- **Class imbalance** distorts the threshold; use class weights, resampling, or move the threshold rather than trusting 0.5.
- **Perfect separation** makes weights diverge to infinity → regularization fixes it.
- Coefficients are in **log-odds**, not probabilities — exponentiate for odds ratios.
- AUC measures ranking, not calibration; a good AUC can still be poorly calibrated.

## 7. Code
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=1000)
)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]   # probabilities for the positive class
```

## 8. Interview / viva questions
- Q: Why can't we use MSE loss for logistic regression?
  - A: With the sigmoid, MSE is non-convex in the weights and gradients vanish; cross-entropy is convex and gives well-behaved gradients.
- Q: What does a coefficient of 0.7 mean?
  - A: A one-unit increase in that feature multiplies the odds of the positive class by $e^{0.7} \approx 2.0$, holding others fixed.
- Q: How do you handle multiclass?
  - A: One-vs-rest or a single softmax (multinomial) model.

## 9. References
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 4.
- scikit-learn User Guide: Logistic Regression.

---
> _Status: 🟢 done (example note)._
