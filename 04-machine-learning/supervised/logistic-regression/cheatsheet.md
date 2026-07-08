# Logistic Regression — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Linear classifier — sigmoid over $w^\top x + b$ gives a probability; trained with cross-entropy.

## Key formulas
- Sigmoid: $\sigma(z) = 1/(1+e^{-z})$
- Prediction: $P(y=1\mid x) = \sigma(w^\top x + b)$
- Logit: $\log\frac{p}{1-p} = w^\top x + b$
- Loss: binary cross-entropy (negative log-likelihood)

## Must-know facts
- It's a **classifier**, not a regressor.
- Decision boundary is **linear**.
- No closed form → gradient descent.
- Coefficients = change in **log-odds**; $e^{w}$ = odds ratio.

## Quick decisions
| Situation | Do this |
|---|---|
| Features on different scales | Standardize |
| Class imbalance | `class_weight="balanced"` or move threshold |
| Overfitting / many features | L1 (sparse) or L2 penalty |
| Perfect separation | Add regularization |

## Common mistakes
- Reading coefficients as probabilities.
- Trusting the 0.5 threshold on imbalanced data.
- Forgetting to scale before regularizing.

## One-liner code
```python
LogisticRegression(class_weight="balanced", max_iter=1000).fit(X, y)
```
