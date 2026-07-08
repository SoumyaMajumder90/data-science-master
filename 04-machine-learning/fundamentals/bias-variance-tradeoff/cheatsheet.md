# Bias-Variance Tradeoff — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Error = Bias² + Variance + Noise. Simple models underfit (bias), complex models overfit (variance).

## Key formulas
- $\mathbb{E}[(y-\hat{f})^2] = \text{Bias}^2 + \text{Var} + \sigma^2$
- $\text{Bias} = \mathbb{E}[\hat{f}] - f$

## Must-know facts
- High bias ⇒ underfitting; high variance ⇒ overfitting.
- Complexity ↑ ⇒ bias ↓, variance ↑ (U-shaped total error).
- More data ↓ variance, **not** bias.
- Double descent: huge models can defy the classic curve.

## Quick decisions
| Symptom | Cause | Fix |
|---|---|---|
| Train & test both bad | Bias | More complexity / features |
| Train good, test bad | Variance | Regularize, more data, simplify |

## Common mistakes
- Thinking more data cures bias.
- Ignoring irreducible noise as a floor on accuracy.

## One-liner code
```python
learning_curve(model, X, y, cv=5)  # gap => variance, both low => bias
```
