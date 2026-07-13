# Overfitting & Underfitting — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Underfit = high bias, bad on train and test. Overfit = high variance, great on train, poor on test (big generalization gap).

## Key formulas
- Generalization gap = test error − train error.
- Test error = bias² + variance + $\sigma^2$.
- Regularized loss: $\frac1n\sum_i L(y_i,f(x_i)) + \lambda\,\Omega(f)$.

## Must-know facts
- **Underfit:** both errors high & close → add capacity/features.
- **Overfit:** low train, high test → regularize, more data, simplify.
- More data reduces **variance**, never bias.
- Early stopping = regularization.
- Deep nets can show **double descent**.

## Quick decisions
| Symptom | Fix |
|---|---|
| High train + high test error | More complexity/features, train longer |
| Low train, high test error | L1/L2/dropout, more data, simplify, early stop |
| Big gap on learning curve | Get more data (variance) |
| Both curves plateau high | Change model (bias) |

## Common mistakes
- Judging fit by training error only.
- Expecting more data to fix underfitting.
- Leakage hiding the real generalization gap.

## One-liner code
```python
validation_curve(model, X, y, param_name="max_depth", param_range=range(1,21), cv=5)
```
