# Train / Test / Validation Split — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Train fits the model, validation tunes it, test reports it once. Split before any preprocessing; respect groups and time order.

## Key formulas
- Test estimate: $\hat{R}_{\text{test}}=\frac1m\sum_i L(y_i,f(x_i))$, standard error $\propto 1/\sqrt{m}$.
- Common ratios: 60/20/20, 70/15/15; huge data → 98/1/1.

## Must-know facts
- **Split first**, then fit transforms on train only.
- Validation is used for decisions → mildly optimistic; test is untouched → unbiased.
- **Stratify** for classification; **time cutoff** for time series; **group split** for repeated entities.
- Small data → use cross-validation instead of a single validation set.

## Quick decisions
| Situation | Do this |
|---|---|
| Classification (esp. imbalanced) | `stratify=y` |
| Time series / forecasting | Chronological split, no shuffle |
| Repeated units (user/patient) | `GroupShuffleSplit` |
| Small dataset | Cross-validation |
| Millions of rows | Small % test still large enough |

## Common mistakes
- Scaling/encoding/selecting features before the split (leakage).
- Random split of grouped or time-ordered data.
- Tuning on the test set.
- Forgetting `stratify` and dropping a rare class from a fold.

## One-liner code
```python
train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```
