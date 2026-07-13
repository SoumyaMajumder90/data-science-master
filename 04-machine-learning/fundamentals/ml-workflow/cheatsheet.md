# ML Workflow — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Frame → collect → clean/EDA → engineer features → split → train/tune → evaluate → deploy → monitor. It's a loop, and most effort is upstream of modeling.

## Key formulas
- Minimize empirical risk $\hat{R}(f)=\frac1n\sum_i L(y_i,f(x_i))$ as a proxy for true risk $R(f)=\mathbb{E}[L(y,f(x))]$.

## Must-know facts
- Always **split before fitting** any transform (prevents leakage).
- **Baseline first**; complex models must beat a trivial one.
- **Test set is touched once**, at the very end.
- Deployment starts the model's life — **monitoring** and retraining are part of the job.
- Biggest wins usually come from **data & features**, not the algorithm.

## Quick decisions
| Situation | Do this |
|---|---|
| Preprocessing + model | Wrap in a `Pipeline` fit on train only |
| Tuning hyperparameters | Use validation set / CV, never the test set |
| Imbalanced target | Pick metric (PR-AUC/F1), not accuracy |
| Model live in prod | Monitor drift; schedule retraining |

## Common mistakes
- Fitting scalers/encoders/feature-selection on the full data (leakage).
- Optimizing accuracy on skewed classes.
- Repeatedly peeking at the test set.
- Train/serve feature skew; no monitoring.

## One-liner code
```python
Pipeline([("pre", preprocessor), ("clf", model)]).fit(X_train, y_train)  # split first
```
