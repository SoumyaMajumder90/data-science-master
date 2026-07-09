# Feature Engineering — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Transform, create, and encode raw data into features that expose patterns to the model — usually more impactful than algorithm choice.

## Key formulas
- Standardize: $x' = (x-\mu)/\sigma$
- Log transform (skewed, positive): $x' = \log(x+1)$
- Cyclic encoding: $\sin(2\pi x/period)$, $\cos(2\pi x/period)$

## Must-know facts
- Fit encoders/scalers on **train only**, apply to test — avoid leakage.
- High-cardinality categoricals: avoid one-hot, use target/frequency/hash encoding.
- Cyclical features (hour, day-of-week) need sin/cos, not raw integers.
- Feature engineering ≠ feature selection (create vs choose).

## Quick decisions
| Situation | Do this |
|---|---|
| Skewed positive numeric feature | Log / Box-Cox transform |
| Distance-based model (KNN, SVM) | Standardize/scale features |
| Low-cardinality categorical | One-hot encode |
| High-cardinality categorical | Target/frequency/hash encoding |
| Periodic feature (hour, month) | Sine/cosine encoding |

## Common mistakes
- Fitting a scaler/encoder on the full dataset before splitting (leakage).
- One-hot encoding a 50k-category column.
- Encoding hour-of-day as a plain integer.

## One-liner code
```python
df["log_x"] = np.log1p(df["x"])
```
