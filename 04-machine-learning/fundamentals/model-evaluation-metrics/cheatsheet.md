# Model Evaluation Metrics — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Pick the metric from the cost of errors and class balance. Never trust accuracy on imbalanced data.

## Key formulas
- Precision $=\frac{TP}{TP+FP}$, Recall $=\frac{TP}{TP+FN}$, Specificity $=\frac{TN}{TN+FP}$
- $F_1 = 2\frac{PR}{P+R}$
- RMSE $=\sqrt{\frac1n\sum(y-\hat y)^2}$, $R^2 = 1-\frac{SS_{res}}{SS_{tot}}$
- LogLoss $=-\frac1n\sum[y\log\hat p+(1-y)\log(1-\hat p)]$

## Must-know facts
- **Precision** = don't cry wolf; **Recall** = don't miss any.
- **PR-AUC** > ROC-AUC under heavy imbalance.
- **AUC ≠ calibration**; use log loss/Brier for probability quality.
- RMSE punishes big errors; **MAE** is robust to outliers.
- Multiclass: **macro** treats classes equally, **weighted/micro** by support.
- $R^2$ always rises with more features → use adjusted $R^2$.

## Quick decisions
| Goal | Metric |
|---|---|
| Imbalanced classification | PR-AUC, F1, recall@precision |
| Threshold-free ranking | ROC-AUC |
| Probability quality | Log loss / Brier |
| Regression w/ outliers | MAE |
| Penalize large errors | RMSE |

## Common mistakes
- Accuracy on skewed data (accuracy paradox).
- Reading ROC-AUC as good under rare positives.
- Leaving threshold at 0.5 when costs are asymmetric.
- Judging regression by $R^2$ alone.

## One-liner code
```python
from sklearn.metrics import classification_report; print(classification_report(y_test, pred))
```
