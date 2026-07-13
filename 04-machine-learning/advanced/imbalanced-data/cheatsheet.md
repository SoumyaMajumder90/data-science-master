# Imbalanced Data — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Rare class gets ignored and accuracy lies. Fix the metric, weight classes, resample inside CV, and tune the threshold.

## Key formulas
- Balanced weights: $w_c = \frac{n}{K\,n_c}$
- MCC $=\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$

## Must-know facts
- Use **PR-AUC / recall / F1 / balanced acc / MCC**, not accuracy.
- **Class weights** first (no data distortion).
- **SMOTE/resampling only on train folds** (leakage otherwise).
- **Threshold tuning** trades precision vs recall (not 0.5).
- Resampling distorts probabilities → recalibrate if needed.

## Quick decisions
| Situation | Do this |
|---|---|
| First move | `class_weight="balanced"` |
| Still low recall | SMOTE / oversample (in-fold) |
| Need true $P(y)$ | Recalibrate after resampling |
| Text/images | Augmentation/weights, not SMOTE |
| Extreme skew | Anomaly detection framing |

## Common mistakes
- Judging by accuracy or ROC-AUC.
- SMOTE before the split / outside CV.
- Leaving threshold at 0.5.
- Forgetting stratified splits.

## One-liner code
```python
imblearn.pipeline.Pipeline([("smote", SMOTE()), ("clf", RandomForestClassifier(class_weight="balanced"))])
```
