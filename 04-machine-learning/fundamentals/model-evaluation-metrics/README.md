# Model Evaluation Metrics

> The scorecards for models — how you turn predictions vs ground truth into a single number that reflects what you actually care about.

| | |
|---|---|
| **Category** | Machine Learning → Fundamentals |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [Logistic Regression](../../supervised/logistic-regression/), [Hypothesis Testing](../../../03-data-analysis/hypothesis-testing/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
"Accuracy" sounds like the obvious score, but on a dataset that is 99% negatives a model that predicts "negative" always scores 99% while being useless. The right metric depends on what a mistake *costs*: missing a fraud (false negative) vs flagging a good customer (false positive) are not equal. Metrics are how you encode those priorities into something optimizable and comparable.

## 2. Formal definition / Key concepts
**Classification** builds on the confusion matrix (TP, FP, TN, FN):
- **Accuracy** — fraction correct; misleading under imbalance.
- **Precision** — of predicted positives, how many are real ($\tfrac{TP}{TP+FP}$).
- **Recall / Sensitivity / TPR** — of actual positives, how many found ($\tfrac{TP}{TP+FN}$).
- **F1** — harmonic mean of precision and recall.
- **Specificity (TNR)** — $\tfrac{TN}{TN+FP}$.
- **ROC-AUC** — ranking quality across all thresholds; TPR vs FPR.
- **PR-AUC** — precision vs recall; preferred under heavy imbalance.
- **Log loss** — penalizes miscalibrated probabilities.

**Regression:**
- **MAE, MSE, RMSE** — average error (RMSE punishes large errors more).
- **$R^2$** — fraction of variance explained.
- **MAPE** — mean absolute *percentage* error.

## 3. Math
$$\text{Precision} = \frac{TP}{TP+FP},\qquad \text{Recall} = \frac{TP}{TP+FN}$$
$$F_1 = 2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}},\qquad F_\beta = (1+\beta^2)\frac{PR}{\beta^2 P + R}$$
$$\text{RMSE} = \sqrt{\tfrac{1}{n}\textstyle\sum_i (y_i-\hat y_i)^2},\qquad R^2 = 1 - \frac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2}$$
$$\text{LogLoss} = -\tfrac{1}{n}\textstyle\sum_i\big[y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)\big]$$

## 4. How it works
1. Choose the metric from the **cost of errors** and class balance *before* modeling.
2. For probabilistic classifiers, decide **threshold-free** (AUC, log loss) vs **thresholded** (precision/recall/F1) evaluation.
3. Pick an operating **threshold** from the precision-recall or ROC curve to match business constraints.
4. For multiclass, aggregate with **macro** (unweighted mean over classes — treats classes equally) or **micro/weighted** (accounts for support).
5. Report a confidence interval or CV std, not just a point estimate.

## 5. When to use / When not to
- ✅ **Imbalanced classification** → PR-AUC, F1, recall at fixed precision; avoid raw accuracy.
- ✅ **Ranking / thresholds unknown** → ROC-AUC.
- ✅ **Calibrated probabilities needed** → log loss / Brier score.
- ✅ **Regression with outliers** → MAE (robust) over RMSE.
- ❌ Accuracy on skewed data; $R^2$ as the sole regression judge; a single metric when costs are asymmetric.

## 6. Common pitfalls & gotchas
- **Accuracy paradox** — high accuracy on imbalanced data can mean the model ignores the minority class.
- **ROC-AUC looks great under imbalance** even for weak models; PR-AUC is more honest.
- **AUC ≠ calibration** — good ranking can still give badly calibrated probabilities.
- **Threshold defaults** — 0.5 is arbitrary; tune it to the cost trade-off.
- **Macro vs micro** averaging changes the story on imbalanced multiclass.
- **$R^2$ can be negative** and always rises when you add features (use adjusted $R^2$).

## 7. Code
```python
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, confusion_matrix)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))         # precision/recall/F1 per class
print("ROC-AUC:", roc_auc_score(y_test, proba))
print("PR-AUC :", average_precision_score(y_test, proba))   # prefer under imbalance
```

## 8. Interview / viva questions
- Q: When is accuracy a bad metric?
  - A: Under class imbalance or asymmetric error costs — a trivial majority-class predictor can score high while being useless.
- Q: Precision vs recall — give an example of favoring each.
  - A: Spam filter favors precision (don't drop real mail); cancer screening favors recall (don't miss a case).
- Q: ROC-AUC vs PR-AUC?
  - A: ROC-AUC can look optimistic on rare positives; PR-AUC focuses on the positive class and is more informative under imbalance.
- Q: Why might a high-AUC model still be unusable?
  - A: It may rank well but produce poorly calibrated probabilities, or its best threshold may not meet the required precision.
- Q: RMSE vs MAE?
  - A: RMSE penalizes large errors more (sensitive to outliers); MAE is robust and in the same units as the target.

## 9. References
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 7.
- Saito & Rehmsmeier (2015) — "The Precision-Recall Plot is More Informative than the ROC Plot…".
- scikit-learn User Guide: Metrics and scoring.

---
> _Status: 🟢 done._
