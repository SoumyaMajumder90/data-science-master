# Imbalanced Data

> Classification where one class vastly outnumbers another (fraud, disease, churn) — accuracy becomes misleading and models ignore the rare-but-important minority unless you intervene.

| | |
|---|---|
| **Category** | Machine Learning → Advanced |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [Model Evaluation Metrics](../../fundamentals/model-evaluation-metrics/), [Logistic Regression](../../supervised/logistic-regression/), [Cross-Validation](../../fundamentals/cross-validation/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
If 99.8% of transactions are legitimate, a model that shouts "legit!" every time is 99.8% accurate and completely useless — it never catches fraud. The rare class is usually the one we care about most. Handling imbalance means changing *what we optimize* (the metric), *how we present the data* (resampling), or *how we penalize errors* (class weights / thresholds) so the minority isn't drowned out.

## 2. Formal definition / Key concepts
- **Class imbalance:** skewed prior $P(y=1)\ll P(y=0)$; ratios of 1:100 or worse are common.
- **Metric fixes:** evaluate with **PR-AUC, recall, F1, balanced accuracy, MCC** — not raw accuracy or (misleadingly) ROC-AUC.
- **Data-level fixes:** **oversample** the minority (random, **SMOTE**), **undersample** the majority, or combine.
- **Algorithm-level fixes:** **class weights** / cost-sensitive learning; **threshold tuning**; anomaly-detection framing when extreme.
- **The rare class is a data problem, not just a modeling one** — more real minority examples beat any trick.

## 3. Math
Cost-sensitive learning reweights the loss so minority errors cost more; scikit-learn's `class_weight="balanced"` sets
$$w_c = \frac{n}{K\,n_c}$$
(inversely proportional to class frequency). **Matthews Correlation Coefficient**, robust under imbalance:
$$\text{MCC} = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$
Threshold moves trade precision vs recall along the PR curve; the optimal threshold depends on the cost of FP vs FN.

## 4. How it works
1. **Pick the right metric first** (PR-AUC / recall at fixed precision) tied to business costs.
2. Establish a baseline with **class weights** (cheap, no data distortion).
3. If needed, **resample** — but fit resampling **inside CV folds** (via `imblearn.Pipeline`) so it never touches validation data.
4. **Tune the decision threshold** on validation to hit the required precision/recall, instead of trusting 0.5.
5. Consider gathering more minority data, better features, or an **anomaly-detection** approach for extreme skew.
6. Use **stratified** splits/CV throughout.

## 5. When to use / When not to
- ✅ Fraud, medical diagnosis, churn, defect/anomaly detection, rare-event prediction.
- ✅ Class weights and threshold tuning: almost always the first, safest levers.
- ✅ SMOTE/oversampling on **tabular** data with a moderately rare class.
- ❌ Applying SMOTE to text/images or very high dimensions (interpolated samples are unrealistic) — prefer weights/augmentation.
- ❌ Resampling when the imbalance is mild and the metric already reflects the goal.
- ❌ Undersampling when it throws away too much majority signal.

## 6. Common pitfalls & gotchas
- **Accuracy trap** — 99% accuracy can mean 0% recall on the minority. Report PR-AUC/recall/F1/MCC.
- **Resampling before the split / outside CV** = leakage; SMOTE using validation neighbors inflates scores badly. Always resample **train folds only**.
- **SMOTE amplifies noise** and mislabels near class boundaries; it assumes meaningful interpolation.
- **Changing the base rate distorts probabilities** — recalibrate if you need true $P(y=1)$.
- **ROC-AUC looks optimistic** under heavy imbalance; PR-AUC is more honest.
- Forgetting **stratification** can leave folds with too few (or zero) minority cases.

## 7. Code
```python
from imblearn.pipeline import Pipeline          # imblearn, not sklearn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

pipe = Pipeline([
    ("smote", SMOTE(random_state=0)),           # applied to train folds only
    ("clf", RandomForestClassifier(class_weight="balanced", n_jobs=-1)),
])
cv = StratifiedKFold(5, shuffle=True, random_state=0)
print(cross_val_score(pipe, X, y, cv=cv, scoring="average_precision").mean())  # PR-AUC
```

## 8. Interview / viva questions
- Q: Why is accuracy a bad metric for imbalanced data?
  - A: A trivial majority-class predictor scores high while never detecting the minority; use PR-AUC, recall, F1, balanced accuracy, or MCC.
- Q: What's the danger of SMOTE and how do you avoid it?
  - A: If applied before splitting/CV it leaks (synthetic points use neighbors that end up in validation); always resample inside the training folds via an imblearn pipeline.
- Q: Class weights vs resampling — which first?
  - A: Class weights are the cheapest, distortion-free first move; resampling is a next step when weights alone don't reach the target recall.
- Q: How does threshold tuning help?
  - A: Lowering the decision threshold raises recall (at some precision cost); the right threshold is set by the relative cost of false positives vs false negatives, not the default 0.5.

## 9. References
- Chawla et al. (2002) — "SMOTE: Synthetic Minority Over-sampling Technique", JAIR.
- He & Garcia (2009) — "Learning from Imbalanced Data", IEEE TKDE.
- `imbalanced-learn` documentation; scikit-learn User Guide (class_weight, metrics).

---
> _Status: 🟢 done._
