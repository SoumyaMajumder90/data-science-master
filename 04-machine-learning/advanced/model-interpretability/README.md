# Model Interpretability

> Understanding *why* a model predicts what it does — through inherently transparent models or post-hoc tools (permutation importance, PDP/ICE, LIME, SHAP).

| | |
|---|---|
| **Category** | Machine Learning → Advanced |
| **Difficulty** | 🟨 Intermediate / 🟥 Advanced |
| **Prerequisites** | [Random Forest](../../supervised/random-forest/), [Feature Selection](../feature-selection/), [Logistic Regression](../../supervised/logistic-regression/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
A model that hits 95% accuracy is worthless if a doctor, regulator, or user can't trust or act on it. Interpretability answers "why this prediction?" and "what does the model rely on?". Some models are transparent by design (linear models, shallow trees); powerful ones (boosted trees, neural nets) are opaque, so we bolt on **explanation tools** that probe the trained model from the outside. The goal is trust, debugging, fairness auditing, and actionable insight.

## 2. Formal definition / Key concepts
- **Intrinsic vs post-hoc:** transparent models (linear/logistic, small trees, GAMs) vs explanations applied after training to any black box.
- **Global vs local:** overall feature effects (which features matter across all data) vs why *one* prediction was made.
- **Model-agnostic tools:**
  - **Permutation importance** — shuffle a feature, measure the performance drop (global).
  - **Partial Dependence Plots (PDP)** / **ICE** — how the prediction changes as one feature varies (global / per-instance).
  - **LIME** — fit a simple local surrogate around one instance.
  - **SHAP** — Shapley-value attributions with a solid game-theoretic foundation (local, aggregable to global).
- **Interpretability ≠ causality** — explanations describe the model, not the world.

## 3. Math
**SHAP** attributes a prediction $f(x)$ to features by the Shapley value — the average marginal contribution of feature $i$ over all subsets $S$:
$$\phi_i = \sum_{S\subseteq F\setminus\{i\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\big[f(S\cup\{i\}) - f(S)\big]$$
with the **local accuracy** (efficiency) property $f(x) = \phi_0 + \sum_i \phi_i$. **Permutation importance** for feature $j$: $\text{Imp}_j = s - \frac{1}{K}\sum_k s_{\pi_j^{(k)}}$, the drop in score $s$ after permuting $j$.

## 4. How it works
- **Permutation importance:** on held-out data, shuffle each feature, re-score, average the degradation over repeats.
- **PDP/ICE:** vary a feature over its range (fixing/averaging others), plot the model output; ICE shows one line per instance to reveal interactions.
- **SHAP:** for trees use exact **TreeSHAP** (fast); for others use KernelSHAP (sampling). Plot beeswarm (global) and force/waterfall (local).
- **LIME:** perturb the instance, weight by proximity, fit an interpretable surrogate, read its coefficients.
- Prefer explanations on a **validation/test** set, and cross-check tools against each other.

## 5. When to use / When not to
- ✅ Regulated/high-stakes domains (finance, healthcare, hiring) needing justifications.
- ✅ Debugging models, detecting leakage (a feature "too" important), and auditing for bias.
- ✅ SHAP for reliable local + global explanations of tree ensembles (TreeSHAP).
- ❌ Treating explanations as ground-truth causal effects.
- ❌ PDP with strongly **correlated features** (creates unrealistic synthetic points) — use ALE or SHAP.
- ❌ When an intrinsically interpretable model is accurate enough — prefer it over explaining a black box.

## 6. Common pitfalls & gotchas
- **Impurity (Gini) importances are biased** toward high-cardinality/continuous features — use permutation importance or SHAP instead.
- **Correlated features** mislead PDP and split/steal importance; interpret with care (ALE handles correlation better).
- **Explanations ≠ causation** — a feature can be important to the model yet not causal.
- **LIME instability** — results vary with the sampling/neighborhood; SHAP is more consistent.
- **KernelSHAP is slow** and approximate; TreeSHAP is exact only for tree models.
- **Global averages hide interactions** — always look at local explanations too.

## 7. Code
```python
import shap
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# global, unbiased importance
imp = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=0)

# SHAP for a tree model (exact, fast)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val)                 # global beeswarm

PartialDependenceDisplay.from_estimator(model, X_val, features=["age", "income"])
```

## 8. Interview / viva questions
- Q: Why not just use a model's built-in feature importances?
  - A: Impurity-based importances are biased toward high-cardinality features and can mislead; permutation importance and SHAP are more trustworthy.
- Q: Global vs local interpretability?
  - A: Global explains overall model behavior (which features matter across all data); local explains a single prediction (why *this* case).
- Q: What makes SHAP principled?
  - A: It's based on Shapley values from cooperative game theory, uniquely satisfying local accuracy, consistency, and missingness — and it decomposes each prediction additively.
- Q: A feature has huge importance but shouldn't be predictive — what do you suspect?
  - A: Data leakage; a feature encoding the target or future information. Investigate before trusting the model.

## 9. References
- Molnar — *Interpretable Machine Learning* (open-access book).
- Lundberg & Lee (2017) — "A Unified Approach to Interpreting Model Predictions" (SHAP).
- Ribeiro et al. (2016) — "Why Should I Trust You?" (LIME).

---
> _Status: 🟢 done._
