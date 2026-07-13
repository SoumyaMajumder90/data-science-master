# Model Interpretability — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Explain *why* a model predicts. Intrinsic (linear/trees) vs post-hoc (perm importance, PDP/ICE, LIME, SHAP). Global vs local.

## Key formulas
- SHAP (Shapley): $\phi_i=\sum_{S}\frac{|S|!(|F|-|S|-1)!}{|F|!}[f(S\cup i)-f(S)]$, with $f(x)=\phi_0+\sum_i\phi_i$
- Perm importance: score drop after shuffling a feature.

## Must-know facts
- **Impurity importances are biased** → use permutation/SHAP.
- **TreeSHAP** exact+fast for trees; KernelSHAP slow/approx.
- **LIME** = local surrogate, less stable than SHAP.
- **PDP** misleads with correlated features → ALE/SHAP.
- Interpretability **≠ causation**.

## Quick decisions
| Need | Tool |
|---|---|
| Global importance | Permutation importance / SHAP beeswarm |
| One prediction | SHAP force/waterfall, LIME |
| Feature effect shape | PDP / ICE (ALE if correlated) |
| Tree ensemble | TreeSHAP |

## Common mistakes
- Trusting Gini importances.
- PDP on correlated features.
- Reading explanations as causal.
- Missing leakage flagged by an over-important feature.

## One-liner code
```python
shap.summary_plot(shap.TreeExplainer(model).shap_values(X_val), X_val)
```
