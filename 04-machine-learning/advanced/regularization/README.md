# Regularization

> Adding a penalty on model complexity to the loss so the model prefers simpler solutions — trading a little bias for a large drop in variance and better generalization.

| | |
|---|---|
| **Category** | Machine Learning → Advanced |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [Bias-Variance Tradeoff](../../fundamentals/bias-variance-tradeoff/), [Overfitting & Underfitting](../../fundamentals/overfitting-underfitting/), [Linear Regression](../../supervised/linear-regression/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
An unconstrained model will contort itself to fit every training point, including the noise — big, wild coefficients that don't generalize. Regularization adds a "keep it simple" tax to the loss: the model can still fit the data, but every unit of complexity now costs something, so it settles for a smoother, more modest solution. It's the single most reliable tool against overfitting.

## 2. Formal definition / Key concepts
- **Penalized objective:** minimize *loss + λ·complexity*. The hyperparameter $\lambda$ (or $\alpha$) sets the strength.
- **L2 (Ridge / weight decay):** penalizes squared weights → shrinks all coefficients smoothly, handles collinearity, keeps everything.
- **L1 (Lasso):** penalizes absolute weights → drives some coefficients exactly to zero → **feature selection** / sparsity.
- **Elastic Net:** a mix of L1 and L2.
- **Other forms of regularization:** early stopping, dropout, data augmentation, max-norm, label smoothing, and even bagging — anything that constrains effective capacity.

## 3. Math
Ridge and Lasso for linear regression:
$$\text{Ridge: } \min_w \tfrac{1}{n}\lVert y - Xw\rVert_2^2 + \lambda\lVert w\rVert_2^2,\qquad
\text{Lasso: } \min_w \tfrac{1}{n}\lVert y - Xw\rVert_2^2 + \lambda\lVert w\rVert_1$$
Elastic Net: $\lambda\big(\alpha\lVert w\rVert_1 + (1-\alpha)\lVert w\rVert_2^2\big)$. Ridge has a closed form $\hat w = (X^\top X + \lambda I)^{-1}X^\top y$ (the $+\lambda I$ also fixes singular $X^\top X$). L1's diamond-shaped constraint has corners on the axes, which is *why* it yields exact zeros; L2's spherical constraint only shrinks.

## 4. How it works
1. Add the penalty term to the training loss.
2. **Standardize features first** — penalties are scale-sensitive; otherwise large-scale features are under-penalized. (Usually the intercept is *not* penalized.)
3. Fit; larger $\lambda$ → smaller weights → simpler model (more bias, less variance).
4. **Tune $\lambda$** by cross-validation (`RidgeCV`, `LassoCV`) — the sweet spot minimizes validation error.
5. For neural nets, apply weight decay, dropout, and early stopping analogously.

## 5. When to use / When not to
- ✅ Almost always for linear/logistic models, especially with many or correlated features or $p > n$.
- ✅ **Lasso/Elastic Net** when you want automatic feature selection or a sparse, interpretable model.
- ✅ **Ridge** when features are correlated and you want to keep them all (multicollinearity).
- ✅ Deep nets: dropout + weight decay + early stopping are standard.
- ❌ When the model is **underfitting** — regularization makes bias worse; reduce $\lambda$ or add capacity instead.
- ❌ Blindly using L1 with groups of correlated features (it arbitrarily keeps one) — prefer Elastic Net.

## 6. Common pitfalls & gotchas
- **Forgetting to standardize** — the most common mistake; unscaled features get inconsistent penalties.
- **Penalizing the intercept** shifts predictions; keep the bias term unpenalized.
- **Wrong $\lambda$** — too large underfits, too small overfits; always cross-validate.
- **L1 with correlated features** picks one arbitrarily and zeros the rest → unstable selection; use Elastic Net.
- **Note the C convention** — in sklearn's LogisticRegression/SVM, `C = 1/λ`, so *smaller* `C` means *stronger* regularization.
- **Leakage** — fit the scaler inside CV, not on the full data.
- Regularization reduces variance, not bias — it won't fix a model that's too simple.

## 7. Code
```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

alphas = np.logspace(-3, 3, 25)     # search lambda on a log scale
ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(X_train, y_train)
lasso = make_pipeline(StandardScaler(), LassoCV(alphas=alphas, cv=5)).fit(X_train, y_train)

print("ridge alpha:", ridge.named_steps["ridgecv"].alpha_)
print("lasso non-zero features:", np.sum(lasso.named_steps["lassocv"].coef_ != 0))
```

## 8. Interview / viva questions
- Q: L1 vs L2 — what's the practical difference?
  - A: L1 (Lasso) produces sparse solutions with exact zeros (feature selection); L2 (Ridge) shrinks all coefficients smoothly and handles collinearity but keeps every feature.
- Q: Why does L1 give exact zeros but L2 doesn't?
  - A: The L1 constraint region is a diamond with corners on the axes, so the optimum often lands on a corner (a zero coefficient); L2's spherical region has no such corners.
- Q: How does regularization relate to the bias-variance trade-off?
  - A: It increases bias slightly while substantially reducing variance, lowering total generalization error when the model was overfitting.
- Q: In sklearn's LogisticRegression, does a larger `C` mean more regularization?
  - A: No — `C = 1/λ`, so larger `C` means weaker regularization.

## 9. References
- Hoerl & Kennard (1970) — Ridge Regression; Tibshirani (1996) — "Regression Shrinkage and Selection via the Lasso".
- Zou & Hastie (2005) — "Regularization and Variable Selection via the Elastic Net".
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 3.4.

---
> _Status: 🟢 done._
