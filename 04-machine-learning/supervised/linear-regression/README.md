# Linear Regression

> A model that predicts a continuous target as a weighted sum of features, fit by minimizing squared error.

| | |
|---|---|
| **Category** | Machine Learning → Supervised |
| **Difficulty** | 🟩 Beginner |
| **Prerequisites** | [Linear Algebra](../../../01-foundations/mathematics/linear-algebra/), [Optimization](../../../01-foundations/mathematics/optimization/), [Correlation vs Causation](../../../03-data-analysis/correlation-vs-causation/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
You believe the outcome moves roughly proportionally with the inputs: more square footage → higher price, and each extra bathroom adds a bit more. Linear regression finds the best straight-line (hyperplane) relationship by drawing the line that makes the vertical gaps to the data points as small as possible — specifically, the sum of *squared* gaps. It's the simplest, most interpretable workhorse of predictive modeling.

## 2. Formal definition / Key concepts
- **Model:** $\hat y = w^\top x + b$ — output is linear in the weights.
- **Ordinary Least Squares (OLS):** choose $w, b$ to minimize the mean squared residual.
- **Coefficient interpretation:** $w_j$ = expected change in $y$ per unit change in $x_j$, holding others fixed.
- **Gauss-Markov:** under linearity, exogeneity, homoscedasticity, and no autocorrelation, OLS is the **B**est **L**inear **U**nbiased **E**stimator.
- Regularized cousins: **Ridge** (L2), **Lasso** (L1), **Elastic Net**.

## 3. Math
Objective (MSE):
$$\mathcal{L}(w) = \frac{1}{n}\sum_{i=1}^{n}\big(y_i - w^\top x_i\big)^2 = \frac{1}{n}\lVert y - Xw\rVert_2^2$$
Closed-form **normal equations** (absorbing $b$ into $w$ via a column of ones):
$$\hat w = (X^\top X)^{-1} X^\top y$$
Gradient for iterative solving:
$$\nabla_w \mathcal{L} = -\frac{2}{n}X^\top (y - Xw)$$

## 4. How it works
1. Assemble the design matrix $X$ (add an intercept column).
2. Solve either in **closed form** (normal equations / SVD, good for modest dimensions) or by **gradient descent** (large/sparse data).
3. In practice libraries use a stable **QR or SVD** decomposition rather than inverting $X^\top X$ directly.
4. Inspect coefficients, residuals, and $R^2$; check assumptions with residual plots.

## 5. When to use / When not to
- ✅ A fast, interpretable **baseline** for any continuous target.
- ✅ When you need coefficient-level explanations or inference (confidence intervals, p-values).
- ✅ Approximately linear relationships, or after feature transforms (logs, polynomials).
- ❌ Strong non-linearity or interactions (unless engineered) — trees/GBMs win.
- ❌ Many correlated features → unstable coefficients; regularize instead.

## 6. Common pitfalls & gotchas
- **Multicollinearity** inflates coefficient variance and flips signs; check VIF, use Ridge.
- **Outliers** dominate squared loss; consider robust (Huber) regression or transform the target.
- **Heteroscedasticity / non-normal residuals** break OLS inference (not the point estimates).
- **Extrapolation** beyond the training range is unreliable.
- **Correlation ≠ causation** — coefficients are associational unless you have a causal design.
- Forgetting to **scale** matters only for regularized/iterative fits, not plain OLS predictions.

## 7. Code
```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

lr = LinearRegression().fit(X_train, y_train)
pred = lr.predict(X_test)

print("RMSE:", mean_squared_error(y_test, pred, squared=False))
print("R^2 :", r2_score(y_test, pred))
print("coefs:", lr.coef_, "intercept:", lr.intercept_)

# regularized alternative when features are many/correlated
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
```

## 8. Interview / viva questions
- Q: What does OLS minimize, and why squared error?
  - A: The sum of squared residuals; squaring makes the objective smooth and convex with a closed-form solution, and corresponds to the MLE under Gaussian noise.
- Q: When does the normal-equation solution fail?
  - A: When $X^\top X$ is singular/ill-conditioned (perfect multicollinearity or $p > n$); use SVD or add regularization.
- Q: How do you interpret a coefficient?
  - A: Expected change in the target per unit change in that feature, holding the others fixed.
- Q: What are the OLS/Gauss-Markov assumptions?
  - A: Linearity, exogeneity (errors uncorrelated with X), homoscedasticity, and no autocorrelation give BLUE; normality is only needed for exact inference.

## 9. References
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 3.
- Gelman & Hill — *Data Analysis Using Regression and Multilevel/Hierarchical Models*.
- scikit-learn User Guide: Linear Models.

---
> _Status: 🟢 done._
