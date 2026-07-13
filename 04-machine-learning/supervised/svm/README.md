# Support Vector Machines (SVM)

> A classifier that finds the decision boundary with the **largest margin** to the nearest points, and uses the kernel trick to draw non-linear boundaries.

| | |
|---|---|
| **Category** | Machine Learning → Supervised |
| **Difficulty** | 🟨 Intermediate / 🟥 Advanced |
| **Prerequisites** | [Linear Algebra](../../../01-foundations/mathematics/linear-algebra/), [Optimization](../../../01-foundations/mathematics/optimization/), [Logistic Regression](../logistic-regression/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
Many lines can separate two classes; which is best? SVM picks the one that sits in the **widest possible street** between them — maximizing the gap (margin) to the closest points of each class. Those closest points are the **support vectors**; they alone define the boundary. When classes aren't linearly separable, the **kernel trick** implicitly lifts the data into a higher-dimensional space where a straight boundary works, without ever computing those coordinates.

## 2. Formal definition / Key concepts
- **Maximum-margin classifier:** find the hyperplane $w^\top x + b = 0$ that maximizes the margin $2/\lVert w\rVert$.
- **Support vectors:** the training points on or inside the margin — the only ones that matter.
- **Soft margin:** slack variables $\xi_i$ allow some violations; hyperparameter $C$ trades margin width against violations.
- **Kernel trick:** replace inner products $x_i^\top x_j$ with $K(x_i, x_j)$ (linear, polynomial, **RBF/Gaussian**, sigmoid) to get non-linear boundaries.
- Equivalent to minimizing **hinge loss** + L2 regularization.

## 3. Math
Soft-margin primal problem:
$$\min_{w,b,\xi}\ \tfrac12\lVert w\rVert^2 + C\sum_i \xi_i \quad \text{s.t. } y_i(w^\top x_i + b)\ge 1-\xi_i,\ \xi_i\ge 0$$
Dual (kernelized), which is what solvers optimize:
$$\max_\alpha\ \sum_i \alpha_i - \tfrac12\sum_{i,j}\alpha_i\alpha_j y_i y_j K(x_i,x_j)\quad \text{s.t. } 0\le\alpha_i\le C,\ \sum_i\alpha_i y_i = 0$$
RBF kernel: $K(x,x') = \exp(-\gamma\lVert x-x'\rVert^2)$. Hinge-loss view: $\min_w \tfrac1n\sum_i \max(0, 1-y_i(w^\top x_i+b)) + \tfrac{\lambda}{2}\lVert w\rVert^2$.

## 4. How it works
1. Standardize features (distances/kernels are scale-sensitive).
2. Choose a kernel: **linear** for high-dim/sparse data (text), **RBF** for general non-linear problems.
3. Solve the (dual) quadratic program; only support vectors get non-zero $\alpha_i$.
4. Tune $C$ (regularization) and, for RBF, $\gamma$ (reach of each point) — usually by grid/randomized search.
5. Predict via $\text{sign}\big(\sum_i \alpha_i y_i K(x_i, x) + b\big)$.

## 5. When to use / When not to
- ✅ Small-to-medium datasets with clear margins; **high-dimensional** data (text with linear SVM).
- ✅ When a robust max-margin boundary and strong theory are desirable.
- ✅ Non-linear boundaries via RBF when you can afford tuning.
- ❌ Very **large datasets** — kernel SVM is roughly $O(n^2)$–$O(n^3)$; use linear SVM (`LinearSVC`/SGD) or trees.
- ❌ When you need calibrated probabilities natively (SVM outputs scores; probabilities need Platt scaling).
- ❌ Lots of noise/overlap where margins are meaningless.

## 6. Common pitfalls & gotchas
- **Not scaling features** wrecks RBF/poly kernels — always standardize.
- **$C$ and $\gamma$** dominate performance: large $C$/$\gamma$ overfit, small ones underfit; tune together.
- **Poor scaling with $n$** — kernel SVMs don't fit millions of rows.
- **No native probabilities** — `probability=True` runs expensive internal CV (Platt scaling).
- **Multiclass** is one-vs-one/one-vs-rest under the hood.
- Imbalance: use `class_weight="balanced"`.

## 7. Code
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
grid = GridSearchCV(pipe, {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.01, 0.1, 1],
}, cv=5, scoring="f1")
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
```

## 8. Interview / viva questions
- Q: What are support vectors?
  - A: The training points lying on or within the margin; they alone determine the hyperplane, so the model is sparse in the data.
- Q: What does the kernel trick achieve?
  - A: It computes inner products in a high-dimensional feature space via $K(x,x')$ without ever mapping the data there, enabling non-linear boundaries cheaply.
- Q: What do $C$ and $\gamma$ control?
  - A: $C$ trades margin width for training violations (regularization); RBF $\gamma$ sets how far each point's influence reaches — both large → overfitting.
- Q: Why doesn't SVM scale to huge datasets?
  - A: Kernel SVM solves a quadratic program that is roughly $O(n^2)$–$O(n^3)$ in the number of samples; linear SVM via SGD scales far better.

## 9. References
- Cortes & Vapnik (1995) — "Support-Vector Networks", *Machine Learning*.
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 12.
- scikit-learn User Guide: Support Vector Machines.

---
> _Status: 🟢 done._
