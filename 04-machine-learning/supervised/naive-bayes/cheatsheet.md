# Naive Bayes — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Bayes' theorem + conditional-independence assumption. Fast, data-efficient, a strong text-classification baseline.

## Key formulas
- $P(y\mid x)\propto P(y)\prod_j P(x_j\mid y)$
- Predict: $\arg\max_c \log P(c) + \sum_j \log P(x_j\mid c)$
- Smoothing: $P(w\mid c)=\frac{N_{wc}+\alpha}{N_c+\alpha|V|}$

## Must-know facts
- **Generative** (models $P(x\mid y)$); log-space to avoid underflow.
- Variants: **Multinomial** (counts), **Bernoulli** (binary), **Gaussian** (continuous), **Complement** (imbalance).
- Always **smooth** ($\alpha>0$) to fix zero-frequency.
- Probabilities are **overconfident** (correlated features double-counted).
- Accuracy stays good because only argmax matters.

## Quick decisions
| Data | Variant |
|---|---|
| Word counts / TF-IDF | Multinomial |
| Binary features | Bernoulli |
| Continuous features | Gaussian |
| Imbalanced text | Complement |

## Common mistakes
- No smoothing → zero probabilities.
- Wrong variant for feature type.
- Trusting the probability values (need calibration).

## One-liner code
```python
make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=1.0)).fit(text_train, y_train)
```
