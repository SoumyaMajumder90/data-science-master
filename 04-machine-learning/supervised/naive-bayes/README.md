# Naive Bayes

> A probabilistic classifier that applies Bayes' theorem with a "naive" assumption that features are conditionally independent given the class — fast, and surprisingly strong for text.

| | |
|---|---|
| **Category** | Machine Learning → Supervised |
| **Difficulty** | 🟩 Beginner / 🟨 Intermediate |
| **Prerequisites** | [Probability](../../../01-foundations/mathematics/probability/), [Bag of Words & TF-IDF](../../../06-nlp/bag-of-words-and-tfidf/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
To decide if an email is spam, look at its words. Bayes' theorem lets you flip "how likely are these words if it's spam?" into "how likely is spam given these words?". The catch is estimating the joint probability of all words together — so Naive Bayes makes a bold simplifying bet: pretend every word appears **independently** of the others given the class. This is almost never literally true, but it makes the math trivially fast and, for high-dimensional text, works remarkably well.

## 2. Formal definition / Key concepts
- **Bayes' theorem:** posterior ∝ likelihood × prior.
- **Naive assumption:** features conditionally independent given the class, so the joint likelihood factorizes into a product.
- **Variants by likelihood model:**
  - **Multinomial NB** — word counts (text classification).
  - **Bernoulli NB** — binary word presence/absence.
  - **Gaussian NB** — continuous features assumed normal per class.
  - **Complement NB** — robust to class imbalance in text.
- A **generative** model (it models $P(x\mid y)$), unlike discriminative logistic regression.

## 3. Math
Bayes' rule and the naive factorization:
$$P(y\mid x) = \frac{P(y)\,P(x\mid y)}{P(x)} \ \propto\ P(y)\prod_{j=1}^{p} P(x_j\mid y)$$
Predict the MAP class:
$$\hat y = \arg\max_{c}\ \log P(y=c) + \sum_{j=1}^{p}\log P(x_j\mid y=c)$$
Multinomial likelihood with **Laplace (add-α) smoothing** to avoid zero probabilities:
$$P(w\mid c) = \frac{N_{wc} + \alpha}{N_c + \alpha\,|V|}$$

## 4. How it works
1. Estimate class **priors** $P(y=c)$ from label frequencies.
2. Estimate per-feature **likelihoods** $P(x_j\mid c)$ (counts for multinomial, mean/variance for Gaussian).
3. Apply **smoothing** so unseen feature/class combinations don't zero out the product.
4. For a new point, sum log-prior and log-likelihoods per class; pick the argmax.
5. Work in **log space** to avoid numerical underflow from multiplying many small probabilities.

## 5. When to use / When not to
- ✅ **Text classification** (spam, sentiment, topic) with bag-of-words / TF-IDF — a classic strong baseline.
- ✅ High-dimensional, sparse data; very fast to train and predict; great with little data.
- ✅ Online/streaming settings (`partial_fit`) and as a quick benchmark.
- ❌ When features are **strongly correlated** — the independence assumption hurts and probabilities become overconfident.
- ❌ When you need **well-calibrated** probabilities (NB estimates are typically pushed toward 0/1).
- ❌ Regression, or problems where feature interactions carry the signal.

## 6. Common pitfalls & gotchas
- **Zero-frequency problem** — an unseen word gives $P=0$ and kills the whole product; always use smoothing ($\alpha>0$).
- **Overconfident probabilities** — correlated features are double-counted, so scores near 0/1 aren't trustworthy; calibrate if you need real probabilities.
- **Wrong variant** — using Gaussian NB on counts or Multinomial NB on continuous features.
- **Class imbalance** skews priors; consider Complement NB or adjust priors.
- Despite violated assumptions, **classification accuracy** often stays good because only the argmax matters, not the exact probabilities.

## 7. Code
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

clf = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2),
    MultinomialNB(alpha=1.0)          # alpha = Laplace smoothing
)
clf.fit(text_train, y_train)
print("accuracy:", clf.score(text_test, y_test))
```

## 8. Interview / viva questions
- Q: What is "naive" about Naive Bayes?
  - A: The assumption that all features are conditionally independent given the class, which lets the joint likelihood factorize into a product.
- Q: Why does it work well for text despite that being false?
  - A: In high dimensions the independence errors often cancel, and classification only needs the correct argmax, not exact probabilities.
- Q: What is the zero-frequency problem and its fix?
  - A: An unseen feature value yields probability 0, zeroing the product; Laplace/add-α smoothing prevents it.
- Q: Naive Bayes vs logistic regression?
  - A: NB is generative (models $P(x\mid y)$) and fast/data-efficient; logistic regression is discriminative (models $P(y\mid x)$) and usually more accurate with enough data and correlated features.

## 9. References
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, ch. 6.6.
- Manning, Raghavan, Schütze — *Introduction to Information Retrieval*, ch. 13 (text classification & NB).
- scikit-learn User Guide: Naive Bayes.

---
> _Status: 🟢 done._
