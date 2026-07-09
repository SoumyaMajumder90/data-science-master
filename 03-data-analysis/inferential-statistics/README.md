# Inferential Statistics

> Using a sample of data to draw conclusions — with quantified uncertainty — about a larger population it was drawn from.

| | |
|---|---|
| **Category** | Data Analysis & Statistics |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [Descriptive Statistics](../descriptive-statistics/), [Probability](../../01-foundations/mathematics/probability/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
You can't survey every voter, but you can survey 1,000 and still say something trustworthy about all of them — as long as you're honest about how uncertain that estimate is. Inferential statistics is the bridge from "what I measured in my sample" to "what's probably true of the population," and it always comes with a margin of error, because a different sample would have given a slightly different answer.

## 2. Formal definition / Key concepts
- **Population** — the entire group you want to know about; **sample** — the subset you actually measured.
- **Parameter** (population, e.g. $\mu$) vs **statistic** (sample, e.g. $\bar{x}$) — the statistic estimates the parameter.
- **Sampling distribution** — the distribution of a statistic (e.g. $\bar x$) across many hypothetical samples of the same size.
- **Central Limit Theorem (CLT)** — the sampling distribution of the mean approaches a normal distribution as $n$ grows, regardless of the population's original distribution.
- **Standard error (SE)** — the standard deviation of a statistic's sampling distribution; quantifies how much the statistic would vary sample to sample.
- **Confidence interval (CI)** — a range that would contain the true parameter in a stated proportion (e.g. 95%) of repeated samples.

## 3. Math
Standard error of the mean:
$$SE_{\bar x} = \frac{\sigma}{\sqrt{n}} \approx \frac{s}{\sqrt{n}}$$

95% confidence interval for the mean (large $n$, or $t$-distribution for small $n$):
$$\bar x \pm z_{0.975} \cdot SE_{\bar x} \quad (z_{0.975} \approx 1.96)$$

Central Limit Theorem (informal): for i.i.d. $X_1,\dots,X_n$ with mean $\mu$ and variance $\sigma^2$,
$$\frac{\bar X_n - \mu}{\sigma/\sqrt n} \xrightarrow{d} \mathcal{N}(0,1) \text{ as } n \to \infty$$

## 4. How it works
1. **Define the population and draw a (ideally random) sample.**
2. **Compute a sample statistic** (mean, proportion, difference of means, etc.).
3. **Quantify uncertainty** — derive the statistic's standard error, either analytically (CLT-based formulas) or via **bootstrapping** (resampling with replacement).
4. **Build a confidence interval** or run a **hypothesis test** to make a probabilistic statement about the population parameter.
5. **Interpret carefully** — a 95% CI means 95% of such intervals (across repeated sampling) would contain the true parameter, not "95% probability the parameter is in this specific interval."

## 5. When to use / When not to
- ✅ When you have a sample and need to generalize to a population (survey results, A/B test outcomes, clinical trial effects).
- ✅ When you need to communicate uncertainty, not just a point estimate.
- ❌ When your sample isn't representative of the population (selection bias) — inference will be systematically wrong, no matter how large $n$ is.
- ❌ When you already have the entire population's data — there's no sampling uncertainty to quantify; just report descriptive statistics.

## 6. Common pitfalls & gotchas
- **Misinterpreting confidence intervals** as "95% probability the true value is in this range" — the correct frequentist interpretation is about the *procedure's* long-run coverage, not this one interval.
- **Non-random samples** (convenience sampling, survivorship bias) break the theoretical guarantees entirely — no amount of statistical machinery fixes a biased sample.
- Forgetting the CLT needs a **large enough $n$** for skewed populations; for small samples from a non-normal population, use exact/non-parametric methods or the $t$-distribution.
- Confusing **standard deviation** (spread of the data) with **standard error** (spread of the *estimate*) — SE shrinks as $n$ grows, SD doesn't.
- Multiple comparisons without correction inflate the effective error rate (see [Hypothesis Testing](../hypothesis-testing/)).

## 7. Code
```python
import numpy as np
from scipy import stats

sample = np.array([...])
mean, se = sample.mean(), stats.sem(sample)
ci = stats.t.interval(0.95, df=len(sample) - 1, loc=mean, scale=se)

# Bootstrap alternative (no distributional assumption)
boot_means = [np.random.choice(sample, size=len(sample), replace=True).mean()
              for _ in range(10_000)]
ci_bootstrap = np.percentile(boot_means, [2.5, 97.5])
```

## 8. Interview / viva questions
- Q: What does a 95% confidence interval actually mean?
  - A: If you repeated the sampling process many times and built a CI each time, about 95% of those intervals would contain the true population parameter. It's a statement about the procedure, not a 95% probability for this specific interval.
- Q: Why does standard error shrink as sample size grows, but standard deviation doesn't?
  - A: SD measures the inherent variability of individual data points, which is a property of the population and doesn't change. SE measures the variability of the *sample mean* across repeated samples, which shrinks as $1/\sqrt{n}$ because averaging more points cancels out noise.
- Q: Your sample is huge (n=1,000,000) but was collected via an opt-in web survey. Is your inference valid?
  - A: Not necessarily — a large $n$ reduces variance but does nothing about bias. If the sample isn't representative (selection bias), the point estimate itself is wrong regardless of how tight the confidence interval looks.

## 9. References
- Wasserman, L. — *All of Statistics*, ch. 5–6, 10.
- Efron, B. & Tibshirani, R. — *An Introduction to the Bootstrap*.
- Casella, G. & Berger, R. — *Statistical Inference*.

---
> _Status: 🟢 done._
