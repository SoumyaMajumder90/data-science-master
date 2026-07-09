# Hypothesis Testing

> A formal procedure for deciding whether observed data provides enough evidence to reject a default assumption (the null hypothesis) in favor of an alternative.

| | |
|---|---|
| **Category** | Data Analysis & Statistics |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [Inferential Statistics](../inferential-statistics/), [Probability](../../01-foundations/mathematics/probability/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
It's a courtroom: you assume innocence (the **null hypothesis**, $H_0$ — "no effect") until the evidence (data) is strong enough to convict beyond reasonable doubt (the **alternative**, $H_1$). You never "prove" $H_0$ true — you either find enough evidence to reject it, or you don't, in which case you simply fail to reject it (innocent doesn't mean proven innocent, just not proven guilty).

## 2. Formal definition / Key concepts
- **Null hypothesis ($H_0$)** — the default, "no effect/no difference" claim.
- **Alternative hypothesis ($H_1$/$H_a$)** — what you're trying to find evidence for.
- **Test statistic** — a number computed from the sample (e.g. $t$, $z$, $\chi^2$, $F$) that measures how far the data deviates from what $H_0$ predicts.
- **p-value** — the probability of observing a test statistic at least as extreme as the one seen, *assuming $H_0$ is true*. It is **not** the probability that $H_0$ is true.
- **Significance level ($\alpha$)** — the pre-chosen threshold (commonly 0.05) below which you reject $H_0$.
- **Type I error (α)** — rejecting a true $H_0$ (false positive). **Type II error (β)** — failing to reject a false $H_0$ (false negative). **Power** $= 1-\beta$ — probability of correctly detecting a true effect.

## 3. Math
One-sample $z$-test statistic:
$$z = \frac{\bar x - \mu_0}{\sigma / \sqrt{n}}$$

Two-sample $t$-test (unequal variance, Welch's):
$$t = \frac{\bar x_1 - \bar x_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$$

Chi-squared statistic (goodness of fit / independence):
$$\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}$$

Decision rule: reject $H_0$ if $p \le \alpha$ (equivalently, if the test statistic falls in the rejection region).

## 4. How it works
1. **State $H_0$ and $H_1$** before looking at the data.
2. **Choose $\alpha$** (and ideally do a power analysis to pick sample size in advance).
3. **Pick the right test** based on data type and design: $t$-test (means), $\chi^2$ (categorical association), ANOVA (>2 group means), Mann-Whitney U (non-parametric), etc.
4. **Compute the test statistic and p-value** from the sample.
5. **Compare $p$ to $\alpha$**: reject $H_0$ if $p \le \alpha$, otherwise fail to reject.
6. **Report effect size and CI**, not just the p-value — statistical significance isn't the same as practical importance.

## 5. When to use / When not to
- ✅ When you need a principled, pre-registered decision rule for "is this effect real or noise?"
- ✅ A/B tests, clinical trials, comparing groups or treatments.
- ❌ Don't run a test after peeking at the data repeatedly and stopping when significant ("p-hacking") — it inflates the false-positive rate.
- ❌ Don't treat a non-significant result as proof of "no effect" — it may just mean insufficient power/sample size.
- ❌ Don't rely on p-values alone for decisions with large practical stakes — pair with effect size and confidence intervals.

## 6. Common pitfalls & gotchas
- **p-value misinterpretation** — $p=0.03$ does NOT mean "3% chance $H_0$ is true"; it means "if $H_0$ were true, we'd see data this extreme 3% of the time."
- **Multiple comparisons problem** — testing many hypotheses at $\alpha=0.05$ each means a high chance of at least one false positive; correct with Bonferroni, Holm, or FDR (Benjamini-Hochberg).
- **Statistical vs practical significance** — with a huge sample, even a trivially small, meaningless effect can be "significant."
- **Optional stopping** — checking results continuously and stopping as soon as significant invalidates the p-value's guarantees.
- Confusing **one-tailed vs two-tailed** tests — using a one-tailed test to more easily hit significance without pre-registering the direction is a form of p-hacking.

## 7. Code
```python
from scipy import stats

# Two-sample t-test (Welch's, unequal variance)
t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

# Chi-squared test of independence
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

if p_value <= 0.05:
    print("Reject H0")
else:
    print("Fail to reject H0")
```

## 8. Interview / viva questions
- Q: What exactly does a p-value of 0.02 mean?
  - A: Assuming $H_0$ is true, there's a 2% chance of observing a test statistic at least this extreme purely by chance. It is not the probability that $H_0$ is true or that $H_1$ is false.
- Q: What's the difference between Type I and Type II error, and how are they related?
  - A: Type I (α) is rejecting a true null (false positive); Type II (β) is failing to reject a false null (false negative). For a fixed sample size, lowering α (stricter significance) raises β (more missed effects) — you need a larger sample to improve both simultaneously.
- Q: You ran 20 independent tests at α=0.05 with no true effects anywhere. How many false positives do you expect, and how do you fix it?
  - A: On average about 1 (20 × 0.05); the chance of at least one false positive is $1-(0.95)^{20}\approx 64\%$. Fix with a multiple-comparisons correction (Bonferroni: $\alpha/m$; or FDR control via Benjamini-Hochberg).

## 9. References
- Wasserman, L. — *All of Statistics*, ch. 10.
- Casella, G. & Berger, R. — *Statistical Inference*, ch. 8.
- Benjamini, Y. & Hochberg, Y. (1995) — "Controlling the False Discovery Rate."

---
> _Status: 🟢 done._
