# Correlation vs Causation

> Two variables moving together (correlation) does not mean one causes the other (causation) — a confounder, reverse causation, or pure chance can produce the same statistical signature.

| | |
|---|---|
| **Category** | Data Analysis & Statistics |
| **Difficulty** | 🟨 Intermediate |
| **Prerequisites** | [Descriptive Statistics](../descriptive-statistics/), [Hypothesis Testing](../hypothesis-testing/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
Ice cream sales and drowning deaths rise and fall together over the year. Ice cream doesn't cause drowning — summer heat drives both (more swimming, more ice cream). Correlation only tells you two things move together; it says nothing about *why*. Establishing causation requires either a controlled experiment or careful statistical reasoning about what else could explain the pattern.

## 2. Formal definition / Key concepts
- **Correlation** — a statistical association between two variables (e.g. Pearson's $r$ for linear association).
- **Causation** — changing $X$ actually produces a change in $Y$, holding everything else fixed.
- **Confounder** — a third variable that influences both $X$ and $Y$, creating a spurious association between them (e.g. summer heat → ice cream & drowning).
- **Reverse causation** — assuming $X \to Y$ when actually $Y \to X$.
- **Selection bias** — the sample itself was chosen in a way that creates an association not present in the population.
- **Counterfactual** — what would have happened to the same unit under the *other* treatment; causal inference is fundamentally about estimating this unobservable quantity.

## 3. Math
Pearson correlation coefficient:
$$r = \frac{\sum_i (x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum_i (x_i-\bar x)^2}\sqrt{\sum_i (y_i-\bar y)^2}}$$

Average Treatment Effect (the causal quantity we actually want), using Rubin's potential outcomes framework:
$$ATE = \mathbb{E}[Y(1) - Y(0)]$$
where $Y(1)$ and $Y(0)$ are a unit's outcome under treatment and control — only one is ever observed per unit, which is why causal inference needs either randomization or strong assumptions.

## 4. How it works
Randomization is what breaks the link between treatment assignment and confounders, which is why an RCT/[A-B test](../ab-testing/) can support causal claims that pure observational correlation cannot. When you can't randomize, causal inference tries to approximate it:
1. **Draw a causal diagram (DAG)** of plausible relationships to identify confounders.
2. **Control for confounders** — regression adjustment, matching, or stratification.
3. **Exploit natural experiments** — instrumental variables, regression discontinuity, difference-in-differences — when a source of "as-if random" variation exists.
4. **Check robustness** — sensitivity analysis for unmeasured confounding (would a plausible unobserved confounder overturn the result?).
5. **Never claim causation from correlation alone** — state the association and the specific untested assumption needed for a causal read.

## 5. When to use / When not to
- ✅ Use correlation freely for **prediction** — if $X$ reliably co-moves with $Y$, it can be a useful predictive feature even without causation.
- ✅ Use it to *generate* causal hypotheses to test experimentally.
- ❌ Never use bare correlation to justify an **intervention** ("if we make $X$ happen, $Y$ will follow") — that requires a causal claim, which needs either an experiment or explicit causal assumptions (DAG + confounder control).
- ❌ Don't compute correlations across dozens of variable pairs and report whichever is highest — with enough pairs, spurious high correlations appear by chance (see "Spurious Correlations" — Tyler Vigen's examples of nonsense correlations like Nicolas Cage films and pool drownings).

## 6. Common pitfalls & gotchas
- **Confounding** — the single most common source of "causal-looking" correlation that isn't causal.
- **Reverse causation** — e.g. "companies with higher revenue spend more on ads" could mean ads drive revenue, or successful (high-revenue) companies simply have bigger ad budgets.
- **Selection bias / collider bias** — conditioning on a variable that's a common effect of $X$ and $Y$ can *create* a spurious correlation between two otherwise unrelated variables.
- **Simpson's paradox** — a correlation can reverse sign when data is split by a lurking subgroup variable, versus aggregated.
- **Nonlinear relationships** — Pearson's $r \approx 0$ doesn't mean "no relationship," only "no *linear* relationship" (see Anscombe's quartet).
- Correlation is **symmetric** ($r_{XY}=r_{YX}$); causation is directional — correlation alone can never tell you which way an arrow points.

## 7. Code
```python
import pandas as pd

corr = df[["x", "y"]].corr(method="pearson").iloc[0, 1]

# Simple confounder check: does correlation persist after controlling for z?
import statsmodels.formula.api as smf
model = smf.ols("y ~ x + z", data=df).fit()   # z = suspected confounder
print(model.summary())   # coefficient on x, controlling for z
```

## 8. Interview / viva questions
- Q: Give a classic example of correlation without causation, and explain the confounder.
  - A: Ice cream sales correlate with drowning deaths. The confounder is hot weather — it drives both more ice cream purchases and more swimming (hence more drownings). Neither causes the other directly.
- Q: How can a randomized experiment establish causation when observational data can't?
  - A: Randomization ensures treatment assignment is independent of all confounders (measured and unmeasured) on average, so any systematic outcome difference between groups can be attributed to the treatment itself, not a lurking third variable.
- Q: You find that companies using Product X have 20% higher retention. Can you say Product X causes retention?
  - A: Not from this alone — companies that adopt X might already be more engaged/higher-quality customers (selection bias/confounding). You'd want a randomized rollout, a natural experiment, or careful confounder adjustment before making a causal claim.

## 9. References
- Pearl, J. — *The Book of Why*.
- Hernán, M. & Robins, J. — *Causal Inference: What If* (free online).
- Vigen, T. — *Spurious Correlations* (illustrative examples).

---
> _Status: 🟢 done._
