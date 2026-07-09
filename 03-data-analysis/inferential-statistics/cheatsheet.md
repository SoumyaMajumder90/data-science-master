# Inferential Statistics — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Use a sample statistic plus its uncertainty (standard error) to make quantified claims about a population.

## Key formulas
- Standard error of mean: $SE = s/\sqrt{n}$
- 95% CI: $\bar x \pm 1.96 \cdot SE$
- CLT: $\bar X_n$ → Normal as $n \to \infty$, regardless of population shape

## Must-know facts
- Parameter (population, fixed) vs statistic (sample, estimate) vs SE (spread of the statistic).
- CI interpretation: long-run coverage of the *procedure*, not a probability on one interval.
- Bigger $n$ shrinks SE (precision) but does **not** fix a biased/non-random sample.
- Bootstrapping estimates uncertainty without assuming a distribution.

## Quick decisions
| Situation | Do this |
|---|---|
| Large $n$, want mean CI | Normal ($z$) approximation |
| Small $n$, unknown population SD | $t$-distribution |
| Don't trust distributional assumptions | Bootstrap resampling |
| Non-random sample | Fix sampling design — stats can't fix bias |

## Common mistakes
- Saying "95% probability the parameter is in this interval."
- Confusing SD (data spread) with SE (estimate spread).
- Trusting a huge biased sample because $n$ is large.

## One-liner code
```python
stats.t.interval(0.95, df=n-1, loc=mean, scale=stats.sem(sample))
```
