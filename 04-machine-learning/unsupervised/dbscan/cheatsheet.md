# DBSCAN — Cheatsheet

> Quick recall. For depth see [README.md](./README.md).

**TL;DR:** Density-based clustering: grow clusters from dense cores, flag sparse points as noise. Arbitrary shapes, no $k$, built-in outliers.

## Key formulas
- $\varepsilon$-neighborhood: $N_\varepsilon(p)=\{q: d(p,q)\le\varepsilon\}$
- Core point: $|N_\varepsilon(p)|\ge$ minPts
- Heuristic: minPts $\approx 2\cdot$ dim; $\varepsilon$ from k-distance knee.

## Must-know facts
- Params: **eps** ($\varepsilon$) and **minPts**.
- Point types: **core / border / noise (-1)**.
- Finds arbitrary shapes, decides cluster count itself.
- **Scale** features first.
- Fails on **varying density** → use HDBSCAN/OPTICS.

## Quick decisions
| Situation | Do this |
|---|---|
| Choose $\varepsilon$ | k-distance elbow plot |
| Varying densities | HDBSCAN / OPTICS |
| High dimensions | Reduce dims first (or avoid) |
| Everything is noise | ↑ $\varepsilon$ or ↓ minPts |
| One giant cluster | ↓ $\varepsilon$ |

## Common mistakes
- Not scaling features.
- Guessing $\varepsilon$ instead of using the k-distance plot.
- Using a single $\varepsilon$ on varying-density data.

## One-liner code
```python
DBSCAN(eps=0.5, min_samples=5).fit_predict(StandardScaler().fit_transform(X))  # -1 = noise
```
