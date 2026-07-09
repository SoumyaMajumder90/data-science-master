# Data Visualization

> The practice of encoding data into visual form (position, length, color, shape) so patterns, comparisons, and outliers are grasped faster than by reading numbers.

| | |
|---|---|
| **Category** | Data Analysis & Statistics |
| **Difficulty** | 🟩 Beginner |
| **Prerequisites** | [Descriptive Statistics](../descriptive-statistics/) |
| **Cheatsheet** | [cheatsheet.md](./cheatsheet.md) |

---

## 1. Intuition
Humans are far better at comparing lines, bars, and positions than scanning tables of numbers. A good chart answers a question at a glance; a bad chart (or the wrong chart type) can mislead just as easily as a lie in text. Visualization is both an exploratory tool (for you) and a communication tool (for others) — the design goals differ for each.

## 2. Formal definition / Key concepts
- **Chart selection** depends on the question: comparison, distribution, relationship, composition, or change over time.
- **Preattentive attributes** — visual properties (position, length, color hue) processed almost instantly by the brain; using the right one for the right data type makes charts readable at a glance.
- **Chart junk** — decorative elements (3D effects, heavy gridlines, unnecessary color) that add no information and increase cognitive load (Tufte's "data-ink ratio").
- **Exploratory vs explanatory** visualization — exploring for yourself favors speed and iteration; explaining to others favors clarity, annotation, and a single takeaway.

## 3. Math
No core formula, but visualization inherits statistics it displays:
- Histogram bin width (Freedman–Diaconis rule): $h = 2\,\dfrac{\text{IQR}(x)}{n^{1/3}}$
- KDE (kernel density estimate): $\hat{f}(x) = \dfrac{1}{nh}\sum_{i=1}^n K\!\left(\dfrac{x-x_i}{h}\right)$, where $K$ is a kernel (e.g. Gaussian) and $h$ the bandwidth.

## 4. How it works
1. **Identify the question** — comparison (bar), distribution (histogram/box/violin), relationship (scatter), composition (stacked bar/pie — use sparingly), trend (line).
2. **Pick the encoding** — position and length are perceived most accurately; area and color saturation are perceived less accurately; avoid encoding a key quantity in angle (pie charts) or 3D.
3. **Choose scale** — linear vs log (log for data spanning orders of magnitude or ratios).
4. **Reduce clutter** — remove unnecessary gridlines/borders, use direct labeling over legends when few series.
5. **Iterate for the audience** — exploratory plots can be rough; explanatory plots need a title that states the takeaway, clear axis labels, and consistent color meaning.

## 5. When to use / When not to
- ✅ Use during EDA to spot distributions, outliers, and relationships quickly.
- ✅ Use to communicate a finding — a well-designed chart persuades faster than a table.
- ✅ Use log scales when comparing quantities across orders of magnitude (e.g. income, population).
- ❌ Don't use pie charts for more than ~3–4 categories or close proportions — angle is hard to compare precisely; a bar chart usually wins.
- ❌ Don't use dual y-axes casually — they can imply a false correlation between unrelated series.
- ❌ Don't visualize as a substitute for statistical testing — "eyeballing" a trend isn't the same as confirming it's significant.

## 6. Common pitfalls & gotchas
- **Truncated y-axis** — starting a bar chart's axis above zero exaggerates differences.
- **Overplotting** — thousands of overlapping points hide density; use alpha transparency, sampling, hexbin, or 2D density plots.
- **Color misuse** — using a rainbow/qualitative palette for ordered (sequential) data, or a non-colorblind-safe palette.
- **Misleading aggregation** — a line chart of averages can hide bimodal or skewed underlying distributions (Anscombe's quartet / the Datasaurus Dozen make this vivid).
- Choosing chart type for aesthetics over clarity (e.g. 3D bar charts distort perceived height).

## 7. Code
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(df["price"], kde=True, ax=axes[0])
sns.scatterplot(data=df, x="sqft", y="price", hue="city", alpha=0.5, ax=axes[1])
plt.tight_layout()
```

## 8. Interview / viva questions
- Q: Why avoid pie charts for many categories?
  - A: Humans compare angle/area poorly compared to position or length; a sorted bar chart lets viewers compare values accurately, especially with more than a few slices.
- Q: When should you use a log scale?
  - A: When data spans several orders of magnitude, or when you care about relative (multiplicative) rather than absolute (additive) change — equal visual distances then represent equal ratios.
- Q: Give an example where visualization reveals something summary statistics hide.
  - A: Anscombe's quartet — four datasets with nearly identical mean, variance, and correlation but wildly different shapes (linear, curved, one outlier, etc.) visible only by plotting.

## 9. References
- Tufte, E. — *The Visual Display of Quantitative Information*.
- Anscombe, F. — "Graphs in Statistical Analysis" (1973) (Anscombe's quartet).
- Wilke, C. — *Fundamentals of Data Visualization* (freely available online).

---
> _Status: 🟢 done._
