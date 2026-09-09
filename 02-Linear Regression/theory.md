# Chapter 2 — Linear Regression

*(Theory + math only in this chapter — no code yet, per Varun's request. Code comes in a follow-up once the concept and math are solid.)*

## 0. Where this sits in the big picture

Recap from Chapter 1: Linear Regression is your first real algorithm, and it's a perfect one to start with because it's:
- **Supervised** (Chapter 1, Section 1) — trained on labeled data (features + a known numeric answer).
- **Regression** (Chapter 1, Section 1) — predicts a **number**, not a category.
- **Parametric** (Chapter 1, Section 6) — learns a small, *fixed* set of numbers (parameters) from the data, then throws the raw training data away and just uses those numbers to predict.

Everything below is building toward one very concrete goal: turning "distill 1,000 houses down to 2 numbers" (the parametric idea from Chapter 1) into an actual, precise, step-by-step method.

---

## 1. What problem is Linear Regression actually solving?

**The question it answers:** *"Given one or more input variables, what straight-line relationship best predicts a numeric output?"*

### The real-world story

Recall the house price example: square footage → price. If you plot 20 real houses on a graph (x-axis = sqft, y-axis = price), the dots won't form a perfect straight line — real-world data is noisy. But they'll often show a clear **trend**: bigger houses tend to cost more, and that trend looks roughly like a straight line, just with scatter around it.

```
 price (₹ lakhs)
   │                                    •
   │                              •  •
   │                        •  •
   │                  •  •
   │            •  •
   │      •  •
   │   •
   └─────────────────────────────────────  sqft
```

Linear Regression's job: **find the one straight line that best fits through that scatter of points**, so that for any new square footage, you can read off a predicted price from the line — even for a square footage value you've never actually seen in your data.

### Why a straight line, specifically?

Because it's the simplest possible shape a relationship between two numbers can take (Chapter 1, Section 6's "parametric = assume a shape" idea, made concrete: the assumed shape here is literally *a straight line*). If a straight line describes your data reasonably well, you get a model that's fast, and — crucially — **interpretable**: every part of the line has a plain-English meaning, which we'll unpack next.

---

## 2. The equation of a line (the formula, unpacked piece by piece)

You've seen this equation in school, possibly written as `y = mx + c`. In ML, it's usually written as:

$$
\hat{y} = \beta_0 + \beta_1 x
$$

This is the **exact same equation**, just relabeled for ML conventions. Here's the translation:

| School notation | ML notation | What it means |
|---|---|---|
| `y` (actual) | `y` | The **actual, real, known** value from your training data (e.g. the real price a house sold for) |
| — | $\hat{y}$ ("y-hat") | The model's **predicted** value — note this is *different* from `y`. The little hat symbol always means "predicted," not "actual." This distinction matters a lot — see Section 3. |
| `x` | `x` | The **input/feature** (e.g. square footage) |
| `m` (slope) | $\beta_1$ ("beta-one") | The **slope** — how much $\hat{y}$ changes for every 1-unit increase in `x` |
| `c` (intercept) | $\beta_0$ ("beta-zero" or "the intercept") | The value of $\hat{y}$ when `x = 0` — where the line crosses the y-axis |

**These two numbers, $\beta_0$ and $\beta_1$, are the entire "model."** This is the parametric idea from Chapter 1 made completely literal: no matter whether you trained on 20 houses or 20,000 houses, the trained model is *always* just these 2 numbers. Once you have them, you throw away the original data — plugging any new `x` into the formula gives you a prediction instantly.

### Worked numeric example

Say training on historical houses produces: $\beta_0 = 20$, $\beta_1 = 4.5$ (price in ₹ lakhs, sqft in hundreds of sqft, just for clean numbers).

$$
\hat{y} = 20 + 4.5x
$$

**Reading this in plain English (this is the interpretability payoff mentioned in Chapter 1, Section 6):**
- $\beta_0 = 20$: a (hypothetical) house with 0 sqft would be predicted at ₹20 lakhs — usually not meaningful on its own, it's just the mathematical anchor point of the line, the baseline the slope builds on top of.
- $\beta_1 = 4.5$: for every extra 100 sqft, predicted price goes up by ₹4.5 lakhs. **This is the number a business person actually cares about** — "each extra 100 sqft is worth about 4.5 lakhs in this market."

For a house with `x = 15` (1,500 sqft): $\hat{y} = 20 + 4.5 \times 15 = 20 + 67.5 = 87.5$ → predicted price ≈ ₹87.5 lakhs.

---

## 3. From one feature to many — Multiple Linear Regression

Real problems rarely have just one input. House price probably depends on square footage **and** bedrooms **and** distance to city center, all at once. The formula extends naturally:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \dots + \beta_n x_n
$$

| Term | Meaning |
|---|---|
| $x_1, x_2, x_3, \dots$ | Each individual feature (sqft, bedrooms, distance, ...) |
| $\beta_1, \beta_2, \beta_3, \dots$ | Each feature's own slope — "how much does $\hat{y}$ change per 1-unit change in *this specific* feature, holding all other features constant" |
| $\beta_0$ | Still the intercept — the baseline value when *every* feature is 0 |

**Why "holding all other features constant" matters:** $\beta_1$ (sqft's slope) tells you sqft's effect *isolated* from bedrooms and distance's effects — each $\beta$ is that feature's own individual pull on the prediction, with the others' influence separated out. This is what makes multiple regression genuinely useful over just eyeballing one variable at a time: it disentangles overlapping effects (e.g. bigger houses often also have more bedrooms — multiple regression can tell you sqft's effect *specifically*, separate from bedroom count's effect).

**Naming:** one feature → **Simple Linear Regression** (Section 2's version). Two or more features → **Multiple Linear Regression** (this section). Same core idea, just more $\beta$'s.

---

## 4. How do you find the "best" line? (the actual learning part)

Section 2 assumed you're handed $\beta_0$ and $\beta_1$ already. This section is about **how training actually produces those numbers** — this is the real "learning" in Linear Regression.

### The problem: infinite possible lines

For any scatter of points, you could draw *infinitely many* straight lines through/near them. Some fit well, some fit terribly. You need a precise, numeric way to say **"this line is better than that line"** — not just eyeballing it.

### Step 1 — Define "error" for a single point: the residual

For any candidate line, and any one training point, compare what the line *predicts* ($\hat{y}$) against what *actually* happened ($y$):

$$
\text{residual} = y - \hat{y}
$$

This is called the **residual** (or error) — how far off the line's prediction was from reality, for that one point. A residual of 0 means a perfect prediction for that point. A large residual (positive or negative) means the line missed badly for that point.

### Step 2 — Why not just add up all the residuals?

Tempting first idea: sum the residuals across all points, and pick the line that makes that sum smallest. **This fails**, for a reason you already know from `statistics-self-learning`: positive and negative residuals **cancel each other out**. A line that's wildly wrong (too high on half the points, too low on the other half) could still sum to ~0, falsely looking "perfect." (This is the exact same cancellation problem that variance's calculation solves by squaring deviations — see `statistics-self-learning` Chapter 1, Section 2. Linear Regression borrows the identical fix.)

### Step 3 — Square the residuals, then sum: the Sum of Squared Errors (SSE)

$$
\text{SSE} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Reading this formula piece by piece:
- $y_i - \hat{y}_i$ → the residual for point $i$ (actual minus predicted)
- $(\dots)^2$ → square it (removes the sign, punishes big misses harder than small ones — same reasoning as variance)
- $\sum_{i=1}^{n}$ → sum this across every single training point, from the 1st to the $n$-th

**SSE is a single number that scores how bad a candidate line is, across the *entire* dataset at once.** Lower SSE = better-fitting line. A perfect line (impossible in real noisy data, but conceptually) would have SSE = 0.

### Step 4 — Ordinary Least Squares (OLS): pick the line that minimizes SSE

**Ordinary Least Squares (OLS)** is simply the *method/rule* for choosing $\beta_0$ and $\beta_1$: **pick whichever values make SSE as small as mathematically possible.** ("Least squares" = literally "smallest sum of squares" — the name directly describes what it does.)

This is why Linear Regression is often described as "fitting a line that minimizes squared error" — that sentence is now fully unpacked: *squared* = Step 3's squaring trick, *error* = Step 1's residual, *minimizes* = Step 4's OLS search for the smallest possible SSE.

### Worked example, small enough to do by hand

4 houses, 1 feature (sqft in hundreds), actual price in ₹ lakhs:

| sqft (x) | actual price (y) |
|---|---|
| 10 | 65 |
| 15 | 88 |
| 20 | 105 |
| 25 | 130 |

Say we're comparing two **candidate lines** (pretend these were proposed, not yet the "trained" answer):

**Candidate Line A:** $\hat{y} = 20 + 4x$

| x | y (actual) | $\hat{y}$ (predicted) | residual ($y-\hat{y}$) | squared residual |
|---|---|---|---|---|
| 10 | 65 | 20+40=60 | 5 | 25 |
| 15 | 88 | 20+60=80 | 8 | 64 |
| 20 | 105 | 20+80=100 | 5 | 25 |
| 25 | 130 | 20+100=120 | 10 | 100 |

SSE for Line A = 25+64+25+100 = **214**

**Candidate Line B:** $\hat{y} = 15 + 4.5x$

| x | y (actual) | $\hat{y}$ (predicted) | residual ($y-\hat{y}$) | squared residual |
|---|---|---|---|---|
| 10 | 65 | 15+45=60 | 5 | 25 |
| 15 | 88 | 15+67.5=82.5 | 5.5 | 30.25 |
| 20 | 105 | 15+90=105 | 0 | 0 |
| 25 | 130 | 15+112.5=127.5 | 2.5 | 6.25 |

SSE for Line B = 25+30.25+0+6.25 = **61.5**

**Line B has a much lower SSE (61.5 vs 214) → Line B fits this data better than Line A.** OLS is the systematic method that searches across *every possible* combination of $\beta_0, \beta_1$ (not just these 2 hand-picked candidates) and mathematically guarantees finding the exact combination with the lowest possible SSE — that guaranteed-best combination is what "training" a Linear Regression model actually produces.

> You won't hand-search candidate lines like this in practice — OLS has a direct mathematical solution (and libraries compute it instantly). This worked example exists purely so "minimizing SSE" stops being an abstract phrase and becomes something you've verified by hand.

---

## 5. Key assumptions Linear Regression makes

Because Linear Regression is **parametric** (Chapter 1, Section 6), it assumes a shape upfront — literally, a straight-line relationship. That assumption only holds up if the real data satisfies a few conditions. Knowing these matters because violating them means your $\beta$'s (and any predictions from them) become unreliable:

| Assumption | Plain-English meaning | Why it matters |
|---|---|---|
| **Linearity** | The real relationship between x and y is actually (roughly) a straight line | If the true pattern is curved (e.g. exponential), a straight line will systematically miss — no amount of "better fitting" fixes a wrong shape |
| **Independence of errors** | One point's residual shouldn't predict another point's residual | Common violation: time-series data where today's error correlates with yesterday's |
| **Homoscedasticity** (constant variance of errors) | The "typical size of a miss" should be roughly the same across the whole range of x, not way bigger for large x than small x | If errors fan out (get bigger for bigger houses, say), the model is more trustworthy for some x-ranges than others, and standard error estimates become unreliable |
| **Normality of errors** | The residuals, plotted as a distribution, should roughly look like a Normal (bell curve) distribution | Ties directly to `statistics-self-learning` Chapter 1's Normal Distribution section — many of Linear Regression's statistical guarantees (like confidence intervals on $\beta$) rely on this |
| **No severe multicollinearity** (multiple regression only) | Input features shouldn't be near-duplicates of each other (e.g. both `length_cm` and `length_inches` as separate features — recall Chapter 1, Section 6's dimensionality reduction example) | If two features carry almost the same information, OLS can't cleanly tell which one deserves credit for the effect, and the individual $\beta$'s become unstable/hard to trust, even if overall predictions still look fine |

**How to check these in practice:** mostly done by plotting residuals (residual vs. predicted value, and a histogram of residuals) *after* fitting — this is a "diagnose after the fact" step, same spirit as checking skewness/kurtosis *after* computing them in `statistics-self-learning`, not something you can fully verify just by staring at raw data beforehand.

---

## 6. Evaluating how good the fit is (beyond just SSE)

SSE (Section 4) is great for *comparing* candidate lines during training, but it's a bad number to *report* to a human — its scale depends on your data's units and how many points you have (100 points will almost always have a bigger SSE than 10 points, even with an equally good fit). Two more human-readable metrics fix this — full computational detail comes in a later evaluation-metrics chapter, but the concepts belong here since they directly follow from Section 4's math:

- **RMSE (Root Mean Squared Error):** $\sqrt{\text{SSE}/n}$ — brings SSE back to the original unit (₹ lakhs, not "lakhs²"), same "undo the squaring" logic as Standard Deviation in `statistics-self-learning`. Reads as: "on average, predictions are off by about X lakhs."
- **R² (R-squared):** a 0-to-1 score (roughly) answering *"what fraction of the variation in y does this line actually explain?"* R² = 1 → the line perfectly explains all the variation in the data. R² = 0 → the line explains nothing (you'd have done just as well predicting the average `y` for everyone, ignoring x entirely).

---

## The full mental model, tied together

1. **Assume a shape** — a straight line, $\hat{y} = \beta_0 + \beta_1 x$ (or the multi-feature version) — this is the parametric assumption from Chapter 1.
2. **Define "wrong"** — for a candidate line, compute each point's residual ($y - \hat{y}$), square it, sum it all up → SSE.
3. **Find the best line** — use Ordinary Least Squares (OLS) to pick the exact $\beta_0, \beta_1, \dots$ that minimizes SSE across the whole dataset.
4. **Sanity-check the assumptions** — linearity, independent/constant-variance/normal errors, no multicollinearity — because the whole method only stays trustworthy if these roughly hold.
5. **Report fit quality in human terms** — RMSE (typical miss, in real units) and R² (how much of the pattern the line actually explains).

Once this is solid, the next step (a follow-up chapter/section) is turning this into actual code — fitting a real Linear Regression model on real data using a library, and seeing these exact concepts (residuals, SSE, R²) show up as real numbers you compute, not hand-worked examples.
