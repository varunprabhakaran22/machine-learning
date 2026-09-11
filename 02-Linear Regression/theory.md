# Chapter 2 — Linear Regression

*(Theory + math only in this chapter — no code yet, per Varun's request. Code comes in a follow-up once the concept and math are solid.)*

**Notation note:** you'll see Linear Regression written a few different ways depending on the book/course — school algebra's `y = mx + b`, or ML notation like $\hat{y} = wx + b$ or $\hat{y} = \beta_0 + \beta_1 x$. **They're all the exact same thing.** This chapter sticks to `m` (slope) and `b` (intercept) throughout, since that's the version that actually made sense on first read — just know that if you see $w$ (weight) instead of `m`, or $\beta_0/\beta_1$ instead of `b`/`m`, it's the identical formula wearing different notation.

## 0. Where this sits in the big picture

Recap from Chapter 1: Linear Regression is your first real algorithm, and it's a perfect one to start with because it's:
- **Supervised** (Chapter 1, Section 1) — trained on labeled data (features + a known numeric answer).
- **Regression** (Chapter 1, Section 1) — predicts a **number**, not a category.
- **Parametric** (Chapter 1, Section 5) — learns a small, *fixed* set of numbers (parameters — here, just `m` and `b`) from the data, then throws the raw training data away and just uses those numbers to predict.

Everything below builds toward one concrete goal: turning "distill a pile of houses down to 2 numbers" (the parametric idea from Chapter 1) into an actual, precise, step-by-step method.

---

## 1. The problem, in plain terms

You have some houses. You know their square footage and the price they sold for. You want a model that, given a **new** house's square footage, predicts its price.

Training data:

| Square Ft (x) | Price in $1000s (y) |
|---|---|
| 500 | 150 |
| 1000 | 200 |
| 1500 | 250 |
| 2000 | 300 |
| 2500 | 350 |

Just eyeballing it: every time square footage goes up by 500, price goes up by 50. That's a **linear relationship** — a straight line.

```
 price ($1000s)
   │                                    •
   │                              •
   │                        •
   │                  •
   │            •
   │      •
   └─────────────────────────────────────  sqft
```

Linear Regression's job: **find the one straight line that best fits through the data**, so that for any new square footage — even one you've never seen — you can read off a predicted price from the line.

---

## 2. The formula

You know this from school as `y = mx + b`. That's it — that's the whole formula, and it never changes shape (more on that in Section 6).

| Symbol | Meaning |
|---|---|
| `x` | The **input/feature** — square footage |
| `y` | The **output/target** — what we're predicting, price |
| `m` | The **slope** — how much price increases per 1-unit increase in square footage |
| `b` | The **intercept** — the "starting price" when square footage is 0 (a math anchor point — often not literally meaningful in real life, since a 0 sqft house doesn't exist, but the formula needs it to place the line correctly) |

### Solving it for our clean data

If you solve for the line that fits the table in Section 1 exactly, you get:

$$
y = 0.1x + 100
$$

**Check it:** at `x = 1000` → `y = 0.1(1000) + 100 = 100 + 100 = 200` ✓ — matches the table exactly.

**Reading this in plain English:** for every 1 sq ft increase, price goes up by $0.1k ($100), plus a base of $100k. This is the "distilled formula" idea from Chapter 1's Parametric section, made completely literal — the entire model is just these 2 numbers, `m = 0.1` and `b = 100`. Whether you trained on 5 houses or 5,000, once you have `m` and `b`, you can throw away the original data — the formula alone predicts any new house's price.

---

## 3. Real data is messy — this is where "learning" actually comes in

The table above was suspiciously perfect (every point lined up exactly). Real data never does. Say your actual data looks like this instead:

| Sq Ft (x) | Actual Price (y) |
|---|---|
| 500 | 160 |
| 1000 | 190 |
| 1500 | 260 |
| 2000 | 290 |
| 2500 | 355 |

Now **no single line passes through every point exactly.** So the question changes from "solve for the line" to: **what's the *best* line — the one closest to all the points overall?**

### Why can't I just pick any 2 points and compute rise/run?

Good instinct to test, and it reveals exactly why "learning" is needed. Try it with 3 different pairs from the messy table above:

- $(500,160)$ and $(1000,190)$: $m = \dfrac{190-160}{1000-500} = \dfrac{30}{500} = 0.06$
- $(1500,260)$ and $(2500,355)$: $m = \dfrac{355-260}{2500-1500} = \dfrac{95}{1000} = 0.095$
- $(500,160)$ and $(2500,355)$: $m = \dfrac{355-160}{2500-500} = \dfrac{195}{2000} = 0.0975$

**Three different pairs → three different slopes (0.06, 0.095, 0.0975).** None of these is "correct" — each is just the local slope between two specific points, ignoring every other point in the dataset. On a perfectly straight line, any 2 points give you the same right answer. The moment data gets messy, that shortcut breaks — you need a method that looks at **every point at once** and finds the slope that's the best overall compromise. That's exactly what Sections 4–5 below build.

---

## 4. Step 1 — Define "how wrong am I": the Cost Function

For any candidate line (any guess at `m` and `b`), and any one house, compare the prediction to reality:

$$
\text{error for one point} = \hat{y} - y
$$

($\hat{y}$, "y-hat," just means "the value **this line predicts**" — as opposed to `y`, the actual real value. You'll see this hat symbol constantly in ML: no hat = real/actual, hat = predicted.)

**Why not just add up all the errors across every house?** Because positive and negative errors cancel out. A line that's way too high on some houses and way too low on others could sum to ~0 and falsely look "perfect." (Same problem, same fix, as variance in `statistics-self-learning` — squaring removes the sign.)

So instead, **square each error, then average across all houses** — this is the **Mean Squared Error (MSE)**, the standard cost function for Linear Regression:

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2
$$

Reading it piece by piece: $\hat{y}_i - y_i$ is one house's error, $(\dots)^2$ squares it (removes sign, punishes big misses harder), $\sum$ adds that up across all $n$ houses, and $\frac{1}{n}$ averages it. **Bigger MSE = worse line. The entire goal of training is to find the `m` and `b` that make MSE as small as possible.**

### Worked example — comparing two candidate lines by hand

Using 4 houses (sqft in hundreds, price in ₹ lakhs, just for clean small numbers):

| sqft (x) | actual price (y) |
|---|---|
| 10 | 65 |
| 15 | 88 |
| 20 | 105 |
| 25 | 130 |

**Candidate Line A:** `y = 4x + 20`

| x | y (actual) | $\hat{y}$ (predicted) | error | squared error |
|---|---|---|---|---|
| 10 | 65 | 20+40=60 | -5 | 25 |
| 15 | 88 | 20+60=80 | -8 | 64 |
| 20 | 105 | 20+80=100 | -5 | 25 |
| 25 | 130 | 20+100=120 | -10 | 100 |

MSE for Line A = (25+64+25+100)/4 = 214/4 = **53.5**

**Candidate Line B:** `y = 4.5x + 15`

| x | y (actual) | $\hat{y}$ (predicted) | error | squared error |
|---|---|---|---|---|
| 10 | 65 | 15+45=60 | -5 | 25 |
| 15 | 88 | 15+67.5=82.5 | -5.5 | 30.25 |
| 20 | 105 | 15+90=105 | 0 | 0 |
| 25 | 130 | 15+112.5=127.5 | -2.5 | 6.25 |

MSE for Line B = (25+30.25+0+6.25)/4 = 61.5/4 = **15.375**

**Line B has a much lower MSE (15.375 vs 53.5) → Line B fits this data better.** This is exactly how "better" gets defined numerically instead of just eyeballed — and it's exactly what training searches over, at scale, across every possible `m`/`b`, not just 2 hand-picked guesses.

---

## 5. Step 2 — Actually finding the best `m` and `b`

There are two standard methods for finding the `m`/`b` that minimizes MSE. Both are worth knowing, because different tools/algorithms use one or the other.

### Method A — Ordinary Least Squares (OLS): solve it directly

For plain Linear Regression, there's actually a direct formula that calculates the exact best `m` and `b` in one shot — no guessing, no trial and error. This is called **Ordinary Least Squares (OLS)** ("least squares" = literally "smallest sum of squared errors" — the name describes exactly what it finds).

Applying OLS to the messy 5-house data from Section 3:

$$
m = 0.098, \quad b = 104
$$

$$
y = 0.098x + 104
$$

**Notice:** `m = 0.098` sits almost exactly between the 3 two-point slopes we computed earlier (0.06, 0.095, 0.0975) — closest to the ones computed from points that were *far apart*. That's not a coincidence: points far apart approximate the overall trend better than points close together, but OLS makes this mathematically rigorous by using **every point**, not a guess about which two to trust.

**Check the fit:** MSE with `m=0, b=0` (a flat, useless line) is **67,885**. MSE with the OLS-solved `m=0.098, b=104` drops to just **82**. That collapse from 67,885 → 82 is the entire payoff of solving for the right `m`/`b` instead of guessing.

### Method B — Gradient Descent: find it by iterating

OLS works cleanly for plain Linear Regression, but many other ML algorithms (neural networks, logistic regression, etc.) don't have a one-shot formula like this — they need an iterative search method instead. Linear Regression is the perfect place to learn that method, since you can compare it against OLS's known-correct answer.

**The idea:** imagine standing on a hill (the MSE cost function, plotted against every possible `m`/`b` combination) blindfolded, and you want to walk downhill to the lowest point (smallest MSE). You take small steps in the downhill direction, repeatedly, until you can't go any lower.

$$
m = m - \alpha \cdot \frac{\partial \text{MSE}}{\partial m}, \qquad b = b - \alpha \cdot \frac{\partial \text{MSE}}{\partial b}
$$

- The $\frac{\partial \text{MSE}}{\partial m}$ term (the "gradient") tells you which direction increases MSE — so you step in the *opposite* direction to decrease it.
- $\alpha$ (**alpha**, the **learning rate**) controls how big each step is. Too big → you overshoot and bounce around without settling. Too small → it takes forever to reach the bottom.
- You repeat this update, thousands of times, each round nudging `m` and `b` a little closer to the values that minimize MSE.

**Worked example — one real step, by hand**, starting from `m=0, b=0` on the messy 5-house data (using a small learning rate $\alpha = 0.0000001$ — the actual value doesn't matter here, just seeing the mechanism work):

1. Start: `m=0, b=0` → MSE = 67,885 (a flat, useless line — predicts price=0 for every house)
2. Compute the gradient at this point: it tells us MSE decreases if we increase `m` (steeply) and increase `b` (slightly)
3. Take one step: `m` moves from `0` → `0.0851`, `b` moves from `0` → `0.00005`
4. Re-check MSE with the new `m, b`: **67,885 → 15,380** — already much better, after just **one** step

Keep repeating this update over and over, and `m`/`b` keep creeping toward OLS's exact answer (`m=0.098, b=104`) — that's "training" a model, literally: repeated small corrections until the error stops shrinking meaningfully.

| | OLS | Gradient Descent |
|---|---|---|
| How it finds `m`, `b` | Solves directly with one formula | Iteratively nudges `m`, `b` closer, step by step |
| Speed | Instant (one calculation) | Takes many iterations |
| Works when a direct formula doesn't exist? | No — only works for problems with a solvable direct formula | Yes — this is *why* gradient descent matters: most other ML algorithms rely on it |
| Answer | Exact best `m`, `b` | Approaches the same best `m`, `b`, gets arbitrarily close with enough steps |

---

## 6. Using the trained model

Once training (either method) has produced the final `m` and `b`, prediction is trivial — just plug in a new `x`:

Say training settled on `y = 0.114x + 106`. A new house comes in: 1800 sq ft, price unknown.

$$
y = 0.114(1800) + 106 = 205.2 + 106 = 311.2
$$

**Predicted price: $311,200.** That's the entire point of training — find the `m` and `b` that fit the historical data well, then reuse that same formula on new, unseen houses, instantly.

---

## 7. Does the formula ever change? (Multiple features)

Real problems rarely have just one input feature. House price probably depends on square footage **and** bedrooms **and** distance to city center, all at once. The formula extends — but its *shape* stays exactly the same idea:

$$
y = m_1x_1 + m_2x_2 + m_3x_3 + \dots + b
$$

Each feature ($x_1$ = sqft, $x_2$ = bedrooms, $x_3$ = distance, ...) gets its **own slope** ($m_1, m_2, m_3, ...$), and there's still just one `b`. This is called **Multiple Linear Regression** (one feature = **Simple Linear Regression**, Sections 1–6 above).

**Why separate slopes per feature matters:** $m_1$ (sqft's slope) tells you sqft's effect on price *in isolation*, holding bedrooms and distance constant. This disentangles overlapping effects — e.g. bigger houses often also have more bedrooms, and Multiple Regression can tell you sqft's effect *specifically*, separate from bedroom count's effect, rather than muddling the two together.

### The big-picture insight: the formula structure never changes — only the numbers do

Whether you're predicting house price from sqft, salary from years of experience, exam scores from hours studied, or crop yield from rainfall + temperature + soil quality — it's **always** `y = mx + b` (or its multi-feature version). What changes per problem:
- **How many `x`'s (features)** you have.
- **The actual learned values** of each `m` and `b` — different, because they're learned from different data.

The *method* to find those numbers is also always the same two tools: **MSE** as the cost function, and **OLS or Gradient Descent** to minimize it. You're not inventing new math per problem — you're feeding the same machinery different data, and it hands back different `m`'s and `b`.

### The boundary: this only works if the relationship really is linear

Linear Regression assumes the true relationship between `x` and `y` is roughly a straight line. That assumption holds great for data like the house-price example. But if the real relationship is curved — e.g. house price vs. sqft showing diminishing returns on very large houses, or salary vs. age rising then plateauing — forcing a straight line onto curved data gives you a systematically bad model, no matter how perfectly you solve for `m` and `b`. The formula stays the same; it's just the wrong tool for that data.

That's the boundary where you'd move to **Polynomial Regression** (bends the line into a curve, similar underlying math) or a genuinely different kind of model (decision trees, neural networks) — covered in later chapters.

---

## 8. Key assumptions Linear Regression makes

Because Linear Regression is **parametric** (Chapter 1, Section 5), it assumes a shape upfront — literally, a straight line. That assumption only holds up if the real data roughly satisfies these conditions. Violating them means your `m`/`b` (and predictions from them) become unreliable:

| Assumption | Plain-English meaning | Why it matters |
|---|---|---|
| **Linearity** | The real relationship is actually (roughly) a straight line | If the true pattern is curved, a straight line systematically misses — no amount of "better fitting" fixes a wrong shape (Section 7's boundary) |
| **Independence of errors** | One house's error shouldn't predict another's | Common violation: time-series data where today's error correlates with yesterday's |
| **Homoscedasticity** (constant error spread) | The "typical size of a miss" should be similar across the whole range of x, not way bigger for large x than small x | If errors fan out (bigger for bigger houses), the model is more trustworthy for some x-ranges than others |
| **Normality of errors** | The errors, plotted as a distribution, should roughly look like a Normal (bell curve) distribution | Ties directly to `statistics-self-learning`'s Normal Distribution section — several of Linear Regression's statistical guarantees rely on this |
| **No severe multicollinearity** (multiple regression only) | Input features shouldn't be near-duplicates of each other (e.g. `length_cm` and `length_inches` as separate features) | If two features carry almost the same info, the model can't cleanly tell which deserves credit, and individual slopes become unstable/hard to trust |

**How to check these in practice:** mostly done by plotting the errors (error vs. predicted value, and a histogram of errors) *after* fitting — a "diagnose after the fact" step, same spirit as checking skewness/kurtosis *after* computing them in `statistics-self-learning`.

---

## 9. Evaluating fit quality in human terms

MSE (Section 4) is great for *comparing* candidate lines during training, but it's an awkward number to *report* to a human — its units are squared (e.g. "lakhs²" means nothing intuitively), and its scale depends on how many data points you have. Two more readable metrics fix this (full computational detail in a later evaluation-metrics chapter, but the concepts belong here since they follow directly from this chapter's math):

- **RMSE (Root Mean Squared Error):** $\sqrt{\text{MSE}}$ — undoes the squaring, back to the original unit. Same "undo the squaring" logic as Standard Deviation in `statistics-self-learning`. Reads as: "on average, predictions are off by about $X."
- **R² (R-squared):** a 0-to-1 score answering *"what fraction of the variation in price does this line actually explain?"* R² = 1 → the line perfectly explains all the variation. R² = 0 → the line explains nothing (you'd have done just as well always predicting the average price, ignoring sqft entirely).

---

## The full mental model, tied together

1. **Assume a shape** — a straight line, `y = mx + b` (or the multi-feature version) — the parametric assumption from Chapter 1.
2. **Define "wrong"** — for a candidate line, compute each house's error ($\hat{y} - y$), square it, average across all houses → **MSE**.
3. **Find the best line** — solve directly with **OLS**, or search iteratively with **Gradient Descent** — either way, the goal is the `m`/`b` that minimizes MSE.
4. **Remember the formula never changes shape** — only the number of features and the learned `m`/`b` values change per problem; the method (MSE + OLS/Gradient Descent) is always the same machinery.
5. **Sanity-check the assumptions** — linearity, independent/constant-spread/normal errors, no multicollinearity.
6. **Report fit quality in human terms** — RMSE (typical miss, in real units) and R² (how much of the pattern the line explains).

Once this is solid, the next step is turning this into actual code — fitting a real Linear Regression model on real data using a library, and watching these exact concepts (errors, MSE, OLS, gradient descent, R²) show up as real numbers you compute, not hand-worked examples.
