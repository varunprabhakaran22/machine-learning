# Chapter 3 — Logistic Regression

*(Theory + math only in this chapter, same as Chapter 2 — code comes later as a separate follow-up step.)*

**Notation note:** same convention as Chapter 2 — plain `m`/`b`/`x`/`y` throughout, not $\beta_0/\beta_1$ or $\hat{y}$. If you see the beta/hat version elsewhere, it's the identical idea, different labels.

## 0. Where this sits in the big picture

Recap from Chapter 1: Logistic Regression is —
- **Supervised** (Chapter 1, Section 1) — trained on labeled data (features + a known answer).
- **Classification** (Chapter 1, Section 1) — predicts a **category**, not a number. (Specifically: this chapter covers **binary** classification — exactly 2 categories, e.g. yes/no, pass/fail, spam/not-spam.)
- **Parametric** (Chapter 1, Section 5) — same as Linear Regression, it learns a small, fixed set of numbers (`m`'s and `b`) from the data and then discards the raw training data.

**The name is a trap, so let's kill the confusion immediately:** despite the name "Logistic **Regression**," this is a **classification** algorithm, not a regression algorithm. The name is historical — it's called that because it's built directly on top of Linear Regression's machinery (you'll see exactly how in Section 2), but its actual job is predicting a category (yes/no), not a number.

---

## 1. The problem: why can't we just reuse Linear Regression?

### The real-world story

Say you're an exam instructor with historical data: hours studied → did the student pass?

| Hours Studied (x) | Passed? (y) |
|---|---|
| 1 | No (0) |
| 2 | No (0) |
| 3 | No (0) |
| 4 | No (0) |
| 5 | Yes (1) |
| 6 | Yes (1) |
| 7 | Yes (1) |
| 8 | Yes (1) |

Notice `y` here is only ever `0` or `1` — never `2.5` or `117`. This is a **categorical target** (Chapter 1's "category → classification" rule), even though it's *written* as numbers `0`/`1` for convenience — `0` and `1` are just labels standing in for "No" and "Yes," not actual quantities you'd average.

### Try fitting a straight line to this anyway — here's why it breaks

If you plot this and run ordinary Linear Regression (`y = mx + b`, Chapter 2's method) straight through it, you get a line that technically "fits," but produces nonsense:

```
 y (pass=1/fail=0)
  1 │                              •  •  •  •
    │                        _,--''
    │                  _,--''
    │            _,--''
  0 │•  •  •  •''
    └──────────────────────────────────────── x (hours studied)
       1  2  3  4  5  6  7  8  9  10 11 12
```

Two concrete problems:
1. **Nonsense outputs.** A straight line keeps going forever in both directions. Plug in `x = 12` (a student who studied a lot) and the line might predict `y = 1.4` — "140% probability of passing"? Plug in `x = -2` and it might predict `y = -0.3` — a negative probability. Neither number means anything; probability must stay between 0 and 1, and a straight line has no way to enforce that.
2. **Wrong shape for this kind of data.** Even ignoring the out-of-range problem, a straight line assumes *constant* change (`m` is the same everywhere). But real pass/fail behavior isn't like that — near the "boundary" (around 4-5 hours here), a small change in study hours flips the outcome dramatically. Far from the boundary (1 hour or 8 hours), more studying barely changes the *certainty* of the outcome (already very unlikely to pass / already very likely to pass). That's not a straight-line pattern — it's an **S-shaped** one.

**This is exactly the problem Logistic Regression solves:** keep Linear Regression's core idea (a weighted combination of inputs), but squash the output through a function that (a) always stays between 0 and 1, and (b) naturally produces that S-shape.

---

## 2. The Sigmoid function — squashing a line into a probability

### Step 1 — Still start with the familiar linear formula

Logistic Regression **still computes** the exact same linear combination as Chapter 2:

$$
z = mx + b
$$

(Or for multiple features: $z = m_1x_1 + m_2x_2 + \dots + b$.) Nothing new yet — `z` is just the straight-line formula from Chapter 2, renamed `z` instead of `y`, because it's *not* the final prediction anymore — it's an intermediate value, about to get squashed.

### Step 2 — Squash `z` through the Sigmoid function

$$
p = \frac{1}{1 + e^{-z}}
$$

This is the **Sigmoid function** (also called the **logistic function** — this is where the algorithm's name comes from). `e` is Euler's number (≈2.71828, a fixed constant, same `e` from compound interest/exponential growth — not a variable).

**What Sigmoid actually does, in plain terms:** it takes *any* number `z` (from $-\infty$ to $+\infty$) and squashes it into a number strictly between 0 and 1 — perfect for representing a **probability**.

| `z` (the linear part) | `p = sigmoid(z)` | Reading it |
|---|---|---|
| -6 | 0.0025 | Very negative `z` → probability near 0 |
| -2 | 0.1192 | Fairly negative → low probability |
| -1 | 0.2689 | Somewhat negative → below 50% |
| **0** | **0.5000** | **Exactly 0 → exactly 50%, the tipping point** |
| 1 | 0.7311 | Somewhat positive → above 50% |
| 2 | 0.8808 | Fairly positive → high probability |
| 6 | 0.9975 | Very positive `z` → probability near 1 |

Notice the shape this creates: it's flat near 0 and flat near 1 (extreme `z` values barely change `p` further), but steep in the middle around `z=0` (small changes in `z` swing `p` a lot). **That's exactly the S-shape Section 1 said we needed** — extreme values are unphased, but the tipping-point region is sensitive.

```
 p
 1 │                     ______________
   │                 __--
   │              _-'
   │            /
   │          /
 0.5│ ─ ─ ─ ─●─ ─ ─ ─ ─ ─ ─ ─        ← z = 0 always gives p = 0.5
   │       /
   │    _-'
   │ __-
 0 │____
   └───────────────────────────────  z
     -6  -4  -2   0   2   4   6
```

### Step 3 — Turn the probability into an actual Yes/No prediction

Sigmoid gives you a probability (e.g. `p = 0.83`), not a category yet. The final step is a simple threshold, almost always **0.5**:

$$
\text{prediction} = \begin{cases} 1 \ (\text{"Yes"}) & \text{if } p \geq 0.5 \\ 0 \ (\text{"No"}) & \text{if } p < 0.5 \end{cases}
$$

### Full worked example — the exam data, end to end

Say training produces `m = 0.8, b = -4.0` (Section 4 covers how these actually get found). Let's run every student through the full 3-step pipeline (`z = mx+b` → `p = sigmoid(z)` → threshold at 0.5):

| Hours (x) | `z = 0.8x - 4.0` | `p = sigmoid(z)` | Predicted | Actual |
|---|---|---|---|---|
| 1 | -3.20 | 0.0392 | No (0) | No (0) ✓ |
| 2 | -2.40 | 0.0832 | No (0) | No (0) ✓ |
| 3 | -1.60 | 0.1680 | No (0) | No (0) ✓ |
| 4 | -0.80 | 0.3100 | No (0) | No (0) ✓ |
| 5 | 0.00 | 0.5000 | **Yes (1)** | Yes (1) ✓ |
| 6 | 0.80 | 0.6900 | Yes (1) | Yes (1) ✓ |
| 7 | 1.60 | 0.8320 | Yes (1) | Yes (1) ✓ |
| 8 | 2.40 | 0.9168 | Yes (1) | Yes (1) ✓ |

Every prediction matches. Notice `x=5` sits exactly at `z=0`, `p=0.5` — right on the threshold — this specific line's **decision boundary** is exactly at 5 hours studied. That's a real, meaningful number: *"this model believes 5 hours of studying is the tipping point between likely failing and likely passing."*

---

## 3. Defining "wrong" for Logistic Regression: why not reuse MSE?

Chapter 2 used Mean Squared Error (MSE) to score how bad a candidate line was. You might expect the same trick here — but MSE turns out to be a poor fit for Logistic Regression, for a subtle but important reason.

### The problem with MSE here

MSE punishes errors in *equal, smooth* proportion to how far off you are. That's fine for predicting continuous numbers (Chapter 2's house prices). But for probabilities, **some wrong answers deserve to be punished far more harshly than others**:

- If the true answer is "Yes" (`y=1`) and the model says `p = 0.51` (barely over 50%, a weak/unsure "yes") — mildly wrong, should get a small penalty.
- If the true answer is "Yes" (`y=1`) and the model says `p = 0.01` (the model is **extremely confident** it's a "No") — this is a *disastrously* wrong, overconfident mistake, and should get punished far more severely than the mild case above.

MSE doesn't capture that difference well — it treats "off by a bit" and "confidently, catastrophically wrong" too similarly, because it just squares the numeric gap. Logistic Regression needs a cost function that specifically **punishes confident wrongness much harder than timid wrongness**.

### The fix: Log Loss (Binary Cross-Entropy)

$$
\text{Log Loss} = -\big[\, y \cdot \log(p) \;+\; (1-y)\cdot \log(1-p) \,\big]
$$

This looks dense, but it's actually just an if/else in disguise — because `y` is always exactly `0` or `1`, one of the two terms always vanishes:

- **If the true answer is `y = 1`:** the formula collapses to just $-\log(p)$ — "how confident was the model in the *correct* (Yes) answer? Punish low confidence in the right answer."
- **If the true answer is `y = 0`:** the formula collapses to $-\log(1-p)$ — "how confident was the model that it's *not* Yes? Punish low confidence in the right answer, mirrored."

### Why the log specifically? Plotting it out

$-\log(p)$, as `p` (confidence in the correct answer) ranges from near-0 to 1:

| `p` (confidence in correct answer) | $-\log(p)$ (the loss) |
|---|---|
| 0.99 (very confident & correct) | **0.010** — tiny penalty |
| 0.50 (a coin flip, unsure) | **0.693** — moderate penalty |
| 0.01 (very confident & *wrong*) | **4.605** — huge penalty |

**This is the entire point of using `log`:** as `p` (confidence in the right answer) gets closer to 0, the penalty doesn't just grow — it **explodes** toward infinity. A model that's *confidently wrong* gets punished dramatically harder than one that's just mildly off. This is exactly the property Section 3's "MSE problem" was missing.

### Worked example — Log Loss on the full exam dataset

Using the same `m=0.8, b=-4.0` model from Section 2:

| x | y (actual) | p (predicted) | Log Loss for this point |
|---|---|---|---|
| 1 | 0 | 0.0392 | 0.040 |
| 2 | 0 | 0.0832 | 0.087 |
| 3 | 0 | 0.1680 | 0.184 |
| 4 | 0 | 0.3100 | 0.371 |
| 5 | 1 | 0.5000 | 0.693 |
| 6 | 1 | 0.6900 | 0.371 |
| 7 | 1 | 0.8320 | 0.184 |
| 8 | 1 | 0.9168 | 0.087 |

**Average Log Loss = 0.252.** Notice the pattern: points where the model was confident *and* correct (x=1, x=8) get tiny losses (0.040, 0.087). The point sitting exactly on the 50/50 boundary (x=5) gets the highest loss (0.693) — because a coin-flip prediction, even when it happens to round to the right side of the threshold, isn't something the model should be rewarded for being "confident" about. **Average Log Loss across every training point is Logistic Regression's cost function** — training means finding the `m`/`b` that makes this number as small as possible, exactly the same "minimize the cost function" idea as Chapter 2's MSE.

---

## 4. Finding the best `m` and `b`

### No direct formula this time — Gradient Descent is required

Recall Chapter 2, Section 5 introduced **two** methods: Ordinary Least Squares (OLS, a direct one-shot formula) and Gradient Descent (iterative). **Logistic Regression only has the iterative option** — because of the `log()` in the cost function, there's no clean direct formula like OLS to jump straight to the answer. This is exactly the case Chapter 2 flagged when it said *"most other ML algorithms rely on [gradient descent]"* — Logistic Regression is the first one you're meeting.

### The mechanism is identical to Chapter 2 — just a different cost function underneath

$$
m = m - \alpha \cdot \frac{\partial \, \text{LogLoss}}{\partial m}, \qquad b = b - \alpha \cdot \frac{\partial \, \text{LogLoss}}{\partial b}
$$

Same idea as Chapter 2, Section 5: start with a guess for `m`/`b`, compute the gradient (which direction makes Log Loss worse), step in the *opposite* direction by a small amount controlled by the learning rate $\alpha$, and repeat thousands of times until Log Loss stops meaningfully shrinking. The mechanics don't change between Linear and Logistic Regression — only *what's being minimized* changes (MSE → Log Loss), because the shape of "wrong" is different for a probability than for a raw number (Section 3).

---

## 5. From probability to multiple categories (a preview)

Everything above was **binary** classification — exactly 2 categories. Real problems sometimes have 3+ categories (e.g. "cat / dog / bird," not just "cat / not-cat"). That extension is called **Multinomial Logistic Regression** (sometimes "Softmax Regression") — same core Sigmoid-and-threshold idea generalized to multiple categories at once, each getting its own probability, with all probabilities summing to 1. Worth knowing the name exists; full treatment belongs in a later chapter once binary classification (this chapter) is solid.

---

## 6. Key assumptions Logistic Regression makes

Same parametric spirit as Chapter 2 (Section 8 there) — Logistic Regression assumes a shape upfront (specifically, that `z = mx+b`, the *pre-sigmoid* linear combination, is a good fit — not that the final S-curve itself is a straight line, that's the whole point of the Sigmoid).

| Assumption | Plain-English meaning | Why it matters |
|---|---|---|
| **Linearity of the log-odds** | The *log-odds* (a transformed version of `z`, see Section 7) should be a linear function of the inputs — not that `y` vs `x` is linear (it's explicitly not, that's why we needed Sigmoid) | If the true relationship between inputs and log-odds is curved, Logistic Regression will systematically misfit, same spirit as Chapter 2's linearity assumption |
| **Independence of observations** | One data point's outcome shouldn't influence another's | Same idea as Chapter 2 — common violation in time-series or repeated-measurement data |
| **Little/no multicollinearity** (multiple features) | Input features shouldn't be near-duplicates of each other | Identical reasoning to Chapter 2, Section 8 — redundant features make individual `m`'s unstable to interpret |
| **Large enough sample size** | Logistic Regression's probability estimates get unreliable with very few data points, especially if one category is rare | Unlike Linear Regression, there's no simple sample-size rule of thumb here — but as a practical habit, be skeptical of a model trained on very few examples of one class (e.g. 950 "no fraud" vs only 5 "fraud" examples) |

---

## 7. A useful side-concept: Odds and Log-Odds (why the assumption says "log-odds")

You'll see the term **log-odds** (or "logit") come up around Logistic Regression, including in Section 6's assumption above — worth defining once, briefly:

- **Odds** = $\dfrac{p}{1-p}$ — "how many times more likely is Yes than No?" (e.g. `p=0.8` → odds = `0.8/0.2 = 4`, i.e. "4-to-1 in favor of Yes.")
- **Log-odds** = $\log\left(\dfrac{p}{1-p}\right)$ — turns out, algebraically, this is **exactly equal to `z` (`mx+b`)** — the plain linear formula from Section 2, Step 1, before Sigmoid was ever applied.

This is genuinely just a side-note for recognizing the term when you see it elsewhere (e.g. "Logistic Regression models the log-odds as a linear function of the inputs" is a common textbook phrasing for exactly what Section 2 walked through step by step) — not something you need to compute by hand routinely.

---

## The full mental model, tied together

1. **Start the same as Linear Regression** — compute `z = mx + b` (or the multi-feature version), the exact same weighted-sum idea as Chapter 2.
2. **Squash `z` into a probability** — `p = sigmoid(z) = 1/(1+e^{-z})`, always between 0 and 1, naturally S-shaped.
3. **Threshold to get a category** — `p ≥ 0.5` → predict "Yes" (1), else "No" (0).
4. **Define "wrong" with Log Loss, not MSE** — because confidently-wrong predictions need to be punished far harder than timid ones; the `log()` makes that penalty explode as confidence-in-the-wrong-answer grows.
5. **Find the best `m`/`b` with Gradient Descent** — no direct formula (unlike OLS in Chapter 2) exists here, because of the `log()` in the cost function.
6. **Sanity-check the assumptions** — linearity of the *log-odds* (not of `y` vs `x` directly), independence, no multicollinearity, reasonable sample size per category.

Once this is solid, the next step is the same as Chapter 2 — turning this into actual code, fitting a real Logistic Regression model on real data, and watching Sigmoid, Log Loss, and Gradient Descent show up as real numbers you compute.
