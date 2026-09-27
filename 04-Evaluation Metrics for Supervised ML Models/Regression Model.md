# Regression Model Evaluation Metrics

*(MAE, MSE, RMSE, R² Score, Adjusted R² Score — all explained using the same trained model from `02-Linear Regression/scripting.ipynb`: Linear Regression on `assets/Cars.csv`, predicting `MPG`. Every number below was actually computed with a script and verified by hand, not estimated.)*

This is the **regression-side counterpart** to `Classification Model.md` in this same folder. Recall the clarification from earlier: "Classification Model" and "Regression Model" aren't new algorithms — they're the two umbrella categories from Chapter 1, Section 1. Logistic Regression (Ch3) sits under Classification Model; Linear Regression (Ch2) sits under Regression Model. This file covers the metrics used to evaluate *any* regression algorithm, not just Linear Regression specifically.

## 0. Why do regression metrics look completely different from classification metrics?

Recall `Classification Model.md`, Section 1 — classification metrics are all built from a Confusion Matrix, because the model's output is a **category** (right or wrong, a discrete match/mismatch). Regression is fundamentally different: the model outputs a **number** (e.g. predicted MPG = 29.7), and "wrong" isn't binary — a prediction of 29.7 vs. an actual of 30.0 is only *slightly* wrong, while a prediction of 15.0 vs. an actual of 30.0 is *very* wrong. There's no TP/FP/TN/FN here — there's only **how far off** each prediction was, in a continuous sense. Every metric below is really just a different way of summarizing "how far off, on average, were the predictions" across all the test rows.

**The one thing to build on:** every metric below starts from the same building block — the **residual** (recall `02-Linear Regression/theory.md` Section 4): `residual = actual - predicted`, one number per test row. MAE, MSE, RMSE, and R² are all just different arithmetic done on this same list of residuals.

---

## 1. MAE — Mean Absolute Error

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

**Plain English:** take every residual (actual − predicted), make it positive (absolute value — ignore whether the model overshot or undershot), and average them all.

**Why take the absolute value?** Recall Chapter 2's theory.md — residuals are positive when the model under-predicted and negative when it over-predicted. If you just averaged the raw residuals, positives and negatives would cancel out, hiding real errors (the exact same cancellation problem MSE's squaring trick solves — see Section 2 below). Taking the absolute value fixes the cancellation without squaring.

**Our number:** **MAE = 4.11** — on average, the model's MPG prediction is off by about 4.11 MPG, in either direction, with every mile-per-gallon of error weighted *equally* regardless of size.

**When to prefer MAE:** when you want an error number that's easy to explain in plain terms ("on average, off by X units") and when you don't want large outlier errors to dominate the score more than they "deserve" — MAE treats a 10-unit miss as exactly 10x worse than a 1-unit miss, nothing more.

---

## 2. MSE — Mean Squared Error

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**Plain English:** take every residual, **square** it (instead of taking the absolute value), and average them all. This is the exact cost function from `02-Linear Regression/theory.md` Section 4 — Ordinary Least Squares literally means finding the `m`/`b` that minimizes this number.

**Why square instead of absolute value?** Squaring does two things absolute value doesn't: (1) it's mathematically smoother, which is what allows Gradient Descent (Chapter 3) and OLS's direct formula (Chapter 2) to work efficiently; (2) it **punishes large errors disproportionately harder** than small ones. A 10-unit miss becomes 100 (squared), while a 1-unit miss stays 1 — so under MSE, one huge miss can dominate the whole score far more than ten small misses combined.

**Our number:** **MSE = 31.96** — but notice the unit problem immediately: this is in "MPG²," a unit that means nothing intuitively. That's precisely why MSE is rarely *reported* directly — it's mainly the thing the model minimizes internally during training, not a human-facing summary number.

---

## 3. RMSE — Root Mean Squared Error

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

**Plain English:** just the square root of MSE — undoing the squaring, so the number lands back in real, interpretable units (MPG, not MPG²). Same "undo the squaring" logic as Standard Deviation in `statistics-self-learning`.

**Our number:** **RMSE = 5.65** — on average (in the squared-then-rooted sense), predictions are off by about 5.65 MPG.

**RMSE vs. MAE — why are they different numbers (5.65 vs. 4.11) for the same model?** Because RMSE's squaring step still leaves outlier errors weighted more heavily even after the square root is taken — RMSE will always be **≥ MAE** for the same data, and the *gap* between them tells you something useful: a big gap means a few large errors are dragging RMSE up; a small gap means the errors are fairly uniform in size. Here, RMSE (5.65) is noticeably above MAE (4.11), hinting the model has at least a few larger misses pulling the RMSE up.

**When to prefer RMSE over MAE:** when large errors are *genuinely* more costly in the real business context (not just mathematically) — e.g. if being off by 20 MPG is disproportionately worse than being off by 2 MPG ten times over, RMSE's extra sensitivity to big misses is a feature, not a bug.

---

## 4. R² Score (Coefficient of Determination) — "how much of the variation did the model actually explain?"

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$

where:
- $SS_{res} = \sum (y_i - \hat{y}_i)^2$ — the **Sum of Squared Residuals** — the model's actual total squared error (this is just $n \times \text{MSE}$, unaveraged)
- $SS_{tot} = \sum (y_i - \bar{y})^2$ — the **Total Sum of Squares** — how much the actual values vary around their own mean, *ignoring the model entirely* (this is the error a "dumbest possible baseline" — always predicting the average — would produce)

**Plain English:** R² compares the model's error against the error of the simplest possible baseline (always guessing the average `y`, recall the baseline comparisons in both `recap.md` files). It answers: *"what fraction of the target's variation does the model explain, compared to just guessing the average every time?"*

**Our number:** **R² = 0.675** → the model explains about 67.5% of the variation in MPG across cars, using HP/VOL/SP/WT. The remaining 32.5% is variation the model's 4 features don't capture.

**Reading the scale:**

| R² value | Meaning |
|---|---|
| **1.0** | Perfect — the model's predictions exactly match every actual value ($SS_{res} = 0$) |
| **0.0** | The model is exactly as good as always predicting the average — it explains nothing beyond the baseline |
| **Our 0.675** | Meaningfully better than the baseline, genuinely useful, but far from perfect |
| **Negative** | Worse than just guessing the average — a real possibility if a model is badly overfit or broken, not just a theoretical edge case |

---

## 5. Adjusted R² Score — fixing R²'s "always add more features" loophole

### The problem R² has, that Adjusted R² fixes

Here's something genuinely counter-intuitive about plain R²: **it can never go down when you add another feature to the model — even a completely useless, random one.** This isn't a hypothetical; it's provable, and it means R² alone can quietly *reward* needlessly complicated models.

**Proof, with real numbers:** take the same Cars.csv model (R² = 0.675 with 4 real features: HP, VOL, SP, WT), and add a 5th feature that's pure random noise — numbers with *zero* real relationship to MPG:

| | 4 real features | + 1 random noise feature |
|---|---|---|
| **R²** | 0.675 | **0.677** (went UP) |
| **Adjusted R²** | 0.567 | **0.530** (went DOWN) |

**R² went up** (0.675 → 0.677) purely from adding a feature that is, by construction, meaningless — proving R² alone can be gamed simply by throwing in more columns, regardless of whether they help. **Adjusted R² correctly went down** (0.567 → 0.530), recognizing that the added complexity bought nothing real.

### The formula

$$
\text{Adjusted } R^2 = 1 - (1 - R^2)\frac{n - 1}{n - k - 1}
$$

where:
- $n$ = number of observations (rows in the test set)
- $k$ = number of features (predictors) used

**Plain English:** Adjusted R² takes plain R² and applies a **penalty that grows with the number of features** relative to the amount of data you have. Adding a feature that genuinely improves the fit by more than that penalty will still raise Adjusted R². Adding a feature that adds little-to-no real signal (like random noise) won't clear that bar, and Adjusted R² will drop — exactly what happened above.

**Our numbers:** with $n=17$, $k=4$: Adjusted R² = $1 - (1 - 0.675)\times\dfrac{16}{12} = 1 - 0.325 \times 1.333 = 1 - 0.433 = 0.567$.

### When to use which

| | R² | Adjusted R² |
|---|---|---|
| **Use when...** | Comparing models with the **same** number of features | Comparing models with a **different** number of features |
| **Risk if misused** | Can be silently inflated just by adding more (even useless) features | None — it's specifically built to resist that |

**The practical rule:** if you're deciding *whether to add a new feature* to a model, don't just check if R² went up — check if **Adjusted R² went up**. If Adjusted R² drops after adding a feature, that feature isn't earning its keep, no matter what plain R² says.

---

## The full picture, tied together

1. **MAE** — average absolute error, in real units, every error weighted equally.
2. **MSE** — average squared error; punishes large errors harder; the actual quantity OLS/Gradient Descent minimize during training; awkward squared units, rarely reported directly.
3. **RMSE** — √MSE, back in real units; always ≥ MAE; the gap between RMSE and MAE hints at how much a few large errors are dragging the score.
4. **R²** — fraction of the target's variation explained, relative to the "always guess the average" baseline; 0 to 1 (can go negative for a badly broken model).
5. **Adjusted R²** — R², penalized for feature count; the honest metric to use when comparing models with different numbers of features, since plain R² can be inflated just by adding more columns, even useless ones.

**The one habit to walk away with:** MAE and RMSE tell you the *size* of a typical error, in interpretable units — pick MAE if you want every error treated equally, RMSE if large errors should count extra. R² and Adjusted R² tell you how much of the target's variation the model actually explains, relative to doing nothing — but only trust R² when comparing models with the *same* feature count; the moment feature counts differ, switch to Adjusted R².
