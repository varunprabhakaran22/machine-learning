# Recap — How do we know if the model is okay to ship?

This is the real question behind "Model Evaluation" (Step 8 of the 10-step pipeline in `scripting.ipynb`) — and the short answer is: **it's evaluation, not p-values.**

## Why not p-values?

p-values are a **statistics** concept (hypothesis testing — recall `statistics-self-learning` Chapter 3), not something every ML algorithm produces. `sklearn`'s `LinearRegression` (what `scripting.ipynb` uses) doesn't output p-values at all — it's built purely for prediction. A stats-focused library like `statsmodels` *can* give you a p-value per coefficient (testing "is this feature's effect statistically significant, or could it be random noise?"), but that's a separate, optional add-on — not the thing that tells you whether a model is good enough to ship. Plenty of excellent, shippable models (Random Forests, Neural Networks, k-Nearest Neighbors) never produce a p-value in their entire lifecycle.

**So if not p-values, what actually decides "ship it"?** A handful of concrete checks — all of which `scripting.ipynb` already ran:

## The actual checklist

| Question | How you check it | What we found in `scripting.ipynb` |
|---|---|---|
| **Does it beat doing nothing?** | Compare against a baseline — here, always predicting the training mean MPG | Model RMSE **5.65** vs. baseline RMSE **10.16** — roughly half the error of guessing the average every time |
| **Is it right in the way that matters for this business problem?** | Pick the metric that matches the cost of being wrong (recall Chapter 1's recap) | RMSE (typical miss, in real MPG units) and R² (0.675 — how much of the variation the model explains) |
| **Does it generalize, or did it just memorize?** | Evaluate only on a held-out test set the model never trained on | Every metric was computed on `X_test`/`y_test` — data `.fit()` never saw |
| **Is any single metric hiding a problem?** | Look at more than one metric/view | RMSE, MAE, R², *and* the actual-vs-predicted scatter plot together — no single number alone |
| **Are the learned coefficients actually sane?** | Sanity-check against domain knowledge, not just accept the numbers | The multicollinearity sign-flip (`SP` and `WT` flipping sign between their solo correlation and the multivariate coefficient) — predictions were still fine, but that specific coefficient isn't safe to read as "heavier cars get better MPG" |
| **Does it meet a pre-agreed bar?** | Define "good enough" *before* modeling, not after | This is the real ship/no-ship gate — a business-specific number agreed upfront, not a universal threshold |

## The honest bottom line

There is no single universal number that means "good model" (no "R² above 0.8 = always ship it"). It's always **relative to**:
1. The baseline (how much better than guessing?),
2. The actual cost of being wrong for this specific business problem, and
3. A bar someone agreed on *before* training started.

An R²=0.5 model can be a genuine win in a noisy, hard-to-predict domain. An R²=0.9 model might be mediocre in a domain that's normally near-deterministic. "Good enough" is a business judgment applied to evaluation numbers — not a property the numbers have on their own.

---

## Side note — do you train one algorithm, or several and compare?

Both, in order:

1. **Narrow the field first**, using the problem and the data shape — target type (regression here, since MPG is a number), how linear the relationship looks, how much data you have, whether you need to *explain* the model to a stakeholder, and whether speed matters. This alone rules out most algorithms before you write any training code.
2. **Train the shortlisted candidates and compare them** on the *same* train/test split with the *same* metric — this is standard practice, not overkill.
3. **Always include the simplest candidate as a first baseline-for-the-baseline** (Chapter 1's recap, Step 6 — "start simple"). If a fancier model only wins by a hair, the extra complexity usually isn't worth it.

This comparison isn't a separate phase — it's just Model Building/Training/Evaluation (Steps 5-8) run once per shortlisted candidate, then compared side by side.
