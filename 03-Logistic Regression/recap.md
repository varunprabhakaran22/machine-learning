# Recap — How do we know if the model is okay to ship?

This is the real question behind "Model Evaluation" (Step 8 of the 10-step pipeline in `scripting.ipynb`) — and the short answer is: **it's evaluation, not p-values.**

## Why not p-values?

p-values are a **statistics** concept (hypothesis testing — recall `statistics-self-learning` Chapter 3), not something every ML algorithm produces. `sklearn`'s `LogisticRegression` (what `scripting.ipynb` uses) doesn't output p-values at all — it's built purely for prediction. A stats-focused library like `statsmodels` *can* give you a p-value per coefficient (testing "is this feature's effect statistically significant, or could it be random noise?"), but that's a separate, optional add-on — not the thing that tells you whether a model is good enough to ship. Plenty of excellent, shippable models (Decision Trees, Random Forests, Neural Networks, k-Nearest Neighbors) never produce a p-value in their entire lifecycle.

**So if not p-values, what actually decides "ship it"?** A handful of concrete checks — all of which `scripting.ipynb` already ran:

## The actual checklist

| Question | How you check it | What we found in `scripting.ipynb` |
|---|---|---|
| **Does it beat doing nothing?** | Compare against a baseline — here, always predicting the majority class | Model accuracy **67.3%** vs. baseline accuracy **52.7%** |
| **Is it right in the way that matters for this business problem?** | Pick the metric that matches the cost of being wrong (recall Chapter 1's recap) — a missed attorney case (False Negative) probably costs the insurer more than a false alarm | Precision/recall per class, not just one accuracy number |
| **Does it generalize, or did it just memorize?** | Evaluate only on a held-out test set the model never trained on | Every metric was computed on `X_test`/`y_test` — data `.fit()` never saw |
| **Is any single metric hiding a problem?** | Look at more than one metric/view | Accuracy alone doesn't show *which kind* of mistake is happening — the confusion matrix does (how many False Positives vs. False Negatives) |
| **Can the model actually separate the two classes, across all thresholds — not just at 0.5?** | ROC-AUC | **0.745** — meaningfully better than 0.5 (random guessing), not near the 1.0 ceiling either — a genuinely useful, imperfect model |
| **Are the learned coefficients actually sane?** | Sanity-check against domain knowledge, not just accept the numbers | `LOSS`'s negative coefficient (higher loss → lower predicted attorney probability, holding other features constant) is counter-intuitive — worth flagging as an association to investigate further, not a settled causal fact |
| **Does it meet a pre-agreed bar?** | Define "good enough" *before* modeling, not after | This is the real ship/no-ship gate — a business-specific number agreed upfront, not a universal threshold |

## The honest bottom line

There is no single universal number that means "good model" (no "accuracy above 80% = always ship it", no "ROC-AUC above 0.7 = always good"). It's always **relative to**:
1. The baseline (how much better than guessing the majority class?),
2. The actual cost of being wrong for this specific business problem (a missed fraud/attorney case vs. a false alarm rarely cost the same), and
3. A bar someone agreed on *before* training started.

A 67% accuracy model can be a real win on a genuinely hard, noisy human-behavior question like "will this person hire an attorney." The same 67% would be unacceptable for something like spam filtering, where far higher accuracy is realistically achievable and expected. "Good enough" is a business judgment applied to evaluation numbers — not a property the numbers have on their own.

---

## Side note — do you train one algorithm, or several and compare?

Both, in order:

1. **Narrow the field first**, using the problem and the data shape — target type (classification here, since `ATTORNEY` is 0/1), how linear the log-odds relationship looks, how much data you have, whether you need to *explain* the model to a stakeholder, and whether speed matters. This alone rules out most algorithms before you write any training code.
2. **Train the shortlisted candidates and compare them** on the *same* train/test split with the *same* metric — this is standard practice, not overkill. For this problem, reasonable candidates besides Logistic Regression would include a Decision Tree or Random Forest classifier, compared on the same ROC-AUC.
3. **Always include the simplest candidate as a first baseline-for-the-baseline** (Chapter 1's recap, Step 6 — "start simple"). If a fancier model only wins by a hair, the extra complexity usually isn't worth it — Logistic Regression is also far easier to explain to an insurance adjuster than a Random Forest's internals.

This comparison isn't a separate phase — it's just Model Building/Training/Evaluation (Steps 5-8) run once per shortlisted candidate, then compared side by side.
