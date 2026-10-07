# Chapter 3 — Logistic Regression (Simple Recap)

> Read only this file for now. `theory.md` and `scripting.ipynb` are there if you ever need depth.

---

## 1. What is it? (one line)

**Linear Regression predicts a number. Logistic Regression predicts YES or NO, plus how sure it is.**

| Linear Regression (Ch. 2) | Logistic Regression (Ch. 3) |
|---|---|
| "This car will give **32.5 MPG**" | "This claimant will hire an attorney: **YES, 73% sure**" |

Despite the name, it's a **classification** model (yes/no).

---

## 2. How it works: 3 simple steps

```
Step 1  SCORE      z = m1·x1 + m2·x2 + … + b     ← same formula as Linear Regression
Step 2  SQUASH     p = sigmoid(z)                ← turns any score into 0% to 100%
Step 3  DECIDE     p ≥ 0.5 → YES,  else NO       ← 0.5 is the "threshold" (you can change it)
```

![sigmoid](images/02_sigmoid.png)

**Sigmoid** is just a squasher: big negative score → close to 0%, big positive score → close to 100%, a score of 0 → exactly 50%.

**Example:** score = −0.32 → sigmoid → **42%** → below 50% → predict **NO**.

---

## 3. How does it learn? Log Loss

Training = adjusting the `m`'s until the model's mistakes are as small as possible. The "mistake score" is called **Log Loss**:

| Model said (for the correct answer) | Penalty |
|---|---|
| 99% sure, and right | 0.01 (tiny) |
| 50/50, unsure | 0.69 (medium) |
| 99% sure, and **wrong** | **4.6 (huge)** |

**Key idea: being confidently wrong is punished very hard.** `.fit()` handles all of this.

---

## 4. The project in 6 lines (`scripting.ipynb`)

1. **Question:** will an insurance claimant hire an attorney? (yes/no → Logistic Regression)
2. **Data:** 1,340 claims → removed the ID column and rows with blanks → 1,096 claims.
3. **Split:** 80% to learn, 20% (220 claims) to test.
4. **Result:** **67% accuracy** vs **53%** from just guessing "No" every time. ✅ Better than guessing.
5. **Mistakes:** 26 real attorney cases were missed, and there were 46 false alarms.
6. **Improvement:** LOSS (the money amount) was very uneven, so we used `log(LOSS)` → the model got a bit better.

---

## 5. Judging a yes/no model

### a) Two kinds of mistakes
| Mistake | Name | Example |
|---|---|---|
| Said YES, was NO | **False alarm** (False Positive) | Spam filter deletes a real email |
| Said NO, was YES | **Miss** (False Negative) | Fraud goes undetected |

### b) Precision vs Recall
- **Precision** = *"When I say YES, how often am I right?"* → care about it when **false alarms** are costly.
- **Recall** = *"Of all real YES cases, how many did I catch?"* → care about it when **misses** are costly.

### c) The threshold is a business decision
| Threshold | Effect |
|---|---|
| Low (0.3) | Says YES easily → **catches almost everything**, but more false alarms |
| High (0.7) | Says YES rarely → **few false alarms**, but misses a lot |

Our project: missing an attorney case is costly → use **0.4 instead of 0.5**.

⚠️ **Accuracy can lie.** If 99% of payments are genuine, "always say genuine" is 99% accurate and catches **zero** fraud. Always check the misses.

### d) Where do "26 missed, 46 false alarms" come from?
`predict()` only gives the model's answers. **Compare them with the real answers (`y_test`), row by row**, and put each row into one of 4 buckets:

| Real | Model said | Bucket |
|---|---|---|
| 1 | 1 | **TP**: correctly caught ✅ |
| 0 | 0 | **TN**: correctly said no ✅ |
| 0 | 1 | **FP**: false alarm ❌ |
| 1 | 0 | **FN**: missed ❌ |

Count all 220 test claims → **TN 70, TP 78, FP 46, FN 26.** One line does the counting:
```python
confusion_matrix(y_test, model.predict(X_test))
# [[70, 46],    ← real NO:  70 said No (TN), 46 said Yes (FP)
#  [26, 78]]    ← real YES: 26 said No (FN), 78 said Yes (TP)
```

### e) What does "tune the threshold" mean?
The model doesn't directly say YES/NO. It gives each claim a **% chance**. The **threshold is the cut-off line**: at or above the line → YES, below → NO. It works like an exam **pass mark**.

| Claim | Model's % | Real | Line at 50% | Line at 40% |
|---|---|---|---|---|
| A | 75% | YES | YES ✅ | YES ✅ |
| B | 45% | YES | NO ❌ **missed** | YES ✅ **caught** |
| C | 42% | NO | NO ✅ | YES ❌ **false alarm** |
| D | 20% | NO | NO ✅ | NO ✅ |

Moving the line from 50% to 40% **caught B** but **wrongly flagged C**. **Tuning the threshold = choosing where to put this line**, based on which mistake hurts the business more. The model itself doesn't change and nothing is retrained:
```python
proba = model.predict_proba(X_test)[:, 1]     # the % for each claim
pred  = (proba >= 0.4).astype(int)            # our own line instead of the default 0.5
```

Real numbers from our model:
| Line | Missed | False alarms |
|---|---|---|
| 50% | 26 | 46 |
| **40%** | **17** | 66 |
| 30% | 6 | 81 |

### f) So is "reduce misses" always the goal?
**No. The business decides which mistake is costlier.**
- Misses are costly (insurance, cancer, fraud) → reduce misses (**recall**).
- False alarms are costly (spam filter, blocking real customers) → reduce false alarms (**precision**).
- Reducing one usually increases the other. **Only a better model** (better features, a stronger algorithm) can reduce both.

---

## 6. What is `log(LOSS)`? (it's NOT part of the model)

⚠️ Three names that sound alike but are unrelated:
| Term | What it is |
|---|---|
| **Log**istic Regression | The model's name |
| **Log** Loss | How the model scores its own mistakes during training |
| **log(LOSS)** | **Our data change.** LOSS is a column (money lost on the claim) |

LOSS was very uneven: most claims were about $1k–4k, but a few were $100k+. `log()` **squeezes the huge values** (170 → 5.1, 10 → 2.4, 2 → 1.1) so they don't distort the model. It's one line **before** `.fit()`:
```python
X["LOSS"] = np.log1p(X["LOSS"])    # change the column, then fit the same model as before
```
- It's **feature engineering** (fixing the input data), the same idea as adding HP² in Chapter 2.
- You *can* do it from the start for money/price columns. But the good habit is: **check first → fix → prove it helped** (in Chapter 2, WT² made things worse).
- It helps straight-line models (Linear/Logistic). Trees (Ch. 5–6) usually don't need it.

---

## 7. ✅ The protocol (same for Linear AND Logistic)
```
1. Get data → understand → clean
2. Split train / test
3. fit() on train
4. predict() on test → compare with the real answers
5. Compare with the "no ML" baseline
6. Good enough for the client? → YES: ship
                               → NO: diagnose → fix → retrain → compare again
```

| | Baseline | Measured by |
|---|---|---|
| Linear (number) | always predict the **average** | RMSE, R² |
| Logistic (yes/no) | always predict the **most common answer** | accuracy, misses, false alarms |

**Fixes, in the order to try:**
| Fix | Example |
|---|---|
| 1. Fix the data / features | drop twin columns, add HP², `log(LOSS)` |
| 2. Tune the threshold (yes/no only) | 0.5 → 0.4 to reduce misses |
| 3. Try a different model | Decision Tree, Random Forest, XGBoost (Ch. 5–6) |
| 4. Get better / more data | new useful columns |

**Always diagnose before fixing:** wrong signs → twin columns, curve → squared column, very uneven column → log, too many misses → lower the threshold. Keep a fix only if the score improves.

This same loop (build → measure against a baseline → diagnose → fix → re-measure) is how you'll improve LLM agents with **evals**.

---

## 8. ⭐ Why this matters for an Agentic AI developer

This chapter is the most directly useful one for LLM work:

| What you learned here | Where you'll meet it in GenAI / Agents |
|---|---|
| **Probability + threshold** | Agents act on confidence: *"if the model is less than 80% sure → ask a human / ask a clarifying question"* (human-in-the-loop routing) |
| **Log Loss (cross-entropy)** | **Exactly how GPT and Claude are trained**: punished when they give low probability to the correct next word |
| **Softmax** (yes/no extended to many options) | An LLM picks the next word by giving a probability to every word in its vocabulary. **Temperature** controls how "spread out" those probabilities are |
| **Classification** | Very common agent tasks: **intent detection** ("is this a refund request or a complaint?"), **routing** to the right tool/agent, **guardrails** ("is this message unsafe?"), sentiment |
| **Precision / Recall / Confusion matrix** | **Evaluating your LLM app (evals)**: *"my guardrail blocked 5 safe messages (false alarms) and let 2 unsafe ones through (misses)"* |
| **The threshold is a business choice** | Guardrails: a strict threshold → safer but annoys users. A loose one → smoother but riskier. You'll tune this. |
| **Baseline** | Always compare: *does my fancy agent beat a simple keyword rule?* |

**Interview-ready one-liner:**
> "Logistic regression computes a linear score, squashes it with sigmoid into a probability, and is trained by minimising cross-entropy. That same loss, with softmax over the vocabulary, is how LLMs are trained."

---

## 9. What you can skip for now
- The Sigmoid and Log Loss formulas (just remember the ideas above)
- Odds, log-odds, odds ratios
- ROC curve / AUC details (just know: **AUC 0.5 = random, 1.0 = perfect**, ours = 0.75)
- The assumptions and the log-odds linearity check
- Cross-validation details (Chapter 7)

---

## 10. Remember in 5 lines
1. Logistic Regression = **YES/NO + a probability**.
2. **Score → sigmoid → threshold.**
3. Trained with **Log Loss**: confident mistakes cost the most (same as LLM training).
4. Judge it by **misses vs false alarms** (recall vs precision), not just accuracy.
5. The **threshold is a business choice**, and it's the same idea as confidence thresholds in AI agents.
