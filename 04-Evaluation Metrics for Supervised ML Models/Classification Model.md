# Classification Model Evaluation Metrics

*(Confusion Matrix, Precision, Recall, ROC, AUC, F1-Score, Accuracy — all explained using the same trained model from `03-Logistic Regression/scripting.ipynb`: Logistic Regression on `assets/claim.csv`, predicting `ATTORNEY`. Every number below was actually computed with a script and verified by hand, not estimated.)*

## 0. Why do we need all these metrics, when we already have "accuracy"?

Recall `03-Logistic Regression/recap.md`: a model's raw predictions alone don't tell you if it's good — you need a *number* to judge it by. "Accuracy" (percent correct) feels like the obvious choice, and it's a real, useful metric... but it can also badly mislead you on its own. This whole chapter exists to fix that.

**The problem with accuracy alone — a concrete story:** imagine a rare disease that only 1% of patients actually have. A model that just says **"No disease"** for every single patient, no matter what, would score **99% accuracy** — and be completely useless, since it catches zero actual disease cases. Accuracy hides *what kind* of mistakes are happening. This is exactly why the other metrics on this page exist — each one answers a *different* question about the model's mistakes that accuracy alone can't answer.

---

## 1. The Confusion Matrix — the foundation everything else is built on

**What it is:** a simple 2×2 grid that counts every possible combination of (what actually happened) × (what the model predicted), for a binary (2-category) target.

Every other metric on this page (Precision, Recall, F1, Accuracy) is just a different arithmetic combination of these same 4 numbers — so this is the one concept to genuinely understand first; everything after this is just "which 2 of these 4 numbers do we divide."

### The 4 cells, defined

| | Predicted: No Attorney (0) | Predicted: Attorney (1) |
|---|---|---|
| **Actual: No Attorney (0)** | **True Negative (TN)** — correctly said "No" | **False Positive (FP)** — wrongly said "Yes" |
| **Actual: Attorney (1)** | **False Negative (FN)** — wrongly said "No" | **True Positive (TP)** — correctly said "Yes" |

**How to read the names — this is the part that trips people up, so slow down here:**
- The **first word** (True/False) tells you **was the model's prediction correct or wrong?** True = correct. False = wrong.
- The **second word** (Positive/Negative) tells you **what the model actually predicted** (not what really happened). Positive = model said "Yes/1." Negative = model said "No/0."

So **"False Positive"** reads as: *"the model's prediction was False (wrong), and what it predicted was Positive (Yes)."* In plain English: the model said "Yes" and was wrong — a **false alarm**. And **"False Negative"** reads as: *the model's prediction was False (wrong), and what it predicted was Negative (No)"* — the model said "No" and was wrong — a **missed case**.

| Term | Model predicted | Actually was | Plain English |
|---|---|---|---|
| **True Positive (TP)** | Yes (1) | Yes (1) | Correctly caught a real "Yes" case |
| **True Negative (TN)** | No (0) | No (0) | Correctly identified a real "No" case |
| **False Positive (FP)** | Yes (1) | No (0) | False alarm — flagged something that wasn't actually true |
| **False Negative (FN)** | No (0) | Yes (1) | Missed case — failed to catch something that actually was true |

### The real numbers, from our claim.csv model

Running the trained Logistic Regression from Chapter 3 on the 220-row test set:

```
                    Predicted: No   Predicted: Attorney
Actual: No               70                46
Actual: Attorney         26                78
```

| Term | Count | Meaning here |
|---|---|---|
| **TN** | 70 | Claims correctly predicted as "No Attorney" |
| **FP** | 46 | Claims wrongly predicted as "Attorney" (false alarms) |
| **FN** | 26 | Claims wrongly predicted as "No Attorney" (missed attorney cases) |
| **TP** | 78 | Claims correctly predicted as "Attorney" |

Check: `70 + 46 + 26 + 78 = 220` — matches the full test set size exactly, since every single test row falls into exactly one of these 4 boxes.

**Why this matters before any single metric:** recall Chapter 1's recap, Step 5 — "is a wrong prediction cheap or costly?" For this insurance problem, a **False Negative** (missing a claim that *will* involve an attorney) probably costs the insurer more than a **False Positive** (over-preparing for a claim that turns out simple) — they'd be caught understaffed on a contested case. Which of FP or FN matters more is *always* a business decision, not a mathematical one — and it's exactly what decides which metric below you should actually optimize for.

---

## 2. Accuracy — "what fraction did I get right, overall?"

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Plain English:** out of every prediction the model made, what fraction were correct (either correctly said "Yes" or correctly said "No")?

**Our numbers:** $\dfrac{78 + 70}{78 + 70 + 46 + 26} = \dfrac{148}{220} = 0.673$ → **67.3%**

**When it's trustworthy:** only when the two classes are reasonably balanced (recall the disease example in Section 0 — with a 99%/1% split, accuracy alone is dangerously misleading). Our `ATTORNEY` target is roughly 51%/49% (checked back in Chapter 3), so accuracy is a *reasonable* metric here — but as the sections below show, it still hides important detail that the confusion matrix and other metrics reveal.

---

## 3. Precision — "when I said Yes, how often was I actually right?"

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Plain English:** of every claim the model *flagged* as "Attorney," what fraction genuinely turned out to involve an attorney? This metric only looks at the model's "Yes" predictions — it completely ignores every case where the model said "No."

**Our numbers:** $\dfrac{78}{78 + 46} = \dfrac{78}{124} = 0.629$ → **62.9%**

**Reading it:** when this model raises a flag ("this claim will likely involve an attorney"), it's right about 63% of the time. The other 37% of flagged claims are false alarms (FP) — claims the model *thought* would involve an attorney but didn't.

**When Precision is the metric to care about most:** when a **False Positive is the expensive mistake**. E.g. spam filtering — if "Positive" means "flag as spam," a False Positive means a real, important email gets buried in the spam folder. You'd want *high precision* there — be very sure before flagging.

---

## 4. Recall (a.k.a. Sensitivity, or True Positive Rate) — "of everyone who actually was Yes, how many did I catch?"

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**Plain English:** of every claim that *genuinely* involved an attorney, what fraction did the model actually catch/flag? This metric only looks at the real "Yes" cases — it completely ignores everything the model said about actual "No" cases.

**Our numbers:** $\dfrac{78}{78 + 26} = \dfrac{78}{104} = 0.750$ → **75.0%**

**Reading it:** of all the claims that really did end up involving an attorney, the model successfully caught 75% of them — and missed the other 25% (FN, the ones it wrongly said "No" to).

**When Recall is the metric to care about most:** when a **False Negative is the expensive/dangerous mistake**. E.g. disease screening or fraud detection — missing a real case (FN) is far worse than a false alarm (FP), which just costs someone a second look. Recall from Section 1's business framing: missing an attorney-bound claim (FN) is likely the costlier mistake for this insurer too, which makes Recall a genuinely important number here, maybe more than Precision.

### Precision vs. Recall — the tradeoff, proven with real numbers

Precision and Recall almost always pull in **opposite directions**, because they're controlled by the same knob: the **decision threshold** — the probability cutoff (default 0.5) above which the model says "Yes." Recall Chapter 3's theory.md: the model doesn't just output "Yes/No" — it first computes a probability via Sigmoid, *then* thresholds it. Move that threshold, and precision/recall move in opposite directions:

| Threshold | TN | FP | FN | TP | Precision | Recall |
|---|---|---|---|---|---|---|
| **0.3** (say "Yes" more easily) | 35 | 81 | 6 | 98 | 0.547 | **0.942** |
| **0.5** (default) | 70 | 46 | 26 | 78 | 0.629 | 0.750 |
| **0.7** (say "Yes" only if very sure) | 111 | 5 | 78 | 26 | **0.839** | 0.250 |

**Reading this table, left to right:** lowering the threshold to 0.3 makes the model flag "Attorney" much more eagerly — it now catches 94.2% of real attorney cases (Recall way up), but at the cost of way more false alarms (Precision drops to 54.7%). Raising the threshold to 0.7 does the opposite: the model only says "Yes" when very confident, so when it does say Yes it's right 83.9% of the time (Precision way up), but it now misses most of the real cases (Recall crashes to 25%).

**This is the entire tradeoff, made concrete, not abstract:** you cannot maximize both Precision and Recall at the same time by moving the threshold — pushing one up pushes the other down. Which threshold to actually pick is, again, a business decision (Section 1) — driven by whether a False Positive or a False Negative costs the business more.

---

## 5. F1-Score — a single number balancing Precision and Recall

$$
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Plain English:** Precision and Recall often trade off against each other (Section 4). F1 is the **harmonic mean** of the two — a single score that's only high when *both* Precision and Recall are reasonably high. (A harmonic mean, unlike a plain average, punishes a big imbalance between the two numbers harder — a model with Precision=0.99 and Recall=0.02 would still score badly on F1, correctly reflecting that it's a bad model overall, even though a plain average would look deceptively okay.)

**Our numbers (at the default 0.5 threshold):** $2 \times \dfrac{0.629 \times 0.750}{0.629 + 0.750} = 2 \times \dfrac{0.472}{1.379} = 0.684$ → **68.4%**

**When to use F1 over Precision or Recall alone:** when you genuinely care about both false alarms *and* missed cases roughly equally, and want one summary number instead of tracking two separately — common when there's no strong business reason to favor one over the other, or when comparing multiple candidate models (recall the earlier "do we train one algorithm or several and compare" discussion) and you want one ranking number.

---

## 6. ROC Curve — visualizing the tradeoff across *every* possible threshold at once

Section 4 showed the Precision/Recall tradeoff at 3 hand-picked thresholds (0.3, 0.5, 0.7). The **ROC curve** (Receiver Operating Characteristic — an old name from signal-processing, don't worry about why) shows this tradeoff across **every possible threshold simultaneously**, as one picture.

**The two axes:**
- **X-axis: False Positive Rate (FPR)** = $\dfrac{FP}{FP + TN}$ — "of everyone who was actually 'No,' what fraction did I *wrongly* flag as 'Yes'?"
- **Y-axis: True Positive Rate (TPR)** = $\dfrac{TP}{TP + FN}$ — this is literally the same formula as **Recall** from Section 4, just renamed for this context.

**How the curve is built:** sweep the decision threshold from 1.0 down to 0.0, and at *every* threshold value, plot one point: (FPR at that threshold, TPR at that threshold). Connecting all those points draws the ROC curve.

**Real points from our model** (a handful of the 94 actual threshold points sklearn computed):

| Threshold | FPR | TPR (Recall) |
|---|---|---|
| very high (model almost never says "Yes") | 0.000 | 0.000 |
| 0.660 | 0.103 | 0.423 |
| 0.594 | 0.259 | 0.596 |
| 0.475 | 0.414 | 0.750 |
| 0.384 | 0.586 | 0.885 |
| 0.000 (model always says "Yes") | 1.000 | 1.000 |

```
 TPR (Recall)
  1.0 │                                              ●  ← threshold near 0 (always "Yes")
      │                                    ●
      │                          ●
      │                ●
      │        ●
      │  ●
      │●  ← threshold near 1 (always "No")
      └──────────────────────────────────────────  FPR
       0.0                                       1.0

       diagonal dashed line (not drawn above) from (0,0) to (1,1)
       = what a coin-flip / random-guessing model would look like
```

**Reading the shape:** at one extreme (very high threshold), the model almost never says "Yes" — so it has 0 false alarms (FPR=0) but also catches almost nothing real (TPR≈0) — bottom-left corner. At the other extreme (threshold=0), the model says "Yes" to everything — it catches every real case (TPR=1) but also falsely flags every real "No" (FPR=1) — top-right corner. A genuinely useful model's curve bulges up and to the **left**, toward the top-left corner (high TPR, low FPR) — meaning there's *some* threshold where you get strong recall without paying for it in tons of false alarms. A useless model's curve would sit right on the diagonal line from (0,0) to (1,1) — exactly matching what random coin-flip guessing would produce, no matter the threshold.

---

## 7. AUC — turning the whole ROC curve into one number

**AUC = Area Under the (ROC) Curve.** Instead of eyeballing the curve's shape, AUC compresses the entire thing into a single number between 0 and 1: literally the area under that curve.

**Our number:** **AUC = 0.745**

**How to read AUC values:**

| AUC | Meaning |
|---|---|
| **1.0** | Perfect — the model can find a threshold that perfectly separates every "Yes" from every "No" |
| **0.5** | Useless — exactly as good as flipping a coin, no matter the threshold (the diagonal line in the picture above) |
| **0.745 (ours)** | Genuinely useful, but far from perfect — meaningfully better than a coin flip, with real room to improve |
| **< 0.5** | Worse than random — actually means the model's predictions are backwards (rare, but possible with a broken setup) |

**A cleaner intuitive definition, worth remembering:** AUC = *"if I randomly pick one real 'Attorney' claim and one real 'No Attorney' claim, what's the probability the model gives the Attorney claim a higher predicted probability than the No-Attorney claim?"* Our AUC of 0.745 means: about 74.5% of the time, the model correctly ranks a true "Attorney" case above a true "No Attorney" case, when picked at random.

**Why AUC is genuinely useful, beyond just "one number for the curve":** it evaluates the model's ability to **rank/separate** the two classes across *every* threshold at once — meaning it doesn't force you to commit to one threshold (like 0.5) before judging the model. This is valuable because the *right* threshold is a business decision (Section 4) that might change later — AUC tells you if the underlying model is fundamentally good at distinguishing the classes, independent of wherever you eventually set that threshold.

---

## The full picture, tied together

1. **Confusion Matrix** — the 4 raw counts (TN, FP, FN, TP) everything else is built from. Always start here.
2. **Accuracy** — overall correctness, but misleading on imbalanced data (Section 0's disease example).
3. **Precision** — "when I say Yes, am I right?" — the metric to prioritize when False Positives are the costly mistake.
4. **Recall** — "of all real Yes cases, how many did I catch?" — the metric to prioritize when False Negatives are the costly mistake.
5. **F1-Score** — one number balancing Precision and Recall, useful when neither type of mistake clearly dominates, or when you need a single number to rank multiple candidate models.
6. **ROC Curve** — visualizes the Precision/Recall-style tradeoff (specifically TPR vs. FPR) across *every* possible threshold at once, not just one.
7. **AUC** — compresses the whole ROC curve into one number (0.5 = useless, 1.0 = perfect), measuring how well the model can *rank/separate* the two classes independent of any one chosen threshold.

**The one habit to walk away with:** never report just accuracy, and never report just one metric in isolation. Look at the confusion matrix first (it's the ground truth every other number is derived from), pick the metric(s) that match the actual business cost of being wrong (Section 1), and use AUC/ROC when you want to judge the model's underlying quality independent of any specific threshold choice.
