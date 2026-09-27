# Chapter 7 — Model Evaluation Techniques

*(Small hand-worked illustrations use the Play Tennis dataset from Chapter 5. Every technique is then run for real in `scripting.py` on `assets/claim.csv`, the same ATTORNEY problem as Chapters 3, 5 and 6, always with the **same model** (Chapter 5's tuned tree). The only thing that changes between sections is **how the model is evaluated**.)*

## 0. Where this sits in the big picture

Chapter 4 answered **"which number do we measure?"** (accuracy, precision, recall, AUC, RMSE, R²...). This chapter answers a different question:

> **"On *which data*, and *how many times*, do we measure it, so that the number is a trustworthy prediction of how the model will do on data it has never seen?"**

The metric is the **ruler**; the evaluation technique is **how carefully you use the ruler**. A perfect ruler used once, on the wrong thing, still gives a misleading answer.

You've already used two of these techniques without studying them properly:
- **Train/test split** (Chapter 4, `train_test_split.md`): every chapter since.
- **K-fold cross-validation** (Chapter 5, Section 13): inside `GridSearchCV(cv=5)`.

This chapter covers the whole family: why each one exists, how it works, what it costs, and when to use which.

| Technique | One-line idea |
|---|---|
| **Holdout** (train/test split) | Split once, test once |
| **Train / Validation / Test** | Split three ways, so tuning and final testing use different data |
| **K-Fold CV** | Split into k chunks, rotate which chunk is the test |
| **Stratified K-Fold** | K-Fold, but every chunk keeps the class ratio |
| **LOOCV** (Leave-One-Out) | K-Fold with k = number of rows |
| **Leave-P-Out** | Test on every possible group of p rows |
| **Repeated K-Fold** | K-Fold, several times with different shuffles |
| **ShuffleSplit** (Monte Carlo CV) | Many independent random train/test splits |
| **TimeSeriesSplit** | Always train on the past, test on the future |
| **GroupKFold** | Keep all rows of one person/customer on the same side |
| **Nested CV** | CV *around* the tuning, for an honest estimate of a tuned model |
| **Bootstrap / OOB** | Resample with replacement; test on rows left out (Chapter 6, Section 4) |

---

## 1. The foundation: train, validation and test data

### The exam analogy

| Data | School analogy | What it's used for | Allowed to influence the model? |
|---|---|---|---|
| **Training set** | The textbook + homework | The model **learns** from it (`.fit()`) | Yes |
| **Validation set** | Mock exams | **Choosing** between models / hyperparameters (Chapter 5, Section 12) | Yes, *indirectly*, through your choices |
| **Test set** | The final board exam | **One** honest, final score | **Never** |

The key subtlety is the middle row. Every time you look at a score and *change something* because of it, that data starts to influence the model. A student who takes the same mock exam 20 times, tweaking their answers each time, will ace that mock but it no longer predicts their real exam. **That's why the final test set must be looked at once, at the very end.**

### The two errors every technique is trying to avoid

1. **An unlucky/lucky estimate (high variance of the estimate):** the score depends heavily on *which* rows happened to be in the test set. Fix: test on more rows, more times (K-Fold, repeats).
2. **An optimistic estimate (bias/leakage):** information from the test data leaked into training or tuning, so the score is too good to be true. Fix: strict separation (validation sets, nested CV, pipelines).

---

## 2. Holdout method (train/test split)

**How it works:** shuffle, then cut the data once, typically **80/20** or **70/30**. Train on the big part, score on the small part. `stratify=y` keeps the class ratio the same on both sides (Chapter 4). Covered in detail in `04-.../train_test_split.md`.

**Pros:** simple, fast (**1 fit**), and it mirrors real life (train on the past, predict the new).

**Cons:**
1. **The score depends on luck.** Which 20% landed in the test set?
2. **Wastes data.** 20% of rows never help train the model; that hurts on small datasets.

### Real proof of the luck problem (`scripting.py`, 7.1)

Same model, same data, 50 different `random_state` values (only the shuffle changes):

| | Test accuracy |
|---|---|
| `random_state=42` (the split Chapters 5–6 used) | 0.732 |
| Worst of 50 splits | **0.664** |
| Best of 50 splits | **0.764** |
| Average of 50 | 0.720 (std 0.023) |

**A 10-point swing from luck alone.** If you'd reported "76.4%" from one lucky split, the business would be disappointed next month. Also note: the 0.732 we've been quoting since Chapter 5 was a *slightly lucky* split.

**When to use holdout:** very large datasets (100,000+ rows), where 20% is plenty for a stable score and K-Fold would be slow, or as the **final** test set wrapped around cross-validation (Section 14).

---

## 3. Train / Validation / Test (the three-way split)

**The problem it solves:** if you tune hyperparameters by checking the *test* score, the test set is secretly part of training (Chapter 5, Section 12's golden rule). So split **three** ways:

```
  All data (1096)
  ├── Test (20% = 220)         ← locked away until the very end
  └── Rest (80% = 876)
       ├── Train (60% = 657)    ← .fit()
       └── Validation (20% = 219) ← compare settings here
```

### Real example (`scripting.py`, 7.2): choosing `max_depth` on validation only

| `max_depth` | 1 | 2 | 3 | 4 | 5 | 6 | 8 | None |
|---|---|---|---|---|---|---|---|---|
| Train acc | 0.729 | 0.729 | 0.734 | 0.756 | 0.785 | 0.795 | 0.836 | 0.998 |
| **Validation acc** | **0.708** | **0.708** | **0.708** | 0.703 | 0.676 | 0.671 | 0.635 | 0.607 |

It's the same overfitting curve as Chapter 5, found **without ever touching the test set**. Depths 1–3 tie, and **when models tie, prefer the simplest**, so `max_depth=1`. Retrain on train + validation (876 rows), then score the test set once: **0.732**.

**Weaknesses:**
- **Even more data is held out:** only 60% trains the model during tuning.
- **The choice rests on one 219-row validation set**, which has the same luck problem as holdout.

Both are fixed by using K-Fold *instead of* a single validation set.

---

## 4. K-Fold Cross-Validation

### How it works

1. Shuffle the rows and cut them into **k equal chunks ("folds")**.
2. **Round 1:** train on folds 2…k, validate on fold 1.
3. **Round 2:** train on folds 1, 3…k, validate on fold 2.
4. …repeat until **every fold has been the validation set exactly once**.
5. Report the **mean ± standard deviation** of the k scores.

```
k = 5                    fold 1   fold 2   fold 3   fold 4   fold 5
Round 1:                [VALID]  [train]  [train]  [train]  [train]   → score₁
Round 2:                [train]  [VALID]  [train]  [train]  [train]   → score₂
Round 3:                [train]  [train]  [VALID]  [train]  [train]   → score₃
Round 4:                [train]  [train]  [train]  [VALID]  [train]   → score₄
Round 5:                [train]  [train]  [train]  [train]  [VALID]   → score₅

CV score = mean(score₁…score₅) ± std(score₁…score₅)
```

### On Play Tennis (14 days, k = 7)

Each fold holds 2 days (shown without shuffling, to keep it readable): fold 1 = days {1, 2}, fold 2 = {3, 4}, fold 3 = {5, 6}, … fold 7 = {13, 14}. Seven models get trained; each is scored on the 2 days it never saw. **Every day is tested exactly once, and used for training 6 times.**

### Why it beats holdout

| | Holdout | K-Fold (k = 5) |
|---|---|---|
| Rows used for testing | 20% | **100%** (each once) |
| Rows used for training | 80% | 80% per round, and every row trains 4 of the 5 models |
| Number of scores | 1 | **5** → a mean *and* a spread |
| Fits (cost) | 1 | 5 |

The **± spread** is new information holdout can't give you. It tells you how much the score depends on luck.

### Real result (`scripting.py`, 7.3)

| Fold | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Validation rows | 220 | 219 | 219 | 219 | 219 |
| Share of ATTORNEY=1 | 0.495 | **0.425** | 0.470 | **0.507** | 0.466 |
| Accuracy | 0.682 | 0.726 | 0.717 | 0.731 | 0.749 |

**K-Fold accuracy = 0.721 ± 0.022.** The share of attorney cases *wanders* from fold to fold (0.425 to 0.507, vs 0.473 overall). That's what stratification fixes.

---

## 5. Stratified K-Fold

**What it changes:** each fold is built to have **the same class ratio as the full dataset**. It's `stratify=y` from Chapter 4, applied to every fold.

### Real result (`scripting.py`, 7.4)

| Fold | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Plain K-Fold: ATTORNEY=1 share | 0.495 | 0.425 | 0.470 | 0.507 | 0.466 |
| **Stratified**: ATTORNEY=1 share | **0.473** | **0.470** | **0.470** | **0.475** | **0.475** |

Stratified K-Fold accuracy = 0.725 ± 0.036. *Honest note:* the spread isn't smaller here. Stratification fixes the **class balance** in each fold, not every source of randomness. **Its value grows with imbalance**: with 2% fraud cases, a plain fold could contain almost *no* fraud at all, making recall impossible to measure.

### The disaster it protects you from — sorted data

Real exports are often **sorted**: by date, by ID, or by outcome. The same data sorted by ATTORNEY (all 0s, then all 1s), with **no shuffling**:

| Method | Fold scores |
|---|---|
| Plain K-Fold, no shuffle | 0.750, **0.470**, 0.721, 0.607, **0.502** |
| Stratified K-Fold, no shuffle | 0.723, 0.749, 0.717, 0.703, 0.735 |

With plain K-Fold, fold 1 validates on almost only 0s while training on a very different mix, and the scores are garbage. **Rules:** always `shuffle=True` for non-time data, and **always use Stratified K-Fold for classification.** (sklearn does this automatically when you pass `cv=5` to a classifier; `cross_val_score` and `GridSearchCV` use `StratifiedKFold` under the hood.)

---

## 6. How to choose k

Choosing k is a bias-variance tradeoff, but *of the estimate*, not of the model:

| | Small k (e.g. 2) | **k = 5 or 10** | Large k (e.g. LOOCV) |
|---|---|---|---|
| Training data per round | 50%, so models are handicapped and the score is **pessimistic** | 80–90% | ~100% |
| Validation fold size | Large, so each score is stable | Medium | Tiny (1 row) |
| Cost | k fits (cheap) | 5–10 fits | n fits (expensive) |
| Verdict | Too pessimistic | **The standard choice** | Only for tiny data |

### Real result (`scripting.py`, 7.5)

| k | Fits | Accuracy |
|---|---|---|
| 2 | 2 | 0.719 ± 0.011 |
| **5** | 5 | **0.725 ± 0.036** |
| 10 | 10 | 0.715 ± 0.044 |

The spread grows with k because each validation fold shrinks (k=10 → ~110 rows each, so individual fold scores bounce more). The means are all within 0.01 of each other. **Practical rule: use k = 5 by default, k = 10 if data is small-to-medium and fits are cheap.**

---

## 7. Leave-One-Out Cross-Validation (LOOCV)

**How it works:** K-Fold taken to the extreme, with **k = number of rows**. Each round trains on *all rows but one* and tests on that single row. For n rows, that's **n fits**, and each round's score is just **1 (right) or 0 (wrong)**.

### Worked example — LOOCV on Play Tennis (14 days → 14 models)

Train a tree on 13 days, predict the 14th, and repeat for every day. *(Computed with sklearn's tree on one-hot-encoded features, which uses binary splits, so it's close to but not identical to Chapter 5's hand-built ID3 tree.)*

| | Result |
|---|---|
| Tree trained on all 14 days, scored on the same 14 days | **14/14 = 100%** |
| **LOOCV** (each day predicted by a tree that never saw it) | **8/14 = 57%** (wrong on days 4, 6, 8, 11, 12, 14) |
| Baseline "predict the majority class of the other 13 days" | **9/14 = 64%** |

The baseline works like this: leave out a **Yes** day and the other 13 are 8 Yes / 5 No, so it predicts Yes, which is correct. Leave out a **No** day and the other 13 are 9 Yes / 4 No, so it predicts Yes, which is wrong. That's 9 right and 5 wrong.

**This is the most important result in the chapter.** Chapter 5's "perfect" tree is a **memorizer**: with only 14 rows, it generalizes *worse than always saying Yes*. Only an honest evaluation technique could reveal that, and with just 14 rows, LOOCV is exactly the right technique, because every row is precious for training.

### Real result on claim.csv (`scripting.py`, 7.6)

**1096 fits**, taking ~3.5 seconds (vs ~0.02 s for 5-fold). LOOCV accuracy = **0.725**, identical to Stratified 5-Fold (0.725).

### Pros and cons

| ✅ Pros | ❌ Cons |
|---|---|
| Uses the maximum possible training data (n − 1 rows) | **n fits**: impractical for big data or slow models |
| No randomness: the same answer every run | Each score is 0 or 1, so the per-fold ± is meaningless |
| Ideal for tiny datasets (tens to a few hundred rows) | The n models are nearly identical, so the estimate can still have high variance |
| | Can't be stratified (a fold of 1 row has no "ratio") |

**When to use:** small datasets, roughly under a few hundred rows (medical studies, lab experiments, Play Tennis). On 1096 rows it bought nothing over 5-fold, at ~100× the cost.

---

## 8. Leave-P-Out (LPO)

The generalization of LOOCV: test on **every possible group of p rows**. The catch is the number of combinations:

| | p = 1 (LOOCV) | p = 2 |
|---|---|---|
| Play Tennis (14 rows) | 14 fits | C(14, 2) = **91** fits |
| claim.csv (1096 rows) | 1,096 fits | C(1096, 2) = **600,060** fits |

It's exhaustive and deterministic, but it explodes combinatorially. **Know the name; you'll almost never use it** beyond tiny datasets. (`sklearn.model_selection.LeavePOut`.)

---

## 9. Repeated K-Fold and ShuffleSplit

### Repeated (Stratified) K-Fold

Run K-Fold **several times, with a different shuffle each time**, then average everything. 5 folds × 10 repeats = **50 scores**.

**Real result (`scripting.py`, 7.7):** 50 scores, mean **0.715**, std 0.027, range 0.662 to 0.785. Even a single fold ranged over 12 points, yet the average of 50 is the most stable estimate in the chapter. **Use it** when you're comparing close models and want to know if a small difference is real.

### ShuffleSplit (Monte Carlo cross-validation)

Make **many independent random train/test splits** (e.g. 10 random 80/20 splits) and average the scores. It's essentially Section 2's "50 seeds" experiment, packaged. Unlike K-Fold, a row can land in several test sets, or in none.

**Real result (`scripting.py`, 7.8):** 10 splits, **0.707 ± 0.021**. **Use it** when you want to control the train/test sizes independently of the number of rounds (e.g. 100 rounds of 90/10).

---

## 10. In code: `cross_val_score` vs `cross_validate`, and comparing models fairly

- **`cross_val_score(model, X, y, cv=...)`** returns one array of scores (one metric).
- **`cross_validate(model, X, y, cv=..., scoring=[...], return_train_score=True)`** returns several metrics at once, **plus the training score** of every fold, so you can see the generalization gap (Chapter 5, Section 11) directly.

**The fairness rule:** when comparing models, give them **the exact same folds** (the same `cv` object with a fixed `random_state`). Otherwise part of the difference is just different luck.

### Real result — and a plot twist (`scripting.py`, 7.9)

Stratified 5-Fold, same folds for all three:

| Model | Train acc | CV acc | CV AUC |
|---|---|---|---|
| Logistic Regression | 0.707 | 0.706 ± 0.022 | **0.757** ± 0.036 |
| Decision Tree, default | 0.996 | 0.620 ± 0.017 | 0.621 ± 0.018 |
| Decision Tree, tuned | 0.726 | **0.725** ± 0.036 | 0.751 ± 0.036 |

**Plot twist:** on Chapter 5's single test split, the tuned tree's AUC *beat* Logistic Regression (0.759 vs 0.745). With 5-fold CV, **Logistic Regression's AUC is slightly higher** (0.757 vs 0.751), and the ±0.036 spreads overlap completely. The honest conclusion is that **they're tied on AUC, and the tree wins on accuracy**. One split had exaggerated the gap in the tree's favour. That's exactly why this chapter exists.

---

## 11. Special data — when rows aren't independent

Every technique so far assumes rows are **independent**: shuffling them is harmless. Two common situations break that assumption, and random K-Fold then gives **optimistic, wrong** scores.

### Time series → `TimeSeriesSplit`

**The problem:** predicting next month's sales from past sales. Random K-Fold would train on *December* and test on *March*, using the future to predict the past. That's impossible in real life, so the score is fake.

**The fix:** folds that always **train on the past and validate on the future**, with the training window growing each round. From `scripting.py` 7.10, on 12 months:

```
Round 1:  train months [0, 1, 2]                    → validate [3, 4, 5]
Round 2:  train months [0 … 5]                      → validate [6, 7, 8]
Round 3:  train months [0 … 8]                      → validate [9, 10, 11]
```

Never shuffle time-series data.

### Grouped data → `GroupKFold`

**The problem:** a customer with 3 claims in the data. If 2 claims land in training and 1 in validation, the model may just *recognize the customer* (same age, same sex…) rather than learn a general pattern. The same happens with patients with several scans, or students with several exam attempts.

**The fix:** all rows of a group stay on the **same side**. From `scripting.py` 7.10 (12 claims from customers A–E): validate on {B, C} and train on {A, D, E}; then validate {A} and train {B, C, D, E}; then validate {D, E} and train {A, B, C}. No customer is ever on both sides.

(There's also `StratifiedGroupKFold` when you need both group separation and class balance.)

---

## 12. Nested cross-validation — honest scores for *tuned* models

### The subtle problem

`GridSearchCV.best_score_` (Chapter 5, Section 13) is the CV score of **the best of many combinations**. Picking the best of many is itself a kind of fitting: the winner is partly *genuinely good* and partly *lucky on those particular folds*. So `best_score_` tends to be **optimistic** as an estimate of real-world performance.

**Story:** 320 students each take the same mock exam and you pick the top scorer. Their mock score overstates their real ability, because part of why they came top was luck on that specific paper.

### The fix: two loops

```
OUTER loop (5 folds) — only for SCORING:
   for each outer fold:
       INNER loop (5 folds) — GridSearchCV picks the best hyperparameters using ONLY the outer-training rows
       score that tuned model on the outer-validation fold (which played no part in the tuning)
   nested CV score = mean of the 5 outer scores
```

In code, you pass a whole `GridSearchCV` object into `cross_val_score`.

### Real proof (`scripting.py`, 7.11) — a small 100-claim project

We simulated having only 100 claims, with a 320-combination grid. Because we actually have 996 *other* claims, we can check which estimate told the truth:

| Estimate | Accuracy |
|---|---|
| `GridSearchCV.best_score_` (non-nested) | **0.770**, which looks great |
| **Nested CV** | **0.670** |
| **The truth:** accuracy on 996 genuinely unseen claims | **0.626** |

The non-nested score was **14 points too optimistic**, while nested CV was far closer to the truth. On the full 1096 rows with Chapter 5's smaller grid, the two agreed to three decimals (0.731 vs 0.731). **The optimism is a small-data, big-grid danger**, which is exactly when people are most tempted to trust it.

**When to use nested CV:** reporting a trustworthy performance number for a *tuned* model when data is small or the grid is large, or comparing *tuning procedures* (e.g. "tuned tree vs tuned XGBoost") fairly. It costs (outer k) × (inner k) × (grid size) fits.

---

## 13. Bootstrap and Out-of-Bag (OOB) evaluation — a quick reminder

Covered in Chapter 6, Section 4: draw n rows **with replacement**, train on that sample, and evaluate on the **≈36.8% of rows never drawn** (the out-of-bag rows). Random Forest and Bagging give you this estimate for free (`oob_score=True`; 0.686 for the bagged trees in Chapter 6). It's also used in statistics to put confidence intervals on any metric.

---

## 14. When to use what — how to decide

### Decision guide

```
Is the data ordered in time?  ──yes──►  TimeSeriesSplit (never shuffle)
        │ no
Can one person/customer/patient have several rows?  ──yes──►  GroupKFold (or StratifiedGroupKFold)
        │ no
How many rows?
   ├── tiny (< ~200)          ──►  LOOCV, or Repeated Stratified K-Fold
   ├── small–medium (~200 to ~100k) ──►  Stratified K-Fold, k = 5 or 10  (Repeated, if comparing close models)
   └── huge (100k+)           ──►  Holdout (or train/validation/test) is enough, and K-Fold is costly
Are you tuning hyperparameters AND need an honest final number on small data?  ──►  Nested CV
```

### Comparison table

| Technique | Fits needed | Uses all data for testing? | Stability of estimate | Best for |
|---|---|---|---|---|
| Holdout | 1 | No (20%) | Low: luck-dependent | Huge datasets; final test set |
| Train / Val / Test | 1 per setting | No | Low | Huge datasets with tuning |
| **K-Fold** | k | Yes | Good | General use (regression) |
| **Stratified K-Fold** | k | Yes | Good | **General use (classification)** |
| LOOCV | n | Yes | Deterministic, but noisy per row | Tiny datasets |
| Leave-P-Out | C(n, p) | Yes | High, but infeasible | Almost never |
| Repeated K-Fold | k × repeats | Yes (many times) | **Best** | Comparing close models |
| ShuffleSplit | chosen | Partly, random | Good with many rounds | Flexible sizes |
| TimeSeriesSplit | k | Future folds only | – | **Time-ordered data** |
| GroupKFold | k | Yes | Good | **Grouped rows** |
| Nested CV | k_out × k_in × grid | Yes | Honest for tuned models | Small data + tuning |

### The recommended everyday workflow (what professionals do)

1. **Lock away a final test set** (holdout, 20%, stratified). Don't look at it.
2. On the remaining 80%, use **Stratified K-Fold CV** (via `GridSearchCV`) to compare models and tune hyperparameters.
3. Score the single chosen model **once** on the locked test set. This checks that CV didn't mislead you.
4. **Retrain the final model on all the data** and ship it (`scripting.py`, Step 9). Quote the CV mean ± spread as its expected performance.

---

## 15. Common mistakes (all of them make the score look better than reality)

| Mistake | Why it's wrong | Fix |
|---|---|---|
| Tuning on the test set / peeking repeatedly | The test set leaks into your choices | Validation set or CV; test once |
| **Preprocessing before splitting** (e.g. fitting a scaler, imputer, or feature selection on *all* data) | Statistics from the validation rows leak into training | Put preprocessing inside a `Pipeline`, so it's re-fitted inside every fold (a later chapter) |
| Not shuffling ordered data (non-time) | Folds get very different class mixes (Section 5's 0.47 fold) | `shuffle=True` / stratify |
| Shuffling time-series data | Trains on the future | `TimeSeriesSplit` |
| Same person on both sides | Model recognizes people, not patterns | `GroupKFold` |
| Reporting `GridSearchCV.best_score_` as the final performance | Best-of-many is optimistic (14 points in Section 12) | Nested CV or a locked test set |
| Reporting only the mean | Hides how luck-dependent it is | Always report **mean ± std** |
| Comparing models on different splits | Part of the difference is luck | Same `cv` object, same `random_state` |

---

## 16. The same model, measured seven ways (`scripting.py`, Step 8)

| Method | Accuracy of Chapter 5's tuned tree |
|---|---|
| Holdout, `random_state=42` | 0.732 |
| Holdout, average of 50 random splits | 0.720 |
| K-Fold (5) | 0.721 |
| Stratified K-Fold (5) | 0.725 |
| LOOCV | 0.725 |
| Repeated Stratified K-Fold (5 × 10) | **0.715** ± 0.027 |
| ShuffleSplit (10) | 0.707 |

They all land between 0.707 and 0.732. The honest statement to the business is **"about 71–72%, give or take ~3 points"**, not "73.2%" from one split.

---

## The full mental model, tied together

1. **Metrics (Chapter 4) are the ruler; evaluation techniques are how carefully you use it.** The goal is a trustworthy estimate of performance on unseen data.
2. **Three kinds of data:** train (learn), validation (choose), test (one final honest score). Anything that influences your choices is no longer a clean test.
3. **Holdout** is simple but luck-dependent: 0.664 to 0.764 on the same model, depending only on the shuffle.
4. **K-Fold** rotates the test fold so every row is tested once, and gives a **mean ± spread**. **Stratified K-Fold** keeps the class ratio in every fold; it's the default for classification. Always shuffle non-time data.
5. **k = 5 or 10** is the standard. **LOOCV** (k = n) is for tiny datasets. On Play Tennis it exposed that the "perfect" tree (100% on training) is worse than always saying Yes (57% vs 64%).
6. **Repeated K-Fold / ShuffleSplit** average over many shuffles, for the most stable estimate.
7. **Non-independent rows need special splitters:** `TimeSeriesSplit` (past → future) and `GroupKFold` (a person stays on one side).
8. **Tuning makes CV scores optimistic.** **Nested CV** gives an honest number for tuned models (0.770 claimed vs 0.670 nested vs 0.626 true, on 100 rows).
9. **Everyday workflow:** lock a test set → Stratified K-Fold + GridSearchCV on the rest → test once → retrain on all data → report the CV mean ± std.
