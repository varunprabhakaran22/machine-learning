# Chapter 6 — Ensemble Techniques

*(Sections 1–10 explain each technique, with small hand-worked examples on the Play Tennis dataset from Chapter 5. Every technique is then run for real in `scripting.py` on `assets/claim.csv`, the same ATTORNEY problem and the same 220 test claims as Chapters 3 and 5, so all the numbers are directly comparable.)*

## 0. Where this sits in the big picture

So far every chapter trained **one** model: one line (Ch2), one S-curve (Ch3), one tree (Ch5). An **ensemble** trains **many** models and combines their answers into one prediction.

> **Ensemble = a team of models that votes.** A team of decent, *different* models usually beats even the best individual member.

**The real-world story:** a hospital doesn't let one doctor decide on a risky surgery; it asks a panel. Each doctor makes different mistakes, and when their opinions are combined, the individual mistakes tend to cancel out. The same idea shows up in a quiz show's "ask the audience" lifeline, which is famously right far more often than any single audience member.

**Why this chapter comes straight after Decision Trees:** almost every serious ensemble (Random Forest, AdaBoost, Gradient Boosting, XGBoost) is a team of **Decision Trees**. Chapter 5 showed a single tree's big weaknesses: it **overfits** (high variance) and it's **unstable**. Ensembles are the fix, and they're the reason trees dominate real-world tabular-data ML today.

**The four families you'll learn:**

| Family | How the team is built | Main problem it fixes (Ch5, Section 11) | Famous examples |
|---|---|---|---|
| **Voting** | Different algorithms, trained independently, vote | A bit of both | `VotingClassifier` |
| **Bagging** | Same algorithm, many copies, each on a random sample of the data, trained **in parallel** | **Variance** (overfitting) | Bagged trees, **Random Forest** |
| **Boosting** | Same algorithm, trained **one after another**, each fixing the previous one's mistakes | **Bias** (underfitting) | **AdaBoost**, **Gradient Boosting**, **XGBoost**, LightGBM, CatBoost |
| **Stacking** | Different algorithms, and a final model *learns* how to combine them | Squeezing out the last bit of performance | `StackingClassifier` |

---

## 1. Why a team beats an individual — the math, and the catch

### The math: majority voting

Say you have 3 models, each **70% accurate**, and their mistakes are **independent** (they fail on different rows). The majority vote is right whenever **at least 2 of the 3** are right:

$$
P(\text{majority right}) = \underbrace{0.7^3}_{\text{all 3 right}} + \underbrace{3 \times 0.7^2 \times 0.3}_{\text{exactly 2 right}} = 0.343 + 0.441 = \mathbf{0.784}
$$

| Number of independent 70%-accurate models | Majority-vote accuracy |
|---|---|
| 1 | 70.0% |
| 3 | **78.4%** |
| 5 | **83.7%** |
| 25 | **98.3%** |

That's the "wisdom of the crowd": individually mediocre, collectively excellent.

### The catch: the models must be *different*

That table assumes the models make **independent** mistakes. If all 3 models are copies of each other, they're wrong on the *same* rows, and voting changes nothing: 3 × 70% still gives 70%.

**Real example (`scripting.py`, Section 5.1):** on the claim data, Logistic Regression and the tuned Decision Tree give the **same answer on 82.3% of test claims**. They're highly correlated, so when one is wrong the other usually is too. That's why the gains from ensembling on this dataset are real but modest.

> **The #1 rule of ensembles: diversity matters more than the number of members.** Every technique below is, at heart, a different trick for making team members disagree usefully.

| Technique | Its trick for creating diversity |
|---|---|
| Voting / Stacking | Use completely **different algorithms** |
| Bagging | Train each copy on a **different random sample of rows** |
| Random Forest | Different rows **and** a different random subset of **features** at every split |
| Boosting | Each new model is **forced to focus on the rows the team currently gets wrong** |

---

## 2. Quick vocabulary

- **Base learner / base model / estimator:** one member of the team (e.g. one tree).
- **Weak learner:** a model only slightly better than random guessing (e.g. a one-question tree, a "stump"). Boosting is built from these.
- **Strong learner:** the accurate combined model.
- **Homogeneous ensemble:** all members use the same algorithm (Bagging, Random Forest, Boosting).
- **Heterogeneous ensemble:** members use different algorithms (Voting, Stacking).
- **Parallel vs sequential:** bagging members can be trained at the same time, independently. Boosting members must be trained one after another, because each depends on the last.

---

## 3. Voting — different algorithms, one vote

### How it works

1. Train several **different** algorithms on the same training data, e.g. Logistic Regression + Decision Tree + something else.
2. For a new row, collect every model's answer.
3. Combine them:
   - **Hard voting:** each model casts one Yes/No vote, and the **majority wins**.
   - **Soft voting:** **average the predicted probabilities**, then apply the usual 0.5 threshold (Chapter 3). Confident models get more say.

### Hard vs soft: a tiny example where they disagree

*(Illustrative probabilities.)* Three models predict P(Attorney) for one claim:

| Model | P(Attorney) | Hard vote |
|---|---|---|
| Model A | **0.90** (very sure: Yes) | Yes |
| Model B | 0.40 (unsure, leaning No) | No |
| Model C | 0.40 (unsure, leaning No) | No |
| **Result** | **Soft:** average = (0.90 + 0.40 + 0.40) / 3 = **0.567 → Yes** | **Hard: 2 No vs 1 Yes → No** |

Hard voting ignored *how sure* each model was. Soft voting let the very confident model outweigh two lukewarm ones. **Soft voting is usually better, *if* the models' probabilities are trustworthy.**

### Real results (`scripting.py`, Section 5.1)

| Team | Hard vote test acc | Soft vote test acc | Soft vote AUC |
|---|---|---|---|
| LogReg + Tree + **Naive Bayes** (a weak member, 0.555 alone) | 0.673 | **0.595** ❌ | 0.740 |
| LogReg + Tree + **Random Forest** (a strong member) | 0.732 | 0.709 | **0.772** ✅ |

**The lesson:** with a weak, badly calibrated member (Naive Bayes), soft voting *collapsed*, because Naive Bayes' probabilities dragged every average the wrong way. Swap in a strong member and the team's AUC (0.772) beats every single model. **A team is only as good as its members; "more models" is not automatically "better".**

**When to use Voting:** you already have a few good, *different* models and want a quick, low-effort boost. If one member is clearly weaker, drop it, or use Stacking (Section 10), which learns to down-weight it.

---

## 4. Bagging (Bootstrap AGGregatING) — same algorithm, different data

### The idea

Bagging trains many copies of the **same** algorithm, but gives each copy a **different random version of the training data**, then averages their predictions (majority vote for classification, mean for regression).

The "different random version" is a **bootstrap sample**: draw rows **at random, with replacement**, until you have as many rows as the original. "With replacement" means the same row can be picked more than once, and some rows are never picked at all.

### Worked example — one bootstrap sample of Play Tennis (14 days)

Drawing 14 days at random, with replacement (a real seeded draw):

```
Picked: 1, 1, 4, 4, 5, 9, 9, 10, 11, 12, 13, 13, 13, 14
```

- Day 1, 4 and 9 appear twice; day 13 appears **three** times.
- Only **9 unique days** made it in.
- Days **2, 3, 6, 7, 8 were never picked.** These are this tree's **Out-Of-Bag (OOB)** rows.

Tree #1 trains on this sample. Tree #2 gets a *different* random draw, and so on for 100 trees. Every tree sees a slightly different "version of history", so every tree comes out slightly different, and that's the diversity.

### The 63% / 37% rule

The chance a particular row is *never* picked in *n* draws is $(1 - \tfrac{1}{n})^n$, which approaches $e^{-1} \approx$ **36.8%** as *n* grows (35.4% for 14 rows, 36.8% for our 876 claim rows). So:
- each tree trains on **≈ 63%** of the unique rows (a 876-row bootstrap sample in `scripting.py` contained 62.2% unique rows);
- the other **≈ 37% are out-of-bag** for that tree.

**The OOB score, a free validation set:** score each row using *only the trees that never saw it*, and you get an honest accuracy estimate **without touching the test set** and without cross-validation. It's a built-in bonus of bagging (`oob_score=True`).

### Why bagging reduces variance

Recall Chapter 5, Section 11: an unlimited tree has **high variance**, meaning retrain it on slightly different data and you get a very different tree. That's bad for one tree, but it's exactly what makes bagging work. Each tree's quirks are *random*, and **random errors cancel out when you average them**, while the real pattern (which every tree picks up) survives.

### Real results (`scripting.py`, Section 5.2): 100 bagged *unlimited* trees

| | Train acc | Test acc | Test AUC |
|---|---|---|---|
| 1 unlimited tree (Ch5 default) | 0.995 | 0.655 | 0.654 |
| **100 bagged unlimited trees** | 0.995 | 0.659 | **0.724** (+0.07) |
| OOB accuracy (no test set used) | | **0.686** | |

Accuracy barely moved, but **AUC jumped by 0.07**. Averaging 100 jumpy trees gives much *smoother, better-ranked* probabilities. Honest caveat: each tree still memorizes (train 0.995). Bagging **reduces** variance, it doesn't eliminate it.

---

## 5. Random Forest — bagging + random features

### The one extra trick

Bagging has a weakness: if one feature is very strong, **every tree asks about it first**. On claim.csv every tree would open with "LOSS ≤ …?". The trees end up similar (correlated), and averaging similar trees doesn't cancel much.

**Random Forest fixes this:** at **every split**, each tree may only choose from a **random subset of the features**. The default for classification is √(number of features).

**On Play Tennis:** 4 features, so √4 = **2 random features per split**. At the root, one tree might only be allowed to pick from {Temperature, Windy}. It *can't* use Outlook, so it has to find the best question among those two. Another tree gets {Outlook, Humidity}. The trees are forced to explore different questions, which makes them **much more diverse**.

**On claim.csv:** 5 features, so √5 ≈ 2 per split.

```
Random Forest = many trees
              × each trained on a bootstrap sample of ROWS       (bagging)
              × each split only sees a random subset of FEATURES  (the "random" in Random Forest)
              → majority vote / average
```

### Key hyperparameters

| Hyperparameter | Meaning | Typical tuning |
|---|---|---|
| `n_estimators` | Number of trees | 100–500. **More is never worse, just slower** (see below) |
| `max_features` | Features considered per split | `"sqrt"` (default), `"log2"`, `None` (= all features, i.e. plain bagging) |
| `max_depth`, `min_samples_leaf` | Shape of each tree (Ch5, Section 12) | Same as a single tree; matters a lot on noisy data |
| `oob_score` | Compute the free OOB estimate | `True` |

### Real results (`scripting.py`, Section 5.3)

**More trees → plateau, not overfitting** (unlimited trees, test AUC):

| Trees | 1 | 5 | 10 | 50 | 100 | 300 |
|---|---|---|---|---|---|---|
| Test AUC | 0.621 | 0.691 | 0.700 | 0.714 | 0.714 | 0.713 |

**Myth-buster:** "Random Forest doesn't overfit" is only half true. *Adding trees* doesn't cause overfitting, but *deep trees* still can. The default forest (unlimited trees) scored train **0.995** vs test **0.645**. The fix is the same as Chapter 5: tune the tree shape with GridSearchCV.

| | Train acc | Test acc | Test AUC |
|---|---|---|---|
| Random Forest, default | 0.995 | 0.645 | 0.714 |
| **Random Forest, GridSearchCV-tuned** (`max_depth=3`, `min_samples_leaf=10`, `max_features="sqrt"`; CV AUC 0.756) | 0.734 | **0.732** | **0.768** |

GridSearchCV picked `max_features="sqrt"` over `None`, so the feature randomness really did help. And feature importance spread out: LOSS dropped from 0.913 (Ch5's single tree) to **0.783**, with CLMAGE 0.109, CLMINSUR 0.062, CLMSEX 0.044. Because some splits couldn't see LOSS, the forest learned more from the other columns.

**When to use Random Forest:** it's the best "first serious model" for almost any tabular problem. It works well with little tuning, is hard to break, handles mixed features, gives feature importances, and needs no scaling.

---

## 6. Boosting — the general idea

Bagging builds its team **in parallel**: independent members, then average. Boosting builds it **in sequence**:

```
Model 1 → look at what it got wrong → Model 2 focuses on those → look at what the team still gets wrong
        → Model 3 focuses on those → ... → final prediction = weighted combination of all models
```

**Story:** a student prepares for an exam by taking a practice test, then spending the next study session **only on the questions they got wrong**, then taking another test and focusing on the *new* wrong answers, and so on. Each session is small, but they add up.

- **Uses weak learners on purpose:** tiny trees (often depth 1 to 3). Each one only has to be a little better than guessing.
- **Targets bias:** it starts simple and keeps adding correction after correction, so the team can capture patterns no single small tree could.
- **Can overfit if you let it run too long:** after enough rounds there's nothing left to correct but noise (real proof in Section 8).

There are two main flavours: **AdaBoost** (re-weight the *rows*) and **Gradient Boosting** (fit the *residual errors*). XGBoost is an engineered version of Gradient Boosting.

---

## 7. AdaBoost (Adaptive Boosting) — re-weight the mistakes

### The algorithm

1. Give every training row an **equal weight** (1/n).
2. Train a **stump**, a tree with **one question** (depth 1), picking the question with the lowest *weighted* error ε.
3. Give the stump a **say** (vote weight) based on how good it was:
$$
\alpha = \tfrac{1}{2}\ln\!\left(\frac{1-\varepsilon}{\varepsilon}\right)
$$
   Low error means a big say. ε = 0.5 (a coin flip) means α = 0, no say at all.
4. **Increase the weights of rows it got wrong** (× e^α) and **decrease the weights of rows it got right** (× e^−α), then rescale so the weights sum to 1.
5. Repeat from step 2 with the new weights. The next stump is forced to care about the rows the last one missed.
6. **Final prediction:** a weighted vote. Add α for every stump that says Yes, subtract α for every stump that says No; a positive total means Yes.

### Worked example — 3 rounds on Play Tennis

**Round 1** (all 14 days weighted 1/14 = 0.0714). The best stump is **"Outlook = Sunny? → No, otherwise → Yes"**. (Tied with "Humidity = High? → No"; both make 4 mistakes.)
- Wrong on days **6, 9, 11, 14** → ε = 4/14 = **0.2857**
- Say: α₁ = ½ ln(0.7143 / 0.2857) = ½ ln(2.5) = **0.4581**
- Re-weight: wrong days get 0.0714 × e^0.4581 = 0.1129, right days get 0.0714 × e^−0.4581 = 0.0452. The total is 0.9035; after dividing by it:

| | Before | After round 1 |
|---|---|---|
| Each of the 4 wrong days | 0.0714 | **0.1250** |
| Each of the 10 right days | 0.0714 | **0.0500** |
| Share of total weight held by the 4 wrong days | 29% | **50%** |

After one round, the 4 misclassified days hold **half the total weight**. The next stump *has* to take them seriously.

**Round 2** (with the new weights): the best stump is **"Humidity = High? → No, otherwise → Yes"**, weighted error ε = 0.2750, **α₂ = 0.4847**. It's wrong on days 3, 4, 6, 12, which get up-weighted.

**Round 3:** the best stump is **"Outlook = Rainy? → No, otherwise → Yes"**, weighted error ε = 0.2633, **α₃ = 0.5144**. On its own, unweighted, this stump is only 8/14 = **57% accurate**. It's weak, but it's exactly the correction the team needed.

**Tracing Day 10** (*Rainy, Mild, Normal, False → actually Yes*):

| Stump | Says | Contribution |
|---|---|---|
| 1: Outlook = Sunny? No, so → Yes | Yes | +0.4581 |
| 2: Humidity = High? No, so → Yes | Yes | +0.4847 |
| 3: Outlook = Rainy? Yes, so → No | No | −0.5144 |
| **Total** | | **+0.4284 → Yes ✓** |

**The payoff:** three stumps that are **71%, 71% and 57%** accurate on their own combine to get **12 of 14 days right (85.7%)**. That's weak learners becoming a strong learner.

### Real result (`scripting.py`, Section 5.4)

100 stumps with sklearn's `AdaBoostClassifier` scored train **0.729**, test **0.732**, AUC **0.773**. That beats the tuned 4-level tree's AUC (0.759), with almost no train/test gap, and each member asks just **one** question.

---

## 8. Gradient Boosting — fit the leftover error

### The idea

Instead of re-weighting rows, each new tree is trained to predict **the residuals**, meaning the error that's still left: *actual − current prediction*. Its output is then **added** to the running prediction, scaled down by a **learning rate** (η).

$$
\text{new prediction} = \text{old prediction} + \eta \times (\text{new tree's prediction of the residual})
$$

It's the same idea as Chapter 2's Gradient Descent: take small steps in the direction that reduces the error. That's where the name comes from. Here, each "step" is a whole tree.

### Worked example — predicting house prices (a tiny made-up dataset, regression makes it easiest to see)

| Size (sq ft) | 600 | 800 | 1000 | 1200 | 1400 |
|---|---|---|---|---|---|
| **Actual price (₹ lakh)** | 150 | 200 | 210 | 280 | 300 |

Learning rate η = 0.5, each tree is a stump.

**Round 0: start with the average.** Predict 228 for every house. MSE = **3016.0**.

**Round 1.** Residuals = actual − 228 = **[−78, −28, −18, +52, +72]**. A stump fit to *these residuals* splits at size ≤ 1100: the left leaf predicts the average residual −41.33, the right leaf +62.0. Update: 228 + 0.5 × (−41.33) = **207.3** for small houses; 228 + 0.5 × 62 = **259.0** for big ones. MSE = **1094.0**.

**Round 2.** New residuals **[−57.3, −7.3, +2.7, +21.0, +41.0]**. The stump now splits at ≤ 700 (the cheapest house is still the worst-fit). Predictions become [178.7, 214.5, 214.5, 266.2, 266.2]. MSE = **477.7**.

**Round 3.** Residuals [−28.7, −14.5, −4.5, +13.8, +33.8]. Split at ≤ 1100 again. Predictions [170.7, 206.6, 206.6, 278.1, 278.1]. MSE = **193.7**.

| Round | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| MSE | 3016.0 | 1094.0 | 477.7 | 193.7 |

Every round, the residuals shrink. Each stump is a crude correction, but the corrections stack up.

**For classification** (our ATTORNEY problem), it's the same mechanism, but the running prediction is in **log-odds** (the `z` from Chapter 3, Section 7), and the "residuals" are *actual (0/1) − predicted probability*. The final log-odds go through the Sigmoid to become a probability, exactly like Logistic Regression.

### The learning rate tradeoff

- A **big η** (e.g. 1.0) takes big steps and learns fast, but overshoots and overfits.
- A **small η** (e.g. 0.01 to 0.1) takes small, careful steps. It needs **more trees**, but usually generalizes better.
- The rule of thumb: **lower learning rate + more trees + small trees = better, just slower.**

### Real proof that boosting *can* overfit (`scripting.py`, Section 5.5)

Gradient Boosting, 300 trees of depth 2, η = 0.05, AUC measured after each number of trees:

| Trees so far | 1 | 10 | 50 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|
| Train AUC | 0.757 | 0.776 | 0.796 | 0.813 | 0.834 | **0.854** ↑ |
| Test AUC | 0.763 | 0.768 | 0.768 | **0.771** | 0.760 | **0.751** ↓ |

Training AUC climbs forever. Test AUC **peaks around 100 trees, then falls**. That's the Chapter 5 overfitting curve again, this time along the "number of trees" axis. **This is the key difference from Random Forest:** in RF, more trees plateaus; in boosting, more trees eventually overfits. So for boosting, `n_estimators` is a real hyperparameter, tuned together with the learning rate (or cut off automatically with *early stopping*).

---

## 9. XGBoost (eXtreme Gradient Boosting) — Gradient Boosting, engineered

XGBoost is the same core algorithm as Section 8, plus engineering that made it the long-time winner of Kaggle competitions on tabular data:

| What it adds | Why it matters |
|---|---|
| **Built-in regularization** (`reg_lambda`, `reg_alpha`, `gamma`) | Penalizes overly complex trees, like Adjusted R² in Ch4: complexity has to earn its keep |
| **Second-order math** (uses the curvature of the loss, not just its slope) | Smarter, faster-converging steps |
| **Handles missing values natively** | Learns which branch missing values should go down; no `dropna()` needed |
| **Speed** (parallel split-finding, clever memory layout) | Much faster on big data |
| **Early stopping** | Stops adding trees once a validation score stops improving, which is the Section 8 fix, automated |
| **Row & column subsampling** (`subsample`, `colsample_bytree`) | Borrows Random Forest's diversity trick |

**Key hyperparameters:** `n_estimators`, `learning_rate`, `max_depth` (usually 2 to 6), `subsample`, `colsample_bytree`, `reg_lambda`.

**Its cousins** (same idea, different engineering): **LightGBM** (Microsoft, very fast on large data) and **CatBoost** (Yandex, great with categorical features). You'll meet them in industry; the concepts transfer directly.

### Real results (`scripting.py`, Section 5.6)

| | Train acc | Test acc | Test AUC |
|---|---|---|---|
| XGBoost, default (100 trees, depth 6) | 0.943 | 0.655 | 0.701 |
| **XGBoost, GridSearchCV-tuned** (`learning_rate=0.01`, `max_depth=2`, `n_estimators=300`; CV AUC 0.757) | 0.737 | **0.741** | **0.773** |

The default overfit badly on this small dataset. GridSearchCV found the textbook boosting recipe, **slow learning rate + tiny trees + more rounds**, and it tops the whole leaderboard.

---

## 10. Stacking — let a model learn how to combine models

### The idea

Voting gives every member an **equal** (or fixed) say. **Stacking trains a final "meta-model" to learn how much to trust each member.**

```
Level 0 (base models):  LogReg ──┐
                        Tree   ──┼──► their predictions become the FEATURES of ─►  Level 1 meta-model  ─► final answer
                        NaiveBayes┘                                               (usually Logistic Regression)
```

**The leakage trap, and how `cv=5` avoids it:** if the base models predicted on rows they were trained on, their predictions would look unrealistically good (they memorized those rows), and the meta-model would learn to over-trust them. So `StackingClassifier(cv=5)` builds the meta-model's training data from **out-of-fold** predictions. Each row's base-model predictions come from models that never saw that row. It's the same k-fold idea as Chapter 5, Section 13.

### Real result (`scripting.py`, Section 5.7)

We used the **same three members that made soft voting collapse** (LogReg + Tree + weak Naive Bayes):

| Same 3 members, combined by… | Test acc | Test AUC |
|---|---|---|
| Soft voting (equal ⅓ say each) | 0.595 | 0.740 |
| **Stacking** (meta-model learns the weights) | **0.682** | **0.770** |

The meta-model learned to give Naive Bayes less say instead of an automatic ⅓. That's exactly what stacking is for.

**When to use Stacking:** squeezing the last bit of performance out of several strong, different models (common in competitions). The costs are that it's slower, more complex, and harder to explain.

---

## 11. When to use what — how to decide

### Match the technique to the problem you see

| What you observe (Chapter 5, Section 11 diagnosis) | Reach for | Why |
|---|---|---|
| Model **overfits**: high train, much lower test (e.g. a deep tree) | **Bagging / Random Forest** | Averaging cancels variance |
| Model **underfits**: low train and low test (e.g. a stump, a shallow tree) | **Boosting** (AdaBoost, Gradient Boosting, XGBoost) | Sequential corrections reduce bias |
| You have a few good but *different* models already | **Voting** (quick) or **Stacking** (stronger) | Combine their different strengths |

### A practical default workflow for tabular data

1. **Baseline:** Logistic/Linear Regression and a single tuned tree (Chapters 2–5). Always keep these on the leaderboard.
2. **Random Forest:** it's robust, hard to break, and good with little tuning. A strong first ensemble.
3. **Gradient boosting (XGBoost / LightGBM), tuned:** usually the most accurate on tabular data, but it *needs* tuning (defaults overfit here).
4. **Voting / Stacking** of the best few, only if the extra accuracy is worth the complexity.
5. **Decide using cross-validation scores**, not a single test split, and weigh accuracy against explainability, speed and maintenance.

### Side-by-side comparison

| | **Voting** | **Bagging** | **Random Forest** | **AdaBoost** | **Gradient Boosting / XGBoost** | **Stacking** |
|---|---|---|---|---|---|---|
| Members | Different algorithms | Same algorithm | Trees | Stumps (usually) | Small trees | Different algorithms |
| Training | Parallel | Parallel | Parallel | Sequential | Sequential | Parallel + meta-model |
| Reduces mostly | Both | **Variance** | **Variance** | **Bias** | **Bias** (+ variance with regularization) | Both |
| Overfits by adding more members? | – | No (plateaus) | No (plateaus) | Can | **Yes, tune `n_estimators`** | – |
| Tuning effort | Low | Low | **Low–medium** | Medium | **High** | High |
| Speed | Sum of members | Fast (parallel) | Fast (parallel) | Medium | Medium (XGBoost: fast) | Slowest |
| Explainability | Low | Low | Feature importance | Low | Feature importance / SHAP | Lowest |
| Sensitive to noise/outliers in labels | – | Robust | Robust | **Sensitive** (keeps up-weighting hard/mislabelled rows) | Medium | – |

### When *not* to use an ensemble

- **You must explain individual decisions** in plain rules (to a regulator, a doctor, a customer). A single tuned tree or Logistic Regression may be the right answer even if it's slightly less accurate.
- **The gain is tiny.** Our best ensemble beat the best single tree by about 0.01 AUC (next section). Is that worth 300 trees instead of 9 rules?
- **Latency or memory is tight.** 300 trees are slower to predict with than one formula.

---

## 12. The full leaderboard — and what it honestly tells us

All models, same 220 test claims (from `scripting.py`, Step 8), sorted by test AUC:

| Model | Train acc | Test acc | Test AUC |
|---|---|---|---|
| **XGBoost, tuned** | 0.737 | **0.741** | **0.773** |
| **AdaBoost (100 stumps)** | 0.729 | 0.732 | **0.773** |
| Voting soft (LogReg + Tree + RF) | 0.734 | 0.709 | 0.772 |
| Stacking (LogReg + Tree + NB → LogReg) | 0.735 | 0.682 | 0.770 |
| Random Forest, tuned | 0.734 | 0.732 | 0.768 |
| *Decision Tree, tuned (Ch5): best single model* | 0.724 | 0.732 | 0.759 |
| *Logistic Regression (Ch3)* | 0.715 | 0.673 | 0.745 |
| Gradient Boosting, default | 0.796 | 0.691 | 0.743 |
| Bagging (100 unlimited trees) | 0.995 | 0.659 | 0.724 |
| Random Forest, default | 0.995 | 0.645 | 0.714 |
| XGBoost, default | 0.943 | 0.655 | 0.701 |
| *Decision Tree, default (Ch5)* | 0.995 | 0.655 | 0.654 |

**What it tells us:**
1. **Ensembles win, but modestly here:** 0.759 → 0.773 AUC. This dataset has only 5 columns, and LOSS carries most of the signal, so we're close to the **irreducible-error ceiling** (Chapter 5, Section 11). Ensembles shine brightest on wide datasets with many weak, interacting signals.
2. **Tuning beats the algorithm's name.** Default XGBoost (0.701) lost to a *single tuned tree* (0.759). Untuned ensembles cluster at the bottom, all overfit (train ≈ 0.94 to 0.995).
3. **The best models all have a near-zero train/test gap.** That's generalization (Chapter 5, Section 11) again.
4. **Differences of ~0.01 on 220 rows are within noise.** The 5-fold CV AUCs (RF 0.756, XGB 0.757) are the more reliable comparison, and they're nearly tied.

---

## The full mental model, tied together

1. **Ensemble = a team of models whose combined answer beats the individuals**, *if* the members are **diverse** (make different mistakes). Three independent 70% models → 78.4%; three copies → still 70%.
2. **Voting:** different algorithms vote. *Hard* = majority of labels; *soft* = average of probabilities. A weak member can sink a soft vote.
3. **Bagging:** the same algorithm on bootstrap samples (≈63% unique rows each, ≈37% out-of-bag for a free OOB score), averaged. It **reduces variance**; ideal for overfit models like deep trees.
4. **Random Forest:** bagging + a random subset of features at every split, which makes the trees more diverse. More trees plateau, never hurt; deep trees still need tuning.
5. **Boosting:** sequential; each model fixes the team's remaining mistakes. It **reduces bias**, and it *can* overfit with too many rounds.
   - **AdaBoost:** re-weight misclassified rows; stumps vote with a say α = ½ ln((1−ε)/ε).
   - **Gradient Boosting:** each tree fits the residuals, added with a learning rate.
   - **XGBoost:** gradient boosting + regularization, speed, missing-value handling, early stopping.
6. **Stacking:** a meta-model learns how to weight the base models, using out-of-fold predictions to avoid leakage.
7. **Pick by diagnosis:** overfitting → bagging / Random Forest; underfitting → boosting. Always tune (GridSearchCV), compare with cross-validation against simple baselines, and weigh the accuracy gain against lost explainability.
