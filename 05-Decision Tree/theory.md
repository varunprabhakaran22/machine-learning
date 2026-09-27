# Chapter 5 — Decision Tree

*(Sections 1–10 are theory + math on the Play Tennis dataset. Sections 11–13 cover Bias-Variance, hyperparameter tuning and GridSearchCV, with real numbers from `scripting.py`, the runnable model on `assets/claim.csv`.)*

**Notation note:** no `m`/`b` this time — a Decision Tree doesn't learn a formula at all. The only math is two "messiness" scores (Entropy and Gini) and some fractions. All logs are base 2 (`log₂`), so Entropy comes out between 0 and 1 for a yes/no target.

## 0. Where this sits in the big picture

Recap from Chapter 1: a Decision Tree is —
- **Supervised** (Chapter 1, Section 1) — trained on labeled data (features + a known answer).
- **Classification *and* Regression** — the same algorithm handles both. This chapter's worked example is classification (predict Yes/No), and Section 9 covers the regression version.
- **Non-Parametric** (Chapter 1, Section 5) — this is the first **non-parametric** algorithm in the track. Linear and Logistic Regression learned a fixed handful of numbers (`m`'s and `b`) no matter how much data you had. A Decision Tree has no fixed formula. It learns a *structure*, a set of if/else questions, and that structure can grow as big as the data allows. That freedom is its biggest strength (it fits curved, messy patterns) and its biggest danger (it can memorize the training data, Section 8).

**The one-line idea:** a Decision Tree is a flowchart of yes/no questions, learned from data. It's the same thing you'd do by hand when deciding whether to play tennis:

> "Is it overcast? → Play. Is it sunny? → Then check the humidity. Is it rainy? → Then check the wind."

The algorithm's only real job is deciding **which question to ask first, which to ask next, and when to stop asking.** Everything else in this chapter (Entropy, Information Gain, Gini, ID3, C4.5, CART) is some version of that job.

---

## 1. The dataset — "Should we play tennis today?"

The classic dataset for this topic (from Ross Quinlan, the inventor of ID3). There are 14 days of history, 4 weather features, and 1 target:

| Day | Outlook | Temperature | Humidity | Windy | PlayTennis |
|---|---|---|---|---|---|
| 1 | Sunny | Hot | High | False | No |
| 2 | Sunny | Hot | High | True | No |
| 3 | Overcast | Hot | High | False | Yes |
| 4 | Rainy | Mild | High | False | Yes |
| 5 | Rainy | Cool | Normal | False | Yes |
| 6 | Rainy | Cool | Normal | True | No |
| 7 | Overcast | Cool | Normal | True | Yes |
| 8 | Sunny | Mild | High | False | No |
| 9 | Sunny | Cool | Normal | False | Yes |
| **10** | **Rainy** | **Mild** | **Normal** | **False** | **Yes** |
| 11 | Sunny | Mild | Normal | True | Yes |
| 12 | Overcast | Mild | High | True | Yes |
| 13 | Overcast | Hot | Normal | False | Yes |
| 14 | Rainy | Mild | High | True | No |

**Totals: 9 Yes, 5 No.**

Things to notice before any math:
- **Every feature is categorical** (Sunny/Overcast/Rainy, Hot/Mild/Cool, and so on). Linear/Logistic Regression would force you to convert these into numbers first. A Decision Tree can work on categories directly, which is part of why trees are so popular for business data.
- **The target is categorical** (Yes/No), so this is **classification** (Chapter 1's rule).
- Day 10 (highlighted in the course slide) is the example we'll trace through the finished tree in Section 5, to show how a prediction actually gets made.

---

## 2. The core question: which feature do we split on first?

Imagine you had to split these 14 days into groups using **one** feature. Here's what each feature would give you:

| Split on... | Groups it creates |
|---|---|
| **Outlook** | Sunny: 2Y/3N · **Overcast: 4Y/0N** · Rainy: 3Y/2N |
| **Temperature** | Hot: 2Y/2N · Mild: 4Y/2N · Cool: 3Y/1N |
| **Humidity** | High: 3Y/4N · Normal: 6Y/1N |
| **Windy** | False: 6Y/2N · True: 3Y/3N |

Just by eye: **Outlook looks the most useful**, because one of its groups (Overcast) is *perfectly pure*, all Yes. Once you know it's overcast you're done, no more questions needed. **Temperature looks the least useful**, since Hot is a 2-vs-2 coin flip and the other groups are still mixed.

That "by eye" judgment is exactly what the algorithm formalizes:

> **A good split creates groups that are as *pure* as possible, meaning each group is mostly one answer.**

We need a number that measures "how mixed/messy is this group?" There are two standard choices: **Entropy** (Section 3) and **Gini Impurity** (Section 4). Which one you use, and how you use it, is basically the difference between ID3, C4.5 and CART (Section 6).

---

## 3. Entropy and Information Gain (the ID3 way)

### Entropy — "how surprised would I be?"

$$
\text{Entropy} = -\sum_{\text{each class}} p \cdot \log_2(p)
$$

For a Yes/No target this is just:

$$
\text{Entropy} = -\big[\, p_{yes}\log_2(p_{yes}) + p_{no}\log_2(p_{no}) \,\big]
$$

**Plain-English meaning:** if you picked a random day from this group, how uncertain would you be about the answer?

| Group | Entropy | Reading it |
|---|---|---|
| 4 Yes / 0 No | **0.000** | Perfectly pure. Zero uncertainty, you already know the answer |
| 6 Yes / 2 No | 0.811 | Mostly one answer, some doubt |
| 3 Yes / 2 No | 0.971 | Pretty mixed |
| 3 Yes / 3 No | **1.000** | Worst possible, a 50/50 coin flip |

So **0 = pure (good)** and **1 = maximally mixed (bad)**. This is the same "surprise" idea from probability: a guaranteed outcome carries no surprise.

### Worked example — Entropy of the whole dataset (before any split)

9 Yes, 5 No, out of 14:

$$
\text{Entropy(root)} = -\left[\tfrac{9}{14}\log_2\tfrac{9}{14} + \tfrac{5}{14}\log_2\tfrac{5}{14}\right] = 0.4098 + 0.5305 = \mathbf{0.9403}
$$

That's pretty high, close to 1. With no questions asked yet, we're quite unsure.

### Information Gain — "how much did this question reduce the mess?"

$$
\text{Information Gain} = \text{Entropy(before)} - \text{Weighted Average Entropy(after)}
$$

"Weighted" means bigger groups count more. A group with 5 days matters more than a group with 1 day.

### Worked example — Information Gain of splitting on Outlook

| Outlook | Days | Yes/No | Entropy | Weight |
|---|---|---|---|---|
| Sunny | 5 | 2Y / 3N | 0.9710 | 5/14 |
| Overcast | 4 | 4Y / 0N | 0.0000 | 4/14 |
| Rainy | 5 | 3Y / 2N | 0.9710 | 5/14 |

$$
\text{Weighted Entropy} = \tfrac{5}{14}(0.9710) + \tfrac{4}{14}(0.0000) + \tfrac{5}{14}(0.9710) = 0.6935
$$

$$
\text{IG(Outlook)} = 0.9403 - 0.6935 = \mathbf{0.2467}
$$

### Doing the same for every feature

| Feature | Weighted Entropy after split | **Information Gain** |
|---|---|---|
| **Outlook** | 0.6935 | **0.2467** ← winner |
| Humidity | 0.7885 | 0.1518 |
| Windy | 0.8922 | 0.0481 |
| Temperature | 0.9111 | 0.0292 |

**Outlook wins, so it becomes the root (first question) of the tree.** This matches the "by eye" guess from Section 2, and Temperature is last, also as guessed. The math just makes the gut feeling precise and repeatable.

---

## 4. Gini Impurity (the CART way)

Gini measures the same "how mixed is this group?" idea with a simpler formula (no logs):

$$
\text{Gini} = 1 - \sum_{\text{each class}} p^2 = 1 - (p_{yes}^2 + p_{no}^2)
$$

**Plain-English meaning:** if you picked a random day from the group and then guessed its label at random *using the group's own Yes/No proportions*, how often would you guess wrong?

| Group | Gini | Entropy (for comparison) |
|---|---|---|
| 4 Yes / 0 No | **0.000** | 0.000 |
| 6 Yes / 2 No | 0.375 | 0.811 |
| 3 Yes / 2 No | 0.480 | 0.971 |
| 3 Yes / 3 No | **0.500** | 1.000 |

For two classes the range is 0 (pure) to 0.5 (a 50/50 mix). **Same direction as Entropy, different scale.** Both say "lower is purer."

### Worked example — Gini of the root, and the Gini of splitting on Outlook

$$
\text{Gini(root)} = 1 - \left[\left(\tfrac{9}{14}\right)^2 + \left(\tfrac{5}{14}\right)^2\right] = 1 - [0.4133 + 0.1276] = \mathbf{0.4592}
$$

Outlook split: Sunny (Gini 0.48), Overcast (0.00), Rainy (0.48), weighted the same way as before:

$$
\tfrac{5}{14}(0.48) + \tfrac{4}{14}(0) + \tfrac{5}{14}(0.48) = 0.3429 \quad\Rightarrow\quad \text{Gini Gain} = 0.4592 - 0.3429 = \mathbf{0.1163}
$$

| Feature (multi-way split) | Weighted Gini after | Gini Gain |
|---|---|---|
| **Outlook** | 0.3429 | **0.1163** ← winner |
| Humidity | 0.3673 | 0.0918 |
| Windy | 0.4286 | 0.0306 |
| Temperature | 0.4405 | 0.0187 |

**Same ranking as Entropy.** This is the normal case in practice: Gini and Entropy agree on the best split the vast majority of the time. The choice between them is rarely what makes or breaks a model (Section 7 comes back to this).

---

## 5. Building the full tree (ID3, step by step)

A tree is built by **repeating the Section 3 recipe inside each branch**, a process called *recursive splitting*:

1. Compute the Information Gain of every remaining feature.
2. Split on the best one.
3. Inside each new branch, go back to step 1 using **only the days in that branch**.
4. Stop when a branch is pure (all Yes or all No), or there are no features left.

### Branch: Outlook = Overcast (4Y / 0N)

Entropy = 0, so it's already pure. **This becomes a leaf: predict Yes.** No more questions.

### Branch: Outlook = Sunny (days 1, 2, 8, 9, 11 → 2Y / 3N, Entropy 0.9710)

| Feature (within Sunny days only) | Groups | Information Gain |
|---|---|---|
| **Humidity** | High: 0Y/3N · Normal: 2Y/0N | **0.9710** ← perfect, both groups pure |
| Temperature | Hot: 0Y/2N · Mild: 1Y/1N · Cool: 1Y/0N | 0.5710 |
| Windy | False: 1Y/2N · True: 1Y/1N | 0.0200 |

Humidity gets an Information Gain equal to the *entire* entropy of the branch (0.9710 − 0 = 0.9710), so it removes all uncertainty. **Split on Humidity.** Both children are pure leaves.

### Branch: Outlook = Rainy (days 4, 5, 6, 10, 14 → 3Y / 2N, Entropy 0.9710)

| Feature (within Rainy days only) | Groups | Information Gain |
|---|---|---|
| **Windy** | False: 3Y/0N · True: 0Y/2N | **0.9710** ← perfect |
| Temperature | Mild: 2Y/1N · Cool: 1Y/1N | 0.0200 |
| Humidity | High: 1Y/1N · Normal: 2Y/1N | 0.0200 |

**Split on Windy.** Both children are pure leaves.

### The finished tree

```
                        ┌───────────┐
                        │  Outlook? │   ← root (IG 0.2467)
                        └─────┬─────┘
            ┌─────────────────┼──────────────────┐
          Sunny            Overcast             Rainy
            │                 │                   │
      ┌─────┴─────┐       ┌───┴───┐         ┌─────┴─────┐
      │ Humidity? │       │  YES  │         │  Windy?   │
      └─────┬─────┘       └───────┘         └─────┬─────┘
       ┌────┴────┐        (4/4 days)         ┌────┴────┐
     High      Normal                      False      True
       │          │                          │          │
    ┌──┴──┐    ┌──┴──┐                    ┌──┴──┐    ┌──┴──┐
    │ NO  │    │ YES │                    │ YES │    │ NO  │
    └─────┘    └─────┘                    └─────┘    └─────┘
    (3/3)      (2/2)                      (3/3)      (2/2)
```

**Vocabulary you'll see everywhere:**
- **Root node**: the first question (Outlook).
- **Internal / decision node**: any question in the middle (Humidity, Windy).
- **Branch**: one answer to a question (Sunny, High, True...).
- **Leaf / terminal node**: a final answer, with no more questions (YES / NO).
- **Depth**: the number of questions from the root down to the deepest leaf. This tree has depth 2.

**Notice what's missing: Temperature never appears.** The algorithm decided it wasn't worth asking about. So a Decision Tree does **automatic feature selection** as a side effect: features that never get picked are, for this data, useless.

### Making a prediction — tracing Day 10 (the highlighted row)

Day 10 = *Rainy, Mild, Normal, False*.

1. **Outlook?** → Rainy → go right.
2. **Windy?** → False → go left.
3. Leaf: **YES** ✓ (actual answer: Yes).

It only took 2 questions, and Temperature and Humidity were never even looked at. That's what prediction is for a tree: **walk down from the root, answering each question, until you hit a leaf.** For a new day the model has never seen, e.g. *Sunny, Cool, High, True*: Outlook = Sunny → Humidity = High → **NO**.

### The tree *is* a set of human-readable rules

```
IF Outlook = Overcast                      THEN Play = Yes
IF Outlook = Sunny  AND Humidity = High    THEN Play = No
IF Outlook = Sunny  AND Humidity = Normal  THEN Play = Yes
IF Outlook = Rainy  AND Windy = False      THEN Play = Yes
IF Outlook = Rainy  AND Windy = True       THEN Play = No
```

This is the single biggest selling point of Decision Trees: **you can hand these rules to a business person and they can read them.** You can't do that with Logistic Regression's coefficients, let alone a neural network.

---

## 6. The algorithms: ID3, C4.5, C5.0, CART, CHAID — what are they?

These are all **the same basic idea** (recursively split on the "best" question). They differ in four practical choices:

1. **How is "best split" measured?** Information Gain, Gain Ratio, Gini, or a Chi-square test.
2. **How many branches per split?** Multi-way (one branch per category) or strictly binary (always yes/no).
3. **What data can it handle?** Categorical only, or also numeric features, missing values, and a numeric target (regression).
4. **How does it stop overgrowing?** No pruning, or some form of pruning.

### ID3 (Iterative Dichotomiser 3) — Quinlan, 1986

This is exactly what Section 5 just did by hand.
- **Split criterion:** Information Gain (Entropy).
- **Branches:** multi-way (Outlook → 3 branches).
- **Handles:** categorical features only, classification only. No missing values.
- **Pruning:** none. It grows until branches are pure, so it overfits easily on real data.
- **Its known flaw: it's biased toward features with many distinct values.** Imagine adding a `Day` column (1 to 14) as a feature. Splitting on it puts each day in its own group of one, and every group of one is trivially "pure", so:

$$
\text{IG(Day)} = 0.9403 - 0 = \mathbf{0.9403}
$$

That's the maximum possible, nearly 4× Outlook's 0.2467. ID3 would happily pick `Day` as the root. That tree is 100% "accurate" on training data and completely useless on any new day, because a new day has a Day number the tree has never seen. **This flaw is the reason C4.5 exists.**

**Use today for:** learning. It's the cleanest algorithm to understand and compute by hand. You won't deploy it.

### C4.5 — Quinlan, 1993 (ID3's successor)

This one fixes ID3's weaknesses.
- **Split criterion: Gain Ratio**, which is Information Gain divided by "Split Info" (how many ways, and how evenly, the feature chops the data):

$$
\text{Split Info} = -\sum \tfrac{|group|}{|total|}\log_2\tfrac{|group|}{|total|} \qquad \text{Gain Ratio} = \frac{\text{Information Gain}}{\text{Split Info}}
$$

A feature that shatters the data into many tiny groups gets a big Split Info, which shrinks its score. Worked on our data:

| Feature | Information Gain | Split Info | **Gain Ratio** |
|---|---|---|---|
| **Outlook** (3 groups: 5/4/5) | 0.2467 | 1.5774 | **0.1564** ← still winner |
| Humidity (2 groups: 7/7) | 0.1518 | 1.0000 | 0.1518 |
| Windy (2 groups: 8/6) | 0.0481 | 0.9852 | 0.0488 |
| Temperature (3 groups: 4/6/4) | 0.0292 | 1.5567 | 0.0188 |

Outlook still wins, but look how close Humidity got (0.1564 vs 0.1518, down from 0.2467 vs 0.1518). Outlook was penalized for having 3 branches while Humidity has only 2. That's Gain Ratio doing its job.

The `Day` column: Split Info = log₂(14) = 3.807, so Gain Ratio = 0.9403 / 3.807 = 0.247. **Its score was cut by 74%**, versus 37% for Outlook. Honest caveat: on a dataset this tiny it still scores above Outlook, so the penalty *helps* but doesn't magically make ID-like columns safe. **The real-world rule stays: drop ID columns (customer ID, row number, case number) before training any tree.** This is the same thing Chapter 3's `scripting.ipynb` did with `CASENUM`.

- **Also adds:** numeric features (it finds the best threshold, e.g. `Temperature ≤ 75.5`), missing values, and **post-pruning** (grow the full tree, then cut back branches that don't help).
- **Use today for:** understanding. It's historically the most influential tree algorithm, and its ideas live on everywhere.

### C5.0 — Quinlan's commercial upgrade of C4.5

Same ideas as C4.5, but faster, less memory-hungry, and smaller trees, plus built-in boosting. It's available in R (`C50` package), rarely in Python. **Know the name.** You'll see it mentioned, but you're unlikely to use it directly.

### CART (Classification And Regression Trees) — Breiman et al., 1984

This is **the one you'll actually use.** scikit-learn's `DecisionTreeClassifier` and `DecisionTreeRegressor` are an optimized version of CART. It's also the base tree inside Random Forest, XGBoost and LightGBM.
- **Split criterion:** **Gini** by default for classification (Entropy is an option), and **MSE / variance reduction** for regression (Section 9).
- **Branches: always binary.** Every question is a yes/no question. Instead of "Outlook = Sunny / Overcast / Rainy?" it asks "Outlook = Overcast? yes / no."
- **Handles:** classification *and* regression, numeric features natively. (sklearn needs categorical features encoded as numbers first, e.g. one-hot. That's a sklearn limitation, not a CART one.)
- **Pruning:** cost-complexity pruning (Section 8).

**Worked: CART's first question on our data.** CART tries every possible "feature = value vs everything else" split:

| Binary split | Weighted Gini after | Gini Gain |
|---|---|---|
| **Outlook = Overcast?** (4 vs 10 days) | 0.3571 | **0.1020** ← winner |
| Humidity = High? (7 vs 7) | 0.3673 | 0.0918 |
| Outlook = Sunny? (5 vs 9) | 0.3937 | 0.0655 |
| Windy = True? (6 vs 8) | 0.4286 | 0.0306 |
| Temperature = Hot? (4 vs 10) | 0.4429 | 0.0163 |
| Outlook = Rainy? (5 vs 9) | 0.4571 | 0.0020 |

So CART's root question is **"Is it Overcast?"** Yes → pure leaf (Play = Yes). No → the remaining 10 days are 5Y/5N and need more questions. Continuing the same greedy procedure by hand gives:

```
Outlook = Overcast?
├── yes → YES (4/4)
└── no  → Humidity = High?
          ├── yes → Outlook = Rainy?
          │         ├── yes → Windy = False?  → yes: YES (1) / no: NO (1)
          │         └── no  → NO (3/3)
          └── no  → Windy = False?
                    ├── yes → YES (3/3)
                    └── no  → Outlook = Rainy? → yes: NO (1) / no: YES (1)
```

Tracing Day 10 (*Rainy, Mild, Normal, False*): Overcast? no → High humidity? no → Windy = False? yes → **YES** ✓. That's the same answer as ID3.

**The interesting lesson:** on this dataset CART's tree is *bigger* (depth 4, 7 leaves) than ID3's (depth 2, 5 leaves), yet both classify all 14 days perfectly. Binary-only questions sometimes need more steps to express what a single 3-way question said at once (Outlook appears twice in CART's tree). On real, larger datasets binary splits are usually the *more* flexible and robust choice, which is part of why CART won. Both algorithms are **greedy**: each picks the best question *right now* without looking ahead, so neither is guaranteed to find the smallest possible tree.

### CHAID (Chi-squared Automatic Interaction Detection) — Kass, 1980

- **Split criterion:** a **Chi-square test of independence**. This is the same chi-square test from `statistics-self-learning`: "is this feature statistically associated with the target?" It splits on the feature with the most significant (lowest p-value) association.
- **Branches:** multi-way, and it merges categories that don't differ significantly (e.g. it might merge Sunny + Rainy into one branch).
- **Stops** when no split is statistically significant, so it has built-in stopping rather than pruning.
- **Use today for:** market research, survey analysis, and customer segmentation, where the goal is *explaining groups to stakeholders* rather than squeezing out maximum prediction accuracy. It's common in SPSS/SAS, rare in Python.

### Side-by-side summary

| | **ID3** | **C4.5 / C5.0** | **CART** | **CHAID** |
|---|---|---|---|---|
| Split criterion | Information Gain | Gain Ratio | Gini (or Entropy); MSE for regression | Chi-square p-value |
| Branches per split | Multi-way | Multi-way | **Always 2** | Multi-way (merges similar categories) |
| Numeric features | ✗ | ✓ (thresholds) | ✓ | ✓ (binned into groups) |
| Regression (numeric target) | ✗ | ✗ | **✓** | Limited |
| Missing values | ✗ | ✓ | ✓ (surrogate splits; sklearn ≥1.3 supports NaN) | ✓ (treated as a category) |
| Overfitting control | None | Post-pruning | Cost-complexity pruning | Significance-based stopping |
| Where you'll meet it | Textbooks, exams | R, Weka, history | **sklearn, Random Forest, XGBoost** | SPSS, SAS, market research |

---

## 7. When to use what — how to decide

### Decision 1 — Which *algorithm*?

In practice this decision is mostly made for you by your tools:

| Your situation | Use |
|---|---|
| Working in Python / sklearn (i.e. almost always, in this track) | **CART.** It's what `DecisionTreeClassifier` / `DecisionTreeRegressor` *are* |
| Target is a number (price, MPG, claim amount) | **CART** (the only one of the four built for regression) |
| Learning, or solving exam questions by hand | **ID3** for the mechanics, **C4.5** to show why Gain Ratio fixes ID3 |
| Working in R and want multi-way splits on categorical data | **C5.0** |
| Survey/marketing segmentation, where the output is a report for stakeholders and you want statistically backed groups | **CHAID** |
| You want the *best accuracy*, not a single readable tree | Not a single tree at all. Use an **ensemble of CART trees** (Random Forest, Gradient Boosting), a later chapter |

**So the honest answer for day-to-day work: you'll use CART, and the real decisions are the settings *inside* CART**, below.

### Decision 2 — Gini or Entropy? (`criterion="gini"` vs `"entropy"`)

- They pick the same split the vast majority of the time. Sections 3 and 4 showed identical rankings on our data.
- **Gini is the default** because it's slightly faster (no logarithm to compute).
- Entropy sometimes produces slightly more balanced trees.
- **How to determine it:** don't agonize. Start with Gini. If you want to check, try both with the same train/test split (Chapter 4) and keep whichever scores better on the *test* set. Usually the difference is tiny.

### Decision 3 — Should I use a Decision Tree at all?

**Reach for a Decision Tree when:**
- **You need to explain the model.** Loan approvals, insurance claims, and medical triage are cases where someone will ask "*why* did it say no?" and a tree can answer with a rule.
- **The relationship isn't a straight line.** E.g. "risk is high for very young *and* very old drivers, low in the middle." Linear/Logistic Regression assume a straight-line relationship (Chapter 2/3 assumptions); trees don't.
- **Features interact.** E.g. "humidity only matters *when it's sunny*." Our tree discovered that automatically. Regression would need you to hand-craft that interaction.
- **Data is mixed categorical + numeric, with different scales.** Trees need **no feature scaling** (a split at `Income ≤ 50,000` works the same whether income is in rupees or lakhs) and don't care about outliers in features much.
- **You want a quick baseline** or want to see which features matter (feature importance).

**Think twice when:**
- **You need the best possible accuracy.** A single tree is usually beaten by Random Forest / Gradient Boosting.
- **The data is small and noisy.** Trees are **unstable**: change a few rows and you can get a completely different tree.
- **The true relationship is smooth and linear.** E.g. MPG dropping steadily with horsepower (Chapter 2). A tree can only approximate a straight line with a staircase of steps, and Linear Regression will do better with far less effort.
- **You need to predict outside the range you trained on.** A regression tree can never predict a value higher than the largest value it saw in training (Section 9).

---

## 8. Overfitting — the tree's biggest weakness, and pruning

### The problem

Left alone, a tree keeps splitting until **every leaf is pure**. On real, noisy data that means it eventually creates leaves for *individual weird rows*: "IF Outlook = Sunny AND Temperature = Mild AND Windy = True AND Humidity = Normal THEN Yes", a rule built from exactly one day. That's memorizing, not learning. It's the non-parametric danger flagged in Section 0.

**How to recognize it:** training accuracy ≈ 100%, test accuracy noticeably lower. This is exactly the train/test gap that Chapter 4's `train_test_split` exists to expose.

### The fix — pruning

**Pre-pruning (stop early), the settings you'll actually tune in sklearn:**

| Setting | What it means | Plain English |
|---|---|---|
| `max_depth` | Max number of questions from root to leaf | "No more than 4 questions deep" |
| `min_samples_split` | A node needs at least this many rows to be split further | "Don't split a group smaller than 20" |
| `min_samples_leaf` | Every leaf must contain at least this many rows | "Every rule must be backed by at least 10 examples" |
| `max_leaf_nodes` | Cap on the total number of leaves (rules) | "At most 15 rules total" |

**Post-pruning (grow fully, then cut back):** grow the whole tree, then remove branches that don't improve performance enough to justify their complexity. sklearn's version is **cost-complexity pruning** (`ccp_alpha`). A bigger `ccp_alpha` means more aggressive cutting and a simpler tree. It's the same spirit as Adjusted R² in Chapter 4 penalizing features that don't earn their keep: **complexity has to pay for itself.**

**How to determine the right values:** there's no formula. Try several (e.g. `max_depth` = 2, 3, 4, 5, 8) and keep the one that scores best on held-out data, not on training data. Chapter 4's train/test split is the minimum; *cross-validation* is the more robust way, and GridSearchCV automates the whole search (Sections 12–13).

---

## 9. Regression Trees (brief) — same idea, numeric target

If the target were a number (say, "minutes of tennis played"), nothing about the *structure* changes. Only two things do:

1. **Purity measure → variance (MSE).** Instead of "how mixed are the Yes/No labels?", it asks "how spread out are the numbers in this group?" The best split is the one that reduces MSE the most. It's the same MSE from Chapter 2, now used to judge groups instead of lines.
2. **Leaf prediction → the average.** A leaf predicts the **mean** of the training values that landed in it, instead of the majority class.

The consequence: a regression tree's predictions are a **staircase**, a set of flat steps, not a smooth line. It also can't extrapolate. If the highest MPG in training was 53, no leaf can ever predict 60.

---

## 10. Pros and cons at a glance

| ✅ Strengths | ❌ Weaknesses |
|---|---|
| Readable, explainable rules (Section 5) | Overfits easily without pruning (Section 8) |
| Handles non-linear patterns and feature interactions automatically | Unstable: small data changes can produce a very different tree |
| No feature scaling needed; works with mixed data types | Greedy: picks the best split *now*, not the best overall tree |
| Built-in feature selection (Temperature was never used) | Staircase predictions for regression; can't extrapolate |
| Fast to train and to predict | Usually less accurate than ensembles (Random Forest / Boosting) |
| The building block of the strongest tabular-data models | ID3/C4.5-style criteria can favor high-cardinality features (Section 6) |

---

## 11. Generalization, Overfitting, Underfitting & the Bias-Variance Tradeoff

Section 8 showed overfitting for trees specifically. Bias-Variance is the general idea behind it, and it applies to **every** ML algorithm. It's the vocabulary for describing the two opposite ways a model can fail.

Before bias and variance, here are the four words they're built on. **Generalization** is the goal. **Overfitting** and **underfitting** are the two ways to miss it. A **good fit** is hitting it.

### The one story that explains all four — three students and an exam

A teacher gives three students last year's question paper to practise on (the **training data**). The real exam (the **test data**) covers the same topics but uses *different* questions.

*(The student scores are made up to tell the story. Each term below also gets a real example from `scripting.py`.)*

| Student | How they studied | Practice paper | Real exam | ML name |
|---|---|---|---|---|
| **A — the memorizer** | Memorized every answer word for word, including the typos | **100%** | **55%** | **Overfitting** |
| **B — the skimmer** | Only learned "most answers are option C" | 60% | 58% | **Underfitting** |
| **C — the understander** | Learned the *concepts* behind the questions | 85% | **82%** | **Good fit / generalizes well** |

The key lesson is that **the practice score alone can't tell A and C apart.** A actually looks *better* on practice. Only the real exam, meaning unseen data, reveals who learned something. That's why every chapter since Chapter 4 keeps a test set locked away.

### Generalization — the actual goal

**Generalization** means how well a model performs on **new data it has never seen**. It's the whole point of ML. A model is only useful if it works on *tomorrow's* claims, not just the ones it trained on.
- **How to measure it:** the test-set score, or the cross-validation score (Section 13). **Never** the training score.
- **Generalization gap** = training score − test score. A small gap means the model generalizes; a big gap means it memorized.

### Overfitting — "learned the noise, not the pattern"

**What it is:** the model is so flexible that it fits the training data *too* well. It captures the real pattern **plus** the random noise and one-off quirks that won't repeat in new data.
- **Symptom:** very high training score, much lower test score, i.e. a **big gap**.
- **Usual causes:** the model is too complex (a deep tree, too many features), too little data, or too little regularization/pruning.
- **Real example (`scripting.py`):** the default tree was 28 levels deep with 304 rules for 876 claims. Train accuracy was **0.995**, test accuracy **0.655**. That's student A.
- **Fixes:** simplify the model (lower `max_depth`, raise `min_samples_leaf`, prune), get more data, drop useless features, or use ensembles (Random Forest, a later chapter).

### Underfitting — "didn't even learn the pattern" (the other key term)

**What it is:** the opposite problem. The model is **too simple** to capture the real pattern, so it does badly *even on the data it trained on*.
- **Symptom:** a low training score *and* a low test score. The gap is small, but both are bad. **A small gap alone is not a sign of success.**
- **Usual causes:** the model is too simple (a tree of depth 1, a straight line through curved data), missing important features, or over-strong pruning.
- **Real example (`scripting.py`):** a depth-1 tree, which gets exactly one question. Train accuracy was 0.724 and CV accuracy 0.702, both capped at the same ceiling. That's student B. (The ceiling here isn't dramatically low, because LOSS alone is a strong signal, but it can't improve without more questions.)
- **Another example:** Chapter 3's straight line through 0/1 pass/fail data, which couldn't bend into the S-shape the data needed.
- **Fixes:** a more flexible model (raise `max_depth`), better or more features, less pruning.

### Good fit — the sweet spot

The training score is good, the test score is nearly as good, and the gap is small. **Real example:** the GridSearchCV-tuned tree scored train **0.724**, test **0.732**. The gap is essentially zero, and it beats Logistic Regression. That's student C.

### How to diagnose any model in 10 seconds

| Training score | Test / CV score | Gap | Diagnosis | Which "error" is to blame |
|---|---|---|---|---|
| Low | Low | Small | **Underfitting** | **High bias** |
| Very high | Much lower | **Big** | **Overfitting** | **High variance** |
| High | Nearly as high | Small | **Good fit** | Both balanced |

The last column is where **bias** and **variance** come in. They're the *causes*; underfitting and overfitting are the *symptoms* you observe:
- **Underfitting is caused by high bias.**
- **Overfitting is caused by high variance.**

### The two kinds of error

**Bias = "too simple, so it misses the pattern."**
The model makes strong, rigid assumptions, so even on the training data it can't capture what's really going on.
- *Story:* predicting house prices using **only** "number of bedrooms". It's wrong in the same direction for every big-plot bungalow, no matter how much data you give it.
- *Tree version:* a tree with `max_depth=1` gets to ask **one** question. It can't express "Sunny AND High humidity → No".
- **How to recognize it: training score is low, and test score is similarly low.** Both are bad, and the gap between them is small. This is called **underfitting**.

**Variance = "too flexible, so it memorizes noise."**
The model bends itself around every quirk of the specific training rows, including random noise that won't repeat in new data.
- *Story:* a student who memorizes last year's exam answers word for word. They get 100% on last year's paper and fail this year's, because the questions changed slightly.
- *Tree version:* an unlimited tree that makes a separate rule for almost every training row.
- **How to recognize it: training score is very high, test score is much lower.** There's a big gap between them. This is **overfitting**.

### The dartboard picture

```
            LOW VARIANCE              HIGH VARIANCE
           (consistent)              (scattered)
         ┌─────────────┐          ┌─────────────┐
LOW      │     ...     │          │  .       .  │
BIAS     │    .(◎).    │  ideal   │     (◎)  .  │  overfit: right on average,
(on      │     ...     │          │ .        .  │  but every retrain lands
target)  └─────────────┘          └─────────────┘  somewhere different
         ┌─────────────┐          ┌─────────────┐
HIGH     │ ...         │          │ .     .     │
BIAS     │ ...  (◎)    │ underfit │    (◎)    . │  worst of both
(off     │             │          │  .      .   │
target)  └─────────────┘          └─────────────┘
```

Each dot is "the model you'd get if you trained on a slightly different sample of data." **Bias** is how far the dots are from the bullseye on average. **Variance** is how spread out the dots are. A deep tree has high variance: retrain it on a slightly different 876 claims and you get a very different tree (the instability from Section 10).

### Why it's a *trade*-off

Making a model more flexible **lowers bias but raises variance**. Making it simpler **lowers variance but raises bias**. You can't push both to zero, so the goal is the **sweet spot** in the middle, where the *total* error on unseen data is lowest.

```
 error on
 unseen data
   │╲                                   ╱
   │ ╲  bias²                  variance╱
   │  ╲___                        ___╱
   │      ╲___      total     ___╱
   │          ╲___  error ___╱
   │              ╲__●__╱   ← sweet spot
   │                                        
   └──────────────────────────────────────── model complexity
     simple (max_depth=1)          complex (max_depth=None)
     UNDERFIT                        OVERFIT
```

### The third piece — Irreducible error (noise)

Bias and variance aren't the whole story. Any model's error on unseen data splits into **three** parts:

$$
\text{Total error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible error}
$$

**Irreducible error** is the randomness in the real world that **no model can ever predict**, however good it is. It lives in the data, not the model.
- *Story:* two claimants with the *exact same* sex, insurance, seatbelt, age and loss amount. One hires an attorney and the other doesn't, maybe because one has a lawyer in the family. Nothing in our 5 columns can tell them apart, so *some* error is guaranteed.
- *Play Tennis version:* maybe on Day 6 the player just felt tired. The weather features can't capture that.
- **Why it matters:** it sets a **ceiling** on how good any model can get with these features. On claim.csv, every model we tried landed between about 0.67 and 0.73 test accuracy. Part of that is probably irreducible given only 5 columns. Tuning moves you between bias and variance, but **it can't reduce irreducible error**. Only **better features** (new information) can, for example "did the claimant have a prior claim?"
- **Practical takeaway:** if extra tuning keeps giving tiny gains (0.742 → 0.741 in Section 13), you've probably reached the floor. Stop tuning and go find better data.

So the full picture is: **bias** (too simple) + **variance** (too jumpy) + **noise** (unpredictable no matter what). You can trade the first two against each other; the third you can only lower with new information.

### Real numbers — the claim.csv tree (from `scripting.py`, Step 7a)

| `max_depth` | Train accuracy | 5-fold CV accuracy (≈ unseen data) | Diagnosis |
|---|---|---|---|
| 1 | 0.724 | 0.702 | Close together but capped. Leaning to **high bias** |
| 3 | 0.739 | 0.708 | |
| 5 | 0.758 | **0.710** | ← **sweet spot** (highest CV) |
| 8 | 0.809 | 0.661 | Gap opening |
| 15 | 0.939 | 0.615 | |
| None (depth 28) | **0.995** | **0.603** | Huge gap. **High variance** |

Read it like the curve above: as depth grows, **training accuracy only ever goes up**, but accuracy on unseen data rises a little, peaks around depth 3 to 5, then falls steadily. Training accuracy alone would tell you "go deeper!" and it would be lying. Same lesson as Chapter 4's Adjusted R²: **never judge a model by how well it fits the data it was trained on.**

### What controls the tradeoff in a Decision Tree

| To reduce **variance** (fix overfitting) | To reduce **bias** (fix underfitting) |
|---|---|
| Lower `max_depth` | Raise `max_depth` |
| Raise `min_samples_leaf` / `min_samples_split` | Lower them |
| Raise `ccp_alpha` (prune harder) | Lower `ccp_alpha` |
| Get more training data | Add more informative features |
| Average many trees (Random Forest, a later chapter) | Use a more flexible algorithm (Boosting, a later chapter) |

Those knobs on the left are **hyperparameters**, which is the next section.

---

## 12. Hyperparameter tweaking — the settings *you* choose

### Parameters vs hyperparameters

This distinction trips everyone up, so here it is plainly:

| | **Parameters** | **Hyperparameters** |
|---|---|---|
| Who sets them? | **The model learns them** from data during `.fit()` | **You choose them** *before* `.fit()` |
| Examples | Linear Regression's `m` and `b` (Chapter 2). A tree's chosen split questions and thresholds (`LOSS <= 0.99`) | `max_depth`, `min_samples_leaf`, `criterion` |
| Analogy | What a student *learns* from studying | The study *plan*: how many hours, which books, how many practice tests |

**Hyperparameter tweaking (tuning)** means trying different settings to find the ones that give the best performance on unseen data. In other words: moving the model along the bias-variance curve (Section 11) until it lands on the sweet spot.

### The Decision Tree hyperparameters that matter most

| Hyperparameter | Default | What it does | Typical values to try |
|---|---|---|---|
| `max_depth` | `None` (unlimited) | Max questions from root to leaf. **The #1 overfitting control** | 2 to 10, plus `None` |
| `min_samples_leaf` | `1` | Minimum training rows in every leaf. It stops "rules backed by 1 example" | 1, 5, 10, 20, 50 (or ~1-5% of rows) |
| `min_samples_split` | `2` | A node needs this many rows before it's allowed to split | 2, 10, 20, 50 |
| `criterion` | `"gini"` | Purity measure, Gini vs Entropy (Section 7, Decision 2) | `"gini"`, `"entropy"` |
| `max_leaf_nodes` | `None` | Cap on total number of rules | 5 to 50 |
| `ccp_alpha` | `0.0` | Post-pruning strength (Section 8) | 0.0 to ~0.02 |
| `class_weight` | `None` | Makes mistakes on a rare class cost more | `None`, `"balanced"` (for imbalanced targets like fraud) |

**Notice the defaults are the *most* overfit settings possible** (unlimited depth, leaves of size 1). sklearn's default tree is deliberately unconstrained, so tuning isn't optional for trees. In `scripting.py` the default tree scored **0.655** test accuracy, worse than Logistic Regression's 0.673.

### The golden rule: never tune on the test set

The obvious approach is "try `max_depth` = 3, 4, 5... and keep whichever gets the best *test* score." **Don't.** If you pick settings by looking at the test set, the test set has quietly become part of training. Its score is no longer an honest "how will this do on brand-new data?" estimate. It's like a student seeing the exam paper before the exam.

The fix is to hold the test set back until the very end, and tune using **cross-validation on the training data only**. Doing that by hand for many combinations is tedious, which is exactly what GridSearchCV automates.

---

## 13. GridSearchCV — automated, honest hyperparameter tuning

The name is two ideas glued together: **Grid Search** + **CV (Cross-Validation)**.

### Part 1 — Cross-Validation (the "CV")

A single train/validation split is noisy: you might just get a lucky or unlucky split. **K-fold cross-validation** fixes that by rotating:

```
Training data (876 rows) split into 5 equal "folds":

Round 1:  [VALID] [train] [train] [train] [train]  → score₁
Round 2:  [train] [VALID] [train] [train] [train]  → score₂
Round 3:  [train] [train] [VALID] [train] [train]  → score₃
Round 4:  [train] [train] [train] [VALID] [train]  → score₄
Round 5:  [train] [train] [train] [train] [VALID]  → score₅

CV score = average(score₁ ... score₅)

                           ┌────────────────┐
The real TEST set (220) →  │ untouched until │
                           │ the very end    │
                           └────────────────┘
```

Every training row gets used for validation exactly once, and the average of 5 scores is much more reliable than 1. **Cross-validation happens inside the training data**, so the test set stays untouched.

### Part 2 — Grid Search

You write down the values you want to try for each hyperparameter. Grid Search tries **every combination** (the "grid"):

```python
param_grid = {
    "criterion":        ["gini", "entropy"],          # 2 options
    "max_depth":        [2, 3, 4, 5, 6, 8, None],     # 7 options
    "min_samples_leaf": [1, 5, 10, 20, 50],           # 5 options
}
# 2 × 7 × 5 = 70 combinations; × 5 folds = 350 trees trained
```

### Putting it together — what `GridSearchCV.fit()` actually does

1. For **each** of the 70 combinations, run 5-fold CV on the training data and record the average score.
2. Pick the combination with the best average score (`best_params_`, `best_score_`).
3. **Refit**: retrain one final model with those settings on **all** the training data (`best_estimator_`).
4. *Then* you evaluate that one final model on the test set, exactly once.

### Real result on claim.csv (from `scripting.py`)

- **Best combination:** `criterion="entropy"`, `max_depth=4`, `min_samples_leaf=50`, with a mean CV ROC-AUC of **0.742**.
- The top 5 combinations **all** had `min_samples_leaf=50`, and depth 6 / 8 / None tied *exactly*. With 50 claims required per leaf, the tree stops growing at depth 6 by itself, so those three caps build the identical tree. **Lesson:** once one hyperparameter is doing the limiting, the others stop mattering.

| Model (same 220-row test set) | Train acc | Test acc | ROC-AUC |
|---|---|---|---|
| Majority-class baseline | – | 0.527 | – |
| Logistic Regression (Chapter 3) | – | 0.673 | 0.745 |
| Decision Tree, **default** | 0.995 | 0.655 | 0.654 |
| Decision Tree, **GridSearchCV-tuned** | 0.724 | **0.732** | **0.759** |

The **same algorithm** went from worst to best purely by choosing hyperparameters well. The train/test gap also collapsed, from 0.995 vs 0.655 to 0.724 vs 0.732. That's the high-variance problem from Section 11, fixed.

### Choosing the `scoring` metric

`GridSearchCV` needs to be told what "best" means via `scoring=`. This is the Chapter 4 decision, reused: `"accuracy"` for balanced classes where both mistakes cost the same, `"roc_auc"` to judge ranking quality across all thresholds, `"recall"` when missing a positive is expensive (fraud, disease), `"precision"` when false alarms are expensive, `"f1"` to balance the two. For regression, use `"neg_root_mean_squared_error"` or `"r2"`. (sklearn maximizes scores, so error metrics come negated.) `scripting.py` uses `"roc_auc"`.

### Practical tips and limits

- **The cost grows fast.** Every extra hyperparameter *multiplies* the combinations: 70 combinations × 5 folds = 350 fits here. Add a 4th hyperparameter with 5 values and it's 1,750. For big grids, **`RandomizedSearchCV`** tries a random sample of combinations (e.g. 50) instead of all of them, and usually finds nearly as good a result much faster.
- **Search coarse, then fine.** First try widely spaced values (`max_depth` 2, 5, 10, None). If 5 wins, search again around it (4, 5, 6).
- **If the best value is at the edge of your grid** (e.g. `min_samples_leaf=50`, the largest we offered), the true best might be beyond it. It's worth extending the grid (70, 100) to check.
- **Diminishing returns.** The #1 and #5 combinations here scored 0.742 vs 0.741. Past a point, better data or features beat more tuning.

---

## The full mental model, tied together

1. **A Decision Tree is a learned flowchart of questions**, non-parametric, usable for classification and regression.
2. **Pick the question that makes the groups purest.** Measure messiness with **Entropy** (0 to 1) or **Gini** (0 to 0.5); the best split has the biggest drop in messiness (**Information Gain** / **Gini Gain**).
3. **Repeat inside every branch** until branches are pure or a stopping rule kicks in. Play Tennis gives Outlook → (Sunny: Humidity) / (Overcast: Yes) / (Rainy: Windy), and Temperature was never needed.
4. **Predict by walking down the tree.** Day 10: Rainy → Not windy → Yes.
5. **The algorithms are variations on one theme:** ID3 (Information Gain, the learning version), C4.5/C5.0 (Gain Ratio, which fixes the many-values bias), CART (Gini, binary, regression too, **what sklearn uses**), CHAID (chi-square, for segmentation reports).
6. **The goal is generalization**, meaning doing well on *unseen* data. **Overfitting** (train high, test much lower) is caused by **high variance**; an unpruned tree memorizes. **Underfitting** (both low) is caused by **high bias**; a tree that's too shallow can't learn the pattern. Total error = bias² + variance + **irreducible error** (noise no model can remove, which only better features can lower).
7. **Hyperparameters** (`max_depth`, `min_samples_leaf`, `criterion`...) are the settings *you* choose to move along that curve. Parameters (the split questions) are what the model learns.
8. **GridSearchCV** tries every hyperparameter combination with k-fold cross-validation *inside the training data*, refits the best one, and leaves the test set for one honest final check. On claim.csv this took the tree from 0.655 to 0.732 test accuracy, beating Logistic Regression (0.673).

The code for all of this lives in `scripting.py` in this folder: the same claim.csv / ATTORNEY problem as Chapter 3, run through the 10-step pipeline, with the depth sweep and GridSearchCV shown live.
