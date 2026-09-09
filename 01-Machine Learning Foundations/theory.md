# Chapter 1 — Machine Learning Foundations

## 0. Why Machine Learning?

### The real-world story

Imagine you run an email service in the early 2000s and you want to filter spam. Your first instinct: write rules.

```
IF subject contains "FREE MONEY" → spam
IF sender is not in contacts AND body contains "click here" → spam
IF email has 10+ exclamation marks → spam
```

This works... for about a week. Spammers adapt immediately — they misspell "FREE M0NEY", drop the exclamation marks, change tactics daily. You're now in an arms race, hand-writing a new `IF` statement every single day, forever. The rules can never keep up, because *you* have to notice the pattern first, then manually encode it.

Now flip the approach: instead of **you** writing the rules, you show a program **thousands of examples** of spam and not-spam emails, and let it figure out the pattern on its own — including patterns you, a human, would never have thought to write a rule for (word combinations, sender behavior, timing, subtle phrasing). That's the entire idea of Machine Learning:

> **Traditional programming:** you write the rules (logic) + give it data → it produces answers.
> **Machine Learning:** you give it data + the answers (examples) → it produces the rules (logic) itself.

```
Traditional:   Rules + Data  ──▶ Program ──▶ Answers
Machine Learning:   Data + Answers ──▶ Program ──▶ Rules (the "model")
```

### Basic definition

**Machine Learning (ML)** = a way of building software that learns patterns from data and improves its predictions from experience, instead of being explicitly programmed with fixed rules for every case.

### Why now, why not always?

ML isn't new as an idea (decades old), but it exploded in the last ~15 years because of three things lining up together:
1. **Data** — the internet generation created oceans of data (clicks, purchases, images, text) to learn from.
2. **Compute** — GPUs made it feasible to crunch that data fast enough.
3. **Algorithms** — better math/techniques (like deep learning) made use of both.

### Why this matters (the "when do I even reach for ML" question)

You don't reach for ML just because it's trendy. You reach for it when:
- The rules are **too complex or unknown** to hand-write (e.g. "what does a cat look like in a photo?" — try writing `IF` statements for that).
- The pattern **changes over time**, so hard-coded rules go stale (spam, fraud, recommendations).
- You have **enough historical data with known outcomes** to learn from (no data → no ML, no matter how fancy the algorithm).

If a simple `IF/ELSE` rule genuinely solves your problem reliably, you don't need ML — ML is a heavier tool (more data, more infrastructure, less interpretable) that you reach for when rules can't keep up. This "do I even need ML" judgment call is itself a core skill — covered more in Section 2 below (approaching a business problem).

---

## 1. Types of Machine Learning

Every ML technique falls into one of a small number of categories, and the category is decided by **one question: what kind of data/feedback does the algorithm get to learn from?**

```
                         MACHINE LEARNING
                                |
        ------------------------------------------------
        |                      |                        |
   SUPERVISED            UNSUPERVISED             REINFORCEMENT
   LEARNING                LEARNING                  LEARNING
        |                      |                        |
  Learns from            Learns from               Learns from
  labeled data           unlabeled data            trial & error
  (has the "answer       (no "answer key" —        + rewards/penalties
  key" during             finds structure/           (no answer key,
  training)                patterns on its own)       feedback comes later,
        |                      |                       as a score)
  -------------          -------------
  |           |          |           |
Classification Regression Clustering  Dimensionality
(predict a     (predict a  (group      Reduction
category)      number)     similar     (simplify/
                            things)     compress data)
```

### 1. Supervised Learning — "learning with an answer key"

**What it is:** you give the algorithm data where each example already has the correct answer (the "label") attached. The algorithm's job is to learn the mapping from input → correct output, so it can predict the answer for *new*, unseen inputs.

**Real-world story:** Think of studying for an exam using **solved past papers**. Every question has the worked-out correct answer right next to it. You study the question-answer pairs, learn the *pattern* of how to solve that type of question, and then you can answer a brand-new question you've never seen before, in the actual exam. The "solved past papers" = labeled training data. The actual exam = new, unlabeled data the model must now handle.

**Two sub-types**, depending on what kind of answer you're predicting:

| Sub-type | Predicts | Example |
|---|---|---|
| **Classification** | A category/label (discrete) | "Is this email spam or not-spam?", "Will this customer churn: yes/no?", "Is this a cat, dog, or bird?" |
| **Regression** | A number (continuous) | "What will this house sell for?", "What will tomorrow's temperature be?", "How many units will we sell next month?" |

**How to tell which one you need:** ask *"is the answer I'm predicting a category/label, or a number?"* Category → classification. Number → regression. (This connects directly to the "types of data" idea from `statistics-self-learning` — categorical target → classification, numerical target → regression.)

**ML/AI relevance:** this is the most common, most mature branch of ML — most real-world business ML (fraud detection, churn prediction, price prediction, medical diagnosis) is supervised learning, because most businesses *do* have historical data with known outcomes (did this transaction turn out to be fraud? yes/no — we found out eventually).

### 2. Unsupervised Learning — "learning without an answer key"

**What it is:** you give the algorithm data with **no labels at all** — no correct answers attached. The algorithm's job is to find hidden structure, patterns, or groupings in the data purely on its own.

**Real-world story:** Imagine you're handed a huge box of mixed photos with **no captions, no folders, nothing** — just raw photos. Nobody tells you "this is a beach photo" or "this is a birthday party photo." But if you spread them out on a table, you'd naturally start noticing photos that *look similar* and grouping them — beach photos together, party photos together — purely based on visual similarity, without ever being told the "correct" category. That's exactly what unsupervised learning does with data.

**Two common sub-types:**

| Sub-type | What it does | Example |
|---|---|---|
| **Clustering** | Groups similar data points together | Segmenting customers into groups by shopping behavior, without ever being told what the groups should be ("high spenders", "bargain hunters" — the model finds these on its own) |
| **Dimensionality Reduction** | Compresses data with many columns/features down to fewer, while keeping the important signal | Simplifying a dataset with 500 columns down to 10 meaningful ones, so it's easier to visualize/model |

**How to tell you need unsupervised learning:** you have data but **no ground-truth answer** to learn from — you're exploring/discovering structure, not predicting a known outcome.

**ML/AI relevance:** often used *before* supervised learning, as an exploration step (e.g. cluster customers first, then build a separate supervised model per cluster), or when labels are genuinely unavailable/expensive to collect (labeling data by hand is often slow and costly — unsupervised learning sidesteps that entirely).

### 3. Reinforcement Learning — "learning by trial and error"

**What it is:** an **agent** takes actions in an **environment**, and instead of being told the "correct" answer directly, it gets a **reward or penalty** based on the outcome of its action. Over many, many attempts, it learns which actions lead to the best long-term reward.

**Real-world story:** Think of how you learned to ride a bicycle. Nobody handed you a labeled dataset of "correct" handlebar angles. Instead, you tried, wobbled, fell (penalty — pain!), tried again, balanced a bit longer (reward — progress!), and slowly, through repeated trial-and-error and the feedback of falling vs. staying upright, your brain learned the right pattern of movements. No one *labeled* each micro-movement as correct/incorrect in advance — you discovered it through consequences.

**Key ingredients:**
| Term | Meaning |
|---|---|
| **Agent** | The learner/decision-maker (e.g. the bike rider, a game-playing bot) |
| **Environment** | The world it interacts with (e.g. the bicycle + road, the game board) |
| **Action** | A choice the agent makes (e.g. lean left, move a chess piece) |
| **Reward/Penalty** | Feedback after the action (e.g. staying upright = +reward, falling = -penalty) |
| **Policy** | The strategy the agent has learned: "in this situation, do this action" |

**Real examples:** game-playing AI (e.g. AlphaGo learning to play Go by playing millions of games against itself), robotics (a robot arm learning to grasp objects through trial and error), self-driving cars (learning driving policies), recommendation systems that adapt based on whether you click or ignore a suggestion.

**How it's different from supervised learning:** supervised learning is told the correct answer *immediately, for every example*. Reinforcement learning only gets a reward/penalty *after acting*, often much later (e.g. you only find out if a chess move was good several moves later, when you win or lose) — the feedback is delayed and indirect, not a direct "correct answer" per input.

---

## 2. How to approach solving a business problem with ML

Before jumping to "which algorithm should I use," a good ML practitioner walks through a structured thought process. Skipping this is the #1 reason ML projects fail in the real world (not bad algorithms — bad problem framing).

### The mental checklist

**Step 1 — Is this actually an ML problem?**
Could a simple rule, a SQL query, or a lookup table solve this reliably? If yes, you don't need ML — don't reach for a hammer when you have a screwdriver. (Recall Section 0: ML is for when rules are too complex/unknown or change over time.)

**Step 2 — What type of ML problem is it?**
This is where Section 1's categories become a practical decision tool:
- Do I have historical data **with known outcomes/labels**? → **Supervised**
  - Is the outcome a category? → **Classification**
  - Is the outcome a number? → **Regression**
- Do I have data but **no known outcome**, and I'm exploring/grouping? → **Unsupervised**
- Am I building something that **takes sequential actions and gets feedback over time** (e.g. a game, a robot, a dynamic pricing system)? → **Reinforcement**

**Step 3 — What data do I actually have (or can get)?**
No data (or too little, or too poor quality) → no amount of algorithm sophistication saves you. This is usually the actual bottleneck in real projects, not the modeling step.

**Step 4 — What does "success" look like, in a measurable way?**
Define this *before* modeling, not after. "Improve customer retention" is not measurable. "Predict which customers will churn in the next 30 days, with at least 80% precision" is. This decision also affects which evaluation metric you'll care about later (precision vs. recall vs. RMSE, etc. — covered in a later chapter).

**Step 5 — Is a wrong prediction cheap or costly?**
A wrong movie recommendation is cheap (minor annoyance). A wrong cancer diagnosis or a wrong loan approval is costly. This affects how conservative your model needs to be, and whether you need a human in the loop reviewing the model's decisions before they take effect.

**Step 6 — Start simple, then get fancy.**
Always try the simplest model that could plausibly work first (e.g. logistic regression before a deep neural network). It's faster to build, easier to explain to stakeholders, and gives you a **baseline** — if a complex model can't beat the simple one by a meaningful margin, the complexity isn't worth it.

### Worked example — turning a vague business ask into an ML framing

**Business ask (vague, as it usually arrives):** *"Our online store is losing money to customers who order and then never pay (fraud). Can AI help?"*

Walking the checklist:
1. **Is this an ML problem?** Rules alone (e.g. "block all orders from country X") are too blunt and go stale as fraudsters adapt → yes, ML fits.
2. **Type?** We have historical orders, and we *know* (eventually) which ones turned out to be fraud (yes/no) → **Supervised → Classification**.
3. **Data?** Past orders with a `is_fraud` column, order amount, account age, delivery address vs. billing address mismatch, etc. — need to confirm this actually exists and is clean.
4. **Success metric?** E.g. "correctly flag at least 90% of fraud cases, while wrongly flagging fewer than 2% of genuine orders" (this is a precision/recall tradeoff — covered later).
5. **Cost of a wrong prediction?** Missing real fraud = lost money. Wrongly blocking a genuine customer = lost trust/sale. Both costly, in different ways → probably needs a human review step for borderline cases, not full auto-blocking.
6. **Start simple:** try logistic regression first with the available features, see how far it gets, before jumping to something complex.

This is the exact translation process every real ML project starts with, before a single line of modeling code is written.

---

## 3. Supervised Learning — Overview

*(Deeper algorithm-level detail — e.g. how linear regression or decision trees actually work internally — comes in later chapters. This section is the conceptual map only.)*

Supervised learning is defined by one thing: **you have labeled data** — every training example comes with the correct answer attached, and the model's job is to learn the relationship between the input features and that answer well enough to predict it for new, unseen data.

```
   Training data (with labels)              New, unseen data (no label)
   ┌─────────────┬───────────┐              ┌─────────────┐
   │   Features   │  Label    │              │   Features   │
   ├─────────────┼───────────┤   learn the  ├─────────────┤   predict
   │ sqft, rooms  │ price=50L │ ───────────▶ │ sqft, rooms  │ ───────▶  price = ?
   │ sqft, rooms  │ price=80L │   pattern    │              │
   │ sqft, rooms  │ price=42L │              │              │
   └─────────────┴───────────┘              └─────────────┘
        MODEL TRAINING                          PREDICTION
```

**The two flavors, revisited with more real examples:**

- **Classification** (predicting a category): spam/not-spam, disease/no-disease, will-churn/won't-churn, digit recognition (0–9), sentiment (positive/negative/neutral).
- **Regression** (predicting a number): house price, stock price, temperature forecast, expected delivery time, expected revenue.

**Common algorithms you'll meet later** (just names to recognize for now, not to master yet): Linear Regression & Logistic Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), k-Nearest Neighbors (KNN), Neural Networks.

**The core workflow, at a glance:**
1. Collect labeled historical data.
2. Split it into a **training set** (model learns from this) and a **test set** (model is evaluated on this, data it never saw during training — to check it actually generalizes, not just memorized).
3. Train the model on the training set.
4. Evaluate on the test set using a metric matching Step 4 of the business-problem checklist above.
5. If good enough, deploy it to predict on genuinely new, real-world data.

**ML/AI relevance:** this workflow (train/test split, fit, evaluate) is the backbone of almost every ML project you'll build, regardless of algorithm — it's covered hands-on with real code in a later chapter.

---

## 4. Unsupervised Learning — Overview

Unsupervised learning is defined by the opposite situation: **no labels**. The model only sees the input features and must find structure entirely on its own, with no "correct answer" to check itself against during training.

```
   Raw data, NO labels                      Model finds structure on its own
   ┌─────────────┐                          ┌─────────────┐
   │  Features    │                         │  Features    │  ← grouped into
   │  only        │  ─────────────────────▶ │  + discovered│    clusters/patterns
   │  (no label   │      find patterns      │  groups      │    the model itself
   │  column)     │                         │              │    decided on
   └─────────────┘                          └─────────────┘
```

**Clustering, worked example:** An e-commerce store has customer data (spend per month, number of orders, average order value) but no pre-existing "customer type" label. A clustering algorithm (e.g. k-means, covered later) looks at the numbers and groups customers into, say, 3 clusters purely based on similarity in the data. *After* clustering, a human looks at each group's characteristics and gives them human-meaningful names — e.g. Cluster 1 = "frequent small spenders", Cluster 2 = "rare big spenders", Cluster 3 = "inactive/at-risk". The algorithm found the grouping; the human interpreted what the groups mean.

**Dimensionality Reduction, worked example:** A dataset has 200 columns describing a product (many of them redundant or barely useful — e.g. `length_cm` and `length_inches` carry the same info twice). A dimensionality reduction technique (e.g. PCA, covered later) compresses those 200 columns down to, say, 15 new columns that still capture almost all the original signal — making the data faster to process, easier to visualize, and often improving downstream model performance by removing noise/redundancy.

**How to know which sub-type you need:** *"Am I trying to group similar things together?"* → clustering. *"Am I trying to simplify/compress a dataset with too many columns?"* → dimensionality reduction.

**ML/AI relevance:** frequently used as a **preprocessing/exploration step** before supervised learning — e.g. cluster users first to segment them, then build a separate targeted supervised model per segment; or reduce dimensions first to speed up and stabilize a downstream supervised model.

---

## 5. Reinforcement Learning — Overview

Reinforcement Learning (RL) is the odd one out — it's not about a fixed dataset at all. It's about an **agent learning a strategy (policy) through interaction and delayed feedback**, in a setting where actions have consequences that unfold over time.

```
        ┌──────────────────────────────────────┐
        │                                       │
        ▼                                       │
   ┌─────────┐   action    ┌─────────────┐      │
   │  AGENT  │ ──────────▶ │ ENVIRONMENT │      │
   │         │             │             │      │
   │         │ ◀────────── │             │      │
   └─────────┘  reward +   └─────────────┘      │
        │       new state                       │
        └───────────────────────────────────────┘
        (this loop repeats thousands/millions of times;
         the agent updates its policy each round to
         chase higher long-term reward)
```

**Worked example — a food delivery app choosing routes:** Imagine an RL agent deciding delivery routes for a driver. Each choice (which street to take) is an **action**. After the delivery, the app finds out the **total time taken** — that's the **reward signal** (shorter time = better reward). The agent doesn't know in advance which specific turn was "correct" — it only learns, over thousands of deliveries, which sequences of choices tend to lead to faster overall deliveries, adjusting its policy after each round. This is exactly the "learning by consequences, not by direct labeling" idea from the bicycle example in Section 1.

**Where RL is actually used today:** game-playing AI (Chess, Go, video games), robotics (learning to walk/grasp), resource allocation and scheduling problems, recommendation systems that adapt to long-term engagement (not just one click), self-driving car decision policies.

**Why RL is harder than supervised/unsupervised learning:**
- Feedback is **delayed** (you might not know if a decision was good until much later).
- The agent's own actions **change future data** it sees (unlike supervised learning, where the training data is fixed upfront).
- It usually needs **huge amounts of trial-and-error** (millions of simulated games/attempts), which is why RL is less common in typical business ML compared to supervised learning — it's reserved for problems that are genuinely sequential/interactive by nature.

**How to know you need RL, not supervised/unsupervised:** ask *"does my problem involve a sequence of decisions, where each decision affects future situations, and I only get a reward/score as feedback (not a direct correct-answer label per decision)?"* If yes → RL. Otherwise, supervised/unsupervised almost always fits better and is far simpler to build.

---

## 6. Parametric vs. Non-Parametric Models

This is a **second, independent way to categorize ML algorithms** — separate from Supervised/Unsupervised/Reinforcement (Section 1). Section 1 asks *"what kind of data/feedback does the algorithm learn from?"*. This section asks a completely different question: ***"once trained, how does the model actually store what it learned?"*** Any algorithm from Section 1 (supervised, unsupervised, whatever) can be further labeled as parametric or non-parametric — the two categorizations sit on different axes and can be combined (e.g. Linear Regression is Supervised + Parametric; k-Nearest Neighbors is Supervised + Non-Parametric).

### The real-world story

Imagine two students prepping for the same exam using 1,000 past solved questions.

**Student A (Parametric approach):** studies all 1,000 questions, then **distills** everything into a short cheat-sheet of, say, 10 general formulas/rules. Once the cheat sheet is done, Student A literally **throws away the 1,000 original questions** — doesn't need them anymore. In the actual exam, Student A answers a new question purely using those 10 formulas.

**Student B (Non-Parametric approach):** doesn't distill anything into a fixed cheat sheet. Instead, Student B **keeps all 1,000 solved questions** close by. In the exam, for a new question, Student B literally flips through the past questions, finds the ones **most similar** to this new question, and bases the answer on those closest matches.

Both students can pass the exam — but they work completely differently, and that difference is exactly what "parametric vs. non-parametric" means in ML.

### The actual definitions

| | Parametric | Non-Parametric |
|---|---|---|
| **What it does** | Learns a **fixed, small set of numbers (parameters)** from the training data, then discards the training data — predictions use only those learned numbers | Keeps some or **all of the training data around**, and the "model" grows/relies on that data at prediction time — no small fixed formula to distill down to |
| **Shape assumption** | **Assumes a fixed shape/form** for the pattern upfront (e.g. "I assume price is a straight-line function of square footage") | **Makes little/no assumption** about the shape of the pattern — lets the data itself decide the shape |
| **Number of parameters** | **Fixed** — stays the same size no matter how much training data you feed it (e.g. always exactly 2 numbers: slope + intercept, whether trained on 100 or 100,000 houses) | **Grows** with the amount of training data (e.g. more stored examples = more comparisons at prediction time) |
| **Speed at prediction time** | **Fast** — just plug numbers into a small formula | **Slower** — often has to search/compare against stored data each time |
| **Flexibility** | Less flexible — if the real pattern doesn't actually match the assumed shape, the model will be systematically wrong (this is called "bias" — more on this in a later chapter on model evaluation) | More flexible — can capture unusual/complex patterns the parametric model's fixed shape would miss, since it isn't locked into one assumed shape |
| **Data needed** | Can work reasonably well with **less data**, because it assumes a shape rather than learning it purely from volume | Usually needs **more data** to work well, since the "knowledge" IS the data (no assumed shortcut shape) |

### Worked example — house prices, both ways

Recall Problem 2 from the practice notebook: predicting house price from square footage.

**Parametric approach (e.g. Linear Regression — Chapter 2):** you assume upfront: *"price probably increases in a straight-line relationship with square footage."* You feed in your 1,000 historical houses, and the algorithm distills that down to exactly 2 numbers: e.g. `price = 500 + 4.2 × sqft` (an intercept of 500 and a slope of 4.2). That's it — 2 numbers, total, is the entire "model." You can throw away the original 1,000 houses; the formula alone predicts any new house's price. **This is exactly what Chapter 2 (Linear Regression) is going to teach you how to build.**

**Non-Parametric approach (e.g. k-Nearest Neighbors):** you make **no** assumption that price follows a straight line at all. Instead, you keep all 1,000 historical houses stored. For a new house, you find the, say, 5 *most similar* past houses (by square footage, bedrooms, location) and average their prices as your prediction. No formula was ever distilled — the "model" is really just the stored data plus a similarity-search rule.

### How to tell which one an algorithm is — the one-question test

Ask: ***"Does the number of things the model needs to remember stay fixed, no matter how much training data I feed it — or does it keep growing as I add more data?"***
- Stays fixed (a handful of numbers, however much data you throw at it) → **Parametric**.
- Keeps growing with the data → **Non-Parametric**.

### Common examples of each

| Parametric | Non-Parametric |
|---|---|
| Linear Regression | k-Nearest Neighbors (KNN) |
| Logistic Regression | Decision Trees |
| Naive Bayes | Random Forests |
| (Simple) Neural Networks with a fixed architecture | Support Vector Machines (with certain kernels) |

*(These are just names to recognize for now — Linear Regression, the first one, is covered in full in Chapter 2 right after this.)*

### Why this distinction actually matters in practice

- **Parametric models** are your **starting point** for most problems (recall Section 2's "start simple, then get fancy" rule) — fast to train, fast to predict, easy to interpret (you can literally read the formula and explain *why* it predicted something — important when explaining a decision to a non-technical stakeholder, e.g. "why was this loan rejected?").
- **Non-parametric models** are reached for when you suspect the real-world pattern is **too complex/irregular** for a simple assumed shape to capture well — at the cost of needing more data, being slower at prediction time, and being harder to interpret ("why" becomes "because these 5 similar past examples said so" rather than a clean formula).
- This is the same "assume a shape vs. let the data speak" tension you'll see again later when checking a statistical distribution's shape (recall the Normal Distribution / skewness discussion in `statistics-self-learning`) — parametric statistics assumes a known distribution shape (like Normal); non-parametric statistics makes no such assumption. Same underlying idea, just applied to ML models instead of statistical tests.

---

## The full picture, tied together

| Type | Has labels? | Learns to... | Example |
|---|---|---|---|
| **Supervised** | Yes | Predict a known kind of answer (category or number) for new data | Spam detection, house price prediction |
| **Unsupervised** | No | Discover hidden structure/groupings on its own | Customer segmentation, data compression |
| **Reinforcement** | No (gets rewards instead) | Learn a strategy through trial, error, and delayed feedback | Game-playing AI, robotics, route optimization |

Before writing a single line of modeling code for any real project, the first job is always: **figure out which of these three boxes the problem belongs in** (Section 2's checklist), because that decision alone determines your entire toolkit, evaluation approach, and even what data you need to collect. Everything from here on (later chapters: how each algorithm actually works, evaluation metrics, the full modeling workflow) builds on top of correctly making this call first.

And remember Section 6 sits on a *second, independent* axis: whichever of the three boxes above a problem lands in, the actual algorithm you pick within it is also either **Parametric** (fixed-size formula, faster, simpler, assumes a shape) or **Non-Parametric** (grows with data, more flexible, no assumed shape). Chapter 2, next, is your first full parametric algorithm — Linear Regression.
