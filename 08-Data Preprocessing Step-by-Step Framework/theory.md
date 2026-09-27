# Chapter 8 — Data Preprocessing: a Step-by-Step Framework

*(Built on the course's "Data Preprocessing Masterclass" material from the zip: the original exercise notebook is kept here as `course_exercise_original.ipynb`, the Excel screenshot as `label_vs_onehot_excel_example.png`, and the dataset as `../assets/environmental_data.csv`, described in `../assets/environmental_data_description.md`. Every number below comes from actually running `scripting.py`.)*

## 0. Where this sits in the big picture

In the course's 10-step pipeline, preprocessing is **Step 4, Data Preparation**, sitting between *understanding* the data and *building* a model. It's the least glamorous step and usually the biggest: practitioners commonly say it takes **most of a project's time**, because real data arrives messy and every model downstream inherits its mess. **Garbage in, garbage out.**

The course notebook splits preprocessing into two stages:

1. **Data Cleaning:** drop, fill missing values with mean / median / mode, change dtypes.
2. **Data Transformation:**
   - **CONTINUOUS** data → Standard Scaler, MinMax Scaler, Robust Scaler
   - **DISCRETE** data → Label Encoder, One-Hot Encoder

This chapter keeps those two stages and adds what industry adds around them: an **audit** first, a **train/test split in the right place**, and a **Pipeline** at the end, so the whole thing is repeatable and leak-free.

> **Terminology fix (from Chapter 1):** the course slide says "DISCRETE → Label/One-Hot Encoder". What it means is **CATEGORICAL** (text labels like Sunny/Cloudy, Bangalore/Mumbai). In statistics, *discrete* means *countable numbers* (number of children, number of claims); those are numeric and usually get **scaled**, not encoded. Rule of thumb: **text categories → encode; numbers → (maybe) scale.**

---

## 1. The framework at a glance — *when to do what*

```
STAGE 0  AUDIT        look at everything; change nothing
STAGE 1  CLEAN        rule-based fixes: wrong values, dtypes, useless columns, duplicates, missing TARGET rows
STAGE 2  SPLIT        ─────── the line: train / test split goes HERE ───────
STAGE 3  TRANSFORM    impute → encode → scale      (every statistic LEARNED FROM TRAIN, applied to both)
STAGE 4  PACKAGE      put Stage 3 + the model into one Pipeline; cross-validate; ship the pipeline
```

**The single most important rule:** anything that **learns a number from the data** (a mean, a median, a mode, a min/max, the list of categories, a standard deviation) must be learned from the **training set only**, and then applied to the test set. Otherwise the test set leaks into training and your evaluation is optimistic (Chapter 7, Section 15).

| Step | Learns a statistic from the data? | Where it goes |
|---|---|---|
| Fix a typo ("May" → 5, "C" → a real number) | No, it's a fixed rule | **Before** split |
| Change a dtype (text → int) | No | **Before** split |
| Drop useless columns (ID, constant, exact duplicate column) | No | **Before** split |
| Remove duplicate rows | No | **Before** split |
| Drop rows with a missing **target** | No | **Before** split |
| Remove impossible values (Wind = −5) | No (a domain rule) | **Before** split |
| Fill missing with mean / median / mode | **Yes** | **After** split, fit on train |
| Group-wise fill (mean per Weather type) | **Yes** | **After** split, fit on train |
| Label / Ordinal / One-Hot encoding | **Yes** (learns the list of categories) | **After** split, fit on train |
| Standard / MinMax / Robust scaling | **Yes** (mean, std, min, max, median, IQR) | **After** split, fit on train |
| Capping outliers at a percentile | **Yes** | **After** split, fit on train |

**Why the course notebook gets away with doing it all on the full table:** it's a *teaching* notebook that stops before modelling, so there's no test set to leak into. The moment you train a model, the order above matters.

---

## 2. Stage 0 — Audit: look before you touch

**Checklist:** shape · dtypes · missing values per column · unique values of every categorical column · `describe()` ranges checked against the *domain* (the data description) · exact duplicates · **duplicates of the real-world key** · outliers.

### What the audit found in `environmental_data.csv` (158 rows × 10 columns)

| Finding | How it was spotted | Evidence |
|---|---|---|
| `Temp C` and `Month` read as **text** | `dtypes` shows `str`/`object` for "numeric" columns | `pd.to_numeric(errors="coerce")` exposes the culprits: `Temp C` has **"C"** (row 11), `Month` has **"May"** (row 24) |
| `Temp C` is a **duplicate** of `Temp` | Compare the columns | Equal on **157 of 157** numeric rows |
| **"Temp C" isn't Celsius** | Range vs domain | Values run **56 to 97**, and 97 °C is impossible; these are **°F** (the data is R's classic `airquality` set). **The description document is wrong**, which is why you always check data against reality, not just against its documentation |
| `Year` is **constant** | `unique()` | Only 2010: zero information |
| Unnamed first column | Column name | A row number exported by accident |
| **Hidden duplicates** | Check the *key* (Month, Day), not just whole rows | Only **1** exact duplicate row, but **5** repeated days: rows 154–158 repeat earlier dates with identical readings, and **4 of 5 have a different Weather label** |
| Missing values | `isna().sum()` | Ozone **38**, Solar.R **7**, Weather **3** |
| Outliers (1.5 × IQR) | IQR rule | Ozone **135, 168** ppb; Wind **20.1, 20.7** mph |

**Lesson:** the course notebook checks `isna()` and `dtypes`, but never checks duplicates, so the 5 repeated days survive into its final table. A key-based duplicate check is one line (`duplicated(subset=["Month", "Day"])`) and it's the one that catches real-world export errors.

---

## 3. Stage 1 — Clean (rule-based)

### 3.1 Remove columns that carry no information

| Column type | Example here | Why drop it |
|---|---|---|
| ID / row number | `Unnamed: 0` | A model can "learn" row order, which is meaningless (Chapter 5's `Day` ID trap) |
| Exact duplicate column | `Temp C` (= `Temp`) | Adds nothing and creates perfect multicollinearity for linear models (Chapter 2, Section 8) |
| Constant column | `Year` (all 2010) | Zero information |

We also renamed `Temp` → `Temp_F`, so the unit is part of the name. Future readers won't repeat the description's mistake.

### 3.2 Fix wrong values, then dtypes

`"May"` → `5`, then `Month.astype(int)`. **Order matters:** `astype(int)` on a column that still contains "May" crashes. Fix the value first, then the type.

### 3.3 Duplicates

- **Exact duplicates** (`df.duplicated()`): every column identical.
- **Key duplicates** (`df.duplicated(subset=[key columns])`): same real-world entity (one day, one customer, one claim) appearing twice, possibly with *conflicting* values.
- **Decision:** keep one row per key. We kept the **first** (the original reading), which took us from **158 → 153 rows**, exactly the 153 days from May 1 to Sep 30. For conflicting duplicates in a real project, ask the data owner which record is correct.

### 3.4 Missing values — the decision table

| Situation | Action | Example here |
|---|---|---|
| **The target is missing** | **Drop the row. Never impute a target**; that would invent the answer the model is meant to learn | Ozone missing on 37 of the 153 days → **116 usable rows** |
| A feature is missing on **very many** rows (rule of thumb: > ~40–50%) and isn't important | Drop the column | – |
| A feature is missing on a **moderate** share (~5–30%) | Impute; consider adding a "was_missing" 0/1 flag if missingness itself might mean something | (The course dropped Ozone at 24% missing; see the note below) |
| A feature is missing on **very few** rows | Impute, or drop those rows if data is plentiful | Solar.R: 5 of 116; Weather: 3 of 116 |
| **Numeric**, roughly symmetric | **Mean** (or median; they're close) | Solar.R skew = −0.46 (fairly symmetric) → median **203.0** |
| **Numeric**, skewed or with outliers | **Median**: the mean gets dragged by extreme values | Ozone itself: 168 ppb would pull the mean up |
| **Categorical** | **Mode** (most frequent) | Weather → **"S"** |
| A related column explains it | **Group-wise** fill, e.g. Solar.R mean per Weather type | Course notebook: C 192, PS 174, S 193 on the full table; train-only: C 193.2, PS 176.2, S 196.5 |
| Time series | Forward-fill / interpolate from neighbouring days | (a later topic) |

**About the course dropping `Ozone`:** "24% missing → drop the column" is a reasonable rule *for an input feature*. But in this dataset Ozone is the headline air-quality measure (the description calls it "the primary air-quality indicator"), so `scripting.py` makes it the **target** instead, and for a target you drop the **rows**, not the column. **Always ask "what is this column's role?" before applying a missing-value rule.**

### 3.5 Outliers — detect, then *decide*

**Detect:** the IQR rule (below Q1 − 1.5·IQR or above Q3 + 1.5·IQR), a z-score (|z| > 3), or domain limits from the description.

**Decide.** An outlier is not automatically an error:

| Is it... | Action |
|---|---|
| **Impossible** (Wind = −5 mph, Temp = 970 °F, age = 250) | An error: fix it if you can, otherwise set it to missing or drop it |
| **Rare but real** (a 168 ppb smog day) | **Keep it.** Often it's exactly what you most need to predict |
| Real but distorting a **linear / distance** model | Cap it (winsorize at, say, the 1st/99th percentile learned from train), log-transform the column, or use **RobustScaler** |
| Tree-based model | Usually leave it alone; trees split on order, so an extreme value is just "above the threshold" |

**Here:** Ozone 135/168 are the *Poor* air-quality days (our rarest, most important class), and Wind 20.1/20.7 mph is a normal windy day. We **kept all of them**.

### 3.6 Choose the features

Kept: `Solar.R`, `Wind`, `Temp_F`, `Month`, `Weather`. Dropped: `Ozone` (it *is* the target; using it as a feature would be leakage) and `Day` (a day-of-month number is a label for *when*, not a physical cause of pollution).

---

## 4. The split — and why it sits in the middle

After cleaning: **116 rows**, target counts Good **82** / Moderate **27** / Poor **7**. We split 75/25 with `stratify=y` → **87 train / 29 test**, with the rare Poor class on both sides (5 train, 2 test). *(With data this small, the real evaluation uses Repeated Stratified K-Fold, Chapter 7, Section 9.)*

**Leakage example:** if you fill Solar.R with the median of *all 116 rows*, the test rows' values helped choose the number used to fill the training rows. The model has seen a trace of the test set. On a dataset this size the effect is small; on real projects it's a classic reason a model scores well in development and disappoints in production.

---

## 5. Stage 3 — Transform, part 1: imputation (fit on train)

`scripting.py` does it by hand so each learned number is visible:

```python
solar_median = X_train["Solar.R"].median()           # 203.0, learned from TRAIN
X_train["Solar.R"] = X_train["Solar.R"].fillna(solar_median)
X_test["Solar.R"]  = X_test["Solar.R"].fillna(solar_median)   # TEST gaps get the TRAIN median
```

It's the same for Weather with the train mode ("S"). In a Pipeline, `SimpleImputer(strategy="median" / "most_frequent")` does exactly this automatically.

---

## 6. Stage 3 — Transform, part 2: encoding categorical features

Models only understand numbers, so text categories must become numbers. There are two main ways.

### Label (Ordinal) Encoding — one column, one integer per category

| Weather | → encoded |
|---|---|
| C (Cloudy) | 0 |
| PS (Partly Sunny) | 1 |
| S (Sunny) | 2 |

**Compact** (1 column, however many categories), but it **implies an order and distances**: "Sunny is 2 × Partly Sunny" and "Partly Sunny is exactly halfway between Cloudy and Sunny."

### One-Hot Encoding — one 0/1 column per category

| Weather | Weather_C | Weather_PS | Weather_S |
|---|---|---|---|
| C | 1 | 0 | 0 |
| PS | 0 | 1 | 0 |
| S | 0 | 0 | 1 |

**No fake order**, and each category gets its own column (its own coefficient in a linear model). The cost is **one new column per category**.

**Two ways to do it:** `pd.get_dummies(df, columns=["Weather"], drop_first=True)` for quick analysis, or `OneHotEncoder` inside a Pipeline for real models (it remembers the categories learned from train, and `handle_unknown="ignore"` stops a never-seen category from crashing production).

**⚠ The course notebook's One-Hot cell fails:** `ohe.fit_transform(df["Weather"])` passes a 1-D Series, but sklearn encoders need a 2-D table: **`df[["Weather"]]`** (double brackets). That's why that cell has no output.

**`drop_first=True` and the dummy-variable trap:** the three one-hot columns always sum to 1, so any one is perfectly predictable from the other two. That's **perfect multicollinearity**, which breaks Linear Regression's coefficients (Chapter 2, Section 8). Dropping one column (it becomes the all-zeros "baseline" category) fixes it. It's needed for plain linear/logistic regression and harmless to skip for trees.

### The course rule — and *why* it's true

> **Input features:** Parametric models → **One-Hot**. Non-parametric models → **Label Encoding**.
> **Output feature (target):** always **Label Encoder**.

**Why parametric → One-Hot.** A parametric model (Chapter 1, Section 5) learns a *weight per column* and multiplies: `salary = m₁·experience + m₂·location + b`. With label encoding, `location` is one number, so the model is forced to believe each step 0 → 1 → 2 changes salary by the *same* amount, in a fixed order that you chose **arbitrarily** (alphabetically).

**Proof: the course's own Excel example** (`label_vs_onehot_excel_example.png`: 5 employees, Experience + Location → Salary), refitted in `scripting.py` Stage 5a with Linear Regression:

| Features | R² on the 5 rows |
|---|---|
| Experience only | 0.584 |
| + Location, label-encoded **alphabetically** (Bangalore=0, Chennai=1, Mumbai=2) | 0.977 |
| + Location, label-encoded in a **different order** (Chennai=0, Bangalore=1, Mumbai=2) | **0.692** |
| + Location, **one-hot** (any column order) | 0.995 |

The data and the cities are the same; **only the arbitrary numbering changed**, and the linear model's fit collapsed from 0.977 to 0.692. The alphabetical order just *happened* to be a lucky order. One-hot gives every city its own coefficient, so there's no order to get lucky or unlucky with. *(Five rows is far too few for a real model; the point is the **sensitivity to an arbitrary order**, not the R² values themselves. I couldn't reproduce the summary numbers in row 10 of the screenshot, so they aren't quoted here.)*

**Why non-parametric (trees) → Label is fine.** A tree only asks threshold questions like "Weather ≤ 0.5?". It never multiplies the code by a weight, and with one or two splits it can isolate any category ({C} vs {PS, S}, then {PS} vs {S}). So the numbering barely matters, and one column is cheaper than many.

**Proof on the real data** (`scripting.py` Stage 5b, 10×5-fold CV on the 87 training rows), scrambling Weather's integer order:

| Weather numbering | Logistic Regression (parametric) | Decision Tree (non-parametric) |
|---|---|---|
| C=0, PS=1, S=2 (sunniness order) | 0.821 | 0.795 |
| PS=0, C=1, S=2 (meaningless order) | 0.831 | 0.792 |

The parametric model's score **moved by a full point from renumbering alone**; the tree moved 0.003. *Honest note:* on this dataset Weather is a weak predictor (the heavy lifting is done by Temp and Wind), so the stakes are small here. On a feature like *City* or *Product category* that really drives the target, the damage from an arbitrary order can be large, as the salary example shows.

### Refinements to the rule (what the course rule simplifies)

| Situation | Better choice | Why |
|---|---|---|
| The category has a **real order** (Low < Medium < High; Cloudy < Partly Sunny < Sunny; education level) | **Ordinal encoding with the order you specify**, even for parametric models | The order is *true*, so the numbers carry real information. `OrdinalEncoder(categories=[["C", "PS", "S"]])` |
| **KNN, SVM, neural networks**: non-parametric or flexible, but **distance / weight based** | **One-Hot** (+ scaling) | "Non-parametric" isn't the real test; the real test is "does the model do arithmetic on the codes?" KNN measures distances, so label codes create fake distances |
| **Many categories** (e.g. 500 pin codes) | Don't one-hot 500 columns: group rare categories into "Other", or use frequency/target encoding (a later topic) | One-hot explosion: more columns than the data can support |
| Encoding an **input** feature in sklearn | `OrdinalEncoder` (works inside `ColumnTransformer`) | `LabelEncoder` is designed for the 1-D target; same idea, but the right tool for X |

**The more precise version of the rule:** *Does the model multiply or measure distances with the numbers?* **Yes** (linear, logistic, KNN, SVM, neural nets) → **One-Hot**, unless the order is real. **No** (trees, Random Forest, boosting) → **Label/Ordinal** is fine.

### The output feature → always Label Encoder

```python
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df["AirQuality"])      # {'Good': 0, 'Moderate': 1, 'Poor': 2}
target_encoder.inverse_transform([1])                   # → 'Moderate'  (turn predictions back into words)
```

**Why a target never needs one-hot:** a classifier treats the target's 0/1/2 as **names of classes**, never as amounts it multiplies. There's no fake-order problem to fix. (For a **regression** target, a number like Ozone in ppb, no encoding at all is needed.)

---

## 7. Stage 3 — Transform, part 3: scaling continuous features

**The problem:** features live on wildly different scales. Here Solar.R runs 7–334, Wind 2.3–20.7, Temp 57–97. For any model that measures distances or uses gradient descent, **the biggest-numbered feature dominates**, simply because of its units.

**Real example (`scripting.py`, Stage 3d):** two training days differ by 41 in Solar.R, 0.6 in Wind and 23 in Temp. In raw units, **76.1% of the (squared) distance between them comes from Solar.R alone**. To a KNN model, "how similar are these days?" is almost purely a sunlight question, which is an accident of units, not physics.

### The three scalers (worked on 5 Wind readings: 4.0, 7.4, 9.7, 12.0, 20.1 mph)

| Scaler | Formula | Result | Output range |
|---|---|---|---|
| **StandardScaler** | (x − mean) / std, with mean 10.64 and std 5.42 | −1.23, −0.60, −0.17, 0.25, 1.75 | Mean 0, std 1 (usually −3…+3) |
| **MinMaxScaler** | (x − min) / (max − min) | 0.00, 0.21, 0.35, 0.50, 1.00 | Exactly 0…1 |
| **RobustScaler** | (x − median) / IQR, with median 9.7 and IQR 12.0 − 7.4 = 4.6 | −1.24, −0.50, 0.00, 0.50, 2.26 | Median 0, not bounded |

### What an outlier does to each (add one faulty 80 mph reading)

| Scaler | The same 5 normal days become… | Verdict |
|---|---|---|
| MinMax | 0.00, 0.04, 0.08, 0.11, 0.21 (and the 80 mph day = 1.00) | **Squashed**: the one outlier defines "1.0" and crushes everyone else together |
| Standard | −0.69, −0.56, −0.47, −0.39, −0.08 | Also squashed: the outlier inflates the std |
| **Robust** | −0.68, −0.34, −0.11, 0.11, 0.92 | **Keeps normal days spread out**: the median and IQR ignore the extreme value |

### Which scaler, when

| Use | When |
|---|---|
| **StandardScaler** | The default. Roughly bell-shaped data; linear/logistic regression (especially with regularization), SVM, PCA, neural nets |
| **MinMaxScaler** | You need a fixed 0–1 range (image pixels, some neural nets), **and** there are no significant outliers |
| **RobustScaler** | The data has outliers you're keeping (like our Ozone and Wind extremes) |
| **No scaling** | Tree-based models (Decision Tree, Random Forest, boosting), since a split at "Wind ≤ 9.7" works identically in any units (Chapter 5, Section 7) |

### Real results (`scripting.py` Stage 5c, 10×5-fold CV accuracy)

| Model | No scaling | Standard | MinMax | Robust |
|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.825 | 0.815 | **0.827** |
| **KNN** (distance-based) | 0.796 | **0.818** | **0.756** | 0.800 |
| Decision Tree | 0.790 | 0.791 | 0.790 | 0.790 |

- **The tree doesn't care.** Every scaler gives the same ~0.790, which proves "trees don't need scaling."
- **KNN cares the most:** there's a 6-point spread between the best and worst scaler. Notice **MinMax *hurt* KNN** (0.756, worse than no scaling). With the numeric columns squeezed into 0–1, the one-hot Weather columns (also 0/1) suddenly weigh as much as Temp in the distance, and Weather is a weak predictor. **Scaling decisions change what a distance model pays attention to.**
- **Logistic Regression** gains 1.5–2 points from Standard/Robust scaling.

**Same rule as always:** scalers learn numbers (mean, std, min, max, median, IQR), so fit them on train only and `.transform()` the test set.

---

## 8. Stage 4 — Package it: `ColumnTransformer` + `Pipeline` (the industry standard)

Doing Stage 3 by hand is great for learning, but it's error-prone in real work (forgetting to apply a step to the test set, or fitting a scaler on the wrong data). Industry bundles every learned step with the model:

```python
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale",  StandardScaler())]),                  ["Solar.R", "Wind", "Temp_F", "Month"]),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("encode", OneHotEncoder(handle_unknown="ignore"))]), ["Weather"]),
])
model = Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=5000))])
```

**Why this is the standard:**
1. **Leakage-proof by construction.** `cross_val_score(model, ...)` re-fits the imputers, encoder and scaler **inside every training fold**, so a validation fold never influences its own preprocessing (Chapter 7).
2. **One object to ship.** `joblib.dump(model, "air_quality_pipeline.pkl")` saves preprocessing *and* model together. Production can hand it a raw, messy row (a missing Solar.R, a text Weather) and it does every step itself. `scripting.py` does exactly this: a hot, calm, sunny July day with Solar.R missing → **"Moderate"**.
3. **Readable and repeatable.** The whole recipe is visible in one place.

### Real results (`scripting.py`, Stage 4)

| Pipeline | 10×5-fold CV accuracy (train) | Test accuracy (29 rows) |
|---|---|---|
| Majority-class baseline (always "Good") | 0.713 | – |
| Logistic Regression + One-Hot + StandardScaler *(parametric recipe)* | 0.825 ± 0.062 | 0.828 |
| Random Forest + Ordinal, no scaling *(non-parametric recipe)* | 0.837 ± 0.062 | 0.828 |

Both clearly beat the baseline. With only 87 training rows the ±0.062 spreads overlap, so the two are effectively tied (Chapter 7, Section 10).

---

## 9. The master decision table — *when to do what*

| Question | If… | Do |
|---|---|---|
| **Column role?** | ID / row number / constant / exact duplicate | Drop (Stage 1) |
| | It's the target | Missing → drop row; categorical → LabelEncoder; numeric → leave as is |
| **Wrong values?** | Typos, wrong units, text in numeric columns | Fix with a rule, then `astype` (Stage 1) |
| **Duplicates?** | Same real-world key twice | Keep one per key (Stage 1) |
| **Missing (feature)?** | Most of the column missing, not important | Drop the column |
| | Numeric, symmetric | Mean (fit on train) |
| | Numeric, skewed / outliers | Median (fit on train) |
| | Categorical | Mode (fit on train) |
| | A related column explains it | Group-wise mean/median (fit on train) |
| **Outlier?** | Impossible value | Fix / set missing / drop |
| | Real value | Keep; if the model is linear/distance-based, consider capping, a log transform, or RobustScaler |
| **Categorical input?** | Real order | OrdinalEncoder with your order |
| | No order + linear/logistic/KNN/SVM/NN | One-Hot (`drop_first=True` for plain linear models) |
| | No order + tree-based | Label/Ordinal is fine |
| | Hundreds of categories | Group rare ones / frequency or target encoding |
| **Continuous input?** | Tree-based model | No scaling |
| | Linear / logistic / KNN / SVM / NN / PCA, no big outliers | StandardScaler (MinMax if 0–1 is required) |
| | Same models, with outliers | RobustScaler |
| **Always** | | Split before any "learned" step; put the learned steps in a Pipeline; cross-validate; ship the pipeline |

---

## 10. Reviewing the course notebook against the framework

| Course notebook step | Verdict | Framework version |
|---|---|---|
| `isna()`, `dtypes`, `describe()`, `value_counts()` | ✅ Good audit habits | Plus duplicates (exact **and** key-based) and domain-range checks |
| Drop `Unnamed: 0` and `Temp C` | ✅ | Plus `Year` (constant) |
| Delete `Ozone` (24% missing) | ⚠ Depends on its role | Here Ozone is the natural **target** → drop rows, keep it |
| Solar.R filled with the mean per Weather group | ✅ Nice technique | Compute the group means on **train only** |
| `dropna()` for the remaining rows | ✅ Fine for 3 rows | – |
| `"May"` → 5, `astype(int)` | ✅ Right order (value, then type) | – |
| Never checks duplicates | ❌ 5 repeated days survive | `drop_duplicates(subset=["Month", "Day"])` |
| `LabelEncoder` on the input Weather | ✅ OK for trees | `OrdinalEncoder` inside a Pipeline; One-Hot for parametric models |
| `pd.get_dummies(..., drop_first=True)` | ✅ | `OneHotEncoder` inside a Pipeline for real models |
| `OneHotEncoder().fit_transform(df["Weather"])` | ❌ Fails (1-D input) | `df[["Weather"]]` |
| Everything done on the full table | ⚠ Fine for a demo with no model | Split first; fit learned steps on train (Pipeline) |
| Accepts "Temp C" as Celsius (from the description) | ⚠ | Values 56–97 are °F; always sanity-check units |

---

## The full mental model, tied together

1. **Preprocessing = Audit → Clean → Split → Transform → Package.** The split goes *between* rule-based cleaning and anything that learns a statistic.
2. **Audit before touching:** dtypes hide typos ("C", "May"), descriptions can be wrong (°F labelled °C), and duplicates hide behind conflicting values (1 exact, but 5 repeated days).
3. **Clean with rules:** drop ID/constant/duplicate columns, fix values then types, dedupe on the real-world key, **never impute the target**, and keep real outliers.
4. **Impute** with mean (symmetric), median (skewed/outliers) or mode (categorical), learned from train.
5. **Encode** categorical inputs: models that multiply or measure distances → **One-Hot** (a label order is an arbitrary fake fact: R² 0.977 vs 0.692 from renumbering alone); trees → **Label/Ordinal**; a *real* order → Ordinal with that order. **Target → LabelEncoder**, always.
6. **Scale** continuous inputs for linear/distance models: Standard (default), MinMax (0–1, no outliers), Robust (outliers). **Trees don't need it** (0.790 under every scaler).
7. **Package it in a `Pipeline`**, cross-validate it, and ship the whole pipeline, so production gets the same preprocessing automatically.
