# Chapter 2 — Linear Regression (Recap)

> **This file = revision through the real project** (`scripting.ipynb`, car MPG).
> For definitions and the "why" behind each concept, see **`theory.md`** (section numbers are linked as §).

---

## 1. The whole chapter on one screen

| Concept | One line | Theory |
|---|---|---|
| Model | `y = mx + b`, one `m` per feature, one `b` | §1, §5 |
| Residual | `actual − predicted` for one row | §3 |
| MSE | average of residual². **Training minimises this** | §3 |
| OLS / Gradient descent | direct formula / walk downhill. `.fit()` does it | §4 |
| Train/test split | learn on 80%, judge on the hidden 20% | §6 |
| Baseline | always guess the average. Must beat it | §6 |
| MAE / RMSE | typical miss in real units (RMSE weights big misses more) | §6 |
| R² | how much better than guessing the average (0 → 1) | §6 |
| Multicollinearity | twin features → coefficients flip, predictions still OK | §8.2 |
| Residual plot | random = good, any pattern = something missing | §8.3 |
| Residuals vs each feature | finds **which** feature is curved | §8.5 |
| Polynomial feature | a new column like HP², so the line can bend | §9.1 |
| Under/overfitting | low/low = too simple; high/low = memorised | §9.3 |

⭐ **For GenAI later:** loss (MSE) + gradient descent is exactly how neural networks and LLMs are trained.

---

## 2. The project: predict a car's MPG

**Business ask:** a car maker wants to estimate MPG from design specs *before* building a prototype.
**Data:** `Cars.csv`, 81 cars. Features: `HP` (horsepower), `VOL` (volume), `SP` (top speed), `WT` (weight). Target: `MPG`.
**Business target (agreed upfront):** predictions within **±4 MPG**.

---

## 3. The 10 steps: what we did and why

### Step 1 — Business Understanding
MPG is a **number** → **Regression** → start with the simplest model: **Linear Regression**.
*Why:* the wrong problem type makes everything after it wrong.

### Step 2 — Data Collection
`pd.read_csv("../assets/Cars.csv")`. In real jobs, this is databases and APIs.

### Step 3 — Data Understanding (look before you build)
```python
df.info()       # 0 missing, all numeric
df.describe()   # min / max / mean per column
df.corr()       # how columns move together
```
![features vs mpg](images/03_features_vs_mpg.png)

Applying the 5 scatter questions (theory §8.1):
- **Direction:** all 4 go ↘. More HP/VOL/SP/WT means lower MPG, so every `m` *should* be negative.
- **Shape:** **HP bends** (drops fast, then flattens).
- **Twins:** VOL looks like WT, and HP looks like SP.

![correlation](images/04_correlation.png)

⚠️ **Red flags written down before modelling:** VOL–WT = **1.00** and HP–SP = **0.97** → multicollinearity.

### Step 4 — Data Preparation
```python
X = df.drop(columns=["MPG"]);  y = df["MPG"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 64 train cars, 17 test cars
```
- Split **before** `fit()`. The test set is only for `predict()`.
- `random_state=42` → the same split every run. No `stratify`, because that's only for classification.

### Step 5 — Model Building
`model = LinearRegression()` → an empty model, no `m`, no `b` yet.

### Step 6 — Model Training
```python
model.fit(X_train, y_train)
# MPG = −0.22×HP − 0.54×VOL + 0.53×SP + 0.95×WT + 17.73
```
`model.coef_` = the `m`'s, `model.intercept_` = `b`. `.fit()` ran OLS. Nothing was computed by hand.

### Step 7 — Model Testing
`y_pred = model.predict(X_test)` on the 17 hidden cars. Examples: actual 39.4 → predicted 33.3 (error +6.2), actual 53.7 → predicted 41.7 (error +12.0).

### Step 8 — Model Evaluation
| Metric | Model | Baseline (always guess 33.96) |
|---|---|---|
| MAE | 4.11 | — |
| **RMSE** | **5.65** | 10.16 |
| **R²** | **0.675** | 0 |

- RMSE = √(543.4 / 17). R² = 1 − 543.4 / 1671.7. The full 17-row working is in theory §6.
- Half the baseline's error ✅, but **5.65 > the ±4 target** ❌ → **improve** (section 4 below).

![baseline vs model](images/05_r2_baseline_vs_model.png)

### Step 9 — Model Deployment
```python
joblib.dump(model, "linear_regression_mpg_model.pkl")   # save the m's and b
loaded = joblib.load("linear_regression_mpg_model.pkl")  # load in any app
loaded.predict(new_car)
```
In a real app, a FastAPI/Flask `/predict` endpoint loads the `.pkl` and the React UI calls it. That's where your FE skills plug in.

### Step 10 — CI/CD (concept)
The world changes, so the model gets worse over time (**model drift**). Automate new data → retrain → check it beats the old model → deploy.

---

## 4. The detective story: from 5.65 to 3.30

**Round 1: first model** → RMSE **5.65**. The target is ±4 ❌. So ask **why**.

**Round 2: Clue 1, strange coefficients**
- SP (+0.53) and WT (+0.95) are **positive**, but Step 3 showed both going ↘.
- `df.corr()`: VOL–WT = 1.00, HP–SP = 0.97 → **twins** (theory §8.2).
- **Fix 1:** keep **HP + WT**, drop VOL + SP → RMSE **4.99**. Signs are now HP −0.09, WT −0.61, both negative ✅. Still above ±4 ❌

**Round 3: Clue 2, the errors have a pattern.** Which feature? Plot residuals vs each feature (theory §8.5):

![residuals per feature](images/09_residuals_per_feature.png)

- vs **HP → U-shape** (the culprit). vs **WT → random** (fine).
- This matches Step 3, where HP was the scatter plot that bent.
- **Fix 2:** add an **HP²** column (theory §9.1), for HP only.

**Round 4: prove it with numbers**
| Try | Test R² | Test RMSE | Verdict |
|---|---|---|---|
| HP + WT | 0.747 | 4.99 | starting point |
| + WT² | 0.721 | 5.23 | ❌ worse, WT didn't need a curve |
| + **HP²** | **0.889** | **3.30** | ✅ big jump |
| + HP² + WT² | 0.890 | 3.29 | no better than HP² alone |

→ **HP + WT + HP²**, RMSE **3.30**, is within ±4 ✅ → **stop and ship.**

```python
df["HP_sq"] = df["HP"] ** 2
X = df[["HP", "WT", "HP_sq"]]
# split → fit → predict → evaluate, exactly as before
```

**Why HP² and not another column?** Three pieces of evidence all pointed at HP: (1) its scatter bent, (2) its residuals made a U, (3) the test showed adding HP² helped while adding WT² hurt.

---

## 5. Final leaderboard (same split for everything)

| Version | Train R² | Test R² | Test RMSE |
|---|---|---|---|
| v1: Linear, all 4 features | 0.774 | 0.675 | 5.65 |
| v2: Linear, twins dropped (HP + WT) | 0.730 | 0.747 | 4.99 |
| **v3: Linear, HP + WT + HP²** | 0.886 | **0.889** | **3.30** ✅ |
| Linear + `PolynomialFeatures(2)` on both (HP², WT², HP×WT) | 0.905 | 0.832 | 4.07 |
| Decision Tree (HP + WT) | **1.000** | 0.791 | 4.53 |
| Random Forest (HP + WT) | 0.980 | 0.801 | 4.43 |

**Lessons:**
1. **Fix before you switch.** Linear went 5.65 → 3.30 without changing the model.
2. **Targeted beats blanket.** Only HP² (with evidence) gave 3.30. Every possible square gave 4.07, because the extras were noise.
3. **Fancier ≠ better.** The tree's train R² of 1.000 means it memorised the training rows (overfitting), and its test score is worse.
4. **Simple + explainable wins** when it's the best score.

---

## 6. Telling the business

> "Given a car's horsepower and weight, we predict mileage to within about **±3.3 MPG** on cars the model has never seen. That's **3× more accurate** than using the average. It's good for early design decisions, but road-test before final numbers."

Pattern: **what it does → accuracy in their units → vs the baseline → limits.** Never lead with "R² = 0.89".

---

## 7. Checklists

### The dev flow
```
1. Business: what are we predicting? agree the target (±4 MPG)
2. Collect → 3. Understand (scatter plots, df.corr()) → 4. Clean
5. SPLIT → 6. fit(train) → 7. predict(test)
8. Metrics vs baseline + residual plot
9. Meets target? YES → ship.  NO → diagnose → fix → retrain → compare (section 4)
10. Explain in business units
```

### Diagnose → fix
| You see | Problem | Fix |
|---|---|---|
| Coefficient sign ≠ scatter direction | Twins (multicollinearity) | Drop one of each pair |
| Residual U / curve | Non-linear | Residuals per feature → add culprit² |
| Residual funnel | Uneven error spread | Predict `log(y)` |
| Train low, test low | Underfit | Better features, a curve, or a more flexible model |
| Train high, test much lower | Overfit | Simpler model, fewer features, more data |

### "Okay to ship?"
- [ ] Beats the baseline? (3.30 vs 10.16 ✅)
- [ ] Measured on the test set only? ✅
- [ ] Several metrics plus the plots checked? ✅
- [ ] Coefficient signs make sense? ✅ (after dropping twins)
- [ ] Residuals look random? ✅ (after HP²)
- [ ] Meets the agreed business target? ✅ (±3.3 within ±4)

### Remember
- **Assumptions** (linearity, multicollinearity, homoscedasticity, normality) are mostly **linear-model** concerns. **Independence** applies to **all** models.
- **MAE / MSE / RMSE / R²** work for **all regression models**. Classification uses accuracy, precision, recall, F1, AUC.
- **Maths to know:** the ideas of MSE and gradient descent. **Skip:** the OLS formula and derivatives.
