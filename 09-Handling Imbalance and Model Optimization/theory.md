# Chapter 9 — Handling Imbalance and Model Optimization

*(Follows the course's "Handling Imbalance Dataset Problem" notebook, used for reference only and not kept in this folder: sklearn's breast cancer dataset, the course's exact split, and its Decision Tree with `class_weight={0:2,1:1}`. `scripting.py` reruns it and adds the techniques below. Every number here comes from that script.)*

## 0. The problem in one paragraph

**Imbalanced data** means one class is much rarer than the other: fraud (≈1%), rare diseases, equipment failures, customer churn. The rare class is almost always **the one you care about**. A model trained naively learns that ignoring it is the easy way to be "accurate".

**Our data:** 569 tumours, **212 malignant (37%)** vs 357 benign (63%). That's a **mild** imbalance.

⚠ **Label trap:** in this dataset **0 = malignant**, 1 = benign. The class we want to catch is **0**, so recall, precision and F1 must be computed with `pos_label=0`. sklearn's default `scoring="f1"` or `"recall"` would score *benign*, the wrong class.

---

## 1. First fix the ruler: accuracy lies on imbalanced data

Always compare against "always predict the majority class":

| Data | "Always benign" accuracy | Cancers caught |
|---|---|---|
| Course data (37% malignant) | 63.2% | **0 of 42** |
| Severe version (20 malignant, 5.3%) | **94.7%** | **0 of 6** |

A 94.7%-accurate model that catches no cancer at all: that's the **accuracy paradox** (Chapter 4). For imbalanced problems, judge models by **recall** of the rare class (cancers caught), **precision** (how many alarms were real), **F1** (their balance), and PR-AUC / balanced accuracy, **never accuracy alone**.

**What each mistake costs here:** a **missed cancer** (false negative) means an untreated patient. A **false alarm** (false positive) means an extra biopsy. So we protect **malignant recall** first.

---

## 2. The toolbox — what each technique does

### 2.1 `stratify` — keep the ratio fair (course ✓)

`train_test_split(..., stratify=y)` keeps the class ratio identical on both sides: malignant is 0.374 of train and 0.368 of test. **It doesn't fix imbalance.** It prevents an *unlucky* split, which matters a lot when the rare class is tiny (6 test cases in the severe version). Use `StratifiedKFold` for CV (Chapter 7, Section 5).

### 2.2 `class_weight` — make rare-class mistakes cost more (course ✓)

The model's training loss (or tree impurity) counts each malignant row as if it were **w** rows, so the model can't cheaply ignore the minority class. **No data is added or removed.**

| Option | Meaning |
|---|---|
| `{0: 2, 1: 1}` (course) | Each malignant row counts double |
| `"balanced"` | Weight = n_samples / (n_classes × n_in_class): **455 / (2 × 170) = 1.338** for malignant, **455 / (2 × 285) = 0.798** for benign, so the two classes carry equal total weight |
| Any dict | A dial you can tune, e.g. with GridSearchCV |

| Decision Tree (depth 3), test set: 42 malignant | Missed cancers | False alarms | Recall |
|---|---|---|---|
| No weight | 4 | 10 | 0.905 |
| `{0:2,1:1}` **(course; reproduces its matrix [[38 4] [7 65]] exactly)** | 4 | 7 | 0.905 |
| `"balanced"` | 5 | 6 | 0.881 |
| `{0:5,1:1}` | 3 | 6 | 0.929 |

Supported by most sklearn classifiers (`LogisticRegression`, `DecisionTree`, `RandomForest`, `SVC`); XGBoost uses `scale_pos_weight`. It's usually the **first thing to try**: zero cost, no data changes.

### 2.3 Resampling — change the training data (`imbalanced-learn`) *(added)*

| Method | What it does | Watch out |
|---|---|---|
| **Random oversampling** | Duplicates minority rows (170 → 285 malignant) | Exact copies make overfitting easier |
| **Random undersampling** | Drops majority rows (285 → 170 benign) | Throws away real data; only sensible when data is plentiful |
| **SMOTE** | *Creates new* minority rows by interpolating between a minority row and its nearest minority neighbours | Needs scaled numeric features; can blur the class boundary |

**Results (tree):** oversampling 5 missed / 6 false alarms, undersampling 3 / 8, SMOTE 5 / 6. That's the same neighbourhood as `class_weight`. **On a mild imbalance there isn't much to fix**, and resampling mostly just trades one error type for the other.

**⚠ The #1 resampling mistake: resample *after* splitting, on training data only.** In the severe version, oversampling *before* the split gave a "perfect" **recall 1.000**. The honest number (split first, oversample only the training set) is **0.833**. Oversampling first puts **copies of the same tumour in both train and test**, so the test just checks memory. Use `imblearn.pipeline.Pipeline`: it resamples only during `.fit()`, including inside every CV fold.

### 2.4 Threshold moving — choose where "malignant" starts *(added)*

Classifiers output a probability; `.predict()` silently uses **0.5**. For a screening test you can say "flag it if P(malignant) ≥ 0.1":

| Scaled Logistic Regression, P(malignant) ≥ | 0.5 | 0.3 | 0.2 | 0.1 |
|---|---|---|---|---|
| Missed cancers | 1 | 1 | 1 | **0** |
| False alarms | 0 | 3 | 6 | 10 |

That's the precision-recall tradeoff (Chapter 4) turned into a **business dial**. It's often the most effective lever and needs no retraining. In a real project, **pick the threshold on validation data / CV**, not the test set.

### 2.5 Other options (know they exist)

Collect more minority data (the best fix of all); ensembles built for imbalance (`BalancedRandomForestClassifier`, `EasyEnsembleClassifier` in imbalanced-learn); anomaly-detection methods for extreme cases (< 0.1%); combined samplers such as SMOTE + Tomek links.

---

## 3. Model optimization — what actually moved the needle

The course's "Model Optimization" section tries **scaling**. Here's the full picture:

**1. Scaling doesn't help a tree.** Tree + MinMaxScaler gives **4 missed / 7 false alarms, identical** to the unscaled tree. A tree splits on "feature ≤ threshold", and scaling doesn't change the order of values (Chapter 8, Section 7). The course's scaled run printed slightly different numbers only because its `DecisionTreeClassifier` has **no `random_state`**, so ties between equal splits are broken randomly on every run. **Always set `random_state` when comparing models.**

**2. Choosing a model that suits the data mattered most.** With 30 smooth numeric measurements, a **scaled Logistic Regression** (here, scaling matters) misses **1** cancer with **0** false alarms (accuracy 0.991), versus 4 and 7 for the course's tree.

**3. Tune with GridSearchCV, scored on the class that matters.** Use `scoring=make_scorer(f1_score, pos_label=0)`, with `class_weight` included in the grid:

| Model | Best settings (5-fold stratified CV) | CV F1 (malignant) | Test: missed / false alarms |
|---|---|---|---|
| Decision Tree | depth 4, min_samples_leaf 5, `{0:2,1:1}` | 0.911 | 4 / 8 |
| Logistic Regression + StandardScaler | C = 0.1, `class_weight="balanced"` | **0.959** | **1 / 0** |

Tuning the tree barely helped: the *tree itself* was the ceiling.

**4. Then move the threshold** (2.4) to trade the last missed cancer against extra biopsies.

---

## 4. When to use what

| Situation | Do this |
|---|---|
| Always | `stratify=y` + StratifiedKFold; judge by rare-class recall/precision/F1 (`pos_label`!) against a majority-class baseline |
| Mild imbalance (e.g. 60/40 to 80/20) | Often nothing beyond the above; try `class_weight="balanced"` and threshold tuning |
| Moderate to severe (e.g. 90/10 to 99/1) | `class_weight` first → threshold tuning → resampling (SMOTE / oversampling) inside an imblearn Pipeline → imbalance-aware ensembles |
| Extreme (< ~0.1%) | Anomaly detection; collect more positives |
| Lots of data, heavy imbalance | Undersampling becomes reasonable (faster training, little information lost) |
| Missing a positive is very costly (cancer, fraud) | Lower the threshold; tune for recall or F-beta (β > 1) |
| False alarms are very costly (spam filter hiding real mail) | Raise the threshold; tune for precision |

---

## 5. Course notebook — quick review (for reference; `scripting.py` already applies every fix)

| Step | Verdict |
|---|---|
| `stratify=y` in the split, pie charts to check the ratio | ✅ |
| `class_weight={0:2,1:1}` upweighting malignant | ✅ The right class, since 0 = malignant |
| Checking train *and* test reports (train 0.98 vs test 0.90) | ✅ A healthy generalization check (Chapter 5, Section 11) |
| Tree with no `random_state` | ⚠ Results change between runs; that, not scaling, explains the scaled run's different numbers |
| Markdown says "Standard Scaler", code uses MinMaxScaler | ⚠ A label mismatch |
| `MinMaxScaler().fit_transform(X)` on **all** data before the split | ⚠ Leakage (Chapter 8, Section 1); fit on train only, or use a Pipeline |
| Scaling a tree as the "optimization" | ⚠ It can't change a tree; it helps LogReg/KNN/SVM |
| `dump(scaled_X, "data_transformation.pkl")` | ❌ That saves the scaled **data table**, not the **fitted scaler**, so it can't transform a new patient. Save the scaler, or better, one Pipeline (scaler + model), as `scripting.py` does with `breast_cancer_pipeline.pkl` |

---

## Mental model

1. **Imbalance = the class you care about is rare, so accuracy stops meaning anything** ("always benign" = 94.7% in the severe case). Measure rare-class recall/precision/F1 with the right `pos_label`.
2. **Stratify** everything, so the imbalance is the same in train, test and every fold.
3. **`class_weight`** makes rare-class mistakes cost more. It's free, so try it first.
4. **Resampling** (over / under / SMOTE) changes the training data, **only after splitting**; otherwise you get fake perfect scores.
5. **Threshold moving** turns the precision-recall tradeoff into a business decision.
6. **Optimization:** pick a model that suits the data (scale it if it needs scaling), tune with GridSearchCV on a rare-class metric, then set the threshold. Ship **one pipeline**, not a pickled data table.
