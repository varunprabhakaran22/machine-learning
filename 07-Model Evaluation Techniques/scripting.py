"""
Model Evaluation Techniques -- Holdout, Train/Validation/Test, K-Fold, Stratified K-Fold, LOOCV,
Repeated K-Fold, ShuffleSplit, TimeSeriesSplit, GroupKFold, Nested CV -- on assets/claim.csv

Run this file directly: python scripting.py
(from inside this chapter's folder, since it uses a relative path to ../assets/claim.csv)

Same business question as Chapters 3, 5 and 6 ("Will this claimant hire an attorney?"). This chapter doesn't
build a new algorithm -- it asks a different question about the models we already have:

    "How do we get a TRUSTWORTHY estimate of how well a model will do on data it has never seen?"

Every technique below is a different answer to that question. Model used throughout (unless stated):
the tuned Decision Tree from Chapter 5 (entropy, max_depth=4, min_samples_leaf=50) -- so the only thing
that changes from section to section is the EVALUATION method, never the model.

This file is meant to be self-explanatory through its comments -- read top to bottom.
"""

import time
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, LeaveOneOut, RepeatedStratifiedKFold,
    ShuffleSplit, TimeSeriesSplit, GroupKFold, GridSearchCV, cross_val_score, cross_validate,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")   # hide harmless library notices so the output stays readable

# ===========================================================================
# STEP 1 -- Business Understanding
# ===========================================================================
# Before an insurer trusts a model to route claims, they'll ask: "How accurate will it be on NEXT month's claims?"
# Quoting a number that turns out to be 7 points too optimistic is a real business problem. This chapter is about
# producing that number honestly.

# ===========================================================================
# STEP 2 -- Data Collection  /  STEP 3 -- Data Understanding  /  STEP 4 -- Data Preparation
# ===========================================================================
# Same as Chapters 3, 5 and 6 -- nothing new here.
df = pd.read_csv("../assets/claim.csv")
df = df.drop(columns=["CASENUM"]).dropna()   # 1340 -> 1096 rows
X = df.drop(columns=["ATTORNEY"])
y = df["ATTORNEY"]
print(f"Rows: {len(df)}, share of ATTORNEY=1: {y.mean():.3f}")   # 1096 rows, 0.473


# ===========================================================================
# STEP 5 -- Model Building
# ===========================================================================
def make_tree():
    """A FRESH, untrained copy of Chapter 5's tuned tree. Every evaluation below must train its own copy --
    reusing one already-trained model would let it 'remember' rows from a previous split."""
    return DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=50, random_state=42)


# ===========================================================================
# STEP 6 -- Model Training  +  STEP 7 -- Model Testing: the evaluation techniques
# ===========================================================================

# ---- 7.1 HOLDOUT (train/test split) -- theory.md Section 2 ----------------------------------------------
print("\n7.1 -- Holdout: one 80/20 train/test split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
holdout_acc = accuracy_score(y_test, make_tree().fit(X_train, y_train).predict(X_test))
print(f"  random_state=42 -> test accuracy {holdout_acc:.3f}")   # 0.732 -- the number Chapters 5 & 6 reported

# Its hidden weakness: the score depends on WHICH 220 rows happened to land in the test set.
# Same model, same data -- only the random shuffle changes:
scores = []
for seed in range(50):
    a, b, c, d = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    scores.append(accuracy_score(d, make_tree().fit(a, c).predict(b)))
scores = np.array(scores)
print(f"  Over 50 different random splits: min {scores.min():.3f}, max {scores.max():.3f}, "
      f"mean {scores.mean():.3f}, std {scores.std():.3f}")
# min 0.664, max 0.764, mean 0.720, std 0.023
# A 10-POINT swing from luck alone. random_state=42 (0.732) happened to be a slightly lucky split.
# If you'd reported "76.4%" from one lucky split, the business would be disappointed next month.

# ---- 7.2 TRAIN / VALIDATION / TEST (3-way split) -- theory.md Section 3 ---------------------------------
print("\n7.2 -- Train / Validation / Test")
# Split twice: first carve off the TEST set (20%), then split the rest into TRAIN (60%) and VALIDATION (20%).
X_temp, X_test3, y_temp, y_test3 = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train3, X_val3, y_train3, y_val3 = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
# 0.25 of the remaining 80% = 20% of the total
print(f"  train {len(X_train3)} / validation {len(X_val3)} / test {len(X_test3)}")   # 657 / 219 / 220

# Use VALIDATION to choose max_depth (the test set is not allowed to influence this choice)
print("  max_depth | train acc | validation acc")
val_results = {}
for depth in [1, 2, 3, 4, 5, 6, 8, None]:
    m = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train3, y_train3)
    val_results[depth] = accuracy_score(y_val3, m.predict(X_val3))
    print(f"  {str(depth):>9} | {accuracy_score(y_train3, m.predict(X_train3)):9.3f} | {val_results[depth]:14.3f}")
# depth 1-3: validation 0.708 (tied) ... depth 8: 0.635 ... None: 0.607 while train climbs to 0.998
# -- the overfitting curve from Chapter 5, found WITHOUT ever touching the test set.
best_depth = max(val_results, key=val_results.get)   # max() keeps the FIRST of tied values -> the simplest tree
print(f"  chosen max_depth = {best_depth}  (depths 1-3 tie; when tied, prefer the simplest model)")   # 1

# Only NOW touch the test set -- once -- with the chosen setting, retrained on train + validation (876 rows)
final = DecisionTreeClassifier(max_depth=best_depth, random_state=42).fit(X_temp, y_temp)
print(f"  final test accuracy (reported once): {accuracy_score(y_test3, final.predict(X_test3)):.3f}")   # 0.732
# The weakness: the choice rested on ONE 219-row validation set, and 219 of our 1096 rows never trained the
# model during tuning. K-Fold (next) fixes both.

# ---- 7.3 K-FOLD CROSS-VALIDATION -- theory.md Section 4 -------------------------------------------------
print("\n7.3 -- K-Fold (k=5), done by hand so you can see every fold")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# shuffle=True -> mix the rows before cutting into 5 chunks (see 7.4 for why this matters a lot)
fold_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    # kf.split() yields (row positions to TRAIN on, row positions to VALIDATE on) -- a different chunk each time
    model = make_tree().fit(X.iloc[train_idx], y.iloc[train_idx])
    acc = accuracy_score(y.iloc[val_idx], model.predict(X.iloc[val_idx]))
    fold_scores.append(acc)
    print(f"  fold {fold}: train on {len(train_idx)} rows, validate on {len(val_idx)} rows "
          f"(ATTORNEY=1 share {y.iloc[val_idx].mean():.3f}) -> accuracy {acc:.3f}")
print(f"  K-Fold accuracy = {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")
# folds: 0.682, 0.726, 0.717, 0.731, 0.749 -> 0.721 ± 0.022
# Every one of the 1096 rows was used for validation EXACTLY once, and for training 4 times.
# Notice the ATTORNEY=1 share per fold wanders from 0.425 to 0.507 (overall: 0.473). That's what 7.4 fixes.

# The one-line version of the loop above:
same = cross_val_score(make_tree(), X, y, cv=kf)
print(f"  cross_val_score gives the same 5 numbers: {np.round(same, 3)}")

# ---- 7.4 STRATIFIED K-FOLD -- theory.md Section 5 -------------------------------------------------------
print("\n7.4 -- Stratified K-Fold (k=5)")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):   # note: split(X, y) -- it needs y to stratify
    print(f"  fold {fold}: ATTORNEY=1 share in validation = {y.iloc[val_idx].mean():.3f}")
# 0.473, 0.470, 0.470, 0.475, 0.475 -- every fold matches the overall 0.473 (vs 0.425-0.507 with plain KFold)
strat_scores = cross_val_score(make_tree(), X, y, cv=skf)
print(f"  Stratified K-Fold accuracy = {strat_scores.mean():.3f} ± {strat_scores.std():.3f}")   # 0.725 ± 0.036
# Honest note: the spread (0.036) is NOT smaller than plain KFold's here. Stratifying fixes the CLASS BALANCE
# per fold; it doesn't remove every source of randomness. The benefit grows as the classes get more imbalanced.

# The real disaster stratification (and shuffling) protects you from -- data stored in sorted order:
sorted_df = df.sort_values("ATTORNEY")   # e.g. an export sorted by outcome: all the 0s first, then all the 1s
Xs, ys = sorted_df.drop(columns=["ATTORNEY"]), sorted_df["ATTORNEY"]
bad = cross_val_score(make_tree(), Xs, ys, cv=KFold(n_splits=5))   # plain KFold, NO shuffle
good = cross_val_score(make_tree(), Xs, ys, cv=StratifiedKFold(n_splits=5))   # stratified, still no shuffle
print(f"  SORTED data, KFold without shuffle : {np.round(bad, 3)}")    # 0.75, 0.47, 0.721, 0.607, 0.502
print(f"  SORTED data, StratifiedKFold       : {np.round(good, 3)}")   # 0.723, 0.749, 0.717, 0.703, 0.735
# Without shuffle, fold 1 validates on (almost) only 0s and trains on a very different mix -- scores are garbage.

# ---- 7.5 CHOOSING K -- theory.md Section 6 ----------------------------------------------------------------
print("\n7.5 -- Effect of k")
for k in [2, 5, 10]:
    start = time.time()
    s = cross_val_score(make_tree(), X, y, cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=42))
    print(f"  k={k:>2}: {k} fits, accuracy {s.mean():.3f} ± {s.std():.3f}, took {time.time() - start:.2f}s")
# k=2: 0.719 ± 0.011 | k=5: 0.725 ± 0.036 | k=10: 0.715 ± 0.044
# Bigger k -> each model trains on more data (k=2: 50%, k=10: 90%), but each validation fold is smaller (k=10:
# ~110 rows), so the individual fold scores bounce around more. k=5 or k=10 is the standard compromise.

# ---- 7.6 LEAVE-ONE-OUT (LOOCV) -- theory.md Section 7 ---------------------------------------------------
print("\n7.6 -- Leave-One-Out CV")
start = time.time()
loo_scores = cross_val_score(make_tree(), X, y, cv=LeaveOneOut())
print(f"  {len(loo_scores)} fits (one per row!), took {time.time() - start:.2f}s")   # 1096 fits
print(f"  each 'fold' score is just 0 or 1: {np.unique(loo_scores)}")                # [0. 1.]
print(f"  LOOCV accuracy = {loo_scores.mean():.3f}")                                  # 0.725
# LOOCV = K-Fold with k = number of rows. Each model trains on 1095 rows and is tested on the 1 row left out.
# Result 0.725 -- the same as 5-fold stratified (0.725) -- at ~100x the cost. The per-fold "std" is meaningless
# (every score is 0 or 1). On 1096 rows LOOCV buys nothing; it's worth it only on TINY datasets (theory.md Section 7).

# ---- 7.7 REPEATED STRATIFIED K-FOLD -- theory.md Section 8 ----------------------------------------------
print("\n7.7 -- Repeated Stratified K-Fold (5 folds x 10 repeats)")
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
# runs Stratified 5-Fold 10 times, reshuffling before each repeat -> 50 scores
rep = cross_val_score(make_tree(), X, y, cv=rskf)
print(f"  {len(rep)} scores: mean {rep.mean():.3f}, std {rep.std():.3f}, min {rep.min():.3f}, max {rep.max():.3f}")
# 50 scores: mean 0.715, std 0.027, min 0.662, max 0.785
# The most stable estimate in this file: one lucky/unlucky shuffle can't move a 50-score average much.

# ---- 7.8 SHUFFLESPLIT (repeated random holdout / Monte Carlo CV) -- theory.md Section 9 ------------------
print("\n7.8 -- ShuffleSplit (10 random 80/20 splits)")
ss = cross_val_score(make_tree(), X, y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=42))
print(f"  accuracy {ss.mean():.3f} ± {ss.std():.3f}")   # 0.707 ± 0.021
# Like 7.1's 50-seed experiment, packaged. Unlike K-Fold, a row can land in several test sets, or none.

# ---- 7.9 cross_validate: several metrics + train scores at once -- theory.md Section 10 ------------------
print("\n7.9 -- Comparing models fairly with cross_validate (Stratified 5-Fold, same folds for everyone)")
same_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in [("Logistic Regression", LogisticRegression(max_iter=1000)),
                    ("Decision Tree, default", DecisionTreeClassifier(random_state=42)),
                    ("Decision Tree, tuned", make_tree())]:
    r = cross_validate(model, X, y, cv=same_folds, scoring=["accuracy", "roc_auc"], return_train_score=True)
    # cross_validate = cross_val_score's big brother: several metrics at once, plus the TRAIN score of every fold
    print(f"  {name:<24} train acc {r['train_accuracy'].mean():.3f} | "
          f"CV acc {r['test_accuracy'].mean():.3f} ± {r['test_accuracy'].std():.3f} | "
          f"CV AUC {r['test_roc_auc'].mean():.3f} ± {r['test_roc_auc'].std():.3f}")
# Logistic Regression     train 0.707 | CV acc 0.706 ± 0.022 | CV AUC 0.757 ± 0.036
# Decision Tree, default  train 0.996 | CV acc 0.620 ± 0.017 | CV AUC 0.621 ± 0.018   <- overfit, gap 0.376
# Decision Tree, tuned    train 0.726 | CV acc 0.725 ± 0.036 | CV AUC 0.751 ± 0.036
# PLOT TWIST: on Chapter 5's single test split, the tuned tree's AUC beat Logistic Regression (0.759 vs 0.745).
# With 5-fold CV, Logistic Regression's AUC is slightly HIGHER (0.757 vs 0.751). The ± 0.036 spreads overlap
# completely -- honestly, the two are tied on AUC; the tree wins on accuracy. One split had exaggerated the gap.

# ---- 7.10 Special data: TimeSeriesSplit and GroupKFold -- theory.md Section 11 -------------------------
print("\n7.10 -- Special cases (shown on 12 made-up rows, so every fold fits on screen)")
toy = np.arange(12).reshape(-1, 1)   # pretend these are 12 months, in time order: month 0 ... month 11
print("  TimeSeriesSplit(n_splits=3) -- always train on the PAST, validate on the FUTURE:")
for train_idx, val_idx in TimeSeriesSplit(n_splits=3).split(toy):
    print(f"    train months {train_idx.tolist()}  ->  validate months {val_idx.tolist()}")
# train [0..2] -> val [3,4,5]; train [0..5] -> val [6,7,8]; train [0..8] -> val [9,10,11]

customers = np.array(["A", "A", "A", "B", "B", "C", "C", "C", "D", "D", "E", "E"])   # 12 claims from 5 customers
print("  GroupKFold(n_splits=3) -- all rows of one customer stay on the SAME side:")
for train_idx, val_idx in GroupKFold(n_splits=3).split(toy, groups=customers):
    print(f"    validate customers {sorted(set(customers[val_idx].tolist()))}, "
          f"train customers {sorted(set(customers[train_idx].tolist()))}")
# validate [B, C] / [A] / [D, E] -- no customer ever appears in both train and validation -> the model can't "recognize" a customer it already saw.

# ---- 7.11 NESTED CROSS-VALIDATION -- theory.md Section 12 ------------------------------------------------
print("\n7.11 -- Nested CV: is GridSearchCV's best_score_ an honest estimate?")
# Simulate a SMALL project: only 100 claims available. The other 996 are kept aside as a "future" we can peek at
# purely to check which estimate told the truth (in real life you would NOT have this).
small = df.sample(100, random_state=2)
future = df.drop(small.index)
Xsm, ysm = small.drop(columns=["ATTORNEY"]), small["ATTORNEY"]
grid = {"criterion": ["gini", "entropy"], "max_depth": [1, 2, 3, 4, 5, 6, 8, None],
        "min_samples_leaf": [1, 2, 5, 10, 20], "max_features": [None, 1, 2, 3]}   # 320 combinations
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)   # used to PICK hyperparameters
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)   # used to SCORE the whole picking procedure

search = GridSearchCV(DecisionTreeClassifier(random_state=42), grid, cv=inner, n_jobs=-1).fit(Xsm, ysm)
nested = cross_val_score(GridSearchCV(DecisionTreeClassifier(random_state=42), grid, cv=inner, n_jobs=-1),
                         Xsm, ysm, cv=outer)
# ^ the WHOLE GridSearchCV is re-run inside each outer fold -- the outer validation rows never influence tuning
truth = search.best_estimator_.score(future.drop(columns=["ATTORNEY"]), future["ATTORNEY"])
print(f"  GridSearchCV best_score_ (non-nested) : {search.best_score_:.3f}")   # 0.770 -- looks great
print(f"  Nested CV estimate                    : {nested.mean():.3f}")        # 0.670
print(f"  Actual accuracy on 996 unseen claims  : {truth:.3f}")                 # 0.626
# The non-nested score was 14 points too optimistic: picking the best of 320 combinations on only 100 rows
# rewards the combination that got LUCKY on those folds. Nested CV was far closer to the truth.
# (On the full 1096 rows with a smaller grid, the two agreed to 3 decimals, 0.731 vs 0.731 -- the optimism
# shrinks as data grows and the grid shrinks. It's a small-data / big-grid danger.)

# ===========================================================================
# STEP 8 -- Model Evaluation: which estimate do we report?
# ===========================================================================
print("\nSTEP 8 -- Summary: the SAME tuned tree, measured different ways")
summary = pd.DataFrame([
    ("Holdout, random_state=42", holdout_acc),
    ("Holdout, average of 50 seeds", scores.mean()),
    ("K-Fold (5)", np.mean(fold_scores)),
    ("Stratified K-Fold (5)", strat_scores.mean()),
    ("LOOCV", loo_scores.mean()),
    ("Repeated Stratified 5x10", rep.mean()),
    ("ShuffleSplit (10)", ss.mean()),
], columns=["method", "accuracy"])
print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
# All land between 0.707 and 0.732. The honest statement for the business is a RANGE, e.g.
# "about 71-72% (Repeated Stratified K-Fold: 0.715 ± 0.027)", not "73.2%" from one lucky split.

# ===========================================================================
# STEP 9 -- Model Deployment
# ===========================================================================
# Cross-validation is for ESTIMATING performance. Once you've decided, train ONE final model on ALL the data
# (every row helps) and ship that -- the CV score is the estimate you quote for it.
final_model = make_tree().fit(X, y)
joblib.dump(final_model, "decision_tree_final_all_data.pkl")
print("\nSTEP 9 -- final tree retrained on all 1096 rows, saved to decision_tree_final_all_data.pkl")

# ===========================================================================
# STEP 10 -- CI/CD (conceptual only)
# ===========================================================================
# In an automated retraining pipeline, the "should we ship the new model?" gate should compare CV scores (mean AND
# spread) of old vs new model on the SAME folds -- not one holdout number, which can swing ~10 points by luck (7.1).
