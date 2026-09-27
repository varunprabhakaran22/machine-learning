"""
Decision Tree -- end-to-end on assets/claim.csv, tuned with GridSearchCV

Run this file directly: python scripting.py
(from inside this chapter's folder, since it uses a relative path to ../assets/claim.csv)

Same business question and same data as 03-Logistic Regression/scripting.ipynb:
    "Will this insurance claimant hire an attorney?"  (ATTORNEY: 0 = no, 1 = yes)
...so at the end we can put the Decision Tree's numbers right next to Logistic Regression's
(accuracy 0.673, ROC-AUC 0.745) and see which algorithm did better on the SAME test set.

Structured around the course's 10-step ML pipeline. Topics from theory.md shown live here:
    - Section 11 (Bias-Variance Tradeoff)      -> Step 7: an untuned tree overfits, then a depth sweep shows both ends
    - Section 12 (Hyperparameter tweaking)     -> Step 5/7: max_depth, min_samples_leaf, criterion
    - Section 13 (GridSearchCV)                -> Step 6: try every combination with 5-fold cross-validation, keep the best

This file is meant to be self-explanatory through its comments -- read top to bottom.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# ===========================================================================
# STEP 1 -- Business Understanding
# ===========================================================================
# An insurer wants to know, as soon as a bodily-injury claim comes in, whether the claimant is likely to
# hire an attorney. Attorney-involved claims cost more and take longer, so flagging them early lets the
# insurer route them to experienced adjusters. A Decision Tree is a good fit here because the business
# will ask "WHY was this claim flagged?" -- and a tree answers with readable IF/THEN rules (theory.md Section 5).

# ===========================================================================
# STEP 2 -- Data Collection
# ===========================================================================
df = pd.read_csv("../assets/claim.csv")   # 1340 claims, 7 columns
print("STEP 2 -- raw data shape:", df.shape)   # (1340, 7)

# ===========================================================================
# STEP 3 -- Data Understanding
# ===========================================================================
# Columns: CASENUM (ID), ATTORNEY (target), CLMSEX, CLMINSUR, SEATBELT (0/1 flags), CLMAGE (age), LOSS (claim amount)
print("\nSTEP 3 -- missing values per column:")
print(df.isnull().sum().to_string())
# CLMSEX 12, CLMINSUR 41, SEATBELT 48, CLMAGE 189 missing -- the rest complete

print("\nTarget balance (ATTORNEY):")
print(df["ATTORNEY"].value_counts().to_string())
# 0 -> 685, 1 -> 655 -- roughly balanced, so accuracy is a fair metric (Chapter 4, Classification Model.md)

# ===========================================================================
# STEP 4 -- Data Preparation
# ===========================================================================
df = df.drop(columns=["CASENUM"])   # an ID column -- theory.md Section 6 shows why a tree would happily split on it and memorize
df = df.dropna()                     # same tradeoff as Chapter 3: drop incomplete rows (1340 -> 1096) rather than guess values
print("\nSTEP 4 -- rows after cleanup:", len(df))   # 1096

X = df.drop(columns=["ATTORNEY"])   # features: CLMSEX, CLMINSUR, SEATBELT, CLMAGE, LOSS
y = df["ATTORNEY"]                   # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    # EXACT same split settings as Chapter 3 -> the SAME 220 test rows -> a fair head-to-head comparison at the end
)
print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")   # 876 / 220

# NOTE what we did NOT do: no feature scaling (StandardScaler etc.). A tree asks questions like "LOSS <= 0.5?",
# and that question works the same whatever units LOSS is in -- scaling changes nothing for a tree (theory.md Section 7).

# ===========================================================================
# STEP 5 -- Model Building
# ===========================================================================
# First, a DEFAULT tree -- no hyperparameters set. Default = max_depth=None, min_samples_leaf=1, i.e.
# "keep splitting until every leaf is pure". That's exactly the overfitting danger from theory.md Section 8.
# random_state=42: when two splits score identically, sklearn breaks the tie randomly -- fixing the seed makes results repeatable.
default_tree = DecisionTreeClassifier(random_state=42)

# ===========================================================================
# STEP 6 -- Model Training
# ===========================================================================
# ---- 6a. Train the default (untuned) tree -------------------------------------------------------------
default_tree.fit(X_train, y_train)

train_acc = accuracy_score(y_train, default_tree.predict(X_train))
test_acc = accuracy_score(y_test, default_tree.predict(X_test))
print("\nSTEP 6a -- DEFAULT tree (no limits):")
print(f"  depth = {default_tree.get_depth()}, leaves = {default_tree.get_n_leaves()}")   # depth 28, 304 leaves!
print(f"  train accuracy = {train_acc:.3f}")   # 0.995 -- it has nearly MEMORIZED the training data
print(f"  test  accuracy = {test_acc:.3f}")    # 0.655 -- ...and it does WORSE than Logistic Regression (0.673) on new data
# 304 leaves for 876 training rows = about 3 rows per rule. That's not learning a pattern, it's memorizing
# individual claims. Huge train/test gap = HIGH VARIANCE = overfitting (theory.md Section 11).

# ---- 6b. Tune it with GridSearchCV ---------------------------------------------------------------------
# The HYPERPARAMETERS (settings WE choose, not things the model learns -- theory.md Section 12) we want to tune:
param_grid = {
    "criterion": ["gini", "entropy"],             # the purity measure (theory.md Sections 3-4)
    "max_depth": [2, 3, 4, 5, 6, 8, None],        # max number of questions from root to leaf (None = unlimited)
    "min_samples_leaf": [1, 5, 10, 20, 50],       # every rule must be backed by at least this many training claims
}
# 2 x 7 x 5 = 70 combinations. With cv=5, each combination is trained 5 times -> 350 trees in total.

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),   # the model "template" -- GridSearchCV fills in each combination
    param_grid=param_grid,                               # the dictionary above -- every combination gets tried
    cv=5,                # 5-fold cross-validation, done ONLY inside the 876 training rows (theory.md Section 13):
                         # split train into 5 chunks, train on 4 and score on the 5th, rotate 5 times, average the 5 scores
    scoring="roc_auc",   # which number decides "best". ROC-AUC (Chapter 4) judges how well the model RANKS claims by risk,
                         # across every threshold -- more informative than accuracy at a single 0.5 cutoff
    n_jobs=-1,           # use every CPU core in parallel -- purely a speed setting, doesn't change the result
)
grid_search.fit(X_train, y_train)
# .fit() here runs all 350 trainings, picks the combination with the best AVERAGE cross-validation score, then
# (because refit=True by default) retrains ONE final tree with those settings on ALL 876 training rows.
# THE TEST SET HAS STILL NOT BEEN TOUCHED -- that's the whole point.

print("\nSTEP 6b -- GridSearchCV:")
print(f"  combinations tried: {len(grid_search.cv_results_['params'])}")   # 70
print(f"  best hyperparameters: {grid_search.best_params_}")
# {'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 50}
print(f"  best mean 5-fold CV ROC-AUC: {grid_search.best_score_:.3f}")    # 0.742

# A peek at the leaderboard -- cv_results_ holds the score of EVERY combination tried
results = pd.DataFrame(grid_search.cv_results_)
top5 = results.sort_values("rank_test_score")[
    ["param_criterion", "param_max_depth", "param_min_samples_leaf", "mean_test_score"]
].head(5)
# (in cv_results_, "test_score" means the score on the held-out FOLD during cross-validation -- NOT our real test set)
print("\n  Top 5 combinations by mean CV ROC-AUC:")
print(top5.to_string(index=False))
# entropy / depth 4 / leaf 50 -> 0.7421,  depth 5 -> 0.7413,  depth None, 8, 6 -> 0.7412 (all three tied)
# Lesson from the leaderboard: ALL top 5 have min_samples_leaf=50 -- that's the setting doing the real work here.
# Depth 6, 8 and None tie EXACTLY because "every leaf needs 50 claims" already stops the tree growing at
# depth 6 on its own -- so capping at 6, 8 or "unlimited" builds the identical tree. Once one hyperparameter
# is the limiting one, the other stops mattering. The scores are also very close
# (0.742 vs 0.741): past a point, tuning gives diminishing returns (theory.md Section 13).

best_tree = grid_search.best_estimator_   # the final tree, already retrained on all training rows with the best settings

# ===========================================================================
# STEP 7 -- Model Testing
# ===========================================================================
# ---- 7a. The Bias-Variance Tradeoff, made visible (theory.md Section 11) ----------------------------------
# Train trees of increasing depth and compare training accuracy vs cross-validated accuracy.
print("\nSTEP 7a -- Bias-Variance: accuracy vs max_depth")
print(f"  {'max_depth':>9} | {'train acc':>9} | {'5-fold CV acc':>13}")
for depth in [1, 2, 3, 4, 5, 6, 8, 10, 15, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    tr = accuracy_score(y_train, tree.predict(X_train))
    cv = cross_val_score(DecisionTreeClassifier(max_depth=depth, random_state=42),
                         X_train, y_train, cv=5).mean()
    # cross_val_score: the same 5-fold idea GridSearchCV uses internally, for ONE setting; .mean() averages the 5 fold scores
    print(f"  {str(depth):>9} | {tr:9.3f} | {cv:13.3f}")
# How to read the table:
#   depth 1-3   -> train ~0.72-0.74 and CV ~0.70: close together (low variance) but both stuck at the same ceiling.
#                  The model is too simple to capture more of the pattern = the HIGH-BIAS end.
#   depth 3-5   -> CV peaks around 0.71 = the sweet spot.
#   depth 8+    -> train keeps climbing (0.81 -> 0.995) while CV FALLS (0.66 -> 0.60). The gap widens every step
#                  = the HIGH-VARIANCE end: it's memorizing noise that doesn't repeat in unseen rows.

# ---- 7b. Predict on the untouched test set with the tuned tree -------------------------------------------
y_pred = best_tree.predict(X_test)               # final 0/1 prediction per test claim
y_proba = best_tree.predict_proba(X_test)[:, 1]  # probability of "Attorney" = the share of Attorney claims in the leaf it landed in

# ===========================================================================
# STEP 8 -- Model Evaluation
# ===========================================================================
tuned_train_acc = accuracy_score(y_train, best_tree.predict(X_train))
tuned_test_acc = accuracy_score(y_test, y_pred)
tuned_auc = roc_auc_score(y_test, y_proba)
default_auc = roc_auc_score(y_test, default_tree.predict_proba(X_test)[:, 1])
baseline_acc = (y_test == y_train.mode()[0]).mean()   # "always predict the majority class" -- the bar to beat

print("\nSTEP 8 -- Evaluation on the 220-row test set:")
print(f"  {'Model':<34}| {'train acc':>9} | {'test acc':>8} | {'ROC-AUC':>7}")
print(f"  {'Majority-class baseline':<34}| {'-':>9} | {baseline_acc:8.3f} | {'-':>7}")
print(f"  {'Logistic Regression (Chapter 3)':<34}| {'-':>9} | {0.673:8.3f} | {0.745:7.3f}")
print(f"  {'Decision Tree, default (untuned)':<34}| {train_acc:9.3f} | {test_acc:8.3f} | {default_auc:7.3f}")
print(f"  {'Decision Tree, GridSearchCV-tuned':<34}| {tuned_train_acc:9.3f} | {tuned_test_acc:8.3f} | {tuned_auc:7.3f}")
# Majority-class baseline      -> 0.527
# Logistic Regression          -> 0.673 acc, 0.745 AUC
# Default tree                 -> 0.995 train / 0.655 test, 0.654 AUC  -- overfit, loses to Logistic Regression
# Tuned tree                   -> 0.724 train / 0.732 test, 0.759 AUC  -- train ~= test (gap closed), beats Logistic Regression
# The SAME algorithm went from worst to best purely by choosing hyperparameters well.

print("\n  Confusion matrix (tuned tree) [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["No Attorney", "Attorney"]))

# Readable rules -- the thing a Decision Tree gives you that Logistic Regression can't (theory.md Section 5)
print(f"  Tuned tree: depth {best_tree.get_depth()}, {best_tree.get_n_leaves()} leaves. Its rules:")   # depth 4, 9 leaves
print(export_text(best_tree, feature_names=list(X.columns), show_weights=True))
# export_text prints the tree as indented IF/THEN rules. show_weights=True adds, per leaf, the count of training
# claims of each class that landed there: "weights: [No-Attorney count, Attorney count] class: <majority>".
# How to read THIS tree:
#   - The root question is "LOSS <= 0.99?" -- the single most useful question (biggest impurity drop, theory.md Section 3).
#       LOSS <= 0.99 -> every leaf says class 1 (Attorney);   LOSS > 0.99 -> every leaf says class 0 (No Attorney).
#   - So why the extra splits, if the label never changes? Because the leaves differ in HOW SURE they are: the
#     weights show different Attorney shares per leaf, which is exactly what predict_proba() returns (and what
#     ROC-AUC rewards -- that's why GridSearchCV, scoring on AUC, kept them).
#   - In this dataset, ATTORNEY=1 claims have much SMALLER losses (median LOSS 0.59 vs 3.19 for ATTORNEY=0) --
#     the tree didn't invent that direction, it read it straight off the data.

print("  Feature importance (share of total impurity reduction each feature contributed):")
for feature, importance in sorted(zip(X.columns, best_tree.feature_importances_), key=lambda p: -p[1]):
    print(f"    {feature:>9}: {importance:.3f}")
# LOSS 0.913, CLMAGE 0.069, CLMSEX 0.018, CLMINSUR 0.000, SEATBELT 0.000
# -> claim amount drives almost everything. CLMINSUR and SEATBELT were never used in a split: the same
#    "automatic feature selection" as Temperature in the Play Tennis tree (theory.md Section 5).
# (Importance says "the tree USED this feature a lot" -- association, not causation, same caveat as Chapter 3.)

# ===========================================================================
# STEP 9 -- Model Deployment
# ===========================================================================
joblib.dump(best_tree, "decision_tree_attorney_model.pkl")   # save the tuned tree (structure + thresholds) to one file
print("\nSTEP 9 -- saved decision_tree_attorney_model.pkl")

loaded_tree = joblib.load("decision_tree_attorney_model.pkl")   # simulate a separate app loading it fresh
new_claim = pd.DataFrame({"CLMSEX": [1], "CLMINSUR": [1], "SEATBELT": [0], "CLMAGE": [34], "LOSS": [2.5]})
# same columns, same order as training -- required for .predict()
print(f"  New claim -> predicted ATTORNEY = {loaded_tree.predict(new_claim)[0]}, "
      f"P(attorney) = {loaded_tree.predict_proba(new_claim)[0, 1]:.3f}")
# Wiring this into a real claims system / API is typically where the DS work hands off to ML/Software Engineering.

# ===========================================================================
# STEP 10 -- CI/CD (conceptual only, not runnable here)
# ===========================================================================
# Claim patterns drift over time (new laws, inflation pushing LOSS values up, changing attorney advertising).
# In production, a pipeline would periodically re-run Steps 2-8 on fresh data -- INCLUDING re-running GridSearchCV,
# since the best max_depth / min_samples_leaf can change as the data changes -- check the new model beats the current
# one on held-out data, and only then ship the new .pkl automatically.
