"""
Ensemble Techniques -- Voting, Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost, Stacking
on assets/claim.csv

Run this file directly: python scripting.py
(from inside this chapter's folder, since it uses a relative path to ../assets/claim.csv)
Needs: scikit-learn, pandas, joblib, xgboost  (xgboost on Mac also needs `brew install libomp`)

Same business question, same cleanup and the SAME train/test split as Chapters 3 and 5:
    "Will this insurance claimant hire an attorney?"  (ATTORNEY: 0 = no, 1 = yes)
So every model here is scored on the identical 220 test claims -- a fair leaderboard against
Logistic Regression (Ch3) and the GridSearchCV-tuned Decision Tree (Ch5).

Structured around the course's 10-step ML pipeline. Steps 5-8 hold one sub-section per ensemble
technique, each pointing at the matching section of theory.md.

This file is meant to be self-explanatory through its comments -- read top to bottom.
"""

import warnings
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    VotingClassifier, BaggingClassifier, RandomForestClassifier,
    AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier,
)
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier   # XGBoost is its own library (not part of sklearn) but follows the same .fit()/.predict() API

warnings.filterwarnings("ignore")   # hides harmless library notices (e.g. convergence/deprecation) so the output stays readable

# ===========================================================================
# STEP 1 -- Business Understanding
# ===========================================================================
# Same as Chapters 3 and 5: flag claims likely to involve an attorney as soon as they arrive, so they can be
# routed to experienced adjusters. The question for THIS chapter: can a TEAM of models beat the best single
# model we have so far (the tuned Decision Tree: test accuracy 0.732, ROC-AUC 0.759)?

# ===========================================================================
# STEP 2 -- Data Collection
# ===========================================================================
df = pd.read_csv("../assets/claim.csv")
print("STEP 2 -- raw data shape:", df.shape)   # (1340, 7)

# ===========================================================================
# STEP 3 -- Data Understanding
# ===========================================================================
# Already explored in Chapters 3 and 5: 4 columns have missing values, target roughly balanced (685 / 655),
# and LOSS carries most of the signal (Chapter 5 feature importance: 0.913). Nothing new to learn here.
print("Target balance:", df["ATTORNEY"].value_counts().to_dict())   # {0: 685, 1: 655}

# ===========================================================================
# STEP 4 -- Data Preparation
# ===========================================================================
df = df.drop(columns=["CASENUM"]).dropna()   # drop the ID column + incomplete rows: 1340 -> 1096 (same as Ch3/Ch5)
X = df.drop(columns=["ATTORNEY"])
y = df["ATTORNEY"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # identical split -> identical 220 test rows as Ch3/Ch5
)
print(f"STEP 4 -- train rows: {len(X_train)}, test rows: {len(X_test)}")   # 876 / 220

# ---------------------------------------------------------------------------
# A small helper so every model is scored the same way (used for the leaderboard at the end)
# ---------------------------------------------------------------------------
leaderboard = []   # one row per model: name, train acc, test acc, test ROC-AUC


def evaluate(name, model):
    """Fit on the training rows, score on train and test, print one line, store it for the leaderboard."""
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    # ROC-AUC needs probabilities. A hard-voting ensemble only outputs votes (0/1), so it has no AUC.
    has_proba = hasattr(model, "predict_proba") and getattr(model, "voting", "soft") != "hard"
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if has_proba else None
    auc_text = f"{auc:.3f}" if auc is not None else "  -  "
    print(f"  {name:<44} train {train_acc:.3f} | test {test_acc:.3f} | AUC {auc_text}")
    leaderboard.append({"model": name, "train_acc": train_acc, "test_acc": test_acc, "test_auc": auc})
    return model


# ===========================================================================
# STEP 5 -- Model Building   (+ STEP 6 Training and STEP 7 Testing, one ensemble technique at a time)
# ===========================================================================

# ---- 5.0 The single-model baselines from earlier chapters -------------------------------------------
# Every ensemble below must beat THESE to be worth its extra complexity.
print("\n5.0 -- single-model baselines")
log_reg = LogisticRegression(max_iter=1000)   # Chapter 3
tuned_tree = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=50, random_state=42)
# ^ the exact hyperparameters GridSearchCV picked in Chapter 5
naive_bayes = GaussianNB()   # a deliberately WEAK third model, used to show what happens when one voter is bad

evaluate("Logistic Regression (Ch3)", log_reg)                                   # 0.673 test, AUC 0.745
evaluate("Decision Tree, default (Ch5)", DecisionTreeClassifier(random_state=42))  # 0.995 train / 0.655 test -- overfit
evaluate("Decision Tree, tuned (Ch5)", tuned_tree)                               # 0.732 test, AUC 0.759 -- the bar to beat
evaluate("Gaussian Naive Bayes (weak voter)", naive_bayes)                       # 0.555 test -- barely above guessing
# (GaussianNB = a simple probability-based classifier, a later chapter. Here it only plays "the weak team member".)

# ---- 5.1 VOTING (theory.md Section 3) -------------------------------------------------------------------
# Train several DIFFERENT algorithms on the same data, then let them vote.
#   voting="hard" -> each model casts one 0/1 vote, the majority wins
#   voting="soft" -> average the models' PROBABILITIES, then threshold at 0.5 (confident models count more)
print("\n5.1 -- Voting")
evaluate("Voting HARD (LogReg + Tree + NaiveBayes)",
         VotingClassifier([("lr", log_reg), ("dt", tuned_tree), ("nb", naive_bayes)], voting="hard"))
evaluate("Voting SOFT (LogReg + Tree + NaiveBayes)",
         VotingClassifier([("lr", log_reg), ("dt", tuned_tree), ("nb", naive_bayes)], voting="soft"))
# hard 0.673 ; soft 0.595 / AUC 0.740 -- WORSE than the tuned tree alone.
# Lesson: Naive Bayes is a bad voter here, and in SOFT voting its badly calibrated probabilities pull the average
# the wrong way. A team is only as good as its members -- "more models" is not automatically "better".

small_forest = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=10, random_state=42)
evaluate("Voting HARD (LogReg + Tree + RandomForest)",
         VotingClassifier([("lr", log_reg), ("dt", tuned_tree), ("rf", small_forest)], voting="hard"))
evaluate("Voting SOFT (LogReg + Tree + RandomForest)",
         VotingClassifier([("lr", log_reg), ("dt", tuned_tree), ("rf", small_forest)], voting="soft"))
# hard 0.732 ; soft 0.709 / AUC 0.772 -- swapping the weak member for a strong one fixes it,
# and soft voting's AUC (0.772) now beats every single model above.

# Why is the gain small? Ensembles help most when members make DIFFERENT mistakes (theory.md Section 1).
lr_pred = log_reg.fit(X_train, y_train).predict(X_test)
dt_pred = tuned_tree.fit(X_train, y_train).predict(X_test)
print(f"  LogReg and Tree give the SAME answer on {(lr_pred == dt_pred).mean():.1%} of test claims")   # 82.3%
# 82% agreement = highly correlated voters. When they're wrong, they're mostly wrong together, so a vote can't fix it.

# ---- 5.2 BAGGING (theory.md Section 4) ------------------------------------------------------------------
# Bootstrap Aggregating: train MANY copies of the SAME algorithm, each on a different random sample of the
# training rows drawn WITH replacement (a "bootstrap sample"), then average their predictions.
# It targets VARIANCE -- so we deliberately give it the most overfit model we have: the default, unlimited tree.
print("\n5.2 -- Bagging")
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),  # the base model: an UNLIMITED tree (high variance on its own)
    n_estimators=100,     # 100 trees, each trained on its own bootstrap sample of the 876 training rows
    oob_score=True,       # score each tree on the rows it did NOT see (~37% of rows) -- a free validation score
    random_state=42,
)
evaluate("Bagging (100 unlimited trees)", bagging)
print(f"  Out-of-bag (OOB) accuracy: {bagging.oob_score_:.3f}")   # 0.686
# Single unlimited tree: test 0.655, AUC 0.654.  100 bagged unlimited trees: test 0.659, AUC 0.724.
# Accuracy barely moved, but AUC jumped +0.07: averaging 100 jumpy trees gives much SMOOTHER, better-ranked
# probabilities. Each tree still memorizes (train 0.995) -- bagging reduces variance, it doesn't remove it.
# The OOB score (0.686) is an honest estimate WITHOUT touching the test set: each row is judged only by the
# trees that never trained on it.

# ---- 5.3 RANDOM FOREST (theory.md Section 5) ------------------------------------------------------------
# Random Forest = Bagging of trees + one extra trick: at EVERY split, each tree may only consider a random
# subset of the features (max_features="sqrt" -> sqrt(5) ~ 2 of our 5 columns). That stops every tree from
# asking the same "LOSS <= ..." question first, so the trees disagree more -> the average cancels more noise.
print("\n5.3 -- Random Forest")
evaluate("Random Forest, default (unlimited trees)", RandomForestClassifier(n_estimators=100, random_state=42))
# train 0.995 / test 0.645 -- even a forest overfits when every tree is unlimited on noisy data.
# "Random Forest doesn't overfit" is a myth -- MORE TREES doesn't overfit, but DEEP trees still can.

print("  Effect of n_estimators (number of trees), unlimited trees:")
for n_trees in [1, 5, 10, 50, 100, 300]:
    forest = RandomForestClassifier(n_estimators=n_trees, random_state=42).fit(X_train, y_train)
    auc = roc_auc_score(y_test, forest.predict_proba(X_test)[:, 1])
    print(f"    {n_trees:>3} trees -> test AUC {auc:.3f}")
# 1 -> 0.621, 5 -> 0.691, 10 -> 0.700, 50 -> 0.714, 100 -> 0.714, 300 -> 0.713
# Big gains from the first few dozen trees, then a PLATEAU. Adding trees never makes it worse, just slower.

# Tune the tree-shape hyperparameters with GridSearchCV (Chapter 5, Section 13 -- same recipe)
rf_grid = GridSearchCV(
    RandomForestClassifier(n_estimators=200, random_state=42),
    param_grid={
        "max_depth": [3, 5, 8, None],
        "min_samples_leaf": [1, 10, 20, 50],
        "max_features": ["sqrt", None],   # None = consider ALL features at every split (= plain bagging of trees)
    },
    cv=5, scoring="roc_auc", n_jobs=-1,
)   # 4 x 4 x 2 = 32 combinations x 5 folds = 160 forests x 200 trees each
rf_grid.fit(X_train, y_train)
print(f"  GridSearchCV best: {rf_grid.best_params_}, CV AUC {rf_grid.best_score_:.3f}")
# {'max_depth': 3, 'max_features': 'sqrt', 'min_samples_leaf': 10}, CV AUC 0.756
# Note max_features="sqrt" beat None: the feature randomness (the thing that makes it a FOREST, not just bagging) helped.
best_forest = evaluate("Random Forest, GridSearchCV-tuned", rf_grid.best_estimator_)   # 0.732 test, AUC 0.768

print("  Feature importance (averaged over all 200 trees):")
for feature, imp in sorted(zip(X.columns, best_forest.feature_importances_), key=lambda p: -p[1]):
    print(f"    {feature:>9}: {imp:.3f}")
# LOSS 0.783, CLMAGE 0.109, CLMINSUR 0.062, CLMSEX 0.044, SEATBELT 0.002
# LOSS still dominates, but less completely than in Chapter 5's single tree (0.913): with max_features="sqrt",
# some trees can't see LOSS at a given split and are forced to learn from the other columns.

# ---- 5.4 ADABOOST (theory.md Section 7) -----------------------------------------------------------------
# BOOSTING trains models ONE AFTER ANOTHER, each focusing on the previous ones' mistakes. It targets BIAS.
# AdaBoost's default weak learner is a "stump" (a tree with max_depth=1 -- ONE question). After each stump,
# the rows it got wrong get MORE weight, so the next stump is forced to care about them.
print("\n5.4 -- AdaBoost")
evaluate("AdaBoost (100 stumps)", AdaBoostClassifier(n_estimators=100, random_state=42))
# train 0.729 / test 0.732 / AUC 0.773 -- 100 one-question stumps combine to beat the tuned 4-level tree's AUC (0.759),
# with almost no train/test gap. Each stump alone is weak (a single depth-1 tree: CV acc ~0.70 in Chapter 5).

# ---- 5.5 GRADIENT BOOSTING (theory.md Section 8) ---------------------------------------------------------
# Each new small tree is trained to predict the RESIDUAL ERROR of all the trees so far, and its prediction is
# added in, scaled down by the learning_rate. "Keep correcting what's still wrong."
print("\n5.5 -- Gradient Boosting")
evaluate("Gradient Boosting, default", GradientBoostingClassifier(random_state=42))
# defaults: 100 trees, depth 3, learning_rate 0.1 -> train 0.796 / test 0.691 -- already starting to overfit
gboost = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=2, random_state=42)
gboost.fit(X_train, y_train)

# staged_predict_proba gives the prediction after 1 tree, after 2 trees, ... -- so we can WATCH boosting learn
print("  AUC as trees are added (smaller trees, learning_rate=0.05):")
train_stages = list(gboost.staged_predict_proba(X_train))
test_stages = list(gboost.staged_predict_proba(X_test))
for n_trees in [1, 10, 50, 100, 200, 300]:
    tr = roc_auc_score(y_train, train_stages[n_trees - 1][:, 1])
    te = roc_auc_score(y_test, test_stages[n_trees - 1][:, 1])
    print(f"    {n_trees:>3} trees -> train AUC {tr:.3f} | test AUC {te:.3f}")
# train: 0.757 -> 0.776 -> 0.796 -> 0.813 -> 0.834 -> 0.854   (keeps climbing forever)
# test : 0.763 -> 0.768 -> 0.768 -> 0.771 -> 0.760 -> 0.751   (peaks around 100 trees, then FALLS)
# Unlike Random Forest, boosting CAN overfit by adding trees -- each new tree chases smaller and smaller
# residuals, which eventually are just noise. n_estimators is a real hyperparameter to tune for boosting.

# ---- 5.6 XGBOOST (theory.md Section 9) ------------------------------------------------------------------
# XGBoost = Gradient Boosting engineered for speed and with built-in regularization. Same idea as 5.5.
print("\n5.6 -- XGBoost")
evaluate("XGBoost, default", XGBClassifier(random_state=42, eval_metric="logloss"))
# defaults: 100 trees, depth 6 -> train 0.943 / test 0.655 -- deep trees + many rounds = overfit on this small dataset
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss"),   # eval_metric: which loss XGBoost reports while training
    param_grid={
        "n_estimators": [100, 300],          # number of boosting rounds (trees)
        "learning_rate": [0.01, 0.05, 0.1],  # how much each new tree's correction counts
        "max_depth": [2, 3, 4],              # boosting wants SMALL trees -- each one only a small correction
    },
    cv=5, scoring="roc_auc", n_jobs=-1,
)   # 2 x 3 x 3 = 18 combinations x 5 folds = 90 fits
xgb_grid.fit(X_train, y_train)
print(f"  GridSearchCV best: {xgb_grid.best_params_}, CV AUC {xgb_grid.best_score_:.3f}")
# {'learning_rate': 0.01, 'max_depth': 2, 'n_estimators': 300}, CV AUC 0.757
# Slow learning (0.01) + tiny trees (depth 2) + more rounds = the classic winning boosting recipe.
best_xgb = evaluate("XGBoost, GridSearchCV-tuned", xgb_grid.best_estimator_)   # 0.741 test, AUC 0.773

# ---- 5.7 STACKING (theory.md Section 10) ----------------------------------------------------------------
# Like soft voting, but instead of a plain average, a final "meta-model" LEARNS how much to trust each member.
print("\n5.7 -- Stacking")
stacking = StackingClassifier(
    estimators=[("lr", log_reg), ("dt", tuned_tree), ("nb", naive_bayes)],   # the base models (level 0)
    final_estimator=LogisticRegression(),   # the meta-model (level 1): learns weights for the 3 base models' outputs
    cv=5,   # base-model predictions used to train the meta-model are made OUT-OF-FOLD, so it never learns from
            # predictions on rows the base models already memorized (that would be leakage)
)
evaluate("Stacking (LogReg + Tree + NaiveBayes -> LogReg)", stacking)
# test 0.682 / AUC 0.770 -- the SAME three members that made soft voting collapse (0.595). The meta-model learned
# to trust Naive Bayes less, instead of giving it an equal 1/3 say. That's the whole point of stacking.

# ===========================================================================
# STEP 8 -- Model Evaluation
# ===========================================================================
print("\nSTEP 8 -- Leaderboard (same 220 test claims), sorted by test ROC-AUC:")
board = pd.DataFrame(leaderboard)
board["gap"] = board["train_acc"] - board["test_acc"]   # generalization gap (Chapter 5, Section 11)
board = board.sort_values("test_auc", ascending=False, na_position="last")
print(board.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
# Top of the board: tuned XGBoost (0.741 acc, 0.773 AUC), AdaBoost (0.732 / 0.773), soft voting with RF (0.772),
# stacking (0.770), tuned Random Forest (0.768). Bottom: the unlimited single tree and its un-tuned relatives.
#
# Honest takeaways:
#   1. The best ensembles beat the best single model, but only by ~0.01 AUC (0.759 -> 0.773). On a 5-column dataset
#      where LOSS carries most of the signal, there isn't much left to find -- that's the irreducible-error ceiling
#      from Chapter 5, Section 11. Ensembles shine more on wide datasets with many weak, interacting signals.
#   2. Tuning still matters more than the algorithm name: default XGBoost (AUC 0.701) lost to a tuned single tree.
#   3. With only 220 test rows, differences of ~0.01 are within noise. The 5-fold CV scores from GridSearchCV
#      (RF 0.756, XGB 0.757) are the more reliable comparison.

# ===========================================================================
# STEP 9 -- Model Deployment
# ===========================================================================
joblib.dump(best_xgb, "xgboost_attorney_model.pkl")   # the top model on the leaderboard
print("\nSTEP 9 -- saved xgboost_attorney_model.pkl")
loaded = joblib.load("xgboost_attorney_model.pkl")
new_claim = pd.DataFrame({"CLMSEX": [1], "CLMINSUR": [1], "SEATBELT": [0], "CLMAGE": [34], "LOSS": [2.5]})
print(f"  New claim -> predicted ATTORNEY = {loaded.predict(new_claim)[0]}, "
      f"P(attorney) = {loaded.predict_proba(new_claim)[0, 1]:.3f}")

# ===========================================================================
# STEP 10 -- CI/CD (conceptual only)
# ===========================================================================
# Same as Chapter 5, with one ensemble-specific note: an ensemble is harder to explain than one tree (there's no single
# set of IF/THEN rules to show a regulator). Production pipelines usually log feature importances (or SHAP values,
# a later topic) on every retrain, so a sudden shift in WHAT the model relies on gets caught before shipping.
