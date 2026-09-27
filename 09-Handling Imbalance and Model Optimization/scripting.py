"""
Handling Imbalance and Model Optimization -- on sklearn's breast cancer dataset

Run this file directly: python scripting.py        (needs: scikit-learn, pandas, imbalanced-learn)

Standalone -- no other file needed. Continues the course's "Handling Imbalance Dataset Problem" notebook
(reference only, not kept in this folder): same data, same split
(test_size=0.20, random_state=12, stratify=y), same starting model (Decision Tree, max_depth=3,
class_weight={0:2, 1:1}) -- then adds the imbalance techniques and optimization steps the course mentions
plus a few it doesn't, each with real numbers. theory.md explains the WHY behind each section.

IMPORTANT label detail: in this dataset 0 = malignant (cancer), 1 = benign.
The class we care about catching is 0 -- so every "recall" below is computed with pos_label=0.

This file is meant to be self-explanatory through its comments -- read top to bottom.
"""

import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, make_scorer,
)
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
# imblearn's Pipeline is like sklearn's, but it allows a RESAMPLING step -- and it applies that step
# ONLY while fitting (training data), never when predicting (test data). That's exactly what we need.

warnings.filterwarnings("ignore")


def report(name, model, X_tr, y_tr, X_te, y_te, threshold=None):
    """Fit, predict on the test set, print the numbers that matter for a cancer screen."""
    model.fit(X_tr, y_tr)
    if threshold is None:
        pred = model.predict(X_te)                       # default: whichever class has probability >= 0.5
    else:
        p_malignant = model.predict_proba(X_te)[:, 0]     # column 0 = probability of class 0 (malignant)
        pred = np.where(p_malignant >= threshold, 0, 1)   # call it malignant above OUR chosen threshold
    cm = confusion_matrix(y_te, pred, labels=[0, 1])      # rows = actual [malignant, benign]
    print(f"  {name:<42} acc {accuracy_score(y_te, pred):.3f} | "
          f"malignant recall {recall_score(y_te, pred, pos_label=0):.3f} | "
          f"precision {precision_score(y_te, pred, pos_label=0, zero_division=0):.3f} | "
          f"missed cancers {cm[0, 1]:>2} | false alarms {cm[1, 0]:>2}")
    return model


# ===========================================================================
# STEP 1 -- Business Understanding
# ===========================================================================
# A screening model flags tumours as malignant (cancer) or benign. The two mistakes are NOT equally bad:
#   missed cancer (malignant predicted benign)  -> the patient goes home untreated      -> very costly
#   false alarm   (benign predicted malignant)  -> an extra biopsy, anxiety, cost        -> bad, but recoverable
# So the metric to protect is MALIGNANT RECALL (Chapter 4), not accuracy.

# ===========================================================================
# STEP 2 -- Data Collection   /   STEP 3 -- Data Understanding
# ===========================================================================
data = datasets.load_breast_cancer()   # built into sklearn -- 569 tumours, 30 numeric measurements each
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="Target")
print("Classes:", dict(enumerate(data.target_names.tolist())))      # {0: 'malignant', 1: 'benign'}
print("Counts :", y.value_counts().to_dict())               # {1: 357, 0: 212}
print(f"Share  : malignant {(y == 0).mean():.0%}, benign {(y == 1).mean():.0%}")   # 37% / 63%
# 63/37 = a MILD imbalance. (Fraud or rare-disease data is often 99/1 -- Section 7 shows that case.)

# ===========================================================================
# STEP 4 -- Data Preparation: the split (technique #1: stratify)
# ===========================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=12, shuffle=True, stratify=y   # exactly the course's split
)
print(f"\nSTEP 4 -- train {len(X_train)} / test {len(X_test)}")   # 455 / 114
print(f"  malignant share: train {(y_train == 0).mean():.3f}, test {(y_test == 0).mean():.3f}")   # 0.374 / 0.368
# stratify=y keeps the 37/63 ratio identical in train and test. It does NOT fix the imbalance --
# it makes sure the imbalance is the SAME on both sides, so the test score is fair (Chapter 7, Section 5).

# ===========================================================================
# STEP 5-8 -- Model Building, Training, Testing, Evaluation
# ===========================================================================
print("\nSTEP 5-8 -- test-set results (42 malignant, 72 benign in the test set)")


def tree(class_weight=None):
    # random_state=42: the course's tree has NO random_state, so its results can change between runs
    # (sklearn breaks ties between equally good splits randomly). Fixing it makes every comparison fair.
    return DecisionTreeClassifier(criterion="gini", max_depth=3, class_weight=class_weight, random_state=42)


# ---- 5.1 Baseline vs the course's model ------------------------------------------------------------
print(" 5.1 Baseline and the course's model")
report("Always predict 'benign' (dummy)", DummyClassifier(strategy="most_frequent"), X_train, y_train, X_test, y_test)
# acc 0.632, malignant recall 0.000 -- 63% accuracy while catching ZERO cancers: the accuracy trap
report("Tree, no class weight", tree(), X_train, y_train, X_test, y_test)
# acc 0.877 | recall 0.905 | precision 0.792 | missed 4 | false alarms 10
report("Tree, class_weight={0:2,1:1} (course)", tree({0: 2, 1: 1}), X_train, y_train, X_test, y_test)
# acc 0.904 | recall 0.905 | precision 0.844 | missed 4 | false alarms 7
# -> reproduces the course's test confusion matrix exactly: [[38 4] [7 65]]

# ---- 5.2 Technique #2: class_weight -------------------------------------------------------------------
print(" 5.2 class_weight -- make mistakes on the rare class cost more")
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
print(f"  'balanced' formula n_samples / (n_classes * n_class): malignant {weights[0]:.3f}, benign {weights[1]:.3f}")
# 455 / (2 x 170) = 1.338 for malignant, 455 / (2 x 285) = 0.798 for benign -> ratio ~1.68 : 1
report("Tree, class_weight='balanced'", tree("balanced"), X_train, y_train, X_test, y_test)
report("Tree, class_weight={0:5,1:1}", tree({0: 5, 1: 1}), X_train, y_train, X_test, y_test)
# balanced: recall 0.881, missed 5, false alarms 6   |   {0:5}: recall 0.929, missed 3, false alarms 6
# The weight is a DIAL, not a magic fix: heavier weight on malignant usually pushes recall up. On a 114-row test
# set, one tumour flipping is a 2.4-point move, so read these as directions, not precise truths.

# ---- 5.3 Technique #3: resampling (imbalanced-learn) -- on TRAINING data only ------------------------------
print(" 5.3 Resampling the training data")
for name, sampler in [("Random OVERsampling (copy minority rows)", RandomOverSampler(random_state=42)),
                      ("Random UNDERsampling (drop majority rows)", RandomUnderSampler(random_state=42)),
                      ("SMOTE (synthesize new minority rows)", SMOTE(random_state=42))]:
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    print(f"  {name}: train counts {y_train.value_counts().to_dict()} -> {y_res.value_counts().to_dict()}")
    report("  -> Tree on resampled train", ImbPipeline([("resample", sampler), ("model", tree())]),
           X_train, y_train, X_test, y_test)
# over: {1: 285, 0: 170} -> {1: 285, 0: 285}   recall 0.881, missed 5, false alarms 6
# under: -> {1: 170, 0: 170}                  recall 0.929, missed 3, false alarms 8
# SMOTE: -> {1: 285, 0: 285}                  recall 0.881, missed 5, false alarms 6
# All land in the same neighbourhood as class_weight. On a MILD imbalance there isn't much to fix.

# ---- 5.4 Technique #4: threshold moving ---------------------------------------------------------------
print(" 5.4 Threshold moving -- call it malignant at a LOWER probability")
log_reg = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=5000))])
for t in [0.5, 0.3, 0.2, 0.1]:
    report(f"LogReg, malignant if P(malignant) >= {t}", log_reg, X_train, y_train, X_test, y_test, threshold=t)
# 0.5: recall 0.976, missed 1, false alarms 0
# 0.3: recall 0.976, missed 1, false alarms 3
# 0.2: recall 0.976, missed 1, false alarms 6
# 0.1: recall 1.000, missed 0, false alarms 10   <- every cancer caught, at the cost of 10 extra biopsies
# The precision-recall tradeoff (Chapter 4) as a business dial: a screening test would pick a low threshold.
# Choose the threshold on VALIDATION data / CV in a real project, not by peeking at the test set like this demo.

# ---- 5.5 Model optimization #1: scaling -- does it help? -----------------------------------------------
print(" 5.5 Scaling (the course's optimization step)")
report("Tree {0:2,1:1} + MinMaxScaler (fit on train)",
       Pipeline([("scale", MinMaxScaler()), ("model", tree({0: 2, 1: 1}))]), X_train, y_train, X_test, y_test)
# acc 0.904 | missed 4 | false alarms 7 -- IDENTICAL to the unscaled tree. Trees split on "feature <= threshold",
# and scaling doesn't change the order of values, so it can't change a tree (Chapter 8, Section 7).
# (The course's scaled tree gave slightly different numbers only because its tree had no random_state.)
# Scaling DOES matter for Logistic Regression -- which is exactly the model that won in 5.4 (see 5.6).

# ---- 5.6 Model optimization #2: tune hyperparameters with GridSearchCV, scored on what matters ----------
print(" 5.6 GridSearchCV, scoring = F1 of the malignant class")
f1_malignant = make_scorer(f1_score, pos_label=0)   # default 'f1' would score class 1 (benign) -- the wrong class
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tree_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    {"max_depth": [2, 3, 4, 5, None], "min_samples_leaf": [1, 5, 10],
     "class_weight": [None, "balanced", {0: 2, 1: 1}, {0: 5, 1: 1}]},   # class_weight is tunable too!
    cv=cv, scoring=f1_malignant,
).fit(X_train, y_train)
print(f"  Tree   best: {tree_search.best_params_}, CV F1 {tree_search.best_score_:.3f}")
# {'class_weight': {0: 2, 1: 1}, 'max_depth': 4, 'min_samples_leaf': 5}, CV F1 0.911
report("Tree, GridSearchCV-tuned", tree_search.best_estimator_, X_train, y_train, X_test, y_test)
# missed 4, false alarms 8 -- tuning the tree barely helps: the tree itself is the ceiling here.

lr_search = GridSearchCV(
    Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=5000))]),
    {"model__C": [0.01, 0.1, 1, 10, 100], "model__class_weight": [None, "balanced"]},
    # "model__C": the <step name>__<parameter> syntax reaches INSIDE a pipeline to tune one step's setting
    cv=cv, scoring=f1_malignant,
).fit(X_train, y_train)
print(f"  LogReg best: {lr_search.best_params_}, CV F1 {lr_search.best_score_:.3f}")
# {'model__C': 0.1, 'model__class_weight': 'balanced'}, CV F1 0.959
best_model = report("LogReg (scaled), GridSearchCV-tuned", lr_search.best_estimator_, X_train, y_train, X_test, y_test)
# acc 0.991 | recall 0.976 | missed 1 | false alarms 0
# THE biggest optimization in this file wasn't a resampling trick -- it was choosing a model that suits the data
# (30 smooth numeric measurements -> a linear boundary works well) and scaling it properly.

# ===========================================================================
# Section 7 -- What SEVERE imbalance looks like (the case these techniques were invented for)
# ===========================================================================
print("\nSEVERE IMBALANCE -- keep only 20 of the 212 malignant tumours (a rare-disease scenario)")
rng = np.random.RandomState(0)
keep = np.concatenate([rng.choice(y[y == 0].index, 20, replace=False), y[y == 1].index])
Xs, ys = X.loc[keep], y.loc[keep]
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xs, ys, test_size=0.3, random_state=12, stratify=ys)
print(f"  counts {ys.value_counts().to_dict()} -> {(ys == 0).mean():.1%} malignant; test set has {(ys_te == 0).sum()} malignant")
# {1: 357, 0: 20} -> 5.3% malignant; only 6 malignant in the test set
report("Always 'benign' (dummy)", DummyClassifier(strategy="most_frequent"), Xs_tr, ys_tr, Xs_te, ys_te)
# acc 0.947, recall 0.000 -- 94.7% "accurate" and useless. This is why accuracy is banned for imbalanced problems.
report("LogReg, no weighting", Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=5000))]),
       Xs_tr, ys_tr, Xs_te, ys_te)
report("LogReg, class_weight='balanced'",
       Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
       Xs_tr, ys_tr, Xs_te, ys_te)
# no weighting: missed 1 of 6 | balanced: missed 0 of 6. With only 6 positives, each one is ~17 points of recall --
# which is itself the lesson: with a rare class, the TEST SET is tiny too. Use repeated stratified CV (Chapter 7).

# The #1 resampling mistake: oversampling BEFORE splitting
Xo, yo = RandomOverSampler(random_state=42).fit_resample(Xs, ys)   # copies of the 20 malignant rows, BEFORE the split
Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(Xo, yo, test_size=0.3, random_state=12, stratify=yo)
report("WRONG: oversample, then split", DecisionTreeClassifier(random_state=42), Xo_tr, yo_tr, Xo_te, yo_te)
report("RIGHT: split, then oversample train only",
       ImbPipeline([("resample", RandomOverSampler(random_state=42)), ("m", DecisionTreeClassifier(random_state=42))]),
       Xs_tr, ys_tr, Xs_te, ys_te)
# WRONG: recall 1.000, a "perfect" model   |   RIGHT: recall 0.833
# Oversampling first puts COPIES of the same malignant tumour in both train and test -- the test just checks
# whether the tree memorized rows it has already seen. The perfect score is fake.

# ===========================================================================
# STEP 9 -- Model Deployment (fixing the course's pickle step)
# ===========================================================================
# The course pickles `scaled_X` -- the scaled DATA table -- as "data_transformation.pkl". That file can't transform
# a new patient's measurements; what production needs is the FITTED SCALER (the learned min/max per column).
# Safest of all: save ONE pipeline that holds the scaler + model together, so they can never get out of sync.
joblib.dump(best_model, "breast_cancer_pipeline.pkl")
loaded = joblib.load("breast_cancer_pipeline.pkl")
new_patient = X_test.iloc[[0]]                          # pretend this row just arrived from the lab, unscaled
print(f"\nSTEP 9 -- saved breast_cancer_pipeline.pkl; new patient -> "
      f"{data.target_names[loaded.predict(new_patient)[0]]}, P(malignant) = {loaded.predict_proba(new_patient)[0, 0]:.3f}")

# ===========================================================================
# STEP 10 -- CI/CD (conceptual)
# ===========================================================================
# Class ratios drift (a new screening programme finds more early, benign cases). Monitor the live class ratio and
# the malignant recall; if either moves, re-tune class_weight / the threshold on fresh data before re-shipping.
