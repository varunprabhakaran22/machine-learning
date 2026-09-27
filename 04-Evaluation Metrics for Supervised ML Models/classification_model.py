"""
Classification Model Evaluation Metrics — a hands-on demo using assets/claim.csv

Run this file directly: python classification_model.py
(from inside this chapter's folder, since it uses a relative path to ../assets/claim.csv)

Trains the SAME Logistic Regression model as 03-Logistic Regression/scripting.ipynb
(claim.csv, predicting ATTORNEY), then walks through every metric from
"Classification Model.md" step by step, computing each one two ways where possible:
once via sklearn's built-in function, once by hand from the raw formula -- so you can
SEE that "the formula" and "the library call" are the exact same thing.

This file is meant to be self-explanatory through its comments -- read top to bottom.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve,
)

# ---------------------------------------------------------------------------
# 1. Rebuild the same trained model from Chapter 3 (needed before any metric makes sense)
# ---------------------------------------------------------------------------
df = pd.read_csv("../assets/claim.csv")
df = df.drop(columns=["CASENUM"]).dropna()   # drop the ID column, drop rows with missing values -- same cleanup as Chapter 3

X = df.drop(columns=["ATTORNEY"])   # features
y = df["ATTORNEY"]                   # target: 0 = no attorney, 1 = attorney

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # same split settings as Chapter 3 -- so we get the SAME test set, and comparable numbers
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)   # .fit() = training -- learns the m's and b that minimize Log Loss (theory.md Chapter 3)

y_pred = model.predict(X_test)               # the final 0/1 prediction per test row (threshold at 0.5 by default)
y_proba = model.predict_proba(X_test)[:, 1]    # the raw probability of class 1 ("Attorney") per test row, BEFORE thresholding

print("Model trained. Now evaluating on the 220-row held-out test set.\n")

# ---------------------------------------------------------------------------
# 2. The Confusion Matrix -- the foundation every other metric is built from
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
# confusion_matrix(actual, predicted) returns a 2x2 grid of counts. By default, sklearn orders it as:
#   [[TN, FP],
#    [FN, TP]]
# i.e. row = actual class (0 then 1), column = predicted class (0 then 1).

print("Confusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()   # .ravel() flattens the 2x2 grid into a flat 1D array of 4 numbers, in reading order: TN, FP, FN, TP
print(f"\nTN (True Negative)  = {tn}   -- correctly predicted 'No Attorney'")
print(f"FP (False Positive) = {fp}   -- wrongly predicted 'Attorney' (false alarm)")
print(f"FN (False Negative) = {fn}   -- wrongly predicted 'No Attorney' (missed a real case)")
print(f"TP (True Positive)  = {tp}   -- correctly predicted 'Attorney'")

assert tn + fp + fn + tp == len(y_test)   # sanity check -- these 4 counts must always add up to the full test set size
print(f"\nCheck: {tn}+{fp}+{fn}+{tp} = {tn+fp+fn+tp}  (matches test set size: {len(y_test)})")

# ---------------------------------------------------------------------------
# 3. Accuracy -- overall correctness
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ACCURACY -- 'what fraction of ALL predictions were correct?'")
print("=" * 60)

accuracy_sklearn = accuracy_score(y_test, y_pred)          # the library function -- does the arithmetic below for you, in one call
accuracy_by_hand = (tp + tn) / (tp + tn + fp + fn)           # the raw formula: (correct predictions) / (all predictions)

print(f"sklearn's accuracy_score(): {accuracy_sklearn:.4f}")
print(f"same number, by hand:       {accuracy_by_hand:.4f}")
assert round(accuracy_sklearn, 4) == round(accuracy_by_hand, 4)   # proves the library call and the formula agree exactly

# ---------------------------------------------------------------------------
# 4. Precision -- "when I said Yes, how often was I right?"
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PRECISION -- 'of everything I flagged as Attorney, how much was real?'")
print("=" * 60)

precision_sklearn = precision_score(y_test, y_pred)   # library function
precision_by_hand = tp / (tp + fp)                      # raw formula: TP / (TP + FP) -- only looks at the model's "Yes" predictions

print(f"sklearn's precision_score(): {precision_sklearn:.4f}")
print(f"same number, by hand:        {precision_by_hand:.4f}")
assert round(precision_sklearn, 4) == round(precision_by_hand, 4)

# ---------------------------------------------------------------------------
# 5. Recall -- "of everyone who really was Yes, how many did I catch?"
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RECALL -- 'of every real Attorney case, how many did I actually catch?'")
print("=" * 60)

recall_sklearn = recall_score(y_test, y_pred)   # library function
recall_by_hand = tp / (tp + fn)                   # raw formula: TP / (TP + FN) -- only looks at the real "Yes" cases

print(f"sklearn's recall_score(): {recall_sklearn:.4f}")
print(f"same number, by hand:     {recall_by_hand:.4f}")
assert round(recall_sklearn, 4) == round(recall_by_hand, 4)

# ---------------------------------------------------------------------------
# 6. F1-Score -- the harmonic-mean balance of Precision and Recall
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("F1-SCORE -- one number balancing Precision and Recall")
print("=" * 60)

f1_sklearn = f1_score(y_test, y_pred)   # library function
f1_by_hand = 2 * (precision_by_hand * recall_by_hand) / (precision_by_hand + recall_by_hand)
# raw formula: harmonic mean of precision and recall -- NOT a plain average. A harmonic mean punishes
# a big gap between the two numbers much harder than a plain average would.

print(f"sklearn's f1_score(): {f1_sklearn:.4f}")
print(f"same number, by hand:  {f1_by_hand:.4f}")
assert round(f1_sklearn, 4) == round(f1_by_hand, 4)

# ---------------------------------------------------------------------------
# 7. Precision vs. Recall tradeoff -- proven by changing the decision threshold
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("THE PRECISION/RECALL TRADEOFF -- same model, 3 different thresholds")
print("=" * 60)
print("model.predict() uses a FIXED 0.5 threshold internally. Here we build our own")
print("predictions at different thresholds directly from y_proba, to see the tradeoff.\n")

for threshold in [0.3, 0.5, 0.7]:
    # (y_proba >= threshold) makes a True/False array; .astype(int) converts True/False into 1/0
    y_pred_at_threshold = (y_proba >= threshold).astype(int)

    cm_t = confusion_matrix(y_test, y_pred_at_threshold)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()

    precision_t = precision_score(y_test, y_pred_at_threshold, zero_division=0)
    # zero_division=0: if a threshold is so extreme that the model NEVER predicts "Yes" (TP+FP=0),
    # precision would otherwise be an undefined 0/0 -- this tells sklearn to report 0 instead of erroring.
    recall_t = recall_score(y_test, y_pred_at_threshold, zero_division=0)

    print(f"threshold={threshold}:  TN={tn_t:>3} FP={fp_t:>3} FN={fn_t:>3} TP={tp_t:>3}"
          f"   precision={precision_t:.3f}   recall={recall_t:.3f}")

print("\nNotice: as the threshold goes UP, precision goes UP but recall goes DOWN.")
print("There is no single threshold that maximizes both at once -- it's always a tradeoff.")

# ---------------------------------------------------------------------------
# 8. ROC Curve -- the tradeoff above, but swept across EVERY possible threshold
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ROC CURVE -- False Positive Rate vs. True Positive Rate, at every threshold")
print("=" * 60)

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
# roc_curve(actual, probabilities) returns 3 same-length arrays: at each of many possible thresholds
# (chosen automatically from the data), it reports the FPR and TPR you'd get at that threshold.
# fpr = FP / (FP + TN)  -- "of real Nos, what fraction did I wrongly flag as Yes?"
# tpr = TP / (TP + FN)  -- this is the SAME formula as Recall, just renamed for this context.

print(f"Total threshold points computed: {len(thresholds)}")
print("\nA few sample points along the curve (threshold -> FPR, TPR):")

# pick 6 evenly-spaced points across the curve just to print a readable sample (not all ~94 of them)
import numpy as np
sample_idxs = np.linspace(0, len(thresholds) - 1, 6).astype(int)
for i in sample_idxs:
    t = thresholds[i]
    t_display = "inf (never says Yes)" if t == float("inf") else f"{t:.3f}"
    print(f"  threshold={t_display:>22}   FPR={fpr[i]:.3f}   TPR={tpr[i]:.3f}")

print("\nReading it: at a very high threshold, FPR and TPR are both ~0 (model almost never says Yes).")
print("At threshold=0, FPR and TPR are both 1.0 (model always says Yes). A useful model's curve")
print("bulges toward the top-left (high TPR, low FPR) somewhere in between those two extremes.")

# ---------------------------------------------------------------------------
# 9. AUC -- the entire ROC curve compressed into one number
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("AUC -- Area Under the ROC Curve, one number summarizing the whole curve")
print("=" * 60)

auc = roc_auc_score(y_test, y_proba)
# roc_auc_score(actual, PROBABILITIES) -- note it takes the raw probability, not the thresholded
# prediction, because AUC measures performance across ALL thresholds at once, not just one.

print(f"AUC = {auc:.4f}")
print("0.5 = no better than random guessing.  1.0 = perfect separation of the two classes.")
print(f"Our {auc:.3f}: meaningfully better than random, genuinely useful, but far from perfect.")

# ---------------------------------------------------------------------------
# 10. Everything side by side
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY -- all metrics together, at the default 0.5 threshold")
print("=" * 60)
print(f"{'Accuracy':<12}: {accuracy_sklearn:.3f}")
print(f"{'Precision':<12}: {precision_sklearn:.3f}")
print(f"{'Recall':<12}: {recall_sklearn:.3f}")
print(f"{'F1-Score':<12}: {f1_sklearn:.3f}")
print(f"{'AUC':<12}: {auc:.3f}")
print("\nNo single one of these tells the whole story on its own -- that's the entire point")
print("of this chapter. Look at the confusion matrix first, then pick the metric(s) that")
print("match which mistake (False Positive vs False Negative) actually costs more here.")
