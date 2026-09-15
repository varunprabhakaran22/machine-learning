"""
Train/Test Split — a hands-on demo using assets/claim.csv

Run this file directly: python train_test_split.py
(from inside this chapter's folder, since it uses a relative path to ../assets/claim.csv)

See train_test_split.md in this same folder for the short concept explanation.
This file is meant to be self-explanatory through its comments — read top to bottom.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------
df = pd.read_csv("../assets/claim.csv")   # pd.read_csv() reads the CSV straight into a DataFrame (a table, like an Excel sheet in code)

# Same cleanup as Chapter 3's scripting.ipynb -- CASENUM is an ID (not a feature),
# and a few columns have missing values that a model can't train on.
df = df.drop(columns=["CASENUM"])   # .drop(columns=[...]) returns a NEW DataFrame with that column removed
df = df.dropna()                     # .dropna() removes any ROW that has a missing (NaN) value in ANY remaining column

print(f"Rows after cleanup: {len(df)}")   # len() on a DataFrame returns its row count -> 1096

# ---------------------------------------------------------------------------
# 2. Separate features (X) from target (y)
# ---------------------------------------------------------------------------
X = df.drop(columns=["ATTORNEY"])   # X = every column EXCEPT the target -- what the model learns FROM
y = df["ATTORNEY"]                   # y = the target column -- what the model is trying to predict (0 or 1)

print(f"\nFeatures (X) columns: {list(X.columns)}")
print(f"Target (y): ATTORNEY -- 0 = no attorney, 1 = attorney")

# ---------------------------------------------------------------------------
# 3. The actual train/test split
# ---------------------------------------------------------------------------
# train_test_split() is sklearn's standard function for this. It takes X and y together
# (so rows stay correctly paired -- row 5's features always stay matched with row 5's
# target, even after shuffling) and randomly divides them into two separate groups.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,     # what FRACTION of rows go into the test set -- 0.2 = 20% test, 80% train.
                        # There's no single "correct" number -- 80/20 or 70/30 are the most common defaults.
                        # More test data -> more reliable evaluation, but less data left to actually train on. It's a tradeoff.
    random_state=42,   # the "seed" for the random shuffle. Without this, you'd get a DIFFERENT random split every time you
                        # rerun the script -- which makes results impossible to reproduce or compare run-to-run. Any fixed
                        # number works (42 is just a common convention in examples, not a special value) -- what matters is
                        # picking ONE number and keeping it, so "rerun this script" always gives the identical split.
    stratify=y,        # tells the function to keep the TARGET'S CLASS RATIO consistent between train and test.
                        # ATTORNEY is roughly 51% / 49% (0s vs 1s) in the full data -- without stratify, a random split
                        # COULD by chance put more 1s in train and more 0s in test (or vice versa), especially on smaller
                        # datasets, which would make the test-set evaluation unfairly easy or hard. stratify=y forces both
                        # train and test to keep that same ~51/49 ratio. This only makes sense for a CATEGORICAL target
                        # (classification) -- there's no "class ratio" to preserve for a continuous number like house price.
)
# train_test_split() ALWAYS returns exactly 4 things, in this exact order:
#   X_train, X_test, y_train, y_test  (features-train, features-test, target-train, target-test)

# ---------------------------------------------------------------------------
# 4. Confirm what we actually got
# ---------------------------------------------------------------------------
print(f"\nX_train shape: {X_train.shape}")   # (rows, columns) -- (876, 5): 876 claims to train on, 5 features each
print(f"X_test shape:  {X_test.shape}")      # (220, 5): 220 claims held back purely for evaluation
print(f"y_train shape: {y_train.shape}")     # (876,): one target value per training row -- always matches X_train's row count
print(f"y_test shape:  {y_test.shape}")      # (220,): one target value per test row -- always matches X_test's row count

# ---------------------------------------------------------------------------
# 5. Prove stratify actually worked -- compare class ratios before/after
# ---------------------------------------------------------------------------
# .value_counts(normalize=True) gives the PROPORTION (not raw count) of each unique value in a column.
print("\nATTORNEY ratio in FULL dataset:")
print(y.value_counts(normalize=True).round(3))

print("\nATTORNEY ratio in TRAIN set:")
print(y_train.value_counts(normalize=True).round(3))

print("\nATTORNEY ratio in TEST set:")
print(y_test.value_counts(normalize=True).round(3))
# All three should show nearly the same ~52.7% / 47.3% split -- that's stratify=y doing its job.
# Try commenting out `stratify=y` above and rerunning -- the train/test ratios will drift apart slightly,
# since a plain random split has no guarantee of preserving the class balance.

# ---------------------------------------------------------------------------
# 6. Why this split matters -- a tiny proof, not just a claim
# ---------------------------------------------------------------------------
# The whole POINT of holding back a test set is to check whether a model actually learned a
# general pattern, or just memorized the training rows. Here's a minimal proof of that idea:
# an untrained/naive guess should perform similarly whether we check it on "seen" or "unseen" rows,
# but a model that TRAINED on one set and is evaluated on a DIFFERENT, held-out set is the only
# setup that can catch a model that just memorized noise instead of learning a real pattern.

train_row_ids = set(X_train.index)   # .index gives the original row-numbers (labels) each row came from in `df`
test_row_ids = set(X_test.index)

overlap = train_row_ids & test_row_ids   # set intersection -- rows appearing in BOTH groups
print(f"\nRows appearing in both train and test: {len(overlap)}")
# Should always print 0 -- train_test_split() guarantees every row lands in exactly ONE of the two groups, never both.
# This "no overlap" guarantee is the entire reason testing on X_test is a fair, honest check.
