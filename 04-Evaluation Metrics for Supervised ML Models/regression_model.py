"""
Regression Model Evaluation Metrics — a hands-on demo using assets/Cars.csv

Run this file directly: python regression_model.py
(from inside this chapter's folder, since it uses a relative path to ../assets/Cars.csv)

Trains the SAME Linear Regression model as 02-Linear Regression/scripting.ipynb
(Cars.csv, predicting MPG), then walks through every metric from
"Regression Model.md" step by step, computing each one two ways where possible:
once via sklearn's built-in function, once by hand from the raw formula -- so you can
SEE that "the formula" and "the library call" are the exact same thing.

This file is meant to be self-explanatory through its comments -- read top to bottom.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# 1. Rebuild the same trained model from Chapter 2 (needed before any metric makes sense)
# ---------------------------------------------------------------------------
df = pd.read_csv("../assets/Cars.csv")

X = df.drop(columns=["MPG"])   # features: HP, VOL, SP, WT
y = df["MPG"]                   # target: MPG (a continuous number)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42   # same split settings as Chapter 2 -- so we get the SAME test set, and comparable numbers
)

model = LinearRegression()
model.fit(X_train, y_train)   # .fit() = training -- OLS solves directly for the m's and b that minimize MSE (theory.md Chapter 2)

y_pred = model.predict(X_test)   # predicted MPG per test row

print(f"Model trained. Evaluating on the {len(y_test)}-row held-out test set.\n")

# The building block EVERY metric below is computed from: the residual (actual - predicted), per row.
residuals = y_test.values - y_pred
print("Residuals (actual - predicted) for each test-set car:")
print(np.round(residuals, 2))

# ---------------------------------------------------------------------------
# 2. MAE -- Mean Absolute Error
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MAE -- Mean Absolute Error: average |residual|, every error weighted equally")
print("=" * 60)

mae_sklearn = mean_absolute_error(y_test, y_pred)   # library function
mae_by_hand = np.mean(np.abs(residuals))              # raw formula: average of |actual - predicted|
# np.abs() takes the absolute value of every element in the array (removes the sign);
# np.mean() then averages all of them into one number.

print(f"sklearn's mean_absolute_error(): {mae_sklearn:.4f}")
print(f"same number, by hand:             {mae_by_hand:.4f}")
assert round(mae_sklearn, 4) == round(mae_by_hand, 4)   # proves the library call and the formula agree exactly

# ---------------------------------------------------------------------------
# 3. MSE -- Mean Squared Error
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MSE -- Mean Squared Error: average (residual)^2, big errors punished harder")
print("=" * 60)

mse_sklearn = mean_squared_error(y_test, y_pred)   # library function
mse_by_hand = np.mean(residuals ** 2)                # raw formula: average of (actual - predicted)^2
# residuals ** 2 squares every element in the array at once (removes sign, punishes large residuals disproportionately)

print(f"sklearn's mean_squared_error(): {mse_sklearn:.4f}")
print(f"same number, by hand:            {mse_by_hand:.4f}")
assert round(mse_sklearn, 4) == round(mse_by_hand, 4)
print("Note the units here are 'MPG-squared' -- not intuitive on its own. That's what RMSE fixes next.")

# ---------------------------------------------------------------------------
# 4. RMSE -- Root Mean Squared Error
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RMSE -- Root Mean Squared Error: sqrt(MSE), back in real MPG units")
print("=" * 60)

rmse = np.sqrt(mse_sklearn)   # sklearn has no separate rmse function -- you just take the square root of MSE yourself
print(f"RMSE = sqrt(MSE) = sqrt({mse_sklearn:.4f}) = {rmse:.4f}")

print(f"\nCompare RMSE ({rmse:.3f}) to MAE ({mae_sklearn:.3f}):")
print("RMSE is always >= MAE for the same data. The bigger the gap, the more a few")
print("large errors are dragging the average up (since squaring punishes them harder).")

# ---------------------------------------------------------------------------
# 5. R^2 Score -- fraction of variation explained, vs. the 'always guess the mean' baseline
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("R^2 SCORE -- how much of MPG's variation does the model explain?")
print("=" * 60)

r2_sklearn = r2_score(y_test, y_pred)   # library function

ss_res = np.sum(residuals ** 2)                        # Sum of Squared Residuals -- the model's actual total squared error
ss_tot = np.sum((y_test.values - y_test.mean()) ** 2)   # Total Sum of Squares -- error of the "always predict the mean" baseline
r2_by_hand = 1 - (ss_res / ss_tot)                       # raw formula: 1 - (model's error / baseline's error)

print(f"SS_res (model's squared error)                    = {ss_res:.4f}")
print(f"SS_tot (baseline-always-predict-mean squared error) = {ss_tot:.4f}")
print(f"sklearn's r2_score(): {r2_sklearn:.4f}")
print(f"same number, by hand:  {r2_by_hand:.4f}")
assert round(r2_sklearn, 4) == round(r2_by_hand, 4)
print(f"\nReading it: the model explains about {r2_sklearn*100:.1f}% of MPG's variation")
print(f"across these cars -- the remaining {(1-r2_sklearn)*100:.1f}% is unexplained by HP/VOL/SP/WT alone.")

# ---------------------------------------------------------------------------
# 6. Adjusted R^2 -- R^2, penalized for the number of features used
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ADJUSTED R^2 -- R^2 penalized for feature count, so it can't be gamed")
print("=" * 60)


def adjusted_r2(r2, n, k):
    """
    r2: the plain R^2 score
    n:  number of observations (rows) the score was computed on
    k:  number of features (predictors) the model used
    Returns Adjusted R^2 -- R^2 with a penalty that grows as k grows relative to n.
    """
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


n = len(y_test)          # number of test-set rows
k = X_test.shape[1]       # number of features -- .shape[1] is the COLUMN count of the X_test DataFrame
adj_r2 = adjusted_r2(r2_sklearn, n, k)

print(f"n (test rows) = {n},  k (features: {list(X.columns)}) = {k}")
print(f"Adjusted R^2 = 1 - (1 - {r2_sklearn:.4f}) * ({n}-1)/({n}-{k}-1) = {adj_r2:.4f}")
print(f"\nPlain R^2:     {r2_sklearn:.4f}")
print(f"Adjusted R^2:  {adj_r2:.4f}   (lower -- this is the penalty for using 4 features)")

# ---------------------------------------------------------------------------
# 7. The proof: adding a USELESS random feature raises R^2 but LOWERS Adjusted R^2
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PROOF -- add a random noise feature (zero real relationship to MPG)")
print("=" * 60)

np.random.seed(0)   # fixed seed -- makes the "random" noise below reproducible every time this script runs
X_train_noisy = X_train.copy()   # .copy() makes an independent copy, so we don't modify the original X_train
X_test_noisy = X_test.copy()

# np.random.rand(n) generates n random numbers between 0 and 1, with NO relationship to MPG whatsoever --
# a pure noise column, added as a 5th "feature" purely to see what it does to each metric.
X_train_noisy["RANDOM_NOISE"] = np.random.rand(len(X_train_noisy))
X_test_noisy["RANDOM_NOISE"] = np.random.rand(len(X_test_noisy))

model_noisy = LinearRegression()
model_noisy.fit(X_train_noisy, y_train)
y_pred_noisy = model_noisy.predict(X_test_noisy)

r2_noisy = r2_score(y_test, y_pred_noisy)
k_noisy = X_test_noisy.shape[1]   # now 5 features instead of 4
adj_r2_noisy = adjusted_r2(r2_noisy, n, k_noisy)

print(f"{'Metric':<15}{'4 real features':>20}{'+ 1 random feature':>22}")
print(f"{'R^2':<15}{r2_sklearn:>20.4f}{r2_noisy:>22.4f}   <- went UP, even though the new feature is pure noise")
print(f"{'Adjusted R^2':<15}{adj_r2:>20.4f}{adj_r2_noisy:>22.4f}   <- went DOWN, correctly penalizing the useless feature")

print("\nThis is exactly why you check Adjusted R^2, not plain R^2, when deciding")
print("whether a new feature is actually worth adding to a model.")

# ---------------------------------------------------------------------------
# 8. Everything side by side
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY -- all metrics together, for the original 4-feature model")
print("=" * 60)
print(f"{'MAE':<15}: {mae_sklearn:.3f}  (MPG, every error weighted equally)")
print(f"{'MSE':<15}: {mse_sklearn:.3f}  (MPG^2 -- awkward units, mainly for training)")
print(f"{'RMSE':<15}: {rmse:.3f}  (MPG, penalizes large errors more than MAE)")
print(f"{'R^2':<15}: {r2_sklearn:.3f}  (fraction of variation explained)")
print(f"{'Adjusted R^2':<15}: {adj_r2:.3f}  (R^2, penalized for using 4 features)")
