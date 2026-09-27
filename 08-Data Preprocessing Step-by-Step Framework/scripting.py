"""
Data Preprocessing -- an industry-standard, step-by-step framework on assets/environmental_data.csv

Run this file directly: python scripting.py
(from inside this chapter's folder, since it uses a relative path to ../assets/environmental_data.csv)

Data: 158 daily environmental readings (May-Sep 2010) -- see ../assets/environmental_data_description.md
Task we prepare the data FOR: predict the day's AIR QUALITY category from the weather readings,
    Good (Ozone 0-50 ppb) / Moderate (51-100) / Poor (101+)   <- the bands from the dataset description

The course notebook (course_exercise_original.ipynb in this folder) does Stage 1 cleaning + Stage 2 transformation
on the whole table. This script follows the same two stages, but in the order industry uses:

    STAGE 0  Audit           -- look before you touch anything
    STAGE 1  Clean           -- RULE-based fixes that don't learn from the data (safe BEFORE the split)
    STAGE 2  Split           -- train/test split BEFORE anything that learns a statistic (Chapter 7)
    STAGE 3  Transform       -- impute / encode / scale, LEARNED FROM TRAIN ONLY, applied to both
    STAGE 4  Package         -- the same steps as one sklearn Pipeline (what actually ships)
    STAGE 5  Prove the rules -- Label vs One-Hot, parametric vs non-parametric, with real numbers

This file is meant to be self-explanatory through its comments -- read top to bottom.
"""

import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import (
    LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler,
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")   # hide harmless library notices so the output stays readable
pd.set_option("display.width", 140)

# ===========================================================================
# STAGE 0 -- AUDIT: look at everything before changing anything   (theory.md Section 2)
# ===========================================================================
raw = pd.read_csv("../assets/environmental_data.csv")
print("STAGE 0 -- AUDIT")
print("  shape:", raw.shape)                                          # (158, 10)
print("  dtypes:", raw.dtypes.astype(str).to_dict())
# 'Temp C' and 'Month' are text ('str' / 'object', depending on pandas version) although they should be numbers
# -> something non-numeric is hiding inside them
print("  missing per column:", raw.isna().sum().to_dict())             # Ozone 38, Solar.R 7, Weather 3

# Find the non-numeric values hiding in "numeric" columns
for col in ["Temp C", "Month"]:
    bad = raw.loc[pd.to_numeric(raw[col], errors="coerce").isna(), col]
    # pd.to_numeric(errors="coerce") turns anything that isn't a number into NaN -- those NaNs ARE the bad values
    print(f"  non-numeric values in '{col}': {bad.to_dict()}")          # Temp C: {10: 'C'}   Month: {23: 'May'}

temp_c = pd.to_numeric(raw["Temp C"], errors="coerce")
print(f"  'Temp C' equals 'Temp' on {(temp_c == raw['Temp']).sum()} of {temp_c.notna().sum()} numeric rows")  # 157 of 157
print(f"  'Temp' range: {raw['Temp'].min()} to {raw['Temp'].max()}  -> these are °F, not °C (97 °C is impossible)")
print(f"  'Year' unique values: {raw['Year'].unique().tolist()}")      # [2010] -- a constant column
print("  Weather counts:", raw["Weather"].value_counts(dropna=False).to_dict())   # S 59, C 49, PS 47, NaN 3

# Duplicates: check BOTH exact duplicates AND duplicates of the real-world KEY (one row per calendar day)
data_cols = raw.columns.drop("Unnamed: 0")   # ignore the row-number column, it's unique by construction
month_num = raw["Month"].replace("May", "5")
print(f"  exact duplicate rows: {raw.duplicated(subset=data_cols).sum()}")                         # 1
print(f"  rows sharing the same (Month, Day): {raw.assign(M=month_num).duplicated(['M', 'Day']).sum()}")  # 5
# Only 1 exact duplicate, but 5 repeated DAYS: rows 154-158 repeat earlier dates with identical readings,
# yet 4 of them have a DIFFERENT Weather label. A plain .duplicated() would have missed 4 of the 5.

# Outliers: the 1.5 x IQR rule (from the statistics track)
for col in ["Ozone", "Solar.R", "Wind", "Temp"]:
    q1, q3 = raw[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    flagged = raw.loc[(raw[col] < q1 - 1.5 * iqr) | (raw[col] > q3 + 1.5 * iqr), col].tolist()
    print(f"  IQR outliers in {col:<7}: {flagged}")
# Ozone [135, 168], Wind [20.1, 20.7] -- both physically plausible (a bad-smog day, a windy day). NOT errors.

# ===========================================================================
# STAGE 1 -- CLEAN: rule-based fixes (theory.md Section 3). Nothing here "learns" from the data,
#            so it's safe to do on the whole table before splitting.
# ===========================================================================
print("\nSTAGE 1 -- CLEAN")
df = raw.copy()   # never modify the raw data in place -- keep it for comparison / re-runs

# 1a. Remove columns that carry no information
df = df.drop(columns=["Unnamed: 0",   # an exported row number -- a model could 'learn' row order, which is meaningless
                      "Temp C",       # an exact duplicate of Temp (plus a typo) -- keeping both adds nothing
                      "Year"])        # constant (2010 everywhere) -- zero information
df = df.rename(columns={"Temp": "Temp_F"})   # say the unit in the name, since the description got it wrong

# 1b. Fix wrong values, then fix dtypes
df["Month"] = df["Month"].replace("May", "5").astype(int)   # 'May' -> 5, then text column -> integer column

# 1c. Remove duplicates on the real-world key, keeping the FIRST (original) reading of each day
before = len(df)
df = df.drop_duplicates(subset=["Month", "Day"], keep="first")
print(f"  duplicates removed: {before} -> {len(df)} rows")   # 158 -> 153 (153 = the real number of days, May 1 - Sep 30)

# 1d. The TARGET: build it, and drop rows where it's missing (NEVER impute a target)
df = df[df["Ozone"].notna()].copy()
# Imputing Ozone would mean inventing the very answer we want the model to learn -- those rows can't be used.
# (The course notebook DELETED the Ozone column because it's 24% missing. That's a sensible rule for an INPUT
#  feature -- but here Ozone is the thing we want to predict, so we drop the ROWS instead.)
df["AirQuality"] = pd.cut(df["Ozone"], bins=[0, 50, 100, np.inf], labels=["Good", "Moderate", "Poor"]).astype(str)
# pd.cut() turns a number into a band: (0, 50] -> Good, (50, 100] -> Moderate, (100, inf) -> Poor
print(f"  rows with a known target: {len(df)}")                                   # 116
print(f"  target counts: {df['AirQuality'].value_counts().to_dict()}")           # Good 82, Moderate 27, Poor 7

# 1e. Outliers: decision = KEEP. Ozone 135/168 are exactly the "Poor" days we most need to learn about;
#     Wind 20.1/20.7 mph is a normal windy day. Outliers are removed only when they're ERRORS (e.g. Wind = -5, Temp = 970).

# 1f. Choose the features
FEATURES_NUM = ["Solar.R", "Wind", "Temp_F", "Month"]
FEATURES_CAT = ["Weather"]
# Dropped: Ozone (it IS the target -- using it as a feature would be leakage), Day (a day-of-month number
# has no physical effect on air quality; it's an identifier of when, not a cause).
X = df[FEATURES_NUM + FEATURES_CAT]
print(f"  missing values still in X (to be handled in STAGE 3): {X.isna().sum()[lambda s: s > 0].to_dict()}")
# Solar.R 5, Weather 3 -- left for later ON PURPOSE: imputing with a mean/median/mode LEARNS a statistic.

df.to_csv("environmental_data_clean.csv", index=False)   # the cleaned table, saved for inspection
print("  saved environmental_data_clean.csv")

# ===========================================================================
# STAGE 2 -- SPLIT before any step that learns from the data (theory.md Section 4, Chapter 7)
# ===========================================================================
print("\nSTAGE 2 -- SPLIT")
# OUTPUT feature: always Label Encoder (course rule). Text classes -> integers 0..k-1.
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df["AirQuality"])
print(f"  target encoding: {dict(zip(target_encoder.classes_, range(len(target_encoder.classes_))))}")
# {'Good': 0, 'Moderate': 1, 'Poor': 2}  (LabelEncoder sorts alphabetically; for a target that's fine -- a classifier
#  treats 0/1/2 as NAMES of classes, never as amounts, so the order can't mislead it)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"  train {len(X_train)} rows / test {len(X_test)} rows")                               # 87 / 29
print(f"  'Poor' days: train {(y_train == 2).sum()}, test {(y_test == 2).sum()}")   # stratify keeps rare class on both sides

# ===========================================================================
# STAGE 3 -- TRANSFORM, step by step: learn every statistic from TRAIN, apply to both (theory.md Sections 5-7)
# ===========================================================================
print("\nSTAGE 3 -- TRANSFORM (done by hand first, so every step is visible)")
Xtr, Xte = X_train.copy(), X_test.copy()

# 3a. Impute numeric: mean or median? Look at the shape (skew) of the TRAIN column.
skew = Xtr["Solar.R"].skew()
print(f"  Solar.R skew (train) = {skew:.2f}")   # -0.46
# |skew| < ~0.5 -> roughly symmetric -> mean and median are close; median is the safer default when unsure.
solar_median = Xtr["Solar.R"].median()        # learned from TRAIN only
Xtr["Solar.R"] = Xtr["Solar.R"].fillna(solar_median)
Xte["Solar.R"] = Xte["Solar.R"].fillna(solar_median)   # the TEST gaps are filled with the TRAIN median, not their own
print(f"  Solar.R missing filled with train median = {solar_median}")   # 203.0

# (The course notebook's cleverer option -- fill by the mean of each Weather group -- is also valid, as long as the
#  group means come from TRAIN only:)
print(f"  (alternative) train Solar.R mean by Weather: {X_train.groupby('Weather')['Solar.R'].mean().round(1).to_dict()}")
# C 193.2, PS 176.2, S 196.5

# 3b. Impute categorical: mode (most frequent value), from TRAIN
weather_mode = Xtr["Weather"].mode()[0]
Xtr["Weather"] = Xtr["Weather"].fillna(weather_mode)
Xte["Weather"] = Xte["Weather"].fillna(weather_mode)
print(f"  Weather missing filled with train mode = '{weather_mode}'")   # 'S'

# 3c. Encode the INPUT categorical feature -- two ways
# Label/Ordinal encoding: one column, categories -> 0, 1, 2
ordinal = OrdinalEncoder(categories=[["C", "PS", "S"]])   # explicit order: Cloudy < Partly Sunny < Sunny (sunniness)
# (for INPUT features use OrdinalEncoder -- LabelEncoder is designed for the 1-D target y, and can't be used inside
#  a ColumnTransformer. Same idea, right tool.)
w_ord = ordinal.fit_transform(Xtr[["Weather"]])
print(f"  Ordinal-encoded Weather, first 5 train rows: {w_ord[:5].ravel().tolist()}")

# One-hot encoding: one 0/1 column per category
onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# handle_unknown="ignore": a category never seen in training becomes all-zeros instead of crashing in production.
# NOTE: the course notebook's `ohe.fit_transform(df["Weather"])` fails, because sklearn encoders need a 2-D input --
#       df[["Weather"]] (double brackets = a one-column TABLE), not df["Weather"] (single brackets = a Series).
w_ohe = onehot.fit_transform(Xtr[["Weather"]])
print(f"  One-hot columns: {onehot.get_feature_names_out().tolist()}")   # ['Weather_C', 'Weather_PS', 'Weather_S']
print(f"  One-hot first 3 train rows: {w_ohe[:3].tolist()}")
print(f"  pandas equivalent: {pd.get_dummies(Xtr['Weather'], prefix='Weather', drop_first=True, dtype=int).columns.tolist()}")
# drop_first=True -> ['Weather_PS', 'Weather_S']: 'C' becomes the all-zeros baseline. Avoids the "dummy variable trap"
# (the 3 columns always sum to 1 = perfect multicollinearity for Linear Regression, Chapter 2 Section 8).

# 3d. Scale the continuous features -- three scalers, fitted on TRAIN
num_train = Xtr[["Solar.R", "Wind", "Temp_F"]]
print("  Scalers, fitted on train (showing Wind):")
print(f"    raw        : min {num_train['Wind'].min():.1f}  median {num_train['Wind'].median():.1f}  max {num_train['Wind'].max():.1f}")
for name, scaler in [("Standard", StandardScaler()), ("MinMax", MinMaxScaler()), ("Robust", RobustScaler())]:
    scaled = pd.DataFrame(scaler.fit_transform(num_train), columns=num_train.columns)
    print(f"    {name:<11}: min {scaled['Wind'].min():6.2f}  median {scaled['Wind'].median():6.2f}  max {scaled['Wind'].max():6.2f}")
# raw 4.0 / 9.7 / 20.1 -> Standard -1.72 / -0.01 / 3.09 -> MinMax 0.00 / 0.35 / 1.00 -> Robust -1.39 / 0.00 / 2.54
# Standard: mean 0, std 1 (values roughly -3..+3)   MinMax: squeezed into exactly 0..1   Robust: median 0, scaled by IQR
# With an outlier (Wind 20.7), MinMax lets that single day define the "1.0" end, squashing every normal day together.

# Why scaling matters for distance-based models: two real days, raw units
a, b = num_train.iloc[0], num_train.iloc[1]
print(f"  Two train days: {a.to_dict()} vs {b.to_dict()}")
diff = (a - b).abs()
print(f"  raw differences: {diff.round(1).to_dict()}")
print(f"  -> share of the (squared) distance due to Solar.R: {(diff['Solar.R']**2 / (diff**2).sum()):.1%}")
# Solar.R differs by 41, Wind by 0.6, Temp by 23 -> Solar.R is 76.1% of the squared distance.
# Solar.R is measured in the hundreds, Wind in single digits: in raw units, 'how far apart are these days?' is
# almost entirely a Solar.R question. Scaling puts every feature on the same footing.

# ===========================================================================
# STAGE 4 -- PACKAGE: the same steps as ONE sklearn Pipeline (theory.md Section 8) -- what industry ships
# ===========================================================================
print("\nSTAGE 4 -- PIPELINE")


def build_pipeline(model, encoder, scaler):
    """Imputer -> (scaler) for numeric columns, imputer -> encoder for the categorical column, then the model.
    Every learned statistic lives INSIDE the pipeline, so .fit() learns them from whatever data it's given --
    inside cross-validation that means from each training fold only. No leakage, by construction."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scaler is not None:
        numeric_steps.append(("scale", scaler))
    preprocess = ColumnTransformer([
        ("num", Pipeline(numeric_steps), FEATURES_NUM),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", encoder)]), FEATURES_CAT),
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


# Parametric model -> One-Hot + scaling (course rule)
logreg_pipe = build_pipeline(LogisticRegression(max_iter=5000), OneHotEncoder(handle_unknown="ignore"), StandardScaler())
# Non-parametric TREE model -> Ordinal (label) encoding, no scaling needed (course rule)
forest_pipe = build_pipeline(RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42),
                             OrdinalEncoder(categories=[["C", "PS", "S"]]), None)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)   # Chapter 7: small data -> repeated CV
baseline = (y_train == np.bincount(y_train).argmax()).mean()
print(f"  majority-class baseline (always 'Good'), train: {baseline:.3f}")
for name, pipe in [("LogisticRegression + OneHot + StandardScaler", logreg_pipe),
                   ("RandomForest + Ordinal, no scaling", forest_pipe)]:
    scores = cross_val_score(pipe, X_train, y_train, cv=cv)   # the whole pipeline is re-fitted inside every fold
    pipe.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f"  {name:<46} CV acc {scores.mean():.3f} ± {scores.std():.3f} | test acc {test_acc:.3f}")

# LogReg + OneHot + Standard : CV 0.825 ± 0.062 | test 0.828
# RandomForest + Ordinal     : CV 0.837 ± 0.062 | test 0.828      (baseline 0.713)
# Both beat the baseline clearly; with only 87 training rows the ± 0.062 spread means they're effectively tied.

# Ship the pipeline, not just the model: preprocessing + model in ONE file, so production can't forget a step
joblib.dump(logreg_pipe, "air_quality_pipeline.pkl")
loaded = joblib.load("air_quality_pipeline.pkl")
new_day = pd.DataFrame({"Solar.R": [np.nan], "Wind": [4.0], "Temp_F": [92], "Month": [7], "Weather": ["S"]})
# a raw, messy new row: missing Solar.R, text Weather -- the pipeline imputes, encodes and scales it itself
pred = target_encoder.inverse_transform(loaded.predict(new_day))[0]   # inverse_transform: 0/1/2 -> 'Good'/...
print(f"  saved air_quality_pipeline.pkl; hot, calm, sunny July day with missing Solar.R -> predicted '{pred}'")

# ===========================================================================
# STAGE 5 -- PROVE THE RULES with real numbers (theory.md Section 6)
# ===========================================================================
print("\nSTAGE 5 -- WHY 'parametric -> One-Hot, non-parametric -> Label'")

# 5a. The course's Excel example (label_vs_onehot_excel_example.png): salary from experience + location
toy = pd.DataFrame({"Experience": [1, 1, 3, 2, 2],
                    "Location": ["Bangalore", "Chennai", "Chennai", "Bangalore", "Mumbai"],
                    "Salary": [40000, 35000, 60000, 55000, 35000]})
print("  Linear Regression (parametric) on the course's 5-row salary table, R² on those rows:")
r2_exp = LinearRegression().fit(toy[["Experience"]], toy["Salary"]).score(toy[["Experience"]], toy["Salary"])
print(f"    Experience only                                    : {r2_exp:.3f}")
for label, mapping in [("Label-encoded, alphabetical (B=0, C=1, M=2)", {"Bangalore": 0, "Chennai": 1, "Mumbai": 2}),
                       ("Label-encoded, different order (C=0, B=1, M=2)", {"Chennai": 0, "Bangalore": 1, "Mumbai": 2})]:
    Xt = toy[["Experience"]].assign(Location=toy["Location"].map(mapping))
    print(f"    {label:<51}: {LinearRegression().fit(Xt, toy['Salary']).score(Xt, toy['Salary']):.3f}")
Xo = pd.concat([toy[["Experience"]], pd.get_dummies(toy["Location"], drop_first=True, dtype=int)], axis=1)
print(f"    {'One-hot (any order gives the same answer)':<51}: {LinearRegression().fit(Xo, toy['Salary']).score(Xo, toy['Salary']):.3f}")
# 0.584 / 0.977 / 0.692 / 0.995
# SAME data, SAME cities -- just renumbering them moves R² from 0.977 to 0.692. Label encoding tells a linear model
# "Chennai is exactly halfway between Bangalore and Mumbai" -- a fake fact the model then tries to fit.
# One-hot gives each city its own coefficient, so there's no fake order to get wrong.
# (5 rows is far too few for a real model -- the point is the ORDER SENSITIVITY, not the R² values themselves.)

# 5b. The same test on our real data: scramble the (meaningless) number order of Weather
print("  Real data, CV accuracy when Weather's integer order is changed (StandardScaler for both):")
for order in [["C", "PS", "S"], ["PS", "C", "S"]]:
    row = []
    for name, model in [("LogReg", LogisticRegression(max_iter=5000)),
                        ("Tree", DecisionTreeClassifier(max_depth=3, random_state=42))]:
        s = cross_val_score(build_pipeline(model, OrdinalEncoder(categories=[order]), StandardScaler()),
                            X_train, y_train, cv=cv).mean()
        row.append(f"{name} {s:.3f}")
    print(f"    order {order}: {' | '.join(row)}")
# ['C','PS','S']: LogReg 0.821 | Tree 0.795      ['PS','C','S']: LogReg 0.831 | Tree 0.792
# The parametric model's score changes when only the (arbitrary) numbering changes; the tree barely notices,
# because a tree only asks "Weather <= 0.5?" style questions and can isolate any category with 1-2 splits.

# 5c. Does scaling matter? Parametric/distance models vs trees
print("  CV accuracy by scaler (One-Hot Weather):")
for name, model in [("LogReg", LogisticRegression(max_iter=5000)),
                    ("KNN", KNeighborsClassifier(n_neighbors=5)),
                    ("Tree", DecisionTreeClassifier(max_depth=3, random_state=42))]:
    row = []
    for sname, scaler in [("none", None), ("Standard", StandardScaler()), ("MinMax", MinMaxScaler()), ("Robust", RobustScaler())]:
        s = cross_val_score(build_pipeline(model, OneHotEncoder(handle_unknown="ignore"), scaler), X_train, y_train, cv=cv).mean()
        row.append(f"{sname} {s:.3f}")
    print(f"    {name:<6}: {' | '.join(row)}")
# LogReg: none 0.809 | Standard 0.825 | MinMax 0.815 | Robust 0.827
# KNN   : none 0.796 | Standard 0.818 | MinMax 0.756 | Robust 0.800
# Tree  : none 0.790 | Standard 0.791 | MinMax 0.790 | Robust 0.790
# KNN (distance-based) moves the most with the scaler; the tree gives the same answer whatever the scaling.
