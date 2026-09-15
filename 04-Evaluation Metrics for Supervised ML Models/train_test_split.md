# Train/Test Split

**What it is:** before training a model, you split your labeled data into two separate groups — a **training set** (the model learns from this) and a **test set** (held back completely, the model never sees it during training). Once trained, you evaluate the model *only* on the test set.

**Why:** a model can score perfectly on data it was trained on without actually having learned anything useful — it could just be memorizing. The test set is the only honest check for whether the model learned a real, generalizable pattern, or just memorized the specific rows it saw. Every metric in Chapter 4 (and every metric already used in Chapter 2/3's `scripting.ipynb`) is only meaningful when computed on the test set, never the training set.

**The common split ratios:** 80/20 or 70/30 (train/test) are the usual defaults — no single "correct" number, it's a tradeoff: more test data gives a more reliable evaluation, but leaves less data to actually train on.

**`stratify`:** for a classification target, pass `stratify=y` so the split keeps the target's class ratio (e.g. ~51%/49%) consistent across both the train and test sets — otherwise a random split could accidentally end up unbalanced in one side, purely by chance. This only applies to classification; a continuous regression target has no "class ratio" to preserve.

**`random_state`:** a fixed seed so the random split is reproducible — same number in, same exact split out, every time you rerun the code.

See `train_test_split.py` in this same folder for a hands-on, heavily-commented walkthrough on `assets/claim.csv`, including proof that `stratify` actually works and that train/test rows never overlap.
