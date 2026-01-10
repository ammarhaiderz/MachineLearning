import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.feature_selection import mutual_info_classif

# ===============================
# CONFIG
# ===============================
PROBLEM_NUM = 36
RANDOM_STATE = 42
N_SPLITS = 5

# 🔴 Put your three binary feature names here
BINARY_FEATURES = ["feat_24", "feat_104", "feat_131"]  # <-- edit

# ===============================
# LOAD DATA
# ===============================
X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"].values

# Sanity check: confirm they are binary
print("Binary feature value counts:")
for f in BINARY_FEATURES:
    vc = X[f].value_counts(dropna=False)
    print(f"\n{f}:\n{vc}")

# ===============================
# CV SETUP
# ===============================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

def fit_gmm_regime(y_train):
    """Fit GMM on y_train and return a function mapping y -> regime label/prob."""
    gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE)
    gmm.fit(y_train.reshape(-1, 1))

    # order by mean: component 0 = lower mean (left mode)
    means = gmm.means_.ravel()
    order = np.argsort(means)

    def regime_prob(y_arr):
        probs = gmm.predict_proba(y_arr.reshape(-1, 1))[:, order]
        # return prob of high-mode (right hump)
        return probs[:, 1]

    return regime_prob

def eval_feature_set(cols):
    """Evaluate how well cols predict the regime (defined from y via GMM inside each fold)."""
    aucs, accs, mis = [], [], []

    for tr_idx, va_idx in skf.split(X, (y > np.median(y)).astype(int)):  # stratification helper only
        X_tr = X.iloc[tr_idx][cols].copy()
        X_va = X.iloc[va_idx][cols].copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # ---- define regime on TRAIN using y only (diagnostic, fold-safe)
        reg_prob_fn = fit_gmm_regime(y_tr)
        p_high_tr = reg_prob_fn(y_tr)
        p_high_va = reg_prob_fn(y_va)

        reg_tr = (p_high_tr >= 0.5).astype(int)
        reg_va = (p_high_va >= 0.5).astype(int)

        # ---- model: logistic regression on binary inputs
        clf = LogisticRegression(solver="liblinear", random_state=RANDOM_STATE)
        clf.fit(X_tr, reg_tr)

        proba = clf.predict_proba(X_va)[:, 1]
        pred = (proba >= 0.5).astype(int)

        aucs.append(roc_auc_score(reg_va, proba))
        accs.append(accuracy_score(reg_va, pred))

        # mutual information (nonlinear dependence measure)
        mi = mutual_info_classif(X_va, reg_va, discrete_features=True, random_state=RANDOM_STATE)
        mis.append(mi.sum())

    return {
        "features": cols,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "mi_mean": float(np.mean(mis)),
        "mi_std": float(np.std(mis)),
    }

# ===============================
# TEST: singles, pairs, all three
# ===============================
feature_sets = []
for k in [1, 2, 3]:
    for cols in combinations(BINARY_FEATURES, k):
        feature_sets.append(list(cols))

results = [eval_feature_set(cols) for cols in feature_sets]
results_df = pd.DataFrame(results).sort_values("auc_mean", ascending=False)

print("\n" + "="*90)
print("REGIME DETECTION FROM 3 BINARY FEATURES (GMM-defined regime, CV)")
print("="*90)
print(results_df.to_string(index=False))
print("="*90)

best = results_df.iloc[0]
print("\nBEST SET:")
print(best)
