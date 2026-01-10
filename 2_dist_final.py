# ============================================================
# 0. Imports
# ============================================================
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

from catboost import CatBoostClassifier, CatBoostRegressor


# ============================================================
# 1. Load data
# ============================================================
PROBLEM_NUM = 36

# =======================
# ONLY CHANGE #1: FEATURE LIST
# =======================
SELECTED_FEATURES = [ 
    'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253', 
    'feat_143', 'feat_221', 'feat_220', 'feat_160', 'feat_266', 
    'feat_138', 'feat_47', 'feat_203', 
]

X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"].values
X_eval = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv")

# =======================
# ONLY CHANGE #2: FEATURE FILTERING
# =======================
X = X[SELECTED_FEATURES]
X_eval = X_eval[SELECTED_FEATURES]

print(f"Train X: {X.shape}, y: {y.shape}")
print(f"Eval  X: {X_eval.shape}")


# ============================================================
# 2. Train / validation split (BEFORE any target analysis)
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"\nTrain split: {len(X_train)} samples")
print(f"Val split:   {len(X_val)} samples")


# ============================================================
# 3. DATA-DRIVEN REGIME DISCOVERY (GMM ON TRAINING TARGET ONLY)
# ============================================================
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(y_train.reshape(-1, 1))

r_train = gmm.predict(y_train.reshape(-1, 1))
r_val = gmm.predict(y_val.reshape(-1, 1))

r_train_proba = gmm.predict_proba(y_train.reshape(-1, 1))
r_val_proba = gmm.predict_proba(y_val.reshape(-1, 1))

means = gmm.means_.ravel()
order = np.argsort(means)

r_train = np.array([np.where(order == r)[0][0] for r in r_train])
r_val = np.array([np.where(order == r)[0][0] for r in r_val])

r_train_proba = r_train_proba[:, order]
r_val_proba = r_val_proba[:, order]

print("\nGMM regime means (from training):", means[order])
print("Train regime proportions:", np.bincount(r_train) / len(r_train))
print("Val   regime proportions:", np.bincount(r_val) / len(r_val))


# ============================================================
# 4. STAGE 1: REGIME CLASSIFIER (X -> regime)
# ============================================================
clf = CatBoostClassifier(
    iterations=600,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    random_seed=42,
    verbose=False
)

clf.fit(X_train, r_train)

r_val_pred = clf.predict(X_val).astype(int)
r_val_proba = clf.predict_proba(X_val)

regime_acc = (r_val_pred == r_val).mean()
print(f"\nRegime classification accuracy (VAL): {regime_acc:.4f}")


# ============================================================
# 5. STAGE 2: REGRESSORS PER REGIME
# ============================================================
regressors = {}

for reg in [0, 1]:
    idx = r_train == reg

    model = CatBoostRegressor(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )

    model.fit(X_train[idx], y_train[idx])
    regressors[reg] = model


# ============================================================
# 6. SOFT MIXTURE-OF-EXPERTS VALIDATION
# ============================================================
y_train_pred = np.zeros(len(X_train))
clf_train_proba = clf.predict_proba(X_train)

for i in range(len(X_train)):
    probs = clf_train_proba[i]
    y_train_pred[i] = (
        probs[0] * regressors[0].predict(X_train.iloc[[i]])[0]
        + probs[1] * regressors[1].predict(X_train.iloc[[i]])[0]
    )

y_val_pred = np.zeros(len(X_val))
clf_val_proba = clf.predict_proba(X_val)

for i in range(len(X_val)):
    probs = clf_val_proba[i]
    y_val_pred[i] = (
        probs[0] * regressors[0].predict(X_val.iloc[[i]])[0]
        + probs[1] * regressors[1].predict(X_val.iloc[[i]])[0]
    )


# ============================================================
# 7. METRICS
# ============================================================
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = root_mean_squared_error(y_train, y_train_pred)

val_r2 = r2_score(y_val, y_val_pred)
val_rmse = root_mean_squared_error(y_val, y_val_pred)

print(f"\n{'='*50}")
print(f"MODEL PERFORMANCE METRICS")
print(f"{'='*50}")
print(f"Train R²:   {train_r2:.4f}")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Val R²:     {val_r2:.4f}")
print(f"Val RMSE:   {val_rmse:.4f}")
print(f"{'='*50}")


# ============================================================
# 8. TRAIN FINAL MODELS ON FULL DATA
# ============================================================
gmm_full = GaussianMixture(n_components=2, random_state=42)
gmm_full.fit(y.reshape(-1, 1))

regime_full = gmm_full.predict(y.reshape(-1, 1))
means_full = gmm_full.means_.ravel()
order_full = np.argsort(means_full)

regime_full = np.array([np.where(order_full == r)[0][0] for r in regime_full])

clf.fit(X, regime_full)

for reg in [0, 1]:
    regressors[reg].fit(X[regime_full == reg], y[regime_full == reg])


# ============================================================
# 9. EVALUATION DATA PREDICTION (NO LABELS)
# ============================================================
eval_regime_proba = clf.predict_proba(X_eval)

print("\nEval regime distribution:",
      np.mean(eval_regime_proba, axis=0))

confidence = np.max(eval_regime_proba, axis=1)
print(f"Low-confidence samples (<0.6): {(confidence < 0.6).mean():.2%}")

y_eval_pred = np.zeros(len(X_eval))

for i in range(len(X_eval)):
    probs = eval_regime_proba[i]
    y_eval_pred[i] = (
        probs[0] * regressors[0].predict(X_eval.iloc[[i]])[0]
        + probs[1] * regressors[1].predict(X_eval.iloc[[i]])[0]
    )


# ============================================================
# 10. SAVE OUTPUT
# ============================================================
output_file = f"EVAL_target01_{PROBLEM_NUM}_GMM_mixture_catboost.csv"
pd.DataFrame({"target01": y_eval_pred}).to_csv(output_file, index=False)

print(f"\nPredictions saved to: {output_file}")
print("Sample predictions:", y_eval_pred[:10])
