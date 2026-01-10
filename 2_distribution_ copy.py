# ============================================================
# 0. Imports
# ============================================================
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import QuantileTransformer

from catboost import CatBoostClassifier, CatBoostRegressor


# ============================================================
# 1. Load data
# ============================================================
PROBLEM_NUM = 36

X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"].values
X_eval = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv")

print(f"Train X: {X.shape}, y: {y.shape}")
print(f"Eval  X: {X_eval.shape}")


# ============================================================
# 2. DATA-DRIVEN REGIME DISCOVERY (GMM ON ORIGINAL TARGET)
# ============================================================
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(y.reshape(-1, 1))

regime = gmm.predict(y.reshape(-1, 1))
regime_proba = gmm.predict_proba(y.reshape(-1, 1))

means = gmm.means_.ravel()
order = np.argsort(means)
regime = np.array([np.where(order == r)[0][0] for r in regime])
regime_proba = regime_proba[:, order]

print("\nGMM regime means:", means[order])
print("GMM regime proportions:", np.bincount(regime) / len(regime))


# ============================================================
# 3. Train / validation split (stratified by regime)
# ============================================================
X_train, X_val, y_train, y_val, r_train, r_val = train_test_split(
    X, y, regime,
    test_size=0.2,
    random_state=42,
    stratify=regime
)

print("\nTrain regime distribution:", np.bincount(r_train) / len(r_train))
print("Val   regime distribution:", np.bincount(r_val) / len(r_val))


# ============================================================
# 4. TARGET QUANTILE TRANSFORM (FIT ON TRAIN ONLY)
# ============================================================
qt = QuantileTransformer(
    n_quantiles=min(1000, len(y_train)),
    output_distribution="normal",
    random_state=42
)

z_train = qt.fit_transform(y_train.reshape(-1, 1)).ravel()
z_val = qt.transform(y_val.reshape(-1, 1)).ravel()


# ============================================================
# 5. STAGE 1: REGIME CLASSIFIER (X -> regime)
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
# 6. STAGE 2: REGRESSORS PER REGIME (ON TRANSFORMED TARGET)
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

    model.fit(X_train[idx], z_train[idx])
    regressors[reg] = model


# ============================================================
# 7. SOFT MIXTURE-OF-EXPERTS VALIDATION (Z-SPACE)
# ============================================================
# Training predictions
z_train_pred = np.zeros(len(X_train))
r_train_proba = clf.predict_proba(X_train)

for i in range(len(X_train)):
    probs = r_train_proba[i]
    z_train_pred[i] = (
        probs[0] * regressors[0].predict(X_train.iloc[[i]])[0]
        + probs[1] * regressors[1].predict(X_train.iloc[[i]])[0]
    )

# Validation predictions
z_val_pred = np.zeros(len(X_val))

for i in range(len(X_val)):
    probs = r_val_proba[i]
    z_val_pred[i] = (
        probs[0] * regressors[0].predict(X_val.iloc[[i]])[0]
        + probs[1] * regressors[1].predict(X_val.iloc[[i]])[0]
    )

# Inverse transform back to original target space
y_train_pred = qt.inverse_transform(z_train_pred.reshape(-1, 1)).ravel()
y_val_pred = qt.inverse_transform(z_val_pred.reshape(-1, 1)).ravel()

# Metrics
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
clf.fit(X, regime)

z_full = qt.fit_transform(y.reshape(-1, 1)).ravel()

for reg in [0, 1]:
    regressors[reg].fit(X[regime == reg], z_full[regime == reg])


# ============================================================
# 9. EVALUATION DATA PREDICTION (NO LABELS)
# ============================================================
eval_regime_proba = clf.predict_proba(X_eval)

print("\nEval regime distribution:",
      np.mean(eval_regime_proba, axis=0))

confidence = np.max(eval_regime_proba, axis=1)
print(f"Low-confidence samples (<0.6): {(confidence < 0.6).mean():.2%}")

z_eval_pred = np.zeros(len(X_eval))

for i in range(len(X_eval)):
    probs = eval_regime_proba[i]
    z_eval_pred[i] = (
        probs[0] * regressors[0].predict(X_eval.iloc[[i]])[0]
        + probs[1] * regressors[1].predict(X_eval.iloc[[i]])[0]
    )

y_eval_pred = qt.inverse_transform(z_eval_pred.reshape(-1, 1)).ravel()


# ============================================================
# 10. SAVE OUTPUT
# ============================================================
output_file = f"EVAL_target01_{PROBLEM_NUM}_GMM_mixture_catboost_QT.csv"
pd.DataFrame({"target01": y_eval_pred}).to_csv(output_file, index=False)

print(f"\nPredictions saved to: {output_file}")
print("Sample predictions:", y_eval_pred[:10])
