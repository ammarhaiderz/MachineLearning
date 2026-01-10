# ============================================================
# 0. Imports
# ============================================================
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from catboost import CatBoostClassifier, CatBoostRegressor


# ============================================================
# 1. Configuration
# ============================================================
PROBLEM_NUM = 36

SELECTED_FEATURES = [
    'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253', 
    'feat_143', 'feat_221', 'feat_220', 'feat_160', 'feat_266', 
    'feat_138', 'feat_47', 'feat_203',
]

# 🔴 PASTE YOUR BEST OPTUNA PARAMS HERE 🔴
BEST_CB_PARAMS = {
    'iterations': 1711,
    'depth': 8,
    'learning_rate': 0.08773275868829458,
    'l2_leaf_reg': 7.791616137902223,
    'random_strength': 1.9831160164613875,
    'bagging_temperature': 0.13907763817404983,
    'border_count': 209,
    'min_data_in_leaf': 16,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': False,
    
}

# ============================================================
# 2. Load data
# ============================================================
X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"].values
X_eval = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv")

# Feature selection
X = X[SELECTED_FEATURES]
X_eval = X_eval[SELECTED_FEATURES]

print(f"X: {X.shape}, y: {y.shape}")
print(f"X_eval: {X_eval.shape}")


# ============================================================
# 3. Train / validation split
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain samples: {len(X_train)}")
print(f"Val samples:   {len(X_val)}")


# ============================================================
# 4. TARGET-DRIVEN REGIME DISCOVERY (GMM on y_train)
# ============================================================
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(y_train.reshape(-1, 1))

r_train = gmm.predict(y_train.reshape(-1, 1))
r_val = gmm.predict(y_val.reshape(-1, 1))

# Ensure ordered regimes
means = gmm.means_.ravel()
order = np.argsort(means)

r_train = np.array([np.where(order == r)[0][0] for r in r_train])
r_val = np.array([np.where(order == r)[0][0] for r in r_val])

print("\nGMM regime means:", means[order])
print("Train regime split:", np.bincount(r_train) / len(r_train))
print("Val regime split:  ", np.bincount(r_val) / len(r_val))


# ============================================================
# 5. REGIME CLASSIFIER (X → regime)
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

regime_acc = (r_val_pred == r_val).mean()
print(f"\nRegime classification accuracy (VAL): {regime_acc:.4f}")


# ============================================================
# 6. OPTUNA-TUNED REGRESSORS PER REGIME
# ============================================================
regressors = {}

for reg in [0, 1]:
    idx = r_train == reg
    model = CatBoostRegressor(**BEST_CB_PARAMS)
    model.fit(X_train[idx], y_train[idx])
    regressors[reg] = model


# ============================================================
# 7. SOFT MIXTURE-OF-EXPERTS PREDICTION
# ============================================================
def moe_predict(X_input):
    probs = clf.predict_proba(X_input)
    preds = np.zeros(len(X_input))
    for i in range(len(X_input)):
        preds[i] = (
            probs[i, 0] * regressors[0].predict(X_input.iloc[[i]])[0]
            + probs[i, 1] * regressors[1].predict(X_input.iloc[[i]])[0]
        )
    return preds


y_train_pred = moe_predict(X_train)
y_val_pred = moe_predict(X_val)


# ============================================================
# 8. GLOBAL METRICS
# ============================================================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print("\n" + "="*50)
print("GLOBAL PERFORMANCE")
print("="*50)
print(f"Train R²: {r2_score(y_train, y_train_pred):.4f}")
print(f"Train RMSE: {rmse(y_train, y_train_pred):.4f}")
print(f"Val R²:   {r2_score(y_val, y_val_pred):.4f}")
print(f"Val RMSE: {rmse(y_val, y_val_pred):.4f}")


# ============================================================
# 9. REGIME-WISE (UNIMODAL) VALIDATION
# ============================================================
print("\n" + "="*50)
print("UNIMODAL REGIME DIAGNOSTIC (VAL)")
print("="*50)

for reg in [0, 1]:
    idx = r_val == reg
    print(f"\nRegime {reg}:")
    print(f" Samples: {idx.sum()}")
    print(f" Mean y:  {y_val[idx].mean():.4f}")
    print(f" Std  y:  {y_val[idx].std():.4f}")
    print(f" R²:      {r2_score(y_val[idx], y_val_pred[idx]):.4f}")
    print(f" RMSE:    {rmse(y_val[idx], y_val_pred[idx]):.4f}")


# ============================================================
# 10. ORACLE REGIME REGRESSION (UPPER BOUND)
# ============================================================
oracle_pred = np.zeros(len(X_val))

for reg in [0, 1]:
    idx = r_val == reg
    oracle_pred[idx] = regressors[reg].predict(X_val[idx])

print("\n" + "="*50)
print("ORACLE REGIME REGRESSION")
print("="*50)
print(f"Oracle R²:   {r2_score(y_val, oracle_pred):.4f}")
print(f"Oracle RMSE:{rmse(y_val, oracle_pred):.4f}")


# ============================================================
# 11. FINAL TRAINING ON FULL DATA
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
# 12. EVAL PREDICTION
# ============================================================
y_eval_pred = moe_predict(X_eval)

out_file = f"EVAL_target01_{PROBLEM_NUM}_MoE_optuna.csv"
pd.DataFrame({"target01": y_eval_pred}).to_csv(out_file, index=False)

print(f"\nSaved predictions → {out_file}")
print("Sample predictions:", y_eval_pred[:10])


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

print("\n--- Regime Classifier ---")
for f, v in sorted(zip(SELECTED_FEATURES, clf.get_feature_importance()),
                   key=lambda x: -x[1])[:10]:
    print(f"{f:25s} {v:8.3f}")

for reg in [0, 1]:
    print(f"\n--- Regressor Regime {reg} ---")
    imp = regressors[reg].get_feature_importance()
    for f, v in sorted(zip(SELECTED_FEATURES, imp),
                       key=lambda x: -x[1])[:10]:
        print(f"{f:25s} {v:8.3f}")
