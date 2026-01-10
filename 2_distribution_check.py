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

X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"].values
X_eval = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv")

print(f"Train X: {X.shape}, y: {y.shape}")
print(f"Eval  X: {X_eval.shape}")


# ============================================================
# 2. Train / validation split
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"\nTrain split: {len(X_train)} samples")
print(f"Val split:   {len(X_val)} samples")


# ============================================================
# 3. TARGET-DRIVEN REGIME DISCOVERY (GMM ON y_train)
# ============================================================
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(y_train.reshape(-1, 1))

r_train = gmm.predict(y_train.reshape(-1, 1))
r_val = gmm.predict(y_val.reshape(-1, 1))

# Enforce consistent ordering
means = gmm.means_.ravel()
order = np.argsort(means)

r_train = np.array([np.where(order == r)[0][0] for r in r_train])
r_val = np.array([np.where(order == r)[0][0] for r in r_val])

print("\nGMM regime means (from training):", means[order])
print("Train regime proportions:", np.bincount(r_train) / len(r_train))
print("Val   regime proportions:", np.bincount(r_val) / len(r_val))


# ============================================================
# 4. STAGE 1: REGIME CLASSIFIER (X → regime)
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
# 6. SOFT MIXTURE-OF-EXPERTS PREDICTION
# ============================================================
def moe_predict(X_input):
    probs = clf.predict_proba(X_input)
    y_pred = np.zeros(len(X_input))

    for i in range(len(X_input)):
        y_pred[i] = (
            probs[i, 0] * regressors[0].predict(X_input.iloc[[i]])[0]
            + probs[i, 1] * regressors[1].predict(X_input.iloc[[i]])[0]
        )
    return y_pred


y_train_pred = moe_predict(X_train)
y_val_pred = moe_predict(X_val)


# ============================================================
# 7. GLOBAL METRICS
# ============================================================
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = root_mean_squared_error(y_train, y_train_pred)

val_r2 = r2_score(y_val, y_val_pred)
val_rmse = root_mean_squared_error(y_val, y_val_pred)

print(f"\n{'='*50}")
print("GLOBAL MODEL PERFORMANCE")
print(f"{'='*50}")
print(f"Train R²:   {train_r2:.4f}")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Val R²:     {val_r2:.4f}")
print(f"Val RMSE:   {val_rmse:.4f}")
print(f"{'='*50}")


# ============================================================
# 8. REGIME-WISE (UNIMODAL) VALIDATION DIAGNOSTIC
# ============================================================
print("\n" + "="*50)
print("UNIMODAL REGIME DIAGNOSTIC (VALIDATION)")
print("="*50)

for reg in [0, 1]:
    idx = r_val == reg

    r2 = r2_score(y_val[idx], y_val_pred[idx])
    rmse = root_mean_squared_error(y_val[idx], y_val_pred[idx])

    print(f"\nRegime {reg}:")
    print(f"  Samples:     {idx.sum()}")
    print(f"  Mean target: {y_val[idx].mean():.4f}")
    print(f"  Std  target: {y_val[idx].std():.4f}")
    print(f"  R²:          {r2:.4f}")
    print(f"  RMSE:        {rmse:.4f}")


# ============================================================
# 9. ORACLE REGIME REGRESSION (NO CLASSIFIER)
# ============================================================
oracle_pred = np.zeros(len(X_val))

for reg in [0, 1]:
    idx = r_val == reg
    oracle_pred[idx] = regressors[reg].predict(X_val[idx])

oracle_r2 = r2_score(y_val, oracle_pred)
oracle_rmse = root_mean_squared_error(y_val, oracle_pred)

print("\n" + "="*50)
print("ORACLE REGIME REGRESSION (UPPER BOUND)")
print("="*50)
print(f"Oracle R²:   {oracle_r2:.4f}")
print(f"Oracle RMSE:{oracle_rmse:.4f}")


# ============================================================
# 10. FINAL TRAINING ON FULL DATA
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
# 11. EVAL DATA PREDICTION
# ============================================================
y_eval_pred = moe_predict(X_eval)

output_file = f"EVAL_target01_{PROBLEM_NUM}_GMM_mixture_catboost.csv"
pd.DataFrame({"target01": y_eval_pred}).to_csv(output_file, index=False)

print(f"\nPredictions saved to: {output_file}")
print("Sample predictions:", y_eval_pred[:10])


# ============================================================
# 12. FEATURE IMPORTANCE ANALYSIS
# ============================================================
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Classifier (Regime Predictor) Feature Importance
clf_importance = clf.get_feature_importance()
clf_feature_names = X.columns
clf_importance_df = pd.DataFrame({
    'feature': clf_feature_names,
    'importance': clf_importance
}).sort_values('importance', ascending=False)

print("\n--- REGIME CLASSIFIER (Top 10 Features) ---")
for idx, row in clf_importance_df.head(10).iterrows():
    print(f"{row['feature']:30s} {row['importance']:8.4f}")

# Regressor Regime 0 Feature Importance
reg0_importance = regressors[0].get_feature_importance()
reg0_importance_df = pd.DataFrame({
    'feature': clf_feature_names,
    'importance': reg0_importance
}).sort_values('importance', ascending=False)

print("\n--- REGRESSOR REGIME 0 (Top 10 Features) ---")
for idx, row in reg0_importance_df.head(10).iterrows():
    print(f"{row['feature']:30s} {row['importance']:8.4f}")

# Regressor Regime 1 Feature Importance
reg1_importance = regressors[1].get_feature_importance()
reg1_importance_df = pd.DataFrame({
    'feature': clf_feature_names,
    'importance': reg1_importance
}).sort_values('importance', ascending=False)

print("\n--- REGRESSOR REGIME 1 (Top 10 Features) ---")
for idx, row in reg1_importance_df.head(10).iterrows():
    print(f"{row['feature']:30s} {row['importance']:8.4f}")

print("\n" + "="*60)
