import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

# ===============================
# CONFIG
# ===============================
PROBLEM_NUM = 36
RANDOM_STATE = 42
TOP_K_LIST = [5, 10, 15, 20, 25, 30, 40, 50]

# ===============================
# LOAD DATA
# ===============================
X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"].values

print(f"Data shape: X={X.shape}, y={y.shape}")

# ===============================
# STANDARDIZE (for ANOVA only)
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# ANOVA F-TEST
# ===============================
F_vals, _ = f_regression(X_scaled, y)

anova_df = pd.DataFrame({
    "feature": X.columns,
    "F_value": F_vals
}).sort_values("F_value", ascending=False).reset_index(drop=True)

# ===============================
# CV SETUP
# ===============================
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def cv_eval(X_sub, y):
    mses, maes = [], []
    for tr, va in kf.split(X_sub):
        model = LGBMRegressor(
            objective="regression",
            n_estimators=800,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=-1
        )
        model.fit(X_sub.iloc[tr], y[tr])
        pred = model.predict(X_sub.iloc[va])
        mses.append(mean_squared_error(y[va], pred))
        maes.append(mean_absolute_error(y[va], pred))
    return np.mean(mses), np.std(mses), np.mean(maes)

# ===============================
# TOP-K EVALUATION
# ===============================
results = []

print("\n=== TOP-K FEATURE SUBSET CV ===")
for k in TOP_K_LIST:
    feats = anova_df["feature"].iloc[:k].tolist()
    mse_mean, mse_std, mae_mean = cv_eval(X[feats], y)

    results.append((k, mse_mean, mse_std, mae_mean))
    print(f"Top-{k:<2d} | MSE: {mse_mean:.6f} ± {mse_std:.6f} | MAE: {mae_mean:.6f}")

# ===============================
# SELECT BEST K
# ===============================
results_df = pd.DataFrame(
    results,
    columns=["K", "MSE", "MSE_std", "MAE"]
).sort_values("MSE")

best_row = results_df.iloc[0]
best_k = int(best_row["K"])

print("\n" + "="*70)
print(f"BEST FEATURE COUNT: Top-{best_k}")
print(f"MSE: {best_row['MSE']:.6f} ± {best_row['MSE_std']:.6f}")
print(f"MAE: {best_row['MAE']:.6f}")
print("="*70)

# ===============================
# FINAL FEATURE LIST
# ===============================
best_features = anova_df["feature"].iloc[:best_k].tolist()

print("\nFINAL SELECTED FEATURES:")
for i, f in enumerate(best_features, 1):
    f_val = anova_df.loc[anova_df["feature"] == f, "F_value"].values[0]
    print(f"{i:2d}. {f:<15s} (F={f_val:.3f})")

print("\n=== DONE ===")

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ===============================
# TRAIN / VALIDATION SPLIT
# ===============================
X_final = X[best_features]

X_train, X_val, y_train, y_val = train_test_split(
    X_final,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    shuffle=True
)

print("\n" + "="*70)
print("TRAIN / VALIDATION SPLIT")
print("="*70)
print(f"Train samples: {X_train.shape[0]}")
print(f"Val samples:   {X_val.shape[0]}")

# ===============================
# FINAL LIGHTGBM MODEL
# ===============================
final_model = LGBMRegressor(
    objective="regression",
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    verbosity=-1
)

# ===============================
# TRAIN
# ===============================
final_model.fit(X_train, y_train)

# ===============================
# PREDICTIONS
# ===============================
y_train_pred = final_model.predict(X_train)
y_val_pred = final_model.predict(X_val)

# ===============================
# METRICS
# ===============================
def report_metrics(y_true, y_pred, label):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"{label:<10s} | MSE: {mse:.6f} | MAE: {mae:.6f} | R²: {r2:.6f}")
    return mse, mae, r2

print("\n" + "="*70)
print("FINAL LIGHTGBM PERFORMANCE (SELECTED FEATURES)")
print("="*70)

train_metrics = report_metrics(y_train, y_train_pred, "TRAIN")
val_metrics   = report_metrics(y_val, y_val_pred, "VALID")

print("\nOverfitting analysis:")
print(f"Δ MSE: {train_metrics[0] - val_metrics[0]:+.6f}")
print(f"Δ MAE: {train_metrics[1] - val_metrics[1]:+.6f}")
print(f"Δ R²:  {train_metrics[2] - val_metrics[2]:+.6f}")

print("\n=== COMPLETE ===")
