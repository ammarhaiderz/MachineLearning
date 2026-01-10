import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

# ===============================
# Load data
# ===============================
PROBLEM_NUM = 36
X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"].values

print(f"X shape: {X.shape}, y shape: {y.shape}")

# ===============================
# Cross-validation setup
# ===============================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_eval(model, X, y):
    mses, maes = [], []

    for fold, (tr, va) in enumerate(kf.split(X), 1):
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[va])

        mses.append(mean_squared_error(y[va], pred))
        maes.append(mean_absolute_error(y[va], pred))

    return (
        np.mean(mses), np.std(mses),
        np.mean(maes), np.std(maes)
    )

# ===============================
# Models
# ===============================
rmse_model = LGBMRegressor(
    objective="regression",
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1
)

mae_model = LGBMRegressor(
    objective="regression_l1",
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1
)

q50_model = LGBMRegressor(
    objective="quantile",
    alpha=0.5,
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1
)

# ===============================
# Evaluation
# ===============================
results = []

for name, mdl in [
    ("RMSE", rmse_model),
    ("MAE", mae_model),
    ("Q50", q50_model)
]:
    mse_mean, mse_std, mae_mean, mae_std = cv_eval(mdl, X, y)

    results.append({
        "Model": name,
        "MSE_mean": mse_mean,
        "MSE_std": mse_std,
        "MAE_mean": mae_mean,
        "MAE_std": mae_std
    })

# ===============================
# Print summary
# ===============================
print("\n" + "=" * 70)
print("CROSS-VALIDATION SUMMARY (5-FOLD)")
print("=" * 70)

for r in results:
    print(
        f"{r['Model']:>4s} | "
        f"MSE: {r['MSE_mean']:.6f} ± {r['MSE_std']:.6f} | "
        f"MAE: {r['MAE_mean']:.6f} ± {r['MAE_std']:.6f}"
    )

print("=" * 70)
print("DONE")
