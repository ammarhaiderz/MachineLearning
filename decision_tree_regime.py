import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
PROBLEM_NUM = 36
N_SPLITS = 5
RANDOM_STATE = 42

# ===============================
# LOAD DATA
# ===============================
X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"].values

print(f"Data shape: X={X.shape}, y={y.shape}")

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# ===============================
# CV EVALUATION FUNCTION
# ===============================
def cv_eval(model, X, y):
    mses, maes = [], []
    for tr, va in kf.split(X):
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[va])
        mses.append(mean_squared_error(y[va], pred))
        maes.append(mean_absolute_error(y[va], pred))
    return np.mean(mses), np.std(mses), np.mean(maes), np.std(maes)

# ===============================
# BASELINE MODEL
# ===============================
base_model = LGBMRegressor(
    objective="regression",
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    verbosity=-1
)

print("\n=== BASELINE (ALL FEATURES) ===")
mse, mse_std, mae, mae_std = cv_eval(base_model, X, y)
print(f"MSE: {mse:.6f} ± {mse_std:.6f}")
print(f"MAE: {mae:.6f} ± {mae_std:.6f}")

# ===============================
# FEATURE IMPORTANCE
# ===============================
base_model.fit(X, y)
importances = pd.Series(base_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False)

# ===============================
# TOP-K FEATURE SUBSETS
# ===============================
print("\n=== TOP-K FEATURE SUBSETS ===")
for k in [5, 10, 20, 50]:
    feats = top_features.head(k).index
    mse, mse_std, mae, mae_std = cv_eval(base_model, X[feats], y)
    print(f"Top-{k:<2d} | MSE: {mse:.6f} ± {mse_std:.6f} | MAE: {mae:.6f}")

# ===============================
# RANDOM SUBSETS FROM TOP-50
# ===============================
print("\n=== RANDOM FEATURE SUBSETS (from top-50) ===")
rng = np.random.default_rng(RANDOM_STATE)
top50 = top_features.head(50).index.tolist()

for i in range(5):
    subset = rng.choice(top50, size=10, replace=False)
    mse, mse_std, mae, mae_std = cv_eval(base_model, X[subset], y)
    print(f"Subset {i+1} | MSE: {mse:.6f} ± {mse_std:.6f}")

# ===============================
# BOOTSTRAP STABILITY TEST
# ===============================
print("\n=== BOOTSTRAP STABILITY (80% samples) ===")
boot_mses = []
for i in range(10):
    idx = rng.choice(len(X), size=int(0.8 * len(X)), replace=False)
    mse, _, _, _ = cv_eval(base_model, X.iloc[idx], y[idx])
    boot_mses.append(mse)

print(f"Bootstrap MSE mean: {np.mean(boot_mses):.6f}")
print(f"Bootstrap MSE std : {np.std(boot_mses):.6f}")

# ===============================
# UNSUPERVISED GMM CLUSTERING (DIAGNOSTIC)
# ===============================
print("\n=== GMM CLUSTERING DIAGNOSTIC (X only) ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE)
clusters = gmm.fit_predict(X_scaled)

for c in [0, 1]:
    print(f"Cluster {c}: n={np.sum(clusters==c)}, "
          f"target mean={y[clusters==c].mean():.4f}, "
          f"std={y[clusters==c].std():.4f}")

# ===============================
# QUANTILE REGRESSION TEST
# ===============================
print("\n=== QUANTILE REGRESSION TEST ===")
for q in [0.1, 0.5, 0.9]:
    q_model = LGBMRegressor(
        objective="quantile",
        alpha=q,
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=-1
    )
    mse, mse_std, mae, mae_std = cv_eval(q_model, X, y)
    print(f"q={q:.1f} | MSE: {mse:.6f} | MAE: {mae:.6f}")

# ===============================
# RESIDUAL STRUCTURE (ONE FOLD)
# ===============================
print("\n=== RESIDUAL STRUCTURE CHECK ===")
tr, va = next(kf.split(X))
base_model.fit(X.iloc[tr], y[tr])
pred = base_model.predict(X.iloc[va])
resid = y[va] - pred

print(f"Residual mean: {resid.mean():.6f}")
print(f"Residual std : {resid.std():.6f}")
print(f"|Residual| mean: {np.abs(resid).mean():.6f}")

print("\n=== DONE ===")
