# ============================================================
# 0. Imports
# ============================================================
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import wilcoxon

from catboost import CatBoostClassifier, CatBoostRegressor


# ============================================================
# 1. Configuration
# ============================================================
PROBLEM_NUM = 36
K = 5

SELECTED_FEATURES = [ 
    'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253', 
    'feat_143', 'feat_221', 'feat_220', 'feat_160', 'feat_266', 
    'feat_138', 'feat_47', 'feat_203',
]

# ---- Single CatBoost parameters (your tuned model)
SINGLE_PARAMS = {
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

# ---- MoE parameters
GATE_PARAMS = dict(
    iterations=600,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    random_seed=42,
    verbose=False
)

EXPERT_PARAMS = dict(
    iterations=800,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)


# ============================================================
# 2. Load data
# ============================================================
X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"].values

X = X[SELECTED_FEATURES]

print(f"Data shape: {X.shape}")


# ============================================================
# 3. OOF setup
# ============================================================
kf = KFold(n_splits=K, shuffle=True, random_state=42)

oof_single = np.zeros(len(X))
oof_moe = np.zeros(len(X))


# ============================================================
# 4. OOF loop
# ============================================================
for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/{K}")

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # --------------------------------------------------------
    # A) SINGLE CATBOOST
    # --------------------------------------------------------
    single = CatBoostRegressor(**SINGLE_PARAMS)
    single.fit(X_tr, y_tr)
    oof_single[va_idx] = single.predict(X_va)

    # --------------------------------------------------------
    # B) MIXTURE OF EXPERTS
    # --------------------------------------------------------
    # --- Regime discovery (TRAIN FOLD ONLY)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(y_tr.reshape(-1, 1))

    r_tr = gmm.predict(y_tr.reshape(-1, 1))
    means = gmm.means_.ravel()
    order = np.argsort(means)
    r_tr = np.array([np.where(order == r)[0][0] for r in r_tr])

    # --- Gate
    gate = CatBoostClassifier(**GATE_PARAMS)
    gate.fit(X_tr, r_tr)

    # --- Experts
    experts = {}
    for reg in [0, 1]:
        idx = r_tr == reg
        model = CatBoostRegressor(**EXPERT_PARAMS)
        model.fit(X_tr[idx], y_tr[idx])
        experts[reg] = model

    # --- Predict fold
    proba = gate.predict_proba(X_va)
    pred = (
        proba[:, 0] * experts[0].predict(X_va)
        + proba[:, 1] * experts[1].predict(X_va)
    )

    oof_moe[va_idx] = pred


# ============================================================
# 5. OOF METRICS
# ============================================================
rmse_single = np.sqrt(mean_squared_error(y, oof_single))
rmse_moe = np.sqrt(mean_squared_error(y, oof_moe))

r2_single = r2_score(y, oof_single)
r2_moe = r2_score(y, oof_moe)

print("\n" + "="*60)
print("OOF PERFORMANCE COMPARISON")
print("="*60)

print(f"Single CatBoost  | R²: {r2_single:.6f} | RMSE: {rmse_single:.6f}")
print(f"MoE              | R²: {r2_moe:.6f} | RMSE: {rmse_moe:.6f}")

print("="*60)


# ============================================================
# 6. PAIRED ERROR TEST (THE DECISIVE TEST)
# ============================================================
err_single = (y - oof_single) ** 2
err_moe = (y - oof_moe) ** 2
delta = err_single - err_moe

print("\nPAIRED PER-SAMPLE ERROR COMPARISON (OOF)")
print("-"*60)
print(f"Mean ΔMSE (single − MoE): {delta.mean():+.8f}")
print(f"Median ΔMSE:             {np.median(delta):+.8f}")
print(f"% samples MoE better:    {(delta > 0).mean()*100:.2f}%")

# Wilcoxon signed-rank test
stat, p_value = wilcoxon(delta)

print("\nWilcoxon signed-rank test:")
print(f"Statistic: {stat:.6f}")
print(f"P-value:   {p_value:.6f}")

if p_value < 0.05:
    print("✓ Difference is statistically significant")
else:
    print("✓ Models are statistically tied (MoE not worse)")

print("="*60)
