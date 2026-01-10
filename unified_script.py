# ============================================================
# 0. Imports
# ============================================================
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.mixture import GaussianMixture

from catboost import CatBoostRegressor, CatBoostClassifier


# ============================================================
# Global config
# ============================================================
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

PROBLEM_NUM = 36
RANDOM_STATE = 42


# ============================================================
# 1. Load data
# ============================================================
X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"]
X_eval = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv")

feature_names = X.columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
)

print(f"X: {X.shape}, y: {y.shape}")


# ============================================================
# 2. Baseline model
# ============================================================
base_model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    random_seed=RANDOM_STATE,
    verbose=False
)
base_model.fit(X_train, y_train)

baseline_r2 = r2_score(y_val, base_model.predict(X_val))
print(f"Baseline R²: {baseline_r2:.4f}")


# ============================================================
# 3. Manual permutation importance
# ============================================================
def permutation_importance_manual(model, X_val, y_val, metric, n_repeats=5, random_state=42):
    rng = np.random.RandomState(random_state)
    baseline_score = metric(y_val, model.predict(X_val))
    importances = []

    for col in tqdm(X_val.columns, desc="Permuting features"):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            scores.append(baseline_score - metric(y_val, model.predict(X_perm)))
        importances.append(np.mean(scores))

    return np.array(importances)


perm_importance = permutation_importance_manual(
    base_model, X_val, y_val, r2_score, n_repeats=5
)

perm_df = (
    pd.DataFrame({"feature": feature_names, "importance": perm_importance})
    .sort_values("importance", ascending=False)
)

print("\nTop 15 features:")
print(perm_df.head(15))


# ============================================================
# 4. Stability selection
# ============================================================
def permutation_run(seed):
    model = CatBoostRegressor(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False
    )
    model.fit(X_train, y_train)

    imp = permutation_importance_manual(
        model, X_val, y_val, r2_score, n_repeats=3, random_state=seed
    )
    return [f for f, v in zip(feature_names, imp) if v > 0]


all_selected = []
for seed in range(5):
    all_selected.extend(permutation_run(seed))

counts = Counter(all_selected)
stable_features = [f for f, c in counts.items() if c >= 3]

print(f"\nStable features: {len(stable_features)}")


# ============================================================
# 5. Feature ranking & ablation
# ============================================================
ranked_features = (
    perm_df[perm_df.feature.isin(stable_features)]
    .sort_values("importance", ascending=False)["feature"]
    .tolist()
)

scores = []
for k in tqdm(range(1, len(ranked_features) + 1), desc="Ablation"):
    feats = ranked_features[:k]
    model = CatBoostRegressor(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False
    )
    model.fit(X_train[feats], y_train)
    scores.append(r2_score(y_val, model.predict(X_val[feats])))

plt.plot(range(1, len(scores) + 1), scores, marker="o")
plt.axhline(baseline_r2, linestyle="--", label="Baseline")
plt.legend()
plt.show()

best_k = int(np.argmax(scores)) + 1
SELECTED_FEATURES = ranked_features[:best_k]

print(f"\nBest k = {best_k}")
print("Selected features:", SELECTED_FEATURES)


# ============================================================
# 6. Regime discovery (TRAIN TARGET ONLY)
# ============================================================
X = X[SELECTED_FEATURES]
X_eval = X_eval[SELECTED_FEATURES]

gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE)
gmm.fit(y_train.values.reshape(-1, 1))

r_train = gmm.predict(y_train.values.reshape(-1, 1))
r_val = gmm.predict(y_val.values.reshape(-1, 1))

means = gmm.means_.ravel()
order = np.argsort(means)

r_train = np.array([np.where(order == r)[0][0] for r in r_train])
r_val = np.array([np.where(order == r)[0][0] for r in r_val])


# ============================================================
# 7. Regime classifier
# ============================================================
clf = CatBoostClassifier(
    iterations=600,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    random_seed=RANDOM_STATE,
    verbose=False
)
clf.fit(X_train[SELECTED_FEATURES], r_train)

r_val_pred = clf.predict(X_val[SELECTED_FEATURES]).astype(int)
val_proba = clf.predict_proba(X_val[SELECTED_FEATURES])


# ============================================================
# 8. Regime regressors
# ============================================================
regressors = {}
for reg in [0, 1]:
    idx = r_train == reg
    model = CatBoostRegressor(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False
    )
    model.fit(X_train.loc[idx, SELECTED_FEATURES], y_train.loc[idx])
    regressors[reg] = model


# ============================================================
# 9. Mixture-of-Experts prediction
# ============================================================
def moe_predict(X):
    p = clf.predict_proba(X)
    return (
        p[:, 0] * regressors[0].predict(X)
        + p[:, 1] * regressors[1].predict(X)
    )


y_val_pred = moe_predict(X_val[SELECTED_FEATURES])


# ============================================================
# 10. ERROR ANALYSIS & VISUALIZATION  ✅ FULLY RESTORED
# ============================================================
val_errors = np.abs(y_val - y_val_pred)
error_threshold = np.percentile(val_errors, 90)
high_error_mask = val_errors > error_threshold

# ----- GMM PDFs -----
y_range = np.linspace(y_val.min(), y_val.max(), 1000)
means_plot = gmm.means_.ravel()[order]
covs_plot = gmm.covariances_.ravel()[order]
weights_plot = gmm.weights_[order]

component_pdfs = []
gmm_pdf = np.zeros_like(y_range)

for mean, cov, w in zip(means_plot, covs_plot, weights_plot):
    pdf = w * (1 / np.sqrt(2 * np.pi * cov)) * np.exp(-0.5 * (y_range - mean) ** 2 / cov)
    component_pdfs.append(pdf)
    gmm_pdf += pdf

overlap_threshold = 0.1 * np.max(component_pdfs)
overlap_mask = np.all([pdf > overlap_threshold for pdf in component_pdfs], axis=0)

# ----- Plot -----
fig = plt.figure(figsize=(20, 12))

ax1 = plt.subplot(2, 3, 1)
ax1.hist(y_val, bins=50, density=True, alpha=0.3)
for i, pdf in enumerate(component_pdfs):
    ax1.plot(y_range, pdf, '--', label=f"Regime {i}")
ax1.plot(y_range, gmm_pdf, 'k-', lw=3)
ax1.hist(y_val[high_error_mask], bins=30, density=True, alpha=0.6, color='red')
ax1.set_title("Target distribution & high-error samples")
ax1.legend()

ax2 = plt.subplot(2, 3, 2)
ax2.scatter(y_val, val_errors, alpha=0.5)
ax2.axhline(error_threshold, linestyle="--")
ax2.set_title("Error vs target")

ax3 = plt.subplot(2, 3, 3)
ax3.scatter(y_val, y_val_pred, alpha=0.5)
ax3.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--')
ax3.set_title("Actual vs predicted")

plt.tight_layout()
plt.savefig(f"error_analysis_{PROBLEM_NUM}.png", dpi=150)
plt.close()

# Save high-error samples
high_error_df = pd.DataFrame({
    "actual": y_val[high_error_mask],
    "predicted": y_val_pred[high_error_mask],
    "error": val_errors[high_error_mask],
    "regime": r_val[high_error_mask],
    "confidence": val_proba[high_error_mask].max(axis=1)
})
high_error_df.to_csv(f"high_error_samples_{PROBLEM_NUM}.csv", index=False)


# ============================================================
# 11. FINAL TRAINING ON FULL DATA
# ============================================================
gmm_full = GaussianMixture(n_components=2, random_state=RANDOM_STATE)
gmm_full.fit(y.values.reshape(-1, 1))

regime_full = gmm_full.predict(y.values.reshape(-1, 1))
means_full = gmm_full.means_.ravel()
order_full = np.argsort(means_full)
regime_full = np.array([np.where(order_full == r)[0][0] for r in regime_full])

clf.fit(X, regime_full)

for reg in [0, 1]:
    regressors[reg].fit(X[regime_full == reg], y[regime_full == reg])


# ============================================================
# 12. EVAL prediction
# ============================================================
y_eval_pred = moe_predict(X_eval)

pd.DataFrame({"target01": y_eval_pred}).to_csv(
    f"EVAL_target01_{PROBLEM_NUM}.csv", index=False
)

print("✓ DONE")
