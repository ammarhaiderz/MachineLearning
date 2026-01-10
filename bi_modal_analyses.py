# ============================================================
# Kaggle-style diagnostic pipeline for multimodal targets
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans

from lightgbm import LGBMRegressor

sns.set(style="whitegrid")


# ============================================================
# 0. Load data
# ============================================================
PROBLEM_NUM = 36

X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"]

print(f"X: {X.shape}, y: {y.shape}")


# ============================================================
# Phase 1 — Sanity & leakage checks
# ============================================================

# ----------------------------
# 1️⃣ Target distribution by CV fold
# ----------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

plt.figure(figsize=(10, 5))
for fold, (_, val_idx) in enumerate(kf.split(X)):
    sns.kdeplot(y.iloc[val_idx], label=f"Fold {fold}", linewidth=2)

plt.title("Target distribution per CV fold")
plt.legend()
plt.show()

print("Check visually: bimodality should appear in every fold.\n")


# ----------------------------
# 2️⃣ Target vs index (ordering artifacts)
# ----------------------------
plt.figure(figsize=(12, 4))
plt.scatter(range(len(y)), y, s=2, alpha=0.5)
plt.title("Target vs sample index")
plt.xlabel("Index")
plt.ylabel("Target")
plt.show()

print("Check: no blocks, no boundary effects.\n")


# ============================================================
# Phase 2 — Conditional structure tests (MOST IMPORTANT)
# ============================================================

# ----------------------------
# 3️⃣ Target vs top features / PCA
# ----------------------------
print("Running PCA for visualization...")
pca = PCA(n_components=2)
Xp = pca.fit_transform(X)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(Xp[:, 0], y, s=3, alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("Target")
plt.title("Target vs PC1")

plt.subplot(1, 2, 2)
plt.scatter(Xp[:, 1], y, s=3, alpha=0.5)
plt.xlabel("PC2")
plt.ylabel("Target")
plt.title("Target vs PC2")
plt.tight_layout()
plt.show()

print(
    "Interpretation:\n"
    " - Vertical separation → conditional structure exists\n"
    " - No separation → averaging risk\n"
)


# ----------------------------
# 4️⃣ Simple model → residuals vs predictions
# ----------------------------
model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

X_tr, X_va = X.iloc[:8000], X.iloc[8000:]
y_tr, y_va = y.iloc[:8000], y.iloc[8000:]

model.fit(X_tr, y_tr)
pred_va = model.predict(X_va)
residuals = y_va - pred_va

plt.figure(figsize=(6, 5))
plt.scatter(pred_va, residuals, s=5, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Prediction")
plt.ylabel("Residual")
plt.title("Residuals vs Predictions")
plt.show()

print(
    "Interpretation:\n"
    " - Two horizontal bands → unresolved modes\n"
    " - Single cloud → model captured structure\n"
)


# ============================================================
# Phase 3 — Metric sensitivity tests (Kaggle-relevant)
# ============================================================

# ----------------------------
# 5️⃣ Loss function swap (MSE vs MAE vs Huber)
# ----------------------------
losses = {
    "MSE": dict(objective="regression"),
    "MAE": dict(objective="regression_l1"),
    "Huber": dict(objective="huber")
}

print("\nLoss function comparison (5-fold CV):")

for name, params in losses.items():
    scores = []
    for tr_idx, va_idx in kf.split(X):
        mdl = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            **params
        )
        mdl.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = mdl.predict(X.iloc[va_idx])
        score = mean_squared_error(y.iloc[va_idx], pred)
        scores.append(score)

    print(f"{name:>6s} | CV MSE: {np.mean(scores):.5f} ± {np.std(scores):.5f}")

print(
    "\nInterpretation:\n"
    " - MAE/Huber better → median > mean (multimodality)\n"
    " - MSE best → averaging acceptable\n"
)


# ----------------------------
# 6️⃣ Fold-wise score variance
# ----------------------------
scores = []
for tr_idx, va_idx in kf.split(X):
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pred = model.predict(X.iloc[va_idx])
    scores.append(mean_squared_error(y.iloc[va_idx], pred))

print("\nFold-wise CV MSE:", scores)
print("Std across folds:", np.std(scores))

print(
    "\nInterpretation:\n"
    " - Large variance → regime instability\n"
    " - Small variance → safe modeling\n"
)


# ============================================================
# Phase 4 — Advanced (ONLY if needed)
# ============================================================

# ----------------------------
# 7️⃣ Quantile regression sanity check
# ----------------------------
quantiles = [0.25, 0.5, 0.75]
print("\nQuantile regression check:")

for q in quantiles:
    q_scores = []
    for tr_idx, va_idx in kf.split(X):
        mdl = LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        mdl.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = mdl.predict(X.iloc[va_idx])
        q_scores.append(mean_absolute_error(y.iloc[va_idx], pred))

    print(f"Quantile {q:.2f} | MAE: {np.mean(q_scores):.5f}")

print(
    "\nInterpretation:\n"
    " - Median (q=0.5) best → multimodality\n"
    " - Quantiles diverge strongly → mixture structure\n"
)


# ----------------------------
# 8️⃣ Mode-aware ablation (careful!)
# ----------------------------
print("\nMode-aware ablation (diagnostic only)")

kmeans = KMeans(n_clusters=2, random_state=42)
modes = kmeans.fit_predict(y.values.reshape(-1, 1))

mode_scores = []
for m in [0, 1]:
    idx = modes == m
    mdl = LGBMRegressor(n_estimators=300, random_state=42)
    mdl.fit(X[idx], y[idx])
    pred = mdl.predict(X[idx])
    mode_scores.append(mean_squared_error(y[idx], pred))

print("Per-mode training MSE:", mode_scores)
print(
    "⚠️ Use only if CV improves consistently. "
    "This is diagnostic, not a submission model."
)

print("\n========================")
print("ANALYSIS COMPLETE")
print("========================")
