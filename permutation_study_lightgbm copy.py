# ===============================
# 0. Imports
# ===============================
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib.pyplot as plt


# ===============================
# 1. Load data
# ===============================
PROBLEM_NUM = 36
RANDOM_SEED = 42

X_path = f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv"
y_path = f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv"

X = pd.read_csv(X_path)
y = pd.read_csv(y_path)["target01"]

print(f"Problem {PROBLEM_NUM}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

feature_names = X.columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    shuffle=True
)


# ===============================
# 2. Train baseline LightGBM (ROBUST)
# ===============================
base_model = lgb.LGBMRegressor(
    objective="regression_l1",   # MAE loss (IMPORTANT)
    n_estimators=600,
    num_leaves=31,
    learning_rate=0.07,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=-1
)

base_model.fit(X_train, y_train)

baseline_pred = base_model.predict(X_val)
baseline_r2 = r2_score(y_val, baseline_pred)

print(f"\nBaseline Validation R² (all features): {baseline_r2:.6f}")


# ===============================
# 3. MANUAL permutation importance (MAE-based)
# ===============================
def permutation_importance_manual(
    model,
    X_val,
    y_val,
    n_repeats=5,
    random_state=42
):
    rng = np.random.RandomState(random_state)

    baseline_mae = mean_absolute_error(
        y_val, model.predict(X_val)
    )

    importances = []

    for col in tqdm(X_val.columns, desc="Permuting features"):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)

            perm_mae = mean_absolute_error(
                y_val, model.predict(X_perm)
            )
            scores.append(perm_mae - baseline_mae)

        importances.append(np.mean(scores))

    return np.array(importances)


perm_importance = permutation_importance_manual(
    base_model,
    X_val,
    y_val,
    n_repeats=7
)

perm_df = pd.DataFrame({
    "feature": feature_names,
    "importance": perm_importance
}).sort_values("importance", ascending=False)

print("\nTop 15 permutation-important features:")
print(perm_df.head(15))


# ===============================
# 4. Stability selection (median > 0)
# ===============================
def permutation_run(seed):
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=400,
        num_leaves=31,
        learning_rate=0.1,
        random_state=seed,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)

    imp = permutation_importance_manual(
        model,
        X_val,
        y_val,
        n_repeats=3,
        random_state=seed
    )

    return imp


all_importances = []

for seed in range(5):
    print(f"\nStability run {seed+1}/5...")
    all_importances.append(permutation_run(RANDOM_SEED + seed))

all_importances = np.vstack(all_importances)

median_importance = np.median(all_importances, axis=0)

stable_features = [
    f for f, v in zip(feature_names, median_importance) if v > 0
]

print(f"\nStable features after stability selection: {len(stable_features)}")
print(stable_features)


# ===============================
# 5. Rank stable features
# ===============================
ranked_features = (
    perm_df[perm_df["feature"].isin(stable_features)]
    .sort_values("importance", ascending=False)["feature"]
    .tolist()
)


# ===============================
# 6. Ablation study
# ===============================
scores = []

print("\nPerforming ablation study...")
for k in tqdm(range(1, len(ranked_features) + 1)):
    feats = ranked_features[:k]

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=400,
        num_leaves=31,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(X_train[feats], y_train)
    preds = model.predict(X_val[feats])

    scores.append(r2_score(y_val, preds))


# ===============================
# 7. Plot ablation curve
# ===============================
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(scores) + 1), scores, marker="o")
plt.axhline(
    baseline_r2,
    linestyle="--",
    color="red",
    label=f"Baseline R² = {baseline_r2:.4f}"
)
plt.xlabel("Number of Features")
plt.ylabel("Validation R²")
plt.title("LightGBM Stable-Permutation Feature Ablation")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ===============================
# 8. Final feature set
# ===============================
best_k = int(np.argmax(scores)) + 1
best_features = ranked_features[:best_k]

print("\n" + "=" * 80)
print(f"Best number of features: {best_k}")
print(f"Best Validation R²: {scores[best_k-1]:.6f}")
print(f"Baseline R²: {baseline_r2:.6f}")
print(f"Improvement: {scores[best_k-1] - baseline_r2:+.6f}")
print("=" * 80)


# ===============================
# 9. Train final model & predict
# ===============================
X_eval = pd.read_csv(
    f"./data_31_40/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv"
)

final_model = lgb.LGBMRegressor(
    objective="regression_l1",
    n_estimators=1200,
    num_leaves=31,
    learning_rate=0.05,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=-1
)

X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

final_model.fit(X_full[best_features], y_full)

y_eval_pred = final_model.predict(X_eval[best_features])

output_file = f"EVAL_target01_{PROBLEM_NUM}_LightGBM_STABLE_{best_k}feat.csv"
pd.DataFrame({"target01": y_eval_pred}).to_csv(output_file, index=False)

print(f"\nPredictions saved to: {output_file}")
print("Done.")
