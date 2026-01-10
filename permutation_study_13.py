# ===============================
# 0. Imports
# ===============================
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

from catboost_model import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt


# ===============================
# 1. Load data
# ===============================
PROBLEM_NUM = 36

# Define the 13 selected features
SELECTED_FEATURES = [
    'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253', 
    'feat_143', 'feat_221', 'feat_220', 'feat_160', 'feat_266', 
    'feat_138', 'feat_47', 'feat_203'
]

X_path = f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv"
y_path = f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv"

X = pd.read_csv(X_path)
y_df = pd.read_csv(y_path)
y = y_df["target01"]

# Filter to selected features only
X = X[SELECTED_FEATURES]

print(f"Problem {PROBLEM_NUM}")
print(f"Using {len(SELECTED_FEATURES)} selected features")
print(f"X shape: {X.shape}, y shape: {y.shape}")

feature_names = X.columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)


# ===============================
# 2. Train baseline CatBoost
# ===============================
base_model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)

base_model.fit(X_train, y_train)

baseline_pred = base_model.predict(X_val)
baseline_r2 = r2_score(y_val, baseline_pred)

print(f"\nBaseline Validation R² (all features): {baseline_r2:.4f}")


# ===============================
# 3. MANUAL permutation importance
# ===============================
def permutation_importance_manual(
    model, X_val, y_val, metric, n_repeats=5, random_state=42
):
    rng = np.random.RandomState(random_state)
    baseline_score = metric(y_val, model.predict(X_val))

    importances = []

    for col in tqdm(X_val.columns, desc="Permuting features"):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            perm_score = metric(y_val, model.predict(X_perm))
            scores.append(baseline_score - perm_score)

        importances.append(np.mean(scores))

    return np.array(importances)


perm_importance = permutation_importance_manual(
    base_model,
    X_val,
    y_val,
    metric=r2_score,
    n_repeats=5
)

perm_df = pd.DataFrame({
    "feature": feature_names,
    "importance": perm_importance
}).sort_values("importance", ascending=False)

print("\nTop 15 permutation-important features:")
print(perm_df.head(15))


# ===============================
# 4. Keep positive-importance features
# ===============================
perm_selected = perm_df[perm_df["importance"] > 0]["feature"].tolist()
print(f"\nFeatures with positive permutation importance: {len(perm_selected)}")


# ===============================
# 5. Stability selection
# ===============================
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
        model,
        X_val,
        y_val,
        metric=r2_score,
        n_repeats=3,
        random_state=seed
    )

    return [
        f for f, v in zip(feature_names, imp) if v > 0
    ]


all_selected = []
for seed in range(5):
    all_selected.extend(permutation_run(seed))

counts = Counter(all_selected)

stable_features = [
    f for f, c in counts.items() if c >= 3
]

print(f"\nStable features after stability selection: {len(stable_features)}")
print(stable_features)


# ===============================
# 6. Rank stable features
# ===============================
ranked_features = (
    perm_df[perm_df["feature"].isin(stable_features)]
    .sort_values("importance", ascending=False)["feature"]
    .tolist()
)


# ===============================
# 7. Ablation study
# ===============================
scores = []

for k in tqdm(range(1, len(ranked_features) + 1)):
    feats = ranked_features[:k]

    model = CatBoostRegressor(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )

    model.fit(X_train[feats], y_train)
    preds = model.predict(X_val[feats])
    scores.append(r2_score(y_val, preds))


# ===============================
# 8. Plot ablation curve
# ===============================
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(scores) + 1), scores, marker="o")
plt.axhline(baseline_r2, linestyle="--", label="All features baseline")
plt.xlabel("Number of Features")
plt.ylabel("Validation R²")
plt.title("Permutation-Based Feature Ablation")
plt.legend()
plt.tight_layout()
plt.show()


# ===============================
# 9. Final feature set
# ===============================
best_k = int(np.argmax(scores)) + 1
best_features = ranked_features[:best_k]

print("\n===============================")
print(f"Best number of features: {best_k}")
print(f"Best Validation R²: {scores[best_k-1]:.4f}")
print(f"Baseline R²: {baseline_r2:.4f}")
print(f"Improvement: {scores[best_k-1] - baseline_r2:+.4f}")
print("Final selected features:")
print(best_features)
print("===============================")


# ===============================
# 10. Train final model & predict
# ===============================
X_eval_path = f"./data_31_40/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv"
X_eval = pd.read_csv(X_eval_path)

final_model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)

X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

final_model.fit(X_full[best_features], y_full)

y_eval_pred = final_model.predict(X_eval[best_features])

output_file = f"EVAL_target01_{PROBLEM_NUM}_permutation_{best_k}feat.csv"
pd.DataFrame({"target01": y_eval_pred}).to_csv(output_file, index=False)

print(f"\nPredictions saved to: {output_file}")
print("Sample predictions:")
print(y_eval_pred[:10])
