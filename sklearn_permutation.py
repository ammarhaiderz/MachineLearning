# ==================================================
# 0. Imports
# ==================================================
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor

import matplotlib.pyplot as plt


# ==================================================
# 1. Configuration
# ==================================================
PROBLEM_NUM = 36
RANDOM_SEED = 42
N_REPEATS = 20        # permutation repeats
STABILITY_RUNS = 5    # number of RF restarts
STABILITY_RATIO = 0.8  # 80% stability

print(f"Problem {PROBLEM_NUM}")
print(f"Permutation repeats: {N_REPEATS}")
print(f"Stability runs: {STABILITY_RUNS}\n")


# ==================================================
# 2. Load data
# ==================================================
X_path = f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv"
y_path = f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv"

X = pd.read_csv(X_path)
y_df = pd.read_csv(y_path)
y = y_df["target01"]

print(f"X shape: {X.shape}, y shape: {y.shape}")

feature_names = X.columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}\n")


# ==================================================
# 3. RandomForest permutation importance (core)
# ==================================================
def rf_permutation_run(seed):
    rf = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    perm = permutation_importance(
        rf,
        X_val,
        y_val,
        n_repeats=N_REPEATS,
        random_state=seed,
        scoring="r2",
        n_jobs=-1
    )

    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    })

    df["lower_bound"] = (
        df["importance_mean"] - df["importance_std"]
    )

    return df


# ==================================================
# 4. Stability selection
# ==================================================
all_selected = []
all_results = []

print("=== Runnin   RandomForest stability selection ===\n")

for seed in range(STABILITY_RUNS):
    print(f"RF run {seed+1}/{STABILITY_RUNS}")
    df_imp = rf_permutation_run(RANDOM_SEED + seed)
    all_results.append(df_imp)

    selected = df_imp[df_imp["lower_bound"] > 0]["feature"].tolist()
    all_selected.extend(selected)

from collections import Counter
counts = Counter(all_selected)

stable_features = [
    f for f, c in counts.items()
    if c >= int(STABILITY_RATIO * STABILITY_RUNS)
]

print(f"\nStable features (≥{int(STABILITY_RATIO*100)}% runs): {len(stable_features)}")
print(stable_features)


# ==================================================
# 5. Rank stable features by mean importance
# ==================================================
mean_importance = (
    pd.concat(all_results)
    .groupby("feature")["importance_mean"]
    .mean()
    .sort_values(ascending=False)
)

ranked_features = [
    f for f in mean_importance.index
    if f in stable_features
]

print("\nRanked stable features:")
print(ranked_features)


# ==================================================
# 6. Ablation study with CatBoost
# ==================================================
scores = []

print("\n=== CatBoost Ablation Study ===\n")

for k in tqdm(range(1, len(ranked_features) + 1)):
    feats = ranked_features[:k]

    model = CatBoostRegressor(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=RANDOM_SEED,
        verbose=False
    )

    model.fit(X_train[feats], y_train)
    preds = model.predict(X_val[feats])
    scores.append(r2_score(y_val, preds))


# ==================================================
# 7. Plot ablation curve
# ==================================================
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(scores) + 1), scores, marker="o")
plt.xlabel("Number of Features")
plt.ylabel("Validation R²")
plt.title("CatBoost Performance vs Number of RF-Selected Features")
plt.tight_layout()
plt.show()


# ==================================================
# 8. Final feature set
# ==================================================
best_k = int(np.argmax(scores)) + 1
best_features = ranked_features[:best_k]

print("\n===============================")
print(f"Best number of features: {best_k}")
print(f"Best Validation R²: {scores[best_k-1]:.4f}")
print("Final selected features:")
print(best_features)
print("===============================")


# ==================================================
# 9. Train final CatBoost & predict
# ==================================================
X_eval_path = f"./data_31_40/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv"
X_eval = pd.read_csv(X_eval_path)

final_model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    random_seed=RANDOM_SEED,
    verbose=False
)

X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

final_model.fit(X_full[best_features], y_full)

y_eval_pred = final_model.predict(X_eval[best_features])

output_file = f"EVAL_target01_{PROBLEM_NUM}_RFperm_{best_k}feat.csv"
pd.DataFrame({"target01": y_eval_pred}).to_csv(output_file, index=False)

print(f"\nPredictions saved to: {output_file}")
print("Sample predictions:")
print(y_eval_pred[:10])


