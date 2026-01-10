# ===============================
# 0. Imports
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance


# ===============================
# 1. Load data
# ===============================
PROBLEM_NUM = 36

X_path = f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv"
y_path = f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv"

X = pd.read_csv(X_path)
y = pd.read_csv(y_path)["target01"]

print(f"Problem {PROBLEM_NUM}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

feature_names = X.columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)


# ===============================
# 2. Train Random Forest
# ===============================
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

val_preds = rf.predict(X_val)
rf_r2 = r2_score(y_val, val_preds)

print(f"\nRandom Forest Validation R²: {rf_r2:.4f}")


# ===============================
# 3. Built-in feature importance
# ===============================
rf_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 15 RF built-in (impurity) important features:")
print(rf_importance.head(15))


# ===============================
# 4. Permutation importance
# ===============================
perm = permutation_importance(
    rf,
    X_val,
    y_val,
    scoring="r2",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

perm_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": perm.importances_mean,
    "std": perm.importances_std
}).sort_values("importance", ascending=False)

print("\nTop 15 RF permutation-important features:")
print(perm_importance.head(15))


# ===============================
# 5. Compare overlap
# ===============================
top_rf = set(rf_importance.head(15)["feature"])
top_perm = set(perm_importance.head(15)["feature"])

print("\nOverlap between top-15 lists:")
print(f"Count: {len(top_rf & top_perm)}")
print(f"Common features: {sorted(top_rf & top_perm)}")


# ===============================
# 6. Plot importance comparison
# ===============================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

rf_importance.head(10).plot.barh(
    x="feature", y="importance", ax=axes[0], legend=False
)
axes[0].set_title("RF Built-in Importance")
axes[0].invert_yaxis()

perm_importance.head(10).plot.barh(
    x="feature", y="importance", ax=axes[1], legend=False
)
axes[1].set_title("RF Permutation Importance")
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()
