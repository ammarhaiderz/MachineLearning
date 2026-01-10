import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# -----------------------------
# Load data
# -----------------------------
PROBLEM_NUM = 36

X = pd.read_csv(f'./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv')
y_df = pd.read_csv(f'./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv')
y = y_df.iloc[:, 0].values

print(f"Loaded data: X shape = {X.shape}, y shape = {y.shape}")

# -----------------------------
# Train / validation split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# -----------------------------
# Base Random Forest (all features)
# -----------------------------
rf_base = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

rf_base.fit(X_train, y_train)

# -----------------------------
# Feature importance ranking
# -----------------------------
importances = rf_base.feature_importances_

feat_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\nTop 20 important features:")
print(feat_importance_df.head(20))

# -----------------------------
# Incremental feature evaluation
# -----------------------------
max_k = min(50, X.shape[1])  # limit to first 50 for stability

train_r2 = []
val_r2 = []
k_values = range(1, max_k + 1)

for k in k_values:
    top_features = feat_importance_df["feature"].iloc[:k].tolist()

    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train[top_features], y_train)

    y_train_pred = rf.predict(X_train[top_features])
    y_val_pred = rf.predict(X_val[top_features])

    train_r2.append(r2_score(y_train, y_train_pred))
    val_r2.append(r2_score(y_val, y_val_pred))

# -----------------------------
# Print R² scores
# -----------------------------
print("\n" + "="*80)
print("R² SCORES BY NUMBER OF FEATURES")
print("="*80)
print(f"{'Features':<12s} {'Train R²':>12s} {'Val R²':>12s} {'Overfit':>12s}")
print("-" * 80)

for i, k in enumerate(k_values):
    overfit = train_r2[i] - val_r2[i]
    print(f"{k:<12d} {train_r2[i]:>12.4f} {val_r2[i]:>12.4f} {overfit:>+12.4f}")

# Find best validation R²
best_idx = np.argmax(val_r2)
best_k = k_values[best_idx]

print("\n" + "="*80)
print("BEST PERFORMANCE")
print("="*80)
print(f"Best number of features: {best_k}")
print(f"Train R² at best: {train_r2[best_idx]:.4f}")
print(f"Val R² at best:   {val_r2[best_idx]:.4f}")
print(f"Overfitting:      {train_r2[best_idx] - val_r2[best_idx]:+.4f}")

print("\nTop features used:")
top_features_best = feat_importance_df["feature"].iloc[:best_k].tolist()
for i, feat in enumerate(top_features_best[:20], 1):  # Show first 20
    importance = feat_importance_df[feat_importance_df["feature"] == feat]["importance"].values[0]
    print(f"  {i:2d}. {feat:20s} (importance: {importance:.4f})")
if best_k > 20:
    print(f"  ... and {best_k - 20} more features")

# -----------------------------
# Plot results
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_r2, label="Train R²", marker="o")
plt.plot(k_values, val_r2, label="Validation R²", marker="o")
plt.xlabel("Number of Top Features Used")
plt.ylabel("R² Score")
plt.title("Random Forest: Train vs Validation R² by Feature Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
