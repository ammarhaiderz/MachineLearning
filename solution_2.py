import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold


# Load
X = pd.read_csv("./data_31_40/problem_36/dataset_36.csv")
y = pd.read_csv("./data_31_40/problem_36/target_36.csv")

# Choose target for Part 2 feature selection
y2 = y["target02"]

# Sanity checks
print("X shape:", X.shape)          # expect (10000, 273)
print("y shape:", y.shape)          # expect (10000, 2)
print("y2 shape:", y2.shape)        # expect (10000,)

# Check column naming / order
assert list(X.columns) == [f"feat_{i}" for i in range(X.shape[1])], "Unexpected feature columns/order!"

# Missing values check
na_counts = X.isna().sum().sum()
print("Total missing values in X:", na_counts)


# decision tree feature importance with cross-validation ---------


X_np = X.values
y_np = y2.values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

importances = []

for train_idx, val_idx in kf.split(X_np):
    tree = DecisionTreeRegressor(
        max_depth=4,
        min_samples_leaf=50,
        random_state=42
    )
    tree.fit(X_np[train_idx], y_np[train_idx])
    importances.append(tree.feature_importances_)

importances = np.array(importances)
mean_importance = importances.mean(axis=0)
std_importance = importances.std(axis=0)

# Rank features
ranked = np.argsort(mean_importance)[::-1]

# Show top 10
top_k = 10
for i in range(top_k):
    idx = ranked[i]
    print(f"Rank {i+1}: feat_{idx}, mean_imp={mean_importance[idx]:.4f}, std={std_importance[idx]:.4f}")


import matplotlib.pyplot as plt

# ---- Step 3: Functional relationship analysis for selected features ----

selected_features = [49, 169, 55]

for feat_idx in selected_features:
    plt.figure()
    plt.scatter(X_np[:, feat_idx], y_np, s=5)
    plt.xlabel(f"feat_{feat_idx}")
    plt.ylabel("target02")
    plt.title(f"target02 vs feat_{feat_idx}")
    plt.show()

# Optional: simple linear fit check (numerical, no plotting assumptions)
from sklearn.linear_model import LinearRegression

for feat_idx in selected_features:
    lr = LinearRegression()
    lr.fit(X_np[:, feat_idx].reshape(-1, 1), y_np)
    r2 = lr.score(X_np[:, feat_idx].reshape(-1, 1), y_np)
    print(f"Linear fit using feat_{feat_idx}: R^2 = {r2:.4f}")



from sklearn.tree import DecisionTreeRegressor

# Tree using ONLY feat_49 to find thresholds
X_feat49 = X_np[:, 49].reshape(-1, 1)

tree_49 = DecisionTreeRegressor(
    max_depth=2,
    min_samples_leaf=100,
    random_state=42
)
tree_49.fit(X_feat49, y_np)

# Extract thresholds
thresholds = tree_49.tree_.threshold
thresholds = thresholds[thresholds != -2]

print("Detected thresholds for feat_49:", thresholds)


# ---- Step 5: per-regime linear analysis ----

regimes = [
    X_np[:, 49] <= 0.20,
    (X_np[:, 49] > 0.20) & (X_np[:, 49] <= 0.50),
    (X_np[:, 49] > 0.50) & (X_np[:, 49] <= 0.70),
    X_np[:, 49] > 0.70
]

for i, mask in enumerate(regimes):
    X_reg = X_np[mask][:, 169].reshape(-1, 1)
    y_reg = y_np[mask]

    lr = LinearRegression()
    lr.fit(X_reg, y_reg)

    r2 = lr.score(X_reg, y_reg)
    a = lr.coef_[0]
    b = lr.intercept_

    print(f"Regime {i+1}: samples={len(y_reg)}, "
          f"target02 ≈ {a:.3f} * feat_169 + {b:.3f}, R^2={r2:.3f}")
    


# ---- Extended evaluation: per-regime error analysis ----
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_true = y_np

regimes = {
    "Regime 1 (feat_49 <= 0.20)": (X[:, 49] <= 0.20),
    "Regime 2 (0.20 < feat_49 <= 0.50)": ((X[:, 49] > 0.20) & (X[:, 49] <= 0.50)),
    "Regime 3 (0.50 < feat_49 <= 0.70)": ((X[:, 49] > 0.50) & (X[:, 49] <= 0.70)),
    "Regime 4 (feat_49 > 0.70)": (X[:, 49] > 0.70),
}

print("\nPer-regime error metrics:")
print("-" * 50)

for name, mask in regimes.items():
    y_t = y_true[mask]
    y_p = y_pred[mask]

    mae = mean_absolute_error(y_t, y_p)
    rmse = mean_squared_error(y_t, y_p, squared=False)
    r2 = r2_score(y_t, y_p)

    print(f"{name}:")
    print(f"  Samples = {len(y_t)}")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R^2  = {r2:.4f}")
    print()

def predict_rule_system(X):
    preds = np.zeros(len(X))

    f49 = X[:, 49]
    f169 = X[:, 169]

    preds[f49 <= 0.20] = -0.638 * f169[f49 <= 0.20] - 0.407

    mask = (f49 > 0.20) & (f49 <= 0.50)
    preds[mask] = 0.760 * f169[mask] + 0.698

    mask = (f49 > 0.50) & (f49 <= 0.70)
    preds[mask] = -1.651 * f169[mask] - 0.100

    preds[f49 > 0.70] = -1.858 * f169[f49 > 0.70] - 0.147

    return preds


y_pred = predict_rule_system(X_np)
