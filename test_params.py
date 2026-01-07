import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor

# Configuration
PROBLEM_NUM = 36
SELECTED_FEATURES = [
    'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253', 
    'feat_143', 'feat_221', 'feat_220', 'feat_160', 'feat_266', 
    'feat_138', 'feat_47', 'feat_203'
]
# SELECTED_FEATURES = [
#     'feat_143', 'feat_221', 'feat_220', 'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253', 'feat_160', 'feat_266', 'feat_209', 'feat_267', 'feat_56'
# ]
# Load data
X_path = f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv"
y_path = f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv"

X = pd.read_csv(X_path)
y_df = pd.read_csv(y_path)
y = y_df["target01"]

# Filter to selected features
X_selected = X[SELECTED_FEATURES]

print(f"Problem {PROBLEM_NUM}")
print(f"Using {len(SELECTED_FEATURES)} features")
print(f"Data shape: {X_selected.shape}")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")

# Use the specified parameters
params = {
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
    'verbose': False
}

print("\n=== Training with specified parameters ===")
for key, value in params.items():
    if key not in ['loss_function', 'random_seed', 'verbose']:
        print(f"  {key}: {value}")

# Train model
model = CatBoostRegressor(**params)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
train_smape = np.mean(2 * np.abs(y_train - y_train_pred) / (np.abs(y_train) + np.abs(y_train_pred))) * 100

val_r2 = r2_score(y_val, y_val_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)
val_mape = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100
val_smape = np.mean(2 * np.abs(y_val - y_val_pred) / (np.abs(y_val) + np.abs(y_val_pred))) * 100

print("\n" + "="*50)
print("PERFORMANCE METRICS")
print("="*50)

print(f"\nTrain Performance:")
print(f"  R²:    {train_r2:.6f}")
print(f"  MSE:   {train_mse:.6f}")
print(f"  RMSE:  {train_rmse:.6f}")
print(f"  MAPE:  {train_mape:.2f}%")
print(f"  SMAPE: {train_smape:.2f}%")

print(f"\nValidation Performance:")
print(f"  R²:    {val_r2:.6f}")
print(f"  MSE:   {val_mse:.6f}")
print(f"  RMSE:  {val_rmse:.6f}")
print(f"  MAPE:  {val_mape:.2f}%")
print(f"  SMAPE: {val_smape:.2f}%")

print(f"\n" + "-"*50)
print("Overfitting Analysis:")
print("-"*50)
print(f"  R² difference (train - val):      {train_r2 - val_r2:+.6f}")
print(f"  RMSE difference (val - train):    {val_rmse - train_rmse:+.6f}")
print(f"  MAPE difference (val - train):    {val_mape - train_mape:+.2f}%")
print(f"  SMAPE difference (val - train):   {val_smape - train_smape:+.2f}%")

if train_r2 - val_r2 < 0.05:
    print("  ✓ Good generalization (low overfitting)")
elif train_r2 - val_r2 < 0.10:
    print("  ⚠ Moderate overfitting")
else:
    print("  ✗ High overfitting detected")

print("\n" + "="*50)

# Feature importance
print("\nFeature Importance:")
print("-"*50)
feature_importance = model.get_feature_importance()
importance_df = pd.DataFrame({
    'Feature': SELECTED_FEATURES,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(importance_df.to_string(index=False))
