"""
GLM (Generalized Linear Model) for Prediction
==============================================
Simple implementation using Tweedie GLM for regression
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Configuration
PROBLEM_NUM = 36
# SELECTED_FEATURES = [
#     'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253',
#     'feat_143', 'feat_221', 'feat_220', 'feat_160', 'feat_266',
#     'feat_138', 'feat_47', 'feat_203'
# ]

print("="*80)
print("GLM PREDICTION MODEL")
print("="*80)

# Load data
print("\nLoading data...")
X = pd.read_csv(f'./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv')
y_df = pd.read_csv(f'./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv')
X_eval = pd.read_csv(f'./data_31_40/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv')

y = y_df.iloc[:, 0].values

print(f"Training data: {X.shape}")
print(f"Evaluation data: {X_eval.shape}")
print(f"Using all {X.shape[1]} features")

# Use all features
X_selected = X
X_eval_selected = X_eval
# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

print(f"\nTrain: {X_train.shape[0]} samples")
print(f"Val:   {X_val.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_eval_scaled = scaler.transform(X_eval_selected)

# Train GLM (Tweedie regressor with power=0 is like Gaussian)
print("\n" + "="*80)
print("TRAINING GLM MODEL")
print("="*80)

glm = TweedieRegressor(
    power=0,           # 0 = Normal/Gaussian distribution
    alpha=0.5,         # Regularization strength
    max_iter=1000
)

print("\nModel parameters:")
print(f"  Distribution: Gaussian (power=0)")
print(f"  Alpha (regularization): 0.5")
print(f"  Max iterations: 1000")

glm.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = glm.predict(X_train_scaled)
y_val_pred = glm.predict(X_val_scaled)

# Metrics
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

val_r2 = r2_score(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

print("\n" + "="*80)
print("PERFORMANCE METRICS")
print("="*80)

print(f"\nTrain Performance:")
print(f"  R²:    {train_r2:.6f}")
print(f"  RMSE:  {train_rmse:.6f}")

print(f"\nValidation Performance:")
print(f"  R²:    {val_r2:.6f}")
print(f"  RMSE:  {val_rmse:.6f}")

print(f"\nOverfitting Analysis:")
print(f"  R² difference:   {train_r2 - val_r2:+.6f}")
print(f"  RMSE difference: {val_rmse - train_rmse:+.6f}")

if train_r2 - val_r2 < 0.05:
    print("  ✓ Good generalization")
elif train_r2 - val_r2 < 0.10:
    print("  ⚠ Moderate overfitting")
else:
    print("  ✗ High overfitting")

# Feature coefficients
print("\n" + "="*80)
print("FEATURE COEFFICIENTS")
print("="*80)

coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': glm.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nTop 20 features by importance (absolute coefficient):")
print(coef_df.head(20).to_string(index=False))
print(f"\n... ({len(coef_df)} total features)")

# Predict on evaluation set
print("\n" + "="*80)
print("GENERATING PREDICTIONS FOR EVALUATION SET")
print("="*80)

y_eval_pred = glm.predict(X_eval_scaled)

# Save predictions
output_file = f"EVAL_target01_{PROBLEM_NUM}_GLM_allfeat.csv"
pd.DataFrame({"target01": y_eval_pred}).to_csv(output_file, index=False)

print(f"\nPredictions saved to: {output_file}")
print(f"Number of predictions: {len(y_eval_pred)}")
print(f"\nSample predictions:")
print(f"  Min:  {y_eval_pred.min():.6f}")
print(f"  Max:  {y_eval_pred.max():.6f}")
print(f"  Mean: {y_eval_pred.mean():.6f}")
print(f"  Std:  {y_eval_pred.std():.6f}")

print("\nFirst 10 predictions:")
for i, pred in enumerate(y_eval_pred[:10], 1):
    print(f"  {i:2d}. {pred:.6f}")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
