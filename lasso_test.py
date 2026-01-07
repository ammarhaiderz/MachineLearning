import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# Configure problem number
PROBLEM_NUM = 36

# Load data
X_path = f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv"
y_path = f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv"

X = pd.read_csv(X_path)
y = pd.read_csv(y_path)
y1 = y["target01"]

print(f"Problem {PROBLEM_NUM}")
print(f"X: {X.shape}, y1: {y1.shape}")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y1, test_size=0.2, random_state=42
)

# Scale features (important for Lasso)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("\n=== Training LassoCV ===")
# LassoCV automatically finds best alpha via cross-validation
lasso = LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1)
lasso.fit(X_train_scaled, y_train)

# Evaluate
y_train_pred = lasso.predict(X_train_scaled)
y_val_pred = lasso.predict(X_val_scaled)

train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"\nBest alpha: {lasso.alpha_:.6f}")
print(f"Train MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
print(f"Val MSE: {val_mse:.6f}, R²: {val_r2:.4f}")

# Feature importance analysis
coefficients = lasso.coef_
feature_names = X.columns

# Create dataframe of features and weights
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Weight': coefficients,
    'Abs_Weight': np.abs(coefficients)
}).sort_values('Abs_Weight', ascending=False)

# Count non-zero features (Lasso does feature selection)
non_zero = (coefficients != 0).sum()
zero_features = (coefficients == 0).sum()

print(f"\n=== Feature Selection by Lasso ===")
print(f"Total features: {len(coefficients)}")
print(f"Non-zero weights: {non_zero}")
print(f"Zero weights (removed): {zero_features}")

print(f"\n=== Top 20 Features by Absolute Weight ===")
print(feature_importance.head(20).to_string(index=False))

print(f"\n=== Features with Zero Weight (First 20) ===")
zero_features_df = feature_importance[feature_importance['Weight'] == 0].head(20)
if len(zero_features_df) > 0:
    print(zero_features_df[['Feature']].to_string(index=False))
else:
    print("No features were zeroed out by Lasso")

# Extract selected features (non-zero weights)
selected_features = feature_importance[feature_importance['Weight'] != 0]['Feature'].tolist()
print(f"\n=== Selected {len(selected_features)} Features ===")
print(selected_features)

# Load evaluation data
X_eval_path = f"./data_31_40/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv"
X_eval = pd.read_csv(X_eval_path)

# Train final model on selected features only
print(f"\n=== Training Final Model on {len(selected_features)} Selected Features ===")
X_train_selected = X_train[selected_features]
X_val_selected = X_val[selected_features]
X_eval_selected = X_eval[selected_features]

# Scale
scaler_final = StandardScaler()
X_train_selected_scaled = scaler_final.fit_transform(X_train_selected)
X_val_selected_scaled = scaler_final.transform(X_val_selected)
X_eval_selected_scaled = scaler_final.transform(X_eval_selected)

# Train on selected features
lasso_final = LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1)
lasso_final.fit(X_train_selected_scaled, y_train)

# Evaluate
y_train_pred_final = lasso_final.predict(X_train_selected_scaled)
y_val_pred_final = lasso_final.predict(X_val_selected_scaled)

train_mse_final = mean_squared_error(y_train, y_train_pred_final)
val_mse_final = mean_squared_error(y_val, y_val_pred_final)
train_r2_final = r2_score(y_train, y_train_pred_final)
val_r2_final = r2_score(y_val, y_val_pred_final)

print(f"Best alpha: {lasso_final.alpha_:.6f}")
print(f"Train MSE: {train_mse_final:.6f}, R²: {train_r2_final:.4f}")
print(f"Val MSE: {val_mse_final:.6f}, R²: {val_r2_final:.4f}")

# Make predictions on evaluation set
print(f"\n=== Making Predictions on EVAL Set ===")
y_eval_pred = lasso_final.predict(X_eval_selected_scaled)

# Save predictions
output_file = f"EVAL_target01_{PROBLEM_NUM}_lasso_18feat.csv"
predictions_df = pd.DataFrame({'target01': y_eval_pred})
predictions_df.to_csv(output_file, index=False)
print(f"Predictions saved to: {output_file}")
print(f"Shape: {predictions_df.shape}")
print(f"Sample predictions:\n{predictions_df.head(10)}")
