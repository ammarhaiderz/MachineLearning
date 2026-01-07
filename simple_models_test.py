import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# ===============================
# Configuration
# ===============================
PROBLEM_NUM = 36
SELECTED_FEATURES = [
    'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253', 
    'feat_143', 'feat_221', 'feat_220', 'feat_160', 'feat_266', 
    'feat_138', 'feat_47', 'feat_203'
]

print(f"Problem {PROBLEM_NUM}")
print(f"Using {len(SELECTED_FEATURES)} selected features\n")

# ===============================
# Load data
# ===============================
X_path = f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv"
y_path = f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv"

X = pd.read_csv(X_path)
y_df = pd.read_csv(y_path)
y = y_df["target01"]

# Filter to selected features
X_selected = X[SELECTED_FEATURES]

print(f"Data shape: {X_selected.shape}")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}\n")

# Scale data (important for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ===============================
# Define models to test
# ===============================
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01, max_iter=10000),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0),
    'KNN': KNeighborsRegressor(n_neighbors=5)
}

# ===============================
# Helper function to calculate metrics
# ===============================
def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    return r2, rmse, smape

# ===============================
# Test each model
# ===============================
results = []

print("="*90)
print(f"{'Model':<20} {'Train R²':<12} {'Train RMSE':<12} {'Train SMAPE':<12} {'Val R²':<12} {'Val RMSE':<12} {'Val SMAPE':<12}")
print("="*90)

for name, model in models.items():
    # Determine if model needs scaled data
    needs_scaling = name in ['Ridge', 'Lasso', 'ElasticNet', 'SVR', 'KNN']
    
    if needs_scaling:
        X_train_use = X_train_scaled
        X_val_use = X_val_scaled
    else:
        X_train_use = X_train
        X_val_use = X_val
    
    # Train model
    try:
        model.fit(X_train_use, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_use)
        y_val_pred = model.predict(X_val_use)
        
        # Calculate metrics
        train_r2, train_rmse, train_smape = calculate_metrics(y_train, y_train_pred)
        val_r2, val_rmse, val_smape = calculate_metrics(y_val, y_val_pred)
        
        # Store results
        results.append({
            'Model': name,
            'Train_R2': train_r2,
            'Train_RMSE': train_rmse,
            'Train_SMAPE': train_smape,
            'Val_R2': val_r2,
            'Val_RMSE': val_rmse,
            'Val_SMAPE': val_smape,
            'Overfit_R2': train_r2 - val_r2
        })
        
        # Print results
        print(f"{name:<20} {train_r2:>11.6f} {train_rmse:>11.6f} {train_smape:>11.2f}% {val_r2:>11.6f} {val_rmse:>11.6f} {val_smape:>11.2f}%")
        
    except Exception as e:
        print(f"{name:<20} FAILED: {str(e)}")

print("="*90)

# ===============================
# Summary
# ===============================
results_df = pd.DataFrame(results).sort_values('Val_R2', ascending=False)

print("\n" + "="*90)
print("BEST MODELS BY VALIDATION R²")
print("="*90)
print(results_df[['Model', 'Val_R2', 'Val_RMSE', 'Val_SMAPE', 'Overfit_R2']].to_string(index=False))

print("\n" + "="*90)
print("BEST MODEL:")
best_model = results_df.iloc[0]
print(f"  {best_model['Model']}")
print(f"  Validation R²: {best_model['Val_R2']:.6f}")
print(f"  Validation RMSE: {best_model['Val_RMSE']:.6f}")
print(f"  Validation SMAPE: {best_model['Val_SMAPE']:.2f}%")
print(f"  Overfitting (R² diff): {best_model['Overfit_R2']:.6f}")
print("="*90)

# Save results
output_file = f"simple_models_comparison_problem_{PROBLEM_NUM}.csv"
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")
