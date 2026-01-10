import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler

# ===============================
# 1. Configuration
# ===============================
PROBLEM_NUM = 36
SELECTED_FEATURES = [
    'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253', 
    'feat_143', 'feat_221', 'feat_220', 'feat_160', 'feat_266', 
    'feat_138', 'feat_47', 'feat_203', "feat_131", "feat_24", "feat_104",
]

# SELECTED_FEATURES = [
#     'feat_143', 'feat_221', 'feat_220', 'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253', 'feat_160', 'feat_266', 'feat_209', 'feat_267', 'feat_56'
# ]

print(f"Problem {PROBLEM_NUM}")
print(f"Using {len(SELECTED_FEATURES)} selected features")

# ===============================
# 2. Load data
# ===============================
X_path = f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv"
y_path = f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv"
X_eval_path = f"./data_31_40/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv"

X = pd.read_csv(X_path)
y_df = pd.read_csv(y_path)
y = y_df["target01"]
X_eval = pd.read_csv(X_eval_path)

# Filter to selected features only
X_selected = X[SELECTED_FEATURES]
X_eval_selected = X_eval[SELECTED_FEATURES]

print(f"\nData shapes:")
print(f"X: {X_selected.shape}, y: {y.shape}")
print(f"X_eval: {X_eval_selected.shape}")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nTrain/Val split:")
print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")

# ===============================
# 3. Optuna objective function
# ===============================
def objective(trial):
    """Optuna objective to maximize validation R2"""
    
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 30, 100),
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': False
    }
    
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
    
    y_val_pred = model.predict(X_val)
    r2_val = r2_score(y_val, y_val_pred)
    
    return r2_val


# ===============================
# 4. Run Optuna optimization
# ===============================
print("\n=== Starting Optuna Optimization ===")
print("Objective: Maximize Validation R²")
print(f"Number of trials: 100\n")

study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='catboost_r2_optimization'
)

study.optimize(objective, n_trials=100, show_progress_bar=True)

# ===============================
# 5. Results
# ===============================
print("\n=== Optimization Results ===")
print(f"Best R² score: {study.best_value:.6f}")
print(f"Best trial: {study.best_trial.number}")
print("\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# ===============================
# 6. Train final model with best params
# ===============================
print("\n=== Training Final Model ===")

best_params = study.best_params.copy()
best_params['loss_function'] = 'RMSE'
best_params['random_seed'] = 42
best_params['verbose'] = False

# First, train on train set only to evaluate train/val performance
eval_model = CatBoostRegressor(**best_params)
eval_model.fit(X_train, y_train)

# Evaluate on both train and validation
y_train_pred = eval_model.predict(X_train)
y_val_pred = eval_model.predict(X_val)

train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)

val_r2 = r2_score(y_val, y_val_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)

print(f"\nTrain Performance:")
print(f"  R²: {train_r2:.6f}")
print(f"  MSE: {train_mse:.6f}")
print(f"  RMSE: {train_rmse:.6f}")

print(f"\nValidation Performance:")
print(f"  R²: {val_r2:.6f}")
print(f"  MSE: {val_mse:.6f}")
print(f"  RMSE: {val_rmse:.6f}")

print(f"\nOverfitting check:")
print(f"  R² difference (train - val): {train_r2 - val_r2:+.6f}")
print(f"  RMSE difference (val - train): {val_rmse - train_rmse:+.6f}")

# Now train on full data for final predictions
final_model = CatBoostRegressor(**best_params)
X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])
final_model.fit(X_full, y_full)

# ===============================
# 7. Make predictions on evaluation set
# ===============================
print("\n=== Making Predictions ===")
y_eval_pred = final_model.predict(X_eval_selected)

# Save predictions
output_file = f"EVAL_target01_{PROBLEM_NUM}_optuna_13feat.csv"
predictions_df = pd.DataFrame({'target01': y_eval_pred})
predictions_df.to_csv(output_file, index=False)

print(f"\nPredictions saved to: {output_file}")
print(f"Shape: {predictions_df.shape}")
print(f"\nSample predictions:")
print(predictions_df.head(10))

# ===============================
# 8. Feature importance
# ===============================
print("\n=== Feature Importance ===")
feature_importance = final_model.get_feature_importance()
importance_df = pd.DataFrame({
    'Feature': SELECTED_FEATURES,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(importance_df.to_string(index=False))
