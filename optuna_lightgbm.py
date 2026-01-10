import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from lightgbm import LGBMRegressor
import optuna
from optuna.samplers import TPESampler

# ===============================
# 1. Configuration
# ===============================
PROBLEM_NUM = 36
SELECTED_FEATURES = [
    'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253', 
    'feat_143', 'feat_221', 'feat_220', 'feat_160', 'feat_266', 
    'feat_138', 'feat_47', 'feat_203'
]

print("="*70)
print(f"LIGHTGBM OPTUNA OPTIMIZATION - Problem {PROBLEM_NUM}")
print("="*70)
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
print(f"  X: {X_selected.shape}, y: {y.shape}")
print(f"  X_eval: {X_eval_selected.shape}")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nTrain/Val split:")
print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}")

# ===============================
# 3. Optuna objective function
# ===============================
def objective(trial):
    """Optuna objective to maximize validation R2"""
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        
        # Tunable parameters
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 1e-5, 1e-2, log=True),
        'max_bin': trial.suggest_int('max_bin', 127, 511, step=32),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 5),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    # Ensure num_leaves < 2^max_depth
    if params['num_leaves'] >= 2 ** params['max_depth']:
        params['num_leaves'] = 2 ** params['max_depth'] - 1
    
    model = LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )
    
    y_val_pred = model.predict(X_val)
    r2_val = r2_score(y_val, y_val_pred)
    
    return r2_val


# ===============================
# 4. Run Optuna optimization
# ===============================
print("\n" + "="*70)
print("STARTING OPTUNA OPTIMIZATION")
print("="*70)
print("Objective: Maximize Validation R²")
print("Number of trials: 100")
print("Sampler: TPE (Tree-structured Parzen Estimator)")
print("\n")

study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='lightgbm_r2_optimization'
)

study.optimize(objective, n_trials=200, show_progress_bar=True)

# ===============================
# 5. Results
# ===============================
print("\n" + "="*70)
print("OPTIMIZATION RESULTS")
print("="*70)
print(f"Best R² score: {study.best_value:.6f}")
print(f"Best trial: {study.best_trial.number}")
print("\nBest hyperparameters:")
print("-"*70)
for key, value in study.best_params.items():
    print(f"  {key:20s}: {value}")

# ===============================
# 6. Train final model with best params
# ===============================
print("\n" + "="*70)
print("TRAINING FINAL MODEL")
print("="*70)

best_params = study.best_params.copy()
best_params['objective'] = 'regression'
best_params['metric'] = 'rmse'
best_params['verbosity'] = -1
best_params['random_state'] = 42

# Ensure num_leaves constraint
if best_params['num_leaves'] >= 2 ** best_params['max_depth']:
    best_params['num_leaves'] = 2 ** best_params['max_depth'] - 1

print("\nFinal model parameters:")
print("-"*70)
for key, value in best_params.items():
    if key not in ['objective', 'metric', 'verbosity', 'random_state']:
        print(f"  {key:20s}: {value}")

# Train on train set only to evaluate train/val performance
eval_model = LGBMRegressor(**best_params)
eval_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# Evaluate on both train and validation
y_train_pred = eval_model.predict(X_train)
y_val_pred = eval_model.predict(X_val)

train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100

val_r2 = r2_score(y_val, y_val_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)
val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100

print("\n" + "="*70)
print("PERFORMANCE METRICS")
print("="*70)

print(f"\nTrain Performance:")
print(f"  R²:    {train_r2:.6f}")
print(f"  MSE:   {train_mse:.6f}")
print(f"  RMSE:  {train_rmse:.6f}")
print(f"  MAPE:  {train_mape:.2f}%")

print(f"\nValidation Performance:")
print(f"  R²:    {val_r2:.6f}")
print(f"  MSE:   {val_mse:.6f}")
print(f"  RMSE:  {val_rmse:.6f}")
print(f"  MAPE:  {val_mape:.2f}%")

print(f"\n" + "-"*70)
print("Overfitting Analysis:")
print("-"*70)
print(f"  R² difference (train - val):      {train_r2 - val_r2:+.6f}")
print(f"  RMSE difference (val - train):    {val_rmse - train_rmse:+.6f}")
print(f"  MAPE difference (val - train):    {val_mape - train_mape:+.2f}%")

if train_r2 - val_r2 < 0.05:
    print("  ✓ Good generalization (low overfitting)")
elif train_r2 - val_r2 < 0.10:
    print("  ⚠ Moderate overfitting")
else:
    print("  ✗ High overfitting detected")

# Now train on full data for final predictions
print("\n" + "-"*70)
print("Training on full dataset for final predictions...")
print("-"*70)

final_model = LGBMRegressor(**best_params)
X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])
final_model.fit(X_full, y_full)

print(f"✓ Final model trained on {len(X_full)} samples")

# ===============================
# 7. Make predictions on evaluation set
# ===============================
print("\n" + "="*70)
print("MAKING PREDICTIONS")
print("="*70)

y_eval_pred = final_model.predict(X_eval_selected)

print(f"\nPrediction statistics:")
print(f"  Count:  {len(y_eval_pred)}")
print(f"  Mean:   {y_eval_pred.mean():.6f}")
print(f"  Std:    {y_eval_pred.std():.6f}")
print(f"  Min:    {y_eval_pred.min():.6f}")
print(f"  Max:    {y_eval_pred.max():.6f}")

# Compare with training target distribution
print(f"\nTraining target statistics:")
print(f"  Mean:   {y.mean():.6f}")
print(f"  Std:    {y.std():.6f}")
print(f"  Min:    {y.min():.6f}")
print(f"  Max:    {y.max():.6f}")

# Save predictions
output_file = f"EVAL_target01_{PROBLEM_NUM}_lightgbm_optuna.csv"
predictions_df = pd.DataFrame({'target01': y_eval_pred})
predictions_df.to_csv(output_file, index=False)

print(f"\n✓ Predictions saved to: {output_file}")
print(f"  Shape: {predictions_df.shape}")

print(f"\nSample predictions:")
print(predictions_df.head(10))

# ===============================
# 8. Feature importance
# ===============================
print("\n" + "="*70)
print("FEATURE IMPORTANCE")
print("="*70)

feature_importance = final_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': SELECTED_FEATURES,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\n" + importance_df.to_string(index=False))

# Calculate cumulative importance
importance_df['Cumulative'] = importance_df['Importance'].cumsum() / importance_df['Importance'].sum() * 100
print(f"\nTop 5 features explain {importance_df.head(5)['Cumulative'].iloc[-1]:.1f}% of total importance")

# ===============================
# 9. Optuna study visualization info
# ===============================
print("\n" + "="*70)
print("OPTIMIZATION SUMMARY")
print("="*70)

print(f"\nTotal trials: {len(study.trials)}")
print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

# Top 5 trials
print(f"\nTop 5 trials:")
print("-"*70)
trials_df = study.trials_dataframe().sort_values('value', ascending=False).head(5)
for idx, row in trials_df.iterrows():
    print(f"  Trial {int(row['number']):3d}: R² = {row['value']:.6f}")

print("\n" + "="*70)
print("OPTIMIZATION COMPLETE")
print("="*70)
