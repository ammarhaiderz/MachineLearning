import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from catboost import CatBoostRegressor
from scipy.interpolate import interp1d

# Configuration
PROBLEM_NUM = 36
SELECTED_FEATURES = [
    'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253', 
    'feat_143', 'feat_221', 'feat_220', 'feat_160', 'feat_266', 
    'feat_138', 'feat_47', 'feat_203'
]

print("="*70)
print("WEIGHTED REGRESSION - Addressing Heteroscedasticity")
print("="*70)
print("Approach: Assign higher weights to high-variance regions")
print("="*70)

# Load data
X_path = f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv"
y_path = f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv"
X_eval_path = f"./data_31_40/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv"

X = pd.read_csv(X_path)
y_df = pd.read_csv(y_path)
y = y_df["target01"]
X_eval = pd.read_csv(X_eval_path)

X_selected = X[SELECTED_FEATURES]
X_eval_selected = X_eval[SELECTED_FEATURES]

print(f"\nData shapes:")
print(f"  X: {X_selected.shape}, y: {y.shape}")
print(f"  X_eval: {X_eval_selected.shape}")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")

# ============================================================================
# STEP 1: Train baseline model to estimate variance structure
# ============================================================================
print("\n" + "="*70)
print("STEP 1: Training Baseline Model")
print("="*70)

baseline_params = {
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

baseline_model = CatBoostRegressor(**baseline_params)
baseline_model.fit(X_train, y_train)

y_train_pred_baseline = baseline_model.predict(X_train)
y_val_pred_baseline = baseline_model.predict(X_val)

baseline_train_r2 = r2_score(y_train, y_train_pred_baseline)
baseline_val_r2 = r2_score(y_val, y_val_pred_baseline)

print(f"\nBaseline Performance:")
print(f"  Train R²: {baseline_train_r2:.6f}")
print(f"  Val R²:   {baseline_val_r2:.6f}")

# ============================================================================
# STEP 2: Estimate variance structure using cross-validation
# ============================================================================
print("\n" + "="*70)
print("STEP 2: Estimating Variance Structure (K-Fold)")
print("="*70)

# Use K-fold to get out-of-fold predictions and residuals
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_predictions = np.zeros(len(X_train))
oof_residuals = np.zeros(len(X_train))

print(f"\nRunning {n_folds}-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]
    
    fold_model = CatBoostRegressor(**baseline_params)
    fold_model.fit(X_fold_train, y_fold_train)
    
    fold_pred = fold_model.predict(X_fold_val)
    oof_predictions[val_idx] = fold_pred
    oof_residuals[val_idx] = y_fold_val - fold_pred
    
    print(f"  Fold {fold}: R² = {r2_score(y_fold_val, fold_pred):.6f}")

print(f"\nOut-of-fold R²: {r2_score(y_train, oof_predictions):.6f}")

# ============================================================================
# STEP 3: Model absolute residuals as function of predicted values
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Modeling Variance Structure")
print("="*70)

# Sort by prediction value and calculate local absolute residuals
sorted_indices = np.argsort(oof_predictions)
sorted_predictions = oof_predictions[sorted_indices]
sorted_abs_residuals = np.abs(oof_residuals[sorted_indices])

# Use rolling window to estimate local variance
window_size = 200  # ~2.5% of training data
variance_estimates = np.zeros(len(sorted_predictions))

for i in range(len(sorted_predictions)):
    start = max(0, i - window_size // 2)
    end = min(len(sorted_predictions), i + window_size // 2)
    variance_estimates[i] = sorted_abs_residuals[start:end].std()

# Create interpolation function for variance
variance_function = interp1d(
    sorted_predictions, 
    variance_estimates, 
    kind='linear',
    fill_value='extrapolate'
)

# Visualize variance structure
print(f"\nVariance structure analysis:")
quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
for q in quantiles:
    idx = int(q * (len(sorted_predictions) - 1))
    pred_val = sorted_predictions[idx]
    var_val = variance_estimates[idx]
    print(f"  Prediction {q*100:.0f}th percentile ({pred_val:+.4f}): σ = {var_val:.4f}")

# ============================================================================
# STEP 4: Calculate sample weights
# ============================================================================
print("\n" + "="*70)
print("STEP 4: Calculating Sample Weights")
print("="*70)

# Estimate variance for each training sample based on its prediction
estimated_variance = variance_function(oof_predictions)

# Weight inversely proportional to variance (higher weight for high variance)
# Add small constant to avoid division by zero
epsilon = 1e-8
raw_weights = 1.0 / (estimated_variance + epsilon)

# Normalize weights to have mean = 1
sample_weights = raw_weights / raw_weights.mean()

print(f"\nSample weight statistics:")
print(f"  Mean:   {sample_weights.mean():.4f}")
print(f"  Median: {np.median(sample_weights):.4f}")
print(f"  Std:    {sample_weights.std():.4f}")
print(f"  Min:    {sample_weights.min():.4f}")
print(f"  Max:    {sample_weights.max():.4f}")

# Show weights by prediction region
print(f"\nWeight distribution by prediction region:")
for q_low, q_high in [(0, 0.33), (0.33, 0.67), (0.67, 1.0)]:
    mask = (oof_predictions >= np.quantile(oof_predictions, q_low)) & \
           (oof_predictions <= np.quantile(oof_predictions, q_high))
    region_weights = sample_weights[mask]
    pred_range = (oof_predictions[mask].min(), oof_predictions[mask].max())
    print(f"  Predictions [{pred_range[0]:+.3f}, {pred_range[1]:+.3f}]: "
          f"mean weight = {region_weights.mean():.4f}")

# ============================================================================
# STEP 5: Train weighted model
# ============================================================================
print("\n" + "="*70)
print("STEP 5: Training Weighted Model")
print("="*70)

weighted_model = CatBoostRegressor(**baseline_params)
weighted_model.fit(X_train, y_train, sample_weight=sample_weights)

y_train_pred_weighted = weighted_model.predict(X_train)
y_val_pred_weighted = weighted_model.predict(X_val)

# ============================================================================
# STEP 6: Compare performance
# ============================================================================
print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)

def calc_metrics(y_true, y_pred, label):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    residuals = y_true - y_pred
    
    print(f"\n{label}:")
    print(f"  R²:           {r2:.6f}")
    print(f"  RMSE:         {rmse:.6f}")
    print(f"  MAPE:         {mape:.2f}%")
    print(f"  Mean Residual: {residuals.mean():.6f}")
    print(f"  Residual Std:  {residuals.std():.6f}")
    
    return r2, rmse, residuals

print("\nBASELINE MODEL:")
print("-"*70)
baseline_train_r2, baseline_train_rmse, baseline_train_res = calc_metrics(
    y_train, y_train_pred_baseline, "Train"
)
baseline_val_r2, baseline_val_rmse, baseline_val_res = calc_metrics(
    y_val, y_val_pred_baseline, "Validation"
)

print("\n\nWEIGHTED MODEL:")
print("-"*70)
weighted_train_r2, weighted_train_rmse, weighted_train_res = calc_metrics(
    y_train, y_train_pred_weighted, "Train"
)
weighted_val_r2, weighted_val_rmse, weighted_val_res = calc_metrics(
    y_val, y_val_pred_weighted, "Validation"
)

print("\n\nIMPROVEMENT:")
print("-"*70)
print(f"  Val R² improvement:   {weighted_val_r2 - baseline_val_r2:+.6f}")
print(f"  Val RMSE improvement: {baseline_val_rmse - weighted_val_rmse:+.6f}")

# ============================================================================
# STEP 7: Check if heteroscedasticity is reduced
# ============================================================================
print("\n" + "="*70)
print("HETEROSCEDASTICITY CHECK")
print("="*70)

# Correlation between predicted values and absolute residuals
baseline_het_corr = np.corrcoef(y_val_pred_baseline, np.abs(baseline_val_res))[0, 1]
weighted_het_corr = np.corrcoef(y_val_pred_weighted, np.abs(weighted_val_res))[0, 1]

print(f"\nCorrelation(Predicted, |Residuals|):")
print(f"  Baseline:  {baseline_het_corr:.4f}")
print(f"  Weighted:  {weighted_het_corr:.4f}")
print(f"  Change:    {weighted_het_corr - baseline_het_corr:+.4f}")

if abs(weighted_het_corr) < abs(baseline_het_corr):
    print(f"  ✓ Heteroscedasticity REDUCED")
else:
    print(f"  ✗ Heteroscedasticity NOT reduced")

# ============================================================================
# STEP 8: Train final model on full data and make predictions
# ============================================================================
print("\n" + "="*70)
print("FINAL MODEL TRAINING")
print("="*70)

# Combine train and validation
X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

# Estimate variance for full dataset
full_model_for_variance = CatBoostRegressor(**baseline_params)
full_model_for_variance.fit(X_full, y_full)
full_predictions = full_model_for_variance.predict(X_full)
full_estimated_variance = variance_function(full_predictions)
full_weights = 1.0 / (full_estimated_variance + epsilon)
full_weights = full_weights / full_weights.mean()

print(f"\nTraining final weighted model on {len(X_full)} samples...")

final_weighted_model = CatBoostRegressor(**baseline_params)
final_weighted_model.fit(X_full, y_full, sample_weight=full_weights)

# Predictions on EVAL
y_eval_pred = final_weighted_model.predict(X_eval_selected)

print(f"\nEVAL prediction statistics:")
print(f"  Count:  {len(y_eval_pred)}")
print(f"  Mean:   {y_eval_pred.mean():.6f}")
print(f"  Std:    {y_eval_pred.std():.6f}")
print(f"  Min:    {y_eval_pred.min():.6f}")
print(f"  Max:    {y_eval_pred.max():.6f}")

# Compare with training distribution
print(f"\nTraining target statistics:")
print(f"  Mean:   {y.mean():.6f}")
print(f"  Std:    {y.std():.6f}")
print(f"  Min:    {y.min():.6f}")
print(f"  Max:    {y.max():.6f}")

# Save predictions
output_file = f"EVAL_target01_{PROBLEM_NUM}_weighted.csv"
predictions_df = pd.DataFrame({'target01': y_eval_pred})
predictions_df.to_csv(output_file, index=False)

print(f"\n✓ Predictions saved to: {output_file}")

# ============================================================================
# STEP 9: Visualization
# ============================================================================
print("\n" + "="*70)
print("Creating Comparison Plots")
print("="*70)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Variance structure
axes[0, 0].scatter(sorted_predictions, sorted_abs_residuals, alpha=0.3, s=10, label='Observed')
axes[0, 0].plot(sorted_predictions, variance_estimates, 'r-', linewidth=2, label='Estimated σ')
axes[0, 0].set_xlabel('Predicted Value', fontweight='bold')
axes[0, 0].set_ylabel('Absolute Residual', fontweight='bold')
axes[0, 0].set_title('Variance Structure', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Sample weights distribution
axes[0, 1].hist(sample_weights, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(1.0, color='red', linestyle='--', linewidth=2, label='Mean=1.0')
axes[0, 1].set_xlabel('Sample Weight', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Weight Distribution', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Weights vs Predictions
axes[0, 2].scatter(oof_predictions, sample_weights, alpha=0.5, s=20, color='purple')
axes[0, 2].set_xlabel('Predicted Value', fontweight='bold')
axes[0, 2].set_ylabel('Sample Weight', fontweight='bold')
axes[0, 2].set_title('Weights vs Predictions', fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Baseline residuals vs predicted
axes[1, 0].scatter(y_val_pred_baseline, baseline_val_res, alpha=0.5, s=20, color='coral')
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Predicted Value', fontweight='bold')
axes[1, 0].set_ylabel('Residuals', fontweight='bold')
axes[1, 0].set_title(f'Baseline Model (r={baseline_het_corr:.3f})', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Weighted residuals vs predicted
axes[1, 1].scatter(y_val_pred_weighted, weighted_val_res, alpha=0.5, s=20, color='green')
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Predicted Value', fontweight='bold')
axes[1, 1].set_ylabel('Residuals', fontweight='bold')
axes[1, 1].set_title(f'Weighted Model (r={weighted_het_corr:.3f})', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: R² comparison
models = ['Baseline\nTrain', 'Baseline\nVal', 'Weighted\nTrain', 'Weighted\nVal']
r2_values = [baseline_train_r2, baseline_val_r2, weighted_train_r2, weighted_val_r2]
colors = ['lightcoral', 'coral', 'lightgreen', 'green']

bars = axes[1, 2].bar(models, r2_values, color=colors, edgecolor='black', alpha=0.7)
axes[1, 2].set_ylabel('R² Score', fontweight='bold')
axes[1, 2].set_title('Model Comparison', fontweight='bold')
axes[1, 2].set_ylim([0.95, 1.0])
axes[1, 2].grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, r2_values):
    height = bar.get_height()
    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Weighted Regression Analysis', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.savefig(f'problem_{PROBLEM_NUM}_weighted_regression.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Comparison plot saved as 'problem_{PROBLEM_NUM}_weighted_regression.png'")

plt.show()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
