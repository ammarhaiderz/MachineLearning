import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from catboost import CatBoostRegressor

# Configuration
PROBLEM_NUM = 36

SELECTED_FEATURES = [
    'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253', 
    'feat_143', 'feat_221', 'feat_220', 'feat_160', 'feat_266', 
    'feat_138', 'feat_47', 'feat_203',
]
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
    'verbose': False,
    
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
train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
train_smape = np.mean(2 * np.abs(y_train - y_train_pred) / (np.abs(y_train) + np.abs(y_train_pred))) * 100

val_r2 = r2_score(y_val, y_val_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)
val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
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

# Plot distributions
import matplotlib.pyplot as plt

print("\n" + "="*50)
print("Distribution Comparison")
print("="*50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Target distribution
axes[0].hist(y_val, bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Target Value')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(y_val.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {y_val.mean():.4f}')
axes[0].legend()

# Plot 2: Predicted distribution
axes[1].hist(y_val_pred, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[1].set_title('Predicted Values Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted Value')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(y_val_pred.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {y_val_pred.mean():.4f}')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'problem_{PROBLEM_NUM}_distributions.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Distribution plot saved as 'problem_{PROBLEM_NUM}_distributions.png'")
plt.show()

# ============================================================================
# RESIDUAL ANALYSIS
# ============================================================================
print("\n" + "="*50)
print("RESIDUAL ANALYSIS")
print("="*50)

# Calculate residuals
residuals = y_val - y_val_pred

# Residual statistics
print(f"\nResidual Statistics:")
print(f"  Mean:   {residuals.mean():.6f}  (should be ~0)")
print(f"  Median: {residuals.median():.6f}")
print(f"  Std:    {residuals.std():.6f}")
print(f"  Min:    {residuals.min():.6f}")
print(f"  Max:    {residuals.max():.6f}")
print(f"  IQR:    {residuals.quantile(0.75) - residuals.quantile(0.25):.6f}")

# Test for normality (Shapiro-Wilk test)
from scipy import stats as scipy_stats
if len(residuals) <= 5000:
    shapiro_stat, shapiro_p = scipy_stats.shapiro(residuals)
    print(f"\nShapiro-Wilk Normality Test:")
    print(f"  Statistic: {shapiro_stat:.6f}")
    print(f"  P-value:   {shapiro_p:.6f}")
    if shapiro_p > 0.05:
        print(f"  ✓ Residuals appear normally distributed (p > 0.05)")
    else:
        print(f"  ✗ Residuals may not be normally distributed (p < 0.05)")

# Create comprehensive residual plots
fig = plt.figure(figsize=(16, 12))

# Plot 1: Residual distribution (histogram)
ax1 = plt.subplot(2, 3, 1)
n, bins, patches = ax1.hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
ax1.axvline(residuals.mean(), color='orange', linestyle='--', linewidth=2, 
           label=f'Mean: {residuals.mean():.4f}')

# Overlay normal distribution
mu, sigma = residuals.mean(), residuals.std()
x = np.linspace(residuals.min(), residuals.max(), 100)
ax1.plot(x, scipy_stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
        label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')

ax1.set_xlabel('Residuals', fontweight='bold')
ax1.set_ylabel('Density', fontweight='bold')
ax1.set_title('Residual Distribution', fontweight='bold', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Q-Q plot (quantile-quantile)
ax2 = plt.subplot(2, 3, 2)
scipy_stats.probplot(residuals, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot (Normal Distribution)', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals vs Predicted
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(y_val_pred, residuals, alpha=0.5, s=20, color='steelblue')
ax3.axhline(0, color='red', linestyle='--', linewidth=2)
ax3.axhline(residuals.std(), color='orange', linestyle=':', linewidth=1.5, label='±1 Std')
ax3.axhline(-residuals.std(), color='orange', linestyle=':', linewidth=1.5)
ax3.axhline(2*residuals.std(), color='red', linestyle=':', linewidth=1, alpha=0.5, label='±2 Std')
ax3.axhline(-2*residuals.std(), color='red', linestyle=':', linewidth=1, alpha=0.5)
ax3.set_xlabel('Predicted Values', fontweight='bold')
ax3.set_ylabel('Residuals', fontweight='bold')
ax3.set_title('Residuals vs Predicted', fontweight='bold', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Absolute Residuals vs Predicted
ax4 = plt.subplot(2, 3, 4)
ax4.scatter(y_val_pred, np.abs(residuals), alpha=0.5, s=20, color='coral')
ax4.set_xlabel('Predicted Values', fontweight='bold')
ax4.set_ylabel('Absolute Residuals', fontweight='bold')
ax4.set_title('Absolute Residuals vs Predicted', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)

# Plot 5: Residuals vs Actual
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(y_val, residuals, alpha=0.5, s=20, color='green')
ax5.axhline(0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Actual Values', fontweight='bold')
ax5.set_ylabel('Residuals', fontweight='bold')
ax5.set_title('Residuals vs Actual', fontweight='bold', fontsize=12)
ax5.grid(True, alpha=0.3)

# Plot 6: Box plot of residuals
ax6 = plt.subplot(2, 3, 6)
bp = ax6.boxplot(residuals, vert=True, patch_artist=True, 
                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))
ax6.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax6.set_ylabel('Residuals', fontweight='bold')
ax6.set_title('Residual Box Plot', fontweight='bold', fontsize=12)
ax6.grid(True, alpha=0.3, axis='y')

# Add outlier count
q1, q3 = residuals.quantile(0.25), residuals.quantile(0.75)
iqr = q3 - q1
outliers = residuals[(residuals < q1 - 1.5*iqr) | (residuals > q3 + 1.5*iqr)]
ax6.text(0.5, 0.02, f'Outliers: {len(outliers)} ({len(outliers)/len(residuals)*100:.1f}%)',
        transform=ax6.transAxes, ha='center', fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Comprehensive Residual Analysis', fontweight='bold', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(f'problem_{PROBLEM_NUM}_residual_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Residual analysis plot saved as 'problem_{PROBLEM_NUM}_residual_analysis.png'")

# ============================================================================
# INTERPRETATION GUIDE
# ============================================================================
print("\n" + "="*70)
print("HOW TO INTERPRET RESIDUAL PLOTS")
print("="*70)

print("\n1. RESIDUAL DISTRIBUTION (Top Left):")
print("   ✓ GOOD: Bell-shaped, centered at 0, symmetric")
print("   ✗ BAD:  Skewed, multi-modal, heavy tails")
print("   → Indicates if errors are normally distributed")

print("\n2. Q-Q PLOT (Top Middle):")
print("   ✓ GOOD: Points follow diagonal line closely")
print("   ✗ BAD:  Points deviate from line, especially at tails")
print("   → Tests normality assumption; deviations = non-normal errors")

print("\n3. RESIDUALS vs PREDICTED (Top Right):")
print("   ✓ GOOD: Random scatter around 0, no pattern, constant spread")
print("   ✗ BAD:  Funnel shape (heteroscedasticity), curve (non-linearity)")
print("   → Checks for systematic bias and homoscedasticity")
print("   → Funnel shape = variance increases with predicted value")
print("   → Curve = model missing non-linear relationships")

print("\n4. ABSOLUTE RESIDUALS vs PREDICTED (Bottom Left):")
print("   ✓ GOOD: Flat horizontal pattern, constant spread")
print("   ✗ BAD:  Increasing trend (heteroscedasticity)")
print("   → Easier to spot variance patterns than plot 3")

print("\n5. RESIDUALS vs ACTUAL (Bottom Middle):")
print("   ✓ GOOD: Random scatter around 0")
print("   ✗ BAD:  Patterns or trends")
print("   → Checks if errors depend on true value")

print("\n6. BOX PLOT (Bottom Right):")
print("   ✓ GOOD: Median at 0, symmetric whiskers, few outliers")
print("   ✗ BAD:  Shifted median, asymmetric, many outliers")
print("   → Shows error distribution symmetry and outliers")

print("\n" + "-"*70)
print("KEY DIAGNOSTICS:")
print("-"*70)
print(f"  Mean residual: {residuals.mean():.6f}")
if abs(residuals.mean()) < 0.01:
    print("  ✓ Near zero - no systematic bias")
else:
    print("  ⚠ Not zero - model may have systematic bias")

print(f"\n  Outliers (>1.5×IQR): {len(outliers)} ({len(outliers)/len(residuals)*100:.1f}%)")
if len(outliers)/len(residuals) < 0.05:
    print("  ✓ Few outliers - model handles most cases well")
else:
    print("  ⚠ Many outliers - some cases poorly predicted")

# Check for heteroscedasticity (Breusch-Pagan test approximation)
correlation = np.corrcoef(y_val_pred, np.abs(residuals))[0, 1]
print(f"\n  Correlation(Predicted, |Residuals|): {correlation:.4f}")
if abs(correlation) < 0.1:
    print("  ✓ Low correlation - homoscedastic (constant variance)")
else:
    print("  ⚠ High correlation - heteroscedastic (variance changes)")

print("\n" + "="*70)

plt.show()


# ============================================================================
# EVAL DATA PREDICTION & DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("EVAL DATA PREDICTION & DISTRIBUTION ANALYSIS")
print("="*70)

# Load and predict on EVAL data
X_eval = pd.read_csv(f"./data_11_20/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv")
X_eval_selected = X_eval[SELECTED_FEATURES]

print(f"\nEVAL data shape: {X_eval_selected.shape}")

# Make predictions on EVAL data
y_eval_pred = model.predict(X_eval_selected)

# TARGET Distribution statistics for EVAL
print("\n" + "-"*70)
print("TARGET DISTRIBUTION FOR EVAL (Model Predictions):")
print("-"*70)
print(f"  Count:      {len(y_eval_pred)}")
print(f"  Mean:       {y_eval_pred.mean():.6f}")
print(f"  Median:     {np.median(y_eval_pred):.6f}")
print(f"  Std:        {y_eval_pred.std():.6f}")
print(f"  Min:        {y_eval_pred.min():.6f}")
print(f"  Max:        {y_eval_pred.max():.6f}")
print(f"  Range:      {y_eval_pred.max() - y_eval_pred.min():.6f}")
print(f"  Q1 (25%):   {np.percentile(y_eval_pred, 25):.6f}")
print(f"  Q2 (50%):   {np.percentile(y_eval_pred, 50):.6f}")
print(f"  Q3 (75%):   {np.percentile(y_eval_pred, 75):.6f}")
print(f"  IQR:        {np.percentile(y_eval_pred, 75) - np.percentile(y_eval_pred, 25):.6f}")
print(f"  Skewness:   {scipy_stats.skew(y_eval_pred):.6f}")
print(f"  Kurtosis:   {scipy_stats.kurtosis(y_eval_pred):.6f}")

# Compare with training data distribution (to check if predictions are in reasonable range)
print("\n" + "-"*70)
print("Distribution Comparison (EVAL predictions vs Training targets):")
print("-"*70)
print(f"                          EVAL Pred    Train Actual    Difference")
print(f"  Mean:                  {y_eval_pred.mean():10.6f}  {y_train.mean():10.6f}    {y_eval_pred.mean() - y_train.mean():+.6f}")
print(f"  Std:                   {y_eval_pred.std():10.6f}  {y_train.std():10.6f}    {y_eval_pred.std() - y_train.std():+.6f}")
print(f"  Min:                   {y_eval_pred.min():10.6f}  {y_train.min():10.6f}    {y_eval_pred.min() - y_train.min():+.6f}")
print(f"  Max:                   {y_eval_pred.max():10.6f}  {y_train.max():10.6f}    {y_eval_pred.max() - y_train.max():+.6f}")
print(f"  Median:                {np.median(y_eval_pred):10.6f}  {np.median(y_train):10.6f}    {np.median(y_eval_pred) - np.median(y_train):+.6f}")

# Check if EVAL predictions fall within training data range
pct_within_range = np.sum((y_eval_pred >= y_train.min()) & (y_eval_pred <= y_train.max())) / len(y_eval_pred) * 100
print(f"\n  % of EVAL predictions within training range: {pct_within_range:.2f}%")
if pct_within_range < 95:
    print(f"  ⚠ Warning: {100-pct_within_range:.2f}% of predictions are outside training range")
else:
    print(f"  ✓ Most predictions within expected range")

# Visualize TARGET distribution for EVAL
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: EVAL TARGET histogram
axes[0].hist(y_eval_pred, bins=50, alpha=0.7, color='purple', edgecolor='black')
axes[0].set_title('EVAL Target Predictions Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Target Value (Predicted)')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(y_eval_pred.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {y_eval_pred.mean():.4f}')
axes[0].axvline(np.median(y_eval_pred), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(y_eval_pred):.4f}')
axes[0].legend()

# Plot 2: Compare EVAL predictions with Training target distribution
axes[1].hist(y_train, bins=50, alpha=0.6, color='blue', edgecolor='black', 
             label=f'Training Targets (Actual) μ={y_train.mean():.4f}')
axes[1].hist(y_eval_pred, bins=50, alpha=0.6, color='purple', edgecolor='black',
             label=f'EVAL Targets (Predicted) μ={y_eval_pred.mean():.4f}')
axes[1].set_title('EVAL Predictions vs Training Target Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Target Value')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Box plots - Training Actual vs Validation Actual vs EVAL Predicted
box_data = [y_train, y_val, y_eval_pred]
bp = axes[2].boxplot(box_data, labels=['Train\n(Actual)', 'Val\n(Actual)', 'EVAL\n(Predicted)'],
                      patch_artist=True)
colors = ['lightblue', 'lightgreen', 'plum']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[2].set_title('Target Distribution Comparison', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Target Value')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'problem_{PROBLEM_NUM}_eval_target_distribution.png', dpi=300, bbox_inches='tight')
print(f"\n✓ EVAL target distribution plot saved as 'problem_{PROBLEM_NUM}_eval_target_distribution.png'")
plt.show()

# Save EVAL predictions
output_file = f"EVAL_target01_{PROBLEM_NUM}_test_params.csv"
pd.DataFrame({"target01": y_eval_pred}).to_csv(output_file, index=False)
print(f"\n✓ EVAL predictions saved to: {output_file}")
print(f"  First 10 predictions: {y_eval_pred[:10]}")

print("\n" + "="*70)
