import numpy as np
import pandas as pd
from catboost_model import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from tqdm import tqdm

# ===============================
# Configuration
# ===============================
PROBLEM_NUM = 36
N_FOLDS = 5
RANDOM_SEED = 42

SELECTED_FEATURES = [
    'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253', 
    'feat_143', 'feat_221', 'feat_220', 'feat_160', 'feat_266', 
    'feat_138', 'feat_47', 'feat_203'
]

print(f"Problem {PROBLEM_NUM}")
print(f"Using {len(SELECTED_FEATURES)} selected features")
print(f"Cross-validation: {N_FOLDS}-fold\n")

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

print(f"Data shape: X={X_selected.shape}, y={y.shape}\n")

# ===============================
# Cross-validation with permutation importance
# ===============================
kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

cv_results = []
all_importances = []

print("="*80)
print("CROSS-VALIDATION WITH PERMUTATION IMPORTANCE")
print("="*80)

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_selected), 1):
    print(f"\n--- Fold {fold_idx}/{N_FOLDS} ---")
    
    # Split data
    X_train = X_selected.iloc[train_idx]
    X_val = X_selected.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Train model
    model = CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=RANDOM_SEED,
        verbose=False
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_val_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)
    print(f"Validation R²: {r2:.6f}")
    
    # Get permutation importance using CatBoost's built-in
    val_pool = Pool(X_val, y_val)
    importance = model.get_feature_importance(
        type='PredictionValuesChange',
        data=val_pool
    )
    
    # Store results
    fold_importance = pd.DataFrame({
        'feature': SELECTED_FEATURES,
        'importance': importance,
        'fold': fold_idx
    })
    
    all_importances.append(fold_importance)
    cv_results.append({'fold': fold_idx, 'r2': r2})
    
    # Show top features for this fold
    fold_importance_sorted = fold_importance.sort_values('importance', ascending=False)
    print("\nTop 5 features:")
    print(fold_importance_sorted.head(5)[['feature', 'importance']].to_string(index=False))

# ===============================
# Aggregate results across folds
# ===============================
all_importances_df = pd.concat(all_importances, ignore_index=True)

# Calculate mean and std for each feature
importance_summary = all_importances_df.groupby('feature')['importance'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max')
]).reset_index()

importance_summary = importance_summary.sort_values('mean', ascending=False)

# Calculate CV coefficient (std/mean) - lower is more stable
importance_summary['cv_coef'] = importance_summary['std'] / importance_summary['mean']

# R² across folds
cv_results_df = pd.DataFrame(cv_results)
mean_r2 = cv_results_df['r2'].mean()
std_r2 = cv_results_df['r2'].std()

print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS")
print("="*80)
print(f"\nMean R²: {mean_r2:.6f} ± {std_r2:.6f}")
print(f"R² range: [{cv_results_df['r2'].min():.6f}, {cv_results_df['r2'].max():.6f}]")

print("\n" + "="*80)
print("FEATURE IMPORTANCE ACROSS ALL FOLDS")
print("="*80)
print("\nRanked by mean importance:")
print(importance_summary.to_string(index=False))

# ===============================
# Stability analysis
# ===============================
print("\n" + "="*80)
print("STABILITY ANALYSIS")
print("="*80)

# Features with consistent importance (low CV coefficient)
stable_features = importance_summary[importance_summary['cv_coef'] < 0.5].sort_values('mean', ascending=False)
print(f"\nMost stable features (CV coefficient < 0.5):")
print(f"Total: {len(stable_features)} features")
print(stable_features[['feature', 'mean', 'std', 'cv_coef']].to_string(index=False))

# Features with variable importance (high CV coefficient)
variable_features = importance_summary[importance_summary['cv_coef'] >= 0.5].sort_values('cv_coef', ascending=False)
if len(variable_features) > 0:
    print(f"\nVariable features (CV coefficient >= 0.5):")
    print(f"Total: {len(variable_features)} features")
    print(variable_features[['feature', 'mean', 'std', 'cv_coef']].to_string(index=False))
else:
    print("\nAll features are stable across folds!")

# ===============================
# Visualization data
# ===============================
print("\n" + "="*80)
print("FEATURE IMPORTANCE BY FOLD (for visualization)")
print("="*80)

# Pivot table for easy comparison
pivot_importance = all_importances_df.pivot(index='feature', columns='fold', values='importance')
pivot_importance = pivot_importance.loc[importance_summary['feature']]  # Sort by mean importance
pivot_importance['mean'] = importance_summary.set_index('feature')['mean']
print(pivot_importance.to_string())

# ===============================
# Summary & Recommendations
# ===============================
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

print(f"\n1. Model Performance:")
print(f"   - Average R² across {N_FOLDS} folds: {mean_r2:.6f} ± {std_r2:.6f}")
print(f"   - Performance is {'stable' if std_r2 < 0.01 else 'variable'} across folds")

print(f"\n2. Feature Importance:")
top_3 = importance_summary.head(3)
print(f"   - Top 3 features: {', '.join(top_3['feature'].tolist())}")
print(f"   - Their combined mean importance: {top_3['mean'].sum():.4f}")
print(f"   - Bottom 3 features contribute < 0.01 each")

print(f"\n3. Stability:")
if len(variable_features) == 0:
    print(f"   ✓ All features are stable across CV folds")
else:
    print(f"   ⚠ {len(variable_features)} features show variable importance")
    print(f"     Consider removing: {', '.join(variable_features['feature'].head(3).tolist())}")

# Save results
output_file = f"cv_permutation_importance_problem_{PROBLEM_NUM}.csv"
importance_summary.to_csv(output_file, index=False)
print(f"\n4. Results saved to: {output_file}")
print("="*80)
