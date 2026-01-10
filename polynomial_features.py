import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from catboost import CatBoostRegressor

# ===============================
# 1. Configuration
# ===============================
PROBLEM_NUM = 36
POLY_DEGREE = 2  # Polynomial degree
INCLUDE_BIAS = False  # Include bias term in polynomial features
INTERACTION_ONLY = False  # Only interaction terms (no x^2, x^3, etc.)

# Selected features from test_params.py
SELECTED_FEATURES = [
    'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253', 
    'feat_143', 'feat_221', 'feat_220', 'feat_160', 'feat_266', 
    'feat_138', 'feat_47', 'feat_203',
]

# Best model parameters from test_params.py
CATBOOST_PARAMS = {
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

print(f"Problem {PROBLEM_NUM}")
print(f"Using {len(SELECTED_FEATURES)} selected features")
print(f"Polynomial degree: {POLY_DEGREE}")
print(f"Interaction only: {INTERACTION_ONLY}")

# ===============================
# 2. Load data
# ===============================
X_path = f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv"
y_path = f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv"
X_eval_path = f"./data_31_40/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv"

X = pd.read_csv(X_path)
y_df = pd.read_csv(y_path)
y = y_df["target01"]  # Change to target02 if needed
X_eval = pd.read_csv(X_eval_path)

# Filter to selected features only
X_selected = X[SELECTED_FEATURES]
X_eval_selected = X_eval[SELECTED_FEATURES]

print(f"\nOriginal data shapes:")
print(f"X: {X_selected.shape}, y: {y.shape}")
print(f"X_eval: {X_eval_selected.shape}")

# ===============================
# 3. Create Polynomial Features
# ===============================
print(f"\n{'='*60}")
print("Creating polynomial features...")
print(f"{'='*60}")

poly = PolynomialFeatures(
    degree=POLY_DEGREE, 
    include_bias=INCLUDE_BIAS,
    interaction_only=INTERACTION_ONLY
)

# Fit and transform training data
X_poly = poly.fit_transform(X_selected)
X_eval_poly = poly.transform(X_eval_selected)

# Get feature names
feature_names = poly.get_feature_names_out(SELECTED_FEATURES)

print(f"\nPolynomial feature expansion:")
print(f"Original features: {X_selected.shape[1]}")
print(f"Polynomial features: {X_poly.shape[1]}")
print(f"Feature expansion ratio: {X_poly.shape[1] / X_selected.shape[1]:.2f}x")

# Convert to DataFrame for easier handling
X_poly_df = pd.DataFrame(X_poly, columns=feature_names)
X_eval_poly_df = pd.DataFrame(X_eval_poly, columns=feature_names)

# ===============================
# 4. Train/Validation Split
# ===============================
X_train, X_val, y_train, y_val = train_test_split(
    X_poly_df, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nTrain/Val split:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")

# ===============================
# 5. BASELINE - Train on Original Features
# ===============================
print(f"\n{'='*60}")
print("BASELINE MODEL - Original Features (No Polynomial)")
print(f"{'='*60}")

# Train/validation split for original features
X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nTraining with {X_train_orig.shape[1]} original features...")

baseline_model = CatBoostRegressor(**CATBOOST_PARAMS)
baseline_model.fit(X_train_orig, y_train_orig)

# Predictions
y_train_pred_baseline = baseline_model.predict(X_train_orig)
y_val_pred_baseline = baseline_model.predict(X_val_orig)

# Calculate metrics
baseline_train_r2 = r2_score(y_train_orig, y_train_pred_baseline)
baseline_train_mse = mean_squared_error(y_train_orig, y_train_pred_baseline)
baseline_train_rmse = np.sqrt(baseline_train_mse)
baseline_train_mape = mean_absolute_percentage_error(y_train_orig, y_train_pred_baseline) * 100
baseline_train_smape = np.mean(2 * np.abs(y_train_orig - y_train_pred_baseline) / 
                               (np.abs(y_train_orig) + np.abs(y_train_pred_baseline))) * 100

baseline_val_r2 = r2_score(y_val_orig, y_val_pred_baseline)
baseline_val_mse = mean_squared_error(y_val_orig, y_val_pred_baseline)
baseline_val_rmse = np.sqrt(baseline_val_mse)
baseline_val_mape = mean_absolute_percentage_error(y_val_orig, y_val_pred_baseline) * 100
baseline_val_smape = np.mean(2 * np.abs(y_val_orig - y_val_pred_baseline) / 
                             (np.abs(y_val_orig) + np.abs(y_val_pred_baseline))) * 100

print("\nBaseline Performance:")
print(f"  Train R²:    {baseline_train_r2:.6f}")
print(f"  Train RMSE:  {baseline_train_rmse:.6f}")
print(f"  Train MAPE:  {baseline_train_mape:.2f}%")
print(f"  Val R²:      {baseline_val_r2:.6f}")
print(f"  Val RMSE:    {baseline_val_rmse:.6f}")
print(f"  Val MAPE:    {baseline_val_mape:.2f}%")

# ===============================
# 6. POLYNOMIAL MODEL - Train on Polynomial Features
# ===============================
print(f"\n{'='*60}")
print("POLYNOMIAL MODEL - Enhanced Features")
print(f"{'='*60}")

print(f"\nTraining with {X_train.shape[1]} polynomial features...")

poly_model = CatBoostRegressor(**CATBOOST_PARAMS)
poly_model.fit(X_train, y_train)

# Predictions
y_train_pred_poly = poly_model.predict(X_train)
y_val_pred_poly = poly_model.predict(X_val)

# Calculate metrics
poly_train_r2 = r2_score(y_train, y_train_pred_poly)
poly_train_mse = mean_squared_error(y_train, y_train_pred_poly)
poly_train_rmse = np.sqrt(poly_train_mse)
poly_train_mape = mean_absolute_percentage_error(y_train, y_train_pred_poly) * 100
poly_train_smape = np.mean(2 * np.abs(y_train - y_train_pred_poly) / 
                           (np.abs(y_train) + np.abs(y_train_pred_poly))) * 100

poly_val_r2 = r2_score(y_val, y_val_pred_poly)
poly_val_mse = mean_squared_error(y_val, y_val_pred_poly)
poly_val_rmse = np.sqrt(poly_val_mse)
poly_val_mape = mean_absolute_percentage_error(y_val, y_val_pred_poly) * 100
poly_val_smape = np.mean(2 * np.abs(y_val - y_val_pred_poly) / 
                         (np.abs(y_val) + np.abs(y_val_pred_poly))) * 100

print("\nPolynomial Performance:")
print(f"  Train R²:    {poly_train_r2:.6f}")
print(f"  Train RMSE:  {poly_train_rmse:.6f}")
print(f"  Train MAPE:  {poly_train_mape:.2f}%")
print(f"  Val R²:      {poly_val_r2:.6f}")
print(f"  Val RMSE:    {poly_val_rmse:.6f}")
print(f"  Val MAPE:    {poly_val_mape:.2f}%")

# ===============================
# 7. COMPARISON AND IMPROVEMENT ANALYSIS
# ===============================
print(f"\n{'='*70}")
print("BASELINE vs POLYNOMIAL - COMPARISON")
print(f"{'='*70}")

print("\n" + "="*70)
print(f"{'Metric':<20} {'Baseline':>15} {'Polynomial':>15} {'Improvement':>15}")
print("="*70)

# R² comparison
r2_improvement = poly_val_r2 - baseline_val_r2
r2_pct = (r2_improvement / baseline_val_r2) * 100 if baseline_val_r2 != 0 else 0
print(f"{'Val R²':<20} {baseline_val_r2:>15.6f} {poly_val_r2:>15.6f} {r2_improvement:>+14.6f} ({r2_pct:+.2f}%)")

# RMSE comparison
rmse_improvement = baseline_val_rmse - poly_val_rmse  # Positive = improvement
rmse_pct = (rmse_improvement / baseline_val_rmse) * 100 if baseline_val_rmse != 0 else 0
print(f"{'Val RMSE':<20} {baseline_val_rmse:>15.6f} {poly_val_rmse:>15.6f} {rmse_improvement:>+14.6f} ({rmse_pct:+.2f}%)")

# MAPE comparison
mape_improvement = baseline_val_mape - poly_val_mape  # Positive = improvement
mape_pct = (mape_improvement / baseline_val_mape) * 100 if baseline_val_mape != 0 else 0
print(f"{'Val MAPE (%)':<20} {baseline_val_mape:>15.2f} {poly_val_mape:>15.2f} {mape_improvement:>+14.2f} ({mape_pct:+.2f}%)")

# SMAPE comparison
smape_improvement = baseline_val_smape - poly_val_smape  # Positive = improvement
smape_pct = (smape_improvement / baseline_val_smape) * 100 if baseline_val_smape != 0 else 0
print(f"{'Val SMAPE (%)':<20} {baseline_val_smape:>15.2f} {poly_val_smape:>15.2f} {smape_improvement:>+14.2f} ({smape_pct:+.2f}%)")

print("="*70)

# Overall conclusion
print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if r2_improvement > 0 and rmse_improvement > 0:
    print("✓ Polynomial features IMPROVED model performance!")
    print(f"  - R² increased by {r2_improvement:.6f} ({r2_pct:+.2f}%)")
    print(f"  - RMSE decreased by {rmse_improvement:.6f} ({rmse_pct:+.2f}%)")
elif r2_improvement > 0:
    print("⚠ Polynomial features show MIXED results")
    print(f"  - R² increased by {r2_improvement:.6f} ({r2_pct:+.2f}%)")
    print(f"  - But RMSE increased by {abs(rmse_improvement):.6f} ({abs(rmse_pct):+.2f}%)")
else:
    print("✗ Polynomial features DID NOT improve model performance")
    print(f"  - R² decreased by {abs(r2_improvement):.6f} ({abs(r2_pct):.2f}%)")
    print(f"  - RMSE increased by {abs(rmse_improvement):.6f} ({abs(rmse_pct):.2f}%)")
    print("\nBaseline model performs better. Polynomial features may be causing overfitting.")

print("\n" + "="*70)

# ===============================
# 8. Generate Predictions for Evaluation Set
# ===============================
print(f"\n{'='*60}")
print("Generating predictions for evaluation set...")
print(f"{'='*60}")

# Use polynomial model for predictions
y_eval_pred = poly_model.predict(X_eval_poly_df)

# Save predictions
output_df = pd.DataFrame({
    'target01': y_eval_pred
})
output_filename = f'EVAL_target01_{PROBLEM_NUM}_polynomial_{len(SELECTED_FEATURES)}feat_deg{POLY_DEGREE}.csv'
output_df.to_csv(output_filename, index=False)
print(f"\nPredictions saved to: {output_filename}")

# ===============================
# 9. Feature Importance Analysis
# ===============================
print(f"\n{'='*60}")
print("Top 20 Most Important Polynomial Features")
print(f"{'='*60}")

feature_importance = poly_model.get_feature_importance()
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(20).to_string(index=False))

# Save feature importance
importance_df.to_csv(f'polynomial_feature_importance_{len(SELECTED_FEATURES)}feat_deg{POLY_DEGREE}.csv', index=False)

# ===============================
# 10. HYBRID MODEL - Original + Top 10 Polynomial Features
# ===============================
print(f"\n{'='*70}")
print("HYBRID MODEL - Original 13 Features + Top 10 Polynomial Features")
print(f"{'='*70}")

# Get top 10 polynomial features (excluding original features)
top_10_poly_features = []
for idx, row in importance_df.iterrows():
    feat_name = row['feature']
    # Skip if it's an original feature (degree 1, no interactions)
    if ' ' not in feat_name and '^' not in feat_name:
        continue  # This is an original feature
    top_10_poly_features.append(feat_name)
    if len(top_10_poly_features) == 10:
        break

print(f"\nTop 10 polynomial features (interactions/powers):")
for i, feat in enumerate(top_10_poly_features, 1):
    importance = importance_df[importance_df['feature'] == feat]['importance'].values[0]
    print(f"  {i:2d}. {feat:<40} (importance: {importance:.4f})")

# Create hybrid dataset: original features + top 10 polynomial features
hybrid_features = list(SELECTED_FEATURES) + top_10_poly_features
X_hybrid = X_poly_df[hybrid_features]
X_eval_hybrid = X_eval_poly_df[hybrid_features]

print(f"\nHybrid feature set size: {len(hybrid_features)} features")
print(f"  - Original features: {len(SELECTED_FEATURES)}")
print(f"  - Top polynomial features: {len(top_10_poly_features)}")

# Train/validation split for hybrid features
X_train_hybrid, X_val_hybrid, y_train_hybrid, y_val_hybrid = train_test_split(
    X_hybrid, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nTraining hybrid model...")
hybrid_model = CatBoostRegressor(**CATBOOST_PARAMS)
hybrid_model.fit(X_train_hybrid, y_train_hybrid)

# Predictions
y_train_pred_hybrid = hybrid_model.predict(X_train_hybrid)
y_val_pred_hybrid = hybrid_model.predict(X_val_hybrid)

# Calculate metrics
hybrid_train_r2 = r2_score(y_train_hybrid, y_train_pred_hybrid)
hybrid_train_rmse = np.sqrt(mean_squared_error(y_train_hybrid, y_train_pred_hybrid))
hybrid_train_mape = mean_absolute_percentage_error(y_train_hybrid, y_train_pred_hybrid) * 100

hybrid_val_r2 = r2_score(y_val_hybrid, y_val_pred_hybrid)
hybrid_val_rmse = np.sqrt(mean_squared_error(y_val_hybrid, y_val_pred_hybrid))
hybrid_val_mape = mean_absolute_percentage_error(y_val_hybrid, y_val_pred_hybrid) * 100

print("\nHybrid Model Performance:")
print(f"  Train R²:    {hybrid_train_r2:.6f}")
print(f"  Train RMSE:  {hybrid_train_rmse:.6f}")
print(f"  Train MAPE:  {hybrid_train_mape:.2f}%")
print(f"  Val R²:      {hybrid_val_r2:.6f}")
print(f"  Val RMSE:    {hybrid_val_rmse:.6f}")
print(f"  Val MAPE:    {hybrid_val_mape:.2f}%")

# ===============================
# 11. FINAL COMPARISON - All Three Models
# ===============================
print(f"\n{'='*80}")
print("FINAL COMPARISON - Baseline vs Full Polynomial vs Hybrid")
print(f"{'='*80}")

print("\n" + "="*80)
print(f"{'Metric':<20} {'Baseline':>15} {'Full Poly':>15} {'Hybrid':>15} {'Best':>10}")
print("="*80)

# R² comparison
best_r2 = max(baseline_val_r2, poly_val_r2, hybrid_val_r2)
baseline_r2_mark = " ←" if baseline_val_r2 == best_r2 else ""
poly_r2_mark = " ←" if poly_val_r2 == best_r2 else ""
hybrid_r2_mark = " ←" if hybrid_val_r2 == best_r2 else ""
print(f"{'Val R²':<20} {baseline_val_r2:>15.6f}{baseline_r2_mark:>2} {poly_val_r2:>15.6f}{poly_r2_mark:>2} {hybrid_val_r2:>15.6f}{hybrid_r2_mark:>2}")

# RMSE comparison
best_rmse = min(baseline_val_rmse, poly_val_rmse, hybrid_val_rmse)
baseline_rmse_mark = " ←" if baseline_val_rmse == best_rmse else ""
poly_rmse_mark = " ←" if poly_val_rmse == best_rmse else ""
hybrid_rmse_mark = " ←" if hybrid_val_rmse == best_rmse else ""
print(f"{'Val RMSE':<20} {baseline_val_rmse:>15.6f}{baseline_rmse_mark:>2} {poly_val_rmse:>15.6f}{poly_rmse_mark:>2} {hybrid_val_rmse:>15.6f}{hybrid_rmse_mark:>2}")

# MAPE comparison
best_mape = min(baseline_val_mape, poly_val_mape, hybrid_val_mape)
baseline_mape_mark = " ←" if baseline_val_mape == best_mape else ""
poly_mape_mark = " ←" if poly_val_mape == best_mape else ""
hybrid_mape_mark = " ←" if hybrid_val_mape == best_mape else ""
print(f"{'Val MAPE (%)':<20} {baseline_val_mape:>15.2f}{baseline_mape_mark:>2} {poly_val_mape:>15.2f}{poly_mape_mark:>2} {hybrid_val_mape:>15.2f}{hybrid_mape_mark:>2}")

# Feature count
print(f"\n{'Feature Count':<20} {len(SELECTED_FEATURES):>15} {X_poly_df.shape[1]:>15} {len(hybrid_features):>15}")

print("="*80)

# Improvement analysis for hybrid vs baseline
print("\n" + "="*80)
print("HYBRID vs BASELINE IMPROVEMENT")
print("="*80)

hybrid_r2_improvement = hybrid_val_r2 - baseline_val_r2
hybrid_r2_pct = (hybrid_r2_improvement / baseline_val_r2) * 100 if baseline_val_r2 != 0 else 0
hybrid_rmse_improvement = baseline_val_rmse - hybrid_val_rmse
hybrid_rmse_pct = (hybrid_rmse_improvement / baseline_val_rmse) * 100 if baseline_val_rmse != 0 else 0
hybrid_mape_improvement = baseline_val_mape - hybrid_val_mape
hybrid_mape_pct = (hybrid_mape_improvement / baseline_val_mape) * 100 if baseline_val_mape != 0 else 0

print(f"  R² change:    {hybrid_r2_improvement:+.6f} ({hybrid_r2_pct:+.2f}%)")
print(f"  RMSE change:  {hybrid_rmse_improvement:+.6f} ({hybrid_rmse_pct:+.2f}%)")
print(f"  MAPE change:  {hybrid_mape_improvement:+.2f}% ({hybrid_mape_pct:+.2f}%)")

# Final conclusion
print("\n" + "="*80)
print("FINAL CONCLUSION")
print("="*80)

if hybrid_val_r2 > baseline_val_r2 and hybrid_val_rmse < baseline_val_rmse:
    print("✓ HYBRID model (Original + Top 10 Polynomial) is the BEST!")
    print(f"  - Improved R² by {hybrid_r2_improvement:.6f} ({hybrid_r2_pct:+.2f}%)")
    print(f"  - Reduced RMSE by {hybrid_rmse_improvement:.6f} ({hybrid_rmse_pct:+.2f}%)")
    print(f"  - Uses only {len(hybrid_features)} features vs {X_poly_df.shape[1]} in full polynomial")
    best_model_for_eval = hybrid_model
    best_features = X_eval_hybrid
    model_type = "hybrid"
elif poly_val_r2 > baseline_val_r2 and poly_val_rmse < baseline_val_rmse:
    print("✓ FULL POLYNOMIAL model is the BEST!")
    print(f"  - Improved R² by {r2_improvement:.6f} ({r2_pct:+.2f}%)")
    print(f"  - Reduced RMSE by {rmse_improvement:.6f} ({rmse_pct:+.2f}%)")
    best_model_for_eval = poly_model
    best_features = X_eval_poly_df
    model_type = "polynomial"
else:
    print("✓ BASELINE model remains the BEST!")
    print("  - Polynomial features did not improve performance significantly")
    print("  - Stick with original 13 features for simplicity and robustness")
    best_model_for_eval = baseline_model
    best_features = X_eval_selected
    model_type = "baseline"

print("\n" + "="*80)

# ===============================
# 12. Generate Final Predictions
# ===============================
print(f"\n{'='*60}")
print(f"Generating final predictions using {model_type.upper()} model...")
print(f"{'='*60}")

y_eval_final = best_model_for_eval.predict(best_features)

# Save final predictions
output_final = pd.DataFrame({
    'target01': y_eval_final
})
output_final_filename = f'EVAL_target01_{PROBLEM_NUM}_{model_type}_best.csv'
output_final.to_csv(output_final_filename, index=False)
print(f"\nFinal predictions saved to: {output_final_filename}")

# Feature importance for hybrid model
if model_type == "hybrid":
    print(f"\n{'='*60}")
    print("Hybrid Model Feature Importance")
    print(f"{'='*60}")
    
    hybrid_importance = hybrid_model.get_feature_importance()
    hybrid_importance_df = pd.DataFrame({
        'feature': hybrid_features,
        'importance': hybrid_importance
    }).sort_values('importance', ascending=False)
    
    print(hybrid_importance_df.to_string(index=False))
    hybrid_importance_df.to_csv(f'hybrid_feature_importance_{len(hybrid_features)}feat.csv', index=False)

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)
