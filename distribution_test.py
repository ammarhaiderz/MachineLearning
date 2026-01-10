import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score

# Load your data
X = pd.read_csv("./data_31_40/problem_36/dataset_36.csv")
y = pd.read_csv("./data_31_40/problem_36/target_36.csv")['target01']

# Split train/val (same as before)
X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load eval set (no labels)
X_eval = pd.read_csv("./data_31_40/problem_36/EVAL_36.csv")

print("="*70)
print("DATA LOADED")
print("="*70)
print(f"Training samples: {len(X_train_full)}")
print(f"Validation samples: {len(X_val_full)}")
print(f"Eval samples: {len(X_eval)}")
print(f"Features: {X_train_full.shape[1]}")

print("\n" + "="*70)
print("DISTRIBUTION ANALYSIS: Train/Val vs Eval")
print("="*70)

# 1. Basic statistics comparison
print("\n1. Feature Statistics Comparison (first 10 features):")
print("-" * 70)

for col in X_train_full.columns[:10]:
    train_mean = X_train_full[col].mean()
    train_std = X_train_full[col].std()
    eval_mean = X_eval[col].mean()
    eval_std = X_eval[col].std()
    
    mean_diff_pct = abs(train_mean - eval_mean) / (abs(train_mean) + 1e-10) * 100
    
    status = "✓" if mean_diff_pct < 10 else "⚠️"
    print(f"{status} {col:15s} | Train: {train_mean:7.3f}±{train_std:.3f} | "
          f"Eval: {eval_mean:7.3f}±{eval_std:.3f} | Diff: {mean_diff_pct:.1f}%")

# 2. Distribution shift detection
print("\n2. Distribution Shift Detection (KS Test):")
print("-" * 70)
print("p-value > 0.05 = similar distributions ✓")
print("p-value < 0.05 = different distributions ⚠️\n")

shift_count = 0
shifted_features = []
for col in X_train_full.columns:
    ks_stat, p_value = stats.ks_2samp(X_train_full[col], X_eval[col])
    
    if p_value < 0.05:
        shift_count += 1
        if shift_count <= 5:
            shifted_features.append(col)
            print(f"⚠️ {col}: p={p_value:.4f} (SHIFTED)")

print(f"\nTotal features with significant shift: {shift_count}/{len(X_train_full.columns)}")

if shift_count < len(X_train_full.columns) * 0.1:
    print("✅ GOOD: Eval set is similar to training data")
elif shift_count < len(X_train_full.columns) * 0.3:
    print("⚠️ CAUTION: Some distribution shift detected")
else:
    print("❌ WARNING: Major distribution shift - performance may degrade")

# 3. Out-of-range values
print("\n3. Out-of-Range Value Detection:")
print("-" * 70)

oor_count = 0
oor_features = []
for col in X_train_full.columns:
    train_min, train_max = X_train_full[col].min(), X_train_full[col].max()
    eval_min, eval_max = X_eval[col].min(), X_eval[col].max()
    
    if eval_min < train_min or eval_max > train_max:
        oor_count += 1
        if oor_count <= 5:
            oor_features.append(col)
            print(f"⚠️ {col}: Train[{train_min:.2f}, {train_max:.2f}] | "
                  f"Eval[{eval_min:.2f}, {eval_max:.2f}]")

print(f"\nFeatures with out-of-range values: {oor_count}/{len(X_train_full.columns)}")

if oor_count == 0:
    print("✅ EXCELLENT: All eval values within training range")
elif oor_count < len(X_train_full.columns) * 0.1:
    print("✅ GOOD: Minimal extrapolation needed")
else:
    print("⚠️ WARNING: Model will need to extrapolate significantly")

# 4. Manual Cross-Validation
print("\n" + "="*70)
print("CROSS-VALIDATION STABILITY CHECK")
print("="*70)

# Combine train + val for CV
X_full = pd.concat([X_train_full, X_val_full], ignore_index=True)
y_full = pd.concat([y_train_full, y_val_full], ignore_index=True)

# Manual 5-fold CV
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

print("Running 5-fold cross-validation manually...")

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_full), 1):
    X_train_cv = X_full.iloc[train_idx]
    X_val_cv = X_full.iloc[val_idx]
    y_train_cv = y_full.iloc[train_idx]
    y_val_cv = y_full.iloc[val_idx]
    
    model_cv = CatBoostRegressor(
        iterations=992,
        depth=9,
        learning_rate=0.0575,
        l2_leaf_reg=1.938,
        random_state=42,
        verbose=0
    )
    
    model_cv.fit(X_train_cv, y_train_cv)
    score = r2_score(y_val_cv, model_cv.predict(X_val_cv))
    cv_scores.append(score)
    
    print(f"  Fold {fold_idx}: {score:.4f}")

cv_scores = np.array(cv_scores)
mean_score = cv_scores.mean()
std_score = cv_scores.std()

print(f"\nMean CV R²: {mean_score:.4f} (+/- {std_score:.4f})")
print(f"Min CV R²:  {cv_scores.min():.4f}")
print(f"Max CV R²:  {cv_scores.max():.4f}")

if std_score < 0.02:
    print(f"\n✅ EXCELLENT: Model is very stable (std = {std_score:.4f})")
    print(f"   Expected eval R²: {mean_score:.4f} ± {2*std_score:.4f}")
elif std_score < 0.05:
    print(f"\n✅ GOOD: Model is reasonably stable (std = {std_score:.4f})")
    print(f"   Expected eval R²: {mean_score:.4f} ± {2*std_score:.4f}")
else:
    print(f"\n⚠️ WARNING: High variance across folds (std = {std_score:.4f})")
    print(f"   Eval performance may be unpredictable")

# 5. Get feature importance for visualization
print("\n4. Generating distribution plots for top 5 features...")
model_importance = CatBoostRegressor(random_state=42, verbose=0)
model_importance.fit(X_train_full, y_train_full)
top_features = X_train_full.columns[np.argsort(model_importance.feature_importances_)[::-1]][:5]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, col in enumerate(top_features):
    ax = axes[idx]
    ax.hist(X_train_full[col], bins=30, alpha=0.5, label='Train', density=True, color='blue')
    ax.hist(X_eval[col], bins=30, alpha=0.5, label='Eval', density=True, color='red')
    ax.set_title(f'{col}')
    ax.legend()
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')

# Hide extra subplot
axes[5].set_visible(False)

plt.tight_layout()
plt.savefig('distribution_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved distribution plots to 'distribution_comparison.png'")
plt.close()

# 6. Prediction Analysis
print("\n" + "="*70)
print("PREDICTION CONFIDENCE ANALYSIS")
print("="*70)

# Train final model on all data
final_model = CatBoostRegressor(
    iterations=992, depth=9, learning_rate=0.0575,
    l2_leaf_reg=1.938, random_state=42, verbose=0
)
final_model.fit(X_train_full, y_train_full)

# Get predictions
train_preds = final_model.predict(X_train_full)
val_preds = final_model.predict(X_val_full)
eval_preds = final_model.predict(X_eval)

print("\nPrediction Statistics:")
print(f"{'Set':<10} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
print("-" * 50)
print(f"{'Train':<10} {train_preds.mean():<8.3f} {train_preds.std():<8.3f} "
      f"{train_preds.min():<8.3f} {train_preds.max():<8.3f}")
print(f"{'Val':<10} {val_preds.mean():<8.3f} {val_preds.std():<8.3f} "
      f"{val_preds.min():<8.3f} {val_preds.max():<8.3f}")
print(f"{'Eval':<10} {eval_preds.mean():<8.3f} {eval_preds.std():<8.3f} "
      f"{eval_preds.min():<8.3f} {eval_preds.max():<8.3f}")

eval_mean_diff = abs(eval_preds.mean() - train_preds.mean())
if eval_mean_diff < train_preds.std():
    print(f"\n✅ GOOD: Eval predictions centered similarly to training")
else:
    print(f"\n⚠️ WARNING: Eval predictions shifted by {eval_mean_diff:.4f}")

# Visual comparison
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(train_preds, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Training Predictions')
plt.xlabel('Predicted Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(val_preds, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.title('Validation Predictions')
plt.xlabel('Predicted Value')

plt.subplot(1, 3, 3)
plt.hist(eval_preds, bins=30, alpha=0.7, color='red', edgecolor='black')
plt.title('Eval Predictions')
plt.xlabel('Predicted Value')

plt.tight_layout()
plt.savefig('prediction_distributions.png', dpi=150, bbox_inches='tight')
print("✓ Saved prediction distributions to 'prediction_distributions.png'")
plt.close()

# 7. Final Summary
print("\n" + "="*70)
print("GENERALIZATION CONFIDENCE SUMMARY")
print("="*70)

confidence_score = 0

# Check 1: Distribution similarity
if shift_count < len(X_train_full.columns) * 0.1:
    confidence_score += 1
    print("✅ [1/5] Distributions are similar")
else:
    print("⚠️ [0/5] Significant distribution shift detected")

# Check 2: Out-of-range values
if oor_count < len(X_train_full.columns) * 0.1:
    confidence_score += 1
    print("✅ [1/5] Minimal extrapolation needed")
else:
    print("⚠️ [0/5] Significant extrapolation required")

# Check 3: CV stability
if std_score < 0.05:
    confidence_score += 1
    print(f"✅ [1/5] CV is stable (std = {std_score:.4f})")
else:
    print(f"⚠️ [0/5] CV is unstable (std = {std_score:.4f})")

# Check 4: Train-val gap
train_r2 = final_model.score(X_train_full, y_train_full)
val_r2 = final_model.score(X_val_full, y_val_full)
gap = train_r2 - val_r2

if gap < 0.05:
    confidence_score += 1
    print(f"✅ [1/5] Low overfitting (gap = {gap:.3f})")
else:
    print(f"⚠️ [0/5] Some overfitting (gap = {gap:.3f})")

# Check 5: Prediction distributions
if eval_mean_diff < train_preds.std():
    confidence_score += 1
    print("✅ [1/5] Eval predictions look reasonable")
else:
    print("⚠️ [0/5] Eval predictions seem unusual")

print("\n" + "="*70)
print(f"CONFIDENCE SCORE: {confidence_score}/5")
print("="*70)

if confidence_score >= 4:
    print("✅ HIGH CONFIDENCE: Eval performance should be close to validation")
    print(f"   Expected Eval R²: {mean_score:.4f} ± {2*std_score:.4f}")
    print(f"   Likely range: [{mean_score - 2*std_score:.4f}, {mean_score + 2*std_score:.4f}]")
elif confidence_score >= 3:
    print("⚠️ MODERATE CONFIDENCE: Some uncertainty in eval performance")
    print(f"   Expected Eval R²: {mean_score:.4f} ± {3*std_score:.4f}")
    print(f"   Likely range: [{mean_score - 3*std_score:.4f}, {mean_score + 3*std_score:.4f}]")
else:
    print("❌ LOW CONFIDENCE: Eval performance may differ significantly")
    print("   Consider investigating the distribution differences further")

print("\n" + "="*70)
print(f"Your validation R²: {val_r2:.4f}")
print(f"Your CV mean R²:    {mean_score:.4f}")
print("="*70)

# Save eval predictions