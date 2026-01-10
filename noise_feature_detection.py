"""
Statistical Noise Feature Detection
====================================
Identifies features that are likely random noise using pure statistical tests:
- Near-zero correlation with target (Pearson & Spearman)
- Low mutual information with target
- Very low variance (quasi-constant)
- Autocorrelation tests
- Normality and randomness tests
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr, normaltest, kstest, shapiro
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROBLEM_NUM = 36
RANDOM_STATE = 42

print("="*80)
print("NOISE FEATURE DETECTION ANALYSIS")
print("="*80)

# Load data
print("\n1. Loading data...")
X = pd.read_csv(f'./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv')
y_df = pd.read_csv(f'./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv')
y = y_df.iloc[:, 0].values

print(f"   Dataset shape: {X.shape}")
print(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")

# ============================================================================
# CRITERION 1: Correlation with Target
# ============================================================================
print("\n" + "="*80)
print("CRITERION 1: Correlation with Target (Pearson & Spearman)")
print("="*80)

correlations = []
for col in X.columns:
    pearson_corr, pearson_p = pearsonr(X[col], y)
    spearman_corr, spearman_p = spearmanr(X[col], y)
    
    correlations.append({
        'Feature': col,
        'Pearson_Corr': pearson_corr,
        'Pearson_P': pearson_p,
        'Spearman_Corr': spearman_corr,
        'Spearman_P': spearman_p,
        'Avg_Abs_Corr': (abs(pearson_corr) + abs(spearman_corr)) / 2
    })

corr_df = pd.DataFrame(correlations).sort_values('Avg_Abs_Corr', ascending=True)

# Features with very low correlation
low_corr_threshold = 0.05
low_corr_features = corr_df[corr_df['Avg_Abs_Corr'] < low_corr_threshold]['Feature'].tolist()

print(f"\nFeatures with average |correlation| < {low_corr_threshold}: {len(low_corr_features)}")
if len(low_corr_features) > 0:
    print("\nTop 20 lowest correlation features:")
    print(corr_df.head(20)[['Feature', 'Pearson_Corr', 'Spearman_Corr', 'Avg_Abs_Corr']].to_string(index=False))

# ============================================================================
# CRITERION 2: Mutual Information
# ============================================================================
print("\n" + "="*80)
print("CRITERION 2: Mutual Information with Target")
print("="*80)

print("\nCalculating mutual information (this may take a moment)...")
mi_scores = mutual_info_regression(X, y, random_state=RANDOM_STATE, n_neighbors=5)

mi_df = pd.DataFrame({
    'Feature': X.columns,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=True)

# Features with very low mutual information
low_mi_threshold = 0.01
low_mi_features = mi_df[mi_df['MI_Score'] < low_mi_threshold]['Feature'].tolist()

print(f"\nFeatures with MI < {low_mi_threshold}: {len(low_mi_features)}")
if len(low_mi_features) > 0:
    print("\nTop 20 lowest MI features:")
    print(mi_df.head(20).to_string(index=False))

# ============================================================================
# CRITERION 3: Variance Analysis
# ============================================================================
print("\n" + "="*80)
print("CRITERION 3: Variance Analysis")
print("="*80)

variance_df = pd.DataFrame({
    'Feature': X.columns,
    'Variance': X.var(),
    'Std': X.std(),
    'Coefficient_of_Variation': X.std() / (X.mean().abs() + 1e-10)
}).sort_values('Variance', ascending=True)

# Very low variance features (quasi-constant)
low_var_threshold = 0.01
low_var_features = variance_df[variance_df['Variance'] < low_var_threshold]['Feature'].tolist()

print(f"\nFeatures with variance < {low_var_threshold}: {len(low_var_features)}")
if len(low_var_features) > 0:
    print("\nLowest variance features:")
    print(variance_df.head(20).to_string(index=False))

# ============================================================================
# CRITERION 4: Statistical Randomness Tests
# ============================================================================
print("\n" + "="*80)
print("CRITERION 4: Statistical Randomness Tests")
print("="*80)

print("\nTesting for statistical properties (sampling 100 features)...")
np.random.seed(RANDOM_STATE)
features_to_test = np.random.choice(X.columns, min(100, len(X.columns)), replace=False)

randomness_tests = []
for i, feature in enumerate(features_to_test, 1):
    if i % 25 == 0 or i == 1:
        print(f"   Testing feature {i}/{len(features_to_test)}: {feature}")
    
    values = X[feature].dropna().values
    
    if len(values) < 20:
        continue
    
    # Test for normality (normal noise should be normally distributed)
    _, normal_p = normaltest(values) if len(values) >= 20 else (np.nan, 1.0)
    
    # Kolmogorov-Smirnov test against uniform distribution
    _, ks_p = kstest(values, 'uniform') if len(values) >= 20 else (np.nan, 1.0)
    
    # Autocorrelation (should be near zero for random noise)
    autocorr = pd.Series(values).autocorr() if len(values) > 1 else 0
    
    randomness_tests.append({
        'Feature': feature,
        'Normal_P_Value': normal_p,
        'KS_P_Value': ks_p,
        'Autocorr': autocorr,
        'Is_Normal': normal_p > 0.05,
        'Is_Random': abs(autocorr) < 0.1
    })

random_df = pd.DataFrame(randomness_tests).sort_values('Autocorr', key=abs, ascending=True)

# Features that look statistically random
random_like_features = random_df[(abs(random_df['Autocorr']) < 0.05)]['Feature'].tolist()

print(f"\nFeatures with very low autocorrelation (|r| < 0.05): {len(random_like_features)}")
if len(random_like_features) > 0:
    print("\nMost random-like features:")
    print(random_df.head(20)[['Feature', 'Autocorr', 'Normal_P_Value', 'Is_Normal']].to_string(index=False))

# ============================================================================
# COMBINED ANALYSIS: High-Confidence Noise Features
# ============================================================================
print("\n" + "="*80)
print("COMBINED STATISTICAL ANALYSIS: High-Confidence Noise Features")
print("="*80)

# Merge all criteria
all_features = set(X.columns)
low_corr_set = set(low_corr_features)
low_mi_set = set(low_mi_features)
low_var_set = set(low_var_features)
random_like_set = set(random_like_features)

# Features failing multiple criteria
noise_score = {}
for feat in all_features:
    score = 0
    criteria_failed = []
    
    if feat in low_corr_set:
        score += 1
        criteria_failed.append('Low_Correlation')
    if feat in low_mi_set:
        score += 1
        criteria_failed.append('Low_MI')
    if feat in low_var_set:
        score += 1
        criteria_failed.append('Low_Variance')
    if feat in random_like_set:
        score += 1
        criteria_failed.append('Random_Like')
    
    noise_score[feat] = {
        'Feature': feat,
        'Noise_Score': score,
        'Criteria_Failed': ', '.join(criteria_failed) if criteria_failed else 'None'
    }

noise_summary = pd.DataFrame(noise_score.values()).sort_values('Noise_Score', ascending=False)

print(f"\nNoise Score Distribution:")
print(noise_summary['Noise_Score'].value_counts().sort_index(ascending=False))

high_confidence_noise = noise_summary[noise_summary['Noise_Score'] >= 3]
moderate_confidence_noise = noise_summary[(noise_summary['Noise_Score'] == 2)]

print(f"\n{'='*80}")
print("HIGH CONFIDENCE NOISE (Failed 3+ statistical criteria):")
print(f"{'='*80}")
print(f"Count: {len(high_confidence_noise)}")
if len(high_confidence_noise) > 0:
    print("\n" + high_confidence_noise.to_string(index=False))
else:
    print("None found")

print(f"\n{'='*80}")
print("MODERATE CONFIDENCE NOISE (Failed 2 statistical criteria):")
print(f"{'='*80}")
print(f"Count: {len(moderate_confidence_noise)}")
if len(moderate_confidence_noise) > 0:
    print("\n" + moderate_confidence_noise.head(30).to_string(index=False))
    if len(moderate_confidence_noise) > 30:
        print(f"\n... and {len(moderate_confidence_noise) - 30} more")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Correlation distribution
axes[0, 0].hist(corr_df['Avg_Abs_Corr'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(low_corr_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {low_corr_threshold}')
axes[0, 0].set_xlabel('Average Absolute Correlation with Target', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Number of Features', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Distribution of Feature-Target Correlations', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Mutual Information distribution
axes[0, 1].hist(mi_df['MI_Score'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].axvline(low_mi_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold: {low_mi_threshold}')
axes[0, 1].set_xlabel('Mutual Information Score', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Number of Features', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Distribution of Mutual Information Scores', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Noise Score distribution
noise_counts = noise_summary['Noise_Score'].value_counts().sort_index()
axes[1, 0].bar(noise_counts.index, noise_counts.values, edgecolor='black', alpha=0.7, color='orange')
axes[1, 0].set_xlabel('Noise Score (# Criteria Failed)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Number of Features', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Feature Noise Score Distribution', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: Autocorrelation distribution
if len(random_df) > 0:
    axes[1, 1].hist(random_df['Autocorr'], bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].axvline(0, color='red', linestyle='-', linewidth=2, label='Zero autocorrelation')
    axes[1, 1].axvline(-0.05, color='orange', linestyle='--', linewidth=1.5, label='Random threshold')
    axes[1, 1].axvline(0.05, color='orange', linestyle='--', linewidth=1.5)
    axes[1, 1].set_xlabel('Autocorrelation', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Number of Features', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Autocorrelation Distribution (Randomness Test)', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('noise_feature_analysis.png', dpi=300, bbox_inches='tight')
print("\nSaved: noise_feature_analysis.png")
plt.show()

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

total_features = len(X.columns)
high_noise_count = len(high_confidence_noise)
moderate_noise_count = len(moderate_confidence_noise)

print(f"\nTotal Features: {total_features}")
print(f"High Confidence Noise: {high_noise_count} ({high_noise_count/total_features*100:.1f}%)")
print(f"Moderate Confidence Noise: {moderate_noise_count} ({moderate_noise_count/total_features*100:.1f}%)")
print(f"Likely Signal Features: {total_features - high_noise_count - moderate_noise_count} ({(total_features - high_noise_count - moderate_noise_count)/total_features*100:.1f}%)")

print("\n" + "="*80)
print("RECOMMENDATIONS:")
print("="*80)

if high_noise_count > 0:
    print(f"\n1. REMOVE HIGH CONFIDENCE NOISE ({high_noise_count} features)")
    print("   These features show no statistical relationship with the target")
    print("   Removing them reduces dimensionality without losing information")
    
if moderate_noise_count > 0:
    print(f"\n2. EVALUATE MODERATE NOISE ({moderate_noise_count} features)")
    print("   These show weak statistical properties")
    print("   Consider removing if model complexity is an issue")

print(f"\n3. STATISTICAL CRITERIA USED:")
print(f"   • Pearson & Spearman correlation < 0.05")
print(f"   • Mutual information < 0.01")
print(f"   • Variance < 0.01 (quasi-constant)")
print(f"   • Autocorrelation |r| < 0.05 (random-like)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Save results
noise_summary.to_csv('noise_feature_summary.csv', index=False)
print("\nSaved: noise_feature_summary.csv")
