# ===============================
# Target EDA and Feature Selection Analysis
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.model_selection import train_test_split

# Configuration
PROBLEM_NUM = 36
SELECTED_FEATURES = [
    'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253', 
    'feat_143', 'feat_221', 'feat_220', 'feat_160', 'feat_266', 
    'feat_138', 'feat_47', 'feat_203'
]

# ===============================
# 1. Load Data
# ===============================
X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"].values
X_eval = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv")

print(f"=" * 60)
print(f"PROBLEM {PROBLEM_NUM} - TARGET & FEATURE ANALYSIS")
print(f"=" * 60)
print(f"\nData shapes: X={X.shape}, y={y.shape}, X_eval={X_eval.shape}")


# ===============================
# 2. Target Distribution Analysis
# ===============================
print(f"\n{'=' * 60}")
print("TARGET DISTRIBUTION ANALYSIS")
print(f"{'=' * 60}")

print(f"\nBasic Statistics:")
print(f"  Mean: {y.mean():.4f}")
print(f"  Std:  {y.std():.4f}")
print(f"  Min:  {y.min():.4f}")
print(f"  Max:  {y.max():.4f}")
print(f"  Median: {np.median(y):.4f}")

# Check for bimodality
from scipy.stats import gaussian_kde
kde = gaussian_kde(y)
y_range = np.linspace(y.min(), y.max(), 1000)
density = kde(y_range)
peaks_idx = np.where((density[1:-1] > density[:-2]) & (density[1:-1] > density[2:]))[0] + 1
n_peaks = len(peaks_idx)

print(f"\nDistribution Properties:")
print(f"  Skewness: {stats.skew(y):.4f}")
print(f"  Kurtosis: {stats.kurtosis(y):.4f}")
print(f"  Number of peaks detected: {n_peaks}")

if n_peaks >= 2:
    print(f"  ⚠️  BIMODAL DISTRIBUTION DETECTED")
    peak_values = y_range[peaks_idx]
    print(f"  Peak locations: {peak_values}")
    
    # Find valley between peaks for regime split
    if n_peaks == 2:
        valley_start = int(peaks_idx[0])
        valley_end = int(peaks_idx[1])
        valley_idx = valley_start + np.argmin(density[valley_start:valley_end])
        valley_threshold = y_range[valley_idx]
        print(f"  Valley threshold: {valley_threshold:.4f}")
        
        regime = (y > valley_threshold).astype(int)
        print(f"  Regime 0 (left): {np.sum(regime == 0)} samples ({100*np.mean(regime == 0):.1f}%)")
        print(f"  Regime 1 (right): {np.sum(regime == 1)} samples ({100*np.mean(regime == 1):.1f}%)")

# Visualize target distribution
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(y, bins=100, edgecolor='black', alpha=0.7)
plt.xlabel('Target Value')
plt.ylabel('Frequency')
plt.title('Target Distribution (Histogram)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(y_range, density, linewidth=2)
if n_peaks >= 2:
    plt.scatter(y_range[peaks_idx], density[peaks_idx], color='red', s=100, zorder=5, label='Peaks')
    if n_peaks == 2:
        plt.axvline(valley_threshold, color='orange', linestyle='--', linewidth=2, label=f'Valley: {valley_threshold:.3f}')
plt.xlabel('Target Value')
plt.ylabel('Density')
plt.title('Target Distribution (KDE)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
stats.probplot(y, dist="norm", plot=plt)
plt.title('Q-Q Plot (Normality Check)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('target_distribution_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: target_distribution_analysis.png")
plt.close()


# ===============================
# 3. Feature Correlation with Target
# ===============================
print(f"\n{'=' * 60}")
print("FEATURE-TARGET CORRELATION ANALYSIS")
print(f"{'=' * 60}")

# Calculate correlations
correlations = X.corrwith(pd.Series(y, index=X.index)).abs()
correlations = correlations.sort_values(ascending=False)

print(f"\nTop 20 features by correlation with target:")
for i, (feat, corr) in enumerate(correlations.head(20).items(), 1):
    marker = "★" if feat in SELECTED_FEATURES else " "
    print(f"  {i:2d}. {marker} {feat:12s}: {corr:.4f}")

# Compare selected vs non-selected features
selected_corrs = correlations[SELECTED_FEATURES]
non_selected_corrs = correlations[~correlations.index.isin(SELECTED_FEATURES)]

print(f"\nCorrelation Statistics:")
print(f"  Selected features (n={len(SELECTED_FEATURES)}):")
print(f"    Mean: {selected_corrs.mean():.4f}")
print(f"    Median: {selected_corrs.median():.4f}")
print(f"    Min: {selected_corrs.min():.4f}, Max: {selected_corrs.max():.4f}")
print(f"\n  Non-selected features (n={len(non_selected_corrs)}):")
print(f"    Mean: {non_selected_corrs.mean():.4f}")
print(f"    Median: {non_selected_corrs.median():.4f}")
print(f"    Min: {non_selected_corrs.min():.4f}, Max: {non_selected_corrs.max():.4f}")


# ===============================
# 4. Mutual Information Analysis
# ===============================
print(f"\n{'=' * 60}")
print("MUTUAL INFORMATION ANALYSIS")
print(f"{'=' * 60}")

mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({
    'feature': X.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print(f"\nTop 20 features by mutual information:")
for i, row in mi_df.head(20).iterrows():
    feat = row['feature']
    marker = "★" if feat in SELECTED_FEATURES else " "
    print(f"  {i+1:2d}. {marker} {feat:12s}: {row['mi_score']:.4f}")

selected_mi = mi_df[mi_df['feature'].isin(SELECTED_FEATURES)]['mi_score']
non_selected_mi = mi_df[~mi_df['feature'].isin(SELECTED_FEATURES)]['mi_score']

print(f"\nMutual Information Statistics:")
print(f"  Selected features: Mean={selected_mi.mean():.4f}, Median={selected_mi.median():.4f}")
print(f"  Non-selected features: Mean={non_selected_mi.mean():.4f}, Median={non_selected_mi.median():.4f}")


# ===============================
# 5. F-statistic Analysis
# ===============================
print(f"\n{'=' * 60}")
print("F-STATISTIC ANALYSIS (LINEAR RELATIONSHIP)")
print(f"{'=' * 60}")

f_scores, p_values = f_regression(X, y)
f_df = pd.DataFrame({
    'feature': X.columns,
    'f_score': f_scores,
    'p_value': p_values
}).sort_values('f_score', ascending=False)

print(f"\nTop 20 features by F-statistic:")
for i, row in f_df.head(20).iterrows():
    feat = row['feature']
    marker = "★" if feat in SELECTED_FEATURES else " "
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    print(f"  {i+1:2d}. {marker} {feat:12s}: F={row['f_score']:.2f}, p={row['p_value']:.4e} {sig}")


# ===============================
# 6. Visualize Selected Features Behavior
# ===============================
print(f"\n{'=' * 60}")
print("VISUALIZING SELECTED FEATURES")
print(f"{'=' * 60}")

# Create comparison plots
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()

for idx, feat in enumerate(SELECTED_FEATURES):
    ax = axes[idx]
    
    # Scatter plot with color-coded by target value
    scatter = ax.scatter(X[feat], y, c=y, cmap='coolwarm', alpha=0.5, s=10)
    
    # Add regression line
    z = np.polyfit(X[feat], y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(X[feat].min(), X[feat].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)
    
    corr = X[feat].corr(pd.Series(y, index=X.index))
    mi_score = mi_df[mi_df['feature'] == feat]['mi_score'].values[0]
    
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel('Target', fontsize=9)
    ax.set_title(f'{feat}\nCorr={corr:.3f}, MI={mi_score:.3f}', fontsize=9)
    ax.grid(True, alpha=0.3)

# Remove extra subplots
for idx in range(len(SELECTED_FEATURES), len(axes)):
    fig.delaxes(axes[idx])

plt.colorbar(scatter, ax=axes[:len(SELECTED_FEATURES)], label='Target Value')
plt.tight_layout()
plt.savefig('selected_features_vs_target.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: selected_features_vs_target.png")
plt.close()


# ===============================
# 7. Feature Importance Comparison
# ===============================
print(f"\n{'=' * 60}")
print("FEATURE IMPORTANCE COMPARISON")
print(f"{'=' * 60}")

# Create a comprehensive comparison
comparison_df = pd.DataFrame({
    'feature': X.columns,
    'correlation': correlations,
    'mutual_info': mi_df.set_index('feature')['mi_score'],
    'f_score': f_df.set_index('feature')['f_score'],
    'is_selected': [feat in SELECTED_FEATURES for feat in X.columns]
})

# Rank features
comparison_df['rank_corr'] = comparison_df['correlation'].rank(ascending=False)
comparison_df['rank_mi'] = comparison_df['mutual_info'].rank(ascending=False)
comparison_df['rank_f'] = comparison_df['f_score'].rank(ascending=False)
comparison_df['avg_rank'] = (comparison_df['rank_corr'] + comparison_df['rank_mi'] + comparison_df['rank_f']) / 3

comparison_df = comparison_df.sort_values('avg_rank')

print(f"\nTop 30 features by average rank:")
for i, row in comparison_df.head(30).iterrows():
    marker = "★" if row['is_selected'] else " "
    print(f"  {int(row['avg_rank']):3d}. {marker} {row['feature']:12s}: "
          f"Corr={row['correlation']:.3f} (#{int(row['rank_corr']):3d}), "
          f"MI={row['mutual_info']:.3f} (#{int(row['rank_mi']):3d}), "
          f"F={row['f_score']:.1f} (#{int(row['rank_f']):3d})")

# Summary of selected features
selected_summary = comparison_df[comparison_df['is_selected']]
print(f"\nSelected Features Summary:")
print(f"  Average rank: {selected_summary['avg_rank'].mean():.1f}")
print(f"  Rank range: {selected_summary['avg_rank'].min():.0f} - {selected_summary['avg_rank'].max():.0f}")
print(f"  {np.sum(selected_summary['avg_rank'] <= 30)} out of {len(SELECTED_FEATURES)} are in top 30")
print(f"  {np.sum(selected_summary['avg_rank'] <= 50)} out of {len(SELECTED_FEATURES)} are in top 50")


# ===============================
# 8. Distribution Comparison: Selected vs Non-Selected
# ===============================
print(f"\n{'=' * 60}")
print("DISTRIBUTION COMPARISON")
print(f"{'=' * 60}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Correlation distribution
axes[0, 0].hist(non_selected_corrs, bins=30, alpha=0.7, label='Non-selected', edgecolor='black')
axes[0, 0].hist(selected_corrs, bins=15, alpha=0.7, label='Selected', edgecolor='black', color='orange')
axes[0, 0].axvline(selected_corrs.mean(), color='orange', linestyle='--', linewidth=2, label=f'Selected mean: {selected_corrs.mean():.3f}')
axes[0, 0].axvline(non_selected_corrs.mean(), color='blue', linestyle='--', linewidth=2, label=f'Non-selected mean: {non_selected_corrs.mean():.3f}')
axes[0, 0].set_xlabel('Absolute Correlation')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Correlation Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Mutual Information distribution
axes[0, 1].hist(non_selected_mi, bins=30, alpha=0.7, label='Non-selected', edgecolor='black')
axes[0, 1].hist(selected_mi, bins=15, alpha=0.7, label='Selected', edgecolor='black', color='orange')
axes[0, 1].axvline(selected_mi.mean(), color='orange', linestyle='--', linewidth=2, label=f'Selected mean: {selected_mi.mean():.3f}')
axes[0, 1].axvline(non_selected_mi.mean(), color='blue', linestyle='--', linewidth=2, label=f'Non-selected mean: {non_selected_mi.mean():.3f}')
axes[0, 1].set_xlabel('Mutual Information')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Mutual Information Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Rank distribution
axes[1, 0].hist(comparison_df[~comparison_df['is_selected']]['avg_rank'], bins=30, alpha=0.7, label='Non-selected', edgecolor='black')
axes[1, 0].hist(comparison_df[comparison_df['is_selected']]['avg_rank'], bins=15, alpha=0.7, label='Selected', edgecolor='black', color='orange')
axes[1, 0].set_xlabel('Average Rank')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Average Rank Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Box plot comparison
data_to_plot = [
    non_selected_corrs,
    selected_corrs,
    non_selected_mi,
    selected_mi
]
axes[1, 1].boxplot(data_to_plot, labels=['Non-sel\nCorr', 'Sel\nCorr', 'Non-sel\nMI', 'Sel\nMI'])
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Feature Quality Comparison')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('feature_distribution_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: feature_distribution_comparison.png")
plt.close()


# ===============================
# 9. If Bimodal: Regime-Specific Analysis
# ===============================
if n_peaks >= 2 and n_peaks == 2:
    print(f"\n{'=' * 60}")
    print("REGIME-SPECIFIC FEATURE ANALYSIS")
    print(f"{'=' * 60}")
    
    regime = (y > valley_threshold).astype(int)
    
    # Check if selected features help distinguish regimes
    regime_correlation = {}
    for feat in SELECTED_FEATURES:
        # Calculate correlation within each regime
        corr_0 = X.loc[regime == 0, feat].corr(pd.Series(y[regime == 0], index=X.loc[regime == 0].index))
        corr_1 = X.loc[regime == 1, feat].corr(pd.Series(y[regime == 1], index=X.loc[regime == 1].index))
        regime_correlation[feat] = {
            'regime_0': corr_0,
            'regime_1': corr_1,
            'diff': abs(corr_0 - corr_1)
        }
    
    print(f"\nSelected features behavior across regimes:")
    regime_df = pd.DataFrame(regime_correlation).T.sort_values('diff', ascending=False)
    for feat, row in regime_df.iterrows():
        print(f"  {feat:12s}: Regime0={row['regime_0']:6.3f}, Regime1={row['regime_1']:6.3f}, Diff={row['diff']:.3f}")
    
    print(f"\nKey Insights:")
    if regime_df['diff'].mean() > 0.1:
        print(f"  ✓ Selected features show DIFFERENT behavior across regimes (avg diff={regime_df['diff'].mean():.3f})")
        print(f"    → Regime-aware modeling may be beneficial")
    else:
        print(f"  ✓ Selected features show CONSISTENT behavior across regimes (avg diff={regime_df['diff'].mean():.3f})")
        print(f"    → Unified model should work well")


# ===============================
# 10. Final Summary
# ===============================
print(f"\n{'=' * 60}")
print("SUMMARY & RECOMMENDATIONS")
print(f"{'=' * 60}")

print(f"\n1. Target Distribution:")
if n_peaks >= 2:
    print(f"   - BIMODAL distribution detected")
    print(f"   - Consider: Regime-aware models or mixture models")
else:
    print(f"   - UNIMODAL distribution")
    print(f"   - Consider: Standard regression models")

print(f"\n2. Selected Features Quality:")
avg_rank = selected_summary['avg_rank'].mean()
if avg_rank <= 30:
    quality = "EXCELLENT"
elif avg_rank <= 50:
    quality = "GOOD"
elif avg_rank <= 100:
    quality = "MODERATE"
else:
    quality = "POOR"
print(f"   - Overall quality: {quality} (avg rank: {avg_rank:.1f})")
print(f"   - {np.sum(selected_summary['avg_rank'] <= 50)}/13 features in top 50")

print(f"\n3. Feature Selection Method:")
if selected_corrs.mean() > non_selected_corrs.mean() * 1.5:
    print(f"   - Selected features are MUCH BETTER than average")
    print(f"   - Correlation ratio: {selected_corrs.mean() / non_selected_corrs.mean():.2f}x")
elif selected_corrs.mean() > non_selected_corrs.mean():
    print(f"   - Selected features are BETTER than average")
    print(f"   - Correlation ratio: {selected_corrs.mean() / non_selected_corrs.mean():.2f}x")
else:
    print(f"   - Selected features are SIMILAR to average")
    print(f"   - May want to reconsider selection")

print(f"\n4. Recommended Approach:")
if n_peaks >= 2 and regime_df['diff'].mean() > 0.1:
    print(f"   ✓ Regime-aware modeling (two-stage approach)")
    print(f"   ✓ Use selected features for better discrimination")
else:
    print(f"   ✓ Unified regression model")
    print(f"   ✓ Consider feature engineering or ensemble methods")

print(f"\n{'=' * 60}")
print("ANALYSIS COMPLETE")
print(f"{'=' * 60}")
