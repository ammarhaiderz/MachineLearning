# ===============================
# Complete Statistical Analysis of Target01 
# and Feature Reduction Techniques
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, anderson, kstest, jarque_bera
from sklearn.feature_selection import (
    mutual_info_regression, f_regression, 
    VarianceThreshold, SelectKBest, RFE
)
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROBLEM_NUM = 36
np.random.seed(42)

# ===============================
# 1. Load Data
# ===============================
print("=" * 80)
print("COMPREHENSIVE STATISTICAL ANALYSIS: TARGET01 & FEATURE REDUCTION")
print("=" * 80)

X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"].values

print(f"\nData: X={X.shape}, y={y.shape}")


# ===============================
# 2. COMPLETE TARGET DISTRIBUTION ANALYSIS
# ===============================
print("\n" + "=" * 80)
print("PART 1: STATISTICAL PROPERTIES OF TARGET01")
print("=" * 80)

print("\n1. DESCRIPTIVE STATISTICS:")
print(f"   Mean:                {y.mean():.6f}")
print(f"   Median:              {np.median(y):.6f}")
print(f"   Mode (approx):       {stats.mode(np.round(y, 2), keepdims=True)[0][0]:.6f}")
print(f"   Std Dev:             {y.std():.6f}")
print(f"   Variance:            {y.var():.6f}")
print(f"   Range:               [{y.min():.6f}, {y.max():.6f}]")
print(f"   IQR:                 {np.percentile(y, 75) - np.percentile(y, 25):.6f}")
print(f"   CV (coefficient):    {(y.std() / y.mean()):.6f}")

print("\n2. MOMENTS:")
print(f"   Skewness:            {stats.skew(y):.6f}")
print(f"   Kurtosis:            {stats.kurtosis(y):.6f}")
print(f"   Interpretation:      ", end="")
if abs(stats.skew(y)) < 0.5:
    print("Approximately symmetric")
elif stats.skew(y) > 0:
    print("Right-skewed (positive)")
else:
    print("Left-skewed (negative)")

print("\n3. NORMALITY TESTS:")
# Shapiro-Wilk test
stat_sw, p_sw = shapiro(y[:5000])  # Limited to 5000 samples
print(f"   Shapiro-Wilk:        W={stat_sw:.6f}, p={p_sw:.6e}", 
      "✗ NOT normal" if p_sw < 0.05 else "✓ Normal")

# D'Agostino-Pearson test
stat_dp, p_dp = normaltest(y)
print(f"   D'Agostino-Pearson:  K²={stat_dp:.6f}, p={p_dp:.6e}",
      "✗ NOT normal" if p_dp < 0.05 else "✓ Normal")

# Jarque-Bera test
stat_jb, p_jb = jarque_bera(y)
print(f"   Jarque-Bera:         JB={stat_jb:.6f}, p={p_jb:.6e}",
      "✗ NOT normal" if p_jb < 0.05 else "✓ Normal")

# Anderson-Darling test
result_ad = anderson(y)
print(f"   Anderson-Darling:    A²={result_ad.statistic:.6f}")
for i, (sl, cv) in enumerate(zip(result_ad.significance_level, result_ad.critical_values)):
    if i == 2:  # 5% significance level
        print(f"                        At 5% level: CV={cv:.3f}",
              "✗ NOT normal" if result_ad.statistic > cv else "✓ Normal")

print("\n4. DISTRIBUTION SHAPE ANALYSIS:")
# Kernel density estimation
from scipy.stats import gaussian_kde
kde = gaussian_kde(y)
y_range = np.linspace(y.min(), y.max(), 1000)
density = kde(y_range)

# Find peaks
from scipy.signal import find_peaks
peaks_idx, properties = find_peaks(density, prominence=0.01)
print(f"   Number of modes:     {len(peaks_idx)}")
if len(peaks_idx) > 0:
    peak_locations = y_range[peaks_idx]
    print(f"   Peak locations:      {peak_locations}")
    print(f"   Peak heights:        {density[peaks_idx]}")
    
if len(peaks_idx) >= 2:
    print(f"   ⚠️  MULTIMODAL DISTRIBUTION DETECTED")
    # Find valleys
    valleys_idx, _ = find_peaks(-density, prominence=0.01)
    if len(valleys_idx) > 0:
        valley_locations = y_range[valleys_idx]
        print(f"   Valley locations:    {valley_locations}")

print("\n5. OUTLIER DETECTION:")
# Z-score method
z_scores = np.abs(stats.zscore(y))
outliers_z = np.sum(z_scores > 3)
print(f"   Z-score (|z|>3):     {outliers_z} outliers ({100*outliers_z/len(y):.2f}%)")

# IQR method
Q1, Q3 = np.percentile(y, [25, 75])
IQR = Q3 - Q1
outliers_iqr = np.sum((y < Q1 - 1.5*IQR) | (y > Q3 + 1.5*IQR))
print(f"   IQR method:          {outliers_iqr} outliers ({100*outliers_iqr/len(y):.2f}%)")

# Modified Z-score (robust)
mad = np.median(np.abs(y - np.median(y)))
modified_z = 0.6745 * (y - np.median(y)) / mad
outliers_mad = np.sum(np.abs(modified_z) > 3.5)
print(f"   Modified Z-score:    {outliers_mad} outliers ({100*outliers_mad/len(y):.2f}%)")

print("\n6. QUANTILE ANALYSIS:")
quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
for q in quantiles:
    print(f"   {int(q*100):2d}th percentile:     {np.percentile(y, q*100):.6f}")


# ===============================
# 3. VISUALIZATION
# ===============================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Histogram
axes[0, 0].hist(y, bins=100, edgecolor='black', alpha=0.7, density=True)
axes[0, 0].plot(y_range, density, 'r-', linewidth=2, label='KDE')
if len(peaks_idx) > 0:
    axes[0, 0].plot(y_range[peaks_idx], density[peaks_idx], 'go', markersize=10, label='Peaks')
axes[0, 0].set_xlabel('Target Value')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Distribution with KDE')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(y, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot (Normal)')
axes[0, 1].grid(True, alpha=0.3)

# Box plot
axes[0, 2].boxplot(y, vert=True)
axes[0, 2].set_ylabel('Target Value')
axes[0, 2].set_title('Box Plot')
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Cumulative distribution
sorted_y = np.sort(y)
cumulative = np.arange(1, len(sorted_y) + 1) / len(sorted_y)
axes[1, 0].plot(sorted_y, cumulative)
axes[1, 0].set_xlabel('Target Value')
axes[1, 0].set_ylabel('Cumulative Probability')
axes[1, 0].set_title('Empirical CDF')
axes[1, 0].grid(True, alpha=0.3)

# Violin plot
axes[1, 1].violinplot([y], vert=True, showmeans=True, showmedians=True)
axes[1, 1].set_ylabel('Target Value')
axes[1, 1].set_title('Violin Plot')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Time series (order in dataset)
axes[1, 2].plot(y, alpha=0.6, linewidth=0.5)
axes[1, 2].set_xlabel('Sample Index')
axes[1, 2].set_ylabel('Target Value')
axes[1, 2].set_title('Target Values by Index (check for temporal patterns)')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('target_statistical_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: target_statistical_analysis.png")
plt.close()


# ===============================
# 4. FEATURE SELECTION METHOD 1: CORRELATION-BASED
# ===============================
print("\n" + "=" * 80)
print("PART 2: STATISTICAL FEATURE SELECTION METHODS")
print("=" * 80)

print("\n" + "-" * 80)
print("METHOD 1: CORRELATION-BASED SELECTION")
print("-" * 80)

# Pearson correlation (linear)
pearson_corr = X.corrwith(pd.Series(y, index=X.index))
pearson_abs = pearson_corr.abs().sort_values(ascending=False)

# Spearman correlation (monotonic, rank-based)
spearman_corr = X.apply(lambda col: stats.spearmanr(col, y)[0])
spearman_abs = spearman_corr.abs().sort_values(ascending=False)

# Kendall correlation (ordinal, robust)
kendall_corr = X.apply(lambda col: stats.kendalltau(col, y)[0])
kendall_abs = kendall_corr.abs().sort_values(ascending=False)

print("\nTop 15 features by Pearson correlation:")
for i, (feat, corr) in enumerate(pearson_abs.head(15).items(), 1):
    print(f"  {i:2d}. {feat:12s}: {corr:.4f} (actual: {pearson_corr[feat]:+.4f})")

print("\nCorrelation method comparison (Top 5):")
print(f"{'Feature':<12} {'Pearson':<10} {'Spearman':<10} {'Kendall':<10}")
print("-" * 45)
for feat in pearson_abs.head(5).index:
    print(f"{feat:<12} {pearson_abs[feat]:>8.4f}  {spearman_abs[feat]:>8.4f}  {kendall_abs[feat]:>8.4f}")


# ===============================
# 5. FEATURE SELECTION METHOD 2: HYPOTHESIS TESTING
# ===============================
print("\n" + "-" * 80)
print("METHOD 2: STATISTICAL HYPOTHESIS TESTING")
print("-" * 80)

# F-test (ANOVA F-statistic)
f_scores, p_values = f_regression(X, y)
f_df = pd.DataFrame({
    'feature': X.columns,
    'f_score': f_scores,
    'p_value': p_values
}).sort_values('f_score', ascending=False)

# Bonferroni correction for multiple testing
alpha = 0.05
bonferroni_threshold = alpha / len(X.columns)
print(f"\nBonferroni correction threshold: p < {bonferroni_threshold:.6e}")

# FDR correction (Benjamini-Hochberg)
rejected_fdr, pvals_corrected_fdr, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
f_df['p_fdr_corrected'] = pvals_corrected_fdr
f_df['significant_bonferroni'] = f_df['p_value'] < bonferroni_threshold
f_df['significant_fdr'] = rejected_fdr

print(f"\nSignificant features:")
print(f"  Uncorrected (p<0.05):        {np.sum(f_df['p_value'] < 0.05)}")
print(f"  Bonferroni corrected:        {np.sum(f_df['significant_bonferroni'])}")
print(f"  FDR corrected (BH):          {np.sum(f_df['significant_fdr'])}")

print(f"\nTop 15 features by F-statistic:")
for i, row in f_df.head(15).iterrows():
    sig = ""
    if row['significant_bonferroni']:
        sig = "*** (Bonferroni)"
    elif row['significant_fdr']:
        sig = "** (FDR)"
    elif row['p_value'] < 0.05:
        sig = "* (uncorrected)"
    print(f"  {i+1:2d}. {row['feature']:12s}: F={row['f_score']:8.2f}, "
          f"p={row['p_value']:.2e}, p_fdr={row['p_fdr_corrected']:.2e} {sig}")


# ===============================
# 6. FEATURE SELECTION METHOD 3: MUTUAL INFORMATION
# ===============================
print("\n" + "-" * 80)
print("METHOD 3: MUTUAL INFORMATION (NON-LINEAR DEPENDENCY)")
print("-" * 80)

mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({
    'feature': X.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

# Normalize MI scores
mi_df['mi_normalized'] = mi_df['mi_score'] / mi_df['mi_score'].max()

print(f"\nMutual Information Statistics:")
print(f"  Max MI:    {mi_df['mi_score'].max():.6f}")
print(f"  Mean MI:   {mi_df['mi_score'].mean():.6f}")
print(f"  Median MI: {mi_df['mi_score'].median():.6f}")
print(f"  Min MI:    {mi_df['mi_score'].min():.6f}")

print(f"\nTop 15 features by Mutual Information:")
for i, row in mi_df.head(15).iterrows():
    print(f"  {i+1:2d}. {row['feature']:12s}: MI={row['mi_score']:.6f} "
          f"(normalized: {row['mi_normalized']:.4f})")


# ===============================
# 7. FEATURE SELECTION METHOD 4: VARIANCE ANALYSIS
# ===============================
print("\n" + "-" * 80)
print("METHOD 4: VARIANCE-BASED SELECTION")
print("-" * 80)

feature_variances = X.var().sort_values(ascending=False)

print(f"\nFeature Variance Statistics:")
print(f"  Max variance:    {feature_variances.max():.6f}")
print(f"  Mean variance:   {feature_variances.mean():.6f}")
print(f"  Median variance: {feature_variances.median():.6f}")
print(f"  Min variance:    {feature_variances.min():.6f}")

# Low variance features (potential noise)
threshold_low = 0.001
low_var_features = feature_variances[feature_variances < threshold_low]
print(f"\nLow variance features (var < {threshold_low}): {len(low_var_features)}")
if len(low_var_features) > 0:
    print(f"  Features: {list(low_var_features.index[:20])}")


# ===============================
# 8. FEATURE SELECTION METHOD 5: REGULARIZATION
# ===============================
print("\n" + "-" * 80)
print("METHOD 5: LASSO REGULARIZATION (L1)")
print("-" * 80)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso with cross-validation
lasso = LassoCV(cv=5, random_state=42, max_iter=5000, n_alphas=100)
lasso.fit(X_scaled, y)

lasso_coefs = pd.Series(lasso.coef_, index=X.columns)
lasso_selected = lasso_coefs[lasso_coefs != 0].abs().sort_values(ascending=False)
lasso_eliminated = lasso_coefs[lasso_coefs == 0]

print(f"\nLasso Results:")
print(f"  Optimal alpha:       {lasso.alpha_:.6f}")
print(f"  Features selected:   {len(lasso_selected)}")
print(f"  Features eliminated: {len(lasso_eliminated)}")

print(f"\nTop 15 features by Lasso coefficient:")
for i, (feat, coef) in enumerate(lasso_selected.head(15).items(), 1):
    print(f"  {i:2d}. {feat:12s}: {coef:.6f}")


# ===============================
# 9. FEATURE SELECTION METHOD 6: TREE-BASED IMPORTANCE
# ===============================
print("\n" + "-" * 80)
print("METHOD 6: RANDOM FOREST FEATURE IMPORTANCE")
print("-" * 80)

rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
rf.fit(X, y)

rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print(f"\nRandom Forest Statistics:")
print(f"  Max importance:    {rf_importance.max():.6f}")
print(f"  Mean importance:   {rf_importance.mean():.6f}")
print(f"  Median importance: {rf_importance.median():.6f}")

print(f"\nTop 15 features by RF importance:")
for i, (feat, imp) in enumerate(rf_importance.head(15).items(), 1):
    print(f"  {i:2d}. {feat:12s}: {imp:.6f}")


# ===============================
# 10. FEATURE REDUNDANCY ANALYSIS
# ===============================
print("\n" + "-" * 80)
print("METHOD 7: FEATURE REDUNDANCY (INTER-CORRELATION)")
print("-" * 80)

feature_corr_matrix = X.corr().abs()

# Find highly correlated feature pairs
threshold_redundancy = 0.95
redundant_pairs = []
for i in range(len(feature_corr_matrix.columns)):
    for j in range(i+1, len(feature_corr_matrix.columns)):
        if feature_corr_matrix.iloc[i, j] > threshold_redundancy:
            redundant_pairs.append((
                feature_corr_matrix.columns[i],
                feature_corr_matrix.columns[j],
                feature_corr_matrix.iloc[i, j]
            ))

print(f"\nHighly correlated feature pairs (|r| > {threshold_redundancy}):")
print(f"  Found {len(redundant_pairs)} redundant pairs")
if len(redundant_pairs) > 0:
    print(f"\n  Top 20 redundant pairs:")
    for feat1, feat2, corr in sorted(redundant_pairs, key=lambda x: x[2], reverse=True)[:20]:
        print(f"    {feat1:12s} ↔ {feat2:12s}: {corr:.4f}")


# ===============================
# 11. PCA ANALYSIS
# ===============================
print("\n" + "-" * 80)
print("METHOD 8: PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("-" * 80)

pca = PCA(random_state=42)
pca.fit(X_scaled)

cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(cumsum_variance >= 0.90) + 1
n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
n_components_99 = np.argmax(cumsum_variance >= 0.99) + 1

print(f"\nPCA Variance Explained:")
print(f"  90% variance: {n_components_90} components")
print(f"  95% variance: {n_components_95} components")
print(f"  99% variance: {n_components_99} components")
print(f"  Total features: {X.shape[1]}")

print(f"\nTop 10 components variance:")
for i in range(min(10, len(pca.explained_variance_ratio_))):
    print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} "
          f"(cumulative: {cumsum_variance[i]:.4f})")


# ===============================
# 12. COMPREHENSIVE RANKING
# ===============================
print("\n" + "=" * 80)
print("PART 3: COMPREHENSIVE FEATURE RANKING & ELIMINATION")
print("=" * 80)

# Create comprehensive dataframe
comparison_df = pd.DataFrame({
    'feature': X.columns,
    'pearson': pearson_abs,
    'spearman': spearman_abs,
    'kendall': kendall_abs,
    'f_score': f_df.set_index('feature')['f_score'],
    'p_value': f_df.set_index('feature')['p_value'],
    'p_fdr': f_df.set_index('feature')['p_fdr_corrected'],
    'mi_score': mi_df.set_index('feature')['mi_score'],
    'variance': feature_variances,
    'lasso_coef': lasso_coefs.abs(),
    'rf_importance': rf_importance,
    'lasso_selected': [feat in lasso_selected.index for feat in X.columns]
})

# Rank by each method
comparison_df['rank_pearson'] = comparison_df['pearson'].rank(ascending=False)
comparison_df['rank_spearman'] = comparison_df['spearman'].rank(ascending=False)
comparison_df['rank_f'] = comparison_df['f_score'].rank(ascending=False)
comparison_df['rank_mi'] = comparison_df['mi_score'].rank(ascending=False)
comparison_df['rank_rf'] = comparison_df['rf_importance'].rank(ascending=False)

# Average rank
comparison_df['avg_rank'] = (
    comparison_df['rank_pearson'] + 
    comparison_df['rank_spearman'] + 
    comparison_df['rank_f'] + 
    comparison_df['rank_mi'] + 
    comparison_df['rank_rf']
) / 5

comparison_df = comparison_df.sort_values('avg_rank')

print("\n" + "-" * 80)
print("TOP 30 FEATURES (综合排名)")
print("-" * 80)
print(f"{'Rank':<5} {'Feature':<12} {'AvgRank':<8} {'Pearson':<8} {'MI':<8} {'F-stat':<10} {'RF_Imp':<8} {'p-FDR':<10}")
print("-" * 80)
for i, row in comparison_df.head(30).iterrows():
    print(f"{int(row['avg_rank']):3d}   {row['feature']:<12} "
          f"{row['avg_rank']:>6.1f}   "
          f"{row['pearson']:>6.4f}   "
          f"{row['mi_score']:>6.4f}   "
          f"{row['f_score']:>8.1f}   "
          f"{row['rf_importance']:>6.4f}   "
          f"{row['p_fdr']:.2e}")


# ===============================
# 13. FEATURE ELIMINATION RECOMMENDATIONS
# ===============================
print("\n" + "=" * 80)
print("PART 4: FEATURE ELIMINATION RECOMMENDATIONS")
print("=" * 80)

# Method 1: Statistical significance
features_significant = comparison_df[comparison_df['p_fdr'] < 0.05]['feature'].tolist()
print(f"\n1. STATISTICALLY SIGNIFICANT (FDR-corrected p<0.05):")
print(f"   Keep: {len(features_significant)} features")
print(f"   Eliminate: {X.shape[1] - len(features_significant)} features")

# Method 2: Top K by average rank
for k in [10, 15, 20, 30, 50]:
    top_k = comparison_df.head(k)['feature'].tolist()
    print(f"\n2. TOP {k} BY AVERAGE RANK:")
    print(f"   Keep: {k} features")
    print(f"   Eliminate: {X.shape[1] - k} features")
    print(f"   Features: {top_k}")

# Method 3: Lasso selection
print(f"\n3. LASSO SELECTION:")
print(f"   Keep: {len(lasso_selected)} features")
print(f"   Eliminate: {len(lasso_eliminated)} features")
print(f"   Features: {list(lasso_selected.index)}")

# Method 4: Combined criteria
threshold_rank = 50
threshold_mi = mi_df['mi_score'].quantile(0.75)
features_combined = comparison_df[
    (comparison_df['avg_rank'] <= threshold_rank) & 
    (comparison_df['mi_score'] >= threshold_mi)
]['feature'].tolist()
print(f"\n4. COMBINED CRITERIA (rank≤{threshold_rank} AND MI≥{threshold_mi:.4f}):")
print(f"   Keep: {len(features_combined)} features")
print(f"   Eliminate: {X.shape[1] - len(features_combined)} features")
print(f"   Features: {features_combined}")


# ===============================
# 14. SAVE RESULTS
# ===============================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save comprehensive ranking
comparison_df.to_csv(f'feature_analysis_problem_{PROBLEM_NUM}.csv', index=False)
print(f"\n✓ Saved: feature_analysis_problem_{PROBLEM_NUM}.csv")

# Save recommended feature sets
recommendations = {
    'significant_fdr': features_significant,
    'top_10': comparison_df.head(10)['feature'].tolist(),
    'top_15': comparison_df.head(15)['feature'].tolist(),
    'top_20': comparison_df.head(20)['feature'].tolist(),
    'top_30': comparison_df.head(30)['feature'].tolist(),
    'lasso_selected': list(lasso_selected.index),
    'combined_criteria': features_combined
}

import json
with open(f'recommended_features_problem_{PROBLEM_NUM}.json', 'w') as f:
    json.dump(recommendations, f, indent=2)
print(f"✓ Saved: recommended_features_problem_{PROBLEM_NUM}.json")


# ===============================
# 15. VISUALIZATION
# ===============================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Correlation comparison
axes[0, 0].scatter(comparison_df['pearson'], comparison_df['spearman'], alpha=0.6)
axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=2)
axes[0, 0].set_xlabel('Pearson Correlation')
axes[0, 0].set_ylabel('Spearman Correlation')
axes[0, 0].set_title('Correlation Methods Comparison')
axes[0, 0].grid(True, alpha=0.3)

# F-statistic vs MI
axes[0, 1].scatter(comparison_df['f_score'], comparison_df['mi_score'], alpha=0.6)
axes[0, 1].set_xlabel('F-statistic')
axes[0, 1].set_ylabel('Mutual Information')
axes[0, 1].set_title('F-stat vs Mutual Information')
axes[0, 1].grid(True, alpha=0.3)

# Feature importance distribution
axes[0, 2].hist(comparison_df['rf_importance'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 2].axvline(comparison_df['rf_importance'].median(), color='r', 
                   linestyle='--', linewidth=2, label='Median')
axes[0, 2].set_xlabel('RF Importance')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('RF Importance Distribution')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Average rank distribution
axes[1, 0].hist(comparison_df['avg_rank'], bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(50, color='r', linestyle='--', linewidth=2, label='Top 50 threshold')
axes[1, 0].set_xlabel('Average Rank')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Average Rank Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# PCA variance
axes[1, 1].plot(np.arange(1, min(101, len(cumsum_variance)+1)), 
                cumsum_variance[:100], linewidth=2)
axes[1, 1].axhline(0.90, color='r', linestyle='--', linewidth=2, label='90%')
axes[1, 1].axhline(0.95, color='orange', linestyle='--', linewidth=2, label='95%')
axes[1, 1].axhline(0.99, color='green', linestyle='--', linewidth=2, label='99%')
axes[1, 1].set_xlabel('Number of Components')
axes[1, 1].set_ylabel('Cumulative Variance Explained')
axes[1, 1].set_title('PCA Cumulative Variance')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Top features comparison
top_15 = comparison_df.head(15)
x_pos = np.arange(len(top_15))
axes[1, 2].barh(x_pos, top_15['mi_score'], alpha=0.7, label='MI')
axes[1, 2].set_yticks(x_pos)
axes[1, 2].set_yticklabels(top_15['feature'], fontsize=8)
axes[1, 2].set_xlabel('Mutual Information')
axes[1, 2].set_title('Top 15 Features by MI')
axes[1, 2].invert_yaxis()
axes[1, 2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('feature_selection_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: feature_selection_analysis.png")
plt.close()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
