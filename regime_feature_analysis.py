import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
PROBLEM_NUM = 36
REGIME_THRESHOLD = -0.15  # Based on target01 bimodal valley

print("="*70)
print("REGIME FEATURE ANALYSIS - Problem 36 Target01")
print("="*70)
print("\nObjective: Identify features that discriminate between regimes")
print("           WITHOUT using target values (feature-only regime detection)")

# Load data
X_path = f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv"
y_path = f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv"

X = pd.read_csv(X_path)
y_df = pd.read_csv(y_path)
y_target01 = y_df["target01"]

print(f"\nData loaded: {X.shape[0]} samples, {X.shape[1]} features")

# ============================================================================
# STEP 1: Define regimes based on target01 (training only)
# ============================================================================
print("\n" + "="*70)
print("STEP 1: Define Regimes from Target01")
print("="*70)

# Create regime labels
regime = (y_target01 >= REGIME_THRESHOLD).astype(int)
regime_names = {0: "Regime 0 (Low)", 1: "Regime 1 (High)"}

print(f"\nRegime definition: target01 < {REGIME_THRESHOLD} → Regime 0")
print(f"                   target01 ≥ {REGIME_THRESHOLD} → Regime 1")

regime_counts = regime.value_counts().sort_index()
print(f"\nRegime distribution:")
for r, count in regime_counts.items():
    pct = count / len(regime) * 100
    print(f"  {regime_names[r]}: {count:,} samples ({pct:.2f}%)")

# Target statistics per regime
print(f"\nTarget01 statistics per regime:")
for r in [0, 1]:
    mask = regime == r
    target_vals = y_target01[mask]
    print(f"\n  {regime_names[r]}:")
    print(f"    Count: {len(target_vals):,}")
    print(f"    Mean:  {target_vals.mean():.6f}")
    print(f"    Std:   {target_vals.std():.6f}")
    print(f"    Min:   {target_vals.min():.6f}")
    print(f"    Max:   {target_vals.max():.6f}")

# ============================================================================
# STEP 2: Feature Statistics per Regime
# ============================================================================
print("\n" + "="*70)
print("STEP 2: Feature Statistics per Regime")
print("="*70)

feature_stats = []
for feat in X.columns:
    regime0_vals = X.loc[regime == 0, feat]
    regime1_vals = X.loc[regime == 1, feat]
    
    # T-test
    t_stat, t_pval = stats.ttest_ind(regime0_vals, regime1_vals)
    
    # KS test
    ks_stat, ks_pval = stats.ks_2samp(regime0_vals, regime1_vals)
    
    # Effect size (Cohen's d)
    mean_diff = regime1_vals.mean() - regime0_vals.mean()
    pooled_std = np.sqrt((regime0_vals.std()**2 + regime1_vals.std()**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    feature_stats.append({
        'feature': feat,
        'regime0_mean': regime0_vals.mean(),
        'regime1_mean': regime1_vals.mean(),
        'mean_diff': mean_diff,
        'regime0_std': regime0_vals.std(),
        'regime1_std': regime1_vals.std(),
        't_stat': abs(t_stat),
        't_pval': t_pval,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
        'cohens_d': abs(cohens_d)
    })

stats_df = pd.DataFrame(feature_stats)

# Sort by KS statistic (distribution difference)
stats_df_sorted = stats_df.sort_values('ks_stat', ascending=False)

print("\nTop 20 discriminating features (by KS-statistic):")
print("-"*70)
for idx, row in stats_df_sorted.head(20).iterrows():
    print(f"{row['feature']:12s}  KS={row['ks_stat']:.4f}  |d|={row['cohens_d']:.4f}  "
          f"t-test p={row['t_pval']:.2e}")

# Significant features (Bonferroni corrected)
alpha = 0.05
bonferroni_threshold = alpha / len(X.columns)
significant_features = stats_df[stats_df['ks_pval'] < bonferroni_threshold].sort_values('ks_stat', ascending=False)

print(f"\n{len(significant_features)} features significantly different between regimes")
print(f"(Bonferroni-corrected p < {bonferroni_threshold:.2e})")

# ============================================================================
# STEP 3: Test Regime Classification from Features Only
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Regime Classification from Features")
print("="*70)
print("\nTesting if features alone can predict regime membership...")

# Split data
X_train, X_test, regime_train, regime_test = train_test_split(
    X, regime, test_size=0.2, random_state=42, stratify=regime
)

print(f"\nTrain: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# Test 1: All features
print("\n" + "-"*70)
print("Test 1: Random Forest with ALL features")
print("-"*70)

rf_all = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_all.fit(X_train, regime_train)
regime_pred_all = rf_all.predict(X_test)
acc_all = accuracy_score(regime_test, regime_pred_all)

print(f"\nAccuracy: {acc_all:.4f}")
print("\nClassification Report:")
print(classification_report(regime_test, regime_pred_all, 
                          target_names=[regime_names[0], regime_names[1]]))

# Test 2: Top 20 discriminating features
print("\n" + "-"*70)
print("Test 2: Random Forest with TOP 20 features (by KS-stat)")
print("-"*70)

top20_features = stats_df_sorted.head(20)['feature'].tolist()
print(f"\nFeatures: {', '.join(top20_features[:10])}, ...")

X_train_top20 = X_train[top20_features]
X_test_top20 = X_test[top20_features]

rf_top20 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_top20.fit(X_train_top20, regime_train)
regime_pred_top20 = rf_top20.predict(X_test_top20)
acc_top20 = accuracy_score(regime_test, regime_pred_top20)

print(f"\nAccuracy: {acc_top20:.4f}")
print("\nClassification Report:")
print(classification_report(regime_test, regime_pred_top20,
                          target_names=[regime_names[0], regime_names[1]]))

# Test 3: Top 10 features
print("\n" + "-"*70)
print("Test 3: Random Forest with TOP 10 features")
print("-"*70)

top10_features = stats_df_sorted.head(10)['feature'].tolist()
print(f"\nFeatures: {', '.join(top10_features)}")

X_train_top10 = X_train[top10_features]
X_test_top10 = X_test[top10_features]

rf_top10 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_top10.fit(X_train_top10, regime_train)
regime_pred_top10 = rf_top10.predict(X_test_top10)
acc_top10 = accuracy_score(regime_test, regime_pred_top10)

print(f"\nAccuracy: {acc_top10:.4f}")
print("\nClassification Report:")
print(classification_report(regime_test, regime_pred_top10,
                          target_names=[regime_names[0], regime_names[1]]))

# Test 4: Top 5 features
print("\n" + "-"*70)
print("Test 4: Random Forest with TOP 5 features")
print("-"*70)

top5_features = stats_df_sorted.head(5)['feature'].tolist()
print(f"\nFeatures: {', '.join(top5_features)}")

X_train_top5 = X_train[top5_features]
X_test_top5 = X_test[top5_features]

rf_top5 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_top5.fit(X_train_top5, regime_train)
regime_pred_top5 = rf_top5.predict(X_test_top5)
acc_top5 = accuracy_score(regime_test, regime_pred_top5)

print(f"\nAccuracy: {acc_top5:.4f}")
print("\nClassification Report:")
print(classification_report(regime_test, regime_pred_top5,
                          target_names=[regime_names[0], regime_names[1]]))

# Test 5: Logistic Regression with Top 10
print("\n" + "-"*70)
print("Test 5: Logistic Regression with TOP 10 features")
print("-"*70)

lr_top10 = LogisticRegression(max_iter=1000, random_state=42)
lr_top10.fit(X_train_top10, regime_train)
regime_pred_lr = lr_top10.predict(X_test_top10)
acc_lr = accuracy_score(regime_test, regime_pred_lr)

print(f"\nAccuracy: {acc_lr:.4f}")
print("\nClassification Report:")
print(classification_report(regime_test, regime_pred_lr,
                          target_names=[regime_names[0], regime_names[1]]))

# ============================================================================
# STEP 4: Feature Importance for Regime Classification
# ============================================================================
print("\n" + "="*70)
print("STEP 4: Feature Importance for Regime Detection")
print("="*70)

# Get feature importance from best model
feature_importance = pd.DataFrame({
    'feature': top10_features,
    'importance': rf_top10.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance (Random Forest on Top 10):")
print("-"*70)
for idx, row in feature_importance.iterrows():
    print(f"{row['feature']:12s}  Importance: {row['importance']:.4f}")

# ============================================================================
# STEP 5: Summary and Recommendations
# ============================================================================
print("\n" + "="*70)
print("SUMMARY & RECOMMENDATIONS")
print("="*70)

print(f"\n1. REGIME DETECTABILITY:")
print(f"   - Regime classification accuracy: {acc_top10:.2%} (using top 10 features)")
print(f"   - This {'IS' if acc_top10 > 0.90 else 'MAY NOT BE'} sufficient for reliable regime detection")

print(f"\n2. KEY DISCRIMINATING FEATURES:")
print(f"   Top 5 features by KS-statistic:")
for feat in top5_features:
    row = stats_df[stats_df['feature'] == feat].iloc[0]
    print(f"   - {feat}: KS={row['ks_stat']:.4f}, Cohen's d={row['cohens_d']:.4f}")

print(f"\n3. REGIME CHARACTERISTICS:")
print(f"   - {regime_counts[0]:,} samples in Regime 0 ({regime_counts[0]/len(regime)*100:.1f}%)")
print(f"   - {regime_counts[1]:,} samples in Regime 1 ({regime_counts[1]/len(regime)*100:.1f}%)")
print(f"   - {len(significant_features)} features significantly different (Bonferroni-corrected)")

print(f"\n4. RECOMMENDED APPROACH FOR EVAL DATA:")
if acc_top10 > 0.90:
    print(f"   ✓ Use feature-based regime classifier (accuracy {acc_top10:.2%})")
    print(f"   ✓ Apply to EVAL data using top {len(top10_features)} features")
    print(f"   ✓ Then apply regime-specific models")
else:
    print(f"   ⚠ Feature-based regime detection may be unreliable ({acc_top10:.2%})")
    print(f"   ⚠ Consider using predicted target01 as proxy for regime")
    print(f"   ⚠ Or use unified model instead of regime-specific")

# ============================================================================
# STEP 6: Visualizations
# ============================================================================
print("\n" + "="*70)
print("STEP 6: Creating Visualizations")
print("="*70)

# Plot 1: Top features distribution comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feat in enumerate(top5_features):
    if i < 6:
        regime0_vals = X.loc[regime == 0, feat]
        regime1_vals = X.loc[regime == 1, feat]
        
        axes[i].hist(regime0_vals, bins=50, alpha=0.6, label='Regime 0', color='blue')
        axes[i].hist(regime1_vals, bins=50, alpha=0.6, label='Regime 1', color='red')
        axes[i].set_title(feat, fontweight='bold')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

# Hide last subplot if needed
if len(top5_features) < 6:
    axes[5].axis('off')

plt.tight_layout()
plt.savefig(f'problem_{PROBLEM_NUM}_regime_features_distributions.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Feature distributions saved as 'problem_{PROBLEM_NUM}_regime_features_distributions.png'")

# Plot 2: Feature importance
fig, ax = plt.subplots(figsize=(10, 6))
feature_importance_plot = feature_importance.sort_values('importance', ascending=True)
ax.barh(feature_importance_plot['feature'], feature_importance_plot['importance'], color='steelblue')
ax.set_xlabel('Importance', fontweight='bold')
ax.set_ylabel('Feature', fontweight='bold')
ax.set_title('Feature Importance for Regime Classification', fontweight='bold', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'problem_{PROBLEM_NUM}_regime_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"✓ Feature importance saved as 'problem_{PROBLEM_NUM}_regime_feature_importance.png'")

# Plot 3: Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# All features
cm_all = confusion_matrix(regime_test, regime_pred_all)
sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=['Regime 0', 'Regime 1'],
            yticklabels=['Regime 0', 'Regime 1'])
axes[0].set_title(f'All Features\nAccuracy: {acc_all:.4f}', fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Top 10 features
cm_top10 = confusion_matrix(regime_test, regime_pred_top10)
sns.heatmap(cm_top10, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Regime 0', 'Regime 1'],
            yticklabels=['Regime 0', 'Regime 1'])
axes[1].set_title(f'Top 10 Features\nAccuracy: {acc_top10:.4f}', fontweight='bold')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

# Top 5 features
cm_top5 = confusion_matrix(regime_test, regime_pred_top5)
sns.heatmap(cm_top5, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['Regime 0', 'Regime 1'],
            yticklabels=['Regime 0', 'Regime 1'])
axes[2].set_title(f'Top 5 Features\nAccuracy: {acc_top5:.4f}', fontweight='bold')
axes[2].set_ylabel('True Label')
axes[2].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(f'problem_{PROBLEM_NUM}_regime_confusion_matrices.png', dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrices saved as 'problem_{PROBLEM_NUM}_regime_confusion_matrices.png'")

plt.show()

# ============================================================================
# STEP 7: Feature Clustering Analysis
# ============================================================================
print("\n" + "="*70)
print("STEP 7: Feature Clustering Analysis")
print("="*70)
print("\nAnalyzing feature clusters to identify patterns...")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 7.1: Correlation-based feature clustering
print("\n" + "-"*70)
print("7.1: Correlation-based Feature Clustering")
print("-"*70)

# Compute correlation matrix
corr_matrix = X.corr()

# Use absolute correlation as distance (1 - |corr|)
corr_distance = 1 - np.abs(corr_matrix)

# Hierarchical clustering on features
linkage_matrix = linkage(squareform(corr_distance), method='ward')

# Cut dendrogram to get feature clusters
from scipy.cluster.hierarchy import fcluster
n_feature_clusters = 10
feature_clusters = fcluster(linkage_matrix, n_feature_clusters, criterion='maxclust')

# Analyze feature clusters
feature_cluster_df = pd.DataFrame({
    'feature': X.columns,
    'cluster': feature_clusters
})

print(f"\nCreated {n_feature_clusters} feature clusters based on correlation")
print(f"\nFeatures per cluster:")
for cluster_id in range(1, n_feature_clusters + 1):
    cluster_features = feature_cluster_df[feature_cluster_df['cluster'] == cluster_id]['feature'].tolist()
    print(f"  Cluster {cluster_id}: {len(cluster_features)} features")

# 7.2: Check if certain feature clusters are more predictive of regimes
print("\n" + "-"*70)
print("7.2: Feature Cluster Discriminative Power")
print("-"*70)

cluster_discriminative_power = []
for cluster_id in range(1, n_feature_clusters + 1):
    cluster_features = feature_cluster_df[feature_cluster_df['cluster'] == cluster_id]['feature'].tolist()
    
    # Get KS statistics for features in this cluster
    cluster_ks_stats = stats_df[stats_df['feature'].isin(cluster_features)]['ks_stat']
    
    cluster_discriminative_power.append({
        'cluster': cluster_id,
        'n_features': len(cluster_features),
        'mean_ks_stat': cluster_ks_stats.mean(),
        'max_ks_stat': cluster_ks_stats.max(),
        'top_features': ', '.join(stats_df[stats_df['feature'].isin(cluster_features)].nlargest(3, 'ks_stat')['feature'].tolist())
    })

cluster_power_df = pd.DataFrame(cluster_discriminative_power).sort_values('mean_ks_stat', ascending=False)

print("\nFeature clusters ranked by discriminative power (mean KS-statistic):")
print("-"*70)
for idx, row in cluster_power_df.iterrows():
    print(f"Cluster {row['cluster']:2d}: {row['n_features']:3d} features, "
          f"mean_KS={row['mean_ks_stat']:.4f}, max_KS={row['max_ks_stat']:.4f}")
    print(f"            Top features: {row['top_features']}")

# 7.3: K-means clustering on samples (colored by regime)
print("\n" + "-"*70)
print("7.3: Sample Clustering Analysis")
print("-"*70)

# K-means on samples
n_sample_clusters = 5
kmeans = KMeans(n_clusters=n_sample_clusters, random_state=42, n_init=10)
sample_clusters = kmeans.fit_predict(X_scaled)

# Create contingency table: sample clusters vs regimes
contingency = pd.crosstab(sample_clusters, regime, 
                          rownames=['Sample Cluster'], 
                          colnames=['Regime'],
                          margins=True)

print(f"\nK-means clustering with {n_sample_clusters} clusters")
print("\nContingency table (Sample Clusters vs Regimes):")
print(contingency)

# Calculate regime purity for each sample cluster
print("\nRegime distribution within each sample cluster:")
for cluster_id in range(n_sample_clusters):
    cluster_mask = sample_clusters == cluster_id
    regime_0_pct = (regime[cluster_mask] == 0).sum() / cluster_mask.sum() * 100
    regime_1_pct = (regime[cluster_mask] == 1).sum() / cluster_mask.sum() * 100
    print(f"  Cluster {cluster_id}: Regime 0: {regime_0_pct:.1f}%, Regime 1: {regime_1_pct:.1f}%")

# 7.4: PCA Analysis
print("\n" + "-"*70)
print("7.4: PCA Analysis")
print("-"*70)

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

print(f"\nExplained variance by first 10 PCA components:")
for i, var in enumerate(pca.explained_variance_ratio_[:10], 1):
    cumsum = pca.explained_variance_ratio_[:i].sum()
    print(f"  PC{i}: {var:.4f} (cumulative: {cumsum:.4f})")

# Check if PCA components separate regimes
print(f"\nPCA component discriminative power (KS-test):")
for i in range(5):
    pc_regime0 = X_pca[regime == 0, i]
    pc_regime1 = X_pca[regime == 1, i]
    ks_stat, ks_pval = stats.ks_2samp(pc_regime0, pc_regime1)
    print(f"  PC{i+1}: KS={ks_stat:.4f}, p={ks_pval:.2e}")

# 7.5: Visualizations
print("\n" + "-"*70)
print("7.5: Creating Clustering Visualizations")
print("-"*70)

# Plot 1: PCA - first 2 components colored by regime
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# PCA colored by regime
scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=regime, cmap='RdBu', alpha=0.5, s=10)
axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontweight='bold')
axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontweight='bold')
axes[0, 0].set_title('PCA: Samples colored by Regime', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0, 0], label='Regime')

# PCA colored by K-means cluster
scatter2 = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=sample_clusters, cmap='tab10', alpha=0.5, s=10)
axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontweight='bold')
axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontweight='bold')
axes[0, 1].set_title('PCA: Samples colored by K-means Cluster', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')

# PCA: PC3 vs PC4 colored by regime
scatter3 = axes[1, 0].scatter(X_pca[:, 2], X_pca[:, 3], c=regime, cmap='RdBu', alpha=0.5, s=10)
axes[1, 0].set_xlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', fontweight='bold')
axes[1, 0].set_ylabel(f'PC4 ({pca.explained_variance_ratio_[3]:.2%})', fontweight='bold')
axes[1, 0].set_title('PCA: PC3 vs PC4 colored by Regime', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=axes[1, 0], label='Regime')

# Scree plot
axes[1, 1].bar(range(1, 11), pca.explained_variance_ratio_[:10], color='steelblue', edgecolor='black')
axes[1, 1].plot(range(1, 11), np.cumsum(pca.explained_variance_ratio_[:10]), 'ro-', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Principal Component', fontweight='bold')
axes[1, 1].set_ylabel('Explained Variance Ratio', fontweight='bold')
axes[1, 1].set_title('PCA Scree Plot', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].set_xticks(range(1, 11))

plt.tight_layout()
plt.savefig(f'problem_{PROBLEM_NUM}_feature_clustering_pca.png', dpi=300, bbox_inches='tight')
print(f"\n✓ PCA visualization saved as 'problem_{PROBLEM_NUM}_feature_clustering_pca.png'")

# Plot 2: Feature correlation heatmap for top discriminative features
fig, ax = plt.subplots(figsize=(12, 10))
top20_corr = X[top20_features].corr()
sns.heatmap(top20_corr, cmap='coolwarm', center=0, vmin=-1, vmax=1, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix: Top 20 Discriminating Features', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(f'problem_{PROBLEM_NUM}_top_features_correlation.png', dpi=300, bbox_inches='tight')
print(f"✓ Feature correlation heatmap saved as 'problem_{PROBLEM_NUM}_top_features_correlation.png'")

# Plot 3: Feature cluster discriminative power
fig, ax = plt.subplots(figsize=(10, 6))
cluster_power_plot = cluster_power_df.sort_values('mean_ks_stat', ascending=True)
bars = ax.barh(cluster_power_plot['cluster'].astype(str), cluster_power_plot['mean_ks_stat'], color='steelblue')
ax.set_xlabel('Mean KS-Statistic', fontweight='bold')
ax.set_ylabel('Feature Cluster', fontweight='bold')
ax.set_title('Discriminative Power by Feature Cluster', fontweight='bold', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

# Add feature count annotations
for i, (idx, row) in enumerate(cluster_power_plot.iterrows()):
    ax.text(row['mean_ks_stat'] + 0.001, i, f"n={row['n_features']}", 
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'problem_{PROBLEM_NUM}_cluster_discriminative_power.png', dpi=300, bbox_inches='tight')
print(f"✓ Cluster discriminative power saved as 'problem_{PROBLEM_NUM}_cluster_discriminative_power.png'")

plt.show()

# ============================================================================
# STEP 8: Clustering Summary
# ============================================================================
print("\n" + "="*70)
print("CLUSTERING ANALYSIS SUMMARY")
print("="*70)

print(f"\n1. FEATURE CLUSTERING:")
print(f"   - Created {n_feature_clusters} feature clusters based on correlation")
best_cluster = cluster_power_df.iloc[0]
print(f"   - Most discriminative cluster: Cluster {best_cluster['cluster']} "
      f"(mean KS={best_cluster['mean_ks_stat']:.4f})")
print(f"   - Top features in best cluster: {best_cluster['top_features']}")

print(f"\n2. SAMPLE CLUSTERING:")
print(f"   - K-means with {n_sample_clusters} clusters shows mixed regime distribution")
max_purity = 0
for cluster_id in range(n_sample_clusters):
    cluster_mask = sample_clusters == cluster_id
    regime_purity = max((regime[cluster_mask] == 0).sum(), (regime[cluster_mask] == 1).sum()) / cluster_mask.sum()
    max_purity = max(max_purity, regime_purity)
print(f"   - Maximum regime purity in any cluster: {max_purity:.2%}")
if max_purity < 0.75:
    print(f"   - ⚠ Low purity indicates regimes do NOT form natural clusters")

print(f"\n3. PCA INSIGHTS:")
print(f"   - First 2 PCs explain {pca.explained_variance_ratio_[:2].sum():.2%} of variance")
print(f"   - First 10 PCs explain {pca.explained_variance_ratio_[:10].sum():.2%} of variance")
best_pc_idx = np.argmax([stats.ks_2samp(X_pca[regime == 0, i], X_pca[regime == 1, i])[0] for i in range(5)])
best_pc_ks = stats.ks_2samp(X_pca[regime == 0, best_pc_idx], X_pca[regime == 1, best_pc_idx])[0]
print(f"   - Best regime-separating PC: PC{best_pc_idx+1} (KS={best_pc_ks:.4f})")
if best_pc_ks < 0.1:
    print(f"   - ⚠ Even best PC shows weak regime separation")

print(f"\n4. KEY FINDING:")
if max_purity < 0.75 and best_pc_ks < 0.1:
    print(f"   ❌ Regimes do NOT form natural clusters in feature space")
    print(f"   ❌ Neither unsupervised clustering nor PCA reveals regime structure")
    print(f"   → Confirms that regimes are NOT feature-driven patterns")
else:
    print(f"   ⚠ Some clustering structure exists but weak")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

# Save detailed results
results_dict = {
    'top_10_features': top10_features,
    'classification_accuracy_all': acc_all,
    'classification_accuracy_top20': acc_top20,
    'classification_accuracy_top10': acc_top10,
    'classification_accuracy_top5': acc_top5,
    'significant_features_count': len(significant_features),
    'regime_counts': regime_counts.to_dict(),
    'n_feature_clusters': n_feature_clusters,
    'best_feature_cluster_ks': best_cluster['mean_ks_stat'],
    'max_sample_cluster_purity': max_purity,
    'pca_variance_2pc': pca.explained_variance_ratio_[:2].sum(),
    'best_pc_ks': best_pc_ks
}

print(f"\nKey Results:")
print(f"  - Top 10 discriminating features: {', '.join(top10_features)}")
print(f"  - Best classification accuracy: {max(acc_all, acc_top10, acc_top5):.4f}")
print(f"  - Significant features (Bonferroni): {len(significant_features)}")
print(f"  - Best feature cluster mean KS: {best_cluster['mean_ks_stat']:.4f}")
print(f"  - Max sample cluster purity: {max_purity:.2%}")
print(f"  - Best PC regime separation: {best_pc_ks:.4f}")
