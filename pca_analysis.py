"""
PCA Analysis for Problem 36 Dataset
====================================
This script performs Principal Component Analysis to understand:
- Dimensionality reduction potential
- Feature redundancy
- Variance explained by components
- Optimal number of components
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("="*80)
print("LOADING DATA")
print("="*80)

PROBLEM_NUM = 36

X_train = pd.read_csv(f'./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv')
y_train = pd.read_csv(f'./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv')

print(f"\nOriginal shape: {X_train.shape}")
print(f"Features: {X_train.shape[1]}")
print(f"Samples: {X_train.shape[0]}")

# ============================================================================
# 2. PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("PREPROCESSING")
print("="*80)

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_train)
print(f"\nMissing values imputed: {X_train.isnull().sum().sum()} total")

# Standardize features (required for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
print(f"Features standardized (mean=0, std=1)")

# ============================================================================
# 3. FULL PCA ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("FULL PCA ANALYSIS")
print("="*80)

# Fit PCA with all components
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

# Calculate cumulative explained variance
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"\nTotal components: {len(explained_variance)}")
print(f"\nVariance explained by top components:")
for i in [1, 5, 10, 20, 50, 100]:
    if i <= len(explained_variance):
        print(f"  Top {i:3d} components: {cumulative_variance[i-1]*100:.2f}%")

# Find components needed for different variance thresholds
thresholds = [0.80, 0.90, 0.95, 0.99]
print(f"\nComponents needed to explain variance:")
for threshold in thresholds:
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    print(f"  {threshold*100:.0f}% variance: {n_components} components ({n_components/X_train.shape[1]*100:.1f}% of original)")

# ============================================================================
# 4. VISUALIZATION: EXPLAINED VARIANCE
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Individual explained variance (first 50 components)
n_plot = min(50, len(explained_variance))
axes[0, 0].bar(range(1, n_plot + 1), explained_variance[:n_plot], alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Component Number', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
axes[0, 0].set_title(f'Individual Explained Variance (First {n_plot} Components)', fontsize=13, fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# Plot 2: Cumulative explained variance
axes[0, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                linewidth=2.5, marker='o', markersize=3)
for threshold in thresholds:
    n_comp = np.argmax(cumulative_variance >= threshold) + 1
    axes[0, 1].axhline(y=threshold, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    axes[0, 1].axvline(x=n_comp, color='green', linestyle='--', alpha=0.6, linewidth=1.5)
    axes[0, 1].text(n_comp + 5, threshold - 0.02, f'{n_comp} comp.', fontsize=9)

axes[0, 1].set_xlabel('Number of Components', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Cumulative Explained Variance', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Cumulative Explained Variance', fontsize=13, fontweight='bold')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_ylim([0, 1.05])

# Plot 3: Scree plot (log scale)
axes[1, 0].plot(range(1, n_plot + 1), explained_variance[:n_plot], 
                linewidth=2.5, marker='o', markersize=5, color='steelblue')
axes[1, 0].set_xlabel('Component Number', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Explained Variance Ratio (log scale)', fontsize=12, fontweight='bold')
axes[1, 0].set_title(f'Scree Plot (First {n_plot} Components)', fontsize=13, fontweight='bold')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(alpha=0.3, which='both')

# Plot 4: Component vs Original Features (dimensionality reduction)
reduction_ratios = []
variance_levels = np.arange(0.5, 1.0, 0.01)
for var_level in variance_levels:
    n_comp = np.argmax(cumulative_variance >= var_level) + 1
    reduction_ratios.append(n_comp / X_train.shape[1] * 100)

axes[1, 1].plot(variance_levels * 100, reduction_ratios, linewidth=2.5, color='purple')
axes[1, 1].axhline(y=50, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='50% reduction')
axes[1, 1].axhline(y=25, color='orange', linestyle='--', alpha=0.6, linewidth=1.5, label='75% reduction')
axes[1, 1].set_xlabel('Variance Explained (%)', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('% of Original Features Needed', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Dimensionality Reduction Efficiency', fontsize=13, fontweight='bold')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('pca_variance_analysis.png', dpi=300, bbox_inches='tight')
print("\nSaved: pca_variance_analysis.png")
plt.show()

# ============================================================================
# 5. FIRST TWO PRINCIPAL COMPONENTS
# ============================================================================
print("\n" + "="*80)
print("FIRST TWO PRINCIPAL COMPONENTS ANALYSIS")
print("="*80)

# Get target values
y = y_train.iloc[:, 0].values

# Create scatter plot of first two PCs colored by target
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PC1 vs PC2 colored by target
scatter = axes[0].scatter(X_pca_full[:, 0], X_pca_full[:, 1], 
                         c=y, cmap='viridis', alpha=0.5, s=20, edgecolors='k', linewidth=0.3)
axes[0].set_xlabel(f'PC1 ({explained_variance[0]*100:.2f}% variance)', fontsize=12, fontweight='bold')
axes[0].set_ylabel(f'PC2 ({explained_variance[1]*100:.2f}% variance)', fontsize=12, fontweight='bold')
axes[0].set_title('First Two Principal Components (colored by Target)', fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.3)
plt.colorbar(scatter, ax=axes[0], label='Target Value')

# PC1 vs PC3 colored by target
scatter2 = axes[1].scatter(X_pca_full[:, 0], X_pca_full[:, 2], 
                          c=y, cmap='viridis', alpha=0.5, s=20, edgecolors='k', linewidth=0.3)
axes[1].set_xlabel(f'PC1 ({explained_variance[0]*100:.2f}% variance)', fontsize=12, fontweight='bold')
axes[1].set_ylabel(f'PC3 ({explained_variance[2]*100:.2f}% variance)', fontsize=12, fontweight='bold')
axes[1].set_title('PC1 vs PC3 (colored by Target)', fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Target Value')

plt.tight_layout()
plt.savefig('pca_components_scatter.png', dpi=300, bbox_inches='tight')
print("Saved: pca_components_scatter.png")
plt.show()

print(f"\nPC1 vs PC2: Shows {(explained_variance[0] + explained_variance[1])*100:.2f}% of total variance")
print(f"PC1 vs PC3: Shows {(explained_variance[0] + explained_variance[2])*100:.2f}% of total variance")

# ============================================================================
# 6. FEATURE IMPORTANCE IN PRINCIPAL COMPONENTS
# ============================================================================
print("\n" + "="*80)
print("TOP FEATURES IN FIRST 3 PRINCIPAL COMPONENTS")
print("="*80)

# Get component loadings
components = pca_full.components_[:3]  # First 3 PCs
feature_names = X_train.columns

for i, component in enumerate(components):
    # Get absolute loadings
    abs_loadings = np.abs(component)
    
    # Get top 10 features
    top_indices = np.argsort(abs_loadings)[-10:][::-1]
    
    print(f"\nPC{i+1} (explains {explained_variance[i]*100:.2f}% variance):")
    print("  Top 10 contributing features:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"    {rank:2d}. {feature_names[idx]:15s} | Loading: {component[idx]:+.4f} | Abs: {abs_loadings[idx]:.4f}")

# ============================================================================
# 7. OPTIMAL NUMBER OF COMPONENTS
# ============================================================================
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

# Calculate optimal components using different criteria
n_90 = np.argmax(cumulative_variance >= 0.90) + 1
n_95 = np.argmax(cumulative_variance >= 0.95) + 1
n_99 = np.argmax(cumulative_variance >= 0.99) + 1

# Kaiser criterion (eigenvalue > 1)
eigenvalues = pca_full.explained_variance_
n_kaiser = np.sum(eigenvalues > 1)

print(f"\n1. VARIANCE-BASED SELECTION:")
print(f"   • For 90% variance: {n_90} components (reduces features by {(1-n_90/X_train.shape[1])*100:.1f}%)")
print(f"   • For 95% variance: {n_95} components (reduces features by {(1-n_95/X_train.shape[1])*100:.1f}%)")
print(f"   • For 99% variance: {n_99} components (reduces features by {(1-n_99/X_train.shape[1])*100:.1f}%)")

print(f"\n2. KAISER CRITERION (eigenvalue > 1):")
print(f"   • Keep {n_kaiser} components")

print(f"\n3. INTERPRETABILITY:")
print(f"   • Original features: {X_train.shape[1]}")
print(f"   • Sample-to-feature ratio: {X_train.shape[0]}/{X_train.shape[1]} = {X_train.shape[0]/X_train.shape[1]:.2f}")
if X_train.shape[0] / X_train.shape[1] < 10:
    print(f"   • ⚠️  Low ratio suggests dimensionality reduction is beneficial")
else:
    print(f"   • ✓ Ratio is adequate, PCA optional")

print(f"\n4. FINAL RECOMMENDATION:")
if n_95 < X_train.shape[1] * 0.5:
    print(f"   → Use PCA with {n_95} components (95% variance)")
    print(f"   → This reduces dimensionality by {(1-n_95/X_train.shape[1])*100:.1f}%")
    print(f"   → Helps prevent overfitting while retaining information")
else:
    print(f"   → PCA provides modest dimensionality reduction")
    print(f"   → Consider feature selection instead")
    print(f"   → Or use tree-based models that handle high dimensions well")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  - pca_variance_analysis.png")
print("  - pca_components_scatter.png")
