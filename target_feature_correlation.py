import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
PROBLEM_NUM = 36

print("="*70)
print(f"TARGET-FEATURE CORRELATION ANALYSIS - Problem {PROBLEM_NUM}")
print("="*70)

# Load data
X_path = f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv"
y_path = f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv"

X = pd.read_csv(X_path)
y_df = pd.read_csv(y_path)

print(f"\nData loaded:")
print(f"  Features: {X.shape[1]} columns, {X.shape[0]} rows")
print(f"  Targets: {y_df.shape[1]} columns, {y_df.shape[0]} rows")

# Concatenate features and targets
data_combined = pd.concat([X, y_df], axis=1)

print(f"\nCombined data: {data_combined.shape[1]} columns, {data_combined.shape[0]} rows")
print(f"Columns: {list(X.columns)[:5]}... + {list(y_df.columns)}")

# Compute correlation matrix
print("\nComputing correlation matrix...")
correlation_matrix = data_combined.corr()
correlation_matrix_abs = correlation_matrix.abs()

print(f"\nFull correlation matrix shape: {correlation_matrix.shape}")
print(f"  Features: {X.shape[1]}, Targets: {y_df.shape[1]}, Total: {correlation_matrix.shape[0]}")

# Save full correlation matrix to CSV
correlation_matrix.to_csv(f'problem_{PROBLEM_NUM}_full_correlation_matrix.csv')
print(f"✓ Full correlation matrix saved to 'problem_{PROBLEM_NUM}_full_correlation_matrix.csv'")

correlation_matrix_abs.to_csv(f'problem_{PROBLEM_NUM}_full_correlation_matrix_abs.csv')
print(f"✓ Absolute correlation matrix saved to 'problem_{PROBLEM_NUM}_full_correlation_matrix_abs.csv'")

# Extract correlations with targets
target_correlations = correlation_matrix[y_df.columns].drop(y_df.columns, axis=0)

print("\n" + "="*70)
print("CORRELATION WITH TARGETS")
print("="*70)

for target in y_df.columns:
    print(f"\n{target}:")
    print("-"*70)
    
    target_corr = target_correlations[target]
    target_corr_abs = target_corr.abs().sort_values(ascending=False)
    
    # Statistics
    print(f"  Max absolute correlation: {target_corr_abs.max():.6f} ({target_corr_abs.idxmax()})")
    print(f"  Mean absolute correlation: {target_corr_abs.mean():.6f}")
    print(f"  Median absolute correlation: {target_corr_abs.median():.6f}")
    
    # Count significant correlations at different thresholds
    strong = (target_corr_abs > 0.3).sum()
    moderate = (target_corr_abs > 0.1).sum()
    weak = (target_corr_abs > 0.05).sum()
    
    print(f"\n  Features with |correlation| > 0.30: {strong}")
    print(f"  Features with |correlation| > 0.10: {moderate}")
    print(f"  Features with |correlation| > 0.05: {weak}")
    
    # Top 20 correlations (by absolute value, showing sign)
    print(f"\n  Top 20 correlated features (by absolute value):")
    for i, feat in enumerate(target_corr_abs.head(20).index, 1):
        abs_corr = target_corr_abs[feat]
        orig_corr = target_corr[feat]
        print(f"    {i:2d}. {feat:12s}  |r| = {abs_corr:.6f}  (r = {orig_corr:+.6f})")

# Visualization: Correlation matrices in batches of 10 features + targets
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Create batches of 10 features each
all_features = list(X.columns)
batch_size = 10
n_batches = (len(all_features) + batch_size - 1) // batch_size  # Ceiling division

print(f"\nCreating {n_batches} correlation matrices (10 features + targets each)...")
print(f"Total features: {len(all_features)}, Targets: {len(y_df.columns)}")

for batch_idx in range(n_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(all_features))
    batch_features = all_features[start_idx:end_idx]
    
    # Combine batch features with targets
    batch_columns = batch_features + list(y_df.columns)
    batch_corr_abs = correlation_matrix_abs.loc[batch_columns, batch_columns]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(batch_corr_abs, annot=True, fmt='.3f', cmap='Reds', vmin=0, vmax=1,
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Absolute Correlation"},
                ax=ax, annot_kws={'size': 8})
    
    # Highlight target rows/columns with boxes
    n_feats_in_batch = len(batch_features)
    for i in range(len(y_df.columns)):
        # Vertical line (left of target column)
        ax.axvline(x=n_feats_in_batch + i, color='blue', linewidth=3)
        # Horizontal line (top of target row)
        ax.axhline(y=n_feats_in_batch + i, color='blue', linewidth=3)
    
    ax.set_title(f'Correlation Matrix: Batch {batch_idx + 1}/{n_batches} '
                 f'(Features {start_idx + 1}-{end_idx} + Targets)',
                 fontweight='bold', fontsize=14, pad=10)
    
    plt.tight_layout()
    print(f"  Batch {batch_idx + 1}/{n_batches}: Features {start_idx + 1}-{end_idx} ({len(batch_features)} features)")

plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

for target in y_df.columns:
    target_corr = target_correlations[target]
    print(f"\n{target}:")
    print(f"  Strongest correlation: {target_corr.abs().max():.6f} ({target_corr.abs().idxmax()})")
    print(f"  Average |correlation|: {target_corr.abs().mean():.6f}")
    
    strong_features = target_corr[target_corr.abs() > 0.1].sort_values(key=abs, ascending=False)
    if len(strong_features) > 0:
        print(f"  Features with |r| > 0.1: {len(strong_features)}")
        print(f"    Top 5: {', '.join(strong_features.head(5).index)}")
    else:
        print(f"  ⚠ No features with |r| > 0.1 (very weak correlations)")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
