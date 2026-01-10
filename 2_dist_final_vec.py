# ============================================================
# 0. Imports
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

from catboost import CatBoostClassifier, CatBoostRegressor

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# ============================================================
# 1. Load data
# ============================================================
PROBLEM_NUM = 36

SELECTED_FEATURES = [ 
    'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253', 
    'feat_143', 'feat_221', 'feat_220', 'feat_160', 'feat_266', 
    'feat_138', 'feat_47', 'feat_203',
]

X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"].values
X_eval = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv")

X = X[SELECTED_FEATURES]
X_eval = X_eval[SELECTED_FEATURES]

print(f"Train X: {X.shape}, y: {y.shape}")
print(f"Eval  X: {X_eval.shape}")


# ============================================================
# 2. Train / validation split
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"\nTrain split: {len(X_train)} samples")
print(f"Val split:   {len(X_val)} samples")


# ============================================================
# 3. Regime discovery (GMM on TRAIN TARGET ONLY)
# ============================================================
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(y_train.reshape(-1, 1))

r_train = gmm.predict(y_train.reshape(-1, 1))
r_val   = gmm.predict(y_val.reshape(-1, 1))

means = gmm.means_.ravel()
order = np.argsort(means)

r_train = np.array([np.where(order == r)[0][0] for r in r_train])
r_val   = np.array([np.where(order == r)[0][0] for r in r_val])

print("\nGMM regime means (train only):", means[order])
print("Train regime proportions:", np.bincount(r_train) / len(r_train))
print("Val   regime proportions:", np.bincount(r_val) / len(r_val))


# ============================================================
# 4. Regime classifier (X → regime)
# ============================================================
clf = CatBoostClassifier(
    iterations=600,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    random_seed=42,
    verbose=False
)

clf.fit(X_train, r_train)

r_val_pred = clf.predict(X_val).astype(int)
regime_acc = (r_val_pred == r_val).mean()

print(f"\nPseudo-regime agreement (VAL, diagnostic only): {regime_acc:.4f}")


# ============================================================
# 5. Regressors per regime
# ============================================================
regressors = {}

for reg in [0, 1]:
    idx = r_train == reg

    model = CatBoostRegressor(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )

    model.fit(X_train[idx], y_train[idx])
    regressors[reg] = model


# ============================================================
# 6. VECTORISED Mixture-of-Experts prediction
# ============================================================
# --- TRAIN ---
train_proba = clf.predict_proba(X_train)

train_pred_0 = regressors[0].predict(X_train)
train_pred_1 = regressors[1].predict(X_train)

y_train_pred = (
    train_proba[:, 0] * train_pred_0
    + train_proba[:, 1] * train_pred_1
)

# --- VALIDATION ---
val_proba = clf.predict_proba(X_val)

val_pred_0 = regressors[0].predict(X_val)
val_pred_1 = regressors[1].predict(X_val)

y_val_pred = (
    val_proba[:, 0] * val_pred_0
    + val_proba[:, 1] * val_pred_1
)


# ============================================================
# 7. Metrics
# ============================================================
print("\n" + "="*50)
print("MODEL PERFORMANCE METRICS")
print("="*50)

print(f"Train R²:   {r2_score(y_train, y_train_pred):.4f}")
print(f"Train RMSE: {root_mean_squared_error(y_train, y_train_pred):.4f}")
print(f"Val R²:     {r2_score(y_val, y_val_pred):.4f}")
print(f"Val RMSE:   {root_mean_squared_error(y_val, y_val_pred):.4f}")

print("="*50)


# ============================================================
# 7a. ERROR ANALYSIS & VISUALIZATION
# ============================================================
print("\n" + "="*60)
print("ERROR ANALYSIS")
print("="*60)

# Calculate errors
val_errors = np.abs(y_val - y_val_pred)
val_squared_errors = (y_val - y_val_pred) ** 2

# Error statistics
print(f"\nValidation Error Statistics:")
print(f"  Mean Absolute Error:  {val_errors.mean():.4f}")
print(f"  Median Absolute Error: {np.median(val_errors):.4f}")
print(f"  Max Error:            {val_errors.max():.4f}")
print(f"  95th percentile:      {np.percentile(val_errors, 95):.4f}")

# Identify high-error samples (top 10%)
error_threshold = np.percentile(val_errors, 90)
high_error_mask = val_errors > error_threshold

print(f"\nHigh-error samples (top 10%, error > {error_threshold:.4f}): {high_error_mask.sum()}")

# Analyze by regime
print("\n--- Error by Regime ---")
for reg in [0, 1]:
    regime_mask = r_val == reg
    if regime_mask.sum() > 0:
        regime_errors = val_errors[regime_mask]
        regime_r2 = r2_score(y_val[regime_mask], y_val_pred[regime_mask])
        print(f"Regime {reg}:")
        print(f"  Samples:     {regime_mask.sum()}")
        print(f"  Mean error:  {regime_errors.mean():.4f}")
        print(f"  Std error:   {regime_errors.std():.4f}")
        print(f"  R²:          {regime_r2:.4f}")
        print(f"  High errors: {(val_errors[regime_mask] > error_threshold).sum()}")

# Analyze by target value range
print("\n--- Error by Target Value Range ---")
target_bins = [y_val.min(), np.percentile(y_val, 25), np.percentile(y_val, 50), 
               np.percentile(y_val, 75), y_val.max()]
bin_labels = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']

for i in range(len(target_bins) - 1):
    mask = (y_val >= target_bins[i]) & (y_val < target_bins[i+1])
    if i == len(target_bins) - 2:  # Include max in last bin
        mask = (y_val >= target_bins[i]) & (y_val <= target_bins[i+1])
    
    if mask.sum() > 0:
        range_errors = val_errors[mask]
        print(f"{bin_labels[i]} [{target_bins[i]:.2f}, {target_bins[i+1]:.2f}]:")
        print(f"  Samples:     {mask.sum()}")
        print(f"  Mean error:  {range_errors.mean():.4f}")
        print(f"  RMSE:        {np.sqrt((range_errors**2).mean()):.4f}")
        print(f"  High errors: {(range_errors > error_threshold).sum()}")

# Analyze misclassified regimes
misclassified = r_val_pred != r_val
if misclassified.sum() > 0:
    print(f"\n--- Regime Misclassification Impact ---")
    print(f"Misclassified samples: {misclassified.sum()} ({100*misclassified.mean():.1f}%)")
    print(f"Mean error (correct regime):     {val_errors[~misclassified].mean():.4f}")
    print(f"Mean error (misclassified regime): {val_errors[misclassified].mean():.4f}")

# Check low-confidence predictions
low_confidence = val_proba.max(axis=1) < 0.6
if low_confidence.sum() > 0:
    print(f"\n--- Low Confidence Predictions (<0.6) ---")
    print(f"Low confidence samples: {low_confidence.sum()} ({100*low_confidence.mean():.1f}%)")
    print(f"Mean error (high conf): {val_errors[~low_confidence].mean():.4f}")
    print(f"Mean error (low conf):  {val_errors[low_confidence].mean():.4f}")

# ============================================================
# 7b. VISUALIZATIONS
# ============================================================
print("\n" + "="*60)
print("CREATING VISUALIZATIONS...")
print("="*60)

# Calculate GMM component PDFs for visualization
y_range = np.linspace(y_val.min(), y_val.max(), 1000)
gmm_pdf = np.zeros_like(y_range)

# Get GMM parameters from training
means_plot = gmm.means_.ravel()[order]
covs_plot = gmm.covariances_.ravel()[order]
weights_plot = gmm.weights_[order]

# Calculate individual component PDFs and total PDF
component_pdfs = []
for i in range(len(means_plot)):
    component_pdf = weights_plot[i] * (1 / np.sqrt(2 * np.pi * covs_plot[i])) * \
                    np.exp(-0.5 * ((y_range - means_plot[i]) ** 2) / covs_plot[i])
    component_pdfs.append(component_pdf)
    gmm_pdf += component_pdf

# Find overlap region (where both components have significant density)
overlap_threshold = 0.1 * np.max([pdf for pdf in component_pdfs])
overlap_mask = np.all([pdf > overlap_threshold for pdf in component_pdfs], axis=0)

# Classify validation samples by their position
val_sample_in_overlap = []
for y_sample in y_val:
    densities = []
    for i in range(len(means_plot)):
        density = weights_plot[i] * (1 / np.sqrt(2 * np.pi * covs_plot[i])) * \
                  np.exp(-0.5 * ((y_sample - means_plot[i]) ** 2) / covs_plot[i])
        densities.append(density)
    val_sample_in_overlap.append(all(d > overlap_threshold for d in densities))

val_sample_in_overlap = np.array(val_sample_in_overlap)

# Analyze errors in overlap vs non-overlap regions
print(f"\n--- Error Analysis by Distribution Region ---")
print(f"Samples in overlap region: {val_sample_in_overlap.sum()} ({100*val_sample_in_overlap.mean():.1f}%)")
print(f"Mean error (overlap):      {val_errors[val_sample_in_overlap].mean():.4f}")
print(f"Mean error (non-overlap):  {val_errors[~val_sample_in_overlap].mean():.4f}")
print(f"High-error in overlap:     {high_error_mask[val_sample_in_overlap].sum()}/{val_sample_in_overlap.sum()}")
print(f"High-error in non-overlap: {high_error_mask[~val_sample_in_overlap].sum()}/{(~val_sample_in_overlap).sum()}")

fig = plt.figure(figsize=(20, 12))

# 1. TARGET DISTRIBUTION WITH HIGH-ERROR SAMPLES HIGHLIGHTED
ax1 = plt.subplot(2, 3, 1)

# Plot histogram of all validation samples
ax1.hist(y_val, bins=50, density=True, alpha=0.3, color='gray', 
         edgecolor='black', label='All val samples')

# Overlay GMM components
for i, (mean, cov, weight, pdf) in enumerate(zip(means_plot, covs_plot, weights_plot, component_pdfs)):
    ax1.plot(y_range, pdf, '--', linewidth=2, 
             label=f'Regime {i} (μ={mean:.2f}, σ²={cov:.2f})', alpha=0.8)

# Plot total GMM
ax1.plot(y_range, gmm_pdf, 'k-', linewidth=3, label='GMM Total', alpha=0.7)

# Highlight overlap region
if overlap_mask.any():
    ax1.fill_between(y_range, 0, gmm_pdf, where=overlap_mask, 
                     alpha=0.2, color='yellow', label='Overlap region')

# Highlight high-error samples
ax1.hist(y_val[high_error_mask], bins=30, density=True, alpha=0.6, 
         color='red', edgecolor='darkred', label=f'High-error samples (top 10%)')

ax1.set_xlabel('Target Value', fontsize=11, fontweight='bold')
ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
ax1.set_title('Target Distribution: High-Error Samples Location', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=8)
ax1.grid(True, alpha=0.3, axis='y')

# 2. SCATTER: Error magnitude by target value position
ax2 = plt.subplot(2, 3, 2)

# Plot all samples
scatter_normal = ax2.scatter(y_val[~high_error_mask], val_errors[~high_error_mask], 
                            c='lightblue', s=40, alpha=0.5, edgecolors='black', 
                            linewidth=0.5, label='Normal error')

# Highlight high-error samples
scatter_high = ax2.scatter(y_val[high_error_mask], val_errors[high_error_mask], 
                          c='red', s=80, alpha=0.8, edgecolors='darkred', 
                          linewidth=1, marker='X', label='High error')

# Mark overlap region
if overlap_mask.any():
    overlap_start = y_range[overlap_mask][0]
    overlap_end = y_range[overlap_mask][-1]
    ax2.axvspan(overlap_start, overlap_end, alpha=0.2, color='yellow', 
                label=f'Overlap region [{overlap_start:.2f}, {overlap_end:.2f}]')

ax2.axhline(y=error_threshold, color='orange', linestyle='--', linewidth=2, 
            label=f'90th percentile ({error_threshold:.3f})')

ax2.set_xlabel('Target Value', fontsize=11, fontweight='bold')
ax2.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
ax2.set_title('Error vs Target Value (High-Error Samples Marked)', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=8)
ax2.grid(True, alpha=0.3)

ax2.grid(True, alpha=0.3)

# 3. Actual vs Predicted with high-error highlighted
ax3 = plt.subplot(2, 3, 3)
scatter_normal_pred = ax3.scatter(y_val[~high_error_mask], y_val_pred[~high_error_mask], 
                                  c='lightblue', s=40, alpha=0.5, edgecolors='black', 
                                  linewidth=0.5, label='Normal error')
scatter_high_pred = ax3.scatter(y_val[high_error_mask], y_val_pred[high_error_mask], 
                                c='red', s=80, alpha=0.8, edgecolors='darkred', 
                                linewidth=1, marker='X', label='High error')
ax3.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 
         'k--', lw=2, label='Perfect prediction')
ax3.set_xlabel('Actual Target', fontsize=11, fontweight='bold')
ax3.set_ylabel('Predicted Target', fontsize=11, fontweight='bold')
ax3.set_title(f'Actual vs Predicted (Val R²={r2_score(y_val, y_val_pred):.3f})', 
              fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. Error distribution by overlap region
ax4 = plt.subplot(2, 3, 4)
error_by_region = [val_errors[~val_sample_in_overlap], val_errors[val_sample_in_overlap]]
labels_region = [f'Non-overlap\n(n={(~val_sample_in_overlap).sum()})', 
                 f'Overlap\n(n={val_sample_in_overlap.sum()})']
bp = ax4.boxplot(error_by_region, labels=labels_region,
                 patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], ['lightgreen', 'lightyellow']):
    patch.set_facecolor(color)
ax4.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
ax4.set_title('Error Distribution: Overlap vs Non-Overlap Region', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.grid(True, alpha=0.3, axis='y')

# 5. Regime distribution with errors
ax5 = plt.subplot(2, 3, 5)
error_by_regime = [val_errors[r_val == 0], val_errors[r_val == 1]]
bp = ax5.boxplot(error_by_regime, labels=['Regime 0\n(Low)', 'Regime 1\n(High)'],
                 patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
    patch.set_facecolor(color)
ax5.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
ax5.set_title('Error Distribution by Regime', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# 6. Sample classification: overlap + high error breakdown
ax6 = plt.subplot(2, 3, 6)

categories = [
    'Non-overlap\nLow error',
    'Non-overlap\nHigh error',
    'Overlap\nLow error',
    'Overlap\nHigh error'
]

counts = [
    (~val_sample_in_overlap & ~high_error_mask).sum(),
    (~val_sample_in_overlap & high_error_mask).sum(),
    (val_sample_in_overlap & ~high_error_mask).sum(),
    (val_sample_in_overlap & high_error_mask).sum()
]

colors_bar = ['lightgreen', 'orange', 'lightyellow', 'darkred']
bars = ax6.bar(range(len(categories)), counts, color=colors_bar, edgecolor='black', linewidth=1.5)

# Add count labels on bars
for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({100*count/len(y_val):.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

ax6.set_xticks(range(len(categories)))
ax6.set_xticklabels(categories, fontsize=9)
ax6.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
ax6.set_title('Sample Breakdown: Region × Error Level', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_filename = f'error_analysis_{PROBLEM_NUM}.png'
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: {plot_filename}")
plt.close()

# Additional: High-error sample analysis
high_error_df = pd.DataFrame({
    'actual': y_val[high_error_mask],
    'predicted': y_val_pred[high_error_mask],
    'error': val_errors[high_error_mask],
    'regime_actual': r_val[high_error_mask],
    'regime_pred': r_val_pred[high_error_mask],
    'confidence': val_proba[high_error_mask].max(axis=1)
})

high_error_file = f'high_error_samples_{PROBLEM_NUM}.csv'
high_error_df.to_csv(high_error_file, index=False)
print(f"✓ High-error samples saved: {high_error_file}")

print(f"\nTop 5 highest errors:")
print(high_error_df.nlargest(5, 'error').to_string(index=False))

print("="*60)


# ============================================================
# 8. Train FINAL models on FULL data
# ============================================================
gmm_full = GaussianMixture(n_components=2, random_state=42)
gmm_full.fit(y.reshape(-1, 1))

regime_full = gmm_full.predict(y.reshape(-1, 1))
means_full = gmm_full.means_.ravel()
order_full = np.argsort(means_full)

regime_full = np.array([np.where(order_full == r)[0][0] for r in regime_full])

clf.fit(X, regime_full)

for reg in [0, 1]:
    regressors[reg].fit(X[regime_full == reg], y[regime_full == reg])


# ============================================================
# 9. EVAL prediction (NO LABELS)
# ============================================================
eval_proba = clf.predict_proba(X_eval)

eval_pred_0 = regressors[0].predict(X_eval)
eval_pred_1 = regressors[1].predict(X_eval)

y_eval_pred = (
    eval_proba[:, 0] * eval_pred_0
    + eval_proba[:, 1] * eval_pred_1
)

print("\nEval regime distribution:", eval_proba.mean(axis=0))
print("Low-confidence samples (<0.6):", (eval_proba.max(axis=1) < 0.6).mean())


# ============================================================
# 10. SAVE OUTPUT  ✅ FIXED FILENAME
# ============================================================
output_file = f"EVAL_target01_{PROBLEM_NUM}.csv"

pd.DataFrame({"target01": y_eval_pred}).to_csv(output_file, index=False)

print(f"\nPredictions saved to: {output_file}")
print("Sample predictions:", y_eval_pred[:10])
