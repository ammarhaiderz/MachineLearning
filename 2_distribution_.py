# ============================================================
# 0. Imports
# ============================================================
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

from catboost import CatBoostClassifier, CatBoostRegressor


# ============================================================
# 1. Load data
# ============================================================
PROBLEM_NUM = 36

X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"].values
X_eval = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv")

print(f"Train X: {X.shape}, y: {y.shape}")
print(f"Eval  X: {X_eval.shape}")


# ============================================================
# 2. Train / validation split (BEFORE any target analysis)
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"\nTrain split: {len(X_train)} samples")
print(f"Val split:   {len(X_val)} samples")


# ============================================================
# 3. DATA-DRIVEN REGIME DISCOVERY (GMM ON TRAINING TARGET ONLY)
# ============================================================
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(y_train.reshape(-1, 1))

# Predict regimes separately for train and val
r_train = gmm.predict(y_train.reshape(-1, 1))
r_val = gmm.predict(y_val.reshape(-1, 1))

r_train_proba = gmm.predict_proba(y_train.reshape(-1, 1))
r_val_proba = gmm.predict_proba(y_val.reshape(-1, 1))

# Ensure consistent ordering: regime 0 = narrow / left mode
means = gmm.means_.ravel()
order = np.argsort(means)
r_train = np.array([np.where(order == r)[0][0] for r in r_train])
r_val = np.array([np.where(order == r)[0][0] for r in r_val])
r_train_proba = r_train_proba[:, order]
r_val_proba = r_val_proba[:, order]

print("\nGMM regime means (from training):", means[order])
print("Train regime proportions:", np.bincount(r_train) / len(r_train))
print("Val   regime proportions:", np.bincount(r_val) / len(r_val))


# ============================================================
# 4. STAGE 1: REGIME CLASSIFIER (X -> regime)
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
r_val_proba = clf.predict_proba(X_val)

regime_acc = (r_val_pred == r_val).mean()
print(f"\nRegime classification accuracy (VAL): {regime_acc:.4f}")


# ============================================================
# 5. STAGE 2: REGRESSORS PER REGIME
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
# 6. SOFT MIXTURE-OF-EXPERTS VALIDATION
# ============================================================
# Training predictions
y_train_pred = np.zeros(len(X_train))
clf_train_proba = clf.predict_proba(X_train)

for i in range(len(X_train)):
    probs = clf_train_proba[i]
    y_train_pred[i] = (
        probs[0] * regressors[0].predict(X_train.iloc[[i]])[0]
        + probs[1] * regressors[1].predict(X_train.iloc[[i]])[0]
    )

# Validation predictions
y_val_pred = np.zeros(len(X_val))
clf_val_proba = clf.predict_proba(X_val)

for i in range(len(X_val)):
    probs = clf_val_proba[i]
    y_val_pred[i] = (
        probs[0] * regressors[0].predict(X_val.iloc[[i]])[0]
        + probs[1] * regressors[1].predict(X_val.iloc[[i]])[0]
    )

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = root_mean_squared_error(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)
val_rmse = root_mean_squared_error(y_val, y_val_pred)

print(f"\n{'='*50}")
print(f"MODEL PERFORMANCE METRICS")
print(f"{'='*50}")
print(f"Train R²:   {train_r2:.4f}")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Val R²:     {val_r2:.4f}")
print(f"Val RMSE:   {val_rmse:.4f}")
print(f"{'='*50}")


# ============================================================
# 7. TRAIN FINAL MODELS ON FULL DATA
# ============================================================
# Refit GMM on full target data for final model
gmm_full = GaussianMixture(n_components=2, random_state=42)
gmm_full.fit(y.reshape(-1, 1))
regime_full = gmm_full.predict(y.reshape(-1, 1))

# Ensure consistent ordering
means_full = gmm_full.means_.ravel()
order_full = np.argsort(means_full)
regime_full = np.array([np.where(order_full == r)[0][0] for r in regime_full])

# Train classifier on full data
clf.fit(X, regime_full)

# Train regressors on full data
for reg in [0, 1]:
    regressors[reg].fit(X[regime_full == reg], y[regime_full == reg])


# ============================================================
# 8. EVALUATION DATA PREDICTION (NO LABELS)
# ============================================================
eval_regime_proba = clf.predict_proba(X_eval)

print("\nEval regime distribution:",
      np.mean(eval_regime_proba, axis=0))

# Confidence diagnostics
confidence = np.max(eval_regime_proba, axis=1)
print(f"Low-confidence samples (<0.6): {(confidence < 0.6).mean():.2%}")

y_eval_pred = np.zeros(len(X_eval))

for i in range(len(X_eval)):
    probs = eval_regime_proba[i]
    y_eval_pred[i] = (
        probs[0] * regressors[0].predict(X_eval.iloc[[i]])[0]
        + probs[1] * regressors[1].predict(X_eval.iloc[[i]])[0]
    )


# ============================================================
# 9. SAVE OUTPUT
# ============================================================
output_file = f"EVAL_target01_{PROBLEM_NUM}_GMM_mixture_catboost.csv"
pd.DataFrame({"target01": y_eval_pred}).to_csv(output_file, index=False)

print(f"\nPredictions saved to: {output_file}")
print("Sample predictions:", y_eval_pred[:10])


# ============================================================
# 10. PLOT PREDICTED VS ACTUAL TARGET DISTRIBUTION
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Histograms
axes[0, 0].hist(y_val, bins=50, alpha=0.6, label='Actual', color='blue', density=True)
axes[0, 0].hist(y_val_pred, bins=50, alpha=0.6, label='Predicted', color='red', density=True)
axes[0, 0].set_xlabel('Target Value')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Validation: Predicted vs Actual Distribution (Histogram)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: KDE (Kernel Density Estimation)
from scipy.stats import gaussian_kde
kde_actual = gaussian_kde(y_val)
kde_pred = gaussian_kde(y_val_pred)
x_range = np.linspace(min(y_val.min(), y_val_pred.min()), 
                      max(y_val.max(), y_val_pred.max()), 500)
axes[0, 1].plot(x_range, kde_actual(x_range), label='Actual', color='blue', linewidth=2)
axes[0, 1].plot(x_range, kde_pred(x_range), label='Predicted', color='red', linewidth=2)
axes[0, 1].fill_between(x_range, kde_actual(x_range), alpha=0.3, color='blue')
axes[0, 1].fill_between(x_range, kde_pred(x_range), alpha=0.3, color='red')
axes[0, 1].set_xlabel('Target Value')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Validation: Predicted vs Actual Distribution (KDE)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Q-Q Plot
from scipy import stats
stats.probplot(y_val_pred - y_val, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot: Residuals (Pred - Actual)')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Scatter plot (Predicted vs Actual)
axes[1, 1].scatter(y_val, y_val_pred, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
axes[1, 1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[1, 1].set_xlabel('Actual Target')
axes[1, 1].set_ylabel('Predicted Target')
axes[1, 1].set_title(f'Predicted vs Actual (R² = {val_r2:.4f})')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'distribution_comparison_{PROBLEM_NUM}.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nDistribution plot saved to: distribution_comparison_{PROBLEM_NUM}.png")


# ============================================================
# 11. EVAL PREDICTIONS DISTRIBUTION ANALYSIS
# ============================================================
from scipy import stats as scipy_stats

print("\n" + "="*60)
print("EVAL TARGET PREDICTIONS - DISTRIBUTION STATISTICS")
print("="*60)
print(f"  Count:      {len(y_eval_pred)}")
print(f"  Mean:       {y_eval_pred.mean():.6f}")
print(f"  Median:     {np.median(y_eval_pred):.6f}")
print(f"  Std:        {y_eval_pred.std():.6f}")
print(f"  Min:        {y_eval_pred.min():.6f}")
print(f"  Max:        {y_eval_pred.max():.6f}")
print(f"  Range:      {y_eval_pred.max() - y_eval_pred.min():.6f}")
print(f"  Q1 (25%):   {np.percentile(y_eval_pred, 25):.6f}")
print(f"  Q2 (50%):   {np.percentile(y_eval_pred, 50):.6f}")
print(f"  Q3 (75%):   {np.percentile(y_eval_pred, 75):.6f}")
print(f"  IQR:        {np.percentile(y_eval_pred, 75) - np.percentile(y_eval_pred, 25):.6f}")
print(f"  Skewness:   {scipy_stats.skew(y_eval_pred):.6f}")
print(f"  Kurtosis:   {scipy_stats.kurtosis(y_eval_pred):.6f}")
print("="*60)

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Histogram
axes[0].hist(y_eval_pred, bins=50, alpha=0.7, color='purple', edgecolor='black')
axes[0].axvline(y_eval_pred.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {y_eval_pred.mean():.4f}')
axes[0].axvline(np.median(y_eval_pred), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(y_eval_pred):.4f}')
axes[0].set_xlabel('Target Value (Predicted)', fontweight='bold')
axes[0].set_ylabel('Frequency', fontweight='bold')
axes[0].set_title('EVAL Target Predictions - Histogram', fontweight='bold', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: KDE (Kernel Density Estimation)
kde_eval = gaussian_kde(y_eval_pred)
x_range = np.linspace(y_eval_pred.min(), y_eval_pred.max(), 500)
axes[1].plot(x_range, kde_eval(x_range), color='purple', linewidth=2)
axes[1].fill_between(x_range, kde_eval(x_range), alpha=0.4, color='purple')
axes[1].axvline(y_eval_pred.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {y_eval_pred.mean():.4f}')
axes[1].set_xlabel('Target Value (Predicted)', fontweight='bold')
axes[1].set_ylabel('Density', fontweight='bold')
axes[1].set_title('EVAL Target Predictions - KDE', fontweight='bold', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Box Plot
bp = axes[2].boxplot(y_eval_pred, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='plum', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
axes[2].set_ylabel('Target Value (Predicted)', fontweight='bold')
axes[2].set_title('EVAL Target Predictions - Box Plot', fontweight='bold', fontsize=14)
axes[2].grid(True, alpha=0.3, axis='y')
axes[2].set_xticklabels(['EVAL Predictions'])

# Add statistics to box plot
q1, q3 = np.percentile(y_eval_pred, 25), np.percentile(y_eval_pred, 75)
iqr = q3 - q1
outliers = y_eval_pred[(y_eval_pred < q1 - 1.5*iqr) | (y_eval_pred > q3 + 1.5*iqr)]
axes[2].text(0.5, 0.02, f'Outliers: {len(outliers)} ({len(outliers)/len(y_eval_pred)*100:.1f}%)',
            transform=axes[2].transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('EVAL Target Predictions Distribution Analysis', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.savefig(f'EVAL_predictions_distribution_{PROBLEM_NUM}.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nEVAL distribution plot saved to: EVAL_predictions_distribution_{PROBLEM_NUM}.png")
