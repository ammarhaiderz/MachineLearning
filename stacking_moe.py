# ===============================
# Mixture of Experts using Stacking Regressor
# Using ALL features
# ===============================
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import time

# Configuration
PROBLEM_NUM = 36

print("=" * 80)
print("MIXTURE OF EXPERTS: STACKING REGRESSOR APPROACH")
print("=" * 80)

# ===============================
# 1. Load Data
# ===============================
print("\n1. Loading data...")
X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"].values
X_eval = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/EVAL_{PROBLEM_NUM}.csv")

print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print(f"   X_eval shape: {X_eval.shape}")
print(f"   Using ALL {X.shape[1]} features")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"   Train: {X_train.shape}, Val: {X_val.shape}")


# ===============================
# 2. Define Base Models (Experts)
# ===============================
print("\n2. Defining expert models...")

# Note: Using LightGBM and XGBoost only for sklearn compatibility
# CatBoost has compatibility issues with newer sklearn versions
experts = [
    ('lightgbm_deep', LGBMRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )),
    ('lightgbm_shallow', LGBMRegressor(
        n_estimators=1200,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=15,
        random_state=43,
        verbose=-1
    )),
    ('xgboost_1', XGBRegressor(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.03,
        random_state=42,
        verbosity=0
    )),
    ('xgboost_2', XGBRegressor(
        n_estimators=1200,
        max_depth=5,
        learning_rate=0.04,
        random_state=43,
        verbosity=0
    ))
]

print(f"   Number of experts: {len(experts)}")
for name, _ in experts:
    print(f"   - {name}")


# ===============================
# 3. Train Individual Experts (Baseline)
# ===============================
print("\n3. Training individual experts (baseline comparison)...")

expert_results = {}
for name, model in experts:
    print(f"\n   Training {name}...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    elapsed = time.time() - start_time
    
    expert_results[name] = {
        'train_r2': train_r2,
        'val_r2': val_r2,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'time': elapsed
    }
    
    print(f"      Train R²: {train_r2:.6f}, Val R²: {val_r2:.6f}")
    print(f"      Train RMSE: {train_rmse:.6f}, Val RMSE: {val_rmse:.6f}")
    print(f"      Time: {elapsed:.2f}s")


# ===============================
# 4. Build Mixture of Experts (Stacking)
# ===============================
print("\n4. Building Mixture of Experts (Stacking Regressor)...")

# Try different meta-learners (gating networks)
meta_learners = {
    'Ridge': RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5),
    'Lasso': LassoCV(cv=5, random_state=42, max_iter=5000),
    'ElasticNet': ElasticNetCV(cv=5, random_state=42, max_iter=5000)
}

stacking_results = {}

for meta_name, meta_learner in meta_learners.items():
    print(f"\n   Stacking with {meta_name} meta-learner...")
    start_time = time.time()
    
    # Recreate experts for stacking (fresh instances)
    # Note: Using only sklearn-compatible models (LightGBM, XGBoost)
    stacking_experts = [
        ('lightgbm_deep', LGBMRegressor(
            n_estimators=1000, max_depth=8, learning_rate=0.03,
            num_leaves=31, random_state=42, verbose=-1
        )),
        ('lightgbm_shallow', LGBMRegressor(
            n_estimators=1200, max_depth=4, learning_rate=0.05,
            num_leaves=15, random_state=43, verbose=-1
        )),
        ('xgboost_deep', XGBRegressor(
            n_estimators=1000, max_depth=7, learning_rate=0.03,
            random_state=42, verbosity=0
        )),
        ('xgboost_shallow', XGBRegressor(
            n_estimators=1200, max_depth=5, learning_rate=0.04,
            random_state=43, verbosity=0
        ))
    ]
    
    stacking_model = StackingRegressor(
        estimators=stacking_experts,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=False  # Only use expert predictions, not original features
    )
    
    stacking_model.fit(X_train, y_train)
    
    train_pred = stacking_model.predict(X_train)
    val_pred = stacking_model.predict(X_val)
    
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    elapsed = time.time() - start_time
    
    stacking_results[meta_name] = {
        'model': stacking_model,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'time': elapsed
    }
    
    print(f"      Train R²: {train_r2:.6f}, Val R²: {val_r2:.6f}")
    print(f"      Train RMSE: {train_rmse:.6f}, Val RMSE: {val_rmse:.6f}")
    print(f"      Time: {elapsed:.2f}s")
    
    # Show meta-learner weights if available
    if hasattr(stacking_model.final_estimator_, 'coef_'):
        weights = stacking_model.final_estimator_.coef_
        print(f"      Expert weights: {weights}")


# ===============================
# 5. Results Comparison
# ===============================
print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)

print("\nIndividual Experts:")
print(f"{'Model':<20} {'Train R²':<12} {'Val R²':<12} {'Val RMSE':<12} {'Time (s)':<10}")
print("-" * 80)
for name, results in expert_results.items():
    print(f"{name:<20} {results['train_r2']:<12.6f} {results['val_r2']:<12.6f} "
          f"{results['val_rmse']:<12.6f} {results['time']:<10.2f}")

print("\nMixture of Experts (Stacking):")
print(f"{'Meta-Learner':<20} {'Train R²':<12} {'Val R²':<12} {'Val RMSE':<12} {'Time (s)':<10}")
print("-" * 80)
for name, results in stacking_results.items():
    print(f"{name:<20} {results['train_r2']:<12.6f} {results['val_r2']:<12.6f} "
          f"{results['val_rmse']:<12.6f} {results['time']:<10.2f}")

# Find best model
best_expert_name = max(expert_results, key=lambda k: expert_results[k]['val_r2'])
best_expert_r2 = expert_results[best_expert_name]['val_r2']

best_stacking_name = max(stacking_results, key=lambda k: stacking_results[k]['val_r2'])
best_stacking_r2 = stacking_results[best_stacking_name]['val_r2']

print("\n" + "=" * 80)
print("BEST MODELS")
print("=" * 80)
print(f"Best Individual Expert: {best_expert_name} (Val R² = {best_expert_r2:.6f})")
print(f"Best Stacking MoE:      {best_stacking_name} (Val R² = {best_stacking_r2:.6f})")

if best_stacking_r2 > best_expert_r2:
    improvement = (best_stacking_r2 - best_expert_r2) / best_expert_r2 * 100
    print(f"\n✓ Stacking improves by {improvement:.2f}%")
else:
    print(f"\n✗ Stacking does not improve over best individual expert")


# ===============================
# 6. Train Final Model on Full Data
# ===============================
print("\n" + "=" * 80)
print("TRAINING FINAL MODEL ON FULL DATA")
print("=" * 80)

# Use best stacking configuration
best_meta = meta_learners[best_stacking_name]
print(f"\nUsing {best_stacking_name} meta-learner")

# Recreate experts
final_experts = [
    ('catboost_deep', CatBoostRegressor(
        iterations=800, depth=8, learning_rate=0.03,
        loss_function='RMSE', random_seed=42, verbose=False
    )),
    ('catboost_shallow', CatBoostRegressor(
        iterations=1000, depth=4, learning_rate=0.05,
        loss_function='RMSE', random_seed=43, verbose=False
    )),
    ('lightgbm', LGBMRegressor(
        n_estimators=800, max_depth=7, learning_rate=0.03,
        num_leaves=31, random_state=42, verbose=-1
    )),
    ('xgboost', XGBRegressor(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        random_state=42, verbosity=0
    ))
]

final_model = StackingRegressor(
    estimators=final_experts,
    final_estimator=best_meta,
    cv=5,
    n_jobs=-1,
    passthrough=False
)

print("Training on full dataset...")
start_time = time.time()
final_model.fit(X, y)
elapsed = time.time() - start_time
print(f"Training completed in {elapsed:.2f}s")

# Validation on full data
full_pred = final_model.predict(X)
full_r2 = r2_score(y, full_pred)
full_rmse = np.sqrt(mean_squared_error(y, full_pred))
print(f"\nFull data R²: {full_r2:.6f}")
print(f"Full data RMSE: {full_rmse:.6f}")


# ===============================
# 7. Predict on EVAL Set
# ===============================
print("\n" + "=" * 80)
print("PREDICTING ON EVAL SET")
print("=" * 80)

y_eval_pred = final_model.predict(X_eval)

print(f"\nPrediction statistics:")
print(f"   Mean: {y_eval_pred.mean():.6f}")
print(f"   Std:  {y_eval_pred.std():.6f}")
print(f"   Min:  {y_eval_pred.min():.6f}")
print(f"   Max:  {y_eval_pred.max():.6f}")

print(f"\nSample predictions (first 10):")
print(y_eval_pred[:10])


# ===============================
# 8. Save Predictions
# ===============================
output_file = f"EVAL_target01_{PROBLEM_NUM}_stacking_moe.csv"
pd.DataFrame({"target01": y_eval_pred}).to_csv(output_file, index=False)

print(f"\n✓ Predictions saved to: {output_file}")


# ===============================
# 9. Summary
# ===============================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nApproach: Mixture of Experts using Stacking Regressor")
print(f"Features: ALL {X.shape[1]} features")
print(f"Experts: {len(final_experts)}")
print(f"Meta-learner: {best_stacking_name}")
print(f"Final validation R²: {best_stacking_r2:.6f}")
print(f"Output file: {output_file}")
print("\n" + "=" * 80)
