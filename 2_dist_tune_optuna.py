# ============================================================
# Optuna hyperparameter tuning for MoE
# ============================================================
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostClassifier, CatBoostRegressor
import optuna

PROBLEM_NUM = 36

SELECTED_FEATURES = [ 
    'feat_155', 'feat_184', 'feat_64', 'feat_232', 'feat_253', 
    'feat_143', 'feat_221', 'feat_220', 'feat_160', 'feat_266', 
    'feat_138', 'feat_47', 'feat_203',
]

X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"].values

X = X[SELECTED_FEATURES]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")


def objective(trial):
    """Optuna objective function"""
    
    # Suggest hyperparameters
    n_components = trial.suggest_int('n_components', 2, 4)
    
    clf_iterations = trial.suggest_int('clf_iterations', 300, 1000, step=100)
    clf_depth = trial.suggest_int('clf_depth', 3, 8)
    clf_lr = trial.suggest_float('clf_lr', 0.01, 0.1, log=True)
    clf_l2 = trial.suggest_float('clf_l2', 1, 10)
    
    reg_iterations = trial.suggest_int('reg_iterations', 400, 1200, step=100)
    reg_depth = trial.suggest_int('reg_depth', 3, 8)
    reg_lr = trial.suggest_float('reg_lr', 0.01, 0.1, log=True)
    reg_l2 = trial.suggest_float('reg_l2', 1, 10)
    
    try:
        # 1. GMM regime discovery
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(y_train.reshape(-1, 1))
        
        r_train = gmm.predict(y_train.reshape(-1, 1))
        means = gmm.means_.ravel()
        order = np.argsort(means)
        r_train = np.array([np.where(order == r)[0][0] for r in r_train])
        
        # 2. Classifier
        clf = CatBoostClassifier(
            iterations=clf_iterations,
            depth=clf_depth,
            learning_rate=clf_lr,
            l2_leaf_reg=clf_l2,
            loss_function="Logloss",
            random_seed=42,
            verbose=False
        )
        clf.fit(X_train, r_train)
        
        # 3. Regressors per regime
        regressors = {}
        for reg in range(n_components):
            idx = r_train == reg
            
            # Skip if too few samples in regime
            if idx.sum() < 20:
                continue
            
            model = CatBoostRegressor(
                iterations=reg_iterations,
                depth=reg_depth,
                learning_rate=reg_lr,
                l2_leaf_reg=reg_l2,
                loss_function="RMSE",
                random_seed=42,
                verbose=False
            )
            model.fit(X_train[idx], y_train[idx])
            regressors[reg] = model
        
        # 4. Validation prediction
        val_proba = clf.predict_proba(X_val)
        y_val_pred = np.zeros(len(y_val))
        
        for reg in regressors:
            y_val_pred += val_proba[:, reg] * regressors[reg].predict(X_val)
        
        val_r2 = r2_score(y_val, y_val_pred)
        
        return val_r2
    
    except Exception as e:
        print(f"Trial failed: {e}")
        return -999  # Return bad score on failure


# ============================================================
# Run Optuna optimization
# ============================================================
study = optuna.create_study(
    direction='maximize',
    study_name='MoE_tuning',
    sampler=optuna.samplers.TPESampler(seed=42)
)

print("\nStarting Optuna optimization...")
print("This will run 50 trials (should take 10-20 minutes)\n")

study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n" + "="*60)
print("OPTIMIZATION COMPLETE")
print("="*60)
print(f"Best Val R²: {study.best_value:.4f}")
print("\nBest Parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Save results
results_df = study.trials_dataframe()
results_df.to_csv(f"optuna_tuning_{PROBLEM_NUM}.csv", index=False)
print(f"\nAll trials saved to: optuna_tuning_{PROBLEM_NUM}.csv")

# Visualizations (if optuna-dashboard installed)
try:
    import optuna.visualization as vis
    
    fig1 = vis.plot_optimization_history(study)
    fig1.write_html(f"optuna_history_{PROBLEM_NUM}.html")
    
    fig2 = vis.plot_param_importances(study)
    fig2.write_html(f"optuna_importance_{PROBLEM_NUM}.html")
    
    print("\nVisualizations saved:")
    print(f"  - optuna_history_{PROBLEM_NUM}.html")
    print(f"  - optuna_importance_{PROBLEM_NUM}.html")
except:
    print("\n(Install plotly for visualizations: pip install plotly)")
