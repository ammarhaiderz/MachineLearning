# ============================================================
# Cross-validation tuning for MoE (most robust)
# ============================================================
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
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

print(f"Full dataset: {len(X)} samples")


def train_and_evaluate_fold(X_train, y_train, X_val, y_val, params):
    """Train MoE on one fold and return validation score"""
    
    # GMM
    gmm = GaussianMixture(n_components=params['n_components'], random_state=42)
    gmm.fit(y_train.reshape(-1, 1))
    
    r_train = gmm.predict(y_train.reshape(-1, 1))
    means = gmm.means_.ravel()
    order = np.argsort(means)
    r_train = np.array([np.where(order == r)[0][0] for r in r_train])
    
    # Classifier
    clf = CatBoostClassifier(
        iterations=params['clf_iterations'],
        depth=params['clf_depth'],
        learning_rate=params['clf_lr'],
        l2_leaf_reg=params['clf_l2'],
        loss_function="Logloss",
        random_seed=42,
        verbose=False
    )
    clf.fit(X_train, r_train)
    
    # Regressors
    regressors = {}
    for reg in range(params['n_components']):
        idx = r_train == reg
        if idx.sum() < 20:
            continue
        
        model = CatBoostRegressor(
            iterations=params['reg_iterations'],
            depth=params['reg_depth'],
            learning_rate=params['reg_lr'],
            l2_leaf_reg=params['reg_l2'],
            loss_function="RMSE",
            random_seed=42,
            verbose=False
        )
        model.fit(X_train[idx], y_train[idx])
        regressors[reg] = model
    
    # Predict
    val_proba = clf.predict_proba(X_val)
    y_val_pred = np.zeros(len(y_val))
    
    for reg in regressors:
        y_val_pred += val_proba[:, reg] * regressors[reg].predict(X_val)
    
    return r2_score(y_val, y_val_pred)


def objective_cv(trial):
    """Optuna objective with 5-fold CV"""
    
    params = {
        'n_components': trial.suggest_int('n_components', 2, 3),
        'clf_iterations': trial.suggest_int('clf_iterations', 400, 800, step=100),
        'clf_depth': trial.suggest_int('clf_depth', 4, 7),
        'clf_lr': trial.suggest_float('clf_lr', 0.02, 0.08, log=True),
        'clf_l2': trial.suggest_float('clf_l2', 1, 8),
        'reg_iterations': trial.suggest_int('reg_iterations', 600, 1000, step=100),
        'reg_depth': trial.suggest_int('reg_depth', 4, 7),
        'reg_lr': trial.suggest_float('reg_lr', 0.02, 0.08, log=True),
        'reg_l2': trial.suggest_float('reg_l2', 1, 8),
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y[val_idx]
        
        try:
            score = train_and_evaluate_fold(
                X_train_fold, y_train_fold, 
                X_val_fold, y_val_fold, 
                params
            )
            scores.append(score)
        except Exception as e:
            print(f"Fold {fold} failed: {e}")
            return -999
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    # Penalize high variance
    final_score = mean_score - 0.1 * std_score
    
    print(f"Trial {trial.number}: CV R² = {mean_score:.4f} ± {std_score:.4f} (score: {final_score:.4f})")
    
    return final_score


# ============================================================
# Run optimization
# ============================================================
study = optuna.create_study(
    direction='maximize',
    study_name='MoE_CV_tuning',
    sampler=optuna.samplers.TPESampler(seed=42)
)

print("\nStarting 5-fold CV optimization...")
print("WARNING: This will take 30-60 minutes!\n")

study.optimize(objective_cv, n_trials=30, show_progress_bar=True)

print("\n" + "="*60)
print("CV OPTIMIZATION COMPLETE")
print("="*60)
print(f"Best CV Score: {study.best_value:.4f}")
print("\nBest Parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

results_df = study.trials_dataframe()
results_df.to_csv(f"optuna_cv_tuning_{PROBLEM_NUM}.csv", index=False)
print(f"\nResults saved to: optuna_cv_tuning_{PROBLEM_NUM}.csv")
