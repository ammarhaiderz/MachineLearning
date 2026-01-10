# ============================================================
# Manual hyperparameter tuning for MoE
# ============================================================
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostClassifier, CatBoostRegressor

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

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ============================================================
# Grid to search
# ============================================================
param_grid = {
    'n_components': [2, 3],
    'clf_iterations': [400, 600, 800],
    'clf_depth': [4, 6],
    'reg_iterations': [600, 800, 1000],
    'reg_depth': [4, 6],
    'learning_rate': [0.03, 0.05, 0.07],
}

best_score = -np.inf
best_params = None
results = []

print("Starting grid search...")
print(f"Total combinations: {2 * 3 * 2 * 3 * 2 * 3} = {2*3*2*3*2*3}")

trial = 0
for n_comp in param_grid['n_components']:
    for clf_iter in param_grid['clf_iterations']:
        for clf_d in param_grid['clf_depth']:
            for reg_iter in param_grid['reg_iterations']:
                for reg_d in param_grid['reg_depth']:
                    for lr in param_grid['learning_rate']:
                        trial += 1
                        
                        # GMM
                        gmm = GaussianMixture(n_components=n_comp, random_state=42)
                        gmm.fit(y_train.reshape(-1, 1))
                        
                        r_train = gmm.predict(y_train.reshape(-1, 1))
                        means = gmm.means_.ravel()
                        order = np.argsort(means)
                        r_train = np.array([np.where(order == r)[0][0] for r in r_train])
                        
                        # Classifier
                        clf = CatBoostClassifier(
                            iterations=clf_iter,
                            depth=clf_d,
                            learning_rate=lr,
                            loss_function="Logloss",
                            random_seed=42,
                            verbose=False
                        )
                        clf.fit(X_train, r_train)
                        
                        # Regressors
                        regressors = {}
                        for reg in range(n_comp):
                            idx = r_train == reg
                            if idx.sum() < 10:  # Skip if too few samples
                                continue
                            
                            model = CatBoostRegressor(
                                iterations=reg_iter,
                                depth=reg_d,
                                learning_rate=lr,
                                loss_function="RMSE",
                                random_seed=42,
                                verbose=False
                            )
                            model.fit(X_train[idx], y_train[idx])
                            regressors[reg] = model
                        
                        # Validation prediction
                        val_proba = clf.predict_proba(X_val)
                        y_val_pred = np.zeros(len(y_val))
                        
                        for reg in regressors:
                            y_val_pred += val_proba[:, reg] * regressors[reg].predict(X_val)
                        
                        val_r2 = r2_score(y_val, y_val_pred)
                        
                        results.append({
                            'n_components': n_comp,
                            'clf_iterations': clf_iter,
                            'clf_depth': clf_d,
                            'reg_iterations': reg_iter,
                            'reg_depth': reg_d,
                            'learning_rate': lr,
                            'val_r2': val_r2
                        })
                        
                        if val_r2 > best_score:
                            best_score = val_r2
                            best_params = results[-1].copy()
                            print(f"Trial {trial}: New best! Val R² = {val_r2:.4f}")
                            print(f"  Params: {best_params}")

print("\n" + "="*60)
print("BEST PARAMETERS:")
print("="*60)
for k, v in best_params.items():
    print(f"{k}: {v}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(f"tuning_results_{PROBLEM_NUM}.csv", index=False)
print(f"\nAll results saved to: tuning_results_{PROBLEM_NUM}.csv")
