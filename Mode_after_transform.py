# ============================================================
# 0. Imports
# ============================================================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import (
    QuantileTransformer,
    PowerTransformer,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)
from scipy import stats

from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')


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
# 2. Train / validation split
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 3. DEFINE ALL TRANSFORMATIONS
# ============================================================
class NoTransform:
    """No transformation (baseline)"""
    def fit(self, y):
        return self
    
    def transform(self, y):
        return y.reshape(-1, 1) if len(y.shape) == 1 else y
    
    def fit_transform(self, y):
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        return y.ravel() if len(y.shape) > 1 else y


class LogTransform:
    """Log transformation with optional shift"""
    def __init__(self):
        self.shift = None
    
    def fit(self, y):
        self.shift = abs(y.min()) + 1 if y.min() <= 0 else 0
        return self
    
    def transform(self, y):
        return np.log(y + self.shift).reshape(-1, 1)
    
    def fit_transform(self, y):
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        return (np.exp(y.ravel()) - self.shift)


class SqrtTransform:
    """Square root transformation with optional shift"""
    def __init__(self):
        self.shift = None
    
    def fit(self, y):
        self.shift = abs(y.min()) + 1 if y.min() < 0 else 0
        return self
    
    def transform(self, y):
        return np.sqrt(y + self.shift).reshape(-1, 1)
    
    def fit_transform(self, y):
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        return (y.ravel() ** 2 - self.shift)


class SquareTransform:
    """Square transformation"""
    def fit(self, y):
        return self
    
    def transform(self, y):
        return (y ** 2).reshape(-1, 1)
    
    def fit_transform(self, y):
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        return np.sqrt(np.abs(y.ravel()))


class CubeRootTransform:
    """Cube root transformation"""
    def fit(self, y):
        return self
    
    def transform(self, y):
        return np.cbrt(y).reshape(-1, 1)
    
    def fit_transform(self, y):
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        return y.ravel() ** 3


class ReciprocalTransform:
    """Reciprocal transformation (1/y) with safety"""
    def __init__(self):
        self.shift = None
    
    def fit(self, y):
        # Ensure no zeros
        self.shift = 0.001 if (y == 0).any() else 0
        return self
    
    def transform(self, y):
        return (1.0 / (y + self.shift)).reshape(-1, 1)
    
    def fit_transform(self, y):
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        return (1.0 / y.ravel() - self.shift)


transformations = {
    "None (Baseline)": NoTransform(),
    "Log": LogTransform(),
    "Sqrt": SqrtTransform(),
    "Square": SquareTransform(),
    "Cube Root": CubeRootTransform(),
    "Reciprocal": ReciprocalTransform(),
    "Quantile-Normal": QuantileTransformer(
        n_quantiles=min(1000, len(y_train)),
        output_distribution="normal",
        random_state=42
    ),
    "Quantile-Uniform": QuantileTransformer(
        n_quantiles=min(1000, len(y_train)),
        output_distribution="uniform",
        random_state=42
    ),
    "Box-Cox": PowerTransformer(method="box-cox", standardize=True),
    "Yeo-Johnson": PowerTransformer(method="yeo-johnson", standardize=True),
}


# ============================================================
# 4. Run experiments for each transformation
# ============================================================
results = []

print(f"\n{'='*70}")
print(f"RUNNING EXPERIMENTS WITH ALL TRANSFORMATIONS")
print(f"{'='*70}\n")

for trans_name, transformer in transformations.items():
    print(f"\n{'─'*70}")
    print(f"🔄 Testing: {trans_name}")
    print(f"{'─'*70}")
    
    try:
        # Transform target
        z_train = transformer.fit_transform(y_train.reshape(-1, 1)).ravel()
        z_val = transformer.transform(y_val.reshape(-1, 1)).ravel()
        
        # Train model
        model = CatBoostRegressor(
            iterations=1200,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=42,
            verbose=False
        )
        
        model.fit(X_train, z_train)
        
        # Predictions
        z_train_pred = model.predict(X_train)
        z_val_pred = model.predict(X_val)
        
        # Inverse transform
        y_train_pred = transformer.inverse_transform(z_train_pred.reshape(-1, 1))
        y_val_pred = transformer.inverse_transform(z_val_pred.reshape(-1, 1))
        
        # Ensure proper shape
        if len(y_train_pred.shape) > 1:
            y_train_pred = y_train_pred.ravel()
        if len(y_val_pred.shape) > 1:
            y_val_pred = y_val_pred.ravel()
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = root_mean_squared_error(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        val_rmse = root_mean_squared_error(y_val, y_val_pred)
        
        results.append({
            "Transformation": trans_name,
            "Train_R2": train_r2,
            "Train_RMSE": train_rmse,
            "Val_R2": val_r2,
            "Val_RMSE": val_rmse,
            "Status": "Success"
        })
        
        print(f"✓ Train R²:   {train_r2:.4f}")
        print(f"✓ Train RMSE: {train_rmse:.4f}")
        print(f"✓ Val R²:     {val_r2:.4f}")
        print(f"✓ Val RMSE:   {val_rmse:.4f}")
        
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        results.append({
            "Transformation": trans_name,
            "Train_R2": np.nan,
            "Train_RMSE": np.nan,
            "Val_R2": np.nan,
            "Val_RMSE": np.nan,
            "Status": f"Failed: {str(e)}"
        })


# ============================================================
# 5. Display Results Summary
# ============================================================
print(f"\n\n{'='*70}")
print(f"📊 RESULTS SUMMARY")
print(f"{'='*70}\n")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Val_R2", ascending=False)

print(results_df.to_string(index=False))

# Save results
results_df.to_csv(f"transformation_comparison_problem_{PROBLEM_NUM}.csv", index=False)
print(f"\n✓ Results saved to: transformation_comparison_problem_{PROBLEM_NUM}.csv")


# ============================================================
# 6. Train final model with BEST transformation on full data
# ============================================================
best_result = results_df.iloc[0]
best_trans_name = best_result["Transformation"]
best_transformer = transformations[best_trans_name]

print(f"\n{'='*70}")
print(f"🏆 BEST TRANSFORMATION: {best_trans_name}")
print(f"{'='*70}")
print(f"Validation R²:   {best_result['Val_R2']:.4f}")
print(f"Validation RMSE: {best_result['Val_RMSE']:.4f}")


# Retrain on full data with best transformation
z_full = best_transformer.fit_transform(y.reshape(-1, 1)).ravel()

final_model = CatBoostRegressor(
    iterations=1200,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)

final_model.fit(X, z_full)


# ============================================================
# 7. EVAL prediction (NO LABELS)
# ============================================================
z_eval_pred = final_model.predict(X_eval)
y_eval_pred = best_transformer.inverse_transform(z_eval_pred.reshape(-1, 1))

if len(y_eval_pred.shape) > 1:
    y_eval_pred = y_eval_pred.ravel()


# ============================================================
# 8. Save output
# ============================================================
output_file = f"EVAL_target01_{PROBLEM_NUM}_BestTransform_{best_trans_name.replace(' ', '_').replace('-', '_')}.csv"

pd.DataFrame({"target01": y_eval_pred}).to_csv(output_file, index=False)

print(f"\n✓ Predictions saved to: {output_file}")
print(f"Sample predictions: {y_eval_pred[:10]}")
print(f"\n{'='*70}\n")
