import numpy as np
import pandas as pd

import pymc as pm
import pymc_bart as pmb
import arviz as az

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def main():

    # ===============================
    # CONFIG
    # ===============================
    PROBLEM_NUM = 36
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    # Conservative BART settings (important for 273 features)
    N_TREES = 100
    N_DRAWS = 1000
    N_TUNE = 1000
    TARGET_ACCEPT = 0.9

    # ===============================
    # LOAD DATA
    # ===============================
    X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
    y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"].values

    print(f"Loaded data: X={X.shape}, y={y.shape}")

    # ===============================
    # TRAIN / VALIDATION SPLIT
    # ===============================
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    print(f"Train samples: {X_train.shape[0]}")
    print(f"Val samples:   {X_val.shape[0]}")

    # ===============================
    # STANDARDIZE (important for BART)
    # ===============================
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # ===============================
    # BUILD BART MODEL
    # ===============================
    with pm.Model() as bart_model:

        X_shared = pm.Data("X", X_train_s)
        y_shared = pm.Data("y", y_train)

        mu = pmb.BART(
            "mu",
            X_shared,
            y_shared,
            m=N_TREES,
            alpha=0.95,
            beta=2
        )

        sigma = pm.HalfNormal("sigma", sigma=1.0)

        pm.Normal(
            "y_obs",
            mu=mu,
            sigma=sigma,
            observed=y_shared
        )

        print("\nSampling BART model (this will take time)...")
        trace = pm.sample(
            draws=N_DRAWS,
            tune=N_TUNE,
            target_accept=TARGET_ACCEPT,
            chains=1,        # 🔴 Windows-safe
            cores=1,         # 🔴 Windows-safe
            random_seed=RANDOM_STATE,
            progressbar=True
        )

    # ===============================
    # TRAIN PREDICTIONS
    # ===============================
    with bart_model:
        pm.set_data({"X": X_train_s})
        post_train = pm.sample_posterior_predictive(
            trace,
            var_names=["mu"],
            random_seed=RANDOM_STATE
        )

    y_train_pred = post_train["mu"].mean(axis=0)

    # ===============================
    # VALIDATION PREDICTIONS
    # ===============================
    with bart_model:
        pm.set_data({"X": X_val_s})
        post_val = pm.sample_posterior_predictive(
            trace,
            var_names=["mu"],
            random_seed=RANDOM_STATE
        )

    y_val_samples = post_val["mu"]
    y_val_pred = y_val_samples.mean(axis=0)

    # ===============================
    # METRICS
    # ===============================
    def report(y_true, y_pred, label):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)
        print(f"{label:<10s} | MSE: {mse:.6f} | MAE: {mae:.6f} | R²: {r2:.6f}")
        return mse, mae, r2

    print("\n" + "="*70)
    print("BART PERFORMANCE (ALL FEATURES)")
    print("="*70)

    train_metrics = report(y_train, y_train_pred, "TRAIN")
    val_metrics   = report(y_val, y_val_pred, "VALID")

    print("\nOverfitting check:")
    print(f"Δ MSE: {train_metrics[0] - val_metrics[0]:+.6f}")
    print(f"Δ MAE: {train_metrics[1] - val_metrics[1]:+.6f}")
    print(f"Δ R²:  {train_metrics[2] - val_metrics[2]:+.6f}")

    # ===============================
    # UNCERTAINTY ANALYSIS
    # ===============================
    pred_std = y_val_samples.std(axis=0)

    print("\nPredictive uncertainty (validation):")
    print(f"Mean std:   {pred_std.mean():.4f}")
    print(f"Median std: {np.median(pred_std):.4f}")
    print(f"95th % std:{np.percentile(pred_std, 95):.4f}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
