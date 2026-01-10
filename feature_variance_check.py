import numpy as np
import pandas as pd

# ===============================
# Load data
# ===============================
PROBLEM_NUM = 36

X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")

print(f"X shape: {X.shape}")

# ===============================
# Choose reference feature
# ===============================
REF_FEATURE = X.columns[0]   # you can change this
ref_values = X[REF_FEATURE].values

# ===============================
# Variance comparison
# ===============================
results = []

ref_var = np.var(ref_values)

for col in X.columns:
    diff = ref_values - X[col].values
    diff_var = np.var(diff)

    ratio = diff_var / ref_var if ref_var > 0 else np.nan

    results.append({
        "feature": col,
        "var_diff": diff_var,
        "var_ratio_to_ref": ratio
    })

df = pd.DataFrame(results).sort_values("var_ratio_to_ref")

# ===============================
# Thresholds (interpretable)
# ===============================
LOW_VARIANCE_THRESHOLD = 0.1   # <10% of reference variance

almost_identical = df[df["var_ratio_to_ref"] < LOW_VARIANCE_THRESHOLD]

# ===============================
# Output
# ===============================
print("\nReference feature:", REF_FEATURE)
print(f"Reference variance: {ref_var:.6f}")

print("\nFeatures with VERY LOW variance of difference:")
print(almost_identical)

print("\nTop 20 most similar features:")
print(df.head(20))

print("\nTop 20 most different features:")
print(df.tail(20))
