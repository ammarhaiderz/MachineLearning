# ============================================================
# 0. Imports
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from scipy.stats import rankdata

sns.set_style("whitegrid")


# ============================================================
# 1. Load target
# ============================================================
PROBLEM_NUM = 36

y = pd.read_csv(
    f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv"
)["target01"].values

print(f"Loaded target01 with shape: {y.shape}")


# ============================================================
# 2. Target transforms
# ============================================================

# 1) Original
y_original = y.copy()

# 2) Log transform (safe)
y_log = np.log1p(y - y.min() + 1e-6)

# 3) Box-Cox (only if strictly positive)
if np.all(y > 0):
    pt_boxcox = PowerTransformer(method="box-cox")
    y_boxcox = pt_boxcox.fit_transform(y.reshape(-1, 1)).ravel()
else:
    y_boxcox = None

# 4) Yeo-Johnson (always works)
pt_yj = PowerTransformer(method="yeo-johnson")
y_yeojohnson = pt_yj.fit_transform(y.reshape(-1, 1)).ravel()

# 5) Quantile → Normal
qt_normal = QuantileTransformer(
    n_quantiles=min(1000, len(y)),
    output_distribution="normal",
    random_state=42
)
y_qt_normal = qt_normal.fit_transform(y.reshape(-1, 1)).ravel()

# 6) Quantile → Uniform
qt_uniform = QuantileTransformer(
    n_quantiles=min(1000, len(y)),
    output_distribution="uniform",
    random_state=42
)
y_qt_uniform = qt_uniform.fit_transform(y.reshape(-1, 1)).ravel()

# 7) Rank-Gauss (manual)
ranks = rankdata(y, method="average")
ranks = (ranks - 0.5) / len(y)
from scipy.stats import norm
y_rank_gauss = norm.ppf(ranks)


# ============================================================
# 3. Collect for plotting
# ============================================================
transforms = {
    "Original": y_original,
    "Log1p": y_log,
    "Yeo-Johnson": y_yeojohnson,
    "Quantile-Normal": y_qt_normal,
    "Quantile-Uniform": y_qt_uniform,
    "Rank-Gauss": y_rank_gauss
}

if y_boxcox is not None:
    transforms["Box-Cox"] = y_boxcox


# ============================================================
# 4. Visualization
# ============================================================
n = len(transforms)
fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))

for i, (name, data) in enumerate(transforms.items()):
    # Histogram
    axes[i, 0].hist(data, bins=50, density=True, alpha=0.7)
    axes[i, 0].set_title(f"{name} – Histogram")
    axes[i, 0].set_ylabel("Density")

    # KDE
    sns.kdeplot(data, ax=axes[i, 1], linewidth=2)
    axes[i, 1].set_title(f"{name} – KDE")

    # Q-Q plot
    from scipy import stats
    stats.probplot(data, dist="norm", plot=axes[i, 2])
    axes[i, 2].set_title(f"{name} – Q-Q Plot")

plt.tight_layout()
plt.savefig(f"target_transforms_comparison_{PROBLEM_NUM}.png", dpi=150)
plt.show()

print(f"\nSaved visualization to: target_transforms_comparison_{PROBLEM_NUM}.png")
