import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# ===============================
# 1. Load data
# ===============================
PROBLEM_NUM = 36

X = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/dataset_{PROBLEM_NUM}.csv")
y = pd.read_csv(f"./data_31_40/problem_{PROBLEM_NUM}/target_{PROBLEM_NUM}.csv")["target01"].values

print("X shape:", X.shape)
print("y shape:", y.shape)

# ===============================
# 2. Standardize X (important for GMM)
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 3. Fit GMM on X only
# ===============================
gmm = GaussianMixture(
    n_components=2,
    covariance_type="full",
    n_init=10,
    random_state=42
)

gmm.fit(X_scaled)

cluster_probs = gmm.predict_proba(X_scaled)
cluster_labels = np.argmax(cluster_probs, axis=1)

# ===============================
# 4. Basic cluster diagnostics (X-only)
# ===============================
print("\n===============================")
print("X-ONLY CLUSTER DIAGNOSTICS")
print("===============================")

sil_score = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette score (X-only): {sil_score:.4f}")

cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
print("\nCluster sizes:")
print(cluster_sizes)

print("\nMean max cluster probability (confidence):")
print(np.mean(np.max(cluster_probs, axis=1)))

# ===============================
# 5. Evaluate clusters using y (diagnostic only)
# ===============================
df = pd.DataFrame({
    "cluster": cluster_labels,
    "target": y
})

print("\n===============================")
print("TARGET DISTRIBUTION PER CLUSTER")
print("===============================")

for c in sorted(df["cluster"].unique()):
    yt = df[df["cluster"] == c]["target"]
    print(f"\nCluster {c}:")
    print(f"  Count: {len(yt)}")
    print(f"  Mean:  {yt.mean():.4f}")
    print(f"  Std:   {yt.std():.4f}")
    print(f"  Min:   {yt.min():.4f}")
    print(f"  Max:   {yt.max():.4f}")

# ===============================
# 6. Plot target distribution per cluster
# ===============================
plt.figure(figsize=(8, 5))
sns.kdeplot(df[df["cluster"] == 0]["target"], label="Cluster 0", fill=True)
sns.kdeplot(df[df["cluster"] == 1]["target"], label="Cluster 1", fill=True)
plt.title("Target distribution per X-only GMM cluster")
plt.xlabel("Target value")
plt.legend()
plt.tight_layout()
plt.show()

# ===============================
# 7. Soft assignment analysis
# ===============================
df["cluster_confidence"] = np.max(cluster_probs, axis=1)

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=df["cluster_confidence"],
    y=df["target"],
    alpha=0.4
)
plt.axvline(0.5, color="red", linestyle="--")
plt.title("Target vs GMM confidence (X-only)")
plt.xlabel("Max cluster probability")
plt.ylabel("Target")
plt.tight_layout()
plt.show()

# ===============================
# 8. Interpretation helper
# ===============================
print("\n===============================")
print("INTERPRETATION GUIDE")
print("===============================")

if sil_score < 0.05:
    print("❌ Very weak or no cluster structure in X.")
else:
    print("✅ Some cluster structure exists in X.")

print("""
Check the KDE plot:
- If both clusters have similar bimodal target distributions:
    → regimes NOT identifiable from X alone (irreducible ambiguity)
- If each cluster concentrates on one mode:
    → regimes partially identifiable from X

Check confidence scatter:
- If high-confidence points still cover both target modes:
    → clustering is not predictive of regime
""")
