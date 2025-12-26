import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# Load latest ic_full_summary.csv
# =====================================

base_dir = "results"
timestamps = [
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d))
]
latest_ts = sorted(timestamps)[-1]
csv_path = os.path.join(base_dir, latest_ts, "ic_full_summary.csv")

print("Loading:", csv_path)
df = pd.read_csv(csv_path)

# =====================================
# Gain ratio (best non-truth vs truth)
# =====================================

df = df[df["honest_utility"] > 0].copy()
df["gain_ratio"] = df["max_lie_utility"] / df["honest_utility"]

EPS = 1e-8
df_no = df[df["gain_ratio"] <= 1.0 + EPS]
df_vi = df[df["gain_ratio"] > 1.0 + EPS]

# =====================================
# Linear bins: min to max
# =====================================

min_val = df["gain_ratio"].min()
max_val = df["gain_ratio"].max()

print(f"Gain ratio range: [{min_val:.4f}, {max_val:.4f}]")

num_bins = 60
bins = np.linspace(min_val, max_val, num_bins + 1)

# =====================================
# Histogram
# =====================================

plt.figure(figsize=(9, 5))

plt.hist(
    df_no["gain_ratio"],
    bins=bins,
    color="steelblue",
    alpha=0.7,
    label="No violation"
)

plt.hist(
    df_vi["gain_ratio"],
    bins=bins,
    color="darkorange",
    alpha=0.7,
    label="Violation"
)

plt.axvline(1.0, color="black", linestyle="--", linewidth=2)

plt.xlabel("Utility ratio (best non-truth report / truth)")
plt.ylabel("Number of profiles")
plt.title("Utility Gain from Misreporting (Linear bins: min–max)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
