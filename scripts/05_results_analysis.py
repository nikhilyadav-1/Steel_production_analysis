import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================
# Results dictionary
# =====================
results = {
    "Model": ["Random Forest", "SVM", "MLP", "Gaussian Process"],
    "RMSE": [0.0587, 0.0669, 0.0801, 0.4424],
    "MAE": [0.0460, 0.0530, 0.0633, 0.4214],
    "R2": [0.3265, 0.1246, -0.2542, -37.2880],
    "Training Time": [5.9384, 0.4149, 1.5127, 10.0151],
    "Inference Time": [0.0495, 0.1914, 0.0020, 0.3173]
}

df = pd.DataFrame(results)

# =====================
# Save results table
# =====================
os.makedirs("results", exist_ok=True)
df.to_csv("results/performance_metrics.csv", index=False)

print("Performance metrics saved to results/performance_metrics.csv")

# =====================
# Visualization setup
# =====================
sns.set(style="whitegrid")
os.makedirs("figures", exist_ok=True)

# =====================
# RMSE comparison
# =====================
plt.figure()
sns.barplot(x="Model", y="RMSE", data=df)
plt.title("RMSE Comparison of Models")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("figures/rmse_comparison.png")
plt.close()

# =====================
# R2 comparison
# =====================
# =====================
# R2 comparison (fixed scale)
# =====================
plt.figure()
sns.barplot(x="Model", y="R2", data=df)
plt.title("R² Score Comparison")
plt.ylim(-1, 0.5)   # limit axis to show meaningful range
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("figures/r2_comparison.png")
plt.close()

# =====================
# Training time comparison
# =====================
plt.figure()
sns.barplot(x="Model", y="Training Time", data=df)
plt.title("Training Time Comparison (seconds)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("figures/training_time.png")
plt.close()

print("All plots saved in figures/")
