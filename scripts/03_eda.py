import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================
# 1. Correlation heatmap
# =====================
def plot_correlation_matrix(df, save_path="figures/correlation_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12,10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Correlation matrix saved to {save_path}")

# =====================
# 2. Feature distributions (histograms)
# =====================
def plot_feature_distributions(df, save_path="figures/feature_histograms.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.hist(figsize=(15,12), bins=30, edgecolor='black')
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Feature histograms saved to {save_path}")

# =====================
# 3. Box plots (for outlier detection)
# =====================
def plot_boxplots(df, save_path="figures/feature_boxplots.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(15,10))
    sns.boxplot(data=df, orient="h")
    plt.title("Feature Boxplots")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Boxplots saved to {save_path}")

# =====================
# 4. Pair plots (feature relationships)
# =====================
def plot_pairplots(df, save_path="figures/pairplots.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.pairplot(df.sample(min(200, len(df))))  # sample to speed up
    plt.suptitle("Pair Plots (Sampled)", y=1.02)
    plt.savefig(save_path)
    plt.close()
    print(f"Pairplots saved to {save_path}")

# =====================
# 5. Target variable distribution
# =====================
def plot_target_distribution(df, target_column="output", save_path="figures/target_distribution.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8,6))
    sns.histplot(df[target_column], bins=30, kde=True, color="skyblue")
    plt.title(f"Distribution of Target Variable: {target_column}")
    plt.xlabel(target_column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Target distribution saved to {save_path}")

# =====================
# 6. Split & normalize data
# =====================
def split_and_normalize_data(df, target_column="output", test_size=0.2, val_size=0.1, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # First split train+val/test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Split train/validation
    val_relative_size = val_size / (1 - test_size)  # adjust relative to remaining
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_relative_size, random_state=random_state
    )

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print("Data split into train/validation/test and normalized")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler
