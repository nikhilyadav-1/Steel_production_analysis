print("=== Phase 3: Model Training Started ===")
# -----------------------------
# Phase 3: Model Training
# -----------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# -----------------------------
# 1. Load cleaned datasets
# -----------------------------
df_train = pd.read_csv("results/clean_train.csv", engine="python")
df_test = pd.read_csv("results/clean_test.csv", engine="python")

# -----------------------------
# 2. Split features and target
# -----------------------------
target_column = "output"  # <-- Replace with your actual target column name

X = df_train.drop(columns=[target_column])
y = df_train[target_column]

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Normalize features
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(df_test.drop(columns=[target_column], errors='ignore'))

# -----------------------------
# 4. Model training functions
# -----------------------------
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    return model, end - start

def train_svm(X_train, y_train):
    model = SVR()
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    return model, end - start

def train_mlp(X_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    return model, end - start

def train_gaussian_process(X_train, y_train):
    model = GaussianProcessRegressor()
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    return model, end - start

# -----------------------------
# 5. Model evaluation function
# -----------------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

def evaluate_model(model, X_val, y_val):
    start = time.time()

    y_pred = model.predict(X_val)

    inference_time = time.time() - start

    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Inference Time": inference_time
    }

    return metrics, y_pred
# -----------------------------
# 6. Train & evaluate all models
# -----------------------------
models = {}
results = {}

# Random Forest
rf_model, rf_time = train_random_forest(X_train, y_train)
models["Random Forest"] = rf_model
results["Random Forest"], rf_pred = evaluate_model(rf_model, X_val, y_val)
results["Random Forest"]["Training Time"] = rf_time

# SVM
svm_model, svm_time = train_svm(X_train, y_train)
models["SVM"] = svm_model
results["SVM"], svm_pred = evaluate_model(svm_model, X_val, y_val)
results["SVM"]["Training Time"] = svm_time

# MLP
mlp_model, mlp_time = train_mlp(X_train, y_train)
models["MLP"] = mlp_model
results["MLP"], mlp_pred = evaluate_model(mlp_model, X_val, y_val)
results["MLP"]["Training Time"] = mlp_time

# Gaussian Process
gp_model, gp_time = train_gaussian_process(X_train, y_train)
models["Gaussian Process"] = gp_model
results["Gaussian Process"], gp_pred = evaluate_model(gp_model, X_val, y_val)
results["Gaussian Process"]["Training Time"] = gp_time

# -----------------------------
# 7. Print results
# -----------------------------
print("\nModel Performance Comparison:")
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
