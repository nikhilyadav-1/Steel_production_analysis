import pandas as pd
import numpy as np

# -------------------------
# 1. Remove duplicates
# -------------------------
def remove_duplicates(df):
    """Remove duplicate rows from dataset"""
    return df.drop_duplicates()

# -------------------------
# 2. Handle missing values
# -------------------------
def handle_missing_values(df):
    """Fill missing numeric values with median, categorical with mode"""
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# -------------------------
# 3. Detect & treat outliers (IQR method)
# -------------------------
def treat_outliers(df, columns=None):
    """Replace outliers with median"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        median = df[col].median()
        df[col] = np.where((df[col] < lower) | (df[col] > upper), median, df[col])
    return df

# -------------------------
# 4. Encode categorical variables
# -------------------------
def encode_categorical(df):
    """Convert categorical columns to numeric using one-hot encoding"""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

# -------------------------
# 5. Full preprocessing pipeline
# -------------------------
def preprocess(df):
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = treat_outliers(df)
    df = encode_categorical(df)
    return df

# -------------------------
# 6. Script entry point
# -------------------------
if __name__ == "__main__":
    # Load datasets
    df_train = pd.read_csv("data/normalized_train_data.csv", engine="python")
    df_test = pd.read_csv("data/normalized_test_data.csv", engine="python")
    
    # Preprocess
    print("Preprocessing training data...")
    df_train_clean = preprocess(df_train)
    print("Training data processed. Shape:", df_train_clean.shape)

    print("\nPreprocessing test data...")
    df_test_clean = preprocess(df_test)
    print("Test data processed. Shape:", df_test_clean.shape)

    # Save cleaned datasets
    df_train_clean.to_csv("results/clean_train.csv", index=False)
    df_test_clean.to_csv("results/clean_test.csv", index=False)
    print("\nCleaned datasets saved in results/")
