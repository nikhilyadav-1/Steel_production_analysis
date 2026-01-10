import pandas as pd
import os

def load_steel_data(file_path):
    df = pd.read_csv(file_path, engine="python")
    return df

if __name__ == "__main__":
    # Debug: List files in data folder
    print("Files in data folder:", os.listdir("data"))

    # Load training dataset
    train_file = "data/normalized_train_data.csv"
    print(f"\nLoading {train_file}...")
    df_train = load_steel_data(train_file)
    print("Training dataset loaded successfully!")
    print(df_train.head(), "\nShape:", df_train.shape)

    # Load test dataset
    test_file = "data/normalized_test_data.csv"
    print(f"\nLoading {test_file}...")
    df_test = load_steel_data(test_file)
    print("Test dataset loaded successfully!")
    print(df_test.head(), "\nShape:", df_test.shape)
