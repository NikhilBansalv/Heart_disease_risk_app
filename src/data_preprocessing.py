import pandas as pd

def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)
    missing = df.isnull().sum()
    if missing.any():
        print("Warning: Missing values detected per column:\n", missing)

    df_encoded = pd.get_dummies(
        df,
        columns=["cp", "restecg", "slope", "thal"],
        drop_first=True
    )

    return df_encoded

def split_features_labels(df_encoded: pd.DataFrame):
    X = df_encoded.drop(columns=["target"])
    y = df_encoded["target"]
    return X, y

if __name__ == "__main__":
    df_clean = load_and_preprocess("data/heart.csv")
    X, y = split_features_labels(df_clean)
    print("Data shape after encoding:", df_clean.shape)
    print("Columns:", list(df_clean.columns))
    print("First few rows:\n", df_clean.head())
