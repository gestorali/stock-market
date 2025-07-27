#technical_indicators

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os


scaler_default_path = "models/price_scaler.pkl"

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds MA25, MA50, MACD, and Bollinger Bands to the DataFrame.
    """
    df = df.copy()
    df["MA25"] = df["Close"].rolling(window=25).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26

    # Bollinger Bands
    ma20 = df["Close"].rolling(window=20).mean()
    std20 = df["Close"].rolling(window=20).std()
    df["Middle Band"] = ma20
    df["Upper Band"] = ma20 + (2 * std20)
    df["Lower Band"] = ma20 - (2 * std20)

    return df

def scale_features(df: pd.DataFrame, feature_cols: list, scaler_path: str = scaler_default_path):
    """
    Scales selected features using StandardScaler and saves the scaler.

    Args:
        df (pd.DataFrame): DataFrame with features to scale.
        feature_cols (list): List of column names to scale.
        scaler_path (str): Path to save the scaler object.

    Returns:
        pd.DataFrame: DataFrame with scaled features.
    """
    X = df[feature_cols].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to '{scaler_path}'")

    df_scaled = df.copy()
    df_scaled[feature_cols] = X_scaled

    return df_scaled

def apply_scaler(df: pd.DataFrame, feature_cols: list, scaler_path: str = scaler_default_path):
    """
    Applies a saved scaler to scale selected features.

    Args:
        df (pd.DataFrame): DataFrame with features to scale.
        feature_cols (list): List of column names to scale.
        scaler_path (str): Path to the saved scaler.

    Returns:
        pd.DataFrame: DataFrame with scaled features.
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file '{scaler_path}' not found. Run `scale_features` first.")

    scaler = joblib.load(scaler_path)
    X = df[feature_cols].astype(float)
    X_scaled = scaler.transform(X)

    df_scaled = df.copy()
    df_scaled[feature_cols] = X_scaled

    return df_scaled

def run_scaling_pipeline():
    """
    Full pipeline for adding indicators, scaling features, and saving the result.
    """
    df = pd.read_csv("data/features/combined.csv")
    df = add_technical_indicators(df)

    feature_cols = ["Close", "sentiment", "MA25", "MA50", "MACD", "Middle Band"]
    df = df.dropna(subset=feature_cols)

    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    df_train_scaled = scale_features(df_train, feature_cols)
    df_test_scaled = apply_scaler(df_test, feature_cols)

    df_scaled = pd.concat([df_train_scaled, df_test_scaled], axis=0)

    os.makedirs("data/features", exist_ok=True)
    df_train_scaled.to_csv("data/features/combined_scaled_train.csv", index=False)
    df_test_scaled.to_csv("data/features/combined_scaled_test.csv", index=False)
    df_scaled.to_csv("data/features/combined_scaled_all.csv", index=False)

    print("✅ Scaled train, test, and all data saved.")
