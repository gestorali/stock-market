#/home/michal/PycharmProjects/stock-market/src/features/technical_indicators.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

scaler_default_path = "models/price_scaler.pkl"

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds common technical indicators without using TA-Lib.
    Works directly on a DataFrame with columns: Open, High, Low, Close, Volume.
    """

    df = df.copy()

    # Moving Averages
    df["MA25"] = df["Close"].rolling(window=25).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()

    # Bollinger Bands
    rolling_mean = df["Close"].rolling(window=20).mean()
    rolling_std = df["Close"].rolling(window=20).std()
    df["BB_upper"] = rolling_mean + (rolling_std * 2)
    df["BB_lower"] = rolling_mean - (rolling_std * 2)

    # MACD (12 EMA - 26 EMA)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

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

    feature_cols = [
        "sentiment", "news_count", "general_sentiment",
        "MA25", "MA50", "MACD", "MACD_signal",
        "BB_upper", "BB_lower", "RSI"
    ]

    # Save feature columns to file so training can load them
    with open("models/feature_columns.json", "w") as f:
        json.dump(feature_cols, f)
    print(f"✅ Feature columns saved to 'models/feature_columns.json'")

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
