# src/features/scaler.py
from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df