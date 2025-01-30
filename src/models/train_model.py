import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.data.feature_engineering import add_features
from src.data.fetch_data import fetch_stock_data


def train_model(ticker, start_date, end_date):
    """Train an ML model to predict stock price movement."""

    # Step 1: Fetch and preprocess data
    print(f"Fetching data for {ticker}...")
    data = fetch_stock_data(ticker, start_date, end_date)
    data = add_features(data,ticker)

    # Step 2: Prepare features and target
    X = data[['MA50', 'MA200', 'EMA12', 'EMA26', 'MACD', 'Signal Line',
              'Middle Band', 'Upper Band', 'Lower Band', 'OBV',
              'Daily Return', 'Lag1', 'Lag2', 'Volatility']]
    y = (data['Close'].shift(-1) > data['Close']).astype(int)  # 1 if price goes up next day, else 0

    # Drop NaN values from shifting
    X = X.iloc[:-1]
    y = y.iloc[:-1]

    # Step 3: Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 5: Train model
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train.to_numpy().ravel())

    # Step 6: Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

    return model, scaler  # Return trained model and scaler for future use
