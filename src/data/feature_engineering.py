import pandas as pd


def add_features(data):
    """
    Adds feature columns to the stock data.

    Args:
        data (pd.DataFrame): The stock data with columns like 'Close', 'Volume', etc.

    Returns:
        pd.DataFrame: The stock data with added features.
    """
    # Moving Averages
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    # Price Returns
    data['Daily Return'] = data['Close'].pct_change()

    # Lag Features
    data['Lag1'] = data['Close'].shift(1)
    data['Lag2'] = data['Close'].shift(2)

    # Drop rows with NaN values
    data = data.dropna()

    return data
