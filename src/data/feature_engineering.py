def add_features(data, ticker):
    """
    Add features to the stock data for a specific ticker.

    Args:
        data (pd.DataFrame): Stock data with columns like 'Close', 'Volume'.
        ticker (str): The ticker symbol for the company (e.g., 'AAPL').

    Returns:
        pd.DataFrame: Stock data with new feature columns.
    """
    # Define the column names dynamically based on the ticker symbol
    close_column = ('Close', ticker)
    volume_column = ('Volume', ticker)

    # Before passing the data to the function, make sure it's a copy
    data = data.copy()

    # Moving averages
    data[('MA50', '')] = data[close_column].rolling(window=50).mean()
    data[('MA200', '')] = data[close_column].rolling(window=200).mean()

    # Daily return
    data[('Daily Return', '')] = data[close_column].pct_change()

    # Lag features
    data[('Lag1', '')] = data[close_column].shift(1)
    data[('Lag2', '')] = data[close_column].shift(2)

    # Volatility
    data[('Volatility', '')] = data[close_column].rolling(window=10).std()

    # Exponential moving averages
    data[('EMA12', '')] = data[close_column].ewm(span=12, adjust=False).mean()
    data[('EMA26', '')] = data[close_column].ewm(span=26, adjust=False).mean()

    # Moving Average onvergence Divergence(MACD)
    # MACD > Signal Line → Buy Signal
    # MACD < Signal Line → Sell Signal

    data[('MACD', '')] = data[('EMA12', '')] - data[('EMA26', '')]
    data[('Signal Line', '')] = data[('MACD', '')].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    # Price near Upper Band → Overbought (Sell Signal)
    # Price near Lower Band → Oversold (Buy Signal)

    data[('Middle Band', '')] = data[close_column].rolling(window=20).mean()
    data[('Upper Band', '')] = data[('Middle Band', '')] + 2 * data[close_column].rolling(window=20).std()
    data[('Lower Band', '')] = data[('Middle Band', '')] - 2 * data[close_column].rolling(window=20).std()

    # RSI
    # RSI > 70 → Overbought (Sell Signal)
    # RSI < 30 → Oversold (Buy Signal)

    def compute_rsi(data, close_column, window=14):
        delta = data[close_column].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / (loss + 1e-9)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi

    data[('RSI', '')] = compute_rsi(data, close_column)

    # Cumulative return
    data[('Cumulative Return', '')] = (1 + data[('Daily Return', '')]).cumprod()

    # Volume-related features
    data[('Volume MA20', '')] = data[volume_column].rolling(window=20).mean()

    # On-Balance Volume (OBV)
    # Rising OBV → Positive momentum
    # Falling OBV → Negative momentum

    data[('OBV', '')] = (data[volume_column] * ((data[close_column].diff() > 0) * 2 - 1)).cumsum()

    # Print column names and a sample of data to check the presence of 'MA50'
    # print("Columns in the data:", data.columns)
    # print(data[[close_column, ('MA50', '')]].head(10))  # Check first few rows of Close and MA50

    # Drop rows with NaN values in 'MA50'
    data = data.dropna(subset=[('MA50', '')])

    # Trend (MA50 should now be populated)
    data.loc[:, ('Trend', '')] = data[close_column] > data[('MA50', '')]

    return data
