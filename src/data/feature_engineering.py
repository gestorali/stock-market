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

    # RSI
    def compute_rsi(data, close_column, window=14):
        delta = data[close_column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    data[('RSI', '')] = compute_rsi(data, close_column)

    # Cumulative return
    data[('Cumulative Return', '')] = (1 + data[('Daily Return', '')]).cumprod()

    # Volume-related features
    data[('Volume MA20', '')] = data[volume_column].rolling(window=20).mean()

    # Print column names and a sample of data to check the presence of 'MA50'
    print("Columns in the data:", data.columns)
    print(data[[close_column, ('MA50', '')]].head(10))  # Check first few rows of Close and MA50

    # Drop rows with NaN values in 'MA50'
    data = data.dropna(subset=[('MA50', '')])

    # Trend (MA50 should now be populated)
    data.loc[:, ('Trend', '')] = data[close_column] > data[('MA50', '')]

    return data
