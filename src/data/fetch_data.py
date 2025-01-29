import yfinance as yf


def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data for a given ticker.

    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL").
        start_date (str): Start date (e.g., "2022-01-01").
        end_date (str): End date (e.g., "2023-01-01").

    Returns:
        pd.DataFrame: Stock data as a pandas DataFrame.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data
