import yfinance as yf

def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

if __name__ == "__main__":
    ticker = "AAPL"  # Example: Apple Inc.
    start_date = "2020-01-01"
    end_date = "2022-01-01"
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    print(stock_data.head())
